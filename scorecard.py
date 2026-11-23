"""Pro-level scorecard of a run (BSRL v2 plan item 10; HISTORY 82): what a strong Showdown player does, measured.

    python scorecard.py RUN [--last 3000]     # RUN = $SIM_WORK/runs/<run> (or a path); CPU only, reads files
    -> one summary line on stdout, details in RUN/scorecard.json

From episodes.jsonl (every finished training agent) + eval_episodes.jsonl (the policy's eval rows), grouped by battle kind (eval / evalb / evalp:<preset> / train),
team size and brawler:
  place      own placement; in teams the team's (best member rank among that battle's rows)
  win, top   share placing 1st / in the top half of the teams
  trade      damage dealt / damage taken (sim term units)
  hit, dps   hits / shots, damage per shot (ammo efficiency: wasted shots lower it)
  cubes, alive_s, gas   cubes picked, seconds survived, share killed by the gas
  behaviour  (rows with env_v5 "beh" counters) shares of alive time hidden by a bush / engaged / in gas, mate 1.5-4
             tiles / < 1 tile (share of time with a mate), range ratio, damage per minute, cubes held at the end
Presets (eval kind evalp, train --eval-presets): per preset the metrics above, the paired difference to balanced on
  the same seeds (mean, SE) and a pass rule per style (PASS_RULES; every preset: place within 0.5 of balanced).
Slider slopes (training rows, sampled weights): OLS of each per-episode metric on the slider weights -> slope and SE per
  (slider, metric); "own" = each slider's own term per minute (s_<slider>): t < 2 = the slider changes nothing.
From replays/*.json (one learner battle per update, true positions):
  lead       slope of (aim - bearing) on the ideal lead offset (target's true velocity, own projectile speed):
             0 = shoots where the target is, 1 = leads perfectly; aim_err = median |aim - ideal| in degrees
  range      median distance to the nearest enemy in view, divided by the brawler's attack range
             (marksmen should sit near 1, not rush to 0.3)
"""
import argparse, collections, glob, json, os
import numpy as np

TICK_S, UNITS_TILE = 0.05, 300.0


def _wrap(d):
    return (np.asarray(d, float) + 180.0) % 360.0 - 180.0


def team_place(rows):
    """{(seed, player): place}: own rank, or in team battles the best rank of its group among the seed's rows"""
    best = {}
    for r in rows:
        g = (r.get("tag") or {}).get("groups") or [0] * 10; k = g[r["player"]]
        if k and r["rank"] is not None:
            key = (r["seed"], k); best[key] = min(best.get(key, 99), r["rank"])
    out = {}
    for r in rows:
        g = (r.get("tag") or {}).get("groups") or [0] * 10; k = g[r["player"]]
        out[(r["seed"], r["player"])] = best.get((r["seed"], k), r["rank"]) if k else r["rank"]
    return out


def ep_stats(rows):
    ok = [r for r in rows if r["rank"] is not None and not r["truncated"] and not r["restart"]]
    if not ok:
        return {"n": 0}
    tp = team_place(ok); S = lambda k: float(sum(r["terms"].get(k, 0.0) for r in ok))
    pl = np.array([tp[(r["seed"], r["player"])] for r in ok], float)
    teams = np.array([len({g for g in (r["tag"].get("groups") or [0] * 10) if g}) + sum(1 for g in (r["tag"].get("groups") or [0] * 10) if not g) for r in ok], float)
    return {"n": len(ok), "place": round(float(pl.mean()), 3), "win": round(float((pl == 1).mean()), 3),
            "top": round(float((pl <= teams / 2).mean()), 3), "trade": round(S("dealt") / max(S("taken"), 1e-6), 3),
            "hit": round(S("hits") / max(S("shots"), 1.0), 3), "dps": round(S("dealt") / max(S("shots"), 1.0), 3),
            "cubes": round(S("cubes") / len(ok), 2), "alive_s": round(float(np.mean([r["ticks"] for r in ok])) * TICK_S, 1),
            "gas": round(float(np.mean([r["storm_death"] for r in ok])), 3),
            **{k: round(_nm(m.get(k, float("nan")) for m in bm), 4) for k in ("hidden", "engaged", "band", "clump", "range", "dealt_min", "cubes_end")
               for bm in [[beh_metrics(r) for r in ok]] if any(bm)}}


def kind_of(r):
    t = r.get("tag") or {}; k = t.get("kind", "")
    return "evalp:%s" % t.get("preset") if k == "evalp" else k if k in ("eval", "evalb") else "train"


STEP_S = 0.1   # env step (2 ticks)


def beh_metrics(r):
    """per-episode behaviour metrics from env_v5's counters (row["beh"]) and slider terms; nan when undefined"""
    b = r.get("beh")
    if not b: return {}
    al = max(b["alive"], 1.0); mt = b["mate"]; nan = float("nan"); mins = al * STEP_S / 60.0
    out = {"hidden": b["hidden"] / al, "bush": b["bush"] / al, "engaged": b["engaged"] / al, "gas": b["gas"] / al,
           "band": b["mate_band"] / mt if mt else nan, "clump": b["clump"] / mt if mt else nan,
           "range": b["range_ratio"] / b["near"] if b["near"] else nan, "dealt_min": b["dealt"] / mins, "cubes_end": b["cubes_end"]}
    for k, v in r["terms"].items():
        if k.startswith("s_"): out[k + "_min"] = v / mins
    return out


# per style: (metric, op, k, how): preset mean op (balanced mean x k | + k)
PASS_RULES = {"bush_camper": [("hidden", ">=", 1.5, "x")], "aggressive": [("engaged", ">=", 1.3, "x"), ("dealt_min", ">", 1.0, "x")],
              "cube_farmer": [("cubes_end", ">=", 1.3, "x"), ("gas", "<=", 0.005, "+")], "team_player": [("clump", "<=", 0.02, "+")],
              "marksman": [("range", ">=", 1.1, "x")], "third_partier": [("s_third_party_min", ">=", 1.3, "x")]}
PLACE_TOL = 0.5   # a style may cost at most half a rank on average


def _nm(xs):
    xs = np.array([x for x in xs if x == x], float)
    return float(xs.mean()) if len(xs) else float("nan")


def preset_report(rows):
    """{preset: {metric: mean, "d_<metric>": [paired mean diff vs balanced, SE], "pass": bool, "why": [...]}} of evalp rows"""
    ev = [r for r in rows if (r.get("tag") or {}).get("kind") == "evalp" and r["player"] == 0 and r["rank"] is not None and not r["restart"]]
    tp = team_place(ev); by = collections.defaultdict(dict)
    for r in ev:
        m = beh_metrics(r); m["place"] = tp[(r["seed"], r["player"])]; by[r["tag"]["preset"]][r["seed"]] = m
    base = by.get("balanced", {}); out = {}
    for p, seeds in sorted(by.items()):
        keys = sorted({k for m in seeds.values() for k in m}); o = {k: round(_nm(m.get(k, float("nan")) for m in seeds.values()), 4) for k in keys}
        o["n"] = len(seeds)
        for k in keys:
            d = [seeds[s][k] - base[s][k] for s in seeds if s in base and k in base[s] and seeds[s][k] == seeds[s][k] and base[s][k] == base[s][k]]
            if len(d) > 1: o["d_" + k] = [round(float(np.mean(d)), 4), round(float(np.std(d, ddof=1) / np.sqrt(len(d))), 4)]
        if p != "balanced" and base:
            bm = {k: _nm(m.get(k, float("nan")) for m in base.values()) for k in keys}; why = []
            for k, op, f, how in PASS_RULES.get(p, []) + [("place", "<=", PLACE_TOL, "+")]:
                tgt = bm.get(k, float("nan")) * f if how == "x" else bm.get(k, float("nan")) + f; v = o.get(k, float("nan"))
                if not (v >= tgt if op == ">=" else v > tgt if op == ">" else v <= tgt): why.append("%s %.3f %s %.3f" % (k, v, op, tgt))
            o["pass"], o["why"] = not why, why
        out[p] = o
    return out


def slider_slopes(rows):
    """OLS per metric on [1, slider weights] over training rows -> {metric: {slider: [slope, SE]}}, + "own" t-values"""
    tr = [r for r in rows if kind_of(r) == "train" and r.get("sliders") and r.get("beh")]
    if len(tr) < 30: return {"n": len(tr)}
    names = [k[2:] for k in tr[0]["terms"] if k.startswith("s_")]
    X = np.c_[np.ones(len(tr)), np.array([r["sliders"] for r in tr], float)]; M = [beh_metrics(r) for r in tr]; out = {"n": len(tr)}
    for k in sorted({k for m in M for k in m}):
        y = np.array([m.get(k, float("nan")) for m in M], float); ok = np.isfinite(y)
        if ok.sum() < X.shape[1] + 10: continue
        b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None); res = y[ok] - X[ok] @ b
        cov = (res @ res) / max(1, ok.sum() - X.shape[1]) * np.linalg.pinv(X[ok].T @ X[ok]); se = np.sqrt(np.maximum(np.diag(cov), 0))
        out[k] = {n: [round(float(b[i + 1]), 5), round(float(se[i + 1]), 5)] for i, n in enumerate(names)}
    out["own_t"] = {n: round(out["s_%s_min" % n][n][0] / max(out["s_%s_min" % n][n][1], 1e-12), 1) for n in names if "s_%s_min" % n in out}
    return out


def replay_stats(frames, chr_, groups, stats_of):
    """lead slope / aim error / range ratio of one replay's recorded player. stats_of(code) -> obs_v5.brawler_stats"""
    lead_i, lead_a, rng = [], [], []
    for k in range(len(frames) - 1):
        f, nx = frames[k], frames[k + 1]
        me = f["f"]; P, Q = np.array(f["p"], float), np.array(nx["p"], float)
        if not P[me, 4] or nx["f"] != me:
            continue
        st = stats_of(chr_[me]); sp = st[3] * 4000.0 / 20.0      # projectile units / tick
        rng_u = st[0] * 30.0 / 3.0 * UNITS_TILE                    # CastingRange (1/3 tiles) -> units
        foe = [p for p in range(10) if p != me and P[p, 4] and not (groups[p] and groups[p] == groups[me])]
        if not foe:
            continue
        d = np.hypot(P[foe, 0] - P[me, 0], P[foe, 1] - P[me, 1]); j = int(np.argmin(d)); t = foe[j]
        if d[j] < 12 * UNITS_TILE and rng_u > 0:
            rng.append(d[j] / rng_u)
        a = f["a"]; dt = max(1, nx["t"] - f["t"])
        if a[1] != 1 or sp <= 0 or d[j] > rng_u * 1.1 or not Q[t, 4]:
            continue
        p = P[t, :2] - P[me, :2]; v = (Q[t, :2] - P[t, :2]) / dt
        kk = sp * sp - v @ v
        if kk <= 1e-6:
            continue
        tt = (p @ v + np.sqrt((p @ v) ** 2 + kk * (p @ p))) / kk; q = p + v * tt
        bear = np.degrees(np.arctan2(p[1], p[0])); ideal = _wrap(np.degrees(np.arctan2(q[1], q[0])) - bear)
        lead_i.append(float(ideal)); lead_a.append(float(_wrap(a[2] - bear)))
    return lead_i, lead_a, rng


def summarize_leads(li, la):
    li, la = np.array(li), np.array(la)
    m = np.abs(li) > 2.0                                           # moving targets only (a lead is visible)
    if m.sum() < 10:
        return {"lead_n": int(m.sum())}
    keep = m & (np.abs(la) < 45)                                   # shots at another target / wild misses out
    s = float((la[keep] * li[keep]).sum() / max((li[keep] ** 2).sum(), 1e-9))
    return {"lead_n": int(keep.sum()), "lead": round(s, 3), "aim_err": round(float(np.median(np.abs(la[keep] - li[keep]))), 2)}


def scorecard(run, last=3000, stats_of=None):
    rows = [json.loads(l) for l in open(os.path.join(run, "episodes.jsonl"))][-last:]
    ev = os.path.join(run, "eval_episodes.jsonl")           # the trainers' eval rows (player 0), not in episodes.jsonl
    if os.path.exists(ev): rows += [json.loads(l) for l in open(ev)][-last:]
    out = {"run": os.path.basename(os.path.normpath(run)), "rows": len(rows), "groups": {}}
    by = collections.defaultdict(list)
    for r in rows:
        t = r.get("tag") or {}; by[(kind_of(r), t.get("team_size", 1), "all")].append(r)
        by[(kind_of(r), t.get("team_size", 1), (t.get("chr") or ["?"] * 10)[r["player"]])].append(r)
    for (k, ts, br), rs in sorted(by.items()):
        out["groups"]["%s/t%d/%s" % (k, ts, br)] = ep_stats(rs)
    if stats_of is None:
        import obs_v5 as O; stats_of = O.brawler_stats
    li, la, rg = [], [], []
    for fn in sorted(glob.glob(os.path.join(run, "replays", "u*.json")))[-30:]:
        c = json.load(open(fn)); a, b, r_ = replay_stats(c["frames"], c["chr"], c.get("groups") or [0] * 10, stats_of)
        li += a; la += b; rg += r_
    out["replays"] = {**summarize_leads(li, la), "range": round(float(np.median(rg)), 3) if rg else None}
    out["presets"] = preset_report(rows); out["slopes"] = slider_slopes(rows)
    return out


def _check():
    """synthetic replays: a shooter that aims at the target's position scores lead ~0, one that leads perfectly ~1"""
    st = lambda c: np.array([1.0, 0.72, 1, 0.75, 0, 1, 0, 0, 0.3, 0, 0, 0, 0], np.float32)   # projectile 3000 u/s, range 10 tiles
    def frames(leading):
        out, sp = [], 0.75 * 4000 / 20
        for k in range(60):
            P = [[0, 0, 1, 1, 0]] * 10; P = [list(x) for x in P]
            P[0] = [5000, 5000, 3000, 3000, 1]; ty = 4000 + 12 * 2 * k
            P[1] = [7000, ty, 3000, 3000, 1]                       # 2000 units right, walking down 12 units / tick
            p, v = np.array([2000.0, ty - 5000.0]), np.array([0.0, 12.0])
            kk = sp * sp - v @ v; tt = (p @ v + np.sqrt((p @ v) ** 2 + kk * (p @ p))) / kk; q = p + v * tt
            ang = np.degrees(np.arctan2(*(q if leading else p)[::-1])) % 360
            out.append({"t": 2 * k, "f": 0, "a": [0, 1, float(ang), 0], "p": P})
        return out
    for lead_, want in ((False, 0.0), (True, 1.0)):
        r = summarize_leads(*replay_stats(frames(lead_), ["X"] * 10, [0] * 10, st)[:2])
        assert abs(r["lead"] - want) < 0.05, (lead_, r)
    rows = [{"seed": 1, "player": p, "rank": rk, "truncated": False, "restart": False, "ticks": 400, "storm_death": 0,
             "terms": {"dealt": 2, "taken": 1, "hits": 1, "shots": 2, "cubes": 1}, "tag": {"kind": "k1", "team_size": 3, "groups": g}}
            for p, rk, g in ((0, 5, [1, 1, 0] + [0] * 7), (1, 2, [1, 1, 0] + [0] * 7))]
    s = ep_stats(rows); assert s["place"] == 2 and s["trade"] == 2 and s["hit"] == 0.5, s
    rng = np.random.default_rng(0); names = ["bush_camp", "collect", "push"]
    def row(kind, preset, seed, w, hidden, engaged, rank=5):   # 1000 alive steps; s_<slider> terms
        beh = {"alive": 1000, "bush": 300, "hidden": hidden * 1000, "engaged": engaged * 1000, "gas": 10, "mate": 0, "mate_band": 0, "clump": 0,
               "mate_tiles": 0, "near": 100, "range_ratio": 80, "dealt": 6.0, "box": 2.0, "cubes_max": 3, "cubes_end": 2}
        return {"seed": seed, "player": 0, "rank": rank, "truncated": False, "restart": False, "ticks": 2000, "storm_death": 0, "sliders": w,
                "terms": {"dealt": 6, "taken": 3, "hits": 9, "shots": 20, "cubes": 2, **{"s_" + k: hidden * w[0] if k == "bush_camp" else 0.1 for k in names}},
                "tag": {"kind": kind, "preset": preset, "groups": [0] * 10}, "beh": beh}
    ev = [row("evalp", "balanced", sd, [1, 1, 1], 0.1, 0.2) for sd in range(20)] + \
         [row("evalp", "bush_camper", sd, [2, 1, 0], 0.2 + 0.01 * rng.random(), 0.15, rank=5 + (sd % 2) * 0.5) for sd in range(20)] + \
         [row("evalp", "aggressive", sd, [0, 1, 2], 0.05, 0.22) for sd in range(20)]
    pr = preset_report(ev)
    assert pr["bush_camper"]["pass"] and not pr["aggressive"]["pass"] and pr["aggressive"]["why"][0].startswith("engaged"), pr["aggressive"]
    assert abs(pr["bush_camper"]["d_hidden"][0] - 0.105) < 0.01 and pr["bush_camper"]["n"] == 20 and "pass" not in pr["balanced"]
    W = rng.uniform(0, 2, (300, 3)); tr = [row("k1", None, k, list(w), 0.1 + 0.05 * w[0] + 0.01 * rng.normal(), 0.2) for k, w in enumerate(W)]
    sl = slider_slopes(tr)
    assert abs(sl["hidden"]["bush_camp"][0] - 0.05) < 0.005 and abs(sl["hidden"]["push"][0]) < 0.005, sl["hidden"]
    assert sl["own_t"]["bush_camp"] > 10 and kind_of(ev[0]) == "evalp:balanced" and ep_stats(ev[:20])["hidden"] == 0.1
    print("scorecard ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("run", nargs="?"); ap.add_argument("--last", type=int, default=3000)
    ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    if a.check or not a.run:
        _check()
    else:
        work = os.path.expanduser(os.environ.get("SIM_WORK", "~/nb68/rl"))   # = env_v5.WORK (no nbpc import here)
        run = a.run if os.path.isdir(a.run) else os.path.join(work, "runs", a.run)
        sc = scorecard(run, a.last); json.dump(sc, open(os.path.join(run, "scorecard.json"), "w"), indent=1)
        ev = sc["groups"].get("eval/t1/all") or next(iter(sc["groups"].values()))
        rp = sc["replays"]
        print("scorecard %s: place %s win %s trade %s hit %s dps %s cubes %s alive %ss | lead %s aim_err %s range %s -> %s" % (
            sc["run"], ev.get("place"), ev.get("win"), ev.get("trade"), ev.get("hit"), ev.get("dps"), ev.get("cubes"),
            ev.get("alive_s"), rp.get("lead"), rp.get("aim_err"), rp.get("range"), os.path.join(run, "scorecard.json")))
        if sc["presets"]:
            print("presets: " + ", ".join("%s %s (place %s%s)" % (p, "PASS" if v.get("pass") else "-" if "pass" not in v else "FAIL", v.get("place"),
                                                                 "; " + "; ".join(v["why"]) if v.get("why") else "") for p, v in sc["presets"].items()))
        if sc["slopes"].get("own_t"): print("slider own-term t (< 2 = no effect):", sc["slopes"]["own_t"])
