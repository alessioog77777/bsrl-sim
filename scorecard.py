"""Pro-level scorecard of a run (BSRL v2 plan item 10; HISTORY 82): what a strong Showdown player does, measured.

    python scorecard.py RUN [--last 3000]     # RUN = $SIM_WORK/runs/<run> (or a path); CPU only, reads files
    -> one summary line on stdout, details in RUN/scorecard.json

From episodes.jsonl (every finished agent: training + evals), grouped by battle kind (eval / evalb / train),
team size and brawler:
  place      own placement; in teams the team's (best member rank among that battle's rows)
  win, top   share placing 1st / in the top half of the teams
  trade      damage dealt / damage taken (sim term units)
  hit, dps   hits / shots, damage per shot (ammo efficiency: wasted shots lower it)
  cubes, alive_s, gas   cubes picked, seconds survived, share killed by the gas
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
            "gas": round(float(np.mean([r["storm_death"] for r in ok])), 3)}


def kind_of(r):
    k = (r.get("tag") or {}).get("kind", "")
    return k if k in ("eval", "evalb") else "train"


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
