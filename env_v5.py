"""Randomised Showdown battles on top of nirkt's MAEnv (run in WSL after `source rl/env.sh`).

    python3 -m env_v5 --n 12 --minutes 5 --agents 10      # benchmark: random actions, one summary line + bench.jsonl
    python3 -m env_v5 --n 3 --minutes 15 --native 1 --episodes nat.jsonl && python3 -m env_v5 --norm nat.jsonl
                                                          # slider_norm factors from native-AI battles (NORM_MEASURED)

battle_fn(): every battle a random Showdown map, a random brawler per player, solo / duo (5 x 2) / trio (3 x 3 + 1
solo: the sim has 10 players, the real mode 12) teams, and `agents` policy-controlled slots (the rest = native AI).
The battle is still Solo Showdown underneath (teams only change who can hurt whom), so MAEnv's rank / place reward is
per player; `Env` holds a grouped agent's final transition until its team's rank is known and swaps in the team's
place / win (`team_ranks`). Solo players: unchanged.
"""
import argparse, json, os, re, sys, time
import numpy as np
from nbpc import diversity as D, proto as P
from nbpc.maenv import MAEnv
import obs_v5 as O

SKIP_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "skip.json")   # {"brawlers": [...], "locations": [...]}


def _skip():
    return json.load(open(SKIP_FILE)) if os.path.exists(SKIP_FILE) else {}


def heroes():
    """playable brawlers = Hero rows that own cards (drops transforms, bosses and the hashed unreleased rows)"""
    owners = {r["Target"] for r in D._rows("cards.csv")}
    bad = set(_skip().get("brawlers", []))
    return [n for n, (_, r) in D.table("characters.csv").items()
            if r.get("Type") == "Hero" and r.get("Disabled") != "true" and n in owners
            and not re.fullmatch(r"[0-9a-f]{40}", n) and n not in bad]


def locations():
    bad = set(_skip().get("locations", []))
    return [n for n, (_, r) in D.table("locations.csv").items()
            if r.get("GameModeVariation") == "Showdown" and r.get("Disabled") != "true" and n not in bad]


WORK = os.path.expanduser(os.environ.get("SIM_WORK", "~/nb68/rl"))   # run outputs + generated maps (env.sh sets it)
GEN_DIR = os.path.join(WORK, "genmaps")   # framegen/genmap.py OUT --seed N: OUT/game = game files with OUT's layout
GEN_BASE = "Survival612"                              # as this location's map (genmap's default base)


def use_genmaps(n_inst, dirs=None):
    """instances 0..n_inst-1 boot from a generated map's game dir (one layout per instance, round robin); call before
    Env(). Returns {instance: dir}. On those instances location GEN_BASE plays the generated layout."""
    from nbpc import instance as NI
    if dirs is None:   # a PayloadTrack tile ('b', from a Payload source map) crashes the boot (rc 139)
        dirs = sorted(os.path.join(GEN_DIR, d, "game") for d in os.listdir(GEN_DIR)) if os.path.isdir(GEN_DIR) else []
        dirs = [d for d in dirs if "b" not in open(os.path.join(os.path.dirname(d), "layout.txt")).read()]
    NI.APK_BY_IDX.clear(); NI.APK_BY_IDX.update({i: dirs[i % len(dirs)] for i in range(n_inst)} if dirs else {})
    return dict(NI.APK_BY_IDX)


def make_battle_fn(agents=10, seed0=10_000_000, roster=None, locs=None, p_team=0.5, gen=None, p_gen=0.5):
    """gen = use_genmaps()'s {instance: dir}: there a battle plays the generated layout with probability p_gen"""
    roster = roster or heroes(); locs = locs or locations(); gen = gen or {}
    seeds = iter(range(seed0, seed0 + 10 ** 9))

    def battle_fn(rng, i):
        br = [roster[int(rng.integers(len(roster)))] for _ in range(10)]
        g = gen.get(i) if i in gen and rng.random() < p_gen else None
        loc = GEN_BASE if g else locs[int(rng.integers(len(locs)))]
        if not g and i in gen and loc == GEN_BASE: loc = locs[(locs.index(loc) + 1) % len(locs)]   # shipped map not on this instance
        div = D.make_div(location=loc, chr=br)
        size = int(rng.integers(2, 4)) if rng.random() < p_team else 1
        groups = [0] * 10                                    # group id per player, 0 = alone (host: nbhost_rl)
        if size > 1:
            for k, p in enumerate(rng.permutation(10)[:10 // size * size]): groups[int(p)] = k // size + 1
            div["team_groups"] = groups
        ctl = sorted(int(p) for p in rng.choice(10, size=agents, replace=False))
        return {"seed": next(seeds), "ctl": ctl, "end": "agents", "div": div,
                "tag": {"loc": os.path.basename(os.path.dirname(g)) if g else loc, "chr": br, "team_size": size, "groups": groups}}
    return battle_fn


SL = O.SLIDERS
_COVER = [O.SEG.index(c) for c in ("indestructible", "destructible_wall", "fragile_wall", "bouncer", "gravity", "destructible")]
_PL = ("x", "y", "hp", "max_hp", "damage_dealt", "bomb_damage", "damage_taken", "gas_damage", "box_damage", "cubes",
       "cubes_total", "in_gas", "kills")
# behaviour counters per player and battle (row["beh"]; scorecard's per-preset / slider-slope metrics), steps unless noted:
# alive, on a bush tile, hidden by a bush from an enemy whose screen it is on, enemy in range with >= 40 % hp, in gas,
# a living mate, mate 1.5-4 tiles, mate < 1 tile, mate distance (tiles, summed), visible enemy <= 12 tiles, its distance /
# own range (summed), damage dealt / 1000, box damage / 1000, then cubes held: max, at the end; then near the gas (gas_safe)
BEH = ("alive", "bush", "hidden", "engaged", "gas", "mate", "mate_band", "clump", "mate_tiles", "near", "range_ratio",
       "dealt", "box", "cubes_max", "cubes_end", "gas_near")
# per-slider factors on the terms so each is worth ~0.5 per battle at weight 1 (0.5 / mean per-battle sum of |term| per
# player, norm_from; against the 8 of place + win). Measured 2026-11-23 on the native AI (the BC target): `env_v5 --n 3
# --minutes 15 --native 1` = 778 battles / 7769 player rows, random maps / brawlers, half in teams, then `--norm`.
# Re-measure when the terms change or the policy is far from the native AI.
NORM_MEASURED = {"bush_camp": 0.232, "collect": 0.0753, "protect": 0.3373, "retreat": 0.2011, "kite": 0.1423, "peek": 0.0525,
                 "focus": 0.0462, "aggressive": 0.0341, "third_party": 2.444, "bush_control": 0.5462, "leash": 1.2308,
                 "gas_safe": 3.2311}


def slider_norm(spec):
    """--slider-norm: '' = all 1.0 (terms as computed), 'measured' = NORM_MEASURED, else a JSON file {slider: factor}"""
    d = {} if not spec else NORM_MEASURED if spec == "measured" else json.load(open(spec))
    return np.array([float(d.get(k, 1.0)) for k in SL])


def sample_sliders(rng, n, mode="uniform"):
    """(n, len(SL)) per-battle slider weights 0-2. uniform: U(0, 2) each (the original sampler, same draws). mix (train
    --slider-sample mix): per player 40 % uniform, 40 % a preset + N(0, 0.3) (density where people steer), 10 % an exact
    preset, 10 % sparse (all 0.5, one or two sliders at 2: single-style coverage)"""
    if mode == "uniform": return rng.uniform(0, 2, (n, len(SL)))
    names = list(O.PRESETS); w = np.zeros((n, len(SL)))
    for p in range(n):
        u = rng.random(); pre = O.preset_vec(names[int(rng.integers(len(names)))])
        if u < 0.4: w[p] = rng.uniform(0, 2, len(SL))
        elif u < 0.8: w[p] = np.clip(pre + rng.normal(0, 0.3, len(SL)), 0, 2)
        elif u < 0.9: w[p] = pre
        else: w[p] = 0.5; w[p, rng.choice(len(SL), int(rng.integers(1, 3)), replace=False)] = 2.0
    return w


def slider_map(sl, tiles, W, H):
    """sl["bush"] / sl["nearcover"] (cover within 2 tiles) from the host's tile ids (H*W,), in place; Env calls it again
    whenever the live tiles change (a bush or wall destroyed)"""
    t = O._tile_lut()[np.clip(np.asarray(tiles).reshape(H, W) % 1000000 + 1, 0, 255)] if tiles is not None else np.zeros((H, W), np.uint8)
    cover = np.pad(np.isin(t, _COVER), 2)
    near = np.zeros((H, W), bool)
    for dy in range(5):
        for dx in range(5): near |= cover[dy:dy + H, dx:dx + W]
    sl["bush"], sl["nearcover"] = t == O.SEG.index("bush"), near
    return sl


def slider_state(tiles, W, H, brawlers, rng, n=10, w=None, mode="uniform"):
    """per-battle state of the slider rewards: weights 0-2 per player and slider (w: (len(SL),) for every player or
    (n, len(SL)); None = sample_sliders(rng, n, mode)), map masks, trackers, behaviour counters"""
    # own attack range (units); floor 600 = 2 tiles, the shortest real melee reach (Rock has no CastingRange)
    rg = [max(float(O.brawler_stats(b)[0]) * 3000, 600.0) if b else 2400.0 for b in (list(brawlers) + [None] * n)[:n]]
    w = sample_sliders(rng, n, mode) if w is None else np.broadcast_to(np.asarray(w, np.float64), (n, len(SL))).copy()
    return slider_map({"w": w, "range": np.array(rg),
                       "hid": np.zeros(n, bool), "low": np.zeros(n, bool), "hp": np.full(n, np.nan),
                       "held": np.zeros(n), "was": np.zeros(n, bool), "mhit": np.zeros(n),
                       "checked": np.zeros((n, H, W), bool), "sum": np.zeros((n, len(SL))), "sumw": np.zeros((n, len(SL))),
                       "asum": np.zeros((n, len(SL))),
                       "ret": np.zeros(n), "beh": np.zeros((n, len(BEH)))}, tiles, W, H)


def gas_margin(x, y, ring, W, H):
    """tiles from the gas edge per position (< 0 = in the gas) and the safe zone's half-size: the host's storm rule
    (env.c in_gas_tile: a tile is gas iff min(tx, ty, W-1-tx, H-1-ty) < ring), ring = the latched storm step
    (NbeResp.reserved[0], 1 tile per step)"""
    tx, ty = np.clip(np.asarray(x) // 300, 0, W - 1), np.clip(np.asarray(y) // 300, 0, H - 1)
    return np.minimum(np.minimum(tx, ty), np.minimum(W - 1 - tx, H - 1 - ty)) - ring, (min(W, H) - 1) / 2.0 - ring


def slider_terms(sl, pl, n=10, ring=0):
    """(n, len(SLIDERS)) slider reward terms of one env step, order SLIDERS; pl = the response's players. Units as MAEnv's terms
    (1000 damage = 1), before the weights and slider_norm. A respawning player (reserved[3] > 0) counts as out. Enemy
    damage taken `te` = damage_taken - gas_damage (the host's damage_taken includes the gas). The host reports damage per
    player, not per target, so `focus`, `protect`, `third_party` and `bush_control` use "that enemy took damage this step"
    instead of "I hit it". Bushes / cover = sl's map masks, the live tiles (Env refreshes them every MAP_EVERY steps).
      bush_camp     0.002 per step hidden by a bush from an enemy that has me on its screen, + damage dealt the step after
      collect       out of the gas: 0.2 x box damage + cubes picked up; - cubes lost (knock-out / elimination drops them)
      protect       damage dealt while a mate within 6 tiles took enemy damage in the last 10 steps (peel)
      retreat       hp regained while low (latched < 40 % until 80 %); not the respawn refill, not a cube's +hp
      kite          damage dealt taking no enemy damage, the nearest enemy >= 0.7 x own range away
      peek          damage dealt within 2 tiles of cover
      focus         damage dealt while the lowest-hp enemy in range is taking enemy damage
      aggressive    damage dealt + 0.5 per kill + 0.002 per step with an enemy in range while >= 40 % hp
      third_party   damage dealt taking none, an enemy in range taking more than I dealt
      bush_control  0.005 per bush tile first within 3 tiles this battle + damage dealt while an enemy hidden in a bush in
                    my range takes damage (checking bushes from range)
      leash         0.002 per step with the nearest living mate 1.5-4 tiles away, -0.002 under 1 tile (clumping into AoE)
      gas_safe      -0.01 per step in the gas, -0.002 per step within 2 tiles of it while the safe zone is >= 4 tiles
                    across from the centre (the endgame zone is all near the gas); ring = the latched storm step
    Also accumulates sl["beh"] (BEH)."""
    A = np.array([[getattr(pl[p], k) for k in _PL] + [pl[p].reserved[2], pl[p].alive and not pl[p].eliminated and pl[p].reserved[3] <= 0]
                  for p in range(n)], np.float64)
    x, y, hp, mhp, dealt, bomb, taken, gasd, box, cubes, held, ingas, kills, team, alive = A.T
    alive = alive > 0; was = sl["was"]; H, W = sl["bush"].shape; dk = (dealt + bomb) / 1000.0; idx = np.arange(n)
    te = np.maximum(taken - gasd, 0.0); ingas = ingas > 0
    d = np.hypot(x[None, :] - x[:, None], y[None, :] - y[:, None])
    both = alive[:, None] & alive[None, :]
    en = both & (team[:, None] != team[None, :]); mate = both & (team[:, None] == team[None, :]) & (idx[:, None] != idx[None, :])
    tx, ty = np.clip(x // 300, 0, W - 1).astype(int), np.clip(y // 300, 0, H - 1).astype(int)
    inb = sl["bush"][ty, tx]
    # screen[q, p]: p is on q's screen (the camera rule of nbpc.fair); sees[q, p]: and not hidden in a bush (> 900 away)
    camx = np.clip(x, 3000.0, W * 300 - 3000.0); ry = y[None, :] - y[:, None]
    screen = (np.abs(x[None, :] - camx[:, None]) < 4500) & (ry > -3500) & (ry < 2900)
    sees = screen & ~(inb[None, :] & (d > 900))
    hid = inb & alive & ~(en & sees).any(0) & (en & screen).any(0)   # the bush is what hides me from someone
    d_en = np.where(en, d, np.inf); near = d_en.min(1); has = np.isfinite(near)
    inr = en & (d <= sl["range"][:, None])
    low_hp = np.where(inr, (hp / np.maximum(mhp, 1))[None, :], np.inf).argmin(1)
    sl["mhit"] = np.where(alive & (te > 0), 10, np.maximum(sl["mhit"] - 1, 0))   # 10 steps = 1 s of "under fire"
    peel = (mate & (d < 1800) & (sl["mhit"] > 0)[None, :]).any(1)               # 6 tiles: near enough to help
    gain = np.nan_to_num(np.maximum(hp - sl["hp"], 0)) / 1000.0 * (cubes == 0)  # sl["hp"] nan while out / respawning
    low = sl["low"] | (hp < 0.4 * mhp)
    engaged = inr.any(1) & (hp >= 0.4 * mhp)                                    # not while low: no suicide rushes
    dm = np.where(mate, d, np.inf).min(1); hm = np.isfinite(dm) & alive
    gm, zone = gas_margin(x, y, ring, W, H); gnear = alive & (gm >= 0) & (gm < 2) & (zone >= 4)
    others = te[None, :] > (dealt + bomb)[:, None]       # others[p, q]: q took more than p dealt = someone else hit it too
    bushhit = (inr & inb[None, :] & ~sees & (te > 0)[None, :]).any(1)          # sees[p, q]: p sees q
    new = np.zeros(n)
    for p in np.nonzero(alive)[0]:                       # bush tiles within 3 tiles, first time this battle
        ys, xs = slice(max(ty[p] - 3, 0), ty[p] + 4), slice(max(tx[p] - 3, 0), tx[p] + 4)
        fresh = sl["bush"][ys, xs] & ~sl["checked"][p, ys, xs]
        new[p] = fresh.sum(); sl["checked"][p, ys, xs] |= fresh
    held = np.where(alive, held, 0.0); lost = np.minimum(held - sl["held"] - cubes, 0) * was   # an elimination: held -> 0
    T = np.stack([
        0.002 * hid + dk * sl["hid"],                                             # bush_camp
        ~ingas * (0.2 * box / 1000.0 + cubes),                                    # collect (+ lost below)
        dk * peel,                                                                # protect
        gain * low,                                                               # retreat
        dk * (te == 0) * (has & (near >= 0.7 * sl["range"])),                     # kite
        dk * sl["nearcover"][ty, tx],                                             # peek
        dk * inr.any(1) * (te[low_hp] > 0),                                       # focus
        dk + 0.5 * kills + 0.002 * engaged,                                       # aggressive
        dk * (te == 0) * (inr & others).any(1),                                   # third_party
        0.005 * new + dk * bushhit,                                               # bush_control
        0.002 * (hm & (dm >= 450) & (dm <= 1200)) - 0.002 * (hm & (dm < 300)),    # leash: 1.5-4 tiles (Pyla), clump < 1 tile
        -0.01 * ingas - 0.002 * gnear], 1) * alive[:, None]                       # gas_safe
    T[:, 1] += lost
    dv = np.where(en & sees, d, np.inf).min(1); nv = alive & (dv < 3600)      # a visible enemy within 12 tiles
    b = sl["beh"]
    b[:, :13] += np.stack([alive, inb & alive, hid, engaged, ingas & alive, hm, hm & (dm >= 450) & (dm <= 1200),
                           hm & (dm < 300), np.where(hm, dm, 0) / 300.0, nv, np.where(nv, dv, 0) / sl["range"], dk * alive, box / 1000.0], 1)
    b[:, 13] = np.maximum(b[:, 13], held); b[:, 14] = held; b[:, 15] += gnear
    sl["hid"], sl["hp"], sl["low"] = hid, np.where(alive, hp, np.nan), low & (hp < 0.8 * mhp)
    sl["held"], sl["was"] = held, alive
    return T


def player_ranks(r):
    """(10,) rank each player has / would get if its episode ended now (MAEnv._finish_step's rule), (10,) eliminated"""
    done, alive = bool(r.done), int(r.alive or 0); pr = np.zeros(10); dead = np.zeros(10, bool)
    for p in range(10):
        pl = r.pl[p]
        if pl.eliminated or not pl.alive: pr[p] = pl.elim_rank if pl.elim_rank > 0 else 10; dead[p] = True
        elif done: pr[p] = 1 if (r.winner == p or alive <= 1) else max(1, alive)
        else: pr[p] = (1 + alive) / 2.0
    return pr, dead


def team_ranks(groups, pr, dead, over):
    """(10,) placement per player: group 0 = own rank; a group = the best rank among its members (alive members outrank
    eliminated ones), nan while a member lives and the battle goes on.
    shortcut: a battle cut short (end "agents" once every agent is out, max_ticks) ranks a living native mate by MAEnv's
    truncation estimate (1 + alive) / 2; exact only with end "done"."""
    g, out = np.asarray(groups), np.asarray(pr, np.float64).copy()
    for k in set(g[g > 0].tolist()):
        m = g == k; out[m] = out[m].min() if over or dead[m].all() else np.nan
    return out


class Env(MAEnv):
    """MAEnv + a log of every instance restart with the battle that caused it + the slider rewards: every player gets
    weights 0-2 per battle (tag "sliders": fixed, one vector for every player or one per player; else sample_sliders
    with slider_sample); reward += slider_scale x sum(weight x slider_norm x term). The transitions' terms carry the raw
    terms ("s_<slider>"); the episode rows carry the raw sums ("s_<slider>"), the reward they gave ("sn_<slider>"), the
    weights ("sliders"), the sums of |term| ("sa_<slider>", slider_norm's measure) and the behaviour counters ("beh", BEH)."""
    crashes = []
    slider_scale = 1.0
    slider_sample = "uniform"     # train --slider-sample
    slider_norm = np.ones(len(SL))   # train --slider-norm (env_v5.slider_norm)
    dead_aids = set()   # agents respawning right now (host: duo / trio knock-out with a living mate, 15 s): dead(ids)
    resp = {"out": 0, "back": 0, "wiped": 0}   # knock-outs into respawn, respawns, waits ended by the team's wipe
    record = False    # True: battles with cfg "native_ctl" put the native AI's action into the transition terms ("_act": BC labels)
    # sim_plan B3 (train_fast --cube-shaping K): the flat per-cube reward becomes potential shaping on the cubes held,
    # K x (gamma x held_after - held_before), held = NbePlayer.cubes_total (the host zeroes it at a team knock-out, as Duo
    # Showdown does) and 0 once eliminated, so losing the cubes gives their reward back.
    cube_k, cube_gamma = 0.0, 0.999
    # the slider rewards' bush / cover masks follow the live tiles: F_MAP every MAP_EVERY steps per instance, masks rebuilt
    # when map_hash changed. 10 steps = 1 s at ticks 2: a destroyed bush keeps paying for <= 1 s. Cost (measured): F_MAP ~45 us
    # of host time per request (us_state 575 vs 530 us), ~4.5 us per step at k = 10; the rebuild (~120 us) only when a tile
    # changed (live_check: ~1.7 refreshes per battle)
    MAP_EVERY = 10

    def _flags(self):
        return super()._flags() | (P.F_REC if self.record else 0) | (P.F_MAP if getattr(self, "_map_now", False) else 0)

    def _send_step(self, i, acts):
        s = self.I[i]; s.nstep = getattr(s, "nstep", 0) + 1; self._map_now = s.nstep % self.MAP_EVERY == 0
        try: super()._send_step(i, acts)
        finally: self._map_now = False

    def _labels(self, s, r):
        """player -> [move, fire, aim, dist] the native AI did this step: the rules of nirkt's collect_demos4 (move = the
        displacement's 16-direction bucket, fire = first unblocked controller record, super wins)"""
        rec = np.frombuffer(s.inst.mm, np.int32, r.rec_n * 8, P.OFF_REC).reshape(-1, 8) if r.rec_n else np.zeros((0, 8), np.int32)
        jump = {int(r.jumps[k][0]) for k in range(r.njumps)}; out = {}
        for p in s.ep:
            pl = r.pl[p]; cid = int(pl.char_id); x0, y0 = s.pos[p]; dx, dy = float(pl.x - x0), float(pl.y - y0); act = [0, 0, 0, 0]
            if cid not in jump and cid not in s.pjump and dx * dx + dy * dy >= (10.0 * self.ticks) ** 2 and pl.alive:
                act[0] = 1 + int(np.round(np.arctan2(dy, dx) / (np.pi / 8))) % 16
            for c in rec[(rec[:, 1] == cid) & (rec[:, 2] == 1)]:
                if c[6] == 1: continue
                f = min(1.0, max(0.0, (float(c[0]) - s.ptick) / max(1.0, float(r.tick - s.ptick))))
                tx, ty = float(c[3] - (x0 + f * dx)), float(c[4] - (y0 + f * dy)); ang = int(round(np.degrees(np.arctan2(ty, tx)))) % 360
                if c[5] == 1 and act[1] != 2: act[1:] = [2, ang, min(7, max(0, int(np.ceil(np.hypot(tx, ty) / 2600.0 * 8)) - 1))]
                elif c[5] == 0 and act[1] == 0: act[1], act[2] = 1, ang
            out[p] = act
        return out

    def _restart(self, i, why):
        self.crashes.append({"why": str(why)[:200], **(self.I[i].cfg or {}).get("tag", {})})
        self._trans += getattr(self.I[i], "hold", {}).values(); self.I[i].hold = {}   # crashed: own placement only
        return super()._restart(i, why)

    def _start_battle(self, i, r):
        super()._start_battle(i, r)
        s = self.I[i]; s.hold = {}                       # player -> its final transition, held until the team's rank is known
        tag = s.cfg.get("tag") or {}
        s.sl = slider_state(s.map_tiles, s.W, s.H, tag.get("chr", []), self.rng, w=tag.get("sliders"), mode=self.slider_sample)
        s.map_hash = int(r.map_hash)
        s.pos = {p: (r.pl[p].x, r.pl[p].y) for p in range(10)}; s.ptick = r.tick; s.pjump = set(); s.wprev = set()
        s.cprev = {p: int(r.pl[p].cubes_total) for p in range(10)}
        s.hud = self._hud(s, r)

    def _hud(self, s, r):
        """the two Showdown HUD numbers per player: own cubes held (cubes_total: 0 after a team knock-out, as B3), teams
        left (a group with a member not eliminated, or a solo player still in)"""
        g = (s.cfg.get("tag") or {}).get("groups") or [0] * 10; inn = [not r.pl[p].eliminated for p in range(10)]
        left = len({g[p] for p in range(10) if inn[p] and g[p]}) + sum(1 for p in range(10) if inn[p] and not g[p])
        return [int(r.pl[p].cubes_total) for p in range(10)], left

    def dead(self, ids):
        """(B,) bool: the agent is respawning (its observation is all zeros, its actions are ignored by the host)"""
        return np.array([int(x) in self.dead_aids for x in ids], bool)

    def _finish_step(self, i, r):
        s = self.I[i]; sl = s.sl
        if getattr(r, "map_written", 0) and int(r.map_hash) != s.map_hash:   # a tile changed (bush / wall destroyed): live masks
            s.map_tiles = s.inst.map_tiles(); s.map_hash = int(r.map_hash); slider_map(sl, s.map_tiles, s.W, s.H)
        T = slider_terms(sl, r.pl, ring=int(r.reserved[0])) if self.slider_scale else np.zeros(np.shape(sl["w"]))   # speed round: off = no reward, skip it (s_* logs 0)
        Tw = T * sl["w"] * self.slider_norm * self.slider_scale; extra = Tw.sum(1); sl["sum"] += T; sl["sumw"] += Tw; sl["ret"] += extra
        sl["asum"] += np.abs(T)
        n0, e0 = len(self._trans), len(self.episodes)
        lab = self._labels(s, r) if self.record and s.cfg.get("native_ctl") else {}
        s.pos = {p: (r.pl[p].x, r.pl[p].y) for p in range(10)}; s.ptick = r.tick; s.pjump = {int(r.jumps[k][0]) for k in range(r.njumps)}
        g = np.array((s.cfg.get("tag") or {}).get("groups") or [0] * 10)
        if g.any(): pr, dead = player_ranks(r); b0 = s.battle   # read r before a reset reuses it
        wait = {p for p in range(10) if r.pl[p].reserved[3] > 0}; agents = dict(s.agents); elim = {p for p in range(10) if r.pl[p].eliminated}
        back = s.wprev - wait                            # respawned this step (or wiped out with the team)
        self.resp["out"] += len(wait - s.wprev); self.resp["wiped"] += len(back & elim); self.resp["back"] += len(back - elim)
        for p, aid in agents.items(): (self.dead_aids.add if p in wait else self.dead_aids.discard)(aid)
        s.wprev = wait
        shaped = {}; s.hud = self._hud(s, r)             # cond's HUD numbers for the next poll (a reset sets them anew)
        if self.cube_k:                                  # B3, from r before super() may reset the instance
            for p in list(s.ep):
                pl = r.pl[p]; after = 0 if (pl.eliminated or not pl.alive) and p not in wait else int(pl.cubes_total)
                shaped[p] = self.cube_k * (self.cube_gamma * after - s.cprev.get(p, 0)); s.cprev[p] = after
        super()._finish_step(i, r)                       # may reset the instance: sl stays this battle's
        for k in range(n0, len(self._trans)):
            aid, rew, fin, t, rank = self._trans[k]; p = aid & 15
            if p in back - elim: rew -= self.w.get("hp", 0) * t["hp"]; t = {**t, "hp": 0.0}   # respawn refill is no hp gain
            if p in shaped: rew += shaped[p] - self.w.get("cubes", 0) * t["cubes"]; t = {**t, "cube_shaped": shaped[p]}
            if fin: self.dead_aids.discard(aid)
            self._trans[k] = (aid, rew + float(extra[p]), fin, {**t, **{"s_" + m: float(v) for m, v in zip(SL, T[p])}, "_act": lab.get(p), "dead": float(p in wait)}, rank)
        if g.any():                                      # duo / trio: placement = the team's (Solo Showdown underneath)
            for k in range(len(self._trans) - 1, n0 - 1, -1):
                p = self._trans[k][0] & 15
                if self._trans[k][2] and g[p]: s.hold[p] = self._trans.pop(k)
            tr = team_ranks(g, pr, dead, s.battle != b0)
            for p in [p for p in s.hold if not np.isnan(tr[p])]:
                aid, rew, fin, t, rank = s.hold.pop(p); place = (10 - tr[p]) / 9.0; win = float(tr[p] == 1)
                rew += self.w.get("place", 0) * (place - t["place"]) + self.w.get("win", 0) * (win - t["win"])
                self._trans.append((aid, rew, fin, {**t, "team_place": place}, rank))
        for row in self.episodes[e0:]:
            p = row["player"]; row["ret"] += float(sl["ret"][p]); row["sliders"] = [round(float(v), 3) for v in sl["w"][p]]
            row["terms"].update({"s_" + m: round(float(v), 4) for m, v in zip(SL, sl["sum"][p])})
            row["terms"].update({"sn_" + m: round(float(v), 4) for m, v in zip(SL, sl["sumw"][p])})
            row["terms"].update({"sa_" + m: round(float(v), 4) for m, v in zip(SL, sl["asum"][p])})
            row["beh"] = {k: round(float(v), 3) for k, v in zip(BEH, sl["beh"][p])}

    def cmd_vel(self, ids):
        """(B, 2) tiles / tick: the agents' last commanded move x their brawler speed (sim_plan A2; what the agent uses live
        as its own velocity: BSRL brain.player / play_live). 0 when stopped or before the first action of a battle."""
        by = {s.battle: s for s in self.I}; out = np.zeros((len(ids), 2), np.float32)
        for k, a in enumerate(ids):
            s, p = by[int(a) >> 4], int(a) & 15; act = s.acts.get(p)
            if act is not None and int(act[0]) > 0:
                sp = O.brawler_stats(s.cfg["tag"]["chr"][p])[1] / 6.0   # stats[1] = Speed / 1000; units/s / 300 / 20 ticks
                t = np.deg2rad((int(act[0]) - 1) * 22.5); out[k] = sp * np.cos(t), sp * np.sin(t)
        return out

    def cond(self, ids):
        """(B, N_COND) conditioning vectors (obs_v5.cond) of the agents of the last poll()"""
        by = {s.battle: s for s in self.I}
        out = []
        for a in ids:
            s, p = by[int(a) >> 4], int(a) & 15; tag = s.cfg["tag"]; ts = tag["team_size"] if tag["groups"][p] else 1
            cubes, left = getattr(s, "hud", (None, None))
            out.append(O.cond(tag["chr"][p], s.sl["w"][p], ts, cubes[p] if cubes else 0, left))
        return np.stack(out) if out else np.zeros((0, O.N_COND), np.float32)


def _check():
    from types import SimpleNamespace as NS
    def players(**over):
        pl = [NS(x=1650 + 3000 * (p % 5), y=1650 + 6000 * (p // 5), hp=3000, max_hp=4000, damage_dealt=0, bomb_damage=0, damage_taken=0,
                 box_damage=0, cubes=0, cubes_total=0, gas_damage=0, in_gas=0, kills=0, reserved=[0, 0, p, 0], alive=1, eliminated=0) for p in range(10)]
        for k, v in over.items(): setattr(pl[int(k[1])], k[3:], v)
        return pl
    cub, left = Env._hud(None, NS(cfg={"tag": {"groups": [1, 1, 1, 2, 2, 2, 3, 3, 3, 0]}}),   # cond's HUD numbers
                         NS(pl=players(p0_eliminated=1, p3_eliminated=1, p4_eliminated=1, p5_eliminated=1, p9_cubes_total=4)))
    assert left == 3 and cub[9] == 4 and cub[0] == 0, (left, cub)
    tiles = np.full((40, 60), 27000000); bush = D.table("tiles.csv")["Forest"][0]; wall = D.table("tiles.csv")["Indestructible"][0]
    tiles[5, 5] = bush; tiles[20:23, 30] = bush; tiles[25, 10] = wall
    sl = slider_state(tiles.ravel(), 60, 40, ["Sniper"] * 10, np.random.default_rng(0))
    assert sl["w"].shape == (10, len(SL)) and 0 <= sl["w"].min() and sl["w"].max() <= 2 and sl["range"][0] == 3000
    assert sl["bush"].sum() == 4 and sl["nearcover"][23, 8] and sl["nearcover"][27, 12] and not sl["nearcover"][28, 10]
    t2 = tiles.copy(); t2[25, 10] = 27000000; t2[5, 5] = 27000000; s2 = slider_map(dict(sl), t2.ravel(), 60, 40)   # destroyed
    assert s2["bush"].sum() == 3 and not s2["bush"][5, 5] and not s2["nearcover"][23, 8] and sl["bush"][5, 5]
    col = lambda T, name: T[:, SL.index(name)]
    fresh = lambda: slider_state(tiles.ravel(), 60, 40, ["Sniper"] * 10, np.random.default_rng(0))
    T = slider_terms(sl, players())                      # p0 on the bush tile (5, 5), on p1's screen 10 tiles away: hidden
    assert col(T, "bush_camp")[0] == 0.002 and col(T, "bush_camp")[1:].sum() == 0 and col(T, "bush_control")[0] == 0.005
    assert sl["beh"][0, BEH.index("hidden")] == 1 and sl["beh"][0, BEH.index("bush")] == 1 and sl["beh"][1, BEH.index("bush")] == 0
    T = slider_terms(sl, players(p0_damage_dealt=500, p0_box_damage=1000, p0_cubes=1, p0_cubes_total=1, p1_damage_taken=800))
    assert np.isclose(col(T, "bush_camp")[0], 0.502) and np.isclose(col(T, "collect")[0], 1.2) and col(T, "bush_control")[0] == 0
    assert col(T, "kite")[0] == 0.5 and col(T, "focus")[0] == 0.5 and col(T, "third_party")[0] == 0.5 and col(T, "peek")[0] == 0
    assert col(T, "kite")[1] == 0 and col(T, "protect").sum() == 0 and np.isclose(col(T, "aggressive")[0], 0.502)   # + p1 in range
    g = fresh(); slider_terms(g, players(p0_eliminated=1, p2_eliminated=1, p3_eliminated=1, p4_eliminated=1, p5_eliminated=1,
                                          p6_eliminated=1, p7_eliminated=1, p8_eliminated=1, p9_eliminated=1, p1_x=15000))
    assert g["beh"][0, BEH.index("hidden")] == 0 and g["beh"][1, BEH.index("alive")] == 1   # A4: a bush with nobody around is no hiding
    gas = players(p0_damage_dealt=500, p1_damage_taken=800, p1_gas_damage=800, p0_box_damage=1000, p0_in_gas=1)
    T = slider_terms(fresh(), gas)                       # A2: gas damage is no enemy hit; A5: no farming in the gas
    assert col(T, "third_party")[0] == 0 and col(T, "focus")[0] == 0 and col(T, "collect")[0] == 0 and col(T, "kite")[1] == 0.0
    T = slider_terms(sl, players(p0_hp=1000)); assert col(T, "retreat")[0] == 0 and sl["low"][0]
    T = slider_terms(sl, players(p0_hp=1600)); assert np.isclose(col(T, "retreat")[0], 0.6)
    T = slider_terms(sl, players(p0_hp=3500)); assert np.isclose(col(T, "retreat")[0], 1.9) and not sl["low"][0]
    T = slider_terms(sl, players(p0_hp=3600, p0_x=1800)); assert col(T, "retreat")[0] == 0
    T = slider_terms(fresh(), players(p0_damage_dealt=1000, p0_kills=1)); assert np.isclose(col(T, "aggressive")[0], 1.502)
    T = slider_terms(fresh(), players(p0_hp=1000)); assert col(T, "aggressive")[0] == 0   # in range but low: no engagement pay
    assert slider_terms(fresh(), players(p1_x=9000))[0, SL.index("aggressive")] == 0      # nobody in range
    assert gas_margin(1650, 1650, 4, 60, 40) == (1, 15.5) and gas_margin(-5, 99999, 0, 60, 40)[0] == 0
    T = slider_terms(fresh(), players(), ring=4); assert col(T, "gas_safe")[0] == -0.002 and col(T, "gas_safe")[6] == 0   # p0 1 tile in, p6 10 in
    T = slider_terms(fresh(), players(p0_in_gas=1), ring=6); assert col(T, "gas_safe")[0] == -0.01
    T = slider_terms(fresh(), players(p0_x=9150, p0_y=6150), ring=18); assert col(T, "gas_safe")[0] == 0   # endgame zone (1.5): no near-gas cost
    g = fresh(); slider_terms(g, players(p0_hp=1000))   # A1 / A7: knocked out low, respawned full: no retreat for the refill
    ko = players(p0_hp=0); ko[0].reserved = [0, 0, 0, 100]; assert not slider_terms(g, ko)[0].any()
    assert col(slider_terms(g, players(p0_hp=4000)), "retreat")[0] == 0 and g["beh"][0, BEH.index("alive")] == 2
    g = fresh(); slider_terms(g, players(p0_hp=1000)); assert col(slider_terms(g, players(p0_hp=1400, p0_cubes=1, p0_cubes_total=1)), "retreat")[0] == 0
    g = fresh(); slider_terms(g, players(p0_cubes=3, p0_cubes_total=3))           # cubes lost at a knock-out / an elimination
    ko = players(); ko[0].reserved = [0, 0, 0, 100]; assert col(slider_terms(g, ko), "collect")[0] == -3
    g = fresh(); slider_terms(g, players(p0_cubes=2, p0_cubes_total=2))
    assert col(slider_terms(g, players(p0_eliminated=1, p0_cubes_total=2)), "collect")[0] == -2 and g["beh"][0, BEH.index("cubes_max")] == 2
    mates = lambda **o: [setattr(q, "reserved", [0, 0, 0 if k < 3 else k, 0]) or q for k, q in enumerate(players(**o))]
    g = fresh(); T = slider_terms(g, mates(p2_damage_taken=1000, p2_x=2500, p0_damage_dealt=400))   # A3: peel, no proximity pay
    assert np.isclose(col(T, "protect")[0], 0.4) and col(T, "protect")[1] == 0
    for k in range(8): slider_terms(g, mates())          # steps k+1 .. k+8
    assert np.isclose(col(slider_terms(g, mates(p0_damage_dealt=400, p2_x=2500)), "protect")[0], 0.4)   # step k+9: still (10 steps)
    assert col(slider_terms(g, mates(p0_damage_dealt=400, p2_x=2500)), "protect")[0] == 0
    T = slider_terms(fresh(), mates(p2_damage_taken=1000, p0_damage_dealt=400)); assert col(T, "protect")[0] == 0   # p2 15+ tiles away
    assert slider_terms(fresh(), mates())[:, SL.index("protect")].sum() == 0   # being near a mate pays nothing by itself
    T = slider_terms(fresh(), mates(p1_x=2500)); assert np.allclose(col(T, "leash")[:3], [0.002, 0.002, 0])   # 850 apart: band
    T = slider_terms(fresh(), mates(p1_x=1800)); assert np.allclose(col(T, "leash")[:3], [-0.002, -0.002, 0])   # 150 apart: clump
    assert not slider_terms(fresh(), mates())[:, SL.index("leash")].any() and not slider_terms(fresh(), players())[:, SL.index("leash")].any()
    hb = players(p1_x=4500, p1_y=6300, p1_damage_taken=500, p0_damage_dealt=300, p0_x=4500, p0_y=4200)   # p1 in the bush (15, 21)
    tiles2 = tiles.copy(); tiles2[21, 15] = bush; g = slider_state(tiles2.ravel(), 60, 40, ["Sniper"] * 10, np.random.default_rng(0))
    assert np.isclose(col(slider_terms(g, hb), "bush_control")[0], 0.3)   # hidden in a bush 7 tiles off, in range, hit
    w = slider_state(tiles.ravel(), 60, 40, [], np.random.default_rng(0), w=O.preset_vec("bush_camper"))["w"]
    assert w.shape == (10, len(SL)) and (w == O.preset_vec("bush_camper")).all()
    r0, r1 = np.random.default_rng(5), np.random.default_rng(5)
    assert np.array_equal(sample_sliders(r0, 10), r1.uniform(0, 2, (10, len(SL))))   # uniform = the original draws
    m = sample_sliders(np.random.default_rng(1), 4000, "mix"); pre = np.stack([O.preset_vec(k) for k in O.PRESETS])
    exact = (m[:, None] == pre[None]).all(2).any(1).mean(); sparse = ((m == 0.5) | (m == 2.0)).all(1).mean()
    assert m.min() >= 0 and m.max() <= 2 and 0.08 < exact < 0.13 and 0.08 < sparse < 0.15, (exact, sparse)
    assert (slider_norm("") == 1).all() and slider_norm("measured").shape == (len(SL),)
    dead = players(p0_alive=0, p0_damage_dealt=900); assert not slider_terms(sl, dead)[0].any()
    pl = players(p0_alive=0, p1_eliminated=1, p2_eliminated=1, p3_eliminated=1)          # team placement
    for p, k in ((0, 9), (1, 8), (2, 10), (3, 7)): pl[p].elim_rank = k
    pr, dd = player_ranks(NS(done=0, alive=6, winner=-1, pl=pl))
    assert list(pr[:5]) == [9, 8, 10, 7, 3.5] and list(dd[:5]) == [1, 1, 1, 1, 0]
    g = [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]                   # team 1 out (best 8), team 2 alive through p4, rest alone
    tr = team_ranks(g, pr, dd, False); assert list(tr[:2]) == [8, 8] and np.isnan(tr[2:5]).all() and tr[5] == 3.5
    assert list(team_ranks(g, pr, dd, True)[:5]) == [8, 8, 3.5, 3.5, 3.5]                     # cut short: p4's estimate
    assert np.array_equal(team_ranks([0] * 10, pr, dd, False), pr)                            # solo: own rank
    def fake(self, i, r):                                # MAEnv._finish_step stand-in: the final transitions of r.fin
        for p in r.fin: rk = r.pl[p].elim_rank; self._trans.append((16 + p, 1.0, True, {"place": (10 - rk) / 9, "win": 0.0}, rk))
        self.I[i].battle += r.over
    env = Env.__new__(Env); env._trans, env.episodes, env.w, env.slider_scale = [], [], {"place": 5.0, "win": 3.0}, 0.0
    for groups, want in (([0] * 10, [1.0, 1.0]), ([1, 1] + [0] * 8, [1.0, 1.0 + 5 * (10 - 3) / 9 - 5 * (10 - 8) / 9])):
        env.I = [NS(sl=sl, cfg={"tag": {"groups": groups}}, hold={}, battle=1, pos={}, ptick=0, pjump=set(), agents={}, wprev=set())]
        mk = lambda fin, over, **o: NS(pl=players(**o), fin=fin, over=over, tick=1, njumps=0, jumps=[], done=0, alive=8, winner=-1, reserved=[0] * 8)
        r1 = mk([1], 0, p1_eliminated=1); r1.pl[1].elim_rank = 8
        r2 = mk([0], 0, p0_eliminated=1, p1_eliminated=1); r2.pl[0].elim_rank = 3; r2.pl[1].elim_rank = 8
        sup, MAEnv._finish_step = MAEnv._finish_step, fake
        try:
            env._finish_step(0, r1); n1 = len(env._trans); env._finish_step(0, r2)
        finally: MAEnv._finish_step = sup
        got = {a & 15: rw for a, rw, *_ in env._trans}; env._trans = []
        assert n1 == (0 if groups[0] else 1) and np.allclose([got[0], got[1]], want), (groups, n1, got)   # p1 held until p0 out
    def fake2(self, i, r):                               # p0 steps on: its hp term is the host's hp change
        self._trans.append((16, 2.0 * r.pl[0].hp / 1000, False, {"hp": r.pl[0].hp / 1000, "place": 0.0, "win": 0.0}, None))
    env.w = {"hp": 2.0}; env.I = [NS(sl=sl, cfg={"tag": {"groups": [1, 1] + [0] * 8}}, hold={}, battle=1, pos={}, ptick=0, pjump=set(), agents={0: 16}, wprev=set())]
    sup, MAEnv._finish_step = MAEnv._finish_step, fake2
    try:
        rw = mk([], 0, p0_hp=0); rw.pl[0].reserved[3] = 120; env._finish_step(0, rw)       # respawning: hp 0, dead
        assert env.dead([16, 17]).tolist() == [True, False] and env._trans[-1][3]["dead"] == 1.0
        env._finish_step(0, mk([], 0, p0_hp=4000))                                         # back with full hp
        assert env._trans[-1][1] == 0.0 and env._trans[-1][3]["hp"] == 0.0 and not env.dead([16]).any()
        assert env.resp == {"out": 1, "back": 1, "wiped": 0}
    finally: MAEnv._finish_step = sup
    def fake3(self, i, r):                               # p0 steps on with base reward 0 and finishes: an episode row
        self._trans.append((16, 0.0, False, {"hp": 0.0, "place": 0.0, "win": 0.0}, None)); self.episodes.append({"player": 0, "ret": 0.0, "terms": {}})
    sl3 = fresh(); sl3["w"][:] = 2.0; env.slider_scale = 0.5; env.slider_norm = np.full(len(SL), 3.0); env.w = {}
    env.I = [NS(sl=sl3, cfg={"tag": {"groups": [0] * 10}}, hold={}, battle=1, pos={}, ptick=0, pjump=set(), agents={0: 16}, wprev=set())]
    sup, MAEnv._finish_step = MAEnv._finish_step, fake3
    try: env._finish_step(0, mk([], 0)); env._finish_step(0, mk([], 0, p0_damage_dealt=1000))   # hidden, then an ambush hit
    finally: MAEnv._finish_step = sup
    want = 2.0 * 3.0 * 0.5 * np.array([0.002 + 0.005 + 0.002, 0.002 + 1.0 + 1.0 + 1.002])   # weight x norm x scale x (bush_camp + bush_control's
    # new tile + aggressive's engagement; bush_camp + ambush + kite + aggressive)
    assert np.allclose([t[1] for t in env._trans[-2:]], want), [t[1] for t in env._trans[-2:]]
    row = env.episodes[-1]; assert np.isclose(row["terms"]["sn_bush_camp"], 3 * 1.004) and np.isclose(row["terms"]["s_bush_camp"], 1.004)
    assert row["beh"]["alive"] == 2 and row["sliders"] == [2.0] * len(SL) and np.isclose(row["terms"]["sa_bush_camp"], 1.004)
    nr = lambda ts, leash, sa: {"sliders": [1.0] * len(SL), "tag": {"team_size": ts},
                                "terms": {**{"s_" + k: 1.0 for k in SL}, "s_leash": leash, **({"sa_leash": sa} if sa is not None else {})}}
    f, m = norm_from([nr(3, 0.01, 0.4), nr(3, -0.01, 0.6), nr(1, 0.0, 0.0)])   # band / clump cancel in the sum, not in |term|
    assert np.isclose(m["leash"], 0.5) and np.isclose(f["leash"], 1.0) and np.isclose(f["kite"], 0.5)   # leash: team rows only
    assert np.isclose(norm_from([nr(3, 0.2, None)])[1]["leash"], 0.2)   # rows without sa_: |s_|
    print("env_v5 sliders ok")


TEAM_SLIDERS = ("protect", "leash")   # fire only with a mate: normalised over team battles


def norm_from(rows, budget=0.5):
    """{slider: factor} so that each term's mean per-battle sum of |term| (row "sa_<slider>"; older rows: |"s_<slider>"|)
    becomes `budget` (0.5: a slider at weight 1 is worth ~0.5 per battle, against the 8 of place + win). |term| per step,
    not |sum|: a signed term (leash: band +, clump -) would cancel and get a huge factor. Team sliders over team battles
    only (in solo they are 0 by definition). A term that never fired: 1.0"""
    rows = [r for r in rows if "sliders" in r and not r.get("restart")]
    a = lambda r, k: r["terms"]["sa_" + k] if "sa_" + k in r["terms"] else abs(r["terms"].get("s_" + k, 0.0))
    use = lambda k: [r for r in rows if k not in TEAM_SLIDERS or r["tag"].get("team_size", 1) > 1]
    m = {k: float(np.mean([a(r, k) for r in use(k)])) if use(k) else 0.0 for k in SL}
    return {k: round(budget / v, 4) if v > 1e-6 else 1.0 for k, v in m.items()}, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12); ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--agents", type=int, default=10); ap.add_argument("--ticks", type=int, default=2)
    ap.add_argument("--out", default="bench.jsonl"); ap.add_argument("--gen", type=int, default=0, help="instances on generated maps")
    ap.add_argument("--native", type=int, default=0, help="1 = the native AI plays the agents (observed; the slider-norm reference)")
    ap.add_argument("--episodes", default="", help="also write every episode row to this jsonl")
    ap.add_argument("--norm", default="", help="RUN dir or episodes jsonl -> slider_norm factors (no game)")
    ap.add_argument("--check", action="store_true", help="slider reward self-check (no game)")
    a = ap.parse_args()
    if a.check: return _check()
    if a.norm:
        fn = os.path.join(a.norm, "episodes.jsonl") if os.path.isdir(a.norm) else a.norm
        f, m = norm_from([json.loads(l) for l in open(fn)])
        print("mean |term| per player and battle:", {k: round(v, 4) for k, v in m.items()}); print(json.dumps(f)); return
    bf = make_battle_fn(a.agents, gen=use_genmaps(a.gen) if a.gen else None)
    battle_fn = (lambda rng, i: {**bf(rng, i), "native_ctl": True}) if a.native else bf
    env = Env(a.n, battle_fn, ticks=a.ticks, fair=True, max_ticks=4000)
    rng = np.random.default_rng(0); t0 = None; steps = 0
    try:
        while t0 is None or time.time() - t0 < a.minutes * 60:
            obs, ids, _ = env.poll(min_ready=1, max_wait=0.003)
            if not len(ids): continue
            if t0 is None: t0 = time.time()     # clock starts at the first observation (after boot)
            B = len(ids); steps += B
            if steps == B: assert env.cond(ids).shape == (B, O.N_COND)
            act = np.stack([rng.integers(0, 17, B), rng.integers(0, 3, B), rng.integers(0, 360, B), rng.integers(0, 8, B)], 1)
            env.send(ids, act)
    finally:
        dt = time.time() - (t0 or time.time()); env.close()
    with open(a.out, "w") as f:
        for c in env.crashes: f.write(json.dumps(c) + "\n")
    if a.episodes:
        with open(a.episodes, "w") as f:
            for r in env.episodes: f.write(json.dumps(r) + "\n")
    print("n %d agents %d ticks %d: %.0f decisions/s, %d battles, %d episodes, %d restarts (%s), %d brawlers, %d maps" % (
        a.n, a.agents, a.ticks, steps / max(dt, 1e-9), env.stats["battles"], len(env.episodes), env.stats["restarts"],
        a.out, len(heroes()), len(locations())))
    rows = [e["terms"] for e in env.episodes if "sliders" in e]
    print("slider terms per episode (mean over %d): %s" % (len(rows), {m: round(float(np.mean([t["s_" + m] for t in rows])), 3) for m in SL} if rows else {}))


if __name__ == "__main__":
    sys.exit(main())
