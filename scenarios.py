"""Hard-scenario eval set (behaviour design 6): fixed seeded situations every checkpoint is scored on, scored by
per-episode numbers (not the reward, so results stay comparable when reward weights change).

    python -m scenarios --check                               # specs, layouts, scores / pass rules (no host)
    python -m scenarios --ckpt X --n 2 [--seeds 4] [--only a,b] # standalone: every scenario, the checkpoint as player 0
    python -m scenarios --native 1 --n 2                      # the native AI as player 0 (reference)

Each scenario = 2 maps (MAPS) x 2 brawler picks x --seeds layouts (default 4) = 16 episodes (seeds past 4 reuse the
layouts with other battle seeds). Player 0 is the agent.
A layout is resolved from the map's tiles (maps.csv via nbpc.diversity.map_reference, so it is deterministic and
needs no host): an anchor tile for the agent and the other roles at given distances with a clear line of fire, the
players not in the scene ("extras") >= FAR tiles away with their native AI off. Layout k of (scenario, map) is the
k-th valid anchor in a fixed shuffled order (anchors of one scenario >= 6 tiles apart).
Set-up on this host, no host change (env_v5.Env calls preroll() from _start_battle, synchronous on that instance, so
MAEnv's trackers start from the post-roll state; scenario steps send send_step()'s forced storm ring):
  - teleport (Instance.setup positions: teleport + stop + HP = max) and spawn protection off for everyone; native AI on
    only for the scenario's "ai" roles through native_mask (setup ai_off would clear a bot's AI flag for good).
  - pre-damage ("damaged": role, gas hits): the storm is forced to the ring that covers that player's tile (its
    margin + 1) until it took that many storm hits (20 % of max HP each, the host's rule for the first 5 s in gas, so
    3 hits = 40 % HP exactly); players that ring would reach wait on park tiles near the map centre and are teleported
    back after (a teleport heals, so the damaged player is never teleported again).
  - endgame: the extras stand deep in a forced ring until the storm eliminated them (~5 s), then the scene is
    teleported again.
  - natural: all 10 natives play NATURAL_S seconds (storm clock), the agent healed by a teleport in place every
    HEAL_EVERY ticks; then the enemy holding the most cubes becomes role e1. A battle whose agent is out after the
    set-up gives an episode row without "scn" ("invalid", counted in the summaries).
Scenarios (horizon; scored; pass; set-up details in SCENARIOS):
  gas_endgame      30 s  3 left, 6-tile safe zone, agent on its edge, 2 enemies inside: outlives >= 1 enemy and gas < 20 % HP
  third_party      15 s  2 fighting enemies 3 tiles apart 8 tiles away, one at 40 %: a kill and HP > 50 % all along
  assassin_rush    15 s  agent Piper, an assassin (Mortis / Edgar) 7 tiles away: alive at the end or a kill
  low_hp_close     15 s  agent at 40 %, full-HP enemy 5 tiles away, a bush 3 tiles away: alive at the end
  cube_rich_low    15 s  after a natural pre-roll the cube-richest enemy at 40 %, 6 tiles away: its kill and alive
  mate_under_fire  15 s  trio, a mate 5 tiles away with 2 enemies 3 tiles from it: mate and agent never knocked out
  bush_memory       5 s  an enemy seen 4 tiles away for 1 s, then teleported into an adjacent bush: attacks within 20 deg
                         of it in the 2 s after >= 2x those toward a control bush at the same distance (and >= 1)
Intents: scenario battles are tagged "intent": "none" (env_v5.Env.intents gives them the zero block), so their results
compare across the intent / planner steps. send_step mirrors MAEnv's step command (unchanged by the intents).
Damage: NbePlayer.damage_dealt / damage_taken are per step (the host clears them after every response), so scores
sum them over the episode's steps.
Trainer hook (train_ppo / train_fast --eval-scenarios 1, every --scenario-every-th eval): the items queue on the eval
instances as kind "evalscn" (tag "scn" = the name), played by the frozen eval copy; finish_eval writes one row per
eval to scenarios.jsonl (summary()) and the episodes to scenario_episodes.jsonl; every scenario episode is recorded
(Recorder) to replays/scn/ with its pass / fail (ScnIndex: replays/scn/index.json, the newest KEEP_UPDATES evals).
"""
import argparse, functools, json, os, sys, time, zlib
import numpy as np

TILE = 300
MAPS = ("Survival651", "Survival612")     # the most open Showdown map + a bushier one; of the 25 shipped maps only 4 have a
                                          # 6 x 6 open centre with 4 gas_endgame layouts (651, 612, 602, 631)
SEED0 = 77_000_000                        # battle seeds of the scenario battles (+ < 1 M; evals use 1..N, training draws >= 10 M)
FAR = 16                                  # tiles (Chebyshev) from the scene to an extra: outside the 15 x 12-tile view
ANCHOR_APART = 6                          # tiles between the anchors of two layouts of one scenario and map (variety)
NATURAL_S = 45                            # natural pre-roll: boxes broken and cubes picked up by then (storm ring ~3)
HEAL_EVERY = 4                            # ticks: the agent's in-place teleport (heal) during the natural pre-roll (20 lost
                                          # 2 of 4 Piper battles: a burst kills her within 1 s)
GAS_HIT = 0.2                             # storm damage per hit (env.c gas_tick: 20 % of max HP, once per 20 ticks, first 5 s)
EVENT_STEP = 10                           # bush_memory: the enemy hides after 10 steps = 1 s in view
MEM_S = (0.5, 1.0, 2.0, 4.0)              # bush_memory windows after the hide (design 3.4)
AIM_TOL = 20.0                            # degrees: an attack "toward" a point
KEEP_UPDATES = 3                          # replays/scn keeps the episodes of the newest 3 scenario evals
STAND = ("floor", "bush")                 # seg classes a brawler is placed on
BLOCK = ("indestructible", "destructible_wall", "fragile_wall", "destructible")   # block a shot (rope fence / water do not)
TRIO = [1, 1, 1, 2, 2, 2, 3, 3, 3, 0]     # trio groups (3 x 3 + 1, as training): agent 0, mates 1-2, enemies 3-5 ...
ALL = 0x3ff

# role spec: (role, ref, d tiles, kind floor|bush|tile, extra constraints). kind "tile" = a bush tile no player stands on.
# near = {role: (lo, hi)} tiles to an already placed role; margin = (lo, hi) tiles from the map edge ("k" = the zone
# ring); same_d = (role, tol): |d(A, me) - d(A, role)| <= tol; apart = (role, deg): angle at A to role >= deg.
SCENARIOS = {
    "gas_endgame": dict(
        horizon=30, team=1, zone=6, apart=3, pre="endgame", ai=("e1", "e2"), who={"A": 0, "e1": 1, "e2": 2},   # apart: anchors on the 6-tile edge
        picks=({"A": "Sniper", "e1": "Gunslinger", "e2": "ShotgunGirl"}, {"A": "ShotgunGirl", "e1": "Gunslinger", "e2": "ShotgunGirl"}),
        roles=(("A", None, 0, "floor", {"margin": (("k", 0), ("k", 0))}),
               ("e1", "A", 3, "floor", {"margin": (("k", 1), 99)}),
               ("e2", "A", 4, "floor", {"margin": (("k", 1), 99), "near": {"e1": (2, 6)}})),
        desc="3 players left (7 eliminated by the storm), the safe zone forced to 6 x 6 tiles, the agent on its edge, two native enemies inside",
        rule="outlives >= 1 enemy and storm damage < 20 % of max HP"),
    "third_party": dict(
        horizon=15, team=1, damaged=("e1", 3), pre="gas", ai=("e1", "e2"), who={"A": 0, "e1": 1, "e2": 2},
        picks=({"A": "Sniper", "e1": "ShotgunGirl", "e2": "Gunslinger"}, {"A": "ShotgunGirl", "e1": "ShotgunGirl", "e2": "Gunslinger"}),
        roles=(("A", None, 0, "floor", {}),
               ("e1", "A", 8, "floor", {"margin": (2, 20)}),
               ("e2", "e1", 3, "floor", {"near": {"A": (7, 11)}, "los": ("A",)})),
        desc="two native enemies 3 tiles apart (fighting) 8 tiles from the agent, one of them at 40 % HP",
        rule="a kill and own HP > 50 % all along"),
    "assassin_rush": dict(
        horizon=15, team=1, pre=None, ai=("e1",), who={"A": 0, "e1": 1},
        picks=({"A": "Sniper", "e1": "Undertaker"}, {"A": "Sniper", "e1": "Enrager"}),
        roles=(("A", None, 0, "floor", {}), ("e1", "A", 7, "floor", {})),
        desc="the agent as Piper, a native assassin (Mortis / Edgar) 7 tiles away; everybody else far away, AI off",
        rule="alive at the end or a kill"),
    "low_hp_close": dict(
        horizon=15, team=1, damaged=("A", 3), pre="gas", ai=("e1",), who={"A": 0, "e1": 1},
        picks=({"A": "Sniper", "e1": "ShotgunGirl"}, {"A": "ShotgunGirl", "e1": "ShotgunGirl"}),
        roles=(("A", None, 0, "floor", {"margin": (2, 20)}), ("e1", "A", 5, "floor", {}),
               ("b", "A", 3, "tile", {"near": {"e1": (4, 99)}})),
        desc="the agent at 40 % HP, a full-HP native enemy 5 tiles away, a bush 3 tiles from the agent",
        rule="alive at the end"),
    "cube_rich_low": dict(
        horizon=15, team=1, damaged=("e1", 3), pre="natural", ai=("e1",), who={"A": 0, "e1": None},
        picks=({"A": "Sniper"}, {"A": "ShotgunGirl"}),
        roles=(("A", None, 0, "floor", {}), ("e1", "A", 6, "floor", {"margin": (2, 20)})),
        desc="45 s of all-native play, then the enemy holding the most cubes is put 6 tiles from the agent at 40 % HP",
        rule="kills that enemy and alive at the end"),
    "mate_under_fire": dict(
        horizon=15, team=3, pre=None, ai=("m", "e1", "e2"), who={"A": 0, "m": 1, "e1": 3, "e2": 4},
        picks=({"A": "Sniper", "m": "ShotgunGirl", "e1": "Gunslinger", "e2": "ShotgunGirl"},
               {"A": "ShotgunGirl", "m": "ShotgunGirl", "e1": "Gunslinger", "e2": "ShotgunGirl"}),
        roles=(("A", None, 0, "floor", {}), ("m", "A", 5, "floor", {}),
               ("e1", "m", 3, "floor", {"near": {"A": (5, 9)}}), ("e2", "m", 3, "floor", {"near": {"A": (5, 9), "e1": (1.5, 5)}})),
        desc="trio: a native mate 5 tiles away, two native enemies (one team) 3 tiles from the mate, the agent at full HP",
        rule="the mate and the agent never knocked out"),
    "bush_memory": dict(
        horizon=5, team=1, pre=None, ai=(), who={"A": 0, "e1": 1}, event=("e1", "eb"),
        picks=({"A": "Sniper", "e1": "ShotgunGirl"}, {"A": "ShotgunGirl", "e1": "ShotgunGirl"}),
        roles=(("A", None, 0, "floor", {}), ("e1", "A", 4, "floor", {}),
               ("eb", "e1", 1, "tile", {"near": {"A": (3.4, 5.5)}, "los": ("A",)}),
               ("ctrl", "A", None, "tile", {"same_d": ("eb", 1.0), "apart": ("eb", 60), "near": {"eb": (4, 99), "e1": (4, 99)}})),
        desc="an enemy (AI off) in view 4 tiles away for 1 s, then teleported into an adjacent bush > 3 tiles away (hidden)",
        rule="attacks within 20 deg of the hidden enemy in the 2 s after >= 2 x those toward a control bush, and >= 1"),
}
OTHER = "ShotgunGirl"                     # the brawler of every player a pick does not name
NATURAL_ROSTER = ("ShotgunGirl", "Gunslinger", "BullDude", "Shaman", "RocketGirl")   # natives of the natural pre-roll


# ------------------------------------------------------------------ map tiles and layouts
def grid_from(cls, box):
    """seg classes (H, W) + spawn / box digit mask -> {stand, bush, block} (H, W) bools"""
    import obs_v5 as O
    S = lambda names: np.isin(cls, [O.SEG.index(n) for n in names])
    return {"stand": S(STAND) & ~box, "bush": S(("bush",)) & ~box, "block": S(BLOCK) | box}


@functools.lru_cache(None)
def grid(loc):
    from nbpc import diversity as D
    import obs_v5 as O
    ids, rows = D.map_reference(loc); H, W = len(rows), len(rows[0])
    cls = O._tile_lut()[np.clip(ids.reshape(H, W) % 1000000 + 1, 0, 255)]
    return grid_from(cls, np.char.isdigit(np.array([list(r) for r in rows])))


def margin(t, W, H):
    """tiles from the map edge (env_v5.gas_margin with ring 0): a tile is storm iff margin < ring"""
    return min(t[0], t[1], W - 1 - t[0], H - 1 - t[1])


def zone_ring(sp, W, H):
    return (min(W, H) - sp["zone"]) // 2 if sp.get("zone") else 0


def los(G, a, b):
    """no shot-blocking tile on the segment between the tile centres a, b"""
    n = int(np.ceil(4 * max(abs(b[0] - a[0]), abs(b[1] - a[1])))) + 1
    f = np.linspace(0, 1, n + 1)
    xs = np.floor(a[0] + 0.5 + f * (b[0] - a[0])).astype(int); ys = np.floor(a[1] + 0.5 + f * (b[1] - a[1])).astype(int)
    return not G["block"][ys, xs].any()


def _dist(a, b): return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _ok(G, t, kind):
    H, W = G["stand"].shape
    if not (0 <= t[0] < W and 0 <= t[1] < H): return False
    return bool(G["bush"][t[1], t[0]]) if kind in ("bush", "tile") else bool(G["stand"][t[1], t[0]] and not G["bush"][t[1], t[0]])


def _fits(G, t, role, kind, ref, d, c, got, k):
    H, W = G["stand"].shape
    if not _ok(G, t, kind) or any(_dist(t, q) < (0.5 if kind == "tile" else 1.5) for q in got.values()): return False   # a tile may touch a player
    if ref is not None and ((d is not None and abs(_dist(t, got[ref]) - d) > 0.75) or not los(G, got[ref], t)): return False
    lo, hi = c.get("margin", (0, 99)); v = lambda x: k + x[1] if isinstance(x, tuple) else x
    if not v(lo) <= margin(t, W, H) <= v(hi): return False
    if any(not lo <= _dist(t, got[q]) <= hi for q, (lo, hi) in c.get("near", {}).items()): return False
    if any(not los(G, got[q], t) for q in c.get("los", ())): return False
    if "same_d" in c:
        q, tol = c["same_d"]
        if abs(_dist(t, got["A"]) - _dist(got[q], got["A"])) > tol: return False
    if "apart" in c:
        q, deg = c["apart"]; a = got["A"]
        u, w = np.subtract(t, a), np.subtract(got[q], a)
        if np.degrees(np.arccos(np.clip(u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-9), -1, 1))) < deg: return False
    if ref is None and c.get("clear", 1):                # the anchor: room to move (3 x 3 standable)
        y, x = t[1], t[0]
        if y < 1 or x < 1 or not G["stand"][y - 1:y + 2, x - 1:x + 2].all(): return False
    return True


def place(sp, G, anchor, k, rng):
    """roles placed in order from the anchor -> {role: (tx, ty)} or None. A role at distance d from its ref: the first
    of 24 directions (15 deg apart, random start) whose tile fits; d None: the first fitting tile of its kind within 8
    tiles of the ref (random order)"""
    got = {}
    for role, ref, d, kind, c in sp["roles"]:
        if ref is None:
            if not _fits(G, anchor, role, kind, None, 0, c, got, k): return None
            got[role] = tuple(anchor); continue
        o = rng.integers(24); found = None
        if d is None:
            x0, y0 = got[ref]; H, W = G["stand"].shape
            ys, xs = np.nonzero(G["bush"] if kind != "floor" else G["stand"] & ~G["bush"])
            near = np.flatnonzero(np.maximum(abs(xs - x0), abs(ys - y0)) <= 8)
            for j in rng.permutation(near):
                t = (int(xs[j]), int(ys[j]))
                if _fits(G, t, role, kind, ref, None, c, got, k): found = t; break
            if found is None: return None
            got[role] = found; continue
        for j in range(24):
            th = np.radians(15.0 * ((o + j) % 24)); x0, y0 = got[ref]
            t = (int(np.floor(x0 + 0.5 + d * np.cos(th))), int(np.floor(y0 + 0.5 + d * np.sin(th))))
            if _fits(G, t, role, kind, ref, d, c, got, k): found = t; break
        if found is None: return None
        got[role] = found
    return got


def _spread(G, cand, n, apart, rng):
    """n tiles of cand (a bool mask) >= apart tiles from each other (Chebyshev), in a shuffled order"""
    ys, xs = np.nonzero(cand); order = rng.permutation(len(xs)); out = []
    for j in order:
        t = (int(xs[j]), int(ys[j]))
        if all(max(abs(t[0] - q[0]), abs(t[1] - q[1])) >= apart for q in out): out.append(t)
        if len(out) == n: break
    return out


def extras_for(sp, G, got, k, rng, n):
    """tiles of the players not in the scene: endgame = deep in the forced ring (margin <= k - 3), else >= FAR tiles
    from every scene tile (relaxed to 12 if the map is too small for n of them)"""
    H, W = G["stand"].shape; free = G["stand"] & ~G["bush"]
    yy, xx = np.mgrid[0:H, 0:W]
    if sp["pre"] == "endgame":
        return _spread(G, free & (np.minimum(np.minimum(xx, yy), np.minimum(W - 1 - xx, H - 1 - yy)) <= k - 3), n, 3, rng)
    for far in (FAR, 12):
        m = free.copy()
        for t in got.values(): m &= np.maximum(abs(xx - t[0]), abs(yy - t[1])) >= far
        out = _spread(G, m, n, 3, rng)
        if len(out) == n: return out
    return out


def park_for(G, got, rng, n=9):
    """tiles near the map centre (margin >= 22, >= 4 tiles from the scene) where players wait while the storm damages
    a scene player (its margin <= 20 by the layout constraints, so the ring never reaches them)"""
    H, W = G["stand"].shape; yy, xx = np.mgrid[0:H, 0:W]
    m = G["stand"] & ~G["bush"] & (np.minimum(np.minimum(xx, yy), np.minimum(W - 1 - xx, H - 1 - yy)) >= 22)
    for t in got.values(): m &= np.maximum(abs(xx - t[0]), abs(yy - t[1])) >= 4
    return _spread(G, m, n, 2, rng)


def solve(name, loc, n=4, G=None):
    """the first n layouts of a scenario on a map: [{tiles {role: (tx, ty)}, extra [...], park [...], k}] (fewer if
    the map has fewer)"""
    sp = SCENARIOS[name]; G = G if G is not None else grid(loc); H, W = G["stand"].shape; k = zone_ring(sp, W, H)
    rng = np.random.default_rng(zlib.crc32(("%s:%s" % (name, loc)).encode()))
    ys, xs = np.nonzero(G["stand"] & ~G["bush"]); order = rng.permutation(len(xs)); out = []
    n_scene = sum(1 for r in sp["who"]) if sp["pre"] != "natural" else 2
    for j in order:
        a = (int(xs[j]), int(ys[j]))
        if any(max(abs(a[0] - o["tiles"]["A"][0]), abs(a[1] - o["tiles"]["A"][1])) < sp.get("apart", ANCHOR_APART) for o in out): continue
        r2 = np.random.default_rng(zlib.crc32(("%s:%s:%d:%d" % (name, loc, a[0], a[1])).encode()))
        got = place(sp, G, a, k, r2)
        if got is None: continue
        ext = extras_for(sp, G, got, k, r2, 10 - n_scene if sp["pre"] != "natural" else 9)
        if len(ext) < (10 - n_scene if sp["pre"] != "natural" else 9): continue
        park = park_for(G, got, r2) if sp.get("damaged") else []
        if sp.get("damaged") and len(park) < 9: continue
        out.append({"tiles": got, "extra": ext, "park": park, "k": k})
        if len(out) == n: break
    return out


@functools.lru_cache(None)
def layouts(name, loc, n=4):
    return solve(name, loc, n)


# ------------------------------------------------------------------ items and battle setup
def items(names=None, seeds=4, maps=MAPS, update=None):
    """the scenario battles: one dict per (scenario, map, pick, seed)"""
    out = []
    for name in names or SCENARIOS:
        for loc in maps:
            for pk in range(len(SCENARIOS[name]["picks"])):
                for sd in range(seeds):
                    out.append({"name": name, "loc": loc, "pick": pk, "seed": sd, "update": update})
    return out


def roster(names=None):
    """every brawler the scenarios use (model.add_heroes)"""
    return sorted({OTHER, *NATURAL_ROSTER} | {b for n in (names or SCENARIOS) for p in SCENARIOS[n]["picks"] for b in p.values()})


def battle_cfg(it, native=False):
    """MAEnv battle config of one item: player 0 = the agent (native=True: observed, played by its native AI)"""
    from nbpc import diversity as D
    import obs_v5 as O
    sp = SCENARIOS[it["name"]]; pick = sp["picks"][it["pick"]]; L = layouts(it["name"], it["loc"]); lay = L[it["seed"] % len(L)]   # seed >= 4: a layout again, new battle seed
    by_p = {p: r for r, p in sp["who"].items() if p is not None}
    if sp["pre"] == "natural": chr_ = [pick["A"]] + [NATURAL_ROSTER[p % len(NATURAL_ROSTER)] for p in range(1, 10)]
    else: chr_ = [pick.get(by_p.get(p), OTHER) for p in range(10)]
    groups = list(TRIO) if sp["team"] == 3 else [0] * 10
    div = D.make_div(location=it["loc"], chr=chr_)
    if sp["team"] > 1: div["team_groups"] = groups
    sd = SEED0 + zlib.crc32(("%s:%s:%d:%d" % (it["name"], it["loc"], it["pick"], it["seed"])).encode()) % 1_000_000
    tag = {"kind": "evalscn", "scn": it["name"], "loc": it["loc"], "chr": chr_, "team_size": sp["team"], "groups": groups,
           "sliders": [float(v) for v in O.preset_vec("balanced")], "preset": "balanced", "item": dict(it),
           "intent": "none"}   # fixed (env_v5.Env.intents): scenario results compare across the intent / planner steps
    cfg = {"seed": sd, "ctl": [0], "end": "agents", "div": div, "tag": tag,
           "scenario": {"name": it["name"], "tiles": lay["tiles"], "extra": lay["extra"], "park": lay["park"], "k": lay["k"]}}
    if native: cfg["native_ctl"] = True
    return cfg


def _u(t): return (int((t[0] + 0.5) * TILE), int((t[1] + 0.5) * TILE))


def preroll(env, i):
    """the scripted set-up of instance i's new battle (cfg "scenario"), synchronous; afterwards s.inst.resp is the
    scenario's start state with the agent's observation, s.native the scenario's native AI mask, cfg["scenario"]
    gets ring / who (cube_rich_low: e1 = the cube-richest enemy) / pre (what the set-up did)"""
    from nbpc import proto as P
    s = env.I[i]; sc = s.cfg["scenario"]; sp = SCENARIOS[sc["name"]]; inst = s.inst; t0 = time.time()
    tiles = sc["tiles"]; W, H = (inst.resp.map_w or 60), (inst.resp.map_h or 60)
    ctl = s.send_mask; idle = {p: (-1, 0, 0.0, 0.0) for p in s.ctl} if ctl else None; n = [0]

    def step(native=0, ring=0, ticks=env.ticks, last=False):
        n[0] += 1
        return inst.step(actions=idle, ticks=ticks, flags=(env._flags() | P.F_MAP) if last else 0, ctl_mask=ctl, native_mask=native,
                         obs_mask=s.obs_mask, storm=ring)

    def setup(pos): inst.setup(positions={p: _u(t) for p, t in pos.items()}, unprotect=range(10))
    who = {k: v for k, v in sp["who"].items()}; r = inst.resp; pre = {}
    if sp["pre"] == "natural":                           # all native, storm clock; the agent healed in place
        for j in range(NATURAL_S * 20 // HEAL_EVERY):
            r = step(native=ALL, ring=None, ticks=HEAL_EVERY)
            if r.pl[0].alive: inst.setup(positions={0: (int(r.pl[0].x), int(r.pl[0].y))})
        alive = [p for p in range(1, 10) if r.pl[p].alive and not r.pl[p].eliminated]
        rich = max(alive, key=lambda p: (int(r.pl[p].cubes_total), -p)); who["e1"] = rich
        pre.update(rich=rich, rich_cubes=int(r.pl[rich].cubes_total), agent_cubes=int(r.pl[0].cubes_total), natural_alive=len(alive) + 1)
        others = [p for p in alive if p != rich]
    else:
        others = [p for p in range(10) if p not in who.values()]
    final = {who[k]: tiles[k] for k in who}; ext = {p: sc["extra"][j] for j, p in enumerate(others)}
    setup({**final, **ext}); r = step()
    if sp["pre"] == "endgame":                           # the extras into the storm until eliminated
        k = sc["k"]; m = 0
        while any(not r.pl[p].eliminated for p in ext) and m < 300: r = step(ring=k, ticks=4); m += 1
        pre["extras_out"] = sum(int(r.pl[p].eliminated) for p in ext); setup(final); r = step(ring=k)
    if sp.get("damaged"):                                # the storm on the damaged player's tile; the others parked
        role, hits = sp["damaged"]; dp = who[role]; k = margin(tiles[role], W, H) + 1
        pos = {**final, **ext}
        park = {p: sc["park"][j] for j, p in enumerate(q for q in pos if q != dp and margin(pos[q], W, H) <= k + 1)}
        if park: setup(park); r = step()
        target = r.pl[dp].max_hp * (1 - GAS_HIT * hits) + 1; m = 0
        while r.pl[dp].hp > target and m < 400: r = step(ring=k); m += 1
        if park: setup({p: pos[p] for p in park})
        r = step(); pre.update(hp_frac=round(r.pl[dp].hp / max(1, r.pl[dp].max_hp), 3), parked=len(park))
    s.native = sum(1 << who[k] for k in sp["ai"]) | (1 if s.cfg.get("native_ctl") else 0)
    ring = sc["k"] if sp["pre"] == "endgame" else 0
    step(native=s.native, ring=ring, last=True)          # the start observation
    pre.update(steps=n[0], s=round(time.time() - t0, 2))
    sc.update(ring=ring, who=who, pre=pre)


# ------------------------------------------------------------------ during the scenario
COLS = ("x", "y", "hp", "max_hp", "up", "cubes", "kills", "dealt", "taken", "gasd", "in_gas", "elim")


def state(r):
    """(10, len(COLS)) int64 of a response: up = alive, not eliminated, not respawning, hp > 0"""
    return np.array([[p.x, p.y, p.hp, p.max_hp, int(bool(p.alive) and not p.eliminated and p.reserved[3] <= 0 and p.hp > 0), p.cubes_total,
                      p.kills, p.damage_dealt + p.bomb_damage, p.damage_taken, p.gas_damage, p.in_gas, p.eliminated]
                     for p in (r.pl[q] for q in range(10))], np.int64)


def start(env, s):
    """the trace of a scenario battle (s.scn): row 0 = the start state, then one row per step with the agent's action"""
    sc = s.cfg["scenario"]; sp = SCENARIOS[sc["name"]]; tg = s.cfg["tag"]
    ev = {}
    if sp.get("event"):
        who, tl = sc["who"], sc["tiles"]; ev[EVENT_STEP] = {who[sp["event"][0]]: _u(tl[sp["event"][1]])}
    return {"name": sc["name"], "item": tg.get("item"), "who": sc["who"], "tiles": sc["tiles"], "ring": sc["ring"], "pre": sc["pre"],
            "rows": [state(s.inst.resp)], "acts": [[0, 0, 0, 0]], "sent": 0, "events": ev, "hide_row": None, "battle": s.battle,
            "horizon": sp["horizon"] * 20, "ticks": env.ticks, "native": bool(s.cfg.get("native_ctl"))}


def send_step(env, i, acts):
    """MAEnv._send_step for a scenario battle: its events (a teleport before the step) and the forced storm ring"""
    from nbpc import proto as P
    from nbpc.maenv import action_to_native
    s = env.I[i]; sc = s.scn; ev = sc["events"].get(sc["sent"])
    if ev:
        s.inst.setup(positions=ev); sc["hide_row"] = len(sc["rows"])   # applied in the next tick: row hide_row is the first hidden state
    sc["sent"] += 1; s.acts = acts
    a = {p: action_to_native(v) for p, v in acts.items()}
    s.inst.send(P.CMD_STEP, ticks=env.ticks, flags=env._flags(), ctl_mask=s.send_mask, native_mask=s.native, obs_mask=s.obs_mask,
                actions=a if s.send_mask else None, gas=sc["ring"] + 1, **env._pix())
    s.state = "stepping"; s.t_sent = time.time()


def trace(sc, r, act):
    sc["rows"].append(state(r)); sc["acts"].append([int(x) for x in act] if act is not None else [0, 0, 0, 0])


# ------------------------------------------------------------------ scores and pass rules
def _first_down(up, p):
    d = np.flatnonzero(up[1:, p] == 0)
    return int(d[0]) + 1 if len(d) else None


def _ang(dx, dy): return np.degrees(np.arctan2(dy, dx))


def _adiff(a, b): d = np.abs(np.asarray(a) - np.asarray(b)) % 360.0; return np.minimum(d, 360.0 - d)


def score(sc):
    """{"scores": {...}, "pass": bool} of a finished scenario trace (numbers from the episode only)"""
    R = np.array(sc["rows"]); A = np.array(sc["acts"]); c = {k: R[:, :, j] for j, k in enumerate(COLS)}; who = sc["who"]
    up = c["up"]; sps = sc["ticks"] / 20.0; dn = _first_down(up, 0); end = dn if dn is not None else len(R) - 1
    mx = max(1, int(c["max_hp"][0, 0]))
    s = {"survived": int(dn is None), "t_alive_s": round(end * sps, 2),
         "dealt_k": round(float(c["dealt"][1:, 0].sum()) / 1000, 3),
         "taken_k": round(float((c["taken"][1:, 0] - c["gasd"][1:, 0]).sum()) / 1000, 3),
         "gas_frac": round(float(c["gasd"][1:, 0].sum()) / mx, 3), "kills": int(c["kills"][1:, 0].sum()),
         "hp_min": round(float((c["hp"][:end + 1, 0] / np.maximum(c["max_hp"][:end + 1, 0], 1)).min()), 3)}

    def killed(role):                                    # the role went down in a step where the agent got a kill
        q = who[role]; d = _first_down(up, q)
        return int(d is not None and (dn is None or d <= dn) and c["kills"][max(1, d - 1):d + 2, 0].sum() > 0)
    n = sc["name"]
    if n == "gas_endgame":
        downs = [_first_down(up, who[r]) for r in ("e1", "e2")]
        s["outlived"] = sum(d is not None and (dn is None or d < dn) for d in downs); s["place"] = 3 - s["outlived"]
        ok = s["outlived"] >= 1 and s["gas_frac"] < 0.2
    elif n == "third_party":
        s["kill_low"] = killed("e1"); ok = s["kills"] >= 1 and s["hp_min"] > 0.5
    elif n == "assassin_rush":
        q = who["e1"]; both = (up[:, 0] > 0) & (up[:, q] > 0)
        s["mean_dist_tiles"] = round(float(np.hypot(c["x"][both, 0] - c["x"][both, q], c["y"][both, 0] - c["y"][both, q]).mean()) / TILE, 2) if both.any() else None
        s["trade"] = round(s["dealt_k"] / max(s["taken_k"], 0.1), 3); ok = bool(s["survived"] or s["kills"] >= 1)
    elif n == "low_hp_close":
        q = who["e1"]; d = np.hypot(c["x"][:, 0] - c["x"][:, q], c["y"][:, 0] - c["y"][:, q])
        G = grid(sc["item"]["loc"]) if sc.get("item") else None
        if G is not None:
            Hh, Ww = G["bush"].shape
            inb = G["bush"][np.clip(c["y"][:, 0] // TILE, 0, Hh - 1), np.clip(c["x"][:, 0] // TILE, 0, Ww - 1)]
        else: inb = np.zeros(len(R), bool)
        live = up[:, 0] > 0
        s["bush_s"] = round(float((inb & live)[1:].sum()) * sps, 2)
        s["hidden_s"] = round(float((inb & live & (d > 900) & (up[:, q] > 0))[1:].sum()) * sps, 2)   # fair rule: bush, > 900 units
        ok = bool(s["survived"])
    elif n == "cube_rich_low":
        s["kill_rich"] = killed("e1"); s["rich_cubes"] = int(c["cubes"][0, who["e1"]])
        s["cubes_gained"] = int(c["cubes"][end, 0] - c["cubes"][0, 0]); ok = bool(s["kill_rich"] and s["survived"])
    elif n == "mate_under_fire":
        s["mate_up"] = int(_first_down(up, who["m"]) is None); ok = bool(s["mate_up"] and s["survived"])
    elif n == "bush_memory":
        s.update(memory(sc, c, A)); ok = s["aim_e_2"] >= 1 and s["aim_e_2"] >= 2 * s["aim_c_2"]
    else:
        raise KeyError(n)
    return {"scores": s, "pass": bool(ok)}


def memory(sc, c, A):
    """bush_memory: per window N in MEM_S (s after the hide) attacks within AIM_TOL of the hidden enemy's bush (aim_e_N)
    / the control bush (aim_c_N), mean cosine of the move toward each (move_e_N / move_c_N); revealed_s = time the
    agent was within 900 units of the hidden enemy (it is seen there: the memory is not needed)"""
    out = {}; h = sc["hide_row"]; q = sc["who"]["e1"]; sps = sc["ticks"] / 20.0
    eb, cb = _u(sc["tiles"]["eb"]), _u(sc["tiles"]["ctrl"])
    if h is None:
        for N in MEM_S:
            key = ("%g" % N).replace(".", "_"); out.update({"aim_e_" + key: 0, "aim_c_" + key: 0, "move_e_" + key: None, "move_c_" + key: None})
        out["revealed_s"] = None; return out
    for N in MEM_S:                                      # action j is decided on state row j - 1
        j = np.arange(h + 1, min(len(A), h + 1 + int(round(N / sps))))
        x, y = c["x"][j - 1, 0], c["y"][j - 1, 0]; fire = A[j, 1] > 0; aim = A[j, 2]; mv = A[j, 0]
        key = ("%g" % N).replace(".", "_")
        for nm, t in (("e", eb), ("c", cb)):
            to = _ang(t[0] - x, t[1] - y)
            out["aim_%s_%s" % (nm, key)] = int((fire & (_adiff(aim, to) <= AIM_TOL)).sum())
            cos = np.where(mv > 0, np.cos(np.radians((mv - 1) * 22.5 - to)), 0.0)
            out["move_%s_%s" % (nm, key)] = round(float(cos.mean()), 3) if len(j) else None
    rv = np.hypot(c["x"][h:, 0] - c["x"][h:, q], c["y"][h:, 0] - c["y"][h:, q]) <= 900
    out["revealed_s"] = round(float(rv.sum()) * sps, 2)
    return out


def summary(rows):
    """per scenario over its episode rows (row["scn"]): episodes, pass rate, mean of every score, pass rate per pick and map"""
    out = {}
    for name in SCENARIOS:
        rs = [r["scn"] for r in rows if (r.get("scn") or {}).get("name") == name]
        if not rs: continue
        keys = sorted({k for r in rs for k, v in r["scores"].items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
        mean = lambda k: (lambda v: round(float(np.mean(v)), 3) if v else None)([r["scores"][k] for r in rs if r["scores"].get(k) is not None])
        grp = lambda f: {str(g): round(float(np.mean([r["pass"] for r in rs if f(r) == g])), 3) for g in sorted({f(r) for r in rs})}
        out[name] = {"n": len(rs), "pass": round(float(np.mean([r["pass"] for r in rs])), 3), **{k: mean(k) for k in keys},
                     "pass_pick": grp(lambda r: (r["item"] or {}).get("pick")), "pass_map": grp(lambda r: (r["item"] or {}).get("loc"))}
    if out: out["_all"] = {"pass": round(float(np.mean([v["pass"] for v in out.values()])), 3), "n": sum(v["n"] for v in out.values())}
    return out


def episode_row(r):
    """the compact per-episode record (scenario_episodes.jsonl)"""
    sc = r["scn"]
    return {"scn": sc["name"], "item": sc["item"], "pass": sc["pass"], "scores": sc["scores"], "pre": sc.get("pre"),
            "restart": r.get("restart"), "steps": r.get("steps"), "rank": r.get("rank"), "truncated": r.get("truncated"),
            "update": (sc["item"] or {}).get("update")}


# ------------------------------------------------------------------ replays (viz: failures first)
class Recorder:
    """every scenario episode -> replays/scn/u<update>_<name>_<map>_p<pick>_s<seed>.json (train_ppo replay format + "scn":
    the scores and pass); finish() returns its index entry (ScnIndex.add writes the index)"""
    def __init__(self, out):
        self.dir = os.path.join(out, "replays", "scn"); os.makedirs(self.dir, exist_ok=True); self.cur = {}

    def frame(self, env, ids, o, act):
        from train_ppo import replay_head, replay_frame
        by = {s.battle: s for s in env.I}
        for b in range(len(ids)):
            s = by.get(int(ids[b]) >> 4)
            if s is None or getattr(s, "scn", None) is None or s.scn["battle"] != s.battle or int(ids[b]) & 15: continue
            c = self.cur.get(s.battle)
            if c is None: c = self.cur[s.battle] = replay_head(s, (s.scn["item"] or {}).get("update") or 0)
            c["frames"].append(replay_frame(s, ids, b, o, act))

    def finish(self, row):
        sc = row.get("scn"); c = self.cur.pop(sc["battle"], None) if sc else None
        if c is None or not c["frames"]: return None
        it = sc["item"] or {}; c["scn"] = {k: sc[k] for k in ("name", "pass", "scores", "pre", "item")}
        name = "u%04d_%s_%s_p%d_s%d.json" % (it.get("update") or 0, sc["name"], it.get("loc"), it.get("pick", 0), it.get("seed", 0))
        json.dump(c, open(os.path.join(self.dir, name), "w"), separators=(",", ":"))
        return {"file": name, "update": it.get("update") or 0, "scn": sc["name"], "loc": it.get("loc"), "pick": it.get("pick"),
                "seed": it.get("seed"), "pass": sc["pass"], "frames": len(c["frames"])}


class ScnIndex:
    """replays/scn/index.json: the episodes of the newest KEEP_UPDATES scenario evals (older files removed)"""
    def __init__(self, out):
        self.dir = os.path.join(out, "replays", "scn"); os.makedirs(self.dir, exist_ok=True); p = os.path.join(self.dir, "index.json")
        self.index = json.load(open(p)) if os.path.exists(p) else []

    def add(self, entries):
        entries = [e for e in entries if e]
        if not entries: return
        self.index = [x for x in self.index if x["file"] not in {e["file"] for e in entries}] + entries
        keep = sorted({x["update"] for x in self.index})[-KEEP_UPDATES:]
        for x in self.index:
            if x["update"] not in keep:
                try: os.remove(os.path.join(self.dir, x["file"]))
                except OSError: pass
        self.index = [x for x in self.index if x["update"] in keep]
        json.dump(self.index, open(os.path.join(self.dir, "index.json"), "w"))


# ------------------------------------------------------------------ self-check
def _synth_grid(W=60, H=60):
    """an open map with a wall line, a few bushes and box digits (no game files)"""
    cls = np.zeros((H, W), np.int64); box = np.zeros((H, W), bool)
    import obs_v5 as O
    cls[:, 0] = cls[:, -1] = cls[0] = cls[-1] = O.SEG.index("indestructible")
    cls[20, 10:30] = O.SEG.index("indestructible")
    for y, x in ((14, 18), (14, 19), (15, 18), (40, 40), (40, 41), (30, 30), (30, 31), (31, 30), (44, 20), (44, 21), (26, 44), (26, 45)):
        cls[y, x] = O.SEG.index("bush")
    for y in range(4, 56, 6):
        for x in range(4, 56, 6): cls[y, x] = O.SEG.index("bush")
    box[50, 50] = True
    return grid_from(cls, box)


def _fake_trace(name, n=150, **ev):
    """a scenario trace with everybody standing still and full HP; ev: per-player edits applied from row k on:
    {(player, col): [(row, value), ...]} via ev["edit"], agent actions via ev["acts"] {row: [m, f, aim, d]}"""
    sp = SCENARIOS[name]; who = {k: (v if v is not None else 5) for k, v in sp["who"].items()}
    base = np.zeros((10, len(COLS)), np.int64); base[:, COLS.index("max_hp")] = 4000; base[:, COLS.index("hp")] = 4000
    base[:, COLS.index("up")] = 1; base[:, 0] = 3000 + 600 * np.arange(10); base[:, 1] = 3000
    rows = [base.copy() for _ in range(n + 1)]; acts = [[0, 0, 0, 0] for _ in range(n + 1)]
    for (p, col), chg in ev.get("edit", {}).items():
        for k, v in chg:
            for j in range(k, n + 1): rows[j][p, COLS.index(col)] = v
    for (p, col), chg in ev.get("once", {}).items():
        for k, v in chg: rows[k][p, COLS.index(col)] = v
    for k, a in ev.get("acts", {}).items(): acts[k] = a
    tiles = {"eb": (13, 10), "ctrl": (10, 13)}
    return {"name": name, "item": None, "who": who, "tiles": tiles, "ring": 0, "pre": {}, "rows": rows, "acts": acts,
            "hide_row": ev.get("hide"), "ticks": 2, "battle": 1}


def _check():
    # specs
    keys = {"horizon", "team", "pre", "ai", "who", "picks", "roles", "desc", "rule"}
    for n, sp in SCENARIOS.items():
        assert keys <= set(sp), (n, keys - set(sp))
        assert len(sp["picks"]) == 2 and sp["roles"][0][0] == "A" and sp["roles"][0][1] is None and sp["who"]["A"] == 0, n
        roles = [r[0] for r in sp["roles"]]
        assert all(k in roles for k in sp["who"]) and all(k in sp["who"] for k in sp["ai"]), n
        assert all(r[1] is None or r[1] in roles[:j] for j, r in enumerate(sp["roles"])), n
        assert sp["pre"] in (None, "gas", "endgame", "natural") and (sp["pre"] == "gas") <= bool(sp.get("damaged")), n
        if sp.get("damaged"): assert dict((r[0], r[4]) for r in sp["roles"])[sp["damaged"][0]].get("margin", (0, 99))[1] <= 20, n   # park tiles exist
        assert sp["team"] == 1 or all(TRIO[sp["who"][k]] != TRIO[0] for k in ("e1", "e2")), n
    assert len(items()) == 7 * 2 * 2 * 4 == 112 and len(items(seeds=1)) == 28
    # geometry on a synthetic map
    G = _synth_grid()
    assert los(G, (12, 15), (12, 25)) is False and los(G, (5, 15), (5, 25)) and margin((3, 50), 60, 60) == 3
    for name in SCENARIOS:
        L = solve(name, "synthetic", 2, G=G); sp = SCENARIOS[name]
        assert len(L) == 2, (name, len(L))
        for lay in L:
            t = lay["tiles"]
            for role, ref, d, kind, c in sp["roles"][1:]:
                assert (d is None or abs(_dist(t[role], t[ref]) - d) <= 0.75) and los(G, t[ref], t[role]), (name, role)
                assert (kind == "floor") == bool(G["stand"][t[role][1], t[role][0]] and not G["bush"][t[role][1], t[role][0]]), (name, role)
            if sp["pre"] == "endgame":
                assert margin(t["A"], 60, 60) == lay["k"] == 27 and all(margin(t[r], 60, 60) > lay["k"] for r in ("e1", "e2"))
                assert all(margin(e, 60, 60) <= lay["k"] - 3 for e in lay["extra"]) and len(lay["extra"]) == 7
            else:
                assert all(min(max(abs(e[0] - q[0]), abs(e[1] - q[1])) for q in t.values()) >= 12 for e in lay["extra"]), name
            if sp.get("damaged"):
                dm = margin(t[sp["damaged"][0]], 60, 60)
                assert len(lay["park"]) == 9 and all(margin(q, 60, 60) >= dm + 2 for q in lay["park"]), name
        a, b = L[0]["tiles"]["A"], L[1]["tiles"]["A"]; assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= sp.get("apart", ANCHOR_APART)
        assert solve(name, "synthetic", 2, G=G) == L                       # deterministic
    # scores / pass on synthetic traces: every pass rule reachable, and failable
    P_ = lambda name, **ev: score(_fake_trace(name, **ev))
    w = lambda n: SCENARIOS[n]["who"]
    f = P_("gas_endgame", n=300); assert not f["pass"] and f["scores"]["outlived"] == 0 and f["scores"]["survived"] == 1
    ok = P_("gas_endgame", n=300, edit={(w("gas_endgame")["e1"], "up"): [(100, 0)], (0, "gasd"): []}); assert ok["pass"] and ok["scores"]["place"] == 2
    g = P_("gas_endgame", n=300, edit={(1, "up"): [(100, 0)]}, once={(0, "gasd"): [(5, 900)]}); assert not g["pass"] and g["scores"]["gas_frac"] == 0.225
    k = P_("third_party", edit={(1, "up"): [(50, 0)], (0, "hp"): [(30, 2100)]}, once={(0, "kills"): [(50, 1)]})
    assert k["pass"] and k["scores"]["kill_low"] == 1 and k["scores"]["hp_min"] == 0.525
    assert not P_("third_party", edit={(0, "hp"): [(30, 1900)]}, once={(0, "kills"): [(50, 1)]})["pass"]
    assert P_("assassin_rush")["pass"] and not P_("assassin_rush", edit={(0, "up"): [(80, 0)]})["pass"]
    assert P_("assassin_rush", edit={(0, "up"): [(80, 0)]}, once={(0, "kills"): [(40, 1)], (0, "dealt"): [(40, 3000)], (0, "taken"): [(70, 1000)]})["scores"]["trade"] == 3.0
    l = P_("low_hp_close", edit={(0, "up"): [(120, 0)]}); assert not l["pass"] and l["scores"]["t_alive_s"] == 12.0
    assert P_("low_hp_close")["pass"]
    c = P_("cube_rich_low", edit={(5, "up"): [(60, 0)], (0, "cubes"): [(61, 4)]}, once={(0, "kills"): [(60, 1)]})
    assert c["pass"] and c["scores"]["cubes_gained"] == 4 and not P_("cube_rich_low", edit={(5, "up"): [(60, 0)]})["pass"]
    assert P_("mate_under_fire")["pass"] and not P_("mate_under_fire", edit={(1, "up"): [(30, 0)]})["pass"]
    assert not P_("mate_under_fire", edit={(0, "up"): [(30, 0)]}, once={(0, "up"): [(140, 1)]})["pass"]   # knocked out then back: fail
    # bush_memory: agent at (3000, 3000), enemy bush (13, 10) right-below-ish, control (10, 13); attacks aimed at each
    to = lambda t: float(_ang(_u(t)[0] - 3000, _u(t)[1] - 3000))
    ae, ac = to((13, 10)), to((10, 13))
    acts = {j: [0, 1, int(round(ae)) % 360, 0] for j in (12, 15, 18)}; acts[20] = [0, 1, int(round(ac)) % 360, 0]
    m = P_("bush_memory", n=50, hide=11, acts=acts)
    assert m["scores"]["aim_e_2"] == 3 and m["scores"]["aim_c_2"] == 1 and m["pass"] and m["scores"]["aim_e_0_5"] == 2, m   # actions 12, 15 of 12-16
    acts[22] = acts[24] = [0, 1, int(round(ac)) % 360, 0]
    assert not P_("bush_memory", n=50, hide=11, acts=acts)["pass"] and not P_("bush_memory", n=50, hide=11)["pass"]
    mv = P_("bush_memory", n=50, hide=11, acts={j: [1 + int(round(ae / 22.5)) % 16, 0, 0, 0] for j in range(12, 32)})
    assert mv["scores"]["move_e_2"] > 0.95 and mv["scores"]["move_c_2"] < mv["scores"]["move_e_2"] and not mv["pass"]
    # summary
    rows = [{"scn": {"name": "low_hp_close", "item": {"pick": p, "loc": "M"}, "pass": ps, "scores": {"survived": int(ps), "x": None}}}
            for p, ps in ((0, True), (0, False), (1, True), (1, True))]
    sm = summary(rows); assert sm["low_hp_close"]["pass"] == 0.75 and sm["low_hp_close"]["pass_pick"] == {"0": 0.5, "1": 1.0}
    assert sm["_all"]["n"] == 4 and sm["low_hp_close"]["survived"] == 0.75 and "x" not in sm["low_hp_close"]
    # real maps (game files from setup.sh), when present: every scenario resolves 4 layouts on both maps
    n_real = 0
    try:
        grid(MAPS[0])
    except Exception as e:
        print("scenarios: game files not available (%s): real-map layouts skipped" % type(e).__name__)
    else:
        import env_v5
        hs = set(env_v5.heroes()); assert set(roster()) <= hs, set(roster()) - hs
        for name in SCENARIOS:
            for loc in MAPS:
                L = layouts(name, loc); assert len(L) == 4, (name, loc, len(L)); n_real += 1
        cfg = battle_cfg(items(["mate_under_fire"])[0]); assert cfg["div"]["team_groups"] == TRIO and cfg["tag"]["kind"] == "evalscn" and cfg["tag"]["intent"] == "none"
    print("scenarios ok: %d scenarios, specs, layouts on a synthetic map, scores / pass rules%s" % (
        len(SCENARIOS), ", 4 layouts each on %s" % ", ".join(MAPS) if n_real else ""))


# ------------------------------------------------------------------ standalone run
def run(a):
    """boot --n instances, play every item (player 0 = the checkpoint's policy, sampled actions, or the native AI), write
    OUT/scenarios.jsonl (one summary row), OUT/scenario_episodes.jsonl, OUT/replays/scn/"""
    import torch
    import env_v5 as E, obs_v5 as O
    from model import Policy, HID, dist_logp_ent
    from train_ppo import priv, policy_s
    its = items([x for x in a.only.split(",") if x] or None, a.seeds); queue = list(its)
    out = a.out or os.path.join(E.WORK, "scenarios", "native" if a.native else os.path.splitext(os.path.basename(a.ckpt or "fresh"))[0])
    os.makedirs(out, exist_ok=True)
    net = Policy().eval()
    if a.ckpt: net.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model"], strict=False)
    net.add_heroes([O.brawler_stats(c) for c in sorted(set(E.heroes()) | set(roster()))])
    E.Env.slider_scale = 0.0; E.Env.record = bool(a.native)   # native: the action labels (_labels) are the trace's actions
    env = E.Env(a.n, lambda rng, i: battle_cfg(queue.pop(0), native=bool(a.native)) if queue else None, ticks=2, fair=True,
                max_ticks=4000, verbose=True, tag="scn%d" % os.getpid())
    ob = O.ObsV5(noise=O.NOISE, seed=a.seed); rec = Recorder(out); idx = ScnIndex(out); hid = {}; z = np.zeros(HID, np.float32)
    T = lambda x, dt=torch.float32: torch.as_tensor(x).to(dt); rows = []; t0 = None; n_ep = 0
    try:
        while len(rows) < len(its):
            obs, ids, _ = env.poll(min_ready=1, max_wait=0.003)
            for r in env.episodes[n_ep:]:
                if r.get("scn"): rows.append(r); idx.add([rec.finish(r)])
                elif r["tag"].get("kind") == "evalscn": rows.append({**r, "scn": None})
            n_ep = len(env.episodes)
            if all(s.state == "idle" for s in env.I): break
            if not len(ids): continue
            if t0 is None: t0 = time.time()
            o = ob.build(obs["vec"], obs["grid"], env.last_meta, ids); s = policy_s(env, ids, o)
            h = np.stack([hid.get(int(x), z) for x in ids])
            with torch.no_grad():
                lg, _, h2 = net(T(s), T(o["ent"]), T(o["grid"], torch.uint8), T(h), T(priv(env, ids)))
                _, _, A = dist_logp_ent(lg)
            A = A.numpy(); [hid.__setitem__(int(x), hh) for x, hh in zip(ids, h2.numpy())]
            rec.frame(env, ids, o, A); env.send(ids, A)
    finally:
        env.close()
    dt = time.time() - (t0 or time.time()); good = [r for r in rows if r.get("scn")]
    sm = {"ckpt": a.ckpt, "native": bool(a.native), "instances": a.n, "seconds": round(dt, 1), "episodes": len(rows),
          "invalid": len(rows) - len(good), "restarts": env.stats["restarts"], **summary(good)}
    open(os.path.join(out, "scenarios.jsonl"), "a").write(json.dumps(sm) + "\n")
    with open(os.path.join(out, "scenario_episodes.jsonl"), "a") as f:
        for r in good: f.write(json.dumps(episode_row(r)) + "\n")
    pre = [r["scn"]["pre"].get("s", 0) for r in good]
    print("%d episodes (%d invalid) on %d instances in %.0f s = %.2f min; pre-roll mean %.2f s; -> %s" % (
        len(rows), len(rows) - len(good), a.n, dt, dt / 60, float(np.mean(pre)) if pre else 0, out))
    for name, v in sm.items():
        if name in SCENARIOS:
            print("  %-16s n %2d pass %.2f  %s" % (name, v["n"], v["pass"], {k: x for k, x in v.items() if k not in ("n", "pass")}))
    return sm


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="self-check (no host)")
    ap.add_argument("--ckpt", default="", help="checkpoint (train_ppo / train_fast); '' = a fresh policy")
    ap.add_argument("--native", type=int, default=0, help="1 = the native AI plays player 0 (reference)")
    ap.add_argument("--n", type=int, default=2, help="game host instances")
    ap.add_argument("--seeds", type=int, default=4, help="layouts per (scenario, map, pick)")
    ap.add_argument("--only", default="", help="comma list of scenario names")
    ap.add_argument("--out", default="", help="output folder (default $SIM_WORK/scenarios/<ckpt name | native>)")
    ap.add_argument("--seed", type=int, default=0, help="observation noise seed")
    a = ap.parse_args()
    if a.check: return _check()
    run(a)


if __name__ == "__main__":
    sys.exit(main())
