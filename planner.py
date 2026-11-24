"""Rule planner (behaviour design 4.5): one intent per agent and second from what the policy sees, with a readable
reason; "none" (free play) when no rule fits. Numpy only, deterministic (no RNG), no host.

    python -m planner        # self-check (synthetic observations, no game)

    pl = RulePlanner()
    pk = pl.decide(self8, ent, grid, cond, sub)   # one agent's observation rows -> pick {rule, kind, dx, dy, slot, obj, reason}
    c = pl.plan(env, s, p, b, w, o, cond)         # env_v5.Env hook: the pick with a world target (intent.new candidate +
                                                  # rule / reason); kind "none" = free play (also when a target is not matched)
    preempts(old_rule, old_xy, new_rule, new_xy)  # may a planner tick replace a held planner intent (hysteresis)?

Input = exactly the policy's rows of one agent: self8 (obs_v5 "self"), ent (N_ENT, F_ENT), grid (25 x 25 seg classes, my
tile at [12, 12]), cond (N_COND: own stats, sliders, team size, HUD) + sub = my position inside my tile (0-1, x / y; the
camera tile offsets obs_v5._cam carry it live too). Offsets dx, dy in tiles (x right, y down). Grid classes count only
where a 4-neighbour has the same class (_solid: one wrong seg tile is not gas / a bush / the map edge).
Rules, first match wins (PRI = order; thresholds from the slider weights in cond, so plans follow the play style):
  1  gas           poison within 2 tiles (or under me) -> move_to the nearest walkable tile >= 3 tiles from the poison,
                   on the side away from the poison centroid           "gas 1 tile away: moving E out of it"
  2  low_hp        hp < 0.35 + 0.1 x (retreat - 1) and an enemy within 1.2 x its reach (known: max(range, super range),
                   else my range) -> hide in a bush 1-4 tiles away that is farther from it, else away from it
                                                                       "hp 30 %, enemy 4 tiles N: disengaging S"
  3  thrower_wall  a KNOWN thrower (ent col ID0+6) within its range + 1 with a wall between us (col LOS), me no thrower
                   -> engage it (around the wall) when hp >= 0.6 and (aggressive >= 1.2 or it is < 50 %), else away
                   from it (out of its range)                          "thrower TntDude 6 tiles NE behind a wall: leaving its range SW"
  4  keep_away     kite >= 0.5, a KNOWN short-range enemy (range <= 5.5 tiles, or melee and <= 8) within its range + 2,
                   my range >= 1.5 x its range -> away from it         "Undertaker (melee, range 3) 4 tiles E: keeping my distance W"
  5  fight         two enemies within 4 tiles of each other, both >= 4 tiles from me, their centre <= 10 tiles, one
                   below 90 % (they are trading)
                   -> third_party >= 1.2 and one < 40 %: engage it, else away from their centre
                                                                       "two fighting 6 tiles N, one at 30 %: third-partying"
  6  endgame       teams left = 2 (cond HUD) and poison in view -> hold a walkable tile 2.5-3.5 tiles from the poison,
                   the one farthest from the nearest enemy (- 0.5 x my distance to it)
                                                                       "endgame (2 left), enemy 5 tiles E: holding the gas edge 3 tiles W"
  7  regroup       team size > 1 and the nearest mate in view farther than 4 + 2 x (1 - leash) tiles (3-6)
                                                                       "mate 7 tiles E: regrouping"
  8  engage        the nearest enemy within 12 tiles that is weaker (known: its hp < mine - 0.2 x my max hp, in hit
                   points from the stats; unknown: its hp fraction < mine - 0.2), or within my range with aggressive >= 1.2
                                                                       "enemy 5 tiles E at 40 %: engaging"
  9  farm          a cube in view -> farm it; else (collect >= 0.5) the nearest box with no enemy within 6 tiles of me;
                   boxes / cubes within 1.5 tiles of poison are skipped (a farm target in the gas is invalid)
                                                                       "cube 3 tiles S: picking up"
  10 hide          bush_camp >= 1.2, no enemy in view, not in a bush, a bush 1-5 tiles away -> hide there
                                                                       "no enemy seen, bush 3 tiles W: hiding"
  11 explore       no enemy, no box in view, the map edge (out_of_bounds) in view -> move_to a walkable tile ~6 tiles
                   away from it                                        "map edge 5 tiles W: exploring E"
  12 none          free play                                           "no plan fits: free play"
Rules 3 / 4 need the enemy's brawler (ID_KNOWN = 1) and skip unknown enemies; 2 / 8 fall back to the unknown readings.
Noisy obs (RulePlanner(noisy=True), trainer --noise 1, live): a row counts only when confirmed: velocity != 0 (tracked
for >= 2 decisions; a new detection, e.g. a false one, has velocity 0: obs_v5 noisy path, live alike) or, for players,
its identity known. Clean obs (noisy=False): every row counts (static boxes / cubes have velocity 0 there).
"Away from" = avoid(that point, R 8) when it is closer than R - 1 = 7 tiles (avoid is achieved at R: farther would pay its
bonus for nothing), else move_to a walkable tile ~4 tiles straight away from it.
Hook (plan): the pick's offset + my true position -> the world target; anchored targets snap to the true object as live
re-anchors to the nearest detection (design 4.2): engage -> the nearest enemy I see within MATCH tiles, regroup -> the
nearest mate on my screen, farm -> the nearest box / cube of my fair state2 row within 1.5 tiles, hide -> the nearest
true bush tile within 1.5 tiles, move_to / hold -> the nearest truly walkable tile within 1.5 tiles (else the tile as
seen). Not matched -> kind "none" with the reason + " (target not matched)".
Preemption (design 4.4, Env.intents at a planner tick while a planner intent is held; random intents are never
preempted): a higher-priority rule replaces it if it is the gas rule or its target is > HYST tiles from the held one;
the same rule re-targets when its target moved > HYST tiles; lower rules and none never replace it.
"""
import sys
import numpy as np
import obs_v5 as O, intent as I

RULES = ("gas", "low_hp", "thrower_wall", "keep_away", "fight", "endgame", "regroup", "engage", "farm", "hide", "explore", "none")
PRI = {r: k for k, r in enumerate(RULES)}
TILE = 300.0
HYST = 2.0                # tiles: preemption / re-target only for a target change larger than this (design 4.4)
MATCH = 2.0               # tiles: a seen player matches the true one this close (1 tile of detector jitter + 2 decisions
                          # of latency at ~0.5 tile each)
SNAP = 1.5                # tiles: boxes / cubes / bushes / walkable tiles re-anchor this close (design 4.2: 1.5 tiles)
AVOID_MAX = I.SPEC["avoid"]["r"] - 1   # 7 tiles: avoid only what is closer (intent.candidates' rule too)
_C = 12                   # my grid cell (row and column)
_S = O.SEG.index
_WALK = np.isin(np.arange(len(O.SEG) + 1), [_S(c) for c in ("floor", "bush", "slow_tile", "fast_tile", "ice")])   # = env_v5._WALK
_T0 = int(O.COND_SLIDERS[9]) + 1          # cond: team size one-hot (3) after the first 10 sliders, then cubes, left
_LEFT = _T0 + 4
_SL = {k: int(i) for k, i in zip(O.SLIDERS, O.COND_SLIDERS)}
_E = {k: 1 + O.ENT.index(k) for k in ("enemy", "teammate", "power_box", "power_cube")}   # class one-hot columns


def dir8(dx, dy):
    """compass word of an offset (x right, y down: N = up on the screen)"""
    a = np.degrees(np.arctan2(-dy, dx)) % 360.0
    return ("E", "NE", "N", "NW", "W", "SW", "S", "SE")[int((a + 22.5) // 45) % 8]


def tiles(d):
    d = max(int(round(float(d))), 1)
    return "%d tile%s" % (d, "" if d == 1 else "s")


def _solid(m):
    """m and at least one 4-neighbour also m (a single wrong seg tile does not count)"""
    p = np.pad(m, 1)
    return m & (p[:-2, 1:-1] | p[2:, 1:-1] | p[1:-1, :-2] | p[1:-1, 2:])


def _name(row):
    """the brawler whose stats a known row carries (obs_v5.id_lut name; live: the countdown name)"""
    names, lut = O.id_lut(); d = ((lut[:-1] - row[O.ID0:O.ID0 + 13].astype(np.float32)) ** 2).sum(1)
    return names[int(d.argmin())]


class View:
    """one agent's observation rows, read for the rules"""

    def __init__(self, self8, ent, grid, cond, sub=(0.5, 0.5), noisy=True):
        self.hp, self.inbush = float(self8[0]), bool(self8[5] > 0.5)
        c = np.asarray(cond, np.float64); self.c = c
        self.w = {k: float(c[i]) for k, i in _SL.items()}
        self.rng = max(c[0] * 10.0, 2.0)          # own range in tiles (floor 2 = env_v5's 600 units)
        self.thrower, self.maxhp = bool(c[6] > 0.5), c[8] * 10000.0
        self.team = int(np.argmax(c[_T0:_T0 + 3])) + 1; self.left = c[_LEFT] * 10.0
        e = np.asarray(ent, np.float64); on = e[:, 0] > 0.5; cls = 1 + np.argmax(e[:, 1:1 + len(O.ENT)], 1)
        rows = lambda k: np.flatnonzero(on & (cls == _E[k]))
        self.e = e
        # noisy obs: a row counts once confirmed: tracked (velocity != 0: matched to the last decision's detection; a new
        # one, e.g. a false detection, has velocity 0) or, players, identity known. Clean obs (noisy False): every row
        trk = (e[:, O.VEL] != 0) | (e[:, O.VEL + 1] != 0)
        ok = np.ones(len(e), bool) if not noisy else trk | (e[:, O.ID_KNOWN] > 0.5)
        okb = np.ones(len(e), bool) if not noisy else trk
        sel = lambda k, m: rows(k)[m[rows(k)]]
        self.foes, self.mates, self.boxes, self.cubes = sel("enemy", ok), sel("teammate", ok), sel("power_box", okb), sel("power_cube", okb)
        self.xy = e[:, O.XY:O.XY + 2] * 10.0; self.d = np.hypot(self.xy[:, 0], self.xy[:, 1])
        self.ehp, self.known = e[:, O.HP], e[:, O.ID_KNOWN] > 0.5
        self.erng, self.esup = e[:, O.ID0] * 10.0, e[:, O.ID0 + 12] * 10.0
        self.thrown, self.melee, self.emax = e[:, O.ID0 + 6] > 0.5, e[:, O.ID0 + 7] > 0.5, e[:, O.ID0 + 8] * 10000.0
        self.wall = e[:, O.LOS] > 0.5
        g = np.asarray(grid); self.g = g
        jj, ii = np.mgrid[0:O.G, 0:O.G]
        self.ox, self.oy = ii - _C + 0.5 - float(sub[0]), jj - _C + 0.5 - float(sub[1])   # tile centre - me, tiles
        self.td = np.hypot(self.ox, self.oy)
        self.walk = _WALK[g] & (g != _S("poison"))
        self.poison, self.bush, self.edge = _solid(g == _S("poison")), _solid(g == _S("bush")), _solid(g == _S("out_of_bounds"))

    def name(self, k):
        """an enemy row as text: its brawler when known"""
        return _name(self.e[k]) if self.known[k] else "enemy"

    def dist_to(self, mask):
        """(G, G) tiles from every tile centre to the nearest tile of mask (inf without one)"""
        if not mask.any(): return np.full(self.g.shape, np.inf)
        px, py = self.ox[mask], self.oy[mask]
        return np.hypot(self.ox[..., None] - px, self.oy[..., None] - py).min(-1)

    def tile_pick(self, ok, key):
        """(dx, dy) of the tile with the smallest key among ok, or None"""
        if not ok.any(): return None
        k = np.where(ok, key, np.inf); j = np.unravel_index(int(np.argmin(k)), k.shape)
        return float(self.ox[j]), float(self.oy[j])


def pick(rule, kind, dx=0.0, dy=0.0, slot=-1, obj=None, reason=""):
    return {"rule": rule, "kind": kind, "dx": float(dx), "dy": float(dy), "slot": int(slot), "obj": obj, "reason": reason}


class RulePlanner:
    """decide(): the rules on one agent's observation rows; plan(): the env_v5.Env hook (world target, anchors).
    noisy = the observation has detector noise (trainer --noise 1, live): rows count once confirmed (module doc)"""

    def __init__(self, noisy=True):
        self.noisy = bool(noisy)

    def decide(self, self8, ent, grid, cond, sub=(0.5, 0.5)):
        v = View(self8, ent, grid, cond, sub, self.noisy)
        for r in RULES[:-1]:
            out = getattr(self, "_" + r)(v)
            if out is not None: return out
        return pick("none", "none", reason="no plan fits: free play")

    # ---------------------------------------------------------------- helpers
    def _away(self, v, rule, x, y, why):
        """leave the point (x, y): avoid it (R 8) when closer than AVOID_MAX, else move_to a walkable tile ~4 tiles away"""
        d = float(np.hypot(x, y)); ux, uy = (-x / d, -y / d) if d > 1e-6 else (1.0, 0.0)
        if d < AVOID_MAX: return pick(rule, "avoid", x, y, reason="%s %s" % (why, dir8(ux, uy)))
        tx, ty = 4 * ux, 4 * uy
        t = v.tile_pick(v.walk & (np.hypot(v.ox - tx, v.oy - ty) <= 2.0), np.hypot(v.ox - tx, v.oy - ty))
        tx, ty = t if t is not None else (tx, ty)
        return pick(rule, "move_to", tx, ty, reason="%s %s" % (why, dir8(tx, ty)))

    # ---------------------------------------------------------------- the rules (RULES order)
    def _gas(self, v):
        if not v.poison.any(): return None
        dme = float(v.td[v.poison].min())
        if dme > 2.0: return None
        pd = v.dist_to(v.poison); cx, cy = v.ox[v.poison].mean(), v.oy[v.poison].mean()
        ok = v.walk & (pd >= 3.0); away = ok & (v.ox * cx + v.oy * cy < 0)
        t = v.tile_pick(away if away.any() else ok, v.td)
        if t is None:                                    # nothing safe in view: straight away from the poison
            n = np.hypot(cx, cy)
            if n < 0.5: return None
            t = (-5 * cx / n, -5 * cy / n)
        where = "in the gas" if v.g[_C, _C] == _S("poison") else "gas %s away" % tiles(max(dme, 1.0))
        return pick("gas", "move_to", *t, reason="%s: moving %s out of it" % (where, dir8(*t)))

    def _low_hp(self, v):
        thr = float(np.clip(0.35 + 0.1 * (v.w["retreat"] - 1.0), 0.25, 0.45))   # retreat style: earlier / later
        if v.hp >= thr or not len(v.foes): return None
        reach = np.where(v.known, np.minimum(np.maximum(v.erng, v.esup), 12.0), v.rng)   # known: its range / super range
        th = [k for k in v.foes if v.d[k] <= 1.2 * reach[k]]
        if not th: return None
        k = min(th, key=lambda j: v.d[j]); ex, ey = v.xy[k]
        who = "%s %s %s" % (v.name(k), tiles(v.d[k]), dir8(ex, ey)) + (" (reach %s)" % tiles(reach[k]) if v.known[k] else "")
        why = "hp %d %%, %s" % (round(100 * v.hp), who)
        de = np.hypot(v.ox - ex, v.oy - ey)
        t = v.tile_pick(v.bush & (v.td >= 1.0) & (v.td <= 4.0) & (de >= v.d[k] + 1.0), v.td)
        if t is not None:
            return pick("low_hp", "hide", *t, reason="%s: hiding in a bush %s %s" % (why, tiles(np.hypot(*t)), dir8(*t)))
        return self._away(v, "low_hp", ex, ey, why + ": disengaging")

    def _thrower_wall(self, v):
        if v.thrower: return None                        # a thrower answers over the wall itself
        th = [k for k in v.foes if v.known[k] and v.thrown[k] and v.wall[k] and v.d[k] <= v.erng[k] + 1.0]
        if not th: return None
        k = min(th, key=lambda j: v.d[j]); ex, ey = v.xy[k]
        why = "thrower %s %s %s behind a wall" % (v.name(k), tiles(v.d[k]), dir8(ex, ey))
        if v.hp >= 0.6 and (v.w["aggressive"] >= 1.2 or v.ehp[k] < 0.5):
            return pick("thrower_wall", "engage", ex, ey, slot=k, reason=why + ": going around it to engage")
        return self._away(v, "thrower_wall", ex, ey, why + ": leaving its range")

    def _keep_away(self, v):
        if v.w["kite"] < 0.5: return None
        close = lambda k: v.erng[k] <= 5.5 or (v.melee[k] and v.erng[k] <= 8.0)   # Bull 5.3, Mortis 2.7 tiles
        th = [k for k in v.foes if v.known[k] and close(k) and v.d[k] <= v.erng[k] + 2.0 and v.rng >= 1.5 * v.erng[k]]
        if not th: return None
        k = min(th, key=lambda j: v.d[j]); ex, ey = v.xy[k]
        why = "%s (%srange %s) %s %s" % (v.name(k), "melee, " if v.melee[k] else "", int(round(v.erng[k])), tiles(v.d[k]), dir8(ex, ey))
        return self._away(v, "keep_away", ex, ey, why + ": keeping my distance")

    def _fight(self, v):
        f = [k for k in v.foes if v.d[k] >= 4.0]
        best = None
        for a in range(len(f)):
            for b in range(a + 1, len(f)):
                i, j = f[a], f[b]
                if np.hypot(*(v.xy[i] - v.xy[j])) <= 4.0 and min(v.ehp[i], v.ehp[j]) < 0.9:
                    c = (v.xy[i] + v.xy[j]) / 2; dc = float(np.hypot(*c))
                    if dc <= 10.0 and (best is None or dc < best[0]): best = (dc, i, j, c)   # farther: not my business
        if best is None: return None
        dc, i, j, c = best; lo = i if v.ehp[i] <= v.ehp[j] else j
        if v.w["third_party"] >= 1.2 and v.ehp[lo] < 0.4:
            return pick("fight", "engage", *v.xy[lo], slot=lo, reason="two fighting %s %s, one at %d %%: third-partying" % (
                tiles(dc), dir8(*c), round(100 * v.ehp[lo])))
        return self._away(v, "fight", c[0], c[1], "2 players fighting %s %s: going" % (tiles(dc), dir8(*c)))

    def _endgame(self, v):
        if not 1.5 < v.left < 2.5 or not v.poison.any(): return None   # 2 teams left (1: nobody to hold against)
        pd = v.dist_to(v.poison); ok = v.walk & (pd >= 2.5) & (pd <= 3.5) & (v.td <= 6.0)   # a spot I reach in the hold's 5 s
        if not ok.any(): return None
        if len(v.foes):
            k = min(v.foes, key=lambda j: v.d[j]); ex, ey = v.xy[k]
            t = v.tile_pick(ok, -(np.hypot(v.ox - ex, v.oy - ey) - 0.5 * v.td))
            why = "endgame (%d left), %s %s %s" % (round(v.left), v.name(k), tiles(v.d[k]), dir8(ex, ey))
        else:
            t = v.tile_pick(ok, v.td); why = "endgame (%d left)" % round(v.left)
        at = "here" if np.hypot(*t) < 0.75 else "%s %s" % (tiles(np.hypot(*t)), dir8(*t))
        return pick("endgame", "hold", *t, reason="%s: holding the gas edge %s" % (why, at))

    def _regroup(self, v):
        if v.team < 2 or not len(v.mates): return None
        thr = float(np.clip(4.0 + 2.0 * (1.0 - v.w["leash"]), 3.0, 6.0))   # leash style: tighter / looser
        k = min(v.mates, key=lambda j: v.d[j])
        if v.d[k] <= thr: return None
        return pick("regroup", "regroup", *v.xy[k], slot=k, reason="mate %s %s: regrouping" % (tiles(v.d[k]), dir8(*v.xy[k])))

    def _engage(self, v):
        mine = v.hp * v.maxhp; ok = []
        for k in v.foes:
            if v.d[k] > 12.0: continue                   # intent.candidates' engage reach
            if v.known[k]: weak = v.ehp[k] * v.emax[k] < mine - 0.2 * v.maxhp   # hit points from the stats
            else: weak = v.ehp[k] < v.hp - 0.2
            agg = v.d[k] <= v.rng and v.w["aggressive"] >= 1.2
            if weak or agg: ok.append((v.d[k], k, weak))
        if not ok: return None
        _, k, weak = min(ok); ex, ey = v.xy[k]
        why = "%s %s %s at %d %%" % (v.name(k), tiles(v.d[k]), dir8(ex, ey), round(100 * v.ehp[k]))
        if weak and v.known[k]: why += " (%d vs my %d hp)" % (round(v.ehp[k] * v.emax[k], -2), round(mine, -2))
        elif not weak: why += ", in my range, aggressive"
        return pick("engage", "engage", ex, ey, slot=k, reason=why + ": engaging")

    def _farm(self, v):
        if v.poison.any():                               # a target in the gas is invalid (intent.py): skip those next to poison
            gas = lambda k: np.hypot(v.ox[v.poison] - v.xy[k, 0], v.oy[v.poison] - v.xy[k, 1]).min() <= 1.5
            v.cubes, v.boxes = [k for k in v.cubes if not gas(k)], [k for k in v.boxes if not gas(k)]
        if len(v.cubes):
            k = min(v.cubes, key=lambda j: v.d[j])
            return pick("farm", "farm", *v.xy[k], slot=k, obj="cube", reason="cube %s %s: picking up" % (tiles(v.d[k]), dir8(*v.xy[k])))
        if not len(v.boxes) or v.w["collect"] < 0.5 or any(v.d[k] <= 6.0 for k in v.foes): return None
        k = min(v.boxes, key=lambda j: v.d[j])
        return pick("farm", "farm", *v.xy[k], slot=k, obj="box", reason="box %s %s, no enemy near: farming" % (tiles(v.d[k]), dir8(*v.xy[k])))

    def _hide(self, v):
        if v.w["bush_camp"] < 1.2 or len(v.foes) or v.inbush: return None
        t = v.tile_pick(v.bush & (v.td >= 1.0) & (v.td <= 5.0), v.td)
        if t is None: return None
        return pick("hide", "hide", *t, reason="no enemy seen, bush %s %s: hiding" % (tiles(np.hypot(*t)), dir8(*t)))

    def _explore(self, v):
        if len(v.foes) or len(v.boxes) or v.edge.sum() < 3: return None
        cx, cy = v.ox[v.edge].mean(), v.oy[v.edge].mean(); n = float(np.hypot(cx, cy))
        if n < 1e-6: return None
        tx, ty = -6 * cx / n, -6 * cy / n
        t = v.tile_pick(v.walk & (np.hypot(v.ox - tx, v.oy - ty) <= 2.0), np.hypot(v.ox - tx, v.oy - ty))
        t = t if t is not None else (tx, ty)
        return pick("explore", "move_to", *t, reason="map edge %s %s: exploring %s" % (tiles(v.td[v.edge].min()), dir8(cx, cy), dir8(*t)))

    # ---------------------------------------------------------------- env_v5.Env hook
    def plan(self, env, s, p, b, w, o, cond):
        """Env._choose / the preemption check: the pick of agent p (row b of this poll's observation o / cond rows) as an
        intent.new candidate {kind, q, pt, obj, rule, reason} on true world positions (w = Env._world); kind "none" = free
        play. o is the frame the policy acts on (noise + latency included)."""
        if o is None or cond is None:
            return dict(kind="none", rule="none", reason="no observation: free play")
        meta = getattr(env, "last_meta", None)
        sub = ((float(meta[b, 0]) % TILE) / TILE, (float(meta[b, 1]) % TILE) / TILE) if meta is not None and b < len(meta) else (0.5, 0.5)
        pk = self.decide(o["self"][b], o["ent"][b], o["grid"][b], cond[b], sub)
        return self.resolve(env, s, p, b, w, pk)

    def resolve(self, env, s, p, b, w, pk):
        """a pick -> a world-target candidate (module doc: anchors snap to the true object)"""
        c = dict(kind=pk["kind"], q=-1, pt=None, obj=pk["obj"], rule=pk["rule"], reason=pk["reason"])
        if pk["kind"] == "none": return c
        mx, my = float(w["x"][p]), float(w["y"][p]); x, y = mx + pk["dx"] * TILE, my + pk["dy"] * TILE
        miss = lambda: dict(c, kind="none", reason=pk["reason"] + " (target not matched)")
        if pk["kind"] in ("engage", "regroup"):
            oth = w["up"].copy(); oth[p] = False; mate = w["team"] == w["team"][p]
            ok = oth & (mate & w["screen"][p] if pk["kind"] == "regroup" else ~mate & w["sees"][p])
            d = np.where(ok, np.hypot(w["x"] - x, w["y"] - y) / TILE, np.inf)
            if d.min() > MATCH: return miss()
            c["q"] = int(np.argmin(d)); return c
        if pk["kind"] == "farm":
            sc = env._scene(s, p, b, w); xy = np.asarray(sc["boxes" if pk["obj"] == "box" else "cubes"], np.float64).reshape(-1, 2)
            d = np.hypot(xy[:, 0] - x, xy[:, 1] - y) / TILE if len(xy) else np.zeros(0)
            if not len(d) or d.min() > SNAP: return miss()
            c["pt"] = (float(xy[int(np.argmin(d)), 0]), float(xy[int(np.argmin(d)), 1])); return c
        if pk["kind"] == "avoid":
            c["pt"] = (x, y); return c
        mask = s.sl["bush"] if pk["kind"] == "hide" else s.sl["walk"]   # hide: a true bush; move_to / hold: walkable
        H, W = mask.shape; tx, ty = int(np.clip(x // TILE, 0, W - 1)), int(np.clip(y // TILE, 0, H - 1))
        if not mask[ty, tx]:
            n = int(np.ceil(SNAP)); y0, x0 = max(ty - n, 0), max(tx - n, 0)
            jj, ii = np.nonzero(mask[y0:ty + n + 1, x0:tx + n + 1]); jj, ii = jj + y0, ii + x0
            dd = np.hypot(ii - tx, jj - ty)
            if len(dd) and dd.min() <= SNAP: k = int(np.argmin(dd)); tx, ty = int(ii[k]), int(jj[k])
            elif pk["kind"] == "hide": return miss()
        c["pt"] = ((tx + 0.5) * TILE, (ty + 0.5) * TILE)
        return c


def preempts(old_rule, old_xy, new_rule, new_xy):
    """may a planner pick (new_rule, target new_xy in tiles) replace the held planner intent (old_rule, old_xy)? Random
    intents (old_rule None) and a none pick never; a higher-priority rule if it is gas or moves the target > HYST; the
    same rule if its target moved > HYST (re-target); lower rules never"""
    if old_rule is None or new_rule in (None, "none"): return False
    far = float(np.hypot(new_xy[0] - old_xy[0], new_xy[1] - old_xy[1])) > HYST
    if new_rule == old_rule: return far
    return PRI[new_rule] < PRI[old_rule] and (new_rule == "gas" or far)


# ------------------------------------------------------------------ self-check
def synth(foes=(), mates=(), boxes=(), cubes=(), hp=1.0, inbush=False, me="Sniper", sliders=None, team=1, left=None,
          grid=None, sub=(0.5, 0.5)):
    """synthetic rows of one agent: foes / mates = [(dx, dy) tiles, hp, brawler name or None (unknown), wall, (new)];
    player rows are tracked (velocity 0.01) unless new = True (velocity 0, a first / false detection); boxes / cubes =
    [(dx, dy)]; grid = (25, 25) seg classes (default all floor)"""
    self8 = np.zeros(O.N_SELF, np.float32); self8[0], self8[5] = hp, float(inbush)
    ent = np.zeros((O.N_ENT, O.F_ENT), np.float32); k = 0
    def row(cls, xy, h=1.0, name=None, wall=False, new=False):
        nonlocal k
        ent[k, 0] = 1; ent[k, 1 + O.ENT.index(cls)] = 1; ent[k, O.XY:O.XY + 2] = np.asarray(xy) / 10.0; ent[k, O.HP] = h
        if not new: ent[k, O.VEL] = 0.01                 # tracked (a noisy-obs row seen at the last decision too)
        if name: ent[k, O.ID0:O.ID0 + 13] = O.brawler_stats(name); ent[k, O.ID_KNOWN] = 1
        ent[k, O.LOS] = float(wall); k += 1
    for f in foes: row("enemy", *f)
    for m in mates: row("teammate", *m)
    k = O.N_ENT - 14                                    # boxes / cubes in the later blocks
    for x in boxes: row("power_box", x)
    for x in cubes: row("power_cube", x)
    g = np.full((O.G, O.G), _S("floor"), np.uint8) if grid is None else grid
    w = np.ones(len(O.SLIDERS), np.float32)
    for n, val in (sliders or {}).items(): w[O.SLIDERS.index(n)] = val
    cond = O.cond(me, w, team, 0, left)
    return self8, ent, g, cond


def _check():
    P = RulePlanner(); Fl = _S("floor")
    def run(**kw):
        sub = kw.pop("sub", (0.5, 0.5)); return P.decide(*synth(**kw), sub=sub)
    def grid(**cls):                                     # grid(poison=[(r0, r1, c0, c1)], bush=...): class blocks (rows, cols)
        g = np.full((O.G, O.G), Fl, np.uint8)
        for n, blocks in cls.items():
            for r0, r1, c0, c1 in blocks: g[r0:r1, c0:c1] = _S(n)
        return g
    seen = {}
    def want(pk, rule, kind, sub=None, sx=0, sy=0):
        assert pk["rule"] == rule and pk["kind"] == kind, (rule, kind, pk)
        assert sub is None or sub in pk["reason"], (sub, pk["reason"])
        if sx: assert np.sign(pk["dx"]) == sx, (rule, pk)
        if sy: assert np.sign(pk["dy"]) == sy, (rule, pk)
        seen[rule] = pk["reason"]; return pk
    # 1 gas: a poison band 2 tiles right (cols 14-24) -> west, >= 3 tiles from it; under me: "in the gas"
    gg = grid(poison=[(0, 25, 14, 25)])
    pk = want(run(grid=gg), "gas", "move_to", "gas 2 tiles away: moving W", sx=-1)
    assert min(np.hypot(pk["dx"] - (c - _C), pk["dy"] - (r - _C)) for r in range(25) for c in range(14, 25)) >= 3
    want(run(grid=grid(poison=[(0, 25, 11, 25)])), "gas", "move_to", "in the gas: moving W", sx=-1)
    assert run(grid=grid(poison=[(0, 25, 16, 25)]))["rule"] != "gas"                     # 4 tiles away: no
    g1 = grid(); g1[_C, _C + 1] = _S("poison"); assert run(grid=g1)["rule"] == "none"       # one wrong seg tile is no gas
    # 2 low hp: unknown enemy 4 tiles N (my range 10): a bush south -> hide; none -> avoid it (S); known reach
    want(run(hp=0.3, foes=[((0, -4), 1.0, None, False)], grid=grid(bush=[(14, 16, 11, 13)])), "low_hp", "hide", "hp 30 %, enemy 4 tiles N: hiding in a bush", sy=1)
    pk = want(run(hp=0.3, foes=[((0, -4), 1.0, None, False)]), "low_hp", "avoid", "disengaging S", sy=-1)   # avoid: target = the enemy
    assert run(hp=0.3, foes=[((0, -11), 1.0, None, False)], me="Undertaker")["rule"] != "low_hp"   # unknown: 1.2 x my range (2.7)
    want(run(hp=0.3, foes=[((0, -9), 1.0, "Gunslinger", False)], me="Undertaker"), "low_hp", "move_to", "Gunslinger 9 tiles N (reach 11 tiles): disengaging S", sy=1)   # known reach, far: walk away
    assert run(hp=0.3, sliders={"retreat": 0}, foes=[((0, -4), 1.0, None, False)])["rule"] != "low_hp"   # retreat 0: threshold 0.25
    # 3 thrower behind a wall (known): leave its range / engage at high hp and aggressive; unknown or no wall: skip
    T = ((5, -3), 1.0, "TntDude", True)
    want(run(hp=0.5, foes=[T]), "thrower_wall", "avoid", "thrower TntDude 6 tiles NE behind a wall: leaving its range SW", sx=1)
    want(run(foes=[T], sliders={"aggressive": 1.5}), "thrower_wall", "engage", "going around it to engage")
    assert run(hp=0.5, foes=[((5, -3), 1.0, None, True)])["rule"] != "thrower_wall"        # ID_KNOWN 0: skipped
    assert run(hp=0.5, foes=[((5, -3), 1.0, "TntDude", False)])["rule"] != "thrower_wall"   # no wall
    assert run(hp=0.5, foes=[T], me="Barkeep")["rule"] != "thrower_wall"                   # I am a thrower too
    # 4 keep away from a known close-range enemy (me Piper, range 10)
    want(run(foes=[((4, 0), 1.0, "Undertaker", False)]), "keep_away", "avoid", "Undertaker (melee, range 3) 4 tiles E: keeping my distance W")
    assert run(foes=[((4, 0), 1.0, None, False)])["rule"] != "keep_away"                    # unknown: skipped
    assert run(foes=[((4, 0), 1.0, "Undertaker", False)], sliders={"kite": 0.2})["rule"] != "keep_away"
    assert run(foes=[((4, 0), 1.0, "Undertaker", False)], me="Luchador")["rule"] != "keep_away"   # not outranging it
    # 5 fight: two enemies 3 tiles apart, 6 tiles N, one at 30 %: third party at >= 1.2, else go the other way
    F2 = [((-1.5, -6), 0.3, None, False), ((1.5, -6), 0.8, None, False)]
    want(run(foes=F2, sliders={"third_party": 1.5}), "fight", "engage", "two fighting 6 tiles N, one at 30 %: third-partying")
    assert run(foes=F2, sliders={"third_party": 1.5})["dx"] < 0                             # the low one
    want(run(foes=F2), "fight", "avoid", "2 players fighting 6 tiles N: going S")
    assert run(foes=[((-1.5, -6), 1.0, None, False), ((1.5, -6), 1.0, None, False)])["rule"] != "fight"   # nobody hurt: not a fight
    # 6 endgame: 2 left, poison 4+ tiles W, an enemy E -> hold 2.5-3.5 tiles from the poison
    ge = grid(poison=[(0, 25, 0, 8)])
    pk = want(run(left=2, grid=ge, foes=[((5, 0), 1.0, None, False)], sliders={"aggressive": 0}), "endgame", "hold", "endgame (2 left), enemy 5 tiles E: holding the gas edge")
    assert 2.5 <= _C + pk["dx"] - 7 <= 3.5 + 1e-9, pk
    assert run(left=3, grid=ge)["rule"] != "endgame" and run(left=1, grid=ge)["rule"] != "endgame"
    # 7 regroup: trio, mate 7 tiles E (leash 1: > 4 tiles)
    want(run(team=3, mates=[((7, 0), 1.0, None, False)]), "regroup", "regroup", "mate 7 tiles E: regrouping", sx=1)
    assert run(team=3, mates=[((3, 0), 1.0, None, False)])["rule"] != "regroup" and run(mates=[((7, 0), 1.0, None, False)])["rule"] != "regroup"
    assert run(team=3, mates=[((5, 0), 1.0, None, False)], sliders={"leash": 0})["rule"] != "regroup"   # leash 0: > 6 tiles
    # 8 engage: weaker by fraction (unknown) / by hit points (known) / aggressive in range
    want(run(foes=[((5, 0), 0.4, None, False)]), "engage", "engage", "enemy 5 tiles E at 40 %: engaging", sx=1)
    assert run(foes=[((5, 0), 0.9, None, False)])["rule"] == "none"
    assert run(foes=[((5, 0), 0.4, None, False, True)])["rule"] == "none"                  # a new unknown detection: not yet
    assert run(foes=[((5, 0), 0.4, "Gunslinger", False, True)])["rule"] == "engage"        # known: counts at once
    want(run(foes=[((5, 0), 0.9, "Sniper", False)], hp=1.0, me="BullDude"), "engage", "engage", "(2500 vs my 5000 hp)")   # 90 % of a weaker brawler
    assert run(foes=[((5, 0), 0.5, "BullDude", False)], hp=0.8, me="Sniper")["rule"] != "engage"   # 50 % of a tank: no
    want(run(foes=[((5, 0), 1.0, None, False)], sliders={"aggressive": 1.5}), "engage", "engage", "in my range, aggressive")
    # 9 farm: a cube; a box with no enemy near; collect < 0.5: no box
    want(run(cubes=[(0, 3)], boxes=[(2, 0)]), "farm", "farm", "cube 3 tiles S: picking up", sy=1)
    pk = want(run(boxes=[(-4, 0)]), "farm", "farm", "box 4 tiles W, no enemy near: farming", sx=-1); assert pk["obj"] == "box"
    assert run(boxes=[(-4, 0)], sliders={"collect": 0.3})["rule"] != "farm"
    assert run(cubes=[(0, -5)], grid=grid(poison=[(0, 7, 0, 25)]))["rule"] != "farm"       # a cube 1 tile from poison: skipped
    assert run(foes=[((-1.5, -11), 0.3, None, False), ((1.5, -11), 0.8, None, False)])["rule"] != "fight"   # 11 tiles: not mine
    s8, e, g, c = synth(cubes=[(0, 3)]); e[:, O.VEL] = 0                                  # new detections (e.g. false ones)
    assert P.decide(s8, e, g, c)["rule"] == "none" and RulePlanner(noisy=False).decide(s8, e, g, c)["rule"] == "farm"   # clean: all count
    # 10 hide: bush camper, no enemy, bush 3 tiles W; not while in a bush; not below 1.2
    gb = grid(bush=[(11, 14, 8, 10)])
    want(run(grid=gb, sliders={"bush_camp": 1.5}), "hide", "hide", "no enemy seen, bush 3 tiles W: hiding", sx=-1)
    assert run(grid=gb, sliders={"bush_camp": 1.5}, inbush=True)["rule"] == "none" and run(grid=gb)["rule"] == "none"
    # 11 explore: the map edge 6 tiles W in view, nothing else -> east
    gx = grid(out_of_bounds=[(0, 25, 0, 7)])
    want(run(grid=gx), "explore", "move_to", "map edge 6 tiles W: exploring E", sx=1)
    assert run(grid=gx, boxes=[(3, 3)])["rule"] == "farm" and run(grid=gx, foes=[((6, 0), 1.0, None, False)])["rule"] == "none"
    # 12 fallback
    want(run(), "none", "none", "no plan fits: free play")
    assert set(seen) == set(RULES), set(RULES) - set(seen)
    # priority order: each state built for rule k also matching a later rule picks k
    rich = dict(team=3, mates=[((7, 0), 1.0, None, False)], cubes=[(0, 3)], sliders={"bush_camp": 1.5})
    order = [("gas", dict(grid=gg, hp=0.3, foes=[((0, -4), 1.0, None, False)])),
             ("low_hp", dict(hp=0.3, foes=[T, ((0, -4), 1.0, None, False)])),
             ("thrower_wall", dict(hp=0.5, foes=[T, ((4, 0), 1.0, "Undertaker", False)])),
             ("keep_away", dict(foes=[((4, 0), 1.0, "Undertaker", False)] + F2)),
             ("fight", dict(foes=F2, left=2, grid=ge)), ("endgame", dict(left=2, grid=ge, **rich)),
             ("regroup", dict(foes=[((5, 0), 0.4, None, False)], **rich)), ("engage", dict(foes=[((5, 0), 0.4, None, False)], cubes=[(0, 3)])),
             ("farm", dict(cubes=[(0, 3)], grid=gb, sliders={"bush_camp": 1.5})), ("hide", dict(grid=np.where(gx == Fl, gb, gx), sliders={"bush_camp": 1.5})),
             ("explore", dict(grid=gx))]
    for rule, kw in order:
        pk = run(**kw); assert pk["rule"] == rule, (rule, pk)
    # the sub-tile position shifts tile offsets (my tile centre = 0.5 - sub)
    pk = run(boxes=[(-4, 0)], sub=(0.9, 0.5)); assert np.isclose(pk["dx"], -4.0)
    pk = run(grid=gb, sliders={"bush_camp": 1.5}, sub=(0.9, 0.5)); assert np.isclose(pk["dx"], -3.4), pk   # entity rows: exact offsets
    # preemption / hysteresis
    assert preempts("farm", (0, 0), "gas", (1, 0)) and not preempts(None, (0, 0), "gas", (5, 0))   # random intents: never
    assert preempts("farm", (0, 0), "low_hp", (5, 0)) and not preempts("farm", (0, 0), "low_hp", (1, 1))   # higher: target > 2 tiles
    assert not preempts("low_hp", (0, 0), "farm", (9, 9)) and not preempts("farm", (0, 0), "none", (9, 9))  # lower / none: never
    assert preempts("engage", (0, 0), "engage", (3, 0)) and not preempts("engage", (0, 0), "engage", (1.5, 1))   # re-target
    print("planner ok: %d rules fire on their states in priority order, identity rules skip unknown enemies, fallback none, "
          "preemption hysteresis %.0f tiles" % (len(RULES), HYST))
    for r in RULES: print("  %-12s %s" % (r, seen[r]))


if __name__ == "__main__":
    sys.exit(_check())
