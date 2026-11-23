"""Planning layer: one named intent + a goal point about once a second, with a readable reason, from the same
observation the reflex policy gets (obs_v5). Written rules first, so every decision can be debugged from its log line.

    python -m planner            # self-check (no game files, no battle)

    pl = Planner()                     # one per agent and episode
    plan = pl.update(o, cond)          # every decision; o = one agent's {"self": (8,), "ent": (39, 16), "grid": (25, 25)},
                                       # cond = its current obs_v5.cond (28,): the HUD numbers (cubes, teams left) change
    plan.intent, plan.goal, plan.reason    # "move_to", (dx, dy) tiles from the player (x right, y down), "gas 1 tile away: ..."
    plan.fired                         # every rule that fit at the last re-plan, most urgent first (the log line)
    encode(plan)                       # (len(INTENTS) + 2,) intent one-hot + goal / 10, input for a goal-following policy

Rules, most urgent first; the first that fits wins (Ryan's table in the BSRL thread, plus `lobbed` and `regen`):
  gas        gas within 2 tiles                        move_to   "gas 1 tile away: moving E out of it"
  low_hp     HP < 35 %, enemy within 6 tiles           hide / avoid  "hp 30 %, enemy 4 tiles N: disengaging S"
  lobbed     enemy behind a wall, its shots land on me engage / avoid  "enemy 5 tiles N behind a wall, lobbing me: flanking NE"
  fight      two enemies fighting within 10 tiles      engage (third-party) / avoid
  endgame    2 or fewer teams left                     hold      "endgame, 2 left: holding the gas edge W"
  regroup    teammate more than 4 tiles away           regroup
  engage     weak enemy, or push slider high           engage    "enemy 5 tiles E at 40 %: engaging"
  farm       power cube, or a box with nobody near     farm      "cube 3 tiles S: picking up"
  regen      HP < 60 %, nobody seen                    hide      "hp 50 %, nobody seen: regenerating in the bush 3 tiles W"
  bush_camp  bush-camp slider high, nobody seen        hide
  explore    map edge in view                          explore   "map edge 5 tiles W: exploring E"
  none       nothing fits                              none      "no plan fits: free play" (the reflex policy plays alone)

Robustness to the measured detector noise (obs_v5.NOISE):
  * an entity counts only once it is seen in 2 of the last 3 decisions at the same place (false positives are up to 6 %
    per empty slot, so a single frame shows ~2 phantom detections);
  * gas, bush, wall and map-edge tiles count only inside a patch of 2+ tiles (seg flips are up to 14 % per tile, but
    scattered), gas only with 3+ and the map edge with 5+ such tiles in view;
  * an enemy confirmed on 2+ decisions stays remembered for MEMORY decisions after its last sighting (bush, missed
    detections); a single confirmation is not remembered (that is where the rare confirmed phantoms are);
  * a plan that hangs on one detection (FLICKER rules) is held for HOLD re-plans while its rule stops fitting (a cube
    missed for a frame), unless a more urgent rule fits or a goal it walked to is reached. Other rules read the patched
    grid or the enemy memory, so their rule not fitting is real news.
  * retreat, bush and cube goals stay out of the gas (GAS_SAFE).
shortcut: own position is dead reckoning from the commanded velocity (self 6 / 7); fine over the 5 s memory, not longer.
shortcut: enemy teams are unknown in the observation, so in team modes two enemies count as fighting only once one of
  them is below 85 % HP.
shortcut: a single isolated wall tile is dropped with the seg noise (rare in Showdown layouts; walls come in patches).
"""
import math, sys
from collections import deque
from dataclasses import dataclass, field
import numpy as np
import obs_v5 as O

INTENTS = ("none", "move_to", "avoid", "hide", "regroup", "engage", "hold", "farm", "explore")
RULES = ("gas", "low_hp", "lobbed", "fight", "endgame", "regroup", "engage", "farm", "regen", "bush_camp", "explore", "none")
PERIOD = 10        # decisions per re-plan: 10 decisions x 2 ticks x 50 ms = 1 s
HOLD = 2           # re-plans a plan survives after its own rule stopped fitting (one missed cube / enemy frame)
FLICKER = ("lobbed", "fight", "engage", "farm", "regroup")   # rules that hang on one detection (held); the rest read the
# patched grid or the enemy memory, so their rule not fitting is real news (no hold)
MEMORY = 50        # decisions an enemy stays remembered after its last confirmed sighting (5 s)
GATE = 1.5         # tiles: same class this close in an earlier decision = the same object (jitter .4 + 2 decisions of motion)
HIGH = 1.4         # slider weight (0-2) that counts as "high" (sampled uniform 0-2 per battle: the top 30 %)
GAS_SAFE = 3       # tiles from the gas a retreat / bush goal must keep: the gas rule pulls back from 2, so 3 avoids ping-pong
LOW_HP, REGEN_HP, FIGHT_HP = 0.35, 0.6, 0.6   # Ryan's table: disengage under 35 %; heal / no new fights under 60 %
LOB_R, SHOT_R = 9, 2.0   # lobbed: an enemy within 9 tiles (throwers' range ~ 7-9) and its shot within 2 tiles of me
PAIR_R, FIGHT_R = 4, 10  # fight: two enemies within 4 tiles of each other (trading range), their middle within 10 of me
MEM_R = 3          # tiles: a confirmed enemy this close to a remembered one is the same enemy (one decision ~ .5 tile move)
G, C = O.G, O.G // 2
_SL = {n: 13 + i for i, n in enumerate(O.SLIDERS)}       # slider -> cond index


def _mask(*names):
    m = np.zeros(len(O.SEG) + 1, bool); m[[O.SEG.index(n) for n in names]] = True
    return m


MOVE_BLOCK = _mask("indestructible", "destructible_wall", "fragile_wall", "water", "rope_fence", "out_of_bounds",
                   "destructible", "power_box")
SHOT_BLOCK = _mask("indestructible", "destructible_wall", "fragile_wall", "destructible", "power_box")
DY, DX = np.mgrid[0:G, 0:G] - C                          # tile offsets from the player, x right / y down


def _patch(m):
    """keep tiles of mask m that have an 8-neighbour in m (drops isolated seg flips)"""
    p = np.pad(m, 1); n = sum(p[1 + y:G + 1 + y, 1 + x:G + 1 + x] for y in (-1, 0, 1) for x in (-1, 0, 1)) - m
    return m & (n > 0)


def _big(m, n):
    return m if m.sum() >= n else np.zeros_like(m)


def compass(dx, dy):
    if math.hypot(dx, dy) < 0.5: return "here"
    return ("E", "NE", "N", "NW", "W", "SW", "S", "SE")[round(math.atan2(-dy, dx) / (math.pi / 4)) % 8]


def tiles(d):
    return "%d tile%s" % (round(d), "" if round(d) == 1 else "s")


@dataclass
class Plan:
    rule: str
    intent: str
    goal: tuple          # (dx, dy) tiles from the player, or None
    reason: str
    fired: list = field(default_factory=list)


def encode(plan):
    """(len(INTENTS) + 2,) float32: intent one-hot + goal / 10 tiles (clipped to +-2; 0 without a goal)"""
    v = np.zeros(len(INTENTS) + 2, np.float32); v[INTENTS.index(plan.intent)] = 1.0
    if plan.goal is not None: v[-2:] = np.clip(np.asarray(plan.goal, np.float32) / 10, -2, 2)
    return v


class View:
    """one decision's cleaned scene: masks, path distances from the player, confirmed / remembered entities (tiles)"""

    def __init__(self, o, cond, ents, remembered):
        s = o["self"]; g = np.asarray(o["grid"])
        self.hp, self.ammo, self.in_bush = float(s[0]), float(s[1]), bool(s[5] > 0.5)
        self.range = max(float(cond[0]) * 10, 2.0)        # CastingRange / 30 -> tiles (3 sub-units per tile)
        self.thrown = bool(cond[6] > 0.5)
        self.sl = {n: float(cond[i]) for n, i in _SL.items()}
        self.team = int(np.argmax(cond[23:26])) + 1; self.cubes = round(float(cond[26]) * 10); self.left = round(float(cond[27]) * 10)
        # gas and the map edge are big areas: under 3 / 5 patched tiles in view = paired seg flips (~.1 per frame at 14 %)
        self.poison = _big(_patch(g == O.SEG.index("poison")), 3); self.bush = _patch(g == O.SEG.index("bush"))
        self.oob = _big(_patch(g == O.SEG.index("out_of_bounds")), 5)
        self.block = _patch(MOVE_BLOCK[g]); self.shot = _patch(SHOT_BLOCK[g])
        self.path = self._bfs()
        pt = np.argwhere(self.poison)
        self.gas_d = (np.hypot(DX[..., None] - (pt[:, 1] - C), DY[..., None] - (pt[:, 0] - C)).min(-1) if len(pt)
                      else np.full((G, G), np.inf))
        by = lambda k: [e for e in ents if e[0] == k]
        self.enemies, self.mates = by("enemy"), by("teammate")
        self.cubes_e, self.boxes = by("power_cube"), by("power_box")
        self.shots = by("enemy_projectile") + by("enemy_area_effect")
        self.remembered = remembered

    def _bfs(self):
        """(G, G) 8-connected path length in tiles from the player (no corner cutting), inf where unreachable"""
        d = np.full((G, G), np.inf); d[C, C] = 0; q = deque([(C, C)])
        while q:
            y, x = q.popleft()
            for ddy, ddx, w in ((0, 1, 1), (1, 0, 1), (0, -1, 1), (-1, 0, 1), (1, 1, 1.4), (1, -1, 1.4), (-1, 1, 1.4), (-1, -1, 1.4)):
                ny, nx = y + ddy, x + ddx
                if not (0 <= ny < G and 0 <= nx < G) or self.block[ny, nx] or d[ny, nx] <= d[y, x] + w: continue
                if ddy and ddx and (self.block[y, nx] or self.block[ny, x]): continue
                d[ny, nx] = d[y, x] + w; q.append((ny, nx))
        return d

    def los(self, dx, dy, fx=0.0, fy=0.0):
        """no shot-blocking tile on the line (fx, fy) -> (dx, dy), endpoints excluded"""
        n = int(max(abs(dx - fx), abs(dy - fy)) * 4) + 1   # 4 samples per tile: no tile on the line is skipped
        ends = {(round(C + dy), round(C + dx)), (round(C + fy), round(C + fx))}   # a wall-labelled own tile (bleed) hides nothing
        for i in range(1, n):
            t = i / n; y, x = round(C + fy + (dy - fy) * t), round(C + fx + (dx - fx) * t)
            if 0 <= y < G and 0 <= x < G and self.shot[y, x] and (y, x) not in ends: return False
        return True

    def pick(self, score, maxd=12.0):     # 12 = the grid's half width
        """reachable tile (path <= maxd) with the lowest score -> (dx, dy), or None"""
        sc = np.where(np.isfinite(self.path) & (self.path <= maxd), score, np.inf)
        if not np.isfinite(sc).any(): return None
        y, x = np.unravel_index(np.argmin(sc), sc.shape)
        return float(x - C), float(y - C)

    def away(self, tx, ty, maxd=8.0, bush_bonus=0.0):
        """reachable tile far from (tx, ty), preferring short paths, bushes and cover from it"""
        far = np.hypot(DX - tx, DY - ty)
        sc = -far + 0.3 * self.path - bush_bonus * self.bush     # .3: 1 tile of distance from it is worth ~3 tiles of walking
        return self.pick(np.where(self.gas_d >= GAS_SAFE, sc, np.inf), maxd) or self.pick(sc + 10 * (self.gas_d < GAS_SAFE), maxd)


def _dist(e):
    return math.hypot(e[1], e[2])


def rules(v):
    """every (rule, intent, goal, reason) that fits this view, most urgent first"""
    out = []
    add = lambda r, i, g, why: g is not None and out.append((r, i, g, why))
    threats = v.enemies + [("enemy", x, y, 1.0) for x, y in v.remembered]
    near = min(threats, key=_dist) if threats else None
    seen = bool(threats)

    gd = float(v.gas_d[C, C])                              # gas: gas within 2 tiles (0 = standing in it)
    if gd <= 2:
        g = v.pick(np.where(v.gas_d >= 4, v.path, 100 + v.path - 10 * np.minimum(v.gas_d, 4)))
        if g: add("gas", "move_to", g, ("in the gas" if gd < 0.5 else "gas %s away" % tiles(gd)) + ": moving %s out of it" % compass(*g))

    if near and v.hp < LOW_HP and _dist(near) <= 6:        # low_hp: disengage (6 tiles ~ most attack ranges), into a bush
        g = v.away(near[1], near[2], bush_bonus=3.0)       # 3: a bush is worth 10 tiles of walking (out of its sight)
        if g: add("low_hp", "hide" if v.bush[round(C + g[1]), round(C + g[0])] else "avoid", g,
                  "hp %d %%, enemy %s %s: disengaging %s" % (v.hp * 100, tiles(_dist(near)), compass(near[1], near[2]), compass(*g)))

    lob = [e for e in v.enemies if _dist(e) <= LOB_R and not v.los(e[1], e[2])]   # lobbed: hit from behind a wall
    if lob and any(_dist(s) <= SHOT_R for s in v.shots):
        e = min(lob, key=_dist); where = "enemy %s %s behind a wall, lobbing me" % (tiles(_dist(e)), compass(e[1], e[2]))
        if v.thrown and v.hp >= FIGHT_HP and v.ammo >= 0.33:
            add("lobbed", "engage", _at_range(v, e), where + ": I throw too, trading over the wall")
        elif not v.thrown and v.hp >= FIGHT_HP:
            ok = np.array([[v.los(e[1], e[2], x - C, y - C) for x in range(G)] for y in range(G)])
            ok &= np.hypot(DX - e[1], DY - e[2]) <= v.range
            g = v.pick(np.where(ok, v.path, np.inf), 8)
            if g: add("lobbed", "engage", g, where + ": flanking %s for a clear shot" % compass(*g))
            else: add("lobbed", "avoid", v.away(e[1], e[2]), where + ": no way round, backing off")
        else: add("lobbed", "avoid", v.away(e[1], e[2]), where + ": backing off out of its range")

    pairs = [(a, b) for i, a in enumerate(v.enemies) for b in v.enemies[i + 1:]       # fight: two enemies trading
             if math.hypot(a[1] - b[1], a[2] - b[2]) <= PAIR_R and (v.team == 1 or min(a[3], b[3]) < 0.85)]
    pairs = [p for p in pairs if math.hypot((p[0][1] + p[1][1]) / 2, (p[0][2] + p[1][2]) / 2) <= FIGHT_R]
    if pairs:
        a, b = pairs[0]; mx, my = (a[1] + b[1]) / 2, (a[2] + b[2]) / 2; weak = min((a, b), key=lambda e: e[3])
        what = "2 players fighting %s %s" % (tiles(math.hypot(mx, my)), compass(mx, my))
        if v.sl["third_party"] >= HIGH and v.hp >= FIGHT_HP and v.ammo >= 0.66 and weak[3] <= 0.5:   # 2 of 3 shots left
            add("fight", "engage", _at_range(v, weak), what + ", one at %d %%: third-partying" % (weak[3] * 100))
        else:
            g = v.away(mx, my); add("fight", "avoid", g, what + ": going the other way" + (" (%s)" % compass(*g) if g else ""))

    if 0 < v.left <= 2:                                     # endgame: hold the gas edge (or cover) and let them come
        if v.poison.any(): g = v.pick(np.where((v.gas_d >= 2) & (v.gas_d <= 4), v.path - 2 * v.bush, np.inf)); at = "the gas edge"
        else: g = v.pick(np.where(v.bush | _near(v.shot), v.path, np.inf), 6); at = "cover"
        if g: add("endgame", "hold", g, "endgame, %d left%s: holding %s %s" % (
            v.left, ", enemy %s %s" % (tiles(_dist(near)), compass(near[1], near[2])) if near else "", at, compass(*g)))

    if v.team > 1 and v.mates:                              # regroup
        m = min(v.mates, key=_dist)
        if _dist(m) > 4: add("regroup", "regroup", (m[1], m[2]), "mate %s %s: regrouping" % (tiles(_dist(m)), compass(m[1], m[2])))

    if v.enemies and v.ammo >= 0.33:                        # engage: a weak enemy, or push slider high and healthy (1 shot left)
        e = min(v.enemies, key=lambda e: (not v.los(e[1], e[2]), e[3], _dist(e)))
        # reach: own range + ~a third to close in, at least 8 tiles so short-range brawlers still chase a weak one
        if _dist(e) <= max(8, v.range * 1.3) and ((e[3] <= 0.5 and v.hp >= 0.4) or (v.sl["push"] >= HIGH and v.hp >= FIGHT_HP)):
            add("engage", "engage", _at_range(v, e), "enemy %s %s at %d %%: engaging" % (tiles(_dist(e)), compass(e[1], e[2]), e[3] * 100))

    gas_at = lambda e: v.gas_d[min(max(round(C + e[2]), 0), G - 1), min(max(round(C + e[1]), 0), G - 1)]
    cubes = [c for c in v.cubes_e if gas_at(c) >= 2]        # farm: cubes out of the gas first, boxes only healthy, nobody close
    if cubes:
        c = min(cubes, key=_dist); add("farm", "farm", (c[1], c[2]), "cube %s %s: picking up" % (tiles(_dist(c)), compass(c[1], c[2])))
    elif v.boxes and v.hp >= 0.5 and not (near and _dist(near) <= 6):   # shooting a box stops regen and shows where I am
        b = min(v.boxes, key=_dist); add("farm", "farm", (b[1], b[2]), "box %s %s: breaking it" % (tiles(_dist(b)), compass(b[1], b[2])))

    bush = v.pick(np.where(v.bush & (v.gas_d >= GAS_SAFE), v.path, np.inf), 8)
    v.in_bush = v.in_bush and gd >= GAS_SAFE               # a bush in the gas heals nothing
    if not seen and v.hp < REGEN_HP and (bush or v.in_bush):    # regen: heal out of sight (stop shooting, stop being seen)
        g = (0.0, 0.0) if v.in_bush else bush
        add("regen", "hide", g, "hp %d %%, nobody seen: regenerating in %s" % (v.hp * 100, "this bush" if v.in_bush else "the bush %s %s" % (tiles(math.hypot(*g)), compass(*g))))
    if not seen and v.sl["bush_camp"] >= HIGH and (bush or v.in_bush):
        g = (0.0, 0.0) if v.in_bush else bush
        add("bush_camp", "hide", g, "no enemy seen, %s: hiding" % ("in a bush" if v.in_bush else "bush %s %s" % (tiles(math.hypot(*g)), compass(*g))))

    if not seen and v.oob.any():                            # explore: away from the map edge
        oy, ox = np.nonzero(v.oob); ex, ey = float((ox - C).mean()), float((oy - C).mean()); n = math.hypot(ex, ey) or 1.0
        g = v.pick(np.hypot(DX + ex / n * 6, DY + ey / n * 6) + 0.1 * v.path)
        d = float(np.hypot(ox - C, oy - C).min())
        if g and math.hypot(*g) >= 1: add("explore", "explore", g, "map edge %s %s: exploring %s" % (tiles(d), compass(ex, ey), compass(*g)))

    out.append(("none", "none", None, "no plan fits: free play"))
    return out


def _near(m):
    return _patch(m | np.roll(m, 1, 0) | np.roll(m, -1, 0) | np.roll(m, 1, 1) | np.roll(m, -1, 1)) & ~m


def _at_range(v, e):
    """goal for an attack: the reachable tile closest to 80 % of own range from the enemy, on my side, with a clear shot"""
    want = min(v.range * 0.8, _dist(e))
    gx, gy = e[1] - e[1] / max(_dist(e), 1e-6) * want, e[2] - e[2] / max(_dist(e), 1e-6) * want
    return v.pick(np.hypot(DX - gx, DY - gy) + 0.1 * v.path) or (e[1], e[2])


class Planner:
    """per agent and episode: entity confirmation + memory, re-plan every PERIOD decisions, hold plans against flicker"""

    def __init__(self, ticks=2, period=PERIOD):
        self.ticks, self.period = ticks, period
        self.t, self.pos = 0, np.zeros(2)                  # pos: dead-reckoned own position, tiles
        self.frames = deque(maxlen=2); self.mem = {}       # frames: the 2 decisions before this one (2 of 3 confirmation)
        # mem: id -> (x, y, last decision, confirmations) of confirmed enemies, world frame
        self.plan, self.goal_w, self.walk, self.hold, self.prio = None, None, False, 0, len(RULES)

    def _entities(self, o):
        ent, dets = o["ent"], []
        for row in ent[ent[:, 0] > 0.5]:
            cls = O.ENT[int(row[1:1 + len(O.ENT)].argmax())]
            dets.append((cls, float(row[O.XY]) * 10 + self.pos[0], float(row[O.XY + 1]) * 10 + self.pos[1], float(row[O.HP])))
        conf = [d for d in dets if any(p[0] == d[0] and math.hypot(p[1] - d[1], p[2] - d[2]) <= GATE for f in self.frames for p in f)]
        self.frames.append(dets)
        for d in conf:
            if d[0] != "enemy": continue
            k = min(self.mem, key=lambda i: math.hypot(self.mem[i][0] - d[1], self.mem[i][1] - d[2]), default=None)
            if k is None or math.hypot(self.mem[k][0] - d[1], self.mem[k][1] - d[2]) > MEM_R: k = max(self.mem, default=-1) + 1
            self.mem[k] = (d[1], d[2], self.t, self.mem.get(k, (0, 0, 0, 0))[3] + 1)
        self.mem = {i: m for i, m in self.mem.items() if self.t - m[2] <= MEMORY}
        rel = [(c, x - self.pos[0], y - self.pos[1], h) for c, x, y, h in conf]
        en = [(x, y) for c, x, y, _ in rel if c == "enemy"]
        rem = [(x - self.pos[0], y - self.pos[1]) for x, y, t, n in self.mem.values()
               if n >= 2 and t < self.t and all(math.hypot(x - self.pos[0] - a, y - self.pos[1] - b) > MEM_R for a, b in en)]
        return rel, rem

    def update(self, o, cond):
        self.t += 1
        self.pos += np.asarray(o["self"][6:8], np.float64) * self.ticks
        ents, rem = self._entities(o)
        if self.t % self.period == 0:                      # first re-plan after PERIOD decisions: entities need 2 sightings
            self._replan(View(o, np.asarray(cond, np.float32), ents, rem))
        if self.plan is None: return Plan("none", "none", None, "warming up: free play")
        g = None if self.goal_w is None else (float(self.goal_w[0] - self.pos[0]), float(self.goal_w[1] - self.pos[1]))
        return Plan(self.plan.rule, self.plan.intent, g, self.plan.reason, self.plan.fired)

    def _replan(self, v):
        cands = rules(v)
        fired = ["%s: %s" % (r, why) for r, _, _, why in cands]
        best = cands[0]; p = RULES.index(best[0])
        reached = self.walk and math.hypot(*(self.goal_w - self.pos)) < 1.0   # only a goal it had to walk to (not "stay here")
        held = self.plan is not None and self.plan.rule != "none" and not reached
        same = [c for c in cands if self.plan is not None and c[0] == self.plan.rule]
        if held and same and p >= self.prio:                # own rule still fits: keep the rule, refresh goal / reason
            best = same[0]; self.hold = HOLD
        elif held and self.plan.rule in FLICKER and self.hold > 0 and p > self.prio:   # missed detection: keep it a little longer
            self.hold -= 1; self.plan = Plan(self.plan.rule, self.plan.intent, None, self.plan.reason.split(" (holding")[0] + " (holding)", fired)
            return
        else: self.hold = HOLD
        r, i, g, why = best
        self.prio = RULES.index(r)
        self.goal_w = None if g is None else self.pos + np.asarray(g, np.float64)
        self.walk = g is not None and math.hypot(*g) >= 1.0
        self.plan = Plan(r, i, g, why, fired)


# ---------------------------------------------------------------------------------------------------------------- check
def _scene(hp=1.0, ammo=1.0, vel=(0, 0), ents=(), grid=None, in_bush=False):
    s = np.zeros(O.N_SELF, np.float32); s[0], s[1], s[5] = hp, ammo, in_bush; s[6:8] = vel
    e = np.zeros((O.N_ENT, O.F_ENT), np.float32)
    for k, (cls, dx, dy, h) in enumerate(ents):
        e[k, 0] = 1; e[k, 1 + O.ENT.index(cls)] = 1; e[k, O.XY:O.XY + 2] = dx / 10, dy / 10; e[k, O.HP] = h
    g = np.full((G, G), O.SEG.index("floor"), np.uint8) if grid is None else grid
    return {"self": s, "ent": e, "grid": g}


def _cond(team=1, left=None, thrown=False, rng=0.8, **sl):
    c = np.zeros(O.N_COND, np.float32); c[0] = rng; c[6] = thrown
    for n in O.SLIDERS: c[_SL[n]] = sl.get(n, 1.0)
    c[22 + team] = 1; c[26] = 0; c[27] = (O.LEFT0[team] if left is None else left) / 10
    return c


def _run(cond, o, n=PERIOD, pl=None):
    pl = pl or Planner()
    for _ in range(n): p = pl.update(o, cond)
    return pl, p


def _noisy(rng, o, seg=0.137, fp=0.06):
    """the top of obs_v5.NOISE's seg / false-positive ranges on a scene (in place)"""
    g = o["grid"]; w = rng.random(g.shape) < seg; g[w] = rng.integers(len(O.SEG), size=int(w.sum()))
    for k in np.nonzero((o["ent"][:, 0] == 0) & (rng.random(O.N_ENT) < fp))[0]:
        o["ent"][k, 0] = 1; o["ent"][k, 1 + rng.integers(len(O.ENT))] = 1
        o["ent"][k, O.XY:O.XY + 2] = rng.uniform(-1.5, 1.5), rng.uniform(-1.17, 0.97); o["ent"][k, O.HP] = rng.random()
    return o


def _check():
    S = O.SEG.index
    def grid(**rects):
        g = np.full((G, G), S("floor"), np.uint8)
        for name, rs in rects.items():
            for x0, y0, x1, y1 in rs: g[C + y0:C + y1 + 1, C + x0:C + x1 + 1] = S(name)
        return g
    fired = lambda p: " ".join(p.fired)
    # gas to the west, 1 tile away -> out of it, east
    _, p = _run(_cond(), _scene(grid=grid(poison=[(-12, -12, -1, 12)])))
    assert p.rule == "gas" and p.goal[0] > 0 and "gas 1 tile away: moving E" in p.reason, p
    # in the gas, wall east: path goes round, never into the wall
    _, p = _run(_cond(), _scene(grid=grid(poison=[(-12, -12, 0, 12)], indestructible=[(1, -12, 2, 4)])))
    assert p.rule == "gas" and "in the gas" in p.reason and p.goal[0] >= 2 and p.goal[1] > 4, p
    # low hp, enemy 4 north, bush to the south: hide there; enemy east, gas 3 west: disengage, but not into the gas
    _, p = _run(_cond(), _scene(hp=0.3, ents=[("enemy", 0, -4, 1.0)], grid=grid(bush=[(-1, 3, 1, 5)])))
    assert p.rule == "low_hp" and p.intent == "hide" and p.goal[1] > 0 and "hp 30 %, enemy 4 tiles N: disengaging S" in p.reason, p
    _, p = _run(_cond(), _scene(hp=0.3, ents=[("enemy", 4, 0, 1.0)], grid=grid(poison=[(-12, -12, -3, 12)])))
    assert p.rule == "low_hp" and p.goal[0] >= 0, p
    # thrower behind a wall lobbing me: flank (not a thrower) / trade from own range (thrower) / back off (low hp)
    sc = _scene(ents=[("enemy", 0, -5, 1.0), ("enemy_area_effect", 0, 0.5, 0)], grid=grid(indestructible=[(-2, -3, 2, -2)]))
    _, p = _run(_cond(), sc)
    assert p.rule == "lobbed" and p.intent == "engage" and abs(p.goal[0]) >= 2 and "flanking" in p.reason, p
    _, p = _run(_cond(thrown=True), sc); assert p.rule == "lobbed" and "trading over the wall" in p.reason, p
    far = _scene(ents=[("enemy", 0, -9, 1.0), ("enemy_area_effect", 0, 0.5, 0)], grid=grid(indestructible=[(-2, -3, 2, -2)]))
    _, p = _run(_cond(thrown=True, rng=0.5), far); assert p.rule == "lobbed" and p.goal[1] <= -3, p   # 5-tile range: steps in
    sc["self"][0] = 0.5; _, p = _run(_cond(thrown=True), sc); assert p.rule == "lobbed" and p.intent == "avoid", p
    # a wall-labelled own tile (bleed) does not hide an enemy in the open
    _, p = _run(_cond(), _scene(ents=[("enemy", 5, 0, 1.0), ("enemy_area_effect", 0, 1, 0)], grid=grid(indestructible=[(0, 0, 0, 1)])))
    assert p.rule != "lobbed", p
    # two players fighting north-west: avoid by default, third-party one at 30 % when the slider asks for it
    fight = [("enemy", -5, -5, 0.3), ("enemy", -3, -6, 0.9)]
    _, p = _run(_cond(), _scene(ents=fight))
    assert p.rule == "fight" and p.intent == "avoid" and p.goal[0] > 0 and p.goal[1] > 0 and "NW: going the other way" in p.reason, p
    _, p = _run(_cond(third_party=2.0), _scene(ents=fight)); assert p.intent == "engage" and "third-partying" in p.reason, p
    # endgame next to the gas: hold its edge; the HUD's teams left changes during the episode
    sc = _scene(grid=grid(poison=[(-12, -12, -6, 12)]))
    pl, p = _run(_cond(left=5), sc); assert p.rule != "endgame", p
    _, p = _run(_cond(left=2), sc, pl=pl)
    assert p.rule == "endgame" and -4 <= p.goal[0] <= -2 and "holding the gas edge W" in p.reason, p
    # duo, mate 7 east: regroup; weak enemy: engage at range; cube: farm (not one in the gas); box: not with an enemy near
    _, p = _run(_cond(team=2), _scene(ents=[("teammate", 7, 0, 1.0)])); assert p.rule == "regroup" and "mate 7 tiles E" in p.reason, p
    _, p = _run(_cond(), _scene(ents=[("enemy", 5, 0, 0.4)])); assert p.rule == "engage" and "enemy 5 tiles E at 40 %" in p.reason, p
    _, p = _run(_cond(), _scene(ents=[("power_cube", 0, 3, 0)])); assert p.rule == "farm" and np.allclose(p.goal, (0, 3)) and "cube 3 tiles S" in p.reason, p
    _, p = _run(_cond(), _scene(ents=[("power_cube", -7, 0, 0)], grid=grid(poison=[(-12, -12, -3, 12)]))); assert p.rule != "farm", p
    _, p = _run(_cond(), _scene(ents=[("power_box", -4, 0, 1), ("enemy", 3, 0, 1.0)])); assert p.rule == "none", p
    # regen / bush camp / explore / none; no bush in the gas
    _, p = _run(_cond(), _scene(hp=0.5, grid=grid(bush=[(-4, -1, -3, 1)]))); assert p.rule == "regen" and "bush 3 tiles W" in p.reason, p
    _, p = _run(_cond(), _scene(hp=0.5, grid=grid(bush=[(-5, -1, -4, 1)], poison=[(-12, -12, -3, 12)]))); assert "regen" not in fired(p), p
    _, p = _run(_cond(bush_camp=2.0), _scene(grid=grid(bush=[(-4, -1, -3, 1)]))); assert p.rule == "bush_camp" and p.intent == "hide", p
    _, p = _run(_cond(), _scene(grid=grid(out_of_bounds=[(-12, -12, -5, 12)])))
    assert p.rule == "explore" and p.goal[0] > 3 and "map edge 5 tiles W: exploring E" in p.reason, p
    _, p = _run(_cond(), _scene()); assert p.rule == "none" and p.goal is None and p.reason == "no plan fits: free play", p
    assert np.array_equal(encode(p)[:len(INTENTS)], np.eye(len(INTENTS))[0]) and np.allclose(encode(Plan("x", "farm", (30, -4), ""))[-2:], [2.0, -0.4])
    # one-frame phantom: not confirmed, no plan
    pl, _ = _run(_cond(), _scene(), 2 * PERIOD - 1)
    assert pl.update(_scene(ents=[("enemy", 3, 0, 0.3)]), _cond()).rule == "none"         # decision 2 x PERIOD = a re-plan
    # memory: an enemy seen, then gone into a bush, still blocks regen / bush camping for MEMORY decisions
    low = _scene(hp=0.5, grid=grid(bush=[(-4, -1, -3, 1)]))
    pl, _ = _run(_cond(), _scene(hp=0.5, ents=[("enemy", 6, 0, 1.0)], grid=low["grid"]))
    pl, p = _run(_cond(), low, pl=pl); assert p.rule == "none" and "regen" not in fired(p), p
    _, p = _run(_cond(), low, MEMORY, pl=pl); assert p.rule == "regen", p
    # hold: a cube missed for a re-plan keeps the farm plan; gas preempts it at once; the goal follows own movement
    pl, _ = _run(_cond(), _scene(ents=[("power_cube", 0, 6, 0)]))
    pl, p = _run(_cond(), _scene(vel=(0, 0.1)), pl=pl)
    assert p.rule == "farm" and "(holding)" in p.reason and abs(p.goal[1] - (6 - 0.2 * PERIOD)) < 1e-6, p
    _, p = _run(_cond(), _scene(grid=grid(poison=[(-12, -12, -1, 12)])), pl=pl); assert p.rule == "gas", p
    # hold also for a goal at the player's own tile (a weak enemy already in range)
    pl, p = _run(_cond(), _scene(ents=[("enemy", 3, 0, 0.3)])); assert p.rule == "engage" and np.allclose(p.goal, 0), p
    _, p = _run(_cond(), _scene(), pl=pl); assert p.rule == "engage" and "(holding)" in p.reason, p
    # measured noise, 3 seeds x 300 re-plans: empty scene -> no phantom-driven plan; low hp by a bush -> still regenerates
    bad = kept = 0
    for seed in range(3):
        rng, pl, pr = np.random.default_rng(seed), Planner(), Planner()
        for t in range(PERIOD * 300):
            p = pl.update(_noisy(rng, _scene()), _cond())
            q = pr.update(_noisy(rng, _scene(hp=0.5, grid=grid(bush=[(-4, -1, -3, 1)]))), _cond())
            if (t + 1) % PERIOD == 0: bad += p.rule != "none"; kept += q.rule == "regen"
    # phantom plans: 10 / 3000 over seeds 0-9 (mostly a confirmed phantom cube, then held); bound ~2 % of 900
    assert bad <= 15 and kept >= 0.9 * 900, (bad, kept)
    print("planner ok: %d rules, %d intents; noisy: %d / 900 phantom plans, %d / 900 regen kept" % (len(RULES), len(INTENTS), bad, kept))


if __name__ == "__main__":
    _check()
