"""Super audit: which brawlers' supers work on this host (a client build of the game: server-only code is bare `ret` stubs,
some of it re-implemented in nirkt's env.c / spawnfix.c). Output is a per-brawler checklist for a GitHub tracking issue.

    python -m super_audit [--n 2] [--only BullDude,Sniper]     # live: boots --n hosts (60-90 s each)
    python -m super_audit --check                              # classification self-check, no host

Setup (one battle per 10 brawlers, map MAP = the most open Showdown map, storm off, every player's native AI off):
players 0-4 shoot in phase A, players 5-9 in phase B; in each phase every shooter stands on its own clear spot (a 3-tile
wide corridor of open tiles in at least two directions, spots >= SPACING apart; A and B use different spots) with one
enemy target at distance d along a corridor. Per phase: teleport + heal everyone and end spawn protection; the target fires
one normal attack at the shooter (so a self-heal is visible); fill the shooters' supers; fire the super (fire 2, aim at the
target, super_dist = d / CastingRange, a charge aims ~1 tile past the target); watch WATCH ticks. A super the player steers
while it runs (FlyJump, Ww) has its shooter walk towards the target once its CastingTime is over (moves sent while it
casts are dropped), so its "move" is plain walking.
Measured per shooter (state2 dump each step, attributed by owner player index): super consumed (charge < half of max),
damage = target HP lost (or the shooter's NbeResp damage_dealt summed over the watch steps: the host reports it per step), own max displacement (dash / jump / charge; host `leaping` flag), target max
displacement (pushback / pull), own HP gain (only when the pre-damage hit), new projectiles / summons / area effects
(name, table radius) / other items owned by the shooter, a change of the brawler's character (transform), the skill's
"active" word (state2 ultiActive, informational), foreign summons near the pair (contamination from phase A).
Expected effects come from the game's tables (skills.csv UltimateSkill row, its projectiles_logic / projectiles_skin
rows, area_effects_logic): damage, move, push, heal, projectile, summon, area, item, change (`expected()`).
Classes (`classify()`): works = consumed and every expected effect seen; partial = consumed, some effect seen but not all
expected (listed as missing); unverified = consumed, but the super's effect is of a kind not visible here (shield,
invisibility, status buff, a heal when the pre-damage missed); no_effect = consumed, nothing seen; not_fired = charge not
consumed (or the brawler was not spawned); crash = the host died with this brawler as the only shooter.
Not detectable: stuns / slows / silences, shields, damage boosts, invisibility, wall breaking, effects that start after
WATCH ticks, effects on allies (every player is solo), how close a re-implementation is to the real game.
Every non-works brawler is retried once (other corridor direction, d from a smaller fraction of its range); the best
attempt counts. A host crash requeues that phase's shooters one at a time (a reboot each) to find the brawler.
Output: $SIM_WORK/super_audit.json (per brawler: name, super, type, class, attempts with measurements) and
$SIM_WORK/super_audit.md (also stdout): a checklist grouped by class then super type. With --only the brawlers are
merged into an existing super_audit.json."""
import argparse, collections, functools, json, os, sys, threading, time
import numpy as np

NIL = -2147483648
MAP = "Survival651"                      # the most open Showdown map (88 % open tiles; others give < 6 clear spots)
TILE = 300
CORR = 8                                 # corridor length in tiles from the shooter's tile: targets up to 7 tiles (2100 u)
SPACING = 13                             # tiles between spot centres (3900 u: a long super past its target stays clear)
WATCH = 50                               # ticks watched after the fire (2.5 s at 20 ticks/s; regen starts 3 s after a hit)
STEP = 2                                 # ticks per env step while watching
DIST_F = (0.6, 0.35)                     # target distance per attempt, as a fraction of the super's CastingRange
DIST_MIN, DIST_MAX = 450, (CORR - 1) * TILE
MOVE_MIN, PUSH_MIN, HEAL_MIN = 300, 100, 100   # units / units / HP: above walk jitter, setup rounding and regen ticks
FILLER = "ShotgunGirl"                   # target-only slot filler when a battle has fewer than 10 brawlers to audit
CLASSES = ("works", "partial", "unverified", "no_effect", "not_fired", "crash")   # best first
P2, C2, J2, I2, HDR = 8, 23, 18, 12, 8   # state2 row sizes (engine/py/state2.py)
BOX_IDS = {16000120}


# ------------------------------------------------------------------ game tables
@functools.lru_cache(None)
def _tables():
    from nbpc import diversity as D
    named = lambda f: {r[next(iter(r))]: r for r in D._rows(f) if r[next(iter(r))]}
    rows = lambda f: [r for r in D._rows(f) if r[next(iter(r))]]
    return {"char": named("characters.csv"), "skill": named("skills.csv"), "plog": named("projectiles_logic.csv"),
            "pskin": named("projectiles_skin.csv"), "alog": named("area_effects_logic.csv"), "askin": named("area_effects_skin.csv"),
            "rows": {6: rows("projectiles_skin.csv"), 16: rows("characters.csv"), 17: rows("area_effects_skin.csv"), 18: rows("items.csv")}}


def data_name(did):
    """global data id (6 projectiles_skin, 16 characters, 17 area_effects_skin, 18 items) -> row name"""
    t, i = divmod(int(did), 1000000); r = _tables()["rows"].get(t)
    if not r or not 0 <= i < len(r): return str(did)
    return r[i][next(iter(r[i]))]


@functools.lru_cache(None)
def _texts():
    from nbpc import diversity as D
    import csv
    p = os.path.join(os.path.dirname(D.ASSETS.rstrip("/")), "assets", "localization", "texts.csv")
    try:
        return {r[0]: r[1] for r in csv.reader(open(p, encoding="utf-8")) if len(r) > 1}
    except OSError:
        return {}


def display(name):
    t = _texts().get(_tables()["char"].get(name, {}).get("TID", ""), "")
    return t.title() if t else name


def area_radius(did):
    """area_effects_skin data id -> table radius (units; obs_v5's lookup)"""
    import obs_v5 as O
    lut, na = O._area_lut(); i = int(did) - 16999999
    return int(round(float(lut[i, 0]) * 3000)) if 0 < i < na else 0


def _num(r, k):
    try: return float((r.get(k) or "0").split(";")[0] or 0)
    except ValueError: return 0.0


def expected(name):
    """the brawler's super from the tables -> dict(skill, type (BehaviorType), kind (finer group label), range (units),
    charge (bool), expect (effects that must be seen), notes)"""
    T = _tables(); ch = T["char"][name]; sk_name = ch.get("UltimateSkill", ""); s = T["skill"].get(sk_name, {})
    bt = s.get("BehaviorType") or "?"; ex = []; notes = []
    projs = [p for p in (s.get("Projectiles") or "").split(";") if p]
    pl = [T["plog"].get(T["pskin"].get(p, {}).get("LogicProjectile") or p, {}) for p in projs]
    ps = [T["pskin"].get(p, {}) for p in projs]
    areas = [a for a in (s.get("AreaEffectObject"), s.get("AreaEffectObject2")) if a]
    areas += [a for r in ps for a in (r.get("SpawnAreaEffectObject"), r.get("SpawnAreaEffectObject2")) if a]
    alog = [T["alog"].get(T["askin"].get(a, {}).get("Logic") or a, {}) for a in areas]
    chain = [r.get(k) for r in alog for k in ("ChainAreaEffect", "ChainAreaEffect2", "ChainAreaEffect3") if r.get(k)]
    alog += [T["alog"].get(T["askin"].get(a, {}).get("Logic") or a, {}) for a in chain]   # an area's follow-up area
    summons = [x for x in (s.get("SummonedCharacters") or "").split(";") if x] + [r["SpawnCharacter"] for r in pl if r.get("SpawnCharacter")]
    items = [x for x in [s.get("SpawnedItem")] + [r.get("SpawnItem") for r in ps] if x]
    charge = bt == "Charge"
    if charge or bt in ("Blink", "BlinkReturn", "BlinkToPlaceholder", "BlinkToProjectile", "FlyJump"): ex.append("move")
    turns = bt == "ChangeCharacter" or (charge and bool(s.get("SummonedCharacters")))   # Bonnie / Moe: the charge ends in the other form
    if turns: ex.append("change")
    if projs: ex.append("projectile")
    if turns: summons = []   # SummonedCharacters is the character it turns into, not a summon
    if summons: ex.append("summon")
    if areas: ex.append("area")
    if items: ex.append("item")
    leap_bombs = charge and bool(items)          # Piper: the damage is the bombs left at the start point, not on the target
    heals_only = any(_num(r, "HealOwnPercent") > 0 for r in pl)   # Poco: the skill Damage is the heal, enemies are not hit
    if ((_num(s, "Damage") > 0 or any(_num(r, "Damage") > 0 for r in alog)) and not leap_bombs and not heals_only
            and bt not in ("ChangeCharacter", "ProjectileShield")): ex.append("damage")
    if _num(s, "ChargePushback") > 0 or any(_num(r, "PushbackStrength") > 0 for r in pl + alog): ex.append("push")
    if (_num(s, "HealSelfOnActivationPower") > 0 or any(_num(r, "HealOwnPercent") > 0 for r in pl)
            or any(r.get("Type") in ("Heal", "Hot", "HealRegen", "HealAndDamage", "HotAndDot") for r in alog)): ex.append("heal")
    if bt in ("Shield", "Invisibility", "StatusEffect", "ProjectileShield", "MountUp", "Split", "Ww", "RecallWeapon"):
        notes.append(f"{bt}: its main effect is not visible here")
    kind = (f"Charge {s.get('ChargeType') or '?'}" if charge else
            "Attack: summon" if bt == "Attack" and summons else "Attack: area" if bt == "Attack" and areas and not projs else
            "Attack: item" if bt == "Attack" and items else "Attack: projectile" if bt == "Attack" else bt)
    cr = _num(s, "CastingRange") * 100
    ring = max((( _num(r, "InnerRadius") + _num(r, "Radius")) / 2 for r in alog
                if _num(r, "InnerRadius") > 0 and _num(r, "InnerRadiusDamagePercent") == 0), default=0.0)
    if cr <= 0 and not projs and alog and not charge:   # no aim range, the effect is an area at the brawler: the target stands inside it
        cr = max((_num(r, "Radius") for r in alog), default=0.0)
    steer = bt in ("FlyJump", "Ww")   # the brawler moves under its player's control while the super runs: walk it to the target
    return {"skill": sk_name, "type": bt, "kind": kind, "range": cr if cr > 0 else 2600.0, "charge": charge, "ring": ring, "steer": steer, "cast": _num(s, "CastingTime"), "expect": ex,
            "projectiles": projs, "summons": summons, "areas": areas, "items": items, "notes": notes}


# ------------------------------------------------------------------ classification
def observed(m):
    """measurements -> the set of effects seen"""
    o = set()
    if m.get("damage", 0) > 0 and not m.get("foreign_near"): o.add("damage")
    if m.get("move", 0) >= MOVE_MIN or m.get("leaping"): o.add("move")
    if m.get("push", 0) >= PUSH_MIN: o.add("push")
    if m.get("heal", 0) >= HEAL_MIN: o.add("heal")
    for k, e in (("projectiles", "projectile"), ("summons", "summon"), ("areas", "area"), ("items", "item")):
        if m.get(k): o.add(e)
    if m.get("changed"): o.add("change")
    return o


def classify(m, expect):
    """(measurements, expected effects) -> (class, missing effects, effects seen)"""
    if m.get("crashed"): return "crash", [], set()
    if not m.get("spawned", True) or not m.get("consumed"): return "not_fired", [], set()
    ex = [e for e in expect if e != "heal" or m.get("pre_damaged")]
    seen = observed(m); missing = [e for e in ex if e not in seen]
    if not ex: return "unverified", [], seen
    if not missing: return "works", [], seen
    return ("partial" if seen else "no_effect"), missing, seen


def describe(m, missing=()):
    """what was seen, short (the checklist line)"""
    if m.get("crashed"): return "host crashed with it as the only shooter"
    if not m.get("spawned", True): return "brawler not spawned (host rejected it)"
    if not m.get("filled", True): return "super could not be filled (super max %s)" % m.get("super_max")
    if not m.get("consumed"): return "super charge not consumed"
    p = []
    if m.get("damage", 0) > 0: p.append("%d damage%s" % (m["damage"], " (foreign summon near: not counted)" if m.get("foreign_near") else ""))
    if m.get("move", 0) >= MOVE_MIN or m.get("leaping"): p.append("moved %d u%s" % (m.get("move", 0), " (leap)" if m.get("leaping") else ""))
    if m.get("push", 0) >= PUSH_MIN: p.append("target pushed %d u" % m["push"])
    if m.get("heal", 0) >= HEAL_MIN: p.append("healed %d" % m["heal"])
    cnt = lambda xs: ", ".join("%s x%d" % (k, v) if v > 1 else k for k, v in collections.Counter(xs).items())
    if m.get("projectiles"): p.append("projectiles: " + cnt(m["projectiles"]))
    if m.get("summons"): p.append("summons: " + cnt(m["summons"]))
    if m.get("areas"): p.append("areas: " + cnt("%s r%d" % (a["name"], a["radius"]) for a in m["areas"]))
    if m.get("items"): p.append("items: " + cnt(m["items"]))
    if m.get("changed"): p.append("character changed")
    s = "; ".join(p) if p else "nothing seen"
    if missing: s += "; missing: " + ", ".join(missing)
    if "heal" in m.get("expect", []) and not m.get("pre_damaged"): s += "; heal not testable (pre-damage missed)"
    return s


def best(attempts):
    return min(attempts, key=lambda a: CLASSES.index(a["class"]))


# ------------------------------------------------------------------ map spots
def find_spots(location=MAP, n=10, seed=0, tries=400):
    """clear spots on the location's map: [(x, y world units of the shooter, [unit direction vectors with a clear corridor])],
    >= SPACING tiles apart; the A / B halves (first / second n/2) sorted by x so the two phases are far apart"""
    from nbpc import diversity as D
    ids, rows = D.map_reference(location); H, W = len(rows), len(rows[0])
    blk = {i for _, (i, r) in D.table("tiles.csv").items() if r.get("BlocksMovement") == "true" or r.get("BlocksProjectiles") == "true"}
    raw = np.array([list(r) for r in rows])
    b = np.isin(ids, list(blk)).reshape(H, W) | np.char.isdigit(raw)   # digits: spawn points and power-cube boxes
    b = np.pad(b, CORR + 2, constant_values=True); o = CORR + 2
    dirs = ((1, 0), (0, 1), (-1, 0), (0, -1))
    cand = []
    for cy in range(H):
        for cx in range(W):
            y, x = cy + o, cx + o
            if b[y - 2:y + 3, x - 2:x + 3].any(): continue
            ok = []
            for dx, dy in dirs:
                xs, ys = sorted((x, x + dx * CORR)), sorted((y, y + dy * CORR))
                if not b[ys[0] - (dx != 0):ys[1] + 1 + (dx != 0), xs[0] - (dy != 0):xs[1] + 1 + (dy != 0)].any(): ok.append((dx, dy))
            if len(ok) >= 2: cand.append((cx, cy, ok))
    rng, top = np.random.default_rng(seed), []
    for t in range(tries):
        order = cand if t == 0 else [cand[i] for i in rng.permutation(len(cand))]
        got = []
        for c in order:
            if all(max(abs(c[0] - g[0]), abs(c[1] - g[1])) >= SPACING for g in got): got.append(c)
            if len(got) == n: break
        if len(got) > len(top): top = got
        if len(top) == n: break
    top = sorted(top, key=lambda c: (c[0], c[1]))
    return [((c[0] + 0.5) * TILE, (c[1] + 0.5) * TILE, c[2]) for c in top]


# ------------------------------------------------------------------ live
def parse_state(buf):
    a = np.frombuffer(buf, "<i4")
    if len(a) < HDR or a[0] != 0x4E423253: return None
    npl, nch, npj, nit = (int(x) for x in a[3:7]); k = HDR
    pl = a[k:k + npl * P2].reshape(npl, P2); k += npl * P2
    ch = a[k:k + nch * C2].reshape(nch, C2); k += nch * C2
    pj = a[k:k + npj * J2].reshape(npj, J2); k += npj * J2
    it = a[k:k + nit * I2].reshape(nit, I2)
    return {"tick": int(a[2]), "pl": pl, "ch": ch, "pj": pj, "it": it}


def brawler_ids(st):
    return {int(p[0]): (int(p[3]) if int(p[3]) not in (0, NIL) else int(p[2])) for p in st["pl"]}


class Job:
    def __init__(self, name, attempt=0, solo=False):
        self.name, self.attempt, self.solo, self.requeues = name, attempt, solo, 0


class Auditor:
    def __init__(self, names, n=1, retry=True, log_dir=None, seed=0, host_args=(), verbose=True):
        from nbpc import diversity as D
        self.D = D; self.info = {b: expected(b) for b in names}; self.retry = retry; self.verbose = verbose
        self.results = {b: [] for b in names}; self.lock = threading.Lock(); self.inflight = 0
        self.queue = collections.deque(Job(b) for b in names); self.seed = 93000000 + seed
        self.spots = find_spots(MAP)
        if len(self.spots) < 2: raise RuntimeError(f"{MAP}: no clear spots")
        self.npair = min(5, len(self.spots) // 2)
        self.log_dir = log_dir or os.path.join(os.environ.get("SIM_WORK", "work"), "super_audit_logs")
        self.n, self.host_args, self.t0 = n, list(host_args), time.time(); self.timing = []

    def log(self, *a):
        if self.verbose: print("[%5.0f s]" % (time.time() - self.t0), *a, flush=True)

    # one attempt's record + the follow-up job
    def record(self, job, m):
        inf = self.info[job.name]; m["expect"] = inf["expect"]
        cls, miss, seen = classify(m, inf["expect"])
        a = {"attempt": job.attempt, "solo": job.solo, "class": cls, "missing": miss, "seen": sorted(seen), "what": describe(m, miss), "m": m}
        with self.lock:
            self.results[job.name].append(a)
            if self.retry and job.attempt == 0 and cls not in ("works", "unverified", "crash"):
                self.queue.append(Job(job.name, 1, solo=job.solo))
        self.log("%-22s try %d %-10s %s" % (job.name, job.attempt, cls, a["what"]))

    def take(self):
        with self.lock:
            if not self.queue: return None
            if self.queue[0].solo: jobs = [self.queue.popleft()]
            else:
                jobs = []
                for j in list(self.queue):
                    if not j.solo and j.name not in {x.name for x in jobs}: jobs.append(j); self.queue.remove(j)
                    if len(jobs) == 2 * self.npair: break
            self.inflight += 1
            return jobs

    def done(self):
        with self.lock: self.inflight -= 1

    def worker(self, idx):
        from nbpc.instance import Instance, InstanceDead
        inst = Instance(idx, tag=f"sa{os.getpid()}", log_dir=self.log_dir, bell=True, extra_args=self.host_args)
        def boot():
            t = time.time(); inst.start(); inst.wait_ready(); self.log(f"instance {idx} ready in {time.time() - t:.0f} s")
        boot()
        try:
            while True:
                jobs = self.take()
                if jobs is None:
                    with self.lock: idle = self.inflight == 0 and not self.queue
                    if idle: return
                    time.sleep(0.5); continue
                try:
                    t = time.time(); self.battle(inst, jobs); self.timing.append((len(jobs), time.time() - t))
                except InstanceDead as e:
                    self.log(f"instance {idx} died: {str(e)[:160]}")
                    self.crashed(jobs)
                    try: inst.close()
                    except Exception: pass
                    boot()
                finally:
                    self.done()
        finally:
            inst.close()

    def crashed(self, jobs):
        """the running phase's shooters (self.cur) go one at a time; a solo crash is final; unrun jobs requeue"""
        undone = [j for j in jobs if not getattr(j, "_done", False)]
        cur = list(getattr(threading.current_thread(), "phase_jobs", ())) or undone   # outside a phase: all suspects
        final = []
        with self.lock:
            for j in undone:
                if j not in cur: self.queue.appendleft(Job(j.name, j.attempt, j.solo))
                elif j.solo: final.append(j)
                else: self.queue.appendleft(Job(j.name, j.attempt, solo=True))
        for j in final: self.record(j, {"crashed": True})

    def battle(self, inst, jobs):
        from nbpc import proto as P
        th = threading.current_thread(); th.phase_jobs = list(jobs)   # a crash in reset: every brawler is a suspect
        A = jobs[:self.npair]; B = jobs[self.npair:2 * self.npair]
        names = [j.name for j in A] + [FILLER] * (self.npair - len(A)) + [j.name for j in B]
        names += [FILLER] * (10 - len(names))
        self.seed += 1
        inst.reset(self.seed, gas=0, flags=P.F_STATE, native_mask=0, obs_mask=0, div=self.D.make_div(location=MAP, chr=names))
        info = inst.battle_info(); got = info["chr"][:10] if info else None
        want = [self.D.cid(x) for x in names]
        inst.step(ticks=10, flags=0, native_mask=0)
        th.phase_jobs = []
        pairs = [(A, list(range(len(A))), [i + 5 for i in range(len(A))], self.spots[:self.npair]),
                 (B, [i + 5 for i in range(len(B))], list(range(len(B))), self.spots[self.npair:2 * self.npair])]
        for js, sh, tg, spots in pairs:
            if not js: continue
            run = []
            for j, s, t in zip(js, sh, tg):
                if got is not None and got[s] != want[s]: j._done = True; self.record(j, {"spawned": False}); continue
                if not inst.resp.pl[s].alive or inst.resp.pl[s].hp <= 0:   # killed as an earlier phase's target: again later
                    with self.lock: self.queue.appendleft(Job(j.name, j.attempt, j.solo))
                    j._done = True; continue
                run.append((j, s, t))
            if run:
                th.phase_jobs = [r[0] for r in run]
                for (j, m) in self.phase(inst, run, spots):
                    j._done = True; self.record(j, m)
                th.phase_jobs = []

    def phase(self, inst, run, spots):
        from nbpc import proto as P
        geo, pos = {}, {}
        for k, (j, s, t) in enumerate(run):
            inf = self.info[j.name]; x, y, dirs = spots[k]
            dx, dy = dirs[min(j.attempt, len(dirs) - 1)]
            d = float(np.clip(DIST_F[min(j.attempt, 1)] * inf["range"], DIST_MIN, DIST_MAX))
            if j.attempt and len(dirs) == 1: d = float(np.clip(d * 0.6, DIST_MIN, DIST_MAX))
            tx, ty = x + dx * d, y + dy * d
            if inf.get("ring") and inf["range"] > inf["ring"]:   # no damage inside InnerRadius (Kaze): land past the target
                d = float(np.clip(inf["range"] - inf["ring"], DIST_MIN, DIST_MAX))
            sd = min(1.0, (d + TILE if inf["charge"] else d + inf.get("ring", 0)) / inf["range"])
            aim = float(np.degrees(np.arctan2(dy, dx)))
            geo[j] = dict(d=round(d), aim=aim, super_dist=round(sd, 3), spot=[round(x), round(y)])
            pos[s] = (int(x), int(y)); pos[t] = (int(tx), int(ty))
        allp = list(range(10)); idle = {i: (-1, 0, 0.0, 0.0) for i in allp}
        stp = lambda acts=None, ticks=STEP: inst.step(actions=acts or idle, ticks=ticks, flags=P.F_STATE, native_mask=0)
        inst.setup(positions=pos, ai_off=allp, unprotect=allp)
        for _ in range(4): stp()
        acts = dict(idle)
        for j, s, t in run: acts[t] = (-1, 1, (geo[j]["aim"] + 180.0) % 360, 0.0)   # pre-damage: the target shoots the shooter once
        stp(acts, 1)
        for _ in range(12): stp()
        inst.setup(super_fill=[s for _, s, _ in run]); r = stp(ticks=1)
        st = parse_state(inst.state_bytes())
        if st is None: raise RuntimeError("no state2 dump in the response (host without F_STATE?)")
        ids0 = set(st["pj"][:, 0].tolist()) | set(st["it"][:, 0].tolist()) | set(st["ch"][:, 0].tolist())
        bid0 = brawler_ids(st)
        M = {}
        for j, s, t in run:
            p, q = r.pl[s], r.pl[t]; ch = st["ch"][st["ch"][:, 0] == bid0.get(s, -1)]
            M[j] = dict(geo[j], spawned=True, filled=bool(p.super_max > 0 and p.super_charge >= p.super_max), super_max=int(p.super_max),
                        sc0=int(p.super_charge), sc_min=int(p.super_charge), hp0=int(p.hp), hp_max=int(p.hp), max_hp=int(p.max_hp),
                        pre_damaged=bool(p.hp < p.max_hp), x0=int(p.x), y0=int(p.y), tx0=int(q.x), ty0=int(q.y), thp0=int(q.hp), thp_min=int(q.hp),
                        move=0, push=0, leaping=False, projectiles={}, summons={}, areas={}, items={},
                        changed=False, data0=int(ch[0, 1]) if len(ch) else NIL, ulti0=int(ch[0, 15]) if len(ch) else NIL,
                        ulti_active=False, foreign_near=False, target_dead=False, s=s, t=t)
        acts = dict(idle)
        for j, s, t in run: acts[s] = (-1, 2, geo[j]["aim"], geo[j]["super_dist"])
        r = stp(acts, 1); n = 1
        while True:
            st = parse_state(inst.state_bytes()); bid = brawler_ids(st) if st is not None else {}
            for j, s, t in run:
                m, p, q = M[j], r.pl[s], r.pl[t]
                m["sc_min"] = min(m["sc_min"], int(p.super_charge)); m["hp_max"] = max(m["hp_max"], int(p.hp))
                m["thp_min"] = min(m["thp_min"], int(q.hp) if q.alive else 0); m["target_dead"] |= bool(not q.alive or q.eliminated)
                m["leaping"] |= bool(p.leaping)
                if p.alive: m["move"] = max(m["move"], int(np.hypot(p.x - m["x0"], p.y - m["y0"])))
                if q.alive and not q.eliminated: m["push"] = max(m["push"], int(np.hypot(q.x - m["tx0"], q.y - m["ty0"])))
                m["dd"] = m.get("dd", 0) + int(p.damage_dealt)   # per step (the host clears it after every response)
                if st is None: continue
                if bid.get(s) != bid0.get(s): m["changed"] = True
                for row in st["pj"]:
                    if row[2] == s and row[0] not in ids0: m["projectiles"][int(row[0])] = data_name(row[1])
                for row in st["it"]:
                    if row[2] == s and row[0] not in ids0:
                        if int(row[1]) // 1000000 == 17: m["areas"][int(row[0])] = {"name": data_name(row[1]), "radius": area_radius(row[1])}
                        else: m["items"][int(row[0])] = data_name(row[1])
                for row in st["ch"]:
                    own = int(row[2]); cid = int(row[0])
                    if row[18] != 0 or int(row[1]) in BOX_IDS or own < 0: continue
                    if cid == bid.get(s):
                        if m["data0"] != NIL and int(row[1]) != m["data0"]: m["changed"] = True
                        if row[15] not in (NIL, m["ulti0"]): m["ulti_active"] = True
                    elif own == s and cid not in bid.values() and cid not in ids0: m["summons"][cid] = data_name(row[1])
                    elif own not in (s, t) and cid not in bid.values():   # someone else's summon near this pair
                        if min(np.hypot(row[3] - m["tx0"], row[4] - m["ty0"]), np.hypot(row[3] - m["x0"], row[4] - m["y0"])) < 2000:
                            m["foreign_near"] = True
            if n >= WATCH: break
            acts = dict(idle)
            for j, s, t in run:   # steered supers (FlyJump, Ww): once cast, the shooter walks to the target (16 joystick directions)
                p, q = r.pl[s], r.pl[t]; inf = self.info[j.name]
                if inf.get("steer") and n * 50 > inf.get("cast", 0) + 100 and q.alive and np.hypot(q.x - p.x, q.y - p.y) > 150:
                    acts[s] = (int(round(np.degrees(np.arctan2(q.y - p.y, q.x - p.x)) / 22.5)) % 16, 0, 0.0, 0.0)
            r = stp(acts); n += STEP
        out = []
        for j, s, t in run:
            m = M[j]
            m["consumed"] = bool(m["filled"] and m["sc_min"] < m["super_max"] / 2)
            m["damage"] = max(m["thp0"] - m["thp_min"], m.get("dd", 0))
            m["heal"] = m["hp_max"] - m["hp0"]
            for k in ("projectiles", "summons", "items"): m[k] = sorted(m[k].values())
            m["areas"] = sorted(m["areas"].values(), key=lambda a: a["name"])
            out.append((j, m))
        return out

    def run(self):
        ths = [threading.Thread(target=self.worker, args=(i,), daemon=True) for i in range(self.n)]
        for i, t in enumerate(ths):
            t.start()
            if i + 1 < self.n: time.sleep(2)
        for t in ths: t.join()
        return self.results


# ------------------------------------------------------------------ output
def summary(names, results, info):
    out = []
    for b in names:
        at = results.get(b) or []
        inf = info[b]
        row = {"name": b, "display": display(b), "skill": inf["skill"], "type": inf["type"], "kind": inf["kind"],
               "expect": inf["expect"], "notes": inf["notes"], "attempts": at}
        if at:
            bb = best(at); row.update({"class": bb["class"], "what": bb["what"], "missing": bb["missing"]})
            if any(a["class"] == "crash" for a in at) and bb["class"] != "crash": row["what"] += " (one attempt crashed the host)"
        else:
            row.update({"class": "not_run", "what": "not run", "missing": []})
        out.append(row)
    return out


def markdown(rows):
    L = ["# Super audit", "", "One battle per 10 brawlers on %s, native AI off: each brawler fires its super at one enemy %s-%d units "
         "away and is watched for %.1f s (`python -m super_audit`). Expected effects come from the game's tables (skills.csv "
         "UltimateSkill + its projectile / area rows). Classes: works (every expected effect seen), partial (some), unverified "
         "(effect not visible to the audit), no_effect, not_fired (charge not consumed), crash. Not visible: stuns, slows, "
         "shields, buffs, wall breaking, ally effects." % (MAP, DIST_MIN, DIST_MAX, WATCH / 20.0), ""]
    cnt = collections.Counter(r["class"] for r in rows)
    L += ["Totals: " + ", ".join("%s %d" % (c, cnt[c]) for c in CLASSES + ("not_run",) if cnt[c]), ""]
    for c in CLASSES + ("not_run",):
        rs = [r for r in rows if r["class"] == c]
        if not rs: continue
        L += ["## %s (%d)" % (c, len(rs)), ""]
        for kind in sorted({r["kind"] for r in rs}):
            L += ["### %s" % kind, ""]
            for r in sorted((r for r in rs if r["kind"] == kind), key=lambda r: r["display"]):
                box = "x" if c == "works" else " "
                tail = "" if c == "works" else ": " + r["what"]
                if c == "works" and r["what"] != "nothing seen": tail = " (" + r["what"] + ")"
                L.append("- [%s] %s (`%s`, %s): %s%s" % (box, r["display"], r["name"], r["kind"], c, tail))
            L.append("")
    return "\n".join(L)


# ------------------------------------------------------------------ self-check
def _check():
    # classification on synthetic measurements
    base = dict(spawned=True, filled=True, consumed=True, damage=0, move=0, push=0, heal=0, pre_damaged=True, projectiles=[],
                summons=[], areas=[], items=[], changed=False, leaping=False, foreign_near=False)
    m = lambda **k: dict(base, **k)
    assert classify(m(), ["damage"])[0] == "no_effect"
    assert classify(m(consumed=False, damage=500), ["damage"])[0] == "not_fired"
    assert classify(m(spawned=False), ["damage"])[0] == "not_fired"
    assert classify({"crashed": True}, ["damage"])[0] == "crash"
    assert classify(m(damage=900, projectiles=["P"]), ["projectile", "damage"])[0] == "works"
    c, miss, seen = classify(m(projectiles=["P"]), ["projectile", "damage"])
    assert c == "partial" and miss == ["damage"] and seen == {"projectile"}
    c, miss, _ = classify(m(move=2100, damage=1960), ["move", "damage", "push", "area"])   # Bull: env charge, no pushback
    assert c == "partial" and miss == ["push", "area"]
    assert classify(m(move=MOVE_MIN - 1), ["move"])[0] == "no_effect" and classify(m(leaping=True), ["move"])[0] == "works"
    assert classify(m(push=PUSH_MIN), ["push"])[0] == "works" and classify(m(push=PUSH_MIN - 1), ["push"])[0] == "no_effect"
    assert classify(m(summons=["ShamanPet"], projectiles=["X"]), ["projectile", "summon"])[0] == "works"
    assert classify(m(areas=[{"name": "A", "radius": 300}]), ["area"])[0] == "works"
    assert classify(m(items=["BoxOfBombs"], move=2000), ["move", "item"])[0] == "works"
    assert classify(m(changed=True), ["change"])[0] == "works"
    assert classify(m(), [])[0] == "unverified" and classify(m(damage=5), [])[0] == "unverified"
    # heal: only checked when the pre-damage landed
    assert classify(m(heal=HEAL_MIN, damage=10, projectiles=["P"]), ["projectile", "damage", "heal"])[0] == "works"
    assert classify(m(damage=10, projectiles=["P"]), ["projectile", "damage", "heal"])[0] == "partial"
    assert classify(m(pre_damaged=False, damage=10, projectiles=["P"]), ["projectile", "damage", "heal"])[0] == "works"
    assert classify(m(pre_damaged=False), ["heal"])[0] == "unverified"
    # damage next to a foreign summon is not counted
    assert classify(m(damage=800, foreign_near=True), ["damage"])[0] == "no_effect"
    # best attempt, descriptions, checklist
    at = [{"class": "no_effect"}, {"class": "works"}]
    assert best(at)["class"] == "works" and best([{"class": "crash"}, {"class": "not_fired"}])["class"] == "not_fired"
    d = describe(m(damage=1960, move=2100, areas=[{"name": "BullSlowArea", "radius": 300}], expect=["move"]), ["push"])
    assert "1960 damage" in d and "moved 2100" in d and "BullSlowArea r300" in d and d.endswith("missing: push"), d
    assert describe(m(consumed=False)) == "super charge not consumed" and "not testable" in describe(m(pre_damaged=False, expect=["heal"]))
    rows = [{"name": "BullDude", "display": "Bull", "kind": "Charge 1", "class": "partial", "what": d},
            {"name": "Shaman", "display": "Nita", "kind": "Attack: summon", "class": "works", "what": "summons: ShamanPet"},
            {"name": "Rosa", "display": "Rosa", "kind": "Shield", "class": "unverified", "what": "nothing seen"}]
    md = markdown(rows)
    assert "- [ ] Bull (`BullDude`, Charge 1): partial: " in md and "- [x] Nita (`Shaman`, Attack: summon): works (summons: ShamanPet)" in md
    assert md.index("## works") < md.index("## partial") < md.index("## unverified") and "Totals: works 1, partial 1, unverified 1" in md
    n_tab = 0
    try:                                                     # the tables (game files from setup.sh), when present
        import env_v5
        hs = env_v5.heroes(); _tables()
    except Exception as e:
        print("super_audit: game tables not available (%s): table checks skipped" % type(e).__name__)
    else:
        inf = {h: expected(h) for h in hs}; n_tab = len(hs)
        b = inf["BullDude"]
        assert b["type"] == "Charge" and b["kind"] == "Charge 1" and {"move", "damage", "push"} <= set(b["expect"]) and b["range"] == 3300, b
        assert {"projectile", "summon"} <= set(inf["Shaman"]["expect"]) and "damage" not in inf["Shaman"]["expect"]
        assert inf["Sniper"]["expect"] == ["move", "item"], inf["Sniper"]            # leap: bombs at the start, no target damage
        assert {"projectile", "area", "damage"} <= set(inf["TntDude"]["expect"])
        assert "heal" in inf["DeadMariachi"]["expect"] and inf["Rosa"]["expect"] == [] and "change" in inf["MechaDude"]["expect"]
        assert all(i["range"] > 0 and i["skill"] for i in inf.values())
        assert display("BullDude") == "Bull" and data_name(16000000 + list(_tables()["char"]).index("BullDude")) == "BullDude"
        sp = find_spots(MAP)
        assert len(sp) == 10 and all(len(d) >= 2 for _, _, d in sp), len(sp)
        assert min(max(abs(a[0] - c[0]), abs(a[1] - c[1])) for i, a in enumerate(sp) for c in sp[i + 1:]) >= SPACING * TILE
    print("super_audit ok: classification%s" % (", tables of %d brawlers, %d spots on %s" % (n_tab, 10, MAP) if n_tab else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="self-check of the classification (no host)")
    ap.add_argument("--n", type=int, default=1, help="game host instances")
    ap.add_argument("--only", default="", help="comma-separated brawler names (characters.csv Name); merged into an existing json")
    ap.add_argument("--no-retry", action="store_true", help="no second attempt for non-works brawlers")
    ap.add_argument("--out", default=os.environ.get("SIM_WORK", "work"), help="folder of super_audit.json / .md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--watch", type=int, default=WATCH, help="ticks watched after the fire (default %d)" % WATCH)
    ap.add_argument("--host-args", default="", help="extra game host arguments, e.g. '--native-charge 0'")
    a = ap.parse_args()
    if a.check: return _check()
    globals()["WATCH"] = a.watch
    import env_v5
    roster = env_v5.heroes()
    names = [x for x in a.only.split(",") if x] or roster
    bad = [x for x in names if x not in _tables()["char"]]
    if bad: sys.exit(f"unknown brawlers: {bad}")
    au = Auditor(names, n=a.n, retry=not a.no_retry, seed=a.seed, host_args=a.host_args.split())
    au.log(f"{len(names)} brawlers, {a.n} instances, {au.npair} pairs per phase on {MAP}")
    res = au.run()
    rows = summary(names, res, au.info)
    os.makedirs(a.out, exist_ok=True); jp = os.path.join(a.out, "super_audit.json")
    if a.only and os.path.exists(jp):
        old = {r["name"]: r for r in json.load(open(jp)).get("brawlers", [])}
        old.update({r["name"]: r for r in rows}); rows = [old[k] for k in sorted(old)]
    bt = [t / max(k, 1) for k, t in au.timing]
    meta = {"map": MAP, "watch_ticks": WATCH, "instances": a.n, "seconds": round(time.time() - au.t0),
            "battles": len(au.timing), "battle_s": round(float(sum(t for _, t in au.timing)), 2),
            "battle_s_per_brawler": round(float(np.mean(bt)), 3) if bt else None, "host_args": a.host_args, "time": time.strftime("%F %T")}
    json.dump({"meta": meta, "brawlers": rows}, open(jp, "w"), indent=1)
    md = markdown(rows); open(os.path.join(a.out, "super_audit.md"), "w").write(md + "\n")
    print(md)
    au.log(f"wrote {jp} and super_audit.md; {meta['battles']} battles in {meta['battle_s']} s ({meta['battle_s_per_brawler']} s per brawler-attempt)")


if __name__ == "__main__":
    main()
