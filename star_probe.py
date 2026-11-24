"""Which star powers does the game read? (issue #20) One boot with the probe host hooks on (NB_CARDPROBE=1,
native/loader/cardprobe.c), Showdown battles of 10 native-AI players that each carry a star power card (div.star), every
card of every playable brawler once per round (env_v5.star_cards: 212 cards in 22 battles), --repeat rounds with new seeds.

libg reads a star power through getters that take an ability type (the card's +0x100 key) and return the card's Value
columns, or -1 / NULL when the player's config holds no card of that type. The probe logs each (player, getter, type)
query and the card each player's config holds after the build. Per card, the best over all rounds:
  READ          the game asked its owner for the card's type and got it: the star power's code exists and ran
  READ_OTHERS   never asked its owner, but the game asked other players for that type: the reader exists, the owner never
                reached it (the situation did not come up, or the reader runs for the victim, not the card's owner)
  NOT_SEEN      the card arrived but nobody was ever asked for its type: its reader is compiled out of the client build,
                or no battle reached the situation that reads it (low hp, a bush, a pet, a super, ...)
  GATED         the card arrived but player+0x238 ("cards off") was set: every getter answers nothing
  NOT_ARRIVED   the host accepted the request but the config the getters read holds another card (or none)
  REJECTED      the host rejected the request (div_err)
Also: every queried type with its getters, call sites and whether any player got an answer.

Output: $SIM_WORK/star_probe.json ({"meta", "cards", "types"}) and star_probe.md (also stdout). Host log in
$SIM_WORK/star_probe_logs.
python -m star_probe [--repeat 2] [--battles N] [--ticks 6000] [--only A,B]      python -m star_probe --check   (no game)"""
import argparse, collections, json, os, re, sys, time

MAP = "Survival651"   # the most open Showdown map (super_audit's): players meet early
STEP = 40             # ticks per env step: the probe only reads the log, so big steps
TICKS = 6000          # tick limit per battle; native-AI battles with the storm end after 1300..3100 ticks
SLOTS = 10
FORCE = ("Sniper",)   # Piper first: ambush (has a reader per the issue) and reload_on_hit (has none) in battles 1 and 2
VERDICTS = ("READ", "READ_OTHERS", "NOT_SEEN", "GATED", "NOT_ARRIVED", "REJECTED")   # best first (rounds combine to the best)

HEX = r"0x[0-9a-f]+|0"
Q = re.compile(rf"\[cardprobe\] q tick (-?\d+) fn (\w+) slot (-?\d+) type ({HEX}) idx (-?\d+) -> (-?\d+) off238 (-?\d+) "
               rf"card (-?\d+) card_t ({HEX}) lr ({HEX}) pl (\S+) cfg (\S+)")
BUILD = re.compile(rf"\[cardprobe\] build slot (\d+) off238 (\d+) cfg348 (\d) card (-?\d+) card_t ({HEX}) meta (-?\d+) n50 (-?\d+) "
                   rf"pl (\S+) cfg (\S+)")
BUILD_SIM = re.compile(r"\[cardprobe\] build sim134 (\d+) list184 n (\d+) has35 (\d)")
CARD_FNS = ("c42838", "c34be0")   # return a card (logged as 1 / 0); the others return a Value column or -1 (0 is a value)


def _i32(s):
    """a %#x field -> signed 32-bit int (the host prints card_t -1 as 0xffffffff)"""
    v = int(s, 16) if s.startswith("0x") else int(s)
    return v - (1 << 32) if v >= 1 << 31 else v


def hit(q): return q["r"] != 0 if q["fn"] in CARD_FNS else q["r"] != -1


def plan(cards, battles=None, force=FORCE):
    """cards = {brawler: [card ids]} -> battles [[(brawler, card)] * <= SLOTS], each brawler at most once per battle. Every
    card once, in as few battles as the brawler limit allows (a brawler has 2..3 cards, so >= that many battles), forced
    brawlers' cards first; battles = None: all of them, else the first `battles`."""
    left = {b: list(v) for b, v in sorted(cards.items())}
    order = [b for b in force if b in left] + [b for b in left if b not in force]
    out = []
    while any(left.values()):
        bt, used = [], set()
        for b in sorted(order, key=lambda b: (b not in force, -len(left[b]), order.index(b))):   # most cards left first
            if left[b] and b not in used and len(bt) < SLOTS: bt.append((b, left[b].pop(0))); used.add(b)
        out.append(bt)
    return out if battles is None else out[:battles]


def parse(lines):
    """host log lines of one battle -> {"sim": {...}, "build": {slot: {...}}, "q": [{...}]}; a query made while the battle
    was built (slot -1) gets the slot whose build line has its player or config pointer"""
    out = {"sim": None, "build": {}, "q": []}
    for ln in lines:
        m = BUILD_SIM.search(ln)
        if m: out["sim"] = dict(cards_flag=int(m[1]), list184=int(m[2]), has35=int(m[3])); continue
        m = BUILD.search(ln)
        if m:
            out["build"][int(m[1])] = dict(off238=int(m[2]), cfg348=int(m[3]), card=int(m[4]), card_t=_i32(m[5]), meta=int(m[6]),
                                           n50=int(m[7]), pl=m[8], cfg=m[9])
            continue
        m = Q.search(ln)
        if m:
            out["q"].append(dict(tick=int(m[1]), fn=m[2], slot=int(m[3]), type=_i32(m[4]), idx=int(m[5]), r=int(m[6]),
                                 off238=int(m[7]), card=int(m[8]), card_t=_i32(m[9]), lr=m[10], pl=m[11], cfg=m[12]))
    by = {}
    for s, b in out["build"].items():
        if b["pl"] != "(nil)": by[b["pl"]] = s
        if b["cfg"] != "(nil)": by.setdefault(b["cfg"], s)
    for q in out["q"]:
        if q["slot"] < 0: q["slot"] = by.get(q["pl"], by.get(q["cfg"], -1))
    return out


def judge(battle, info, parsed):
    """battle [(brawler, card)], info = Instance.battle_info(), parsed = parse() -> per slot a row with this battle's verdict
    (READ / NOT_SEEN / GATED / NOT_ARRIVED / REJECTED; READ_OTHERS is decided over all battles in combine)"""
    rows = []
    err, star = info["err"], info["star_mask"]
    for s, (b, card) in enumerate(battle):
        bd = parsed["build"].get(s, {})
        mine = [q for q in parsed["q"] if q["slot"] == s and bd and q["type"] == bd.get("card_t")]
        row = dict(brawler=b, card=card, slot=s, card_t=bd.get("card_t", -1), meta=bd.get("meta", -1), off238=bd.get("off238", -1),
                   hits=[list(h) for h in sorted({(q["fn"], q["lr"]) for q in mine if hit(q)})],
                   first_tick=min((q["tick"] for q in mine if hit(q)), default=None))
        if err >> s & 1: v = "REJECTED"
        elif not (star >> s & 1) or bd.get("card") != card: v = "NOT_ARRIVED"
        elif bd.get("off238"): v = "GATED"
        elif row["hits"]: v = "READ"
        else: v = "NOT_SEEN"
        row["verdict"] = v; rows.append(row)
    return rows


def type_table(parsed_all):
    """all battles' queries -> {type: {"fns": [...], "sites": [...], "hit": bool}}"""
    t = collections.defaultdict(lambda: {"fns": set(), "sites": set(), "hit": False})
    for p in parsed_all:
        for q in p["q"]:
            e = t[q["type"]]; e["fns"].add(q["fn"]); e["sites"].add(q["lr"]); e["hit"] |= hit(q)
    return {k: {"fns": sorted(v["fns"]), "sites": sorted(v["sites"]), "hit": v["hit"]} for k, v in sorted(t.items())}


def combine(rows, types):
    """per battle rows (a card once per round) + type_table -> one row per card: its best verdict over the rounds, NOT_SEEN
    turned into READ_OTHERS when its type was queried for some player"""
    best = {}
    for r in rows:
        v = "READ_OTHERS" if r["verdict"] == "NOT_SEEN" and r["card_t"] in types else r["verdict"]
        o = best.get(r["card"])
        if o is None or VERDICTS.index(v) < VERDICTS.index(o["verdict"]):
            best[r["card"]] = {**r, "verdict": v, "rounds": (o or {}).get("rounds", 0) + 1}
        else: o["rounds"] += 1
    return sorted(best.values(), key=lambda r: (r["brawler"], r["card"]))


def card_rows():
    """card id -> (name, Type) from cards.csv (ids as env_v5.star_cards)"""
    from nbpc import diversity as D
    out, k = {}, 0
    for r in D._rows("cards.csv"):
        if not r[next(iter(r))]: continue
        out[23000000 + k] = (r[next(iter(r))], r.get("Type", "")); k += 1
    return out


def markdown(cards, types, meta):
    c = collections.Counter(r["verdict"] for r in cards)
    out = [f"# Star power probe ({meta['rounds']} x {meta['battles']} battles on {meta['map']}, native AI, {time.strftime('%Y-%m-%d')})", "",
           "Cards: " + ", ".join(f"{k} {c[k]}" for k in VERDICTS if c[k]), "",
           f"Queried types: {len(types)} ({sum(v['hit'] for v in types.values())} answered for some player).", ""]
    for v in VERDICTS:
        rs = [r for r in cards if r["verdict"] == v]
        if not rs: continue
        out += [f"## {v} ({len(rs)})", ""]
        if v == "NOT_SEEN":
            out += [", ".join(f"{r['brawler']} {r['type']}" for r in rs), ""]; continue
        out += ["| brawler | card | type | card_t | first read (tick) | call sites |", "|---|---|---|---|---|---|"]
        for r in rs:
            sites = sorted({h[1] for h in r["hits"]}) if r["hits"] else types.get(r["card_t"], {}).get("sites", [])
            out.append(f"| {r['brawler']} | {r['name']} | {r['type']} | {r['card_t']:#x} | {'' if r['first_tick'] is None else r['first_tick']} "
                       f"| {' '.join(sites)} |")
        out.append("")
    known = {r["card_t"] for r in cards}
    other = {k: v for k, v in types.items() if k not in known}
    if other:
        out += ["## Queried types no card in the probe has", "", ", ".join(f"{k:#x} ({' '.join(v['sites'])})" for k, v in other.items()), ""]
    return "\n".join(out)


def run(a):
    import env_v5 as E
    from nbpc import diversity as D
    from nbpc.instance import Instance
    sc = E.star_cards()
    if a.only: sc = {b: v for b, v in sc.items() if b in a.only.split(",")}
    battles = plan(sc, a.battles)
    names = card_rows()
    os.environ["NB_CARDPROBE"] = "1"   # the instance passes the environment on to the host
    inst = Instance(0, tag=f"sp{os.getpid()}", log_dir=os.path.join(a.out, "star_probe_logs"), bell=True, extra_args=a.host_args.split())
    t = time.time(); inst.start(); inst.wait_ready(timeout=a.boot_timeout); print(f"instance ready in {time.time() - t:.0f} s", flush=True)
    rows, parsed_all, sims = [], [], []
    try:
        for rd in range(a.repeat):
            for k, bt in enumerate(battles):
                div = D.make_div(location=MAP, chr=[b for b, _ in bt])
                div["star"] = [c for _, c in bt]
                off = os.path.getsize(inst.log_path)
                inst.reset(a.seed + 1000 * rd + k, gas=1, flags=0, native_mask=0xffffffff, obs_mask=0, div=div)
                info = inst.battle_info()
                for _ in range(a.ticks // STEP):
                    if inst.step(ticks=STEP, flags=0, native_mask=0xffffffff).done: break
                with open(inst.log_path, "rb") as f: f.seek(off); lines = f.read().decode("utf-8", "replace").splitlines()
                p = parse(lines)
                if info is None or p["sim"] is None:   # no data is not a verdict
                    raise SystemExit(f"battle {k + 1}: " + ("the host publishes no battle info" if info is None else
                                     "no [cardprobe] build line (host built without cardprobe.c?)") + f"; log {inst.log_path}")
                parsed_all.append(p); sims.append(p["sim"])
                got = judge(bt, info, p)
                for row in got: row["name"], row["type"] = names.get(row["card"], ("?", "?")); row["round"] = rd
                rows += got
                c = collections.Counter(r["verdict"] for r in got)
                print(f"round {rd + 1} battle {k + 1}/{len(battles)} ended at tick {inst.resp.tick}: "
                      + " ".join(f"{v} {n}" for v, n in sorted(c.items())) + f"  (sim {p['sim']})", flush=True)
    finally:
        inst.close()
    types = type_table(parsed_all)
    cards = combine(rows, types)
    meta = dict(rounds=a.repeat, battles=len(battles), ticks=a.ticks, map=MAP, seed=a.seed, sims=sims)
    with open(os.path.join(a.out, "star_probe.json"), "w") as f:
        json.dump({"meta": meta, "cards": cards, "battles": rows, "types": {f"{k:#x}": v for k, v in types.items()}}, f, indent=1)
    md = markdown(cards, types, meta)
    with open(os.path.join(a.out, "star_probe.md"), "w") as f: f.write(md + "\n")
    print(md)


def _check():
    cards = {"Sniper": [91, 152], "Colt": [1, 2], "Bull": [3, 4], "Lightyear": [5, 6, 7]}
    bs = plan(cards)
    assert sorted(c for bt in bs for _, c in bt) == sorted(c for v in cards.values() for c in v)            # every card once
    assert all(len({b for b, _ in bt}) == len(bt) <= SLOTS for bt in bs) and len(bs) == 3                  # no brawler twice
    assert bs[0][0] == ("Sniper", 91) and bs[1][0] == ("Sniper", 152) and plan(cards) == bs and len(plan(cards, 1)) == 1
    big = {f"B{i}": [i * 10, i * 10 + 1] for i in range(25)}
    assert len(plan(big)) == 5 and all(len(bt) == 10 for bt in plan(big)[:4])                                # 50 cards / 10
    log = ["[cardprobe] q tick -1 fn c42874 slot -1 type 0x7b idx 1 -> 40 off238 0 card 152 card_t 0x7b lr 0xc28100 pl 0xb00 cfg 0xc00",
           "[cardprobe] q tick -1 fn c34c6c slot -1 type 0x10 idx 1 -> 0 off238 -1 card 1 card_t 0x10 lr 0xc28144 pl (nil) cfg 0xc02",
           "[cardprobe] build sim134 1 list184 n 0 has35 0",
           "[cardprobe] build slot 0 off238 0 cfg348 0 card 91 card_t 0x2a meta 4 n50 0 pl 0xa00 cfg 0xa02",
           "[cardprobe] build slot 1 off238 0 cfg348 0 card 152 card_t 0x7b meta 4 n50 0 pl 0xb00 cfg 0xc00",
           "[cardprobe] build slot 2 off238 1 cfg348 0 card 1 card_t 0x10 meta 4 n50 0 pl 0xb02 cfg 0xc02",
           "[cardprobe] build slot 3 off238 0 cfg348 0 card 0 card_t 0xffffffff meta -1 n50 0 pl 0xb03 cfg (nil)",
           "[cardprobe] build slot 4 off238 0 cfg348 0 card 4 card_t 0x55 meta 4 n50 0 pl 0xb04 cfg 0xc04",
           "[cardprobe] q tick 50 fn c42874 slot 0 type 0x2a idx 1 -> 0 off238 0 card 91 card_t 0x2a lr 0xb3e958 pl 0xa00 cfg 0xa02",
           "[cardprobe] q tick 51 fn c42874 slot 4 type 0x2a idx 1 -> -1 off238 0 card 4 card_t 0x55 lr 0xb3e958 pl 0xb04 cfg 0xc04",
           "[cardprobe] q tick 52 fn c42838 slot 4 type 0x77 idx 0 -> 0 off238 0 card 4 card_t 0x55 lr 0xb3e000 pl 0xb04 cfg 0xc04",
           "[cardprobe] unrelated line"]
    p = parse(log)
    assert p["sim"] == dict(cards_flag=1, list184=0, has35=0) and len(p["build"]) == 5 and len(p["q"]) == 5
    assert p["build"][3]["card_t"] == -1 and [q["slot"] for q in p["q"]] == [1, 2, 0, 4, 4]                # build-time queries matched
    bt = [("Sniper", 91), ("Sniper", 152), ("Colt", 1), ("Bull", 3), ("Bull", 4), ("Colt", 2)]
    info = dict(err=1 << 5, star_mask=0b011111 & ~(1 << 3))
    got = judge(bt, info, p)
    assert [r["verdict"] for r in got] == ["READ", "READ", "GATED", "NOT_ARRIVED", "NOT_SEEN", "REJECTED"], [r["verdict"] for r in got]
    assert got[0]["hits"] == [["c42874", "0xb3e958"]] and got[0]["first_tick"] == 50 and got[1]["first_tick"] == -1   # Value 0 is a read
    t = type_table([p])
    assert t[0x2a]["hit"] and t[0x10]["hit"] and not t[0x77]["hit"] and t[0x2a]["sites"] == ["0xb3e958"]       # c34c6c 0 = a value
    later = [dict(r, verdict="READ") if r["card"] == 4 else r for r in got]                                    # a second round
    cb = {r["card"]: r for r in combine(got + later, t)}
    assert cb[4]["verdict"] == "READ" and cb[4]["rounds"] == 2 and cb[91]["verdict"] == "READ"
    cb = {r["card"]: r for r in combine(got, {0x55: {}})}
    assert cb[4]["verdict"] == "READ_OTHERS" and cb[2]["verdict"] == "REJECTED"                                  # type asked of others
    md = markdown([dict(r, name="n", type="t") for r in combine(got, t)], t, dict(rounds=1, battles=1, map=MAP))
    assert "## READ (2)" in md and "0x77" in md, md
    try:   # with the game files: the real card list plans into 22 battles with Piper first
        import env_v5 as E
        sc = E.star_cards(); bs = plan(sc)
        assert sum(map(len, sc.values())) == sum(map(len, bs)) and bs[0][0][0] == "Sniper" and len(bs) <= 23, (len(bs), sum(map(len, sc.values())))
        names = card_rows(); assert names[23000091][1] == "ambush" and names[23000152][1] == "reload_on_hit", (names[23000091], names[23000152])
        extra = f", {sum(map(len, sc.values()))} cards in {len(bs)} battles"
    except (ImportError, OSError) as e:
        extra = f" (game tables skipped: {e.__class__.__name__})"
    print("star_probe self-check ok" + extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="self-check without the game host")
    ap.add_argument("--repeat", type=int, default=2, help="rounds of the whole plan, new seeds each (a card's best verdict counts)")
    ap.add_argument("--battles", type=int, default=None, help="only the first N battles of the plan (default all)")
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated brawlers (characters.csv names)")
    ap.add_argument("--out", default=os.environ.get("SIM_WORK", "work"), help="folder of star_probe.json / .md")
    ap.add_argument("--host-args", default="", help="extra game host arguments")
    ap.add_argument("--boot-timeout", type=float, default=900, help="seconds to wait for the boot (60-300 s, slower on a busy machine)")
    a = ap.parse_args()
    if a.check: return _check()
    os.makedirs(a.out, exist_ok=True); run(a)


if __name__ == "__main__":
    sys.exit(main())
