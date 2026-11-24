"""Star power probe (issue #20, step 1): which star power cards the shipped libg v68.279 still reads in battle.

Every star power read in libg goes through three loadout-config lookups (engine/native/loader/star.c). With NB_STAR_PROBE=1
the host counts the calls per card Type enum and whether the config had a card of that type ("hit"), and logs the counts
per battle. This module boots one host with the probe on, plays battles of 10 native-AI players where every player carries
one star power card not yet probed (cards.csv MetaType 4, Target = its brawler), until every card has been carried
--rounds times, then reads the host log and puts each card in one class:
  read     its type was asked for on a config that holds it: a live reader in this build took the card's values
  asked    its type was asked for, but never on the carrier's config (the event it waits for did not happen to the carrier)
  generic  never asked for, but the row lists Traits or Components (attached at spawn, 0xc28600: no lookup needed)
  none     never asked for: no reader left in this build (likely compiled out with the server code)
  missing  the card did not reach the player (the host rejected it, or the player list never had it)
  skipped  its brawler is not in the battle pool (env_v5.heroes: a form, or skip.json), so it was not carried
Each card also gets "site", from a static scan of libg (lookup_sites): "sim" = a call with the card's type in the battle
logic, "ui" = only in client code (it never runs in the sim-only battles training plays), "nosite" = no call at all (the
reader is gone from this build).
LIVE is the list of cards found "read" by the last probe (python -m star_audit, 2026-11-24): what
env_v5.with_star_powers hands out. "asked" cards have a reader but its effect is unproven; they stay out until a per-card
A/B check (issue #20 "How to verify") shows them working.

python -m star_audit [--rounds 2] [--ticks 1600]      (one boot, about 25 battles; writes runs/star_audit/probe.json)
python -m star_audit --check                           (no game: the log parser and the classes on a made-up log)"""
import argparse, collections, json, os, re, sys, time

TICKS = 1600   # per battle (80 s at 20 ticks/s): most native-AI showdowns have fights, supers and low-hp moments by then
STEP = 40      # ticks per env step: the native AI plays every player, nothing to decide per tick

# card names (cards.csv Name) the 2026-11-24 probes found "read" (all 212 cards x 3 battles of 3000 ticks, then the 27
# "asked" / sim-site cards x 10 battles of 4000 ticks; filled in by hand from runs/star_audit/probe*.json)
LIVE = frozenset({
    "Alternator_unique_2", "Blower_unique", "BowDude_unique_2", "Duelist_unique", "Duelist_unique_2", "FutureGirl_unique_2",
    "JetpackGirl_unique", "Leaper_unique_2", "Maisie_unique", "Mechanic_unique_2", "Meeple_unique", "Samurai_unique",
    "Speedy_unique_2", "Voodoo_unique", "Voodoo_unique_2"})

# the lookups (star.c) and the wrappers that tail-call them with the type in w1: a call site with a constant type is a reader
LOOKUPS = (0xc34be0, 0xc34c6c, 0xc34d10, 0xc42838, 0xc42874, 0xc428b0, 0xb6a75c, 0xb6a7b0)
TEXT_END = 0x11a68e0   # end of libg's first (code) LOAD segment: file offset = address there
SIM_CODE = 0xa00000    # call sites below this are client / UI code (menus, aim indicators); the battle logic is above

Q_RE = re.compile(r"\[star\] q battle=(\d+) type=(\d+) n=(\d+) hit=(\d+) lr=0x([0-9a-f]+)")
P_RE = re.compile(r"\[star\] p battle=(\d+) slot=(\d+) card=(-?\d+) type=(-?\d+) off=(\d+)")


def star_rows():
    """cards.csv star power rows (MetaType 4, Target set, not Disabled) -> [(data id, name, brawler, Type, generic)]; the
    data id counts the named rows (as env_v5.gadget_cards)"""
    from nbpc import diversity as D
    import env_v5 as E
    out, k = [], 0
    for r in D._rows("cards.csv"):
        if not r[next(iter(r))]: continue
        if r.get("MetaType") == "4" and r.get("Target") and not E._off(r):
            out.append((E.T_CARD * 1000000 + k, r["Name"], r["Target"], r.get("Type", ""), bool(r.get("Traits") or r.get("Components"))))
        k += 1
    return out


def lookup_sites(libg):
    """libg.so -> {type: [call site, ...]}: every BL / B to a lookup with a constant type (movz w1, #type within the 9
    instructions before it, and w1 not written in between). A type without a site has no reader in this build."""
    import numpy as np
    w = np.fromfile(libg, dtype="<u4", count=TEXT_END // 4); ix = np.arange(len(w), dtype=np.int64)
    br = ((w & 0xfc000000) == 0x94000000) | ((w & 0xfc000000) == 0x14000000)
    imm = (w & 0x03ffffff).astype(np.int64); imm[imm >= 1 << 25] -= 1 << 26; tgt = (ix + imm) * 4
    out = {}
    for k in np.nonzero(br & np.isin(tgt, LOOKUPS))[0]:
        for j in range(k - 1, max(k - 10, 0), -1):
            if (w[j] & 0xffe0001f) == 0x52800001: out.setdefault(int((w[j] >> 5) & 0xffff), []).append(int(k * 4)); break
            if (w[j] & 0x1f) == 1 and (w[j] & 0x7f000000) in (0x2a000000, 0x11000000): break   # w1 = another register
    return out


def site_kind(sites):
    """a type's call sites -> "sim" (battle logic), "ui" (client code only) or "nosite" """
    return "sim" if any(a >= SIM_CODE for a in sites) else "ui" if sites else "nosite"


def parse_log(lines):
    """host log lines -> (carried {battle: {slot: (card, type, off)}}, asked {battle: {type: (n, hit, lr)}})"""
    carried, asked = collections.defaultdict(dict), collections.defaultdict(dict)
    for ln in lines:
        m = P_RE.search(ln)
        if m:
            b, s, c, t, off = map(int, m.groups()); carried[b][s] = (c, t, off); continue
        m = Q_RE.search(ln)
        if m:
            b, t, n, h = map(int, m.groups()[:4]); asked[b][t] = (n, h, int(m.group(5), 16))
    return carried, asked


def classify(rows, carried, asked):
    """the classes above, per card id: {id: {"cls", "type", "n", "hit", "lr", "battles"}}"""
    res = {cid: {"cls": "missing", "type": None, "n": 0, "hit": 0, "lr": 0, "battles": 0} for cid, *_ in rows}
    gen = {cid: g for cid, _, _, _, g in rows}
    for b, slots in carried.items():
        for s, (c, t, off) in slots.items():
            if c not in res or t < 0: continue
            r = res[c]; r["type"] = t; r["battles"] += 1; r["off"] = max(r.get("off", 0), off)
            n, h, lr = asked.get(b, {}).get(t, (0, 0, 0))
            r["n"] += n; r["hit"] += h; r["lr"] = r["lr"] or lr
    for c, r in res.items():
        if r["type"] is None: continue
        r["cls"] = "read" if r["hit"] else "asked" if r["n"] else "generic" if gen[c] else "none"
    return res


def plan(rows, rounds, per=10, seed=0):
    """battles of `per` players: each card carried `rounds` times, a battle never carries the same card twice"""
    import random
    rng = random.Random(seed); todo = [c for c in rows for _ in range(rounds)]; rng.shuffle(todo); out = []
    while todo:
        b, seen = [], set()
        for c in list(todo):
            if c[0] in seen: continue
            b.append(c); seen.add(c[0]); todo.remove(c)
            if len(b) == per: break
        out.append(b)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="the log parser and the classes on a made-up log (no game)")
    ap.add_argument("--rounds", type=int, default=2, help="battles each card is carried in")
    ap.add_argument("--ticks", type=int, default=TICKS)
    ap.add_argument("--limit", type=int, default=0, help="only the first N battles (a quick look)")
    ap.add_argument("--cards", default="", help="only these cards (cards.csv Names, comma separated): a longer look at a few")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--out", default=None, help="json result (default runs/star_audit/probe.json)")
    a = ap.parse_args()
    if a.check: return _check()
    os.environ["NB_STAR_PROBE"] = "1"   # star.c: hook the three lookups (read by the host at boot)
    from nbpc.instance import Instance
    from nbpc import proto as P, diversity as D
    import env_v5 as E
    rows = star_rows(); hs = set(E.heroes())
    skip = [r for r in rows if r[2] not in hs]   # the card's Target is not in the battle pool (a form, or skip.json)
    want = set(a.cards.split(",")) if a.cards else None
    battles = plan([r for r in rows if r[2] in hs and (not want or r[1] in want)], a.rounds)[: a.limit or None]
    print("%d star power cards, %d battles of %d ticks" % (len(rows), len(battles), a.ticks), flush=True)
    inst = Instance(0, tag="star_audit", log_dir=a.log_dir); t0 = time.time(); inst.start(); inst.wait_ready()
    print("host ready in %.0f s" % (time.time() - t0), flush=True)
    start = os.path.getsize(inst.log_path); rej = 0
    try:
        for i, b in enumerate(battles):
            div = D.make_div(location="Survival651", chr=[c[2] for c in b]); div["star"] = [c[0] for c in b]
            r = inst.reset(1000 + i, gas=1, flags=P.F_STATE, native_mask=0xffffffff, obs_mask=0, div=div)
            rej += bin(int(getattr(r, "div_err", 0)) & 0x3ff).count("1")
            for _ in range(a.ticks // STEP): inst.step(ticks=STEP, flags=0, native_mask=0xffffffff)
            print("battle %d/%d done (%.0f s)" % (i + 1, len(battles), time.time() - t0), flush=True)
        inst.reset(9999, gas=1, flags=0, native_mask=0xffffffff, obs_mask=0)   # the next build logs the last battle's counts
        inst.step(ticks=STEP, flags=0, native_mask=0xffffffff)
    finally:
        inst.close()
    with open(inst.log_path, errors="replace") as f: f.seek(start); lines = f.readlines()
    if not any("[star] probe on" in ln for ln in open(inst.log_path, errors="replace")): raise SystemExit("host without star.c (rebuild: ./setup.sh)")
    res = classify(rows, *parse_log(lines)); by = {cid: (n, t, ty) for cid, n, t, ty, _ in rows}
    for c, *_ in skip: res[c]["cls"] = "skipped"
    st = lookup_sites(os.path.join(os.environ.get("NBHOST_APK", "engine/game"), "lib", "arm64-v8a", "libg.so"))
    for r in res.values(): r["site"] = site_kind(st.get(r["type"], [])) if r["type"] is not None else None
    if a.cards: res = {c: r for c, r in res.items() if by[c][0] in want}
    cnt = collections.Counter(r["cls"] for r in res.values())
    out = a.out or os.path.join(os.environ.get("SIM_WORK", "work"), "runs", "star_audit", "probe.json"); os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"cards": {by[c][0]: {"brawler": by[c][1], "Type": by[c][2], **r} for c, r in res.items()}, "classes": cnt,
               "rejected": rej, "ticks": a.ticks, "rounds": a.rounds}, open(out, "w"), indent=1)
    for k in ("read", "asked", "generic", "none", "missing", "skipped"): print("%-8s %d" % (k, cnt.get(k, 0)))
    print("by call site: %s" % dict(collections.Counter("%s/%s" % (r["cls"], r["site"]) for r in res.values())))
    print("rejected requests %d; written %s" % (rej, out))
    print("star_audit ok")


def _check():
    rows = [(23000001, "A", "X", "medikit", False), (23000002, "B", "X", "reload_on_hit", False),
            (23000003, "C", "Y", "heal_forest", True), (23000004, "D", "Y", "cripple", False), (23000005, "E", "Z", "x", False)]
    log = ["[star] p battle=3 slot=0 card=23000001 type=10 off=0", "[star] p battle=3 slot=1 card=23000002 type=11 off=0",
           "[star] p battle=3 slot=2 card=23000003 type=12 off=0", "[star] p battle=4 slot=0 card=23000004 type=13 off=0",
           "[star] q battle=3 type=10 n=50 hit=7 lr=0xb30278", "[star] q battle=3 type=11 n=9 hit=0 lr=0xc00000",
           "[star] q battle=4 type=10 n=5 hit=0 lr=0xb30278", "[star] q battle=3 type=13 n=4 hit=4 lr=0x1", "noise"]
    carried, asked = parse_log(log)
    assert carried[3][1] == (23000002, 11, 0) and asked[3][10] == (50, 7, 0xb30278) and len(asked[4]) == 1
    res = classify(rows, carried, asked)
    assert [res[c]["cls"] for c, *_ in rows] == ["read", "asked", "generic", "none", "missing"], res
    assert res[23000001]["lr"] == 0xb30278 and res[23000004]["n"] == 0   # battle 3's type 13 is not battle 4's
    p = plan(rows, 3, per=2)
    assert sum(len(b) for b in p) == 15 and all(len({c[0] for c in b}) == len(b) for b in p)
    assert all(isinstance(n, str) for n in LIVE) and len(LIVE) == 15
    assert site_kind([0x52ccd4, 0xb30278]) == "sim" and site_kind([0x52ccd4]) == "ui" and site_kind([]) == "nosite"
    import numpy as np, tempfile                                   # a made-up code segment: two calls, one without a constant
    code = np.zeros(TEXT_END // 4, dtype="<u4"); at = 0xc00000 // 4
    code[at] = 0x52800001 | (59 << 5); code[at + 1] = 0x94000000 | ((0xc42874 // 4 - (at + 1)) & 0x3ffffff)   # mov w1,#59; bl
    code[at + 4] = 0x2a0203e1; code[at + 5] = 0x14000000 | ((0xc34be0 // 4 - (at + 5)) & 0x3ffffff)           # mov w1,w2; b
    with tempfile.NamedTemporaryFile(suffix=".so") as f:
        code.tofile(f.name); assert lookup_sites(f.name) == {59: [(at + 1) * 4]}
    print("star_audit --check ok")


if __name__ == "__main__":
    sys.exit(main())
