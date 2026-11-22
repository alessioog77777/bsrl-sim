"""Put a measure_noise.py report into obs_v5.py (HISTORY 85): NOISE jitter / fp / seg upper bounds = 2 x measured,
DROP_CLS = measured per-class miss chance (classes the report lacks keep their value). Refuses values more than 4 x
off the current measurement (a broken run), so a bad log never reaches training.
  python apply_noise.py REPORT.log TAG [--dry]     TAG names the det in the comment, e.g. "det_v7 (HISTORY 85)"
Self-check: python apply_noise.py --test"""
import re, sys

FIELDS = {"jitter": r'("jitter": \(0\.0, )([\d.]+)(\),.*\(measured )([\d.]+)(\))',
          "fp": r'("fp": \(0\.0, )([\d.]+)(\),.*\(measured )([\d.]+)(\))',
          "seg": r'("seg": \(0\.0, )([\d.]+)(\),.*\(measured )([\d.]+)(\))'}
DROP = r"(# measured miss chance per class, ).*?(, ENT order\r?\nDROP_CLS = np\.array\(\[)([^\]]*)(\], np\.float32\))"


def parse(report):
    m = re.search(r"^measured: (.*?)\s{2,}", report, re.M)
    if not m:
        raise ValueError("no 'measured:' line")
    meas = {k: float(v) for k, v in re.findall(r"(\w+) ([\d.]+)", m.group(1))}
    meas["frames"] = int(re.search(r"^frames (\d+)", report, re.M).group(1))
    drop = {c: float(p) / 100 for c, p in re.findall(r"^\s+(\w+)\s+n\s+\d+\s+drop ([\d.]+)%", report, re.M)}
    return meas, drop


def apply(src, meas, drop, ent, tag):
    out = src
    for k, pat in FIELDS.items():
        m = re.search(pat, out)
        old, new = float("0" + m.group(4)) if m.group(4).startswith(".") else float(m.group(4)), meas[k]
        if not (old / 4 <= new <= old * 4):
            raise ValueError(f"{k}: measured {new:.4f} vs {old:.4f} before (> 4 x off): not applied")
        out = out[:m.start()] + re.sub(pat, lambda g: f"{g[1]}{2 * new:.3f}{g[3]}{new:.4f}{g[5]}", m.group(0)) + out[m.end():]
    m = re.search(DROP, out, re.S)
    vals = [float(v) for v in m.group(3).split(",")]
    for i, c in enumerate(ent):
        if c in drop:
            if not (vals[i] / 4 <= drop[c] <= max(vals[i] * 4, 0.05)):
                raise ValueError(f"drop {c}: {drop[c]:.3f} vs {vals[i]:.3f} before (> 4 x off): not applied")
            vals[i] = drop[c]
    body = ", ".join(f"{v:.3f}".lstrip("0") if v < 1 else f"{v:.3f}" for v in vals)
    note = f"{tag} vs labels on {meas['frames']} val frames"
    return out[:m.start()] + m.group(1) + note + m.group(2) + body + m.group(4) + out[m.end():]


def _test():
    ent = ("enemy", "teammate")
    src = ('NOISE = {"jitter": (0.0, 0.039),   # x (measured .0195)\n         "fp": (0.0, 0.060),   # y (measured .030)\n'
           '         "seg": (0.0, 0.137),   # z (measured .068)\n# measured miss chance per class, det_v6 vs labels on '
           '1000 val frames (HISTORY 61), ENT order\nDROP_CLS = np.array([.079, .085], np.float32)\n')
    rep = ("frames 1000; ...\nmeasured: jitter 0.0180, drop 0.1000, fp 0.0250, seg 0.0600   (jitter = sigma)\n"
           "  enemy                  n   500  drop 6.0%  fp   10  median err 0.20 tiles\n")
    meas, drop = parse(rep)
    out = apply(src, meas, drop, ent, "det_v7 (HISTORY 85)")
    assert '"jitter": (0.0, 0.036)' in out and "(measured 0.0180)" in out and '"fp": (0.0, 0.050)' in out, out
    assert '"seg": (0.0, 0.120)' in out and "[.060, .085]" in out and "det_v7 (HISTORY 85) vs labels on 1000 val frames, ENT" in out, out
    try:
        apply(src, {**meas, "seg": 0.9}, drop, ent, "x"); raise AssertionError("4 x guard")
    except ValueError:
        pass
    print("apply_noise self-check ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        _test(); sys.exit()
    sys.path.insert(0, "."); import obs_v5   # noqa: E401
    rep, tag = open(sys.argv[1]).read(), sys.argv[2]
    src = open("obs_v5.py", encoding="utf-8", newline="").read()   # keep CRLF
    new = apply(src, *parse(rep), obs_v5.ENT, tag)
    if "--dry" in sys.argv:
        import difflib; sys.stdout.writelines(difflib.unified_diff(src.splitlines(1), new.splitlines(1), "obs_v5.py", "new"))
    else:
        open("obs_v5.py", "w", encoding="utf-8", newline="").write(new); print("obs_v5.py updated from", sys.argv[1])
