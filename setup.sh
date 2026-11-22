#!/usr/bin/env bash
# One-command setup: nirkt's host and environment (deps/, git submodules), this project's host changes, the Python
# environment. Safe to run again: every step skips what is already done.
#
#   APK=/path/to/nb_68.279_release_7c5c5152.apk ./setup.sh [--no-apt] [--no-python]
#
#   --no-apt      do not install system packages (you install the APT list below yourself)
#   --no-python   do not create .venv (you manage numpy / torch / nirkt's requirements yourself)
#   APK           the game build the host is made for (checked against deps/bulletlabels/native/LIBG_SHA256).
#                 Only needed the first time; this script never downloads it.
#   SIM_VENV      where the venv goes (default .venv); TORCH_INDEX_URL picks the torch build (CUDA default, ROCm / CPU)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
BL=$ROOT/deps/bulletlabels; NB=$ROOT/deps/nbrl-env; VENV=${SIM_VENV:-$ROOT/.venv}
APT=(gcc gcc-aarch64-linux-gnu libgles-dev libegl-dev ninja-build meson pkg-config flex bison libglib2.0-dev
     libpixman-1-dev zlib1g-dev libzstd-dev unzip xz-utils curl qemu-user python3-venv python3-pip)
NOAPT=0; NOPY=0
for a in "$@"; do case $a in --no-apt) NOAPT=1;; --no-python) NOPY=1;; *) echo "unknown option $a"; exit 2;; esac; done
say() { printf '\n== %s\n' "$*"; }

say "nirkt's repos (deps/)"
git submodule update --init --depth 1 deps/bulletlabels deps/nbrl-env 2>/dev/null \
  || git submodule update --init deps/bulletlabels deps/nbrl-env

if [ $NOAPT = 0 ] && command -v apt-get >/dev/null; then
  say "system packages"; sudo apt-get update -qq && sudo apt-get install -y -qq "${APT[@]}"
fi

say "nbpc: nirkt's environment files next to the host's (vendor/nbpc)"
for f in fair.py maenv.py; do [ -f "$BL/vendor/nbpc/$f" ] || cp "$NB/py/nbpc/$f" "$BL/vendor/nbpc/$f"; done

say "this project's host changes (host/*.patch)"
for p in host/training_round.patch host/fair_view.patch; do
  if git -C "$BL" apply --reverse --check "$ROOT/$p" 2>/dev/null; then echo "$p: already applied"
  else git -C "$BL" apply "$ROOT/$p" && echo "$p: applied"; rm -f "$BL/native/loader/nbhost_rl"; fi
done

say "host binary (native/loader/nbhost_rl)"
if [ ! -x "$BL/native/loader/nbhost_rl" ]; then
  (cd "$BL/native/loader" && sed 's/nbhost_spawnfix2/nbhost_rl/g' build.sh | sh 2>&1 | grep -E "error:|^built ")
fi
[ -x "$BL/native/loader/nbhost_rl" ] || { echo "host build failed (needs gcc-aarch64-linux-gnu, libgles-dev)"; exit 1; }

say "patched qemu (~5 min once; the system qemu-aarch64 works too, ~2x slower)"
[ -x "$BL/native/qemu/bin/qemu-aarch64-nb" ] || "$BL/native/qemu/build.sh" || echo "qemu build failed: env.sh falls back to the system qemu-aarch64"

if [ $NOPY = 0 ]; then
  say "Python environment ($VENV)"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "$BL/requirements.txt"
  "$VENV/bin/python" -c "import torch" 2>/dev/null || "$VENV/bin/pip" install -q torch ${TORCH_INDEX_URL:+--index-url "$TORCH_INDEX_URL"}
fi

say "game files ($BL/game)"
if [ ! -f "$BL/game/lib/arm64-v8a/libg.so" ]; then
  if [ -z "${APK:-}" ] || [ ! -f "$APK" ]; then
    echo "Missing. Rerun with the game build the host is made for: APK=/path/to/the.apk ./setup.sh --no-apt"
    echo "(its lib/arm64-v8a/libg.so must match deps/bulletlabels/native/LIBG_SHA256)"; exit 1
  fi
  rm -rf "$BL/game" && mkdir -p "$BL/game" && unzip -q -o "$APK" 'lib/arm64-v8a/*' 'assets/*' -d "$BL/game"
fi
WANT=$(cut -d' ' -f1 "$BL/native/LIBG_SHA256"); HAVE=$(sha256sum "$BL/game/lib/arm64-v8a/libg.so" | cut -d' ' -f1)
[ "$HAVE" = "$WANT" ] || { echo "wrong game build: libg.so sha256 $HAVE, the host needs $WANT"; exit 1; }
echo "game files ok"

say "done"
echo "Next:  source env.sh && python -m obs_v5 && python -m env_v5 --check && python -m live_check --n 2 --minutes 8"
