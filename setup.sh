#!/usr/bin/env bash
# One-command setup: builds the battle host in engine/ (nirkt's code with this project's changes already applied), the
# Python environment, and unpacks the game files. Safe to run again: every step skips what is already done.
#
#   ./setup.sh [--no-apt] [--no-python]
#
#   --no-apt      do not install system packages (you install the APT list below yourself)
#   --no-python   do not create .venv (you manage numpy / torch / the requirements yourself)
#   APK           a local copy of the game build the host is made for (checked against
#                 engine/native/LIBG_SHA256). Without it the build is downloaded once from APK_URL
#                 into work/ (nirkt's copy, the link from nirkt's original setup.sh; not an official Null's Brawl host).
#   APK_URL       where to download it from (default: nirkt's copy)
#   SIM_VENV      where the venv goes (default .venv); TORCH_INDEX_URL picks the torch build (CUDA default, ROCm / CPU)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd); cd "$ROOT"
ENG=$ROOT/engine; VENV=${SIM_VENV:-$ROOT/.venv}
APK_URL=${APK_URL:-https://pub-3e0f25080eb949a08641fd278fad96d1.r2.dev/nb_68.279_release_7c5c5152.apk}
APT=(gcc gcc-aarch64-linux-gnu libgles-dev libegl-dev ninja-build meson pkg-config flex bison libglib2.0-dev
     libpixman-1-dev zlib1g-dev libzstd-dev unzip xz-utils curl qemu-user python3-venv python3-pip)
NOAPT=0; NOPY=0
for a in "$@"; do case $a in --no-apt) NOAPT=1;; --no-python) NOPY=1;; *) echo "unknown option $a"; exit 2;; esac; done
say() { printf '\n== %s\n' "$*"; }

if [ $NOAPT = 0 ] && command -v apt-get >/dev/null; then
  say "system packages"; sudo apt-get update -qq && sudo apt-get install -y -qq "${APT[@]}"
fi

say "host binary (engine/native/loader/nbhost_rl)"
if [ ! -x "$ENG/native/loader/nbhost_rl" ]; then
  (cd "$ENG/native/loader" && sed 's/nbhost_spawnfix2/nbhost_rl/g' build.sh | sh 2>&1 | grep -E "error:|^built ")
fi
[ -x "$ENG/native/loader/nbhost_rl" ] || { echo "host build failed (needs gcc-aarch64-linux-gnu, libgles-dev)"; exit 1; }

say "patched qemu (~5 min once; the system qemu-aarch64 works too, ~2x slower)"
[ -x "$ENG/native/qemu/bin/qemu-aarch64-nb" ] || "$ENG/native/qemu/build.sh" || echo "qemu build failed: env.sh falls back to the system qemu-aarch64"

if [ $NOPY = 0 ]; then
  say "Python environment ($VENV)"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install -q -r "$ENG/requirements.txt"
  "$VENV/bin/python" -c "import torch" 2>/dev/null || "$VENV/bin/pip" install -q torch ${TORCH_INDEX_URL:+--index-url "$TORCH_INDEX_URL"}
fi

say "game files ($ENG/game)"
if [ ! -f "$ENG/game/lib/arm64-v8a/libg.so" ]; then
  if [ -n "${APK:-}" ]; then
    [ -f "$APK" ] || { echo "APK=$APK: no such file"; exit 1; }
  else
    APK=${SIM_WORK:-$ROOT/work}/$(basename "$APK_URL")
    if [ ! -f "$APK" ]; then
      echo "downloading $APK_URL"; mkdir -p "$(dirname "$APK")"
      curl -fL --retry 3 -o "$APK.part" "$APK_URL" && mv "$APK.part" "$APK"
    fi
  fi
  rm -rf "$ENG/game" && mkdir -p "$ENG/game" && unzip -q -o "$APK" 'lib/arm64-v8a/*' 'assets/*' -d "$ENG/game"
fi
WANT=$(cut -d' ' -f1 "$ENG/native/LIBG_SHA256"); HAVE=$(sha256sum "$ENG/game/lib/arm64-v8a/libg.so" | cut -d' ' -f1)
[ "$HAVE" = "$WANT" ] || { rm -rf "$ENG/game"   # so the next run unpacks again instead of keeping a bad build
  echo "wrong game build: libg.so sha256 $HAVE, the host needs $WANT (delete or replace ${APK:-the APK})"; exit 1; }
echo "game files ok"

say "done"
echo "Next:  source env.sh && python -m obs_v5 && python -m env_v5 --check && python -m live_check --n 2 --minutes 8"
