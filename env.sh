# source this in WSL / Linux before any script here: points nirkt's MAEnv at the battle host.
# Every path can be overridden before sourcing. Defaults: engine/ (nirkt's host and environment, built by setup.sh) when
# it exists, else the original layout (~/nb68, nirkt's two repos side by side: BULLETLABELS and NBRL).
export RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this folder, wherever it is checked out
if [ -n "${SIM_ENGINE:-}" ] || [ -d "$RL_ROOT/engine/native" ]; then
  B=${SIM_ENGINE:-$RL_ROOT/engine}; PY=$B/py:$B/train
  export SIM_WORK=${SIM_WORK:-$RL_ROOT/work}; V=${SIM_VENV:-$RL_ROOT/.venv}
else
  B=${BULLETLABELS:-$HOME/nb68/bulletlabels}; export NBRL=${NBRL:-$(dirname "$RL_ROOT")/reference/nirkt/nbrl-env}
  PY=$B/vendor:$NBRL/py:$NBRL/train
  export SIM_WORK=${SIM_WORK:-$HOME/nb68/rl}; V=${SIM_VENV:-$HOME/nb68/venv}
fi
# B = the engine (host built, engine/NOTICE.md); PY = its Python (nbpc, models4); SIM_WORK = runs/ + genmaps/
export NBHOST_BIN=$B/native/loader/nbhost_rl   # host + the training round (team groups, obs fields); setup.sh builds it
export NBHOST_APK=$B/game
export NB_PIX_CAP=10368000   # = 2400*1080*4, the frame slot the host is built with (a mismatch kills every boot silently)
export NBGL_BIN=$B/native/glhost/nbgl
export NBPC_PAYLOADS=$B/native/payloads
q=$(ls $B/native/qemu/bin/qemu-aarch64-nb $HOME/v5dl/qemu-nb/qemu-aarch64-nb 2>/dev/null | head -1)
q=${q:-$(command -v qemu-aarch64 || true)}   # no patched build: the system qemu-user (~2x slower, same results)
export NBHOST_QEMU=$(readlink -f "$q")
export PYTHONPATH=$RL_ROOT:$PY   # the engine's nbpc (team_mask / map_location, the host PIX_CAP) before nirkt's py/ scripts
[ -f "$V/bin/activate" ] && source "$V/bin/activate"
mkdir -p "$SIM_WORK"
