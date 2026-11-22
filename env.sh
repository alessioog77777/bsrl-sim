# source this in WSL / Linux before any script here: points nirkt's MAEnv at the battle host.
# Every path can be overridden before sourcing. Defaults: deps/ (the public repo's submodules, set up by setup.sh) when
# it exists, else the original layout (~/nb68).
export RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this folder, wherever it is checked out
D=$RL_ROOT/deps
if [ -d "$D/bulletlabels/native" ]; then
  B=${BULLETLABELS:-$D/bulletlabels}; export NBRL=${NBRL:-$D/nbrl-env}
  export SIM_WORK=${SIM_WORK:-$RL_ROOT/work}; V=${SIM_VENV:-$RL_ROOT/.venv}
else
  B=${BULLETLABELS:-$HOME/nb68/bulletlabels}; export NBRL=${NBRL:-$(dirname "$RL_ROOT")/reference/nirkt/nbrl-env}
  export SIM_WORK=${SIM_WORK:-$HOME/nb68/rl}; V=${SIM_VENV:-$HOME/nb68/venv}
fi
# B = nirkt's bulletlabels with the host built (NOTICE.md); NBRL = nirkt's nbrl-env; SIM_WORK = runs/ + genmaps/
export NBHOST_BIN=$B/native/loader/nbhost_rl   # host + the training round (team groups, obs fields); setup.sh builds it
export NBHOST_APK=$B/game
export NB_PIX_CAP=10368000   # = 2400*1080*4, the frame slot the host is built with (a mismatch kills every boot silently)
export NBGL_BIN=$B/native/glhost/nbgl
export NBPC_PAYLOADS=$B/native/payloads
q=$(ls $B/native/qemu/bin/qemu-aarch64-nb $HOME/v5dl/qemu-nb/qemu-aarch64-nb 2>/dev/null | head -1)
q=${q:-$(command -v qemu-aarch64 || true)}   # no patched build: the system qemu-user (~2x slower, same results)
export NBHOST_QEMU=$(readlink -f "$q")
export PYTHONPATH=$RL_ROOT:$B/vendor:$NBRL/py:$NBRL/train   # vendor nbpc first: ours has team_mask / map_location and the host PIX_CAP
[ -f "$V/bin/activate" ] && source "$V/bin/activate"
mkdir -p "$SIM_WORK"
