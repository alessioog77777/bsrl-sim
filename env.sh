# source this in WSL / Linux before any script here: points nirkt's MAEnv at the libg 68 host.
# Every path can be overridden before sourcing; the defaults are the original layout (~/nb68).
B=${BULLETLABELS:-$HOME/nb68/bulletlabels}     # nirkt's bulletlabels checkout with the host built (NOTICE.md)
export RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)   # this folder, wherever it is checked out
export NBRL=${NBRL:-$(dirname "$RL_ROOT")/reference/nirkt/nbrl-env}   # nirkt's nbrl-env checkout
export SIM_WORK=${SIM_WORK:-$HOME/nb68/rl}     # run outputs (runs/) and generated maps (genmaps/)
export NBHOST_BIN=$B/native/loader/nbhost_rl   # = nbhost_maps + the rl round (team groups, obs fields); build: README
export NBHOST_APK=$B/game
export NB_PIX_CAP=10368000   # = 2400*1080*4, the frame slot nbhost_maps is built with (a mismatch kills every boot silently)
export NBGL_BIN=$B/native/glhost/nbgl
export NBPC_PAYLOADS=$B/native/payloads
q=$(ls $B/native/qemu/bin/qemu-aarch64-nb $HOME/v5dl/qemu-nb/qemu-aarch64-nb 2>/dev/null | head -1)
export NBHOST_QEMU=$(readlink -f "$q")
export PYTHONPATH=$RL_ROOT:$B/vendor:$NBRL/py:$NBRL/train   # vendor nbpc first: ours has team_mask / map_location and the host PIX_CAP
V=${SIM_VENV:-$HOME/nb68/venv}
[ -f "$V/bin/activate" ] && source "$V/bin/activate"
mkdir -p "$SIM_WORK"
