# Making the sim run

The battles run in **nirkt's battle host** (`bulletlabels`), driven through nirkt's `MAEnv` (`nbrl-env`). This
repository adds the observation, the battle randomisation and the trainer on top. You need all of the pieces below.
Without the host, only the offline self-checks run.

## What you need

| Piece | From | Notes |
|---|---|---|
| `bulletlabels` at commit `52184b2` | [nirkt/bulletlabels](https://github.com/nirkt/bulletlabels) | Battle host: `native/loader` (host binary), `native/qemu` (patched `qemu-aarch64`), `vendor/nbpc` (Python side: instances, observation encoding, `MAEnv` glue). |
| `nbrl-env` | [nirkt/nbrl-env](https://github.com/nirkt/nbrl-env) | `py/` (`models4`: network blocks, masked log-prob / entropy) and `train/`. |
| Game data folder | you | Put it in `$BULLETLABELS/game`. Not distributed by anyone here. |
| `host/fair_view.patch` | this repo | Camera-view window for `vendor/nbpc/fair.py` (measured trapezoid view, per-episode zoom, teammates never bush-hidden). |
| Training-round host changes | maintainers | Team groups, summons / area effects in the observation, per-instance map folders, teammate respawn. Needed for duo / trio and generated maps; solo battles work without them. Not published yet. |
| Linux x86-64 or WSL2 (Ubuntu 24.04), Python 3.11+ | | `numpy`, `torch` (CUDA or ROCm for real runs) and nirkt's Python requirements. |

## Steps

```bash
# 1. nirkt's code
git clone https://github.com/nirkt/bulletlabels bulletlabels && git -C bulletlabels checkout 52184b2
git clone https://github.com/nirkt/nbrl-env nbrl-env
git -C bulletlabels apply ../bsrl-sim/host/fair_view.patch

# 2. build the host (nirkt's native/README.md has the details); the binary must be native/loader/nbhost_rl
cd bulletlabels/native/loader && sed 's/nbhost_spawnfix2/nbhost_rl/g' build.sh | sh && cd -
#    game data -> bulletlabels/game

# 3. Python
python3 -m venv ~/sim-venv && ~/sim-venv/bin/pip install numpy torch

# 4. environment (every variable has a default; see env.sh)
export BULLETLABELS=$PWD/bulletlabels NBRL=$PWD/nbrl-env SIM_WORK=$HOME/sim-work SIM_VENV=$HOME/sim-venv
cd bsrl-sim && source env.sh

# 5. checks
python -m obs_v5 && python -m model && python -m env_v5 --check   # offline, no battle
python -m env_v5 --n 2 --minutes 2 --agents 10                    # boots the host: random-action benchmark
python -m live_check --n 2 --minutes 8                             # team / respawn checks (need the round changes)
```

Then train: README "Setup".

## Troubleshooting
- **Nothing printed after start.** `MAEnv` retries a failed boot forever without a message. Boot one `nbpc.instance`
  with `log_dir=.` and read its `game_*.log`.
- **Every boot dies at once.** `NB_PIX_CAP` (set in `env.sh`, `2400*1080*4`) must equal the frame slot the host was
  built with.
- **Variables come out empty.** `env.sh` must have LF line endings (CRLF breaks every variable). From Windows, pass
  scripts to `wsl.exe ... bash -s` on stdin.
- **Slow start.** One boot takes 60-90 s. Batch your checks into one run.
- **A generated map crashes its instance (exit 139).** Layouts with a payload-track tile crash at boot.
  `env_v5.use_genmaps` skips them.
- **The fire head never uses gadgets or hypercharge.** The host cannot trigger them, so actions are none / attack /
  super.
