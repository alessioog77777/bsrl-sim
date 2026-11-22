# Making the sim run

This repository is the master checkout: it pulls in **nirkt's** two repositories as git submodules under `deps/`
and adds this project's changes on top. `setup.sh` does all of it.

## Quick start (Linux x86-64 or WSL2 Ubuntu 24.04)

```bash
git clone --recursive https://github.com/mlgdoggynoob/bsrl-sim && cd bsrl-sim
APK=/path/to/nb_68.279_release_7c5c5152.apk ./setup.sh     # ~10 min the first time
source env.sh
python -m obs_v5 && python -m model && python -m env_v5 --check   # self-checks: no battle, need the game files
python -m live_check --n 2 --minutes 8                             # boots the host: teams, respawn, maps, HUD mask
python -m train_ppo --run t --smoke                                # 3-update plumbing check of the trainer
```

The game build is not distributed here and `setup.sh` never downloads it: you supply it. Its `lib/arm64-v8a/libg.so`
must match `deps/bulletlabels/native/LIBG_SHA256`. `./setup.sh --no-apt` skips system packages, `--no-python` skips
the venv. `TORCH_INDEX_URL` picks the torch build (for example ROCm or CPU wheels).

## What setup.sh puts together

| Piece | Where | From |
|---|---|---|
| Battle host: `native/loader` (host binary sources), `native/qemu` (patched `qemu-aarch64` build), `vendor/nbpc` (instances, observation encoding) | `deps/bulletlabels` @ `52184b2` | [nirkt/bulletlabels](https://github.com/nirkt/bulletlabels) |
| Multi-agent environment `MAEnv` and camera rules (`py/nbpc/maenv.py`, `fair.py`), network blocks (`py/models4.py`) | `deps/nbrl-env` @ `7ff57bc` | [nirkt/nbrl-env](https://github.com/nirkt/nbrl-env) |
| Training round of the host: team groups, summons and area effects in the observation, per-instance map folders, teammate respawn | `host/training_round.patch` | this repo |
| Camera view window: measured trapezoid, per-episode zoom, teammates never bush-hidden | `host/fair_view.patch` | this repo |
| Game files | `deps/bulletlabels/game` | you (`APK=`) |

Steps, in order:
1. `git submodule update --init` (nirkt's repos at the pinned commits).
2. System packages (APT list at the top of `setup.sh`).
3. Copy nirkt's `maenv.py` and `fair.py` from `deps/nbrl-env/py/nbpc` into `deps/bulletlabels/vendor/nbpc`, so one
   `nbpc` package has the host side and the environment side.
4. Apply `host/training_round.patch`, then `host/fair_view.patch`, to `deps/bulletlabels`.
5. Build the host: `native/loader/build.sh` with the output named `nbhost_rl` (static ARM64 binary, runs under qemu).
6. Build nirkt's patched qemu (~5 min). If that fails, `env.sh` uses the system `qemu-aarch64` (about 2x slower,
   same results).
7. `.venv` with nirkt's requirements and torch.
8. Unpack the game files from the APK and check them against `LIBG_SHA256`.

`env.sh` finds all of it under `deps/`. Each path can be overridden: `BULLETLABELS`, `NBRL`, `SIM_WORK`
(runs and generated maps, default `work/`), `SIM_VENV`.

The submodules are nirkt's code, unchanged in git. `setup.sh` only changes the working tree under `deps/`.
`git -C deps/bulletlabels status` shows what it changed.

## Troubleshooting
- **Nothing printed after start.** `MAEnv` retries a failed boot forever without a message. Boot one `nbpc.instance`
  with `log_dir=.` and read its `game_*.log`.
- **Every boot dies at once.** `NB_PIX_CAP` (set in `env.sh`, `2400*1080*4`) must equal the frame slot the host was
  built with (`NBE_PIX_CAP` in `native/loader/nbenv_proto.h`).
- **Variables come out empty.** `env.sh` must have LF line endings (CRLF breaks every variable). From Windows, pass
  scripts to `wsl.exe ... bash -s` on stdin.
- **A patch does not apply.** The submodule is not at the pinned commit: `git submodule update --init --force`, then
  rerun `setup.sh`.
- **Slow start.** One boot takes 60-90 s. Batch your checks into one run.
- **A generated map crashes its instance (exit 139).** Layouts with a payload-track tile crash at boot.
  `env_v5.use_genmaps` skips them.
- **The fire head never uses gadgets or hypercharge.** The host cannot trigger them, so actions are none / attack /
  super.
