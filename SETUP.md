# Making the sim run

Everything the sim needs is in this repository: nirkt's battle host and environment are in `deps/` with this
project's changes already applied (README "Credits"). `setup.sh` builds it.

## Quick start (Linux x86-64 or WSL2 Ubuntu 24.04)

```bash
git clone https://github.com/mlgdoggynoob/bsrl-sim && cd bsrl-sim
./setup.sh                                                         # ~10 min the first time
source env.sh
python -m obs_v5 && python -m model && python -m env_v5 --check   # self-checks: no battle, need the game files
python -m live_check --n 2 --minutes 8                             # boots the host: teams, respawn, maps, HUD mask
python -m train_ppo --run t --smoke                                # 3-update plumbing check of the trainer
```

The game build is not stored in this repository. `setup.sh` downloads it (1.2 GB) once into `work/` from `APK_URL`, by default
nirkt's copy (the link from nirkt's original `setup.sh`; not an official Null's Brawl host, so it may go away). To use
a copy you already have: `APK=/path/to/nb_68.279_release_7c5c5152.apk ./setup.sh`. Either way its
`lib/arm64-v8a/libg.so` must match `deps/bulletlabels/native/LIBG_SHA256`, or `setup.sh` stops before anything runs it. `./setup.sh --no-apt` skips system packages, `--no-python` skips
the venv. `TORCH_INDEX_URL` picks the torch build (for example ROCm or CPU wheels).

## What is in deps/

| Folder | What | From |
|---|---|---|
| `deps/bulletlabels` | Battle host: `native/loader` (host binary sources), `native/qemu` (patched `qemu-aarch64` build), `vendor/nbpc` (instances, observation encoding, `MAEnv`, camera rules) | [nirkt/bulletlabels](https://github.com/nirkt/bulletlabels) @ `52184b2` |
| `deps/nbrl-env` | `py/` (`models4.py`: network blocks, masked log-prob / entropy; the original `nbpc`) and `train/` | [nirkt/nbrl-env](https://github.com/nirkt/nbrl-env) @ `7ff57bc` (models, qemu binary and duplicate host sources left out) |

Changes made to nirkt's code, all recorded in `host/`:
- `host/training_round.patch`: team groups, summons and area effects in the observation, per-instance map folders,
  teammate respawn.
- `host/fair_view.patch`: camera view window (measured trapezoid, per-episode zoom, teammates never bush-hidden).
- `host/live_state.patch`: live map tiles in `F_MAP` and the observation grid (a destroyed wall / bush reads as open),
  area effects' data id in item feature 5 (obs_v5 looks up radius and kind), power cubes lost at a team knock-out.
- `maenv.py` and `fair.py` copied from `deps/nbrl-env/py/nbpc` into `deps/bulletlabels/vendor/nbpc`, so one `nbpc`
  package holds the host side and the environment side.
- The game download link removed from `deps/bulletlabels/setup.sh` (this repo's `setup.sh` downloads it instead).

What `setup.sh` does: system packages (APT list at its top), builds the host as `native/loader/nbhost_rl` (static
ARM64 binary, runs under qemu), builds nirkt's patched qemu (~5 min; if that fails, `env.sh` uses the system
`qemu-aarch64`, about 2x slower, same results), creates `.venv`, downloads (unless `APK=` is given), unpacks and checks the game files.

`env.sh` finds all of it under `deps/`. Each path can be overridden: `BULLETLABELS`, `NBRL`, `SIM_WORK` (runs and
generated maps, default `work/`), `SIM_VENV`.

## Troubleshooting
- **Nothing printed after start.** `MAEnv` retries a failed boot forever without a message. Boot one `nbpc.instance`
  with `log_dir=.` and read its `game_*.log`.
- **Every boot dies at once.** `NB_PIX_CAP` (set in `env.sh`, `2400*1080*4`) must equal the frame slot the host was
  built with (`NBE_PIX_CAP` in `native/loader/nbenv_proto.h`).
- **Variables come out empty.** `env.sh` must have LF line endings (CRLF breaks every variable). From Windows, pass
  scripts to `wsl.exe ... bash -s` on stdin.
- **Slow start.** One boot takes 60-90 s. Batch your checks into one run.
- **A generated map crashes its instance (exit 139).** Layouts with a payload-track tile crash at boot.
  `env_v5.use_genmaps` skips them.
- **The fire head never uses gadgets or hypercharge.** The host cannot trigger them, so actions are none / attack /
  super.
