# Making the sim run

Everything the sim needs is in this repository: nirkt's battle host and environment are in `engine/` with this
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
`lib/arm64-v8a/libg.so` must match `engine/native/LIBG_SHA256`, or `setup.sh` stops before anything runs it. `./setup.sh --no-apt` skips system packages, `--no-python` skips
the venv. `TORCH_INDEX_URL` picks the torch build (for example ROCm or CPU wheels).

## What is in engine/

One tree of nirkt's code (`engine/NOTICE.md` says which file came from which repo):

| Folder | What | From |
|---|---|---|
| `engine/native` | Battle host: `loader` (host binary sources), `qemu` (patched `qemu-aarch64` build), `glhost`, `payloads`, `LIBG_SHA256` | [nirkt/bulletlabels](https://github.com/nirkt/bulletlabels) @ `52184b2` |
| `engine/py` | `nbpc` (instances, observation encoding, `MAEnv`, camera rules), `state2.py`, `dumps.py` (bulletlabels' `vendor/`); `models4.py` (network blocks, masked log-prob / entropy) and nbrl-env's other `py/` scripts | bulletlabels; [nirkt/nbrl-env](https://github.com/nirkt/nbrl-env) @ `7ff57bc` |
| `engine/train` | nbrl-env's example training loop | nbrl-env |
| `engine/labeller`, `framegen`, `debug`, `data`, `docs`, `tests`, `bl.py` | bulletlabels' vision-label tools (not used by the sim) | bulletlabels |
| `engine/patches` | This project's changes, listed below | this project |

Changes made to nirkt's code, all recorded in `engine/patches/`:
- `training_round.patch`: team groups, summons and area effects in the observation, per-instance map folders,
  teammate respawn.
- `fair_view.patch`: camera view window (measured trapezoid, per-episode zoom, teammates never bush-hidden).
- `live_state.patch`: live map tiles in `F_MAP` and the observation grid (a destroyed wall / bush reads as open),
  area effects' data id in item feature 5 (obs_v5 looks up radius and kind), power cubes lost at a team knock-out.
- `gadgets.patch` (issue #3, a diff on top of the three above): per-player gadget state in the response (`NbeResp`
  `gad_*`, mirrored in `nbpc/proto.py`) and the host's own gadget activation in `native/loader/gadget.c`, which
  `env.c` `gadget_input` tries first. No gadget kind is implemented yet, so every gadget still reads as not usable.
- `hypercharge.patch` (issue #4, a diff on top of `gadgets.patch`): the hypercharge card on request (`NbeDiv.hyper_mask`,
  put in the player config by `offline.c`), activation by `NbeAction.fire` 4 through libg's own activation, and in
  `native/loader/hyper.c` the parts the client build lacks: the drain to 0 over 5 s, the end, and the speed / damage /
  shield buffs from `characters.csv` (`env.c` `rec_move` / `my_root`). State in `NbeResp` `hyp_*`, mirrored in `nbpc/proto.py`.
- `maenv.py` and `fair.py` copied from nbrl-env's `py/nbpc` into bulletlabels' `vendor/nbpc` (now `engine/py/nbpc`),
  so one `nbpc` package holds the host side and the environment side.
- The game download link removed from bulletlabels' `setup.sh` (`engine/setup.sh`) (this repo's `setup.sh` downloads it instead).

What `setup.sh` does: system packages (APT list at its top), builds the host as `native/loader/nbhost_rl` (static
ARM64 binary, runs under qemu), builds nirkt's patched qemu (~5 min; if that fails, `env.sh` uses the system
`qemu-aarch64`, about 2x slower, same results), creates `.venv`, downloads (unless `APK=` is given), unpacks and checks the game files.

`env.sh` finds all of it under `engine/`. Each path can be overridden: `SIM_ENGINE` (the engine folder), `SIM_WORK` (runs and the
downloaded game build, default `work/`), `SIM_VENV`.

## Troubleshooting
- **Nothing printed after start.** `MAEnv` retries a failed boot forever without a message. Boot one `nbpc.instance`
  with `log_dir=.` and read its `game_*.log`.
- **Every boot dies at once.** `NB_PIX_CAP` (set in `env.sh`, `2400*1080*4`) must equal the frame slot the host was
  built with (`NBE_PIX_CAP` in `native/loader/nbenv_proto.h`).
- **Variables come out empty.** `env.sh` must have LF line endings (CRLF breaks every variable). From Windows, pass
  scripts to `wsl.exe ... bash -s` on stdin.
- **Slow start.** One boot takes 60-90 s. Batch your checks into one run.
- **The fire head never uses gadgets.** The game build compiles gadget activation out, and the host activates only the
  kinds `native/loader/gadget.c` implements (none yet), so the gadget class stays masked. `python -m gadget_check` shows
  which kinds the built host publishes.
- **The fire head never uses hypercharge.** It needs `--hyper 1` (players get their card) and a host built with
  `native/loader/hyper.c`. `python -m hyper_check` checks the built host.
- **Every boot crashes at "loading 45%".** A game file is a Git LFS pointer, not the file (seen with `assets/sc/ui.sc`
  from a git clone of the game files): run `git lfs pull` in that clone.
- **`gadget_check` / `hyper_check` say the host publishes no gadget / hypercharge state.** `setup.sh` builds the host only when
  `native/loader/nbhost_rl` is missing: after a change in `engine/native/loader`, delete it and rerun
  `./setup.sh --no-apt --no-python`.
