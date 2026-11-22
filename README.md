# bsrl-sim

Reinforcement-learning environment and trainer for a 10-player top-down battle royale (Showdown mode). The policy
does not get the true game state. It gets what vision models would read from the screen: a 25 x 25 tile grid of terrain
classes, a set of detected entities with measured detector noise and latency, and a few HUD numbers. That makes this a
testbed for training agents that have to act from imperfect perception, and policies trained here can run behind real
vision models without retraining.

This repository is the RL layer only. The battles themselves run in **nirkt's** `nbrl-env` / `bulletlabels` host
(see [NOTICE.md](NOTICE.md)), which this code drives through its `MAEnv`. Neither that host nor any game file is
included here, and nothing here runs without it.

## What is here

| File | What it does |
|---|---|
| `env_v5.py` | Randomised Showdown battles on top of nirkt's `MAEnv`: random map, brawlers, solo / duo / trio teams, policy-controlled slots vs the native AI, team placement reward, behaviour-slider reward terms. `python -m env_v5 --check` |
| `obs_v5.py` | The observation: `self` (8), `ent` (39 x 16 entity slots), `grid` (25 x 25 tile classes), `cond` (28: brawler stats, 10 slider weights, team size, cubes, players left). Per-episode detector noise, latency and HUD masking. `python -m obs_v5` |
| `fast_obs.py`, `fast_act.py` | Batched, vectorised versions of the observation builder and the actor for the multi-process trainer (bit-equal to the reference paths). |
| `model.py` | Policy: entity set + tile grid + conditioning -> GRU -> move (17) / fire (3) / aim (360, pointer over targets) / distance heads, critic with a privileged branch. `python -m model` |
| `train_ppo.py` | Behaviour cloning from the native AI, then PPO with a KL anchor, league / self-play, evals, replays. |
| `train_fast.py` | Same trainer with environment collection split over worker processes (shared memory). |
| `scorecard.py` | Skill metrics of a run from its logs: placement, win rate, trade, hit rate, aim lead, cube and gas stats. |
| `live_check.py` | One-boot check of the host round: team reward, area effects, generated maps, teammate respawn, HUD mask. |
| `test_fast.py` | Equality and timing of the fast paths against the reference paths on real battles. |
| `apply_noise.py` | Writes a measured detector-noise report into `obs_v5.NOISE` (with sanity limits). |
| `viz.html` | Run dashboard: training / eval curves, action histograms, battle replay with the policy's view. |
| `hud_ui.npy` | Measured share of each tile hidden by the game HUD (used by the noisy observation). |
| `env.sh` | Environment variables for the host, nirkt's packages and the work folder. |
| `host/fair_view.patch` | Camera-view change for nirkt's `vendor/nbpc/fair.py` (SETUP.md). |
| `pro_strategies.md` | Community Showdown habits and what each means for the observation / reward. |

## Requirements

- Linux or WSL2 (developed on Ubuntu 24.04), Python 3.11+, PyTorch (CUDA or ROCm for real runs; CPU is fine for the
  self-checks).
- nirkt's `nbrl-env` (its `py/` and `train/` folders) and a built `bulletlabels` host with this project's round
  applied. You supply the game files the host needs yourself. None are distributed here.

## Setup

The full list of pieces and how to build the host: **[SETUP.md](SETUP.md)**. Once the host is built:

```bash
export BULLETLABELS=/path/to/bulletlabels      # nirkt's host checkout, built
export NBRL=/path/to/nbrl-env                  # nirkt's nbrl-env checkout
export SIM_WORK=$HOME/bsrl-sim-work            # where runs/ and genmaps/ go
export SIM_VENV=/path/to/venv                  # optional: activated if it exists
source env.sh
```

Self-checks (need nirkt's `nbpc` on `PYTHONPATH`, which `env.sh` sets; no battle is booted):

```bash
python -m obs_v5 && python -m model && python -m fast_obs && python -m env_v5 --check && python apply_noise.py --test
```

Live checks (boot the host; each boot takes about 60-90 s):

```bash
python -m live_check --n 2 --minutes 8 --gen 1
python -m env_v5 --n 4 --minutes 2 --agents 10      # random-action benchmark
```

Training:

```bash
python -m train_ppo --run t --smoke                 # plumbing check: 2 instances, 3 updates
python -m train_fast --run my_run --hours 6 [flags]  # outputs $SIM_WORK/runs/my_run/
python scorecard.py my_run
cd $SIM_WORK/runs && python -m http.server 8766      # dashboard: http://localhost:8766/my_run/
```

`python -m train_ppo --help` lists every flag (sliders, teams, generated maps, league, KL anchor, reward weights).

## Notes

- Comments cite numbered entries of the project's private decision log (`HISTORY N`). They explain why a value is
  what it is; the log itself is not published.
- The fire head is none / attack / super: gadgets and hypercharge cannot be triggered on this host.
- Observation layout changes must be mirrored by anything that consumes exported policies. `obs_v5` carries the
  version (`v5.2`) that exported policies are checked against.

## Licence

GNU Affero General Public License v3.0, see [LICENSE](LICENSE). Credits: [NOTICE.md](NOTICE.md).
