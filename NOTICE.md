# Credits

**This sim exists because of [nirkt](https://github.com/nirkt).** nirkt built the battle host that runs the game's
battle logic headless, fast enough for reinforcement learning, and the multi-agent environment around it. Everything
in this repository sits on top of that work.

nirkt's repositories are included as git submodules under `deps/`. They are referenced, not copied: their code stays
nirkt's, under nirkt's own terms, and comes straight from nirkt's GitHub at the pinned commits.

- **[bulletlabels](https://github.com/nirkt/bulletlabels)** (`deps/bulletlabels`): the battle host
  (`native/loader`), the patched `qemu-aarch64` it runs under, and the `vendor/nbpc` package (instances, observation
  encoding).
- **[nbrl-env](https://github.com/nirkt/nbrl-env)** (`deps/nbrl-env`): the multi-agent battle environment (`MAEnv`)
  that `env_v5.py` drives, the camera view rules (`fair.py`), and the `models4` building blocks `model.py` imports
  (`UnfoldConv2d`, masked log-prob / entropy / sampling).
- **[nb-offline-play](https://github.com/nirkt/nb-offline-play)**: nirkt's single-window offline player (emulator,
  renderer, game host). Not needed for training, credited as the related project.
- Design ideas used here: the PointerAim aim head, the behaviour-cloning-then-PPO recipe with a decaying KL anchor, and
  the native-AI action labelling rules (`collect_demos4`) that `env_v5.Env` follows.

`host/training_round.patch` and `host/fair_view.patch` are this project's changes to nirkt's files. They ship as
patches against the pinned commits and `setup.sh` applies them.

Everything else in this repository is the BSRL project's own code, licensed under the AGPL-3.0 (see LICENSE).
