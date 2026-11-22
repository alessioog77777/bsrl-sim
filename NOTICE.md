# Credits

**This sim exists because of [nirkt](https://github.com/nirkt).** nirkt built the battle host that runs the game's battle logic headless, fast
enough for reinforcement learning, and the multi-agent environment around it. Everything in this repository sits on
top of that work.

From nirkt:
- **[bulletlabels](https://github.com/nirkt/bulletlabels)**: the battle host (`native/loader`), the patched `qemu-aarch64` it runs under, and the
  `vendor/nbpc` package (instances, observation encoding, camera view rules in `fair.py`, the `MAEnv` glue).
- **[nbrl-env](https://github.com/nirkt/nbrl-env)**: the multi-agent battle environment (`MAEnv`) that `env_v5.py` drives, and the `models4` building
  blocks `model.py` imports (`UnfoldConv2d`, masked log-prob / entropy / sampling).
- Design ideas used here: the PointerAim aim head, the behaviour-cloning-then-PPO recipe with a decaying KL anchor, and
  the native-AI action labelling rules (`collect_demos4`) that `env_v5.Env` follows.

nirkt's code is not included in this repository and stays under nirkt's own terms. `host/fair_view.patch` is this
project's change to nirkt's `vendor/nbpc/fair.py` and is shipped as a patch against it. See [SETUP.md](SETUP.md) for
how the pieces fit together.

Everything else in this repository is the BSRL project's own code, licensed under the AGPL-3.0 (see LICENSE).
