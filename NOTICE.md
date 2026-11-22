# Credits

This project is built on **nirkt**'s work. Their code is not included here. It is a separate dependency.

- **nbrl-env** (nirkt): the multi-agent battle environment (`MAEnv`, `nbpc` package) that `env_v5.py` drives, the
  `models4` building blocks `model.py` imports (`UnfoldConv2d`, masked log-prob / entropy / sampling), the PointerAim
  aim head design, the BC-then-PPO recipe with a decaying KL anchor, and the native-AI action labelling rules
  (`collect_demos4`) that `env_v5.Env` follows.
- **bulletlabels** (nirkt): the battle host and its `vendor/nbpc` package (camera view rules in `fair.py`, instance
  management). This project's training round adds team groups, extra observation fields, generated maps and
  teammate respawn on top of it. Those host changes are not part of this repository.

Everything else in this repository is the BSRL project's own code, licensed under the AGPL-3.0 (see LICENSE).
