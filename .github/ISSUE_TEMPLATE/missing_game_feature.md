---
name: Missing or broken game feature
about: Something that works in the real game but not in the sim (a super, gadget, hypercharge, area effect, ...)
labels: help wanted
---

## What doesn't work
<!-- Brawler(s), ability, what the real game does, what the sim does. Output of `python -m super_audit --only <Name>` helps. -->

## Where it breaks
<!-- If known: the libg address that is a bare `ret` stub, the host function involved (deps/bulletlabels/native/loader/*.c),
     or "unknown". The shipped libg.so is a client build: server-only code is compiled out (see
     deps/bulletlabels/docs/OPEN_ISSUES.md and the spawnfix.c header). -->

## How to verify a fix
<!-- A live check that fires the ability and asserts its effect (damage, displacement, spawned object, status, charge consumed). -->

## Contract changes
<!-- Does the fix change the action or observation? Then bump obs_v5.OBS_VERSION and keep old checkpoints loading (AGENTS.md). -->

Changes under `deps/` are nirkt's code: keep them small and record every one in `host/*.patch` and SETUP.md (AGENTS.md).
