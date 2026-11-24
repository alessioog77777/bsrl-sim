# Issue #21: Showdown mode logic (working notes)

Branch claude/project-thread-a2aisf from main fac4e76. Game files: /home/user/bsrl-game symlinked as engine/game (.git/info/exclude).
Setup: pytorch.org and the qemu source download are blocked by the proxy here: torch from PyPI, system qemu-aarch64 (2x slower).

## Done (code, not yet verified live)
- Team end: env.c elimination_tick ends the battle when one team is left (g_div_team set; teams = player +0xc), not only at
  <= 1 alive player. winner = the last team's lowest-index player, so MAEnv ranks it 1 and env_v5.team_ranks gives the team 1.
  local_rank = 1 for the local player on that team.
- Spawned walls: they already expire natively (the setter puts them on the dynamic tile list). Sprout audit
  (super_audit --only Wally --watch 300, NB_SPAWNFIX_DBG=1): 13 of 13 ring tiles open again between placed + 199 and + 220
  (tiles.csv Lifetime 200). A backstop was written, never fired, and was removed; only a comment in spawnfix.c.

## Data found
- area_effects_logic WallSpawn rows: WallyUltiArea R 400 T 500, BushGrowArea R 600, MutantBushGrowArea R 500, TankBushGrow* R 100/500.
- The ring shape (2 tiles out) is still a guess; not changed.

- Cube drops (subagent, static): libg's death drop is in the damage root 0xb422d0 (from 0xb44f20), not stubbed: hero with a
  player at hp 0 drops min(+0x874/3 + 1, MaxPowerCubesToDrop) DroppedItemOnDeath items (variation record sim+0x108 +0x2c8 /
  +0x318), spawner 0xb55b74 (angle random, radius 300-600 with LOS + spacing, then fallbacks). +0x874 was never filled, so every
  death dropped 1 cube. Fix: env mirrors its count into +0x874 (item_tick, 0 at a respawn knock-out); trio groups capped at 3
  (Trio max 2; the sim is variation 6, max 5). cube84_w shows max(+0x874, env) instead of the sum.
- Spawn protection: every char tick sets +0x431 = (+0x4f0 > tick) (0xb3995c); root blocks damage while set. Battle start:
  +0x4f0 = tick + intro 182 + 60. Respawn: +0x4f0 = tick + 60, +0x431 = 1 (60 is a guess = the post-intro value).
- Respawn delay 15 s -> 17 s (game_mode_variations RespawnSeconds 17 for Duo / Trio).
- showdown_check.py: --check (no host) and live (native AI, solo/duo/trio): end lag, drops vs drop rule, protection flag.

## Open
- Bush-grow mapping (code 3) needs the area data id -> name; not done.

## Live results (showdown_check, native AI, 2 instances)
- 12 min run: 345 endings (86 solo, 136 duo, 123 trio), every one within 0 ticks of the tick one team was left, winner on the
  last team every time; 1518 of 1518 respawns came back protected. Mean end tick ~2250 for duo / trio.
- 5 min run after excluding wiped waiting players (no body) and DROP_R 1300: 1548 of 1647 falls left exactly
  min(k/3+1, 5 or 2) cubes (94 %); misses are mostly 0 seen (likely picked up within the step) or 2 seen (a box's cube nearby).
- Without the mirror (main), every death should drop exactly 1 cube (+0x874 = 0; from the static read, not measured on main).

## State
- PR #26 open (not merged), commit pushed on claude/project-thread-a2aisf, subscribed to its activity.
- Not done from #21: storm numbers, event modifiers, WallSpawn shape from Radius, bush-grow mapping.
