# Issue #21 leftovers (storm, modifiers, WallSpawn shape): working notes

Branch claude/project-thread-y36m53 from main fe49fe2. Other sessions own gadget.c, hyper.c, env_v5.py / star-power files: don't touch.

## Plan
1. Storm (env.c gas_tick): Trio ramps after 1 s (Solo / Duo 5 s); healing inside the gas halved (all sources).
   Knob NB_GAS_RULES=0 restores the phone agent's storm.
2. Modifiers (new showdown_mods.c, tick job from pre_snapshot): NB_SD_MODIFIERS="3,5" (event_modifiers.csv ids:
   3 Meteors, 5 HealingMushrooms). Default off. Env-side damage / heal (no native area: an area needs an owner and then
   filters by its team).
3. spawnfix.c WallSpawn: tiles whose centre is < Radius from the area's tile centre (Rosa's BushGrowArea R 600 -> 3x3,
   as the wiki says). Bush-grow areas -> ExtraBush (DynamicCode 3).

## Sources
- area_effects_logic: EventExplosion (row 37) TimeMs 700 R 1200 Damage 2000 DestroysEnvironment, pushback 150;
  EventHealing (row 44) Hot TimeMs 5000 R 1000 Damage -1000. WallSpawn: WallyUltiArea 77 R 400, WallyOvercharged 79 R 400,
  BushGrowArea 81 R 600, MutantBushGrowArea 260 R 500, TankBushGrowArea 499 R 100, TankBushGrowAreaLarge 500 R 500.
- tiles.csv: ExtraBush code 3 (no Lifetime), ExtraBushTemp 25, Wally walls Lifetime 200.
- fandom zh Modifiers page: meteor 2000 dmg + knockback + destroys obstacles, red circle 3 s before impact, one at a time,
  increasing frequency, prefers grouped brawlers. Mushrooms heal 1000/s in a ring, near lone players, increasing frequency,
  last "a couple of seconds". Rosa wiki: Grow Light = 3x3 bushes, stay until destroyed.
- #22 comment (alessioog77777, wiki + patch notes): Trio ramp after 1 s; gas heal -50 % since 04/09/24; depth damage
  since 22/04/26 (amount unknown).

## Still guessed
- Storm shrink schedule (tick 440, 1 tile / 100 ticks), depth damage.
- Meteor / mushroom intervals and first spawn; mushroom grow time.

## Live results (system qemu, native AI)
- showdown_check --n 2 --minutes 6 --mods 3,5: 158 ends (39 solo / 57 duo / 62 trio), all within 0 ticks of one team
  left; cube drops 1661 of 1788 exact; 621 of 621 respawns protected. Storm: 697 hits logged, all on gas_pct (96 Trio hits
  past the 1 s ramp), 58 heal cuts. Meteors: 1050 impacts, 1218 hits: 1078 exact 2000, 39 lethal, 26 protected, 63 knocked
  out into respawn (back to full hp), 12 changed by the victim's damage modifiers. Mushrooms: 1260, 3641 heals <= 1000.
- super_audit --only Wally (Sprout): works; area data id = area_effects_skin row (17000292), radius 400, 2 tiles placed
  (the rest of the plus was not open ground).
- Bush mapping: no live trigger (Rosa's Grow Light is not in gadget.c's AREA kind yet); static only.

## State
- Merged in #35. All host checks passed; see the PR body.
