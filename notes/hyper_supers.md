# Hypercharged supers that fail only hypercharged (issue #32 item 3, #6)

Resume notes for this branch. Scope: `engine/native/loader/hyper.c` (+ its `engine/patches/*.patch`).

Setup on a fresh box: `git lfs pull` in bsrl-game (else the host segfaults at "loading 45%"), link or unpack the game into
`engine/game`, `./setup.sh`. Check one brawler: `python -m super_audit --hyper --only <Name>`.

Main fe49fe2, all 7 partial (`super_audit --hyper`):
- Edgar `Enrager`: areas EnragerOverchargedUlti x3 (type EnragerOverchargeSuper), no damage.
- Sandy `Sandstorm`: OC projectile + 3 OC areas, no damage.
- Hank `FishTank`: OC area + 6 homing hankOverchargedUltiProjectile, no damage.
- Belle `ElectroSniper`: OC homing projectile, no damage.
- Pam `MinigunDude`: OC projectile and heal area, but the normal HealingStation summon; no heal.
- Gigi `Daredevil`: OC damage area, no damage.
- Lumi `Morningstar`: normal projectiles 1-3, not the Overcharged list.
Leads: Belle and Hank are the only OC projectiles with TravelType 4 (homing) besides Doug's (heal, works).

## Causes found (fixes in this branch)
- Belle, Hank: homing (TravelType 4) projectiles move to proj+0x94/+0x98, which only the compiled-out steer code writes, so
  they flew to (0,0). hyper.c `on_travel` (enter hook on 0xb709e4) steers them. Changes every homing projectile.
- Lumi: AttackPattern 11's per-attack pick (0xb4b978) reads only Projectiles. hyper.c `hyper_proj_site`.
- Pam: the Attack handler reads the normal summon (8 bl's to 0xb0ddf8). hyper.c `hyper_summon_bl`. Instant heal: hyper.c `oc_heal_areas` (Heal areas carry a negative amount the area apply skips).
- Gigi: the OC enter area is a Damage area with no Damage: superfix_moves.c passes the scaled damage.
- Sandy, Edgar: audit expectation (their hypercharged supers deal no damage, texts.csv): super_audit.py NO_DAMAGE_AREAS.
  Edgar's real effect (reload / charge buff, area type EnragerOverchargeSuper) is compiled out and not built.
## Status
- all 7 work in `super_audit --hyper --only ...`; normal supers unchanged; hyper_check passes. Full --hyper: 95 of 115 work.
- Hank misses at close range (450-600 u) in a full run: the homing turn rate (SteerStrength as degrees per tick) is a guess.
- PR open for review.
