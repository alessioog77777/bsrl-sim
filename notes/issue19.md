# Issue #19 (status effects and movement): running notes

Branch `claude/project-thread-2z4j2e`, on main fac4e76 (includes #18's superfix status store).

## Facts (static reading of libg v68.279)
- 0xb47358 (`mov w0, wzr; ret`) is not a push: both callers (area apply 0xb29d80, projectile hit 0xb78054) first check
  the owner's card type 46 = `stun_trap` (Bo's star power, Value 40 ticks). Args: x0 victim, w1 ticks, w2..w5 0; return
  unused. So it is a stun. Area / projectile pushback already go through the real knockback 0xb49834 (area 0xb2a718,
  strength data +0x154; projectile 0xb7816c).
- 0xb35c3c (`ret`) is a general change-skill(ch, slot 0 weapon / 1 super, new SkillData, w3, w4, w5), 24 call sites.
  Draco: dismount 0xba0178 (ret addr 0xba017c) passes CharacterData +0x260 (DragonRiderWeapon); mount 0xba02e0 (ret
  0xba02e4) passes the weapon SkillData +0x158 (SecondarySkill = DragonRiderDragonWeapon).
- Skill instances: char +0x458 array, +0x464 count; new(0x58) 0x11a4d90 + ctor 0xb7aa98(inst, data).
- Draco's mount speed: MountUp component data +0xb8 = DragonRiderUltiSpeed (SpeedBoostPercent 20, shield 20, 2 ticks,
  re-added every tick). Base 720 -> 864 mounted.

- Projectile stun length: ProjectileData +0x1d0 (StunLengthMS / 50), read nowhere. The hit code at 0xb73eb8 is not
  reached for Frank's super hits (no hook call in a live run), so the stun is applied from env.c's damage root hook
  (0xb422d0, x6 = source: a projectile, VT 0x1209318).
- Bea's super (BeeSniperUltiProjectile, StatusEffectEnemy BeeSniperUltiProjectileSlow -350 u/s 60 ticks) is added
  through the projectile hit's status add 0xb73cd8 -> 0xba3230, i.e. our store.
- Building: build.sh writes nbhost_spawnfix2; nbhost_rl comes from `sed 's/nbhost_spawnfix2/nbhost_rl/g' build.sh | sh`
  (setup.sh). env.sh runs nbhost_rl.

## Decisions
- Status speed: v = base * (100 + boost %) / 100 + boost abs, then * (100 - reduce %) / 100 - reduce abs, floor 0
  (order is ours; the client has no formula). Speed 0 = rooted (move target = own position).
- Damage boost / reduce applied in env.c's my_root (attacker DamageBoostPercent - DamageReducePercent, victim
  DamageReceivedBoost % / abs), not through libg's char +0x968 / +0x970 (their mapping is not proven).
- Knockback outside charges already works (area 0xb2a718 and projectile 0xb7816c call the real 0xb49834); the stub
  that looked like a push (0xb47358) is Bo's stun trap and is now a stun.
- Stunned brawlers send no input (host gadgets would otherwise still fire); silenced ones no super / gadget /
  hypercharge.

## Open
- RequireLineOfSight and stun area shapes other than circles are ignored (range = 0xb2b298).
- Draco's ammo while mounted is the new instance's (full); Taunt / Fear / LifeSteal / reload statuses not modelled.
- Bo's trap stun is not covered by a live check.
- Bea's super bees (BeeSniperUltiProjectile, TravelType 3) spawn ~200 u in front of Bea, never move and vanish after
  ~6 ticks (live probe, targets at 600 and 1200 u; super_audit still classes her super as works, so it may only
  reach point blank). Not a status bug; a separate issue. The
  check uses Spike's super slow instead.
