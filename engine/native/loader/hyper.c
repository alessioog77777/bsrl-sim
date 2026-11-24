/* hypercharge round (bsrl-sim issue #4): the host's own hypercharge ("Overcharge" in libg) for the battle.
 *
 * What the shipped libg v68.279 (a client build) still has: the activation 0xc42584(player, char) (reached in the game by
 * input kind 17 and by the bot AI; the input path is gated off offline) and the charging inside the super charge 0xc420d8
 * (player+0x98 += damage-based charge x CharacterData OverchargeChargeMul, 0 while active: OverchargeChargeActiveMul).
 * The activation needs the brawler's hypercharge card (cards.csv MetaType 6) in the player config (+0x28, the slot 0xc33e84
 * fills for MetaType 6), a full charge (+0x98 >= +0x9c) and not already active (+0xa0); it sets +0xa0 = 1, char+0x68 = 1
 * (overcharged) and the drain per tick +0xa4 = 40 (divided by 1 + x/100 for an "extend" card modifier), and spends nothing.
 * What is compiled out (server code): the drain and the expiry, and the three buffs. Measured (2026-11-24, Shelly, host
 * without this file): speed, shot damage and damage taken identical active and inactive; active never ended (400+ ticks),
 * the charge stayed full.
 *
 * Here: equip (offline.c my_player_ctor puts hyper_card_for(character) in cfg+0x28 for the RESET's div.hyper_mask), activate
 * (NbeAction.fire 4 -> hyper_use -> libg's own 0xc42584), drain +0xa4 per tick from the charge and end at 0 (hyper_tick:
 * 4000 / 40 = 100 ticks = 5 s at 20 ticks/s, as in the real game), and the buffs from the brawler's characters.csv row while
 * active: OverchargeSpeedPercent on the move speed (env.c rec_move), OverchargeDamagePercent on damage dealt and
 * OverchargeShieldPercent off damage taken (env.c my_root). The state goes to Python in NbeResp hyp_* (every response).
 *
 * The hypercharged super (and the other "overcharged" variants). The game's skill code asks Character::isOvercharged 0xb35bf0
 * (~170 call sites in the skill / projectile code, e.g. 0xb82d34 latches it into the skill instance +0x31 when a skill
 * starts, and 0xb7adfc then returns the SkillData's OverchargedVersion, skills.csv) — in this build that function is a bare
 * `return false`. Here it returns true for a brawler while its hypercharge is active (hyper_install), so the game's own
 * code runs every Overcharged* column of skills.csv: OverchargedVersion (3 supers and ~25 main attacks in this table),
 * OverchargedProjectiles, OverchargedChainProjectile, OverchargedSummonedCharacters, OverchargedAreaEffectObject/2,
 * OverchargedSpawnedItem and the Overcharged*Effect visuals. Seven brawlers (Jester, Beamer, Twins, Skater, Domain, Dancer,
 * MechaDudeBig) have a separate super instead (characters.csv OverchargedUltimateSkill): the game builds it as the brawler's
 * third skill instance (char+0x458[2], 0xb34560) but what picks it for a super cast is compiled out; here instances 1
 * (the super) and 2 are swapped while hypercharged, so the game's own super code casts it, and swapped back once the
 * hypercharge ended and that super is idle (hyper_unswap). Meg's mech (MechaDudeBig, what MechaDude's super turns into)
 * takes Meg's card. NbeResp hyp_supers / hyp_super_id count the supers cast while hypercharged and name the SkillData of
 * the last one (hyper_super_data, from env.c fire_input_now).
 * The visual effects (client side) are whatever the client draws from +0xa0 and the overcharged skill data. */
#include "nbhost.h"
#include "nbenv_proto.h"
#include <string.h>

#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
#define U8(p, o) (*(uint8_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void **)((uint8_t *)(p) + (o)))
extern Module *g_libg;
#define B(off) ((uint8_t *)g_libg->base + (off))
void *div_data(int id);   /* env.c: data row by global id, NULL unless the table has it */
void *hook_replace(Module *m, uint64_t off, void *repl);

#define DRAIN_DEFAULT 40   /* +0xa4 that 0xc42584 sets without a duration modifier (4000 max / 40 = 100 ticks = 5 s) */

static void *player_cfg(void *p) {   /* the active loadout's config (player+0x30[player+0x40] -> +0x08); +0x28 hypercharge card */
    void **lo = PTR(p, 0x30); int li = I32(p, 0x40); void **ld = lo && li >= 0 && li < 8 ? lo[li] : NULL;
    return ld ? ld[1] : NULL;
}
static void *player_chardata(void *p) {
    void **lo = PTR(p, 0x30); int li = I32(p, 0x40); void **ld = lo && li >= 0 && li < 8 ? lo[li] : NULL;
    return ld ? ld[0] : NULL;
}
static int cd_pct(void *cd, unsigned off) { return cd ? ((int (*)(void *))B(off))(cd) : 0; }
/* CharacterData getters (characters.csv Overcharge* columns) */
#define CD_DAMAGE_PCT 0xab32a4
#define CD_SPEED_PCT 0xab32b0
#define CD_SHIELD_PCT 0xab32bc

/* the character's hypercharge card (cards.csv MetaType 6 whose target, card+0xd8, is this CharacterData), NULL if it has
 * none; cached per character (the cards table does not change while the host runs) */
void *hyper_card_for(void *chardata) {
    static struct { void *d, *card; } cache[128]; static int n;
    if (!chardata) return NULL;
    for (int k = 0; k < n; k++) if (cache[k].d == chardata) return cache[k].card;
    void *found = NULL;
    for (int idx = 0, miss = 0; idx < 20000 && miss < 64; idx++) {   /* table 23 = cards; stop after a run of missing rows */
        uint8_t *card = div_data(23000000 + idx);
        if (!card) { miss++; continue; }
        miss = 0;
        if (*(void **)(card + 0xd8) == chardata && I32(card, 0x108) == 6) { found = card; break; }
    }
    /* a brawler another one turns into (Meg's mech, MechaDudeBig, the SummonedCharacters of MechaDude's ChangeCharacter
     * super): the card of the brawler whose super it is (CharacterData +0x268 UltimateSkill, SkillData +0x80 / +0x8c) */
    for (int idx = 0, miss = 0; !found && idx < 20000 && miss < 64; idx++) {
        uint8_t *card = div_data(23000000 + idx);
        if (!card) { miss++; continue; }
        miss = 0;
        uint8_t *t = I32(card, 0x108) == 6 ? *(uint8_t **)(card + 0xd8) : NULL, *u = t ? PTR(t, 0x268) : NULL;
        void **sc = u ? PTR(u, 0x80) : NULL; int nsc = u ? I32(u, 0x8c) : 0;
        for (int k = 0; sc && k < nsc && k < 8; k++) if (sc[k] == chardata) { found = card; break; }
    }
    if (n < 128) { cache[n].d = chardata; cache[n].card = found; n++; }
    return found;
}

typedef struct { int card, used, dmg_pct, speed_pct, shield_pct, supers, super_id; } Hyp;
static Hyp g_hyp[NBE_MAXP];
static uint8_t *g_oc[NBE_MAXP];   /* player i's brawler while its hypercharge is active (what isOvercharged answers for) */
static uint8_t *g_sw[NBE_MAXP];   /* player i's brawler whose skill instances 1 / 2 are swapped (OverchargedUltimateSkill) */

void hyper_clear(void) { memset(g_hyp, 0, sizeof g_hyp); memset(g_oc, 0, sizeof g_oc); memset(g_sw, 0, sizeof g_sw); }

/* Character::isOvercharged 0xb35bf0 (a bare `mov w0, wzr; ret` in this build): true for a brawler whose hypercharge is on.
 * Only the brawlers hyper_use activated answer true (not a read of char+0x68: some call sites may pass other objects). */
static int my_is_oc(void *ch) {
    if (!ch) return 0;
    for (int i = 0; i < NBE_MAXP; i++) if (g_oc[i] == ch && U8(ch, 0x68)) return 1;
    return 0;
}
void hyper_install(void) {
    uint32_t *f = (uint32_t *)B(0xb35bf0);
    if (f[0] != 0x2a1f03e0 || f[1] != 0xd65f03c0) {   /* mov w0, wzr ; ret */
        LOG("[hyper] isOvercharged 0xb35bf0: unexpected bytes %08x %08x, hypercharged skills off\n", f[0], f[1]); return;
    }
    hook_replace(g_libg, 0xb35bf0, (void *)my_is_oc);
}

/* The OverchargedUltimateSkill is the brawler's third skill instance (char+0x458[2]); the game's super code uses instance 1.
 * While hypercharged the two are swapped, so every super path (input, charge, the skill's own timers) runs the hypercharged
 * one with the game's own code; swapped back once the hypercharge ended and the hypercharged super is not running. */
static void swap_ulti(uint8_t *ch) {
    uint8_t **sk = PTR(ch, 0x458); uint8_t *t = sk[1]; sk[1] = sk[2]; sk[2] = t;
}
static int has_oc_ulti(uint8_t *ch) {
    uint8_t **sk = ch ? PTR(ch, 0x458) : NULL; int n = ch ? I32(ch, 0x464) : 0;
    return sk && n >= 3 && n <= 16 && sk[1] && sk[2];
}
/* the super player i casts now (sd = its SkillData, SKILL_SUPER): counted in hyp_supers / hyp_super_id while hypercharged */
void *hyper_super_data(int i, uint8_t *ch, void *sd) {
    if (i < 0 || i >= NBE_MAXP || !sd || !my_is_oc(ch)) return sd;
    g_hyp[i].supers++; g_hyp[i].super_id = I32(sd, 0x20);
    static int nlog; if (nlog++ < 40) LOG("[hyper] player %d hypercharged super: data %d%s\n", i, I32(sd, 0x20), g_sw[i] == ch ? " (OverchargedUltimateSkill)" : "");
    return sd;
}
/* after every tick: a swapped brawler whose hypercharge is over gets its normal super back once the swapped one is idle
 * (skill instance +0x30 = running; char+0x6d8 2 = a charge of instance 1 in progress) */
void hyper_unswap(int i, uint8_t *ch) {
    if (i < 0 || i >= NBE_MAXP || !g_sw[i] || g_oc[i]) return;
    if (ch != g_sw[i]) { g_sw[i] = NULL; return; }   /* the brawler is gone (a new object): nothing to undo */
    uint8_t **sk = PTR(ch, 0x458);
    if (!has_oc_ulti(ch) || U8(sk[1], 0x30) || I32(ch, 0x6d8) == 2) return;
    swap_ulti(ch); g_sw[i] = NULL;
}

/* RESET response (env.c div_publish), per player: the card the game's config holds and the brawler's buff percents */
void hyper_reset_player(int i, void *p) {
    if (i < 0 || i >= NBE_MAXP) return;
    Hyp *h = &g_hyp[i]; memset(h, 0, sizeof *h);
    void *cfg = p ? player_cfg(p) : NULL, *card = cfg ? PTR(cfg, 0x28) : NULL, *cd = p ? player_chardata(p) : NULL;
    if (!card) return;
    h->card = I32(card, 0x20);
    h->dmg_pct = cd_pct(cd, CD_DAMAGE_PCT); h->speed_pct = cd_pct(cd, CD_SPEED_PCT); h->shield_pct = cd_pct(cd, CD_SHIELD_PCT);
    static int nlog; if (nlog++ < 20) LOG("[hyper] player %d card %d: damage +%d%%, speed +%d%%, shield %d%%\n", i, h->card,
                                          h->dmg_pct, h->speed_pct, h->shield_pct);
}

int hyper_active(void *p) { return p && U8(p, 0xa0); }

/* NbeAction.fire 4 for player i (its brawler ch, alive): libg's activation; -> 1 if the hypercharge is now active and was not */
int hyper_use(int i, void *p, uint8_t *ch) {
    if (i < 0 || i >= NBE_MAXP || !p || !ch || !g_hyp[i].card || U8(p, 0xa0) || I32(p, 0x98) < I32(p, 0x9c)) return 0;
    ((void (*)(void *, void *))B(0xc42584))(p, ch);
    if (!U8(p, 0xa0)) return 0;   /* refused by the game's own checks (e.g. a state that forbids it) */
    g_hyp[i].used++; g_oc[i] = ch;
    if (has_oc_ulti(ch) && g_sw[i] != ch) { swap_ulti(ch); g_sw[i] = ch; }
    static int nlog; if (nlog++ < 40) LOG("[hyper] player %d activated (charge %d, drain %d/tick)\n", i, I32(p, 0x98), I32(p, 0xa4));
    return 1;
}

/* after every tick, per player i (ch = its brawler or NULL): drain the charge; -> 1 if the hypercharge ended at this tick */
int hyper_tick_player(int i, void *p, uint8_t *ch, int alive) {
    if (!p || !U8(p, 0xa0)) return 0;
    int drain = I32(p, 0xa4) > 0 ? I32(p, 0xa4) : DRAIN_DEFAULT, c = I32(p, 0x98) - drain;
    if (c > 0 && alive) { I32(p, 0x98) = c; return 0; }
    I32(p, 0x98) = alive ? 0 : (c > 0 ? c : 0); U8(p, 0xa0) = 0;   /* ended (charge used up, or the brawler died) */
    if (ch) U8(ch, 0x68) = 0;
    if (i >= 0 && i < NBE_MAXP) g_oc[i] = NULL;
    return 1;
}

/* damage through the root 0xb422d0: d > 0 dealt by player ai's brawler to player vi's brawler (-1 = not a player's brawler) */
int hyper_damage(void *pa, int ai, void *pv, int vi, int d) {
    if (d <= 0) return d;
    if (ai >= 0 && ai < NBE_MAXP && hyper_active(pa) && g_hyp[ai].dmg_pct) d = d * (100 + g_hyp[ai].dmg_pct) / 100;
    if (vi >= 0 && vi < NBE_MAXP && hyper_active(pv) && g_hyp[vi].shield_pct) d = d * (100 - g_hyp[vi].shield_pct) / 100;
    return d;
}

/* the move speed (units/s) player i's brawler moves with now, 0 = unchanged (not hypercharged) */
int hyper_speed(int i, void *p, uint8_t *ch) {
    if (i < 0 || i >= NBE_MAXP || !hyper_active(p) || !g_hyp[i].speed_pct || !ch) return 0;
    void *cd = ((void *(*)(void *))B(0xb6a384))(ch);   /* the brawler's CharacterData (what the mover reads its speed from) */
    int base = cd ? ((int (*)(void *))B(0xab31e0))(cd) : 0;
    return base > 0 ? base * (100 + g_hyp[i].speed_pct) / 100 : 0;
}

void hyper_publish(NbeResp *r, void **ps, int np) {   /* every response */
    r->hyp_valid = 1;
    for (int i = 0; i < NBE_MAXP; i++) {
        const Hyp *h = &g_hyp[i]; void *p = i < np ? ps[i] : NULL;
        int on = hyper_active(p), drain = p && I32(p, 0xa4) > 0 ? I32(p, 0xa4) : DRAIN_DEFAULT;
        r->hyp_card[i] = h->card; r->hyp_used[i] = h->used;
        r->hyp_left[i] = on ? (I32(p, 0x98) + drain - 1) / drain : 0;
        r->hyp_damage_pct[i] = h->card ? h->dmg_pct : 0; r->hyp_speed_pct[i] = h->card ? h->speed_pct : 0;
        r->hyp_shield_pct[i] = h->card ? h->shield_pct : 0;
        r->hyp_supers[i] = h->supers; r->hyp_super_id[i] = h->super_id;
    }
}
