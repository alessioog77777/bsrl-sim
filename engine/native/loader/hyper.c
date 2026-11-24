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
 * NOT done: the hypercharged super (characters.csv OverchargedUltimateSkill / skills.csv OverchargedVersion, 7 brawlers in
 * this table): the super stays the normal one. The visual effects (client side) are whatever the client draws from +0xa0. */
#include "nbhost.h"
#include "nbenv_proto.h"
#include <string.h>

#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
#define U8(p, o) (*(uint8_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void **)((uint8_t *)(p) + (o)))
extern Module *g_libg;
#define B(off) ((uint8_t *)g_libg->base + (off))
void *div_data(int id);   /* env.c: data row by global id, NULL unless the table has it */

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
    if (n < 128) { cache[n].d = chardata; cache[n].card = found; n++; }
    return found;
}

typedef struct { int card, used, dmg_pct, speed_pct, shield_pct; } Hyp;
static Hyp g_hyp[NBE_MAXP];

void hyper_clear(void) { memset(g_hyp, 0, sizeof g_hyp); }

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
    g_hyp[i].used++;
    static int nlog; if (nlog++ < 40) LOG("[hyper] player %d activated (charge %d, drain %d/tick)\n", i, I32(p, 0x98), I32(p, 0xa4));
    return 1;
}

/* after every tick, per player (ch = its brawler or NULL): drain the charge; -> 1 if the hypercharge ended at this tick */
int hyper_tick_player(void *p, uint8_t *ch, int alive) {
    if (!p || !U8(p, 0xa0)) return 0;
    int drain = I32(p, 0xa4) > 0 ? I32(p, 0xa4) : DRAIN_DEFAULT, c = I32(p, 0x98) - drain;
    if (c > 0 && alive) { I32(p, 0x98) = c; return 0; }
    I32(p, 0x98) = alive ? 0 : (c > 0 ? c : 0); U8(p, 0xa0) = 0;   /* ended (charge used up, or the brawler died) */
    if (ch) U8(ch, 0x68) = 0;
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
    }
}
