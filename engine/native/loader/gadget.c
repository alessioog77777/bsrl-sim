/* gadget round (bsrl-sim issue #3): the host's own gadget activation.
 *
 * The shipped libg v68.279 is a client build: the gadget activation that input kind 8 reaches (NbeAction.fire 3 -> env.c
 * gadget_input -> 0xc2cef4 -> 0xb27c58(player+0x190, char, x, y, 0)) is a bare ret, and with it the server's bookkeeping of
 * uses and cooldown. Here the host keeps that state per player for the battle and runs a re-implemented activation per
 * gadget kind (NBE_GAD_*), the way spawnfix.c re-implements the compiled-out children of explosions with libg's own functions.
 * The state goes to Python in NbeResp gad_* (every response); bsrl-sim's env_v5.Env.gadget turns it into obs_v5 self 3.
 *
 * What the game gives the host: every player with a gadget card has an accessory object at player+0x190 (0xe0 B, built by
 * 0xb2731c(acc, AccessoryData, player)): +0x10 its AccessoryData (accessories.csv row), +0x24 the type enum the constructor
 * parses from the Type column (2 dash, 8 heal, ...), +0xc4 the cooldown in ticks (the row's Cooldown, scaled by gear).
 * AccessoryData columns are read with libg's own getters (SubType 0xa93258, ActiveTicks 0xa932c8, CustomValue1/2
 * 0xa933d0 / 0xa933e0).
 *
 * Implemented kinds (RE-IMPLEMENTED, not the game's server code, which is compiled out):
 *   DASH  type 2 (the 7 dash rows: Cannon Girl, Maisie, Twins, Crab, Digger, Lightyear, Skater): the brawler moves
 *         CustomValue1 units (600..1300) along the aim at CustomValue2 u/s (2000..3500) with the native charge mover
 *         (env.c env_dash_start). Movement only: a row's extras (Maisie's damage area, Skater's taunt, CustomValue3/4) are
 *         not applied, and the "requires ..." conditions (MissingTargetText) are not checked.
 *   HEAL  type 8, SubType 2 (Bibi's Vitamin Booster, Byron's Shot in the Arm): heals CustomValue1 % of max hp per second
 *         (the card texts say "per second for N seconds") for ActiveTicks ticks, at most up to max hp, ended by the
 *         brawler's death. Byron's ammo cost (ConsumesAmmo) is not applied. The other heal SubTypes need state the host
 *         does not have (Ash's rage, Meg's mech, Sandy's sleep, area heals) and stay unimplemented.
 * Every other gadget publishes kind NONE (not usable): skill gadgets (Type empty + Skill: the gadget's own skill, fired
 * through any input kind or installed as the brawler's 4th skill, spawns nothing in this build), next_attack_change, spawn,
 * shields, ... Each player gets GAD_USES uses per battle and the accessory's cooldown between uses.
 *
 * To add a kind: write classify (accessory -> 1 if this implementation handles it) and fire (the effect at the brawler, with
 * libg's own functions where they exist), add the entry to g_impl, then run `python -m gadget_check` (bsrl-sim) on the built
 * host: it fires each published kind and asserts that a use was spent and that the kind's effect happened. */
#include "nbhost.h"
#include "nbenv_proto.h"
#include <math.h>
#include <string.h>

#define I32(p, o) (*(volatile int32_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void *volatile *)((uint8_t *)(p) + (o)))
#define B(off) ((uint8_t *)g_libg->base + (off))
extern Module *g_libg;
extern int32_t g_div_gadget[NBE_MAXP];   /* env.c: the RESET's requested gadget cards (offline.c puts them in the configs) */
void *env_obj_by_id(void *sim, int id);  /* env.c */
int env_dash_start(uint8_t *ch, int tx, int ty, int speed);   /* env.c: -> path length, 0 = no dash */

#define GAD_USES 3        /* Brawl Stars: 3 gadget uses per battle (the count lives in server code, not in this build) */
#define TICKS_PER_S 20    /* 50 ms ticks */
enum { ACC_DASH = 2, ACC_HEAL = 8 };   /* accessory +0x24 type enum (0xb2731c's Type parser) */

typedef int (*AdGet)(void *ad);
static int ad_get(void *ad, uint64_t getter) { return ((AdGet)B(getter))(ad); }
#define AD_SUBTYPE 0xa93258
#define AD_ACTIVE_TICKS 0xa932c8
#define AD_CV1 0xa933d0
#define AD_CV2 0xa933e0

typedef struct Gad Gad;
typedef struct {
    int kind;                                                                     /* NBE_GAD_* */
    int (*classify)(void *ad, int type);                                          /* 1 = handles this accessory */
    int (*fire)(void *sim, Gad *g, uint8_t *ch, double deg, int tick);           /* 1 = activated (a use is spent) */
} GadImpl;
struct Gad {
    int card, uses, cd, cd_until, used; void *ad; const GadImpl *im;
    int hot_id, hot_start, hot_end, hot_total, hot_done;   /* heal over time: char id, ticks, hp to give, hp given */
};
static Gad g_gad[NBE_MAXP];
static long g_fired[8];

/* ---- DASH */
static int dash_classify(void *ad, int type) { return type == ACC_DASH && ad_get(ad, AD_CV1) > 0 && ad_get(ad, AD_CV2) > 0; }
static int dash_fire(void *sim, Gad *g, uint8_t *ch, double deg, int tick) {
    (void)sim; (void)tick;
    int dist = ad_get(g->ad, AD_CV1), spd = ad_get(g->ad, AD_CV2), X = I32(ch, 0x38), Y = I32(ch, 0x3c);
    double r = deg * M_PI / 180;
    return env_dash_start(ch, (int)floor(X + dist * cos(r) + 0.5), (int)floor(Y + dist * sin(r) + 0.5), spd) > 0;
}

/* ---- HEAL (over time) */
static int heal_classify(void *ad, int type) {
    return type == ACC_HEAL && ad_get(ad, AD_SUBTYPE) == 2 && ad_get(ad, AD_CV1) > 0 && ad_get(ad, AD_ACTIVE_TICKS) > 0;
}
static int heal_fire(void *sim, Gad *g, uint8_t *ch, double deg, int tick) {
    (void)sim; (void)deg;
    int dur = ad_get(g->ad, AD_ACTIVE_TICKS), maxhp = I32(ch, 0x3c8);
    if (I32(ch, 0x3c4) <= 0 || maxhp <= 0) return 0;
    g->hot_id = I32(ch, 8); g->hot_start = tick; g->hot_end = tick + dur; g->hot_done = 0;
    g->hot_total = (int)floor((double)maxhp * ad_get(g->ad, AD_CV1) / 100.0 * dur / TICKS_PER_S + 0.5);
    return 1;
}
static void heal_tick(void *sim, Gad *g, int tick) {
    uint8_t *ch = env_obj_by_id(sim, g->hot_id);
    if (!ch || I32(ch, 0x3c4) <= 0 || tick <= g->hot_start) { if (!ch || I32(ch, 0x3c4) <= 0) g->hot_end = 0; return; }
    int t = tick < g->hot_end ? tick : g->hot_end, dur = g->hot_end - g->hot_start;
    int due = (int)floor((double)g->hot_total * (t - g->hot_start) / dur + 0.5), d = due - g->hot_done;   /* spread evenly */
    g->hot_done = due;
    int hp = I32(ch, 0x3c4), mx = I32(ch, 0x3c8);
    if (d > 0 && hp < mx) I32(ch, 0x3c4) = hp + d < mx ? hp + d : mx;
    if (tick >= g->hot_end) g->hot_end = 0;
}

static const GadImpl g_impl[] = {
    {NBE_GAD_DASH, dash_classify, dash_fire},
    {NBE_GAD_HEAL, heal_classify, heal_fire},
    {NBE_GAD_NONE, NULL, NULL}};

void gadget_clear(void) { memset(g_gad, 0, sizeof g_gad); }

/* RESET response (env.c div_publish), per player: its requested card and the accessory object the game made for it */
void gadget_reset_player(int i, void *acc) {
    if (i < 0 || i >= NBE_MAXP) return;
    Gad *g = &g_gad[i]; memset(g, 0, sizeof *g);
    if (!acc || !g_div_gadget[i]) return;
    g->card = g_div_gadget[i]; g->ad = PTR(acc, 0x10);
    int type = I32(acc, 0x24);
    for (const GadImpl *im = g_impl; g->ad && im->classify; im++)
        if (im->classify(g->ad, type)) { g->im = im; g->uses = GAD_USES; g->cd = I32(acc, 0xc4) > 0 ? I32(acc, 0xc4) : 0; break; }
    static int nlog; if (nlog++ < 20) LOG("[gadget] player %d card %d type %d: %s, %d uses, cooldown %d ticks\n", i, g->card, type,
                                          g->im ? "implemented" : "kind not implemented (not usable)", g->uses, g->cd);
}

/* NbeAction.fire 3 for player i at sim tick `tick`: -> 1 = the host handled it (activated, or refused: no use left / cooldown /
 * the effect could not start), 0 = no implementation for this gadget (the caller sends the raw input kind 8 as before) */
int gadget_use(void *sim, int i, uint8_t *ch, double deg, int tick) {
    if (i < 0 || i >= NBE_MAXP || !g_gad[i].im) return 0;
    Gad *g = &g_gad[i];
    if (g->uses <= 0 || tick < g->cd_until || !ch) return 1;
    if (g->im->fire(sim, g, ch, deg, tick)) {
        g->uses--; g->used++; g->cd_until = tick + g->cd;
        if (g_fired[g->im->kind & 7]++ < 10) LOG("[gadget] player %d kind %d fired at tick %d, %d uses left\n", i, g->im->kind, tick, g->uses);
    }
    return 1;
}

void gadget_tick(void *sim, int tick) {   /* env.c pre_snapshot, once per tick */
    for (int i = 0; i < NBE_MAXP; i++) if (g_gad[i].hot_end) heal_tick(sim, &g_gad[i], tick);
}

void gadget_publish(NbeResp *r, int tick) {   /* every response */
    r->gad_valid = 1;
    for (int i = 0; i < NBE_MAXP; i++) {
        const Gad *g = &g_gad[i];
        r->gad_card[i] = g->card; r->gad_kind[i] = g->im ? g->im->kind : NBE_GAD_NONE; r->gad_charges[i] = g->im ? g->uses : 0;
        r->gad_cd[i] = g->im && g->cd_until > tick ? g->cd_until - tick : 0; r->gad_used[i] = g->used;
    }
}
