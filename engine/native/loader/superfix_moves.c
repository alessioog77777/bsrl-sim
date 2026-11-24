/* superfix_moves (bsrl-sim issue #6): supers that move the brawler or hit around it, whose server half is a bare `ret`
 * in the shipped client libg. The skill activation 0xb7b268 switches on the skill's BehaviorType (data +0x1a8) and
 * calls these stubs; the client half (move speed, "airborne" checks, no attacks in the air) is shipped and runs.
 *
 * FlyJump (Mico, LeaperUlti, type 19): handler 0xb7ceec calls the jump start 0xb55598(char) (`ret`, one caller 0xb7cf00),
 * then spawns the skill's AreaEffectObject LeaperUltiArea (Effect, follows the owner, its damage word +0xe8 = the
 * damage of its ChainAreaEffect, computed by the handler at 0xb7cf54..0xb7cfa0) through spawnAreaEffect 0xb3970c
 * (superfix.c). While the ulti skill's active counter +8 >= 1 the client treats Mico as airborne (0xb477f4: no fire,
 * MovementSpeed). Missing: the height and the landing hit. Here: the start hook starts a job; while the skill is active
 * the character's z (+0x40) follows a ramp up / hold / ramp down; the landing is the indicator area's ChainAreaEffect
 * (data +0xe0, getter 0xa97c04: LeaperUltiAreaExplosion, Damage + Pushback) spawned at the indicator when it times out:
 * the area update 0xb28c0c calls expire 0xb28f84 only from 0xb28d08, whose `bl` is redirected here. The chain area gets
 * the indicator's kind (+248) and is spawned and started with the shipped helper 0xb35c40 (its damage word is then the
 * chain data's own Damage), so libg's area apply deals the damage and the push. Moves sent while airborne are taken by
 * the shipped mover (the client only drops them during the 1000 ms CastingTime). The indicator's expired byte +244 is set before expire runs (expire sets it itself as
 * its first store), so a generic "chain on expire" hook on 0xb28f84 that checks +244 does not spawn it a second time.
 * Fallback without an indicator (or if it never expires): the chain is spawned at Mico LAND_WAIT ticks after the skill
 * ended.
 *
 * Ww (Carl, WhirlwindUlti, type 8): handler 0xb7d1c8 calls the spin 0xb5a704(char, scaled damage, Damage,
 * MsBetweenAttacks) (`ret`, one caller 0xb7d204); the client changes the move speed while the skill is active. Missing:
 * the contact damage. Here (modelled on env.c charge_contact_tick): while the ulti skill is active, each living enemy
 * character within WW_R of Carl is hit for the scaled damage through the damage root 0xb422d0 (so env's damage model,
 * power cubes and event records apply), at most once per MsBetweenAttacks per target; the owner's super charge is
 * kept (a super's hits do not recharge it, as env.c does for the charge contact and the leap bombs).
 *
 * Blink (Gigi, DaredevilUlti, type 1): handler 0xb7c198 -> 0xb80110(skill, char) calls the teleport 0xb4ffe0(char, tx,
 * ty, areaLeave, areaEnter, scaled damage, Damage, ...) (`ret`; 5 callers). Only its two Blink call sites are taken here
 * (return address 0xb804a8 / 0xb80334; the projectile / component callers 0xb7495c, 0xb75dc0, 0xb9beb0 keep the stub):
 * the leave area is spawned at the start, the brawler is moved to the target (pulled back along the line to a walkable
 * tile: ForceValidTile), the enter area is spawned there, and when the enter area does not deal damage itself (an
 * Effect area: DaredevilUltiNoDamageEnter; the overcharged DaredevilOverchargedUltiDamage is a Damage area and does)
 * every living enemy character within its radius is hit once for the scaled damage through the damage root. A
 * BlinkReturn skill (type 23, the swapped-in DaredevilUltiReturn) teleports back to where the last Blink started.
 *
 * The per-tick work runs from the character tick 0xb3ae74 (enter hook, per character; only characters with a job do
 * anything). NB_SUPERFIX=0 turns every part off (superfix_install does not call this file's install). */
#include "nbhost.h"
#include <math.h>
#include <stdlib.h>
extern Module *g_libg;
extern int g_superfix, g_superfix_dbg;
void hook_enter(Module *m, uint64_t off, void (*fn)(uint64_t *ctx));
void hook_bl(Module *m, uint64_t off, void *fn);
#define B(off) ((uint8_t *)g_libg->base + (off))
#define I32(p, o) (*(volatile int32_t *)((uint8_t *)(p) + (o)))
#define U8(p, o) (*(volatile uint8_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void *volatile *)((uint8_t *)(p) + (o)))
#define VTOF(o) ((int)((uint8_t *)PTR(o, 0) - g_libg->base))
#define VT_CHAR 0x1209020                     /* character vtable (env.c VT_CHAR) */
#define RET_INSN 0xd65f03c0
static int mode(void) { return g_superfix && I32(g_libg->base, 0x12eefb4) == 3; }   /* battle running (spawnfix.c mode()) */

typedef void *(*getp_t)(void *);
typedef int (*geti_t)(void *);
static void *sim_of(void *o) { return ((getp_t)B(0xb6a378))(o); }
static int now_of(void *o) { void *s = sim_of(o); return s ? I32(s, 0x10) : 0; }             /* sim tick (0xc26bd8) */
static void *ulti_skill(uint8_t *ch) {   /* skill slot 1 = ulti (ctor 0xb34560: +0x458 array, +0x464 count) */
    void **sk = PTR(ch, 0x458); return sk && I32(ch, 0x464) > 1 ? sk[1] : NULL;
}
static uint8_t *skill_data(void *sk) { return sk ? ((uint8_t *(*)(void *, int))B(0xb7adfc))(sk, U8(sk, 49) & 1) : NULL; }   /* +49 = overcharged */
static int area_radius(uint8_t *ad) { int r = ad ? I32(ad, 0x12c) : 0; return r > 0 && r < 5000 ? r : 0; }   /* AreaEffectData Radius (+300, getter a97bb4; +292 is TimeMs) */
/* the shipped helper 0xb35c40(owner, data, x, y, extra damage, kind, modifiers): create 0xb2aefc, add 0xb63198, start
 * 0xb29494; kind 2 = what the skill execute passes (0xb7bebc). The area's damage word is extra + the data's own Damage
 * (0xb2b0b4), so extra is 0 here, as at every shipped caller. */
static void *spawn_area(uint8_t *owner, void *data, int x, int y, int dmg, int kind) {
    if (!owner || !data || !PTR(owner, 0x20)) return NULL;
    return ((void *(*)(void *, void *, int, int, int, int, void *))B(0xb35c40))(owner, data, x, y, dmg, kind, NULL);
}

/* damage `v` for `dmg` through the (env-hooked) damage root, the way env.c charge_contact_tick calls it; the owner's
 * super charge (player +0x54) is restored afterwards */
typedef uint64_t (*root_t)(void *, int, int, int, int, void *, void *, int, int, int, void *, int, int, int, int, int, int, int);
static void hit(uint8_t *ch, uint8_t *v, int dmg) {
    void *sim = sim_of(ch); int owner = I32(ch, 0x80), np = sim ? I32(sim, 0x7c) : 0;
    uint8_t *pl = sim && owner >= 0 && owner < np && np <= 64 ? ((uint8_t **)PTR(sim, 0x70))[owner] : NULL;
    int c0 = pl ? I32(pl, 0x54) : 0;
    ((root_t)B(0xb422d0))(v, owner, dmg, dmg, 0, ch, NULL, 1, I32(ch, 0x38), I32(ch, 0x3c), NULL, 0, 0, 0, 1, 0, 0, 0);
    if (pl) I32(pl, 0x54) = c0;
}
static int enemy(uint8_t *ch, uint8_t *c) {   /* a living enemy character (another owner; env.c's charge contact rule) */
    return c && c != ch && VTOF(c) == VT_CHAR && I32(c, 0x3c4) > 0 && !(I32(c, 0x80) == I32(ch, 0x80) && I32(ch, 0x80) >= 0);
}
static void **objs(void *sim, int *n) { void *h = sim ? PTR(sim, 0) : NULL; *n = h ? I32(h, 0xc) : 0; if (*n < 0 || *n > 4096) *n = 0; return h ? (void **)PTR(h, 0) : NULL; }

/* ======================= FlyJump */
#define FJ_H 700           /* peak z: the height env.c's leap uses (Piper) */
#define FJ_RAMP 6          /* ticks up and down (300 ms) */
#define LAND_WAIT 10       /* ticks after the skill ended without an indicator expiry: land anyway */
typedef struct { uint8_t *ch; int id, t, seen, done_t, landed; void *ind; } FlyJob;
static FlyJob g_fj[16]; static int g_nfj;
static long g_fj_start, g_fj_land, g_fj_fallback;
static void flyjump_start(uint64_t *ctx) {   /* enter 0xb55598(char) */
    uint8_t *ch = (uint8_t *)ctx[0]; if (!mode() || !ch) return;
    void *sk = ulti_skill(ch); uint8_t *sd = skill_data(sk); if (!sd || I32(sd, 0x1a8) != 19) return;
    int k; for (k = 0; k < g_nfj && g_fj[k].ch != ch; k++) {}
    if (k == g_nfj) { if (g_nfj >= 16) return; g_nfj++; }
    g_fj[k] = (FlyJob){ch, I32(ch, 8), 0, 0, 0, 0, PTR(sd, U8(sk, 49) & 1 ? 176 : 160)};   /* AreaEffectObject (OC +176) */
    g_fj_start++;
    if (g_superfix_dbg) LOG("[superfix] flyjump start char %d at %d,%d active %d indicator %p chain %p\n", I32(ch, 8),
                            I32(ch, 0x38), I32(ch, 0x3c), I32(sk, 8), g_fj[k].ind, g_fj[k].ind ? PTR(g_fj[k].ind, 0xe0) : NULL);
}
static void flyjump_land(FlyJob *j, uint8_t *at, int kind, const char *how) {
    void *chain = j->ind ? PTR(j->ind, 0xe0) : NULL;   /* ChainAreaEffect */
    j->landed = 1; I32(j->ch, 0x40) = 0;
    if (!chain || I32(j->ch, 0x3c4) <= 0) return;
    void *a = spawn_area(j->ch, chain, I32(at, 0x38), I32(at, 0x3c), 0, kind);
    g_fj_land++;
    if (g_superfix_dbg) LOG("[superfix] flyjump land (%s) char %d at %d,%d: area %p type %d r %d dmg %d kind %d\n", how, j->id,
                            I32(at, 0x38), I32(at, 0x3c), a, I32(chain, 0x118), area_radius(chain), a ? I32(a, 232) : 0, kind);
}
static void flyjump_tick(FlyJob *j) {
    uint8_t *ch = j->ch; void *sk = ulti_skill(ch); int left = sk ? I32(sk, 8) : 0;
    if (I32(ch, 0x3c4) <= 0) { I32(ch, 0x40) = 0; j->landed = 2; return; }
    j->t++;
    if (left >= 1) j->seen = 1;
    else if (!j->seen && j->t < 40) return;              /* the active counter may be set after the start hook */
    if (left >= 1 && !j->done_t) {
        int up = j->t < FJ_RAMP ? j->t : FJ_RAMP, down = left < FJ_RAMP ? left : FJ_RAMP;
        I32(ch, 0x40) = FJ_H * (up < down ? up : down) / FJ_RAMP;
        return;
    }
    I32(ch, 0x40) = 0;
    if (!j->done_t) j->done_t = j->t;
    if (!j->landed && j->t - j->done_t >= LAND_WAIT) {   /* no indicator expiry seen: land at the brawler */
        g_fj_fallback++; flyjump_land(j, ch, 2, "fallback");
    }
}
/* in place of `bl 0xb28f84` (area expire) at 0xb28d08 */
static void on_area_expire(uint8_t *a) {
    if (mode() && a && !U8(a, 244)) {
        uint8_t *own = PTR(a, 280);   /* the character the area follows */
        for (int i = 0; i < g_nfj; i++) {
            FlyJob *j = &g_fj[i];
            if (j->landed || own != j->ch || !j->ind || ((getp_t)B(0xb6a384))(a) != j->ind) continue;
            if (now_of(a) < I32(a, 216)) continue;           /* a natural timeout only (+216 = end tick) */
            U8(a, 244) = 1;                                  /* expire's own first store, done early (see the header) */
            flyjump_land(j, a, I32(a, 248), "indicator");
        }
    }
    ((void (*)(void *))B(0xb28f84))(a);
}

/* ======================= Ww */
#define WW_R 300           /* contact radius: not in the tables; ~ Carl's spin ring (env.c's charge contact uses 250) */
typedef struct { uint8_t *ch; int id, dmg, every, n; int hid[24], ht[24]; } WwJob;
static WwJob g_ww[16]; static int g_nww;
static long g_ww_start, g_ww_hits;
static void ww_start(uint64_t *ctx) {   /* enter 0xb5a704(char, scaled damage, Damage, MsBetweenAttacks) */
    uint8_t *ch = (uint8_t *)ctx[0]; if (!mode() || !ch) return;
    int ms = (int)ctx[3], k;
    for (k = 0; k < g_nww && g_ww[k].ch != ch; k++) {}
    if (k == g_nww) { if (g_nww >= 16) return; g_nww++; }
    g_ww[k] = (WwJob){ch, I32(ch, 8), (int)ctx[1], ms / 50 > 0 ? ms / 50 : 1, 0, {0}, {0}};   /* 50 ms per tick */
    g_ww_start++;
    if (g_superfix_dbg) LOG("[superfix] ww start char %d dmg %d (Damage %d) every %d ms\n", I32(ch, 8), (int)ctx[1], (int)ctx[2], ms);
}
static int ww_tick(WwJob *j) {   /* 0 = job over */
    uint8_t *ch = j->ch; void *sk = ulti_skill(ch); uint8_t *sd = skill_data(sk);
    if (I32(ch, 0x3c4) <= 0 || !sd || I32(sd, 0x1a8) != 8 || I32(sk, 8) < 1) return 0;
    void *sim = sim_of(ch); int now = sim ? I32(sim, 0x10) : 0, n, x = I32(ch, 0x38), y = I32(ch, 0x3c);
    void **v = objs(sim, &n);
    for (int i = 0; i < n; i++) {
        uint8_t *c = v[i]; if (!enemy(ch, c)) continue;
        long dx = I32(c, 0x38) - x, dy = I32(c, 0x3c) - y; if (dx * dx + dy * dy > (long)WW_R * WW_R) continue;
        int cid = I32(c, 8), h; for (h = 0; h < j->n && j->hid[h] != cid; h++) {}
        if (h < j->n && now - j->ht[h] < j->every) continue;
        if (h == j->n) { if (j->n >= 24) continue; j->n++; j->hid[h] = cid; }
        j->ht[h] = now;
        int hp0 = I32(c, 0x3c4); hit(ch, c, j->dmg); g_ww_hits++;
        if (g_superfix_dbg) LOG("[superfix] ww hit char %d -> %d: hp %d -> %d\n", j->id, cid, hp0, I32(c, 0x3c4));
        v = objs(sim, &n);   /* the root may add objects */
    }
    return 1;
}

/* ======================= Blink */
static int tile_blocks_move(int id) {   /* tiles.csv BlocksMovement rows (env.c SF_BLOCK) */
    static const int rows[] = {1,2,3,4,6,9,10,11,12,13,14,15,16,18,20,21,39,40,41,42,43,44,45,46,53,54,55,56,57,58,59,60,61,62,63,65,66,69,70,71,72,73};
    for (unsigned i = 0; i < sizeof rows / sizeof *rows; i++) if (id == 27000000 + rows[i]) return 1;
    return 0;
}
static int walkable(void *sim, int x, int y) {   /* env.c walkable(): 300-unit tiles of the sim's map (+0x68) */
    void *m = PTR(sim, 0x68); if (!m) return 1;
    int W = I32(m, 0xc4), Hh = I32(m, 0xc8), tx = (int)floor(x / 300.0), ty = (int)floor(y / 300.0);
    if (tx < 0 || ty < 0 || tx >= W || ty >= Hh) return 0;
    void *t = ((void **)PTR(m, 0x20))[tx + W * ty]; if (!t) return 1;
    void *d = PTR(t, 8); return !d || !tile_blocks_move(I32(d, 0x20));
}
static void stop_mover(void *ch) {   /* env.c sf_stop: not charging, can stop, moving -> stop 0xb4103c */
    if (!U8(ch, 0x490) && (((geti_t)B(0xb50ac8))(ch) & 1) && (((geti_t)B(0xb393f0))(ch) & 1)) ((void (*)(void *))B(0xb4103c))(ch);
}
typedef struct { uint8_t *ch; int x, y; } BlinkHome;
static BlinkHome g_home[16]; static int g_nhome;
static long g_blink, g_blink_back, g_blink_hits;
static void blink_start(uint64_t *ctx) {   /* enter 0xb4ffe0(char, tx, ty, areaLeave, areaEnter, scaled damage, Damage, ...) */
    uint64_t lr = ctx[9] - (uint64_t)g_libg->base;
    if (!mode() || (lr != 0xb804a8 && lr != 0xb80334)) return;   /* the two calls in the Blink handler 0xb80110 */
    uint8_t *ch = (uint8_t *)ctx[0]; void *sim = ch ? sim_of(ch) : NULL; if (!sim || I32(ch, 0x3c4) <= 0) return;
    uint8_t *sd = skill_data(ulti_skill(ch)); int type = sd ? I32(sd, 0x1a8) : -1;
    int tx = (int)ctx[1], ty = (int)ctx[2], dmg = (int)ctx[5], x0 = I32(ch, 0x38), y0 = I32(ch, 0x3c), k;
    uint8_t *aLeave = (uint8_t *)ctx[3], *aEnter = (uint8_t *)ctx[4];
    for (k = 0; k < g_nhome && g_home[k].ch != ch; k++) {}
    if (type == 23) {                                        /* BlinkReturn: back to the remembered start */
        if (k == g_nhome) return;
        tx = g_home[k].x; ty = g_home[k].y; g_blink_back++;
    } else {
        if (k == g_nhome) { if (g_nhome >= 16) return; g_nhome++; }
        g_home[k] = (BlinkHome){ch, x0, y0};
    }
    double L = hypot(tx - x0, ty - y0), back = 0; int lx = tx, ly = ty;   /* ForceValidTile: back along the line */
    while (!walkable(sim, lx, ly) && back < L) {
        back += 50; double f = (L - back) / (L > 1 ? L : 1); if (f < 0) f = 0;
        lx = (int)floor(x0 + (tx - x0) * f + 0.5); ly = (int)floor(y0 + (ty - y0) * f + 0.5);
    }
    if (aLeave) spawn_area(ch, aLeave, x0, y0, 0, 2);
    stop_mover(ch);
    ((void (*)(void *, int, int, int))B(0xb6a61c))(ch, lx, ly, 0);
    if (aEnter) spawn_area(ch, aEnter, lx, ly, 0, 2);
    int r = area_radius(aEnter), hits = 0;
    if (type != 23 && dmg > 0 && r > 0 && (!aEnter || I32(aEnter, 0x118) != 0)) {   /* the enter area does not hit (not Damage) */
        int n; void **v = objs(sim, &n);
        for (int i = 0; i < n; i++) {
            uint8_t *c = v[i]; if (!enemy(ch, c)) continue;
            long dx = I32(c, 0x38) - lx, dy = I32(c, 0x3c) - ly; if (dx * dx + dy * dy > (long)r * r) continue;
            hit(ch, c, dmg); hits++; v = objs(sim, &n);
        }
    }
    g_blink++; g_blink_hits += hits;
    if (g_superfix_dbg) LOG("[superfix] blink type %d char %d %d,%d -> %d,%d (asked %d,%d) leave %p enter %p r %d type %d dmg %d hits %d\n",
                            type, I32(ch, 8), x0, y0, lx, ly, tx, ty, aLeave, aEnter, r, aEnter ? I32(aEnter, 0x118) : -1, dmg, hits);
}

/* ======================= per character tick (enter 0xb3ae74) */
static void on_char_tick(uint64_t *ctx) {
    uint8_t *ch = (uint8_t *)ctx[0]; if (!ch || !mode() || (!g_nfj && !g_nww)) return;
    for (int i = 0; i < g_nfj; i++) if (g_fj[i].ch == ch) {
        if (I32(ch, 8) != g_fj[i].id) { g_fj[i--] = g_fj[--g_nfj]; continue; }   /* object reused */
        flyjump_tick(&g_fj[i]);
        if (g_fj[i].landed) g_fj[i--] = g_fj[--g_nfj];
    }
    for (int i = 0; i < g_nww; i++) if (g_ww[i].ch == ch)
        if (I32(ch, 8) != g_ww[i].id || !ww_tick(&g_ww[i])) g_ww[i--] = g_ww[--g_nww];
}

void superfix_moves_install(void) {
    int n = 0;
    if (*(uint32_t *)B(0xb3ae74) != 0xa9ba7bfd) { LOG("[superfix] moves: 0xb3ae74 is not the character tick: left out\n"); return; }   /* stp x29, x30, [sp, #-96]! */
    hook_enter(g_libg, 0xb3ae74, on_char_tick);
    if (*(uint32_t *)B(0xb55598) == RET_INSN && *(uint32_t *)B(0xb7cf00) == 0x97ff61a6 &&   /* bl 0xb55598 */
        *(uint32_t *)B(0xb28d08) == 0x9400009f) {                                            /* bl 0xb28f84 */
        hook_enter(g_libg, 0xb55598, flyjump_start); hook_bl(g_libg, 0xb28d08, (void *)on_area_expire); n |= 1;
    } else LOG("[superfix] moves: FlyJump sites differ: left out\n");
    if (*(uint32_t *)B(0xb5a704) == RET_INSN && *(uint32_t *)B(0xb5a708) == RET_INSN) { hook_enter(g_libg, 0xb5a704, ww_start); n |= 2; }
    else LOG("[superfix] moves: 0xb5a704 is not the Ww stub: left out\n");
    if (*(uint32_t *)B(0xb4ffe0) == RET_INSN) { hook_enter(g_libg, 0xb4ffe0, blink_start); n |= 4; }
    else LOG("[superfix] moves: 0xb4ffe0 is not the teleport stub: left out\n");
    LOG("[superfix] moves installed:%s%s%s\n", n & 1 ? " flyjump" : "", n & 2 ? " ww" : "", n & 4 ? " blink" : "");
}
void superfix_moves_status(void) {
    LOG("[superfix] moves: flyjump %ld starts, %ld landings (%ld fallback); ww %ld starts, %ld hits; blink %ld (%ld back), %ld hits\n",
        g_fj_start, g_fj_land, g_fj_fallback, g_ww_start, g_ww_hits, g_blink, g_blink_back, g_blink_hits);
}
