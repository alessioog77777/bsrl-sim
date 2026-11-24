/* star power probe (bsrl-sim issue #20, step 1): which star power card types the shipped libg v68.279 still reads.
 *
 * Every star power read in libg goes through three loadout-config lookups (the player wrappers 0xc42838 / 0xc42874 /
 * 0xc428b0 and the projectile / area owner wrappers 0xb6a75c / 0xb6a7b0 tail-call them with the same arguments):
 *   0xc34be0(cfg, type)       -> the config's card of that type (cfg+0x08 star power, then the cfg+0x50 list, count +0x5c),
 *                                NULL if none
 *   0xc34c6c(cfg, type, idx)  -> that card's Value<idx> (0xaa9568), -1 if none
 *   0xc34d10(cfg, type)       -> an int of that card (its first skill's +0xc0), -1 if none
 * The type is the card's Type enum (cards.csv Type, read by 0xaa9500 from the card). With NB_STAR_PROBE=1 this file
 * hooks the three, counts calls and hits (a card of that type was found) per type with the first caller's libg offset,
 * and logs the counts once per battle ("[star] q ..." lines, at the next build and at star_dump). A type that is never
 * asked for while a player carries its card has no reader left in this build (likely compiled out with the server code).
 * 40 ticks into each battle it also logs every player's star power card ("[star] p ..."): data id, Type enum, the "cards off" byte
 * (player+0x238: the wrappers return nothing while it is set). Python: star_audit.py. Without NB_STAR_PROBE nothing is hooked. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>

#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
#define U8(p, o) (*(uint8_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void **)((uint8_t *)(p) + (o)))
extern Module *g_libg;
#define B(off) ((uint8_t *)g_libg->base + (off))
void *hook_replace(Module *m, uint64_t off, void *repl);

#define NT 4096   /* Type enum values seen so far are < 1000 (the enum map 0xaa3000-0xaa8860); larger ones share the last slot */
static int g_on;
static uint32_t q_n[NT], q_hit[NT], q_lr[NT];
static long g_battle;

static void *(*o_card)(void *, int);
static int (*o_val)(void *, int, int);
static int (*o_int)(void *, int);
static int slot(int t) { return t < 0 ? NT - 1 : t >= NT ? NT - 1 : t; }
static void note(int t, int hit, void *lr) {
    int s = slot(t); q_n[s]++; if (hit) q_hit[s]++;
    if (!q_lr[s]) q_lr[s] = (uint32_t)((uint8_t *)lr - (uint8_t *)g_libg->base);
}
/* __builtin_return_address(0): the hook stub is a plain branch, so this is the libg call site (the wrappers tail-call) */
static void *my_card(void *cfg, int t) { void *r = o_card(cfg, t); note(t, r != NULL, __builtin_return_address(0)); return r; }
static int my_val(void *cfg, int t, int i) { int r = o_val(cfg, t, i); note(t, r != -1, __builtin_return_address(0)); return r; }
static int my_int(void *cfg, int t) { int r = o_int(cfg, t); note(t, r != -1, __builtin_return_address(0)); return r; }

void star_dump(void) {   /* the counts since the last dump, then zero */
    if (!g_on) return;
    int k = 0;
    for (int t = 0; t < NT; t++) if (q_n[t]) {
        LOG("[star] q battle=%ld type=%d n=%u hit=%u lr=0x%x\n", g_battle, t, q_n[t], q_hit[t], q_lr[t]); k++;
    }
    LOG("[star] dump battle=%ld types=%d\n", g_battle, k);
    memset(q_n, 0, sizeof q_n); memset(q_hit, 0, sizeof q_hit); memset(q_lr, 0, sizeof q_lr);
}
static void *player_cfg(void *p) {   /* what the player wrappers use: player+0x348, else the active loadout's config */
    void *c = PTR(p, 0x348); if (c) return c;
    void **lo = PTR(p, 0x30); int li = I32(p, 0x40); void **ld = lo && li >= 0 && li < 8 ? lo[li] : NULL;
    return ld ? ld[1] : NULL;
}
static void *g_ssim; static int g_sticks;
void star_built(void *sim) { if (g_on && sim) { g_battle++; g_ssim = sim; g_sticks = 0; } }   /* after a build */
void star_tick(void *sim) {   /* 40 ticks (2 s) into the battle, once the players have spawned: their star power cards */
    if (!g_on || sim != g_ssim || ++g_sticks != 40) return;
    int np = I32(sim, 0x7c); void **ps = PTR(sim, 0x70); if (np < 0 || np > 64 || !ps) np = 0;
    for (int i = 0; i < np; i++) {
        void *p = ps[i]; if (!p) continue;
        void *cfg = player_cfg(p), *card = cfg ? PTR(cfg, 0x08) : NULL;
        int t = card ? ((int (*)(void *))B(0xaa9500))(card) : -1;
        LOG("[star] p battle=%ld slot=%d card=%d type=%d off=%d\n", g_battle, i, card ? I32(card, 0x20) : 0, t, U8(p, 0x238));
    }
}
void star_install(void) {
    const char *e = getenv("NB_STAR_PROBE"); g_on = e && atoi(e);
    if (!g_on) return;
    o_card = hook_replace(g_libg, 0xc34be0, (void *)my_card);
    o_val = hook_replace(g_libg, 0xc34c6c, (void *)my_val);
    o_int = hook_replace(g_libg, 0xc34d10, (void *)my_int);
    LOG("[star] probe on: 0xc34be0 / 0xc34c6c / 0xc34d10 counted per card type\n");
}
