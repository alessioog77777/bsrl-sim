/* test-only (issue #20): NB_CARDPROBE=1 logs which star power types the game's code asks a player's config for, and what it gets.
 *
 * libg reads a star power through a player-level getter that takes an ability type (the card's +0x100 key, not its MetaType):
 *   0xc42838 getCard(player, type)            -> card or NULL      (cfg-level 0xc34be0)
 *   0xc42874 getValue(player, type, idx)      -> Value<idx> or -1  (cfg-level 0xc34c6c; 67 call sites, 20 more via 0xb6a7b0)
 *   0xc428b0 (player, type)                   -> int or -1         (cfg-level 0xc34d10)
 *   0xc428ec (player, type, idx)              -> int or -1         (cfg-level 0xc34e10: reads cfg+0x38, the hypercharge card)
 * Each returns nothing while player+0x238 ("cards off") is set, and reads cfg = player+0x348 if set, else the loadout's config
 * (player+0x30[player+0x40] +0x08), the one offline.c my_player_ctor writes the requested star power into (+0x08).
 * The cfg-level getters are hooked too, for the few callers that skip the player wrapper (0xc28144, 0xc42168, 0xc42b28).
 *
 * Log lines (parsed by star_probe.py), one per new (slot, getter, type) and once more when that query first finds a card:
 *   [cardprobe] q tick T fn F slot S type 0xTT idx I -> R off238 O card ID card_t 0xCT lr 0xLR pl P cfg C
 *   [cardprobe] build slot S off238 O cfg348 0/1 card ID card_t 0xCT meta M n50 N pl P cfg C   (env.c div_publish)
 *   [cardprobe] build sim134 C list184 n N has35 H                                         (the battle-wide "cards off" trigger)
 * A query while the battle is being built (offline.c g_building: the sim's player list is not the new one yet) is logged
 * with tick -1 and slot -1; star_probe matches its pl / cfg to the build lines. The dedup state is cleared when a RESET is
 * accepted (env.c div_reset -> cardprobe_reset), before the build. card_t -1 (no card) prints as 0xffffffff.
 * Unset: nothing is hooked. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
extern Module *g_libg; extern void *g_sim; extern int g_building;
void *hook_replace(Module *m, uint64_t off, void *repl);
#define I32(p, o) (*(volatile int32_t *)((uint8_t *)(p) + (o)))
#define U8(p, o) (*(volatile uint8_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void *volatile *)((uint8_t *)(p) + (o)))
#define MAXP 16
#define NTYPE 0x400   /* the queried types seen in libg go up to 0x10b */
#define MAXLINES 6000 /* a 10-player battle logs a few hundred */

static int g_on;
static __thread int in_wrap;   /* a player-level getter is running: its cfg-level tail call is not a separate query */
static uint8_t seen[6][MAXP + 1][NTYPE];   /* bit 0 = logged, bit 1 = logged a hit; slot MAXP = no player matched */
static int nlines;

static int tick(void) { return g_sim && !g_building ? I32(g_sim, 0x10) : -1; }
static void **players(int *n) { if (!g_sim || g_building) { *n = 0; return NULL; }   /* building: g_sim is the old sim (freed) */ *n = I32(g_sim, 0x7c); if (*n < 0 || *n > 64) *n = 0; return (void **)PTR(g_sim, 0x70); }
static void *pl_cfg(void *pl) {   /* the cfg libg's player getters read */
    if (PTR(pl, 0x348)) return PTR(pl, 0x348);
    void **lo = PTR(pl, 0x30); int li = I32(pl, 0x40); void **ld = lo && li >= 0 ? lo[li] : NULL;
    return ld ? ld[1] : NULL;
}
static int pl_slot(void *pl) { int np; void **ps = players(&np); for (int i = 0; i < np && i < MAXP; i++) if (ps[i] == pl) return i; return MAXP; }
static int cfg_slot(void *cfg, void **pl) {
    int np; void **ps = players(&np);
    for (int i = 0; i < np && i < MAXP; i++) if (ps[i] && pl_cfg(ps[i]) == cfg) { *pl = ps[i]; return i; }
    *pl = NULL; return MAXP;
}
static const unsigned FN_ADDR[6] = {0xc42838, 0xc42874, 0xc428b0, 0xc428ec, 0xc34be0, 0xc34c6c};
static void note(int fn, void *pl, void *cfg, int type, int idx, long r, int hit, uint64_t lr) {
    int s = pl ? pl_slot(pl) : cfg_slot(cfg, &pl), t = type >= 0 && type < NTYPE ? type : NTYPE - 1;
    uint8_t want = hit ? 3 : 1;
    if (nlines >= MAXLINES) return;
    if (s != MAXP || !g_building) { if ((seen[fn][s][t] & want) == want) return; seen[fn][s][t] |= want; }   /* build: every query */
    nlines++;
    void *c = pl ? pl_cfg(pl) : cfg, *card = c ? PTR(c, 0x08) : NULL;
    LOG("[cardprobe] q tick %d fn %x slot %d type %#x idx %d -> %ld off238 %d card %d card_t %#x lr %#lx pl %p cfg %p\n", tick(),
        FN_ADDR[fn], s == MAXP ? -1 : s, type, idx, r, pl ? U8(pl, 0x238) : -1, card ? I32(card, 0x20) : 0,
        card ? I32(card, 0x100) : -1, (unsigned long)(lr - (uint64_t)g_libg->base), pl, c);
}

static void *(*o_gc)(void *, int); static int (*o_gv)(void *, int, int), (*o_g2)(void *, int), (*o_g3)(void *, int, int);
static void *(*o_cc)(void *, int); static int (*o_cv)(void *, int, int);
static void *my_gc(void *pl, int type) {
    in_wrap++; void *r = o_gc(pl, type); in_wrap--;
    note(0, pl, NULL, type, 0, r ? 1 : 0, r != NULL, (uint64_t)__builtin_return_address(0)); return r;
}
static int my_gv(void *pl, int type, int idx) {
    in_wrap++; int r = o_gv(pl, type, idx); in_wrap--;
    note(1, pl, NULL, type, idx, r, r != -1, (uint64_t)__builtin_return_address(0)); return r;
}
static int my_g2(void *pl, int type) {
    in_wrap++; int r = o_g2(pl, type); in_wrap--;
    note(2, pl, NULL, type, 0, r, r != -1, (uint64_t)__builtin_return_address(0)); return r;
}
static int my_g3(void *pl, int type, int idx) {
    in_wrap++; int r = o_g3(pl, type, idx); in_wrap--;
    note(3, pl, NULL, type, idx, r, r != -1, (uint64_t)__builtin_return_address(0)); return r;
}
static void *my_cc(void *cfg, int type) {   /* fn 4 / 5: cfg-level, direct callers only */
    void *r = o_cc(cfg, type);
    if (!in_wrap) note(4, NULL, cfg, type, 0, r ? 1 : 0, r != NULL, (uint64_t)__builtin_return_address(0));
    return r;
}
static int my_cv(void *cfg, int type, int idx) {
    int r = o_cv(cfg, type, idx);
    if (!in_wrap) note(5, NULL, cfg, type, idx, r, r != -1, (uint64_t)__builtin_return_address(0));
    return r;
}

void cardprobe_reset(void) { if (g_on) { memset(seen, 0, sizeof seen); nlines = 0; } }   /* env.c div_reset: a RESET was accepted */

void cardprobe_build(void *sim) {   /* env.c div_publish: a new battle was built */
    if (!g_on || !sim) return;
    int n = I32(sim, 0x184), has = 0; int32_t *l = PTR(sim, 0x190);
    if (n < 0 || n > 256) n = 0;
    for (int i = 0; l && i < n; i++) if (l[i] == 0x23) has = 1;
    LOG("[cardprobe] build sim134 %d list184 n %d has35 %d\n", U8(sim, 0x134), n, has);
    int np = I32(sim, 0x7c); void **ps = PTR(sim, 0x70); if (np < 0 || np > 64) np = 0;
    for (int i = 0; i < np && i < MAXP; i++) {
        void *pl = ps[i]; if (!pl) continue;
        void *c = pl_cfg(pl), *card = c ? PTR(c, 0x08) : NULL;
        LOG("[cardprobe] build slot %d off238 %d cfg348 %d card %d card_t %#x meta %d n50 %d pl %p cfg %p\n", i, U8(pl, 0x238),
            PTR(pl, 0x348) != NULL, card ? I32(card, 0x20) : 0, card ? I32(card, 0x100) : -1, card ? I32(card, 0x108) : -1,
            c ? I32(c, 0x5c) : -1, pl, c);
    }
}

void cardprobe_install(void) {
    const char *e = getenv("NB_CARDPROBE"); if (!e || !*e || *e == '0') return;
    g_on = 1;
    o_gc = hook_replace(g_libg, 0xc42838, my_gc); o_gv = hook_replace(g_libg, 0xc42874, my_gv);
    o_g2 = hook_replace(g_libg, 0xc428b0, my_g2); o_g3 = hook_replace(g_libg, 0xc428ec, my_g3);
    o_cc = hook_replace(g_libg, 0xc34be0, my_cc); o_cv = hook_replace(g_libg, 0xc34c6c, my_cv);
    LOG("[cardprobe] on: star power getters hooked\n");
}
