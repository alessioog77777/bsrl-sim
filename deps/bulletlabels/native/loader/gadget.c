/* gadget round (bsrl-sim issue #3): the host's own gadget activation.
 *
 * The shipped libg v68.279 is a client build: the gadget activation that input kind 8 reaches (NbeAction.fire 3 -> env.c
 * gadget_input -> 0xc2cef4 -> 0xb27c58(player+0x190, char, x, y, 0)) is a bare ret, and with it the server's bookkeeping of
 * uses and cooldown. Here the host keeps that state per player for the battle and runs a re-implemented activation per
 * gadget kind (NBE_GAD_*), the way spawnfix.c re-implements the compiled-out children of explosions with libg's own functions.
 * The state goes to Python in NbeResp gad_* (every response); bsrl-sim's env_v5.Env.gadget turns it into obs_v5 self 3.
 *
 * STATUS: the bookkeeping and the response fields are in place, NO kind is implemented yet (g_impl is empty): every gadget
 * publishes kind NONE and 0 uses (not usable), gadget_use leaves the input to the old path (the stub: nothing happens) and the
 * policy's gadget class stays masked. To add a kind:
 *   1. find in libg how the gadget card (cards.csv data id = the RESET's div.gadget[i], div_data(id) = its CardData) leads to
 *      its skill / ability data and the fields the kind needs: uses per battle, cooldown, distance / amount / spawned data;
 *   2. write classify (CardData -> 1 if this implementation handles it, with its uses and cooldown in ticks) and fire (the
 *      effect at the brawler, with libg's own functions where they exist) and add the entry to g_impl;
 *   3. run `python -m gadget_check` (bsrl-sim) on the built host: it fires each published kind and asserts its effect and
 *      that a use was spent. */
#include "nbhost.h"
#include "nbenv_proto.h"
#include <string.h>

#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
extern int32_t g_div_gadget[NBE_MAXP];   /* env.c: the RESET's requested gadget cards (offline.c puts them in the configs) */
void *div_data(int id);                  /* env.c: data row by global id, NULL unless the table has it */

typedef struct {
    int kind;                                                                     /* NBE_GAD_* */
    int (*classify)(void *card, int *uses, int *cd_ticks);                        /* 1 = handles this card */
    int (*fire)(void *sim, int pidx, uint8_t *ch, void *card, double deg);       /* 1 = activated (a use is spent) */
} GadImpl;
static const GadImpl g_impl[] = {
    /* {NBE_GAD_DASH, dash_classify, dash_fire}, ... (none yet) */
    {NBE_GAD_NONE, NULL, NULL}};

typedef struct { int card, uses, cd, cd_until, used; void *data; const GadImpl *im; } Gad;
static Gad g_gad[NBE_MAXP];

void gadget_clear(void) { memset(g_gad, 0, sizeof g_gad); }

/* RESET response (env.c div_publish), per player: its requested card, if the game gave the player a gadget object */
void gadget_reset_player(int i, int has_object) {
    if (i < 0 || i >= NBE_MAXP) return;
    Gad *g = &g_gad[i]; memset(g, 0, sizeof *g);
    if (!has_object || !g_div_gadget[i]) return;
    g->card = g_div_gadget[i]; g->data = div_data(g->card);
    for (const GadImpl *im = g_impl; g->data && im->classify; im++) {
        int uses = 0, cd = 0;
        if (im->classify(g->data, &uses, &cd)) { g->im = im; g->uses = uses; g->cd = cd; break; }
    }
    static int nlog; if (nlog++ < 20) LOG("[gadget] player %d card %d: %s, %d uses, cooldown %d ticks\n", i, g->card,
                                          g->im ? "implemented" : "kind not implemented (not usable)", g->uses, g->cd);
}

/* NbeAction.fire 3 for player i at sim tick `tick`: -> 1 = the host handled it (activated, or refused: no use left / cooldown),
 * 0 = no implementation for this gadget (the caller sends the raw input kind 8 as before) */
int gadget_use(void *sim, int i, uint8_t *ch, double deg, int tick) {
    if (i < 0 || i >= NBE_MAXP || !g_gad[i].im) return 0;
    Gad *g = &g_gad[i];
    if (g->uses <= 0 || tick < g->cd_until) return 1;
    if (g->im->fire(sim, i, ch, g->data, deg)) { g->uses--; g->used++; g->cd_until = tick + g->cd; }
    return 1;
}

void gadget_publish(NbeResp *r, int tick) {   /* every response */
    r->gad_valid = 1;
    for (int i = 0; i < NBE_MAXP; i++) {
        const Gad *g = &g_gad[i];
        r->gad_card[i] = g->card; r->gad_kind[i] = g->im ? g->im->kind : NBE_GAD_NONE; r->gad_charges[i] = g->im ? g->uses : 0;
        r->gad_cd[i] = g->im && g->cd_until > tick ? g->cd_until - tick : 0; r->gad_used[i] = g->used;
    }
}
