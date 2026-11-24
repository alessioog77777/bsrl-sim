/* superfix_status.c (bsrl-sim issue #6, see superfix.c): status effects. The shipped client libg compiles the character's
 * status-effect manager (char +0x890) out: add 0xba3230(cont, ch, data, w3 source player, w4 team, x5 source object,
 * x6 source character, w7 kind, ...) is `mov x0, xzr; ret` (40 callers: the skill's ActivationSelfStatusEffects 0xb8061c via
 * 0xb807ac, the wrapper 0xb35dd0(ch, data, ...) = 0xba3230(ch+0x890, ch, data, ...), the area enemy status 0xb2a71c, the
 * MountUp tick 0xba0048, ...), the removes 0xba3260(cont, data, -2) / 0xba3264(cont, data, source) are bare `ret`, and
 * nothing writes the aggregates the character code reads. So no status effect ever existed in a battle here.
 *
 * Store: a host-side list of instances (character, data, source, start / end tick, next trigger). The add still returns
 * NULL, the stub's own result (every caller tests it and then skips its own bookkeeping on the instance, e.g. 0xb807b0),
 * so libg's instance list (char +0x918) stays empty and nothing libg-side points at our records.
 * Per character and tick (enter hook on the character update 0xb3941c = VT_CHAR slot +0x28, before libg's own update):
 *   expiry (DurationTicks +0x98, until = add tick + duration; a non-Stacking re-add (+0xac) refreshes the end tick),
 *   Damage +0xb8 / Healing +0xf0 every TriggerRate +0xa0 ticks (TriggerOnFirstTick +0xab: also at the add; TriggerRate 0:
 *   once) through the damage root 0xb422d0 (env.c's hook on it applies the env's damage model and events, as for its own
 *   calls) and the heal 0xb41554(ch, source player, amount, 1, NULL, 0, 0, 0, 0, 0) (the argument pattern of 0xb43ae8),
 *   then the aggregates: char +0x954 = max ShieldPercent +0xf4 (the damage root 0xb422d0 reads it at 0xb428fc:
 *   pct = max(skill shield, +0x954), damage * (100 - pct) / 100), char +0x97c = max Invisibility +0x12c (0 none ..
 *   3 full; libg's reader 0xb3931c and the char update). Pulsating effects (+0xb0 on / +0xb4 off ticks) count only in
 *   their on phases. Nothing else in this libg writes those two words.
 * Not modelled (no client reader found): speed (SpeedBoostPercent +0xf8 / Absolute +0xfc / reduces +0x100 / +0x104: the
 *   move start 0xb4790c times a path with the CharacterData speed 0xab31e0 only), damage boosts (+0x108, +0x114: the damage
 *   percent 0xb4b660 reads char +0x968 / +0x970, whose status mapping is not proven), stuns, silences, boolean attributes
 *   (Kaze's HasStormVision), CancelOn* conditions. Those statuses are stored (so their duration and target are right)
 *   but change nothing. 0xba3274 (called at stop / charge end / respawn) and the other manager stubs are left alone.
 *
 * StatusEffect areas (AreaEffectData +0x118 == 36: Max, Jae-Yong, Najia's jars, Kaze's storm, slows ...): the area apply
 * 0xb29494 returns before its status code for type 36 (its type gate) and libg reads StatusEffectSelf +0x88 / Ally +0x90
 * nowhere. From the area tick (superfix.c's enter hook on 0xb28c0c calls superfix_status_area): at the area's first tick
 * (or after StatusEffectTickInterval +0xa0 ticks with DelayFirstTick +0x17c) and then every TickInterval ticks (0: once),
 * every live character within 0xb2b298(data, ch) (libg's own range: radius + the character's radius - 50) of the area gets
 * Self (the area's owner +0x108), Ally (same team, 0xb6a820) or Enemy +0x98 through 0xb35dd0 with the arguments of the
 * area enemy call 0xb2a71c (source player 0xb6a754(area), team 0xb6a820(area), area, owner, kind +0xf8, 0, -1, -1, 0).
 * AffectOnlyHeroes +0x1b4 keeps brawlers only (0xb34de4); IgnoreCharacters +0x1b6 areas (projectile slows) are skipped.
 * MountUp (Draco): the component's tick 0xba0048 is never called in this libg; it is run from the character tick (below).
 * NB_SUPERFIX=0 turns this off with the rest, NB_SUPERFIX_STATUS=0 this part alone; NB_SUPERFIX_DBG=1 logs adds, area hits, triggers and brawler HP drops. */
#include "nbhost.h"
#include <stdlib.h>
#include <string.h>
extern Module *g_libg;
extern int g_superfix, g_superfix_dbg;
void *hook_replace(Module *m, uint64_t off, void *repl);
void hook_enter(Module *m, uint64_t off, void (*fn)(uint64_t *ctx));
#define B(off) ((uint8_t *)g_libg->base + (off))
#define I32(p, o) (*(int32_t *)((uint8_t *)(p) + (o)))
#define U8(p, o) (*(uint8_t *)((uint8_t *)(p) + (o)))
#define PTR(p, o) (*(void **)((uint8_t *)(p) + (o)))
static int mode(void) { return I32(g_libg->base, 0x12eefb4) == 3; }   /* battle running (spawnfix.c mode()) */
typedef void *(*getp_t)(void *);
typedef int (*geti_t)(void *);

/* StatusEffectData fields (loader 0xba3814) */
enum { SE_DUR = 0x98, SE_RATE = 0xa0, SE_SRCDEATH = 0xa8, SE_FIRST = 0xab, SE_STACK = 0xac, SE_REFRESH = 0xad,
       SE_PON = 0xb0, SE_POFF = 0xb4, SE_DMG = 0xb8, SE_HEAL = 0xf0, SE_SHIELD = 0xf4, SE_INVIS = 0x12c };
/* character aggregates (manager +0xc4 / +0xec) */
enum { CH_SHIELD = 0x954, CH_INVIS = 0x97c };
#define VT_CHAR 0x1209020

typedef struct {
    uint8_t *ch; int ch_id; void *data, *src_obj, *src_ch; int src_id, src_player, start, until, next;
} St;
#define ST_MAX 512            /* 10 brawlers + summons x a few effects; Draco re-adds a 2-tick status every tick */
static St g_st[ST_MAX]; static int g_nst;
static long g_nmount, g_nadd, g_nrefresh, g_nfull, g_nexpire, g_nremove, g_ndmg, g_nheal, g_narea_hits, g_narea_apply;

static int now_tick(void *obj) { uint8_t *sim = ((getp_t)B(0xb6a378))(obj); return sim ? I32(sim, 0x10) : -1; }
static int data_id(void *d) { return d ? I32(d, 0x20) : -1; }            /* LogicData global id (table * 1e6 + row) */
static int is_char(void *o) { return o && (int)((uint8_t *)PTR(o, 0) - g_libg->base) == VT_CHAR; }
static void **objects(uint8_t *ch, int *n) {                            /* the object manager's list (spawnfix.c) */
    uint8_t *mgr = ch ? PTR(ch, 0x20) : NULL; *n = 0;
    if (!mgr) return NULL;
    int c = I32(mgr, 0x34); if (c < 0 || c > 4096) return NULL;
    *n = c; return PTR(mgr, 0x28);
}
static uint8_t *by_id(uint8_t *any, int id) {
    int n; void **v = objects(any, &n);
    for (int i = 0; v && i < n; i++) if (v[i] && I32(v[i], 8) == id) return v[i];
    return NULL;
}

/* 0xba3230 */
static void *my_status_add(void *cont, uint8_t *ch, void *data, int src_player, int team, void *src_obj, uint8_t *src_ch) {
    (void)cont; (void)team;
    if (!g_superfix || !mode() || !ch || !data) return NULL;
    int dur = I32(data, SE_DUR), t = now_tick(ch);
    if (dur <= 0 || t < 0) return NULL;
    if (!U8(data, SE_STACK))
        for (int i = 0; i < g_nst; i++) if (g_st[i].ch == ch && g_st[i].ch_id == I32(ch, 8) && g_st[i].data == data) {
            if (U8(data, SE_REFRESH) || g_st[i].until < t + dur) g_st[i].until = t + dur;   /* refresh: a new full duration */
            g_nrefresh++; return NULL;
        }
    if (g_nst >= ST_MAX) { g_nfull++; return NULL; }
    int rate = I32(data, SE_RATE);
    St *s = &g_st[g_nst++];
    *s = (St){ch, I32(ch, 8), data, src_obj, src_ch, src_ch ? I32(src_ch, 8) : -1, src_player, t, t + dur,
              U8(data, SE_FIRST) || rate <= 0 ? t : t + rate};
    g_nadd++;
    if (g_superfix_dbg) LOG("[superfix] st: add %d to ch %d (hp %d) from ch %d player %d: %d ticks, tick %d, shield %d dmg %d heal %d invis %d, live %d\n",
                            data_id(data), I32(ch, 8), I32(ch, 0x3c4), s->src_id, src_player, dur, t, I32(data, SE_SHIELD),
                            I32(data, SE_DMG), I32(data, SE_HEAL), I32(data, SE_INVIS), g_nst);
    return NULL;
}
/* 0xba3264(cont, data, source) / 0xba3260(cont, data, -2 = any source) */
static void my_status_remove(void *cont, void *data, void *src) {
    if (!g_superfix || !cont) return;
    uint8_t *ch = (uint8_t *)cont - 0x890;
    for (int i = g_nst - 1; i >= 0; i--) {
        St *s = &g_st[i];
        if (s->ch != ch || s->data != data) continue;
        if ((int32_t)(intptr_t)src != -2 && src && src != s->src_ch && src != s->src_obj) continue;
        g_st[i] = g_st[--g_nst]; g_nremove++;
    }
}

static int pulse_on(const St *s, int t) {
    int on = I32(s->data, SE_PON), off = I32(s->data, SE_POFF);
    return on <= 0 || off <= 0 || (t - s->start) % (on + off) < on;
}
static void trigger(St *s, int t) {
    uint8_t *ch = s->ch, *src = s->src_id >= 0 ? by_id(ch, s->src_id) : NULL;
    int dmg = I32(s->data, SE_DMG), heal = I32(s->data, SE_HEAL), hp0 = I32(ch, 0x3c4);
    if (dmg > 0) {   /* the argument pattern of env.c's own root calls (charge contact, leap bombs) */
        ((uint64_t (*)(void *, int, int, int, int, void *, void *, int, int, int, void *, int, int, int, int, int, int, int))B(0xb422d0))(
            ch, s->src_player, dmg, dmg, 0, src, NULL, 1, I32(ch, 0x38), I32(ch, 0x3c), NULL, 0, 0, 0, 1, 0, 0, 0);
        g_ndmg++;
    }
    if (heal > 0 && I32(ch, 0x3c4) > 0) {
        ((void (*)(void *, int, int, int, void *, int, int, int, long, long))B(0xb41554))(ch, s->src_player, heal, 1, NULL, 0, 0, 0, 0, 0);
        g_nheal++;
    }
    if (g_superfix_dbg && (dmg > 0 || heal > 0))
        LOG("[superfix] st: trigger %d on ch %d tick %d (+%d): dmg %d heal %d, hp %d -> %d\n", data_id(s->data), I32(ch, 8), t, t - s->start,
            dmg, heal, hp0, I32(ch, 0x3c4));
}

/* enter hook on the character update 0xb3941c: ctx[0] = character */
static int g_last_tick = -1;
static void on_char_update(uint64_t *ctx) {
    uint8_t *ch = (uint8_t *)ctx[0];
    if (!g_superfix || !mode() || !is_char(ch)) return;
    int t = now_tick(ch); if (t < 0) return;
    if (t != g_last_tick) {   /* once per tick: drop the records of characters that left the object list */
        g_last_tick = t;
        int n; void **v = objects(ch, &n);
        for (int i = g_nst - 1; i >= 0 && v; i--) {
            int alive = 0;
            for (int k = 0; k < n && !alive; k++) alive = v[k] == g_st[i].ch && I32(v[k], 8) == g_st[i].ch_id;
            if (!alive) { g_st[i] = g_st[--g_nst]; g_nexpire++; }
        }
    }
    {   /* MountUp (Draco): libg never runs this component's tick (vtable +0x40 = 0xba0048; its mount ticks +0x38 stayed at
         * CustomValue 200 for 100 ticks after the mount in a host run, so a mounted Draco stayed mounted): run it once per
         * tick while mounted. It counts +0x38 down, adds the component data's +0xb8 / +0xc8 statuses (DragonRiderUltiSpeed:
         * shield 20, 2 ticks) through 0xb35dd0 each tick and dismounts at 0 (weapon swap 0xb35c3c, still a stub). */
        int nc = I32(ch, 0xa6c); void **c = PTR(ch, 0xa60);
        for (int i = 0; c && i < nc && i < 64; i++)
            if (c[i] && I32(c[i], 0x10) == 21 && I32(c[i], 0x38) > 0 && PTR(c[i], 8) == ch &&
                (*(void ***)c[i])[0x40 / 8] == (void *)B(0xba0048)) { ((void (*)(void *))B(0xba0048))(c[i]); g_nmount++; }
    }
    int shield = 0, invis = 0, any = 0;
    for (int i = 0; i < g_nst; i++) {
        St *s = &g_st[i];
        if (s->ch != ch || s->ch_id != I32(ch, 8)) continue;
        int rate = I32(s->data, SE_RATE), dur = s->until - s->start;
        /* last trigger: e < duration with TriggerOnFirstTick, e <= duration without (both give duration / rate triggers) */
        int last = U8(s->data, SE_FIRST) ? s->until - 1 : s->until;
        while (s->next <= t && s->next <= last && I32(ch, 0x3c4) > 0) {
            trigger(s, t);
            s->next = rate > 0 ? s->next + rate : 0x7fffffff;
            if (rate > 0 && dur / rate > 64) break;   /* belt and braces for long effects added long ago */
        }
        if (t >= s->until || (U8(s->data, SE_SRCDEATH) && s->src_id >= 0 && !by_id(ch, s->src_id))) {
            if (g_superfix_dbg) LOG("[superfix] st: end %d on ch %d tick %d (%d ticks)\n", data_id(s->data), I32(ch, 8), t, t - s->start);
            g_st[i--] = g_st[--g_nst]; g_nexpire++; continue;
        }
        any = 1;
        if (!pulse_on(s, t)) continue;
        int sp = I32(s->data, SE_SHIELD), iv = I32(s->data, SE_INVIS);
        if (sp > shield) shield = sp;
        if (iv > invis) invis = iv;
    }
    (void)any;
    I32(ch, CH_SHIELD) = shield;   /* nothing in this libg writes these two: ours */
    I32(ch, CH_INVIS) = invis;
    if (g_superfix_dbg) {   /* MountUp (component type 21, char +0xa60 / count +0xa6c): its mount ticks +0x38 (0xb50be8) */
        int nc = I32(ch, 0xa6c); void **c = PTR(ch, 0xa60);
        for (int i = 0; c && i < nc && i < 64; i++) if (c[i] && I32(c[i], 0x10) == 21) {
            static void *last_c; static int last_v = -1;
            if (c[i] != last_c || I32(c[i], 0x38) != last_v) {
                if (I32(c[i], 0x38) == 0 || last_v <= 0 || I32(c[i], 0x38) % 10 == 0)
                    LOG("[superfix] st: mount ch %d ticks %d tick %d\n", I32(ch, 8), I32(c[i], 0x38), t);
                last_c = c[i]; last_v = I32(c[i], 0x38);
            }
        }
    }
    if (g_superfix_dbg && ((geti_t)B(0xb34de4))(ch) & 1) {   /* brawlers: HP drops, with the shield in force */
        static struct { int id, hp, tick; } hp[64]; static int nhp;
        int id = I32(ch, 8), h = I32(ch, 0x3c4), k;
        for (k = 0; k < nhp && hp[k].id != id; k++) {}
        if (k == nhp) { if (nhp == 64) return; hp[nhp++] = (typeof(hp[0])){id, h, t}; return; }
        if (h < hp[k].hp) LOG("[superfix] st: hp ch %d %d -> %d (-%d) tick %d shield %d invis %d\n", id, hp[k].hp, h, hp[k].hp - h, t, shield, invis);
        hp[k].hp = h; hp[k].tick = t;
    }
}

/* StatusEffect (type 36) areas, from superfix.c's area tick hook (0xb28c0c) */
typedef struct { void *a, *d; int start, end, next; } AreaRec;
#define AR_MAX 128
static AreaRec g_ar[AR_MAX]; static int g_nar;
typedef void (*apply_t)(void *, void *, int, int, void *, void *, int, int, int, int, int);   /* 0xb35dd0 */
void superfix_status_area(uint8_t *a) {
    if (!g_superfix || !mode() || !a) return;
    uint8_t *d = ((getp_t)B(0xb6a384))(a);
    if (!d || I32(d, 0x118) != 36) return;
    void *self = PTR(d, 0x88), *ally = PTR(d, 0x90), *enemy = PTR(d, 0x98);
    uint8_t *owner = PTR(a, 0x108);
    if ((!self && !ally && !enemy) || U8(d, 0x1b6) || !owner) return;   /* IgnoreCharacters: it slows projectiles */
    int t = now_tick(a), interval = I32(d, 0xa0), k;
    for (k = 0; k < g_nar && !(g_ar[k].a == a && g_ar[k].d == d && g_ar[k].start == I32(a, 0xd0)); k++) {}
    if (k == g_nar) {   /* first tick of this area: forget the areas whose end tick (+0xd8, start +0xd0) has passed */
        for (int i = g_nar - 1; i >= 0; i--) if (g_ar[i].end < t) g_ar[i] = g_ar[--g_nar];
        if (g_nar >= AR_MAX) { g_nar = 0; LOG("[superfix] status: area records full, reset\n"); }
        k = g_nar++;
        g_ar[k] = (AreaRec){a, d, I32(a, 0xd0), I32(a, 0xd8), U8(d, 0x17c) && interval > 0 ? t + interval : t};
    }
    if (t < g_ar[k].next) return;
    g_ar[k].next = interval > 0 ? t + interval : 0x7fffffff;
    int n; void **v = objects(owner, &n); if (!v) return;
    int team = ((geti_t)B(0xb6a820))(a), player = ((geti_t)B(0xb6a754))(a), ax = I32(a, 0x38), ay = I32(a, 0x3c);
    int heroes = U8(d, 0x1b4), hits = 0;
    g_narea_apply++;
    for (int i = 0; i < n; i++) {
        uint8_t *o = v[i];
        if (!is_char(o) || I32(o, 0x3c4) <= 0) continue;
        if (heroes && !(((geti_t)B(0xb34de4))(o) & 1)) continue;
        long dx = I32(o, 0x38) - ax, dy = I32(o, 0x3c) - ay, r = ((int (*)(void *, void *))B(0xb2b298))(d, o);
        if (dx * dx + dy * dy > r * r) continue;
        void *st = o == owner ? self : ((geti_t)B(0xb6a820))(o) == team ? ally : enemy;
        if (!st) continue;
        if (g_superfix_dbg) LOG("[superfix] st: area %d (owner ch %d) tick %d: %s %d -> ch %d at %ld u\n", data_id(d), I32(owner, 8), t,
                                st == enemy ? "enemy" : o == owner ? "self" : "ally", data_id(st), I32(o, 8), (long)__builtin_sqrt((double)(dx * dx + dy * dy)));
        ((apply_t)B(0xb35dd0))(o, st, player, team, a, owner, I32(a, 0xf8), 0, -1, -1, 0);
        hits++; g_narea_hits++;
        v = objects(owner, &n); if (!v) break;   /* a trigger cannot run here (deferred to the char tick), but stay safe */
    }
}

static void on_mount_trigger(uint64_t *ctx) {   /* dbg: 0xba0244(component, skillData), the MountUp "skill triggered" handler */
    uint8_t *comp = (uint8_t *)ctx[0], *sd = (uint8_t *)ctx[1]; uint8_t *ch = comp ? PTR(comp, 8) : NULL;
    LOG("[superfix] st: mount handler ch %d skill type %d bt %d mounted %d\n", ch ? I32(ch, 8) : -1, sd ? I32(sd, 0x1ac) : -1,
        sd ? I32(sd, 0x1a8) : -1, comp ? I32(comp, 0x38) : -1);
}
void superfix_status_install(void) {
    const char *e = getenv("NB_SUPERFIX_STATUS");
    if (e && !atoi(e)) { LOG("[superfix] status effects off (NB_SUPERFIX_STATUS=0)\n"); return; }
    if (*(uint32_t *)B(0xba3230) != 0xaa1f03e0 || *(uint32_t *)B(0xba3234) != 0xd65f03c0 ||   /* mov x0, xzr; ret */
        *(uint32_t *)B(0xba3260) != 0xd65f03c0 || *(uint32_t *)B(0xba3264) != 0xd65f03c0 ||   /* ret; ret */
        *(uint32_t *)B(0xb3941c) != 0xa9bd7bfd) {                                              /* stp x29, x30, [sp, #-48]! */
        LOG("[superfix] status: libg bytes differ from v68.279, status effects left out\n"); return;
    }
    hook_replace(g_libg, 0xba3230, (void *)my_status_add);
    hook_replace(g_libg, 0xba3260, (void *)my_status_remove);
    hook_replace(g_libg, 0xba3264, (void *)my_status_remove);
    hook_enter(g_libg, 0xb3941c, on_char_update);
    if (g_superfix_dbg && *(uint32_t *)B(0xba0244) == 0xa9bd7bfd) hook_enter(g_libg, 0xba0244, on_mount_trigger);
    LOG("[superfix] status effects installed (store, shield / invisibility aggregates, damage / heal triggers, type-36 areas)\n");
}
void superfix_status_status(void) {
    LOG("[superfix] status: added %ld, refreshed %ld, full %ld, ended %ld, removed %ld, live %d; damage %ld, heal %ld; "
        "type-36 area applies %ld, characters hit %ld; MountUp ticks %ld\n", g_nadd, g_nrefresh, g_nfull, g_nexpire, g_nremove, g_nst, g_ndmg, g_nheal,
        g_narea_apply, g_narea_hits, g_nmount);
}
