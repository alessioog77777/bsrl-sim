"""Equality + timing of the train_fast parts on real game batches (WSL, after `source env.sh`; boots --n instances).

    python -m test_fast --n 2 --polls 400 [--ckpt runs/fast_eq/ckpt.pt --base runs/fast_eq/base.pt]   # GPU if there is one

1. Plays train_ppo-style battles (self-play with league opponents + k vs native) on --n instances, the policy acting
   through fast_act.StackedActor (learner = --ckpt, opponents = --base), and records every poll's raw inputs.
2. Observations: obs_v5.ObsV5 and fast_obs.FastObsV5 (same seed) over the recorded polls must be bit-equal;
   train_ppo.priv == fast_obs.priv_fast (live, per poll).
3. Acting: every poll's batch through model.act_stacked (the train_ppo path) vs StackedActor: max |diff| < 1e-4.
4. Per-poll build / priv / act timings, old vs new.
Intents are on (--intent-scale, default 0.25; p_steer / p_rand_intent 1): the policy input s carries real intent blocks
(env_v5.Env.intents, shared by both paths), so the intent columns of s go through both actors too.
"""
import argparse, time
import numpy as np
import torch
from nbpc import diversity as D
import env_v5 as E, obs_v5 as O, fast_obs as FO, fast_act as FA
from model import Policy, HID
from train_ppo import priv, policy_s


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=2); ap.add_argument("--polls", type=int, default=400)
    ap.add_argument("--ckpt", default="", help="learner checkpoint ('' = a fresh Policy with every input column in use)")
    ap.add_argument("--base", default="", help="league checkpoint ('' = another fresh Policy)"); ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--intent-scale", type=float, default=0.25, help="intents on (> 0): real intent blocks in s")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); rng_b = np.random.default_rng(a.seed); loc = "Survival604"
    made = []
    def net_of(p):
        m = Policy().to(dev).eval()
        if p: m.load_state_dict(torch.load(p, map_location="cpu", weights_only=False)["model"], strict=False)
        else:                                            # fresh: the zero-initialised branches made live, so they are compared too
            torch.manual_seed(a.seed * 10 + len(made)); made.append(p)
            with torch.no_grad():
                m.hemb.weight.normal_(0, 0.1); m.ent[0].weight.normal_(0, 0.1); m.vp[2].weight.normal_(0, 0.1); m.vec[0].weight[:, O.S_INT:].normal_(0, 0.1)
        return m
    nets = [net_of(a.ckpt), None, None, net_of(a.base), net_of(a.base)]    # learner / (eval) / (base) / league slots 0, 1

    def battle_fn(rng, i):
        tag = {"loc": loc, "chr": ["Sniper"] * 10, "team_size": 1, "groups": [0] * 10}
        cfg = {"div": D.make_div(location=loc, chr=tag["chr"]), "tag": tag, "end": "agents", "seed": int(rng_b.integers(10_000_000, 2 ** 31 - 1))}
        if rng_b.random() < 0.6:
            opp = sorted(rng_b.choice(10, int(rng_b.integers(0, 8)), replace=False).tolist())
            return {**cfg, "ctl": list(range(10)), "tag": {**tag, "kind": "sp", "opp": opp, "slot": int(rng_b.integers(2)), "snap": "x"}}
        k = int(rng_b.integers(1, 6))
        return {**cfg, "ctl": sorted(rng_b.choice(10, k, replace=False).tolist()), "tag": {**tag, "kind": "k%d" % k}}

    E.Env.slider_scale = 0.0; E.Env.intent_scale, E.Env.p_steer, E.Env.p_rand_intent = a.intent_scale, 1.0, 1.0
    env = E.Env(a.n, battle_fn, ticks=2, fair=True, max_ticks=4000, verbose=True, seed=a.seed, tag="fasteq%d" % a.seed)
    ob_live = FO.FastObsV5(seed=a.seed); actor = FA.StackedActor(dev); hid = {}; z = np.zeros(HID, np.float32)
    rec, dmax, t_priv = [], {"fwd": 0, "fwd_allclose_ratio": 0, "logp": 0, "v": 0, "h": 0}, np.zeros(2); rows = 0; n_int = 0
    try:
        while len(rec) < a.polls:
            obs, ids, tr = env.poll(min_ready=a.n, max_wait=0.004)
            for (aid, rew, done, t, rank) in tr:
                if done: hid.pop(aid, None)
            if not len(ids): env.send([], []); continue
            rec.append((obs["vec"].copy(), obs["grid"].copy(), env.last_meta.copy(), ids.copy()))
            o = ob_live.build(obs["vec"], obs["grid"], env.last_meta, ids)
            tq = time.perf_counter(); p0 = priv(env, ids); t1 = time.perf_counter(); p1 = FO.priv_fast(env, ids); t2 = time.perf_counter()
            assert np.array_equal(p0, p1), "priv"
            t_priv += [t1 - tq, t2 - t1]
            s = policy_s(env, ids, o); h = np.stack([hid.get(int(x), z) for x in ids]); n_int += int(o["intent"].any(1).sum())
            kind = np.array([t["kind"] for t in env.last_tags]); opp = np.array([int(x) & 15 in t.get("opp", ()) for x, t in zip(ids, env.last_tags)], bool)
            gidx = np.where(opp, 3 + np.array([t.get("slot", 0) for t in env.last_tags]), 0)
            d = FA.compare(nets, gidx, s, o["ent"], o["grid"], h, p1, actor)
            dmax = {k: max(dmax[k], d[k]) for k in d}
            A, lp, v, h2 = actor(nets, gidx, s, o["ent"], o["grid"], h, p1); rows += len(ids)
            for x, hh in zip(ids, h2): hid[int(x)] = hh
            env.send(ids, A)
    finally:
        env.close()
    print("acting on %d real polls (%d rows, learner = %s): max |d| vs model.act_stacked %s" % (len(rec), rows, a.ckpt, {k: "%.1e" % x for k, x in dmax.items()}))
    assert FA.ok(dmax), dmax
    cnt = {}
    for r in env.episodes:
        for k, v in (r.get("intents") or {}).items(): cnt[k] = [x + y for x, y in zip(cnt.get(k, [0] * 4), v)]
    print("intents: %d of %d rows had an active intent; ended episodes' counts [started, achieved, timeout, invalid] %s" % (n_int, rows, cnt))
    assert not a.intent_scale or n_int > 0
    print("priv == priv_fast on every poll; per poll %.3f ms -> %.3f ms" % tuple(t_priv / len(rec) * 1e3))
    # observations on the recorded polls: same seed, bit-equal; timing
    res = {}
    for name, ob in (("ObsV5", O.ObsV5(seed=11)), ("FastObsV5", FO.FastObsV5(seed=11))):
        outs, t = [], 0.0
        for vec, grid, meta, ids in rec:
            x = (vec.copy(), grid.copy(), meta.copy(), ids)
            tq = time.perf_counter(); outs.append(ob.build(*x)); t += time.perf_counter() - tq
        res[name] = (outs, t / len(rec) * 1e3)
    bad = [i for i, (x, y) in enumerate(zip(res["ObsV5"][0], res["FastObsV5"][0])) if not FO.same(x, y)]
    print("observations: %d recorded polls, %d differ (bit-equality); build per poll ObsV5 %.2f ms -> FastObsV5 %.2f ms (mean B %.1f)"
          % (len(rec), len(bad), res["ObsV5"][1], res["FastObsV5"][1], rows / len(rec)))
    assert not bad, bad[:10]
    print("test_fast ok")


if __name__ == "__main__":
    main()
