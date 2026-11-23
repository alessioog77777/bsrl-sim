"""Policy for the ObsV5 observation: entity set + seg-class grid + self / conditioning vector -> GRU -> heads.

    net = Policy(); logits, v, h = net(s, ent, grid, h)      # one decision; s = cat[self (8), cond (26)], h (B, HID)
    logits, v = net.seq(s, ent, grid, h0)                     # (N, L, ...) chunks from stored h0 -> flat (N * L) outputs
    logits = move 17 | fire 3 (none / attack / super) | aim 360 (pointer over visible enemies / boxes + free) | distance 8

    python -m model      # self-check (CPU, no game)
The critic also gets `priv` (PRIV = 40: every player's true offset, hp, alive; train_ppo.priv) through a zero-initialised
branch, so the policy heads never see it and old checkpoints load with strict=False.
Stage 2 (BSRL v2 plan), both zero-initialised so s3 / s4 checkpoints act the same (load_state_dict pads):
* lead hint: 2 extra entity columns computed here from the obs (no obs / live change): sin of the angle from a foe to
  where a shot fired now meets it, and the flight time; own projectile speed = cond stats[3].
* per-brawler embedding `hemb` (a learned trunk bias per hero): the hero is found by matching cond's 13 stats against
  the `hero_stats` table (add_heroes; saved in the checkpoint, rows never reorder), unknown = the last row.
Obs v5.3 (obs_v5.OBS_VERSION): 4 area-effect columns at the end of the observation's entity row, before the lead columns;
load_state_dict zero-pads v5.2 checkpoints there (same outputs).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import obs_v5 as O
from models4 import UnfoldConv2d, dist_logp_ent      # nirkt (MIOpen-safe conv; masked log-prob / entropy / sampling)

NVEC, HID, PRIV = [17, 3, 360, 8], 256, 40
_CAND = list(range(0, 9)) + list(range(25, 31))       # ent slots that can be aimed at: the enemy block + the power boxes
_FOE = [1 + O.ENT.index(c) for c in ("enemy", "enemy_summon", "power_box")]
N_LEAD, HMAX, N_STAT = 2, 160, 13                     # lead columns; hero table rows (+1 unknown); cond's brawler stats
F_IN = O.F_ENT + N_LEAD                               # entity row width the network sees
_PSPEED = 4000 / 300 / 20                             # cond stats[3] (projectile Speed / 4000, units/s) -> tiles / tick
LAT = 3.0                                             # ticks between the frame and the shot: obs_v5 NOISE latency 1-2 decisions x 2


def lead(ent, s):
    """(..., N_ENT, 2) aim hint per present foe row: sin of the angle between the foe as seen and where a shot fired now
    meets it, and the flight time / 2 s. 0 for other rows, melee and targets faster than the shot. The frame is LAT ticks
    old when the shot leaves (obs latency 1-2 decisions, sim and live), so the foe is first moved on by its velocity
    relative to ours (s3 aimed behind movers: scorecard lead slope -0.41, HISTORY 82).
    ent (..., N_ENT, F_ENT): dx, dy in 10-tile units, vx, vy tiles / tick; s (..., N_SELF + N_COND), self[6:8] own velocity."""
    p0 = ent[..., O.XY:O.XY + 2] * 10.0; v = ent[..., O.VEL:O.VEL + 2]
    p = p0 + (v - s[..., None, 6:8]) * LAT
    sp = (s[..., O.N_SELF + 3] * _PSPEED)[..., None]
    k = sp * sp - (v * v).sum(-1); bh = (p * v).sum(-1); c = (p * p).sum(-1)
    ok = ((k > 1e-4) & (ent[..., 0] > 0.5) & (ent[..., _FOE].sum(-1) > 0.5)).to(ent.dtype)
    t = ((bh + torch.sqrt((bh * bh + k * c).clamp_min(0.0))) / k.clamp_min(1e-4)).clamp(0.0, 40.0)
    q = p + v * t[..., None]
    sn = (p0[..., 0] * q[..., 1] - p0[..., 1] * q[..., 0]) / (torch.sqrt((p0 * p0).sum(-1) * (q * q).sum(-1)) + 1e-6)
    return torch.stack([sn, t / 40.0], -1) * ok[..., None]


def hero_index(s, table):
    """(...,) row of the hero whose stats match cond's (s[..., 8:21]); HMAX when none does. table (..., HMAX, 13)"""
    d2 = ((s[..., None, O.N_SELF:O.N_SELF + N_STAT] - table) ** 2).sum(-1)
    m, i = d2.min(-1)
    return torch.where(m < 1e-6, i, torch.full_like(i, HMAX))


class Policy(nn.Module):
    def __init__(self, d=512):
        super().__init__()
        self.emb = nn.Embedding(O.UNKNOWN + 1, 8)
        self.grid = nn.Sequential(UnfoldConv2d(8, 32, 3, 2, 1), nn.ReLU(), UnfoldConv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                  nn.Flatten(), nn.Linear(64 * 7 * 7, 256), nn.ReLU())
        self.ent = nn.Sequential(nn.Linear(F_IN, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
        self.vec = nn.Sequential(nn.Linear(O.N_SELF + O.N_COND, 128), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(512, d), nn.ReLU())
        self.hemb = nn.Embedding(HMAX + 1, d); nn.init.zeros_(self.hemb.weight)
        self.register_buffer("hero_stats", torch.full((HMAX, N_STAT), 1e4))   # empty rows match nothing
        self.gru = nn.GRUCell(d, HID)
        self.pi = nn.Linear(d + HID, 17 + 3 + 8); self.v = nn.Linear(d + HID, 1)
        self.cand = nn.Sequential(nn.Linear(d + HID + F_IN, 128), nn.ReLU(), nn.Linear(128, 3))
        self.vp = nn.Sequential(nn.Linear(d + HID + PRIV, 256), nn.ReLU(), nn.Linear(256, 1))
        nn.init.zeros_(self.vp[2].weight); nn.init.zeros_(self.vp[2].bias)
        self.free = nn.Linear(d + HID, 360); self.free_score = nn.Linear(d + HID, 1)
        nn.init.orthogonal_(self.pi.weight, 0.01); nn.init.zeros_(self.pi.bias); nn.init.zeros_(self.free.weight)
        with torch.no_grad():   # the lead columns start unused
            self.ent[0].weight[:, O.F_ENT:] = 0; self.cand[0].weight[:, -N_LEAD:] = 0
        self.register_buffer("bins", torch.arange(360).float())

    def add_heroes(self, stats):
        """put each (13,) stats row not in the table yet into the next free row (order-stable across resumes) -> count"""
        t = self.hero_stats
        with torch.no_grad():
            for r in torch.as_tensor(np.asarray(stats, np.float32), device=t.device):
                if hero_index(torch.cat([r.new_zeros(O.N_SELF), r]), t) < HMAX:
                    continue
                free = (t[:, 0] >= 1e3).nonzero()
                assert len(free), "hero table full (HMAX %d)" % HMAX
                t[free[0, 0]] = r
        return int((t[:, 0] < 1e3).sum())

    def load_state_dict(self, sd, strict=True, **kw):
        """old checkpoints: the entity input weights grew by the lead columns and (obs v5.3) by the area-effect columns
        between the observation's row and the lead columns -> zero-padded in place (same outputs); no hemb -> zeros (a
        reused league slot must not keep the previous snapshot's rows; the hero table may stay)"""
        own = self.state_dict(); sd = dict(sd)
        if "hemb.weight" not in sd: sd["hemb.weight"] = torch.zeros_like(own["hemb.weight"])
        for k in ("ent.0.weight", "cand.0.weight"):     # input = [x part (cand)] | obs row (F_ENT) | lead (N_LEAD)
            v, o = sd.get(k), own[k]
            if v is None or v.dim() != 2 or v.shape[1] >= o.shape[1]: continue
            pre = o.shape[1] - F_IN; n_lead = v.shape[1] - pre - O.F_ENT_V52   # v5.2 row, with (2) or without (0) lead
            assert n_lead in (0, N_LEAD) and v.shape[0] == o.shape[0], (k, tuple(v.shape))
            t = torch.zeros_like(o); t[:, :pre + O.F_ENT_V52] = v[:, :pre + O.F_ENT_V52]
            if n_lead: t[:, -N_LEAD:] = v[:, -N_LEAD:]
            sd[k] = t
        for k, v in sd.items():
            o = own.get(k)
            if o is not None and v.shape != o.shape and v.dim() == o.dim() and all(a <= b for a, b in zip(v.shape, o.shape)):
                t = torch.zeros_like(o); t[tuple(slice(0, n) for n in v.shape)] = v; sd[k] = t
        return super().load_state_dict(sd, strict=strict, **kw)

    def encode(self, s, ent, grid):
        ent = torch.cat([ent, lead(ent, s)], -1)
        e = self.ent(ent).masked_fill(ent[..., :1] < 0.5, 0.0).amax(1)             # set pooling over the present slots
        g = self.grid(self.emb(grid.long()).permute(0, 3, 1, 2))
        x = self.trunk[0](torch.cat([self.vec(s), e, g], -1)) + self.hemb(hero_index(s, self.hero_stats))
        return F.relu(x)

    def heads(self, x, ent, s, priv=None):
        """x = cat[trunk, gru]. The aim head is nirkt's PointerAim: a bump at every target's angle (+ a learned lead of
        up to 25 degrees, learned width) mixed with a free 360-bin component; log-probabilities."""
        mv, fi, di = torch.split(self.pi(x), [17, 3, 8], -1)
        c = ent[:, _CAND]; c = torch.cat([c, lead(c, s)], -1); B, K, _ = c.shape
        here = (c[..., 0] > 0.5) & (c[..., _FOE].sum(-1) > 0.5)
        o = self.cand(torch.cat([x[:, None].expand(B, K, x.shape[-1]), c], -1))
        mu = torch.rad2deg(torch.atan2(c[..., O.XY + 1], c[..., O.XY])) % 360.0 + 25.0 * torch.tanh(o[..., 1])
        dd = (self.bins - mu[..., None]).abs() % 360.0; dd = torch.minimum(dd, 360.0 - dd)
        bump = torch.log_softmax(-0.5 * (dd / (F.softplus(o[..., 2]) + 0.7)[..., None]) ** 2, -1)
        mix = torch.log_softmax(torch.cat([o[..., 0].masked_fill(~here, -1e9), self.free_score(x)], -1), -1)
        comp = torch.cat([bump, torch.log_softmax(self.free(x), -1)[:, None]], 1)
        v = self.v(x).squeeze(-1)
        if priv is not None: v = v + self.vp(torch.cat([x, priv], -1)).squeeze(-1)
        return (mv, fi, torch.logsumexp(mix[..., None] + comp, 1), di), v

    def forward(self, s, ent, grid, h, priv=None):
        x = self.encode(s, ent, grid); h = self.gru(x, h)
        (mv, fi, ai, di), v = self.heads(torch.cat([x, h], -1), ent, s, priv)
        return (mv, mask_super(fi, s), ai, di), v, h

    def seq(self, s, ent, grid, h, priv=None):
        N, L = s.shape[:2]; f = lambda t: t.flatten(0, 1)
        x = self.encode(f(s), f(ent), f(grid)).view(N, L, -1); hs = []
        for t in range(L):
            h = self.gru(x[:, t], h); hs.append(h)
        (mv, fi, ai, di), v = self.heads(f(torch.cat([x, torch.stack(hs, 1)], -1)), f(ent), f(s), None if priv is None else f(priv))
        return (mv, mask_super(fi, f(s)), ai, di), v


def mask_super(fi, s):
    """fire head (none, attack, super): super only while obs_v5 self[2] (super ready) is set; s = the self (+ cond) rows.
    HISTORY 77: 12 % of decisions pressed an uncharged super (live that blocks the fire finger 150 ms)"""
    return fi.masked_fill((s[..., 2] < 0.5)[..., None] & (torch.arange(fi.shape[-1], device=fi.device) == 2), -1e9)


class _Act(nn.Module):
    """acting forward of one Policy with the GRU step written out (torch.func.vmap has no gru_cell rule; same math as
    nn.GRUCell): used by act_stacked"""
    def __init__(self, p): super().__init__(); self.p = p

    def forward(self, s, ent, grid, h, priv):
        p = self.p; x = p.encode(s, ent, grid); g = p.gru
        ir, iz, i_n = F.linear(x, g.weight_ih, g.bias_ih).chunk(3, -1); hr, hz, h_n = F.linear(h, g.weight_hh, g.bias_hh).chunk(3, -1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz); h = (1 - z) * torch.tanh(i_n + r * h_n) + z * h
        (mv, fi, ai, di), v = p.heads(torch.cat([x, h], -1), ent, s, priv)
        return (mv, mask_super(fi, s), ai, di), v, h


_ACT_META = None


def act_stacked(nets, xs):
    """one vmapped forward for several Policies (different weights) on their own row sets (speed round, HISTORY 68: the
    per-net loop paid ~8 ms of fixed GPU cost per net). nets: list of Policy; xs: list of (s, ent, grid, h, priv) device
    tensors, one tuple per net (any batch sizes, padded to the largest). -> logits tuple, v, h with a leading net axis
    (G, Bmax, ...); rows past each net's own B are padding."""
    global _ACT_META
    from torch.func import functional_call, vmap
    acts = [_Act(n) for n in nets]
    if _ACT_META is None: _ACT_META = _Act(Policy()).to("meta")
    st = lambda ds: {k: torch.stack([d[k].detach() for d in ds]) for k in ds[0]}   # detached: the learner has grads, frozen nets not
    P, Bf = st([dict(a.named_parameters()) for a in acts]), st([dict(a.named_buffers()) for a in acts])
    Bm = max(x[0].shape[0] for x in xs)
    pad = lambda t: torch.cat([t, t.new_zeros((Bm - t.shape[0],) + t.shape[1:])]) if t.shape[0] < Bm else t
    X = [torch.stack([pad(x[i]) for x in xs]) for i in range(5)]
    r = vmap(lambda p, b, *x: functional_call(_ACT_META, (p, b), x))(P, Bf, *X)
    return r[0], r[1], r[2]


def masked_kl(ref, new):
    """KL(ref || new) of the factored policy: move + fire + P_ref(fire > 0) aim + P_ref(super) distance (nirkt F1.4)"""
    kl = []
    for a, b in zip(ref, new):
        pa, pb = torch.log_softmax(a.float(), -1), torch.log_softmax(b.float(), -1)
        kl.append((pa.exp() * (pa - pb)).sum(-1))
    pf = torch.softmax(ref[1].float(), -1)
    return kl[0] + kl[1] + (1 - pf[:, 0]) * kl[2] + pf[:, 2] * kl[3]


def _check():
    torch.manual_seed(0); net = Policy(); N, L = 3, 4
    s = torch.randn(N, L, O.N_SELF + O.N_COND); ent = torch.zeros(N, L, O.N_ENT, O.F_ENT)
    grid = torch.randint(0, O.UNKNOWN + 1, (N, L, O.G, O.G), dtype=torch.uint8)
    ent[:, :, 0, 0] = 1; ent[:, :, 0, 1 + O.ENT.index("enemy")] = 1; ent[:, :, 0, O.XY] = 0.3      # an enemy straight right
    ent[:, :, 1, 0] = 1; ent[:, :, 1, 1 + O.ENT.index("teammate")] = 1; ent[:, :, 1, O.XY + 1] = 0.3   # a teammate below
    pv = torch.randn(N, L, PRIV); h = torch.zeros(N, HID); outs = []
    assert torch.equal(net.seq(s, ent, grid, h, pv)[1], net.seq(s, ent, grid, h)[1])     # zero-init branch: an old checkpoint's value
    nn.init.normal_(net.vp[2].weight, std=0.1)
    for t in range(L):                                    # step by step == seq from the same h0
        lg, v, h = net(s[:, t], ent[:, t], grid[:, t], h, pv[:, t]); outs.append((lg, v))
    lg2, v2 = net.seq(s, ent, grid, torch.zeros(N, HID), pv)
    assert [x.shape[-1] for x in lg2] == NVEC and v2.shape == (N * L,)
    assert torch.allclose(torch.stack([o[1] for o in outs], 1).flatten(), v2, atol=1e-5)
    assert torch.allclose(torch.stack([o[0][2] for o in outs], 1).flatten(0, 1), lg2[2], atol=1e-4)
    p = lg2[2].exp(); assert torch.allclose(p.sum(-1), torch.ones(N * L), atol=1e-4)
    near = lambda a: p[:, [(a + k) % 360 for k in range(-30, 31)]].sum(-1).mean().item()
    assert near(0) > 3 * near(90), (near(0), near(90))    # mass at the enemy (0 deg), none pulled to the teammate (90 deg)
    lp, en, a = dist_logp_ent(lg2); assert a.shape == (N * L, 4) and torch.isfinite(lp).all()
    assert masked_kl(lg2, lg2).abs().max() < 1e-5
    (lp.mean() + v2.mean()).backward()
    _check_stage2()
    print("model ok: %d parameters" % sum(x.numel() for x in net.parameters()))


def _check_stage2():
    """lead hint math; an old-format checkpoint (no lead columns / hero table) acts the same after loading; hero rows"""
    s = torch.zeros(1, O.N_SELF + O.N_COND); s[0, O.N_SELF + 3] = 1.0                    # shot 0.667 tiles / tick
    ent = torch.zeros(1, O.N_ENT, O.F_ENT); ent[0, :2, 0] = 1; ent[0, :2, 1 + O.ENT.index("enemy")] = 1
    ent[0, :2, O.XY] = 0.3; ent[0, 0, O.VEL + 1] = 0.12                                  # 3 tiles right, slot 0 walks down
    ld = lead(ent, s)[0]; sp = 4000 / 6000                                               # slot 0: brute-force intercept
    ts = np.linspace(0, 40, 400001); y0 = 0.12 * LAT                                     # seen at y 0, now at 0.12 x LAT
    t = ts[np.argmin(np.abs(np.hypot(3.0, y0 + 0.12 * ts) - sp * ts))]; qy = y0 + 0.12 * t
    assert abs(ld[0, 1] - t / 40) < 1e-3 and abs(ld[0, 0] - qy / np.hypot(3.0, qy)) < 1e-3, (ld[0], t, qy)
    assert abs(ld[1, 0]) < 1e-6 and abs(ld[1, 1] - 3.0 / sp / 40) < 1e-3 and ld[2:].abs().max() == 0, ld[:3]
    s2 = s.clone(); s2[0, 6:8] = torch.tensor([0.0, 0.12])     # we walk along with it: no latency shift, still a lead
    t0 = 3.0 / (sp ** 2 - 0.12 ** 2) ** 0.5
    assert abs(lead(ent, s2)[0, 0, 0] - 0.12 * t0 / np.hypot(3.0, 0.12 * t0)) < 1e-3
    torch.manual_seed(1); N, L = 2, 3
    s = torch.randn(N, L, O.N_SELF + O.N_COND); s[..., O.N_SELF:O.N_SELF + N_STAT] = s[:, :1, O.N_SELF:O.N_SELF + N_STAT]
    s[..., O.N_SELF + 3] = 1.0                                                            # one hero per sequence
    ent = torch.zeros(N, L, O.N_ENT, O.F_ENT); ent[..., :3, 0] = 1; ent[..., :3, 1 + O.ENT.index("enemy")] = 1
    ent[..., :3, O.XY:O.HP] = torch.randn(N, L, 3, 4) * 0.2
    grid = torch.randint(0, O.UNKNOWN + 1, (N, L, O.G, O.G), dtype=torch.uint8); h0 = torch.zeros(N, HID)
    ent[..., :3, O.RAD:] = torch.rand(N, L, 3, 4)                                        # new inputs present: an old net ignores them
    a = Policy(); nn.init.normal_(a.pi.weight, std=0.1); na = O.F_ENT - O.F_ENT_V52; pre = a.cand[0].weight.shape[1] - F_IN
    with torch.no_grad():                                                                 # a v5.2 net: lead used, no area columns
        nn.init.normal_(a.ent[0].weight[:, O.F_ENT:], std=0.3); nn.init.normal_(a.cand[0].weight[:, -N_LEAD:], std=0.3)
        a.ent[0].weight[:, O.F_ENT_V52:O.F_ENT] = 0; a.cand[0].weight[:, pre + O.F_ENT_V52:pre + O.F_ENT] = 0
    ref = a.seq(s, ent, grid, h0)[0]
    old = {k: v for k, v in a.state_dict().items() if k not in ("hemb.weight", "hero_stats")}
    v52 = lambda w, p: torch.cat([w[:, :p + O.F_ENT_V52], w[:, p + O.F_ENT:]], 1)        # the v5.2 layout: row 16 | lead 2
    old["ent.0.weight"] = v52(old["ent.0.weight"], 0); old["cand.0.weight"] = v52(old["cand.0.weight"], pre)
    assert old["ent.0.weight"].shape[1] == O.F_ENT_V52 + N_LEAD and na == 4
    c = Policy(); c.load_state_dict({**old, "ent.0.weight": old["ent.0.weight"][:, :O.F_ENT_V52]}, strict=False)   # pre-lead format
    assert torch.equal(c.ent[0].weight[:, :O.F_ENT_V52], a.ent[0].weight[:, :O.F_ENT_V52]) and not c.ent[0].weight[:, O.F_ENT_V52:].any()
    b = Policy(); b.load_state_dict(old, strict=False)
    assert b.add_heroes(s[:, 0, O.N_SELF:O.N_SELF + N_STAT]) == N and b.add_heroes(s[:1, 0, O.N_SELF:O.N_SELF + N_STAT]) == N
    got = b.seq(s, ent, grid, h0)[0]
    assert max((x - y).abs().max().item() for x, y in zip(ref, got)) < 1e-6, "old checkpoint acts differently"
    assert (hero_index(s, b.hero_stats) < N).all() and (hero_index(torch.randn(4, s.shape[-1]), b.hero_stats) == HMAX).all()
    nn.init.normal_(b.hemb.weight, std=0.5)
    assert max((x - y).abs().max().item() for x, y in zip(ref, b.seq(s, ent, grid, h0)[0])) > 1e-3   # the embedding is used


if __name__ == "__main__":
    _check()
