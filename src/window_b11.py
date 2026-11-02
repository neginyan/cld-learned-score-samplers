"""H5 (B11): which interval of the learned score plants the seed of the blow-up?  Plus the tracking of the outliers.
  python window_b11.py <metric> <net_seed> <samp_seed>
     -> results/win_<metric>_n<n>_s<s>.json        (every window: L, outliers, broken)
     -> results/track_<metric>_n<n>_s<s>.json      (outliers followed back through sigma = 1 ... 0.005; comparison points)
     -> results/pool_<metric>_n<n>_s<s>.npz        (learned-minus-exact path difference in Z at the band edges; used by H6)
Probability-flow ODE, Heun, h ~ 0.002 (K = 4496), prior draws seed 100 + s: the grid and the draws of B9/B10.
The score is learned inside a window W = [sigma_1, sigma_2] (a run of adjacent bands) and exact outside.  A step starting at t_k
belongs to the band of sigma(t_k) (the rule of B10).  Windows with the same top edge share one learned integration from the
exact path's state at that edge; its states at the lower edges are continued with the exact score.  This gives the same numbers
as integrating each window from the start (the steps are identical)."""
import sys, json, time
import numpy as np
from learned_b11 import load, stats, far_mask, N
from cld_mixture import drift_fn, mode_stats, POS

EDGES = [40.0, 1.0, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]          # 8 bands b1..b8 between consecutive edges
metric, ns, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
m, ls = load(metric, ns); p = m.proc; ex = m.score(); f = drift_fn(p, 'ode')
K = int(np.ceil(p.T / 0.002 / 16)) * 16; ts = np.linspace(p.T, 0.0, K + 1); sk = p.sigma(ts[:-1])
kb = [0] + [int(np.argmax(sk < c)) for c in EDGES[1:-1]] + [K]           # first step of each band; kb[8] = K (end)
y0 = m.prior(N, np.random.default_rng(100 + sd)); ref = mode_stats(m.truth(400000, np.random.default_rng(999)), m.mu)

def step(score, z, k):
    t, t2 = ts[k], ts[k + 1]; h = t - t2
    f1 = f(z, t, score(z, t)); zp = z - h * f1; f2 = f(zp, t2, score(zp, t2)); return z - 0.5 * h * (f1 + f2)   # == run_sampler 'ode'

def integrate(score, z, k0, k1, keep=()):
    saved = {}
    with np.errstate(all='ignore'):
        for k in range(k0, k1):
            if k in keep: saved[k] = z.copy()
            z = step(score, z, k)
    if k1 in keep: saved[k1] = z.copy()
    return z, saved

def summarize(zY):
    z = m.to_Z(zY); st = stats(z, m, ref)
    bad = ~np.isfinite(z).all(1); bad[~bad] = far_mask(z[~bad], m.mu); st['n_bad'] = int(bad.sum())
    st['broken'] = (not st['finite']) or abs(st['L']) > 1 or st['n_far'] / N > 0.01
    return st, bad

t0 = time.time()
zE, E = integrate(ex, y0.copy(), 0, K, set(kb))                        # exact only; E[kb[j]] = state at the top of band j+1
out = {'metric': metric, 'net_seed': ns, 'seed': sd, 'N': N, 'K': K, 'edges': EDGES, 'k_edge': kb,
       'steps_per_band': [kb[j + 1] - kb[j] for j in range(8)], 'windows': {}}
out['exact_only'] = summarize(zE)[0]
L0 = None
for j2 in range(8):                                                      # top edge index of the window (sigma_2 = EDGES[j2])
    zL, S = integrate(ls, E[kb[j2]].copy(), kb[j2], K, set(kb[j2 + 1:]))
    if j2 == 0: L0 = S; zL0 = zL
    for j1 in range(j2 + 1, 9):                                          # bottom edge index (sigma_1 = EDGES[j1])
        zW = S[kb[j1]] if j1 < 8 else zL
        if j1 < 8: zW = integrate(ex, zW.copy(), kb[j1], K)[0]
        st = summarize(zW)[0]
        out['windows'][f'{EDGES[j1]}-{EDGES[j2]}'] = {'lo': EDGES[j1], 'hi': EDGES[j2], 'bands': list(range(j2 + 1, j1 + 1)), **st}
    print(metric, ns, sd, 'top %g done (%.0f s)' % (EDGES[j2], time.time() - t0), flush=True)
st, bad = summarize(zL0); out['learned_only'] = st; out['sec'] = time.time() - t0
json.dump(out, open(f'results/win_{metric}_n{ns}_s{sd}.json', 'w'), indent=1)

# ---------------- tracking of the outliers (observation) and the path-difference pool for H6
mt, Pi = ex.params(p.T); d = y0[:, None] - mt[None]; dmah0 = np.sqrt(np.einsum('nki,ij,nkj->nk', d, Pi, d).min(1))
rng = np.random.default_rng(1200 + 10 * ns + sd)
ok = np.where(~bad)[0]
tail_ok = ok[(dmah0[ok] >= 3.5) & (dmah0[ok] <= 5.0)]; bulk_ok = ok[dmah0[ok] < 2.0]
groups = {'outlier': np.where(bad)[0], 'tail_ref': rng.choice(tail_ok, min(200, len(tail_ok)), replace=False),
          'bulk_ref': rng.choice(bulk_ok, min(200, len(bulk_ok)), replace=False)}
CK = kb[1:] ; CS = EDGES[1:]                                             # checkpoints sigma = 1, 0.3, ..., 0.01, 0.005(end)
LS = {k: (L0[k] if k < K else zL0) for k in CK}; ES = {k: (E[k] if k < K else zE) for k in CK}
tr = {'metric': metric, 'net_seed': ns, 'seed': sd, 'sigma': CS, 'groups': {}}
pool = {}
for g, idx in groups.items():
    rows = []
    for k, sgm in zip(CK, CS):
        yl, ye = LS[k][idx], ES[k][idx]; zl, ze = m.to_Z(yl), m.to_Z(ye); fin = np.isfinite(zl).all(1)
        tk = ts[k]
        with np.errstate(all='ignore'):
            e = np.full(len(idx), np.nan)
            if fin.any():
                s, sh = ex(yl[fin], tk), ls(yl[fin], tk); sZ, dZ = s @ m.P, (sh - s) @ m.P
                e[fin] = np.sqrt((dZ**2).sum(1) / (sZ**2).sum(1))
        dz = np.linalg.norm(zl - ze, axis=1); pos = np.linalg.norm(zl[:, POS], axis=1)
        rows.append({'sigma': sgm, 'n_finite': int(fin.sum()), 'median_score_err': float(np.nanmedian(e)) if fin.any() else None,
                     'median_dev': float(np.median(dz[fin])) if fin.any() else None, 'max_dev': float(np.max(dz[fin])) if fin.any() else None,
                     'median_pos_norm': float(np.median(pos[fin])) if fin.any() else None,
                     'per_point': {'score_err': e.tolist(), 'dev': np.where(fin, dz, np.nan).tolist(), 'pos_norm': np.where(fin, pos, np.nan).tolist()}})
        if g == 'outlier': pool[f'sig{sgm}'] = (zl - ze)[fin]
    tr['groups'][g] = {'idx': idx.tolist(), 'prior_mahalanobis': dmah0[idx].tolist(), 'rows': rows}
tr['prior_mahalanobis_quantiles_all'] = np.quantile(dmah0, [0.5, 0.9, 0.99, 0.999]).tolist()
json.dump(tr, open(f'results/track_{metric}_n{ns}_s{sd}.json', 'w'), indent=1, default=lambda v: None)
np.savez(f'results/pool_{metric}_n{ns}_s{sd}.npz', **pool)
fmt = lambda s: ('%.3g' % s['L'] if s['finite'] else 'NF') + '/%d' % s['n_bad'] + ('X' if s['broken'] else '')
W = out['windows']
print(metric, ns, sd, 'learned', fmt(out['learned_only']), 'exact', fmt(out['exact_only']), '| [0.03,0.1]', fmt(W['0.03-0.1']),
      '[0.01,0.1]', fmt(W['0.01-0.1']), '| outliers', len(groups['outlier']), 'prior dmah', np.round(np.sort(dmah0[groups['outlier']]), 2).tolist(),
      '%.0f s' % (time.time() - t0), flush=True)
