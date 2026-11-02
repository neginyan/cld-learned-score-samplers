"""H9 (B12): is the second stage (the final jump) an extrapolation of the learned score?
  python inject_b12.py <metric> <net_seed>   -> results/h9_<metric>_n<n>.json
The B11 H6 perturbations (inject_b11.py: same points, same pool, the same draws of rng 1100 in the same order) are added at
sigma_inj on the exact-score path; from there the state is carried to the end (i) with the exact score and (ii) with the learned
score of the network.  A = |dx_end| / |dz_inj|, where dx_end is the position difference in Z at the end against the unperturbed
path carried with the SAME score.  Non-finite -> A = inf.
B12_FAKE=1 replaces the learned score by the exact one (code check only: R must be 1)."""
import sys, os, json, glob, time
import numpy as np
from learned_b11 import load, stats, far_mask, N
from b8core import MetricSetup
from cld_mixture import drift_fn, mode_stats, POS, VEL

B11 = os.environ.get('B11_DIR', '/home/claude/B11')
SIG_ALL = [0.3, 0.1, 0.03, 0.01]; SIG_RUN = [0.03, 0.01]; NPER = 4
metric, ns = sys.argv[1], int(sys.argv[2]); fake = os.environ.get('B12_FAKE') == '1'
if fake: m = MetricSetup(metric, 1.0); ls = m.score()
else: m, ls = load(metric, ns)
p = m.proc; ex = m.score(); f = drift_fn(p, 'ode')
K = int(np.ceil(p.T / 0.002 / 16)) * 16; ts = np.linspace(p.T, 0.0, K + 1); sk = p.sigma(ts[:-1])
kin = {c: int(np.argmax(sk < c)) for c in SIG_ALL}
y0 = m.prior(N, np.random.default_rng(100)); ref = mode_stats(m.truth(400000, np.random.default_rng(999)), m.mu)
mt, Pi = ex.params(p.T); d = y0[:, None] - mt[None]; dmah0 = np.sqrt(np.einsum('nki,ij,nkj->nk', d, Pi, d).min(1))
pts = {'tail': np.where((dmah0 >= 3.5) & (dmah0 <= 5.0))[0], 'bulk': np.where(dmah0 < 2.0)[0][:300]}

def step(score, z, k):
    t, t2 = ts[k], ts[k + 1]; h = t - t2
    f1 = f(z, t, score(z, t)); zp = z - h * f1; f2 = f(zp, t2, score(zp, t2)); return z - 0.5 * h * (f1 + f2)
def integrate(score, z, k0, k1, keep=()):
    saved = {}
    with np.errstate(all='ignore'):
        for k in range(k0, k1):
            if k in keep: saved[k] = z.copy()
            z = step(score, z, k)
    return z, saved

t0 = time.time()
zE, S = integrate(ex, y0.copy(), 0, K, set(kin.values()))
st0 = stats(m.to_Z(zE), m, ref)
pools = sorted(glob.glob(f'{B11}/results/pool_I_n*_s*.npz'))
rng = np.random.default_rng(1100)
out = {'metric': metric, 'net_seed': ns, 'fake': fake, 'kappa_P': float(m.kappa), 'N': N, 'K': K, 'unperturbed_exact': st0,
       'n_points': {g: int(len(v)) for g, v in pts.items()}, 'pool_files': [os.path.basename(x) for x in pools], 'by_sigma': {}}

def summ(ze, base_end, dz):
    fin = np.isfinite(ze).all(1) & np.isfinite(base_end).all(1)
    dx = np.full(len(ze), np.inf); dx[fin] = np.linalg.norm(ze[fin][:, POS] - base_end[fin][:, POS], axis=1)
    A = dx / np.linalg.norm(dz, axis=1)
    finz = np.isfinite(ze).all(1); far = np.ones(len(ze), bool); far[finz] = far_mask(ze[finz], m.mu)
    return {'A_median': float(np.median(A)), 'A_q90': float(np.quantile(A, 0.9)), 'A_q10': float(np.quantile(A, 0.1)),
            'A_q99': float(np.quantile(A, 0.99)), 'n_nonfinite': int((~finz).sum()), 'far_fraction': float(far.mean())}, A

for c in SIG_ALL:                                   # the draws follow inject_b11.py exactly (also for sigma levels not run here)
    D = np.concatenate([np.load(x)[f'sig{c}'] for x in pools if f'sig{c}' in np.load(x).files])
    D = D[np.isfinite(D).all(1)]; eps = float(np.median(np.linalg.norm(D, axis=1)))
    res = {'k_inj': kin[c], 'pool_size': int(len(D)), 'eps': eps}
    for g, idx in pts.items():
        rep = np.repeat(np.arange(len(idx)), NPER)
        def iso(sub, size):
            v = np.zeros((len(rep), 4)); v[:, sub] = rng.standard_normal((len(rep), len(sub)))
            return size * v / np.linalg.norm(v, axis=1, keepdims=True)
        kinds = {'pool': D[rng.integers(0, len(D), len(rep))], 'iso': iso([0, 1, 2, 3], eps), 'iso_small': iso([0, 1, 2, 3], eps / 10),
                 'pos_only': iso(list(np.arange(4)[POS]), eps), 'vel_only': iso(list(np.arange(4)[VEL]), eps)}
        if c not in SIG_RUN: continue
        base_in = S[kin[c]][idx]
        endE = m.to_Z(integrate(ex, base_in.copy(), kin[c], K)[0])[rep]
        endL0 = m.to_Z(integrate(ls, base_in.copy(), kin[c], K)[0])
        finL0 = np.isfinite(endL0).all(1); farL0 = np.ones(len(idx), bool); farL0[finL0] = far_mask(endL0[finL0], m.mu)
        res[f'{g}_unperturbed_learned'] = {'n_far_or_nonfinite': int(farL0.sum()), 'n_nonfinite': int((~finL0).sum()), 'n': int(len(idx))}
        for kname in ('pool', 'iso'):
            dz = kinds[kname]; yp = base_in[rep] + dz @ m.P.T
            sE, AE = summ(m.to_Z(integrate(ex, yp.copy(), kin[c], K)[0]), endE, dz)
            sL, AL = summ(m.to_Z(integrate(ls, yp.copy(), kin[c], K)[0]), endL0[rep], dz)
            with np.errstate(all='ignore'):
                R = sL['A_median'] / sE['A_median']; R90 = sL['A_q90'] / sE['A_q90']
                lr = np.log10(AL / AE); fin = np.isfinite(lr)
            res[f'{g}_{kname}'] = {'exact': sE, 'learned': sL, 'R': R, 'R90': R90,
                                   'pointwise_log10_ratio_median': float(np.median(np.where(np.isfinite(AL / AE), lr, np.inf))),
                                   'frac_ratio_ge_10': float(np.mean(AL >= 10 * AE))}
            print(metric, ns, 'sig', c, g, kname, 'A_E %.3g A_L %.3g R %.3g | q90 %.3g %.3g R90 %.3g | far E %.3f L %.3f nf L %d' % (
                sE['A_median'], sL['A_median'], R, sE['A_q90'], sL['A_q90'], R90, sE['far_fraction'], sL['far_fraction'], sL['n_nonfinite']),
                '(%.0f s)' % (time.time() - t0), flush=True)
    out['by_sigma'][str(c)] = res
tag = 'fake_' if fake else ''
json.dump(out, open(f'results/{tag}h9_{metric}_n{ns}.json', 'w'), indent=1, default=lambda v: None if v != v or v in (np.inf, -np.inf) else float(v))
