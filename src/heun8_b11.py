"""H8 (B11): the Heun-type corrected SDE (ii) with the learned score in every metric, plus a coarser step h ~ 0.064.
  python heun8_b11.py <metric> <net_seed> <samp_seed> [only64]  -> results/heun8_<metric>_n<n>_s<s>.json
The integrator is that of B10's heun_learned_b10.py (predictor-corrector, the same Gaussian increment in both stages, noise from G_+
at the mid-time).  Prior draws 100 + s and noise seed 200 + s as in B9, so for h ~ 0.032, 0.016, 0.008 the Brownian path is that of
the B9 Euler-Maruyama runs (paired comparison).  h ~ 0.064: K = 140 steps, each increment the normalised sum of 32 draws from the
same stream; this grid does not line up exactly with the 4496-step grid, so it is not strictly coupled to the finer levels.
At h ~ 0.064 the Euler-Maruyama run is made here as well (run_sampler 'fixed', K = 140, sub = 32).
'only64': just the h ~ 0.064 level (used for I and diag2, whose finer levels are in B10)."""
import sys, json, time
import numpy as np
from learned_b11 import load, stats, far_mask, N
from cld_mixture import drift_fn, mode_stats, run_sampler

metric, ns, sd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); only64 = len(sys.argv) > 4 and sys.argv[4] == 'only64'
m, ls = load(metric, ns); p = m.proc; f = drift_fn(p, 'fixed')
Kf = int(np.ceil(p.T / 0.002 / 16)) * 16
y0 = m.prior(N, np.random.default_rng(100 + sd)); ref = mode_stats(m.truth(400000, np.random.default_rng(999)), m.mu)

def summ(zY):
    zZ = m.to_Z(zY); st = stats(zZ, m, ref)
    bad = ~np.isfinite(zZ).all(1); bad[~bad] = far_mask(zZ[~bad], m.mu); st['n_bad'] = int(bad.sum()); return st

def heun(K, sub):
    ts = np.linspace(p.T, 0.0, K + 1); rng = np.random.default_rng(200 + sd); z = y0.copy()
    with np.errstate(all='ignore'):
        for k in range(K):
            t, t2 = ts[k], ts[k + 1]; h = t - t2
            xi = rng.standard_normal((sub,) + z.shape).sum(0) / np.sqrt(sub)
            dW = np.sqrt(h) * xi @ p.sqrtGplus(0.5 * (t + t2)).T
            f1 = f(z, t, ls(z, t)); zp = z - h * f1 + dW
            f2 = f(zp, t2, ls(zp, t2)); z = z - 0.5 * h * (f1 + f2) + dW
    return z

out = {'metric': metric, 'net_seed': ns, 'seed': sd, 'levels': []}
plan = [(140, 32)] + ([] if only64 else [(Kf // 16, 16), (Kf // 8, 8), (Kf // 4, 4)])
for K, sub in plan:
    t0 = time.time(); row = {'K': K, 'h': p.T / K, 'sub': sub, 'heun': summ(heun(K, sub))}
    if K == 140:
        with np.errstate(all='ignore'): row['em'] = summ(run_sampler(ls, p, y0, 'fixed', K, np.random.default_rng(200 + sd), sub=sub))
    out['levels'].append(row)
    print(metric, ns, sd, 'h %.4f heun L %s bad %d' % (p.T / K, row['heun']['L'], row['heun']['n_bad']),
          ('| em L %s bad %d' % (row['em']['L'], row['em']['n_bad'])) if 'em' in row else '', '(%.0f s)' % (time.time() - t0), flush=True)
json.dump(out, open(f'results/heun8_{metric}_n{ns}_s{sd}.json', 'w'), indent=1)
