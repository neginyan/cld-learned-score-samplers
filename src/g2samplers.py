"""Samplers of the paper, with a learned or exact score, in the metric Y = P Z (setting D, r = 1).
The code is that of the lab stages (heun8_b11.py, learned_b11.py, cld_mixture.run_sampler); it is collected here so that
verification/run_all.py can call it directly.  Networks are read from nets/ (B11_LOCAL_ONLY=1, B11_NETS=<repo>/nets)."""
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('B11_LOCAL_ONLY', '1'); os.environ.setdefault('B11_NETS', os.path.join(_root, 'nets'))
import numpy as np
from learned_b11 import load, stats, far_mask, N
from cld_mixture import drift_fn, mode_stats, run_sampler

KF = 4496                                            # finest grid, h ~ 0.002

def setup(metric, net_seed, seed):
    m, ls = load(metric, net_seed); p = m.proc
    y0 = m.prior(N, np.random.default_rng(100 + seed)); ref = mode_stats(m.truth(400000, np.random.default_rng(999)), m.mu)
    return m, ls, p, y0, ref

def summary(m, ref, zY):
    z = m.to_Z(zY); st = stats(z, m, ref)
    bad = ~np.isfinite(z).all(1); bad[~bad] = far_mask(z[~bad], m.mu); st['n_bad'] = int(bad.sum())
    st['broken'] = (not st['finite']) or abs(st['L']) > 1 or bad.sum() / len(z) > 0.01
    return st

def heun(ls, p, y0, K, sub, seed):
    """Heun-type corrected SDE (ii): predictor-corrector with one Gaussian increment shared by both stages, noise G_+ at the mid-time.
    Increments: normalized sums of `sub` draws of rng(200 + seed) -> coupled to the 4496-step grid when K * sub = 4496."""
    f = drift_fn(p, 'fixed'); ts = np.linspace(p.T, 0.0, K + 1); rng = np.random.default_rng(200 + seed); z = y0.copy()
    with np.errstate(all='ignore'):
        for k in range(K):
            t, t2 = ts[k], ts[k + 1]; h = t - t2
            xi = rng.standard_normal((sub,) + z.shape).sum(0) / np.sqrt(sub)
            dW = np.sqrt(h) * xi @ p.sqrtGplus(0.5 * (t + t2)).T
            f1 = f(z, t, ls(z, t)); zp = z - h * f1 + dW
            f2 = f(zp, t2, ls(zp, t2)); z = z - 0.5 * h * (f1 + f2) + dW
    return z

def em(ls, p, y0, K, sub, seed):
    with np.errstate(all='ignore'): return run_sampler(ls, p, y0, 'fixed', K, np.random.default_rng(200 + seed), sub=sub)

def ode(ls, p, y0, K=KF):
    with np.errstate(all='ignore'): return run_sampler(ls, p, y0, 'ode', K)
