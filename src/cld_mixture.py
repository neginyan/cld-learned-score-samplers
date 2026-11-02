"""Stage B1: the CLD flow with an isotropic marginal noise schedule, exact mixture score.

Definitions are taken unchanged from the threshold paper's code (psdt.py, copied verbatim from
neginyan/phase-space-diffusion-threshold @ 029ee94):
  forward law   Z_t = M(t) Z_0 + sigma(t) eps,  M(t) = expm(B t),  sigma(t) = s0 e^{r t}
  required noise G(t) = 2 sigma^2 (r I - Sym B),  kappa = lambda_max(Sym B)
  samplers (forward time, integrated from T down to 0, score s = grad log p_t):
    'naive' (i)   clipped Anderson SDE : drift Bz - G_+ s,             noise sqrt(G_+)
    'fixed' (ii)  corrected SDE        : drift Bz - 1/2 (G + G_+) s,   noise sqrt(G_+)
    'ode'   (iii) probability flow     : drift Bz - 1/2 G s,           noise 0
CLD flow: B = beta [[0, M^-1], [-1, -Gamma M^-1]] per position coordinate, z = (x1, v1, x2, v2).
CLD defaults (nv-tlabs/CLD-SGM, configs/default_cifar10.txt and default_toy_data.txt):
  beta0 = 4, beta1 = 0 (constant beta = 4), m_inv = 4, Gamma = f = 2/sqrt(m_inv) = 1, gamma = 0.04
  -> initial velocity v0 ~ N(0, gamma/m_inv) = N(0, 0.01)   (sde_lib.py, CLD.var: var0v = gamma / m_inv)
"""
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from psdt import cld_B, kappa_of, Process

# ------------------------------------------------------------------ CLD defaults and data
BETA, M_INV, GAMMA_CLD = 4.0, 4.0, 0.04
GAMMA_DAMP = 2.0 / np.sqrt(M_INV)                 # CLD-SGM: self.f = 2/sqrt(m_inv)  (critical damping)
V0_VAR = GAMMA_CLD / M_INV                        # 0.01
B_CLD = cld_B(beta=BETA, M=1.0 / M_INV, Gamma=GAMMA_DAMP)
KAPPA = kappa_of(B_CLD)                           # = 2
N_MODES, RADIUS, S_POS = 8, 1.0, 0.08             # 8 Gaussians on a unit circle, position std 0.08
S_MIN, S_MAX = 0.005, 40.0                        # common blur range for every r
POS, VEL = [0, 2], [1, 3]

def mixture():
    ang = 2 * np.pi * np.arange(N_MODES) / N_MODES
    mu = np.zeros((N_MODES, 4)); mu[:, 0] = RADIUS * np.cos(ang); mu[:, 2] = RADIUS * np.sin(ang)
    S = np.diag([S_POS**2, V0_VAR, S_POS**2, V0_VAR])
    w = np.full(N_MODES, 1.0 / N_MODES)
    return w, mu, np.repeat(S[None], N_MODES, 0)

def sample_mixture(w, mu, S, n, rng, extra_var=0.0):
    k = rng.choice(len(w), n, p=w)
    L = np.linalg.cholesky(S + extra_var * np.eye(4))
    return mu[k] + np.einsum('nij,nj->ni', L[k], rng.standard_normal((n, 4))), k

def process(ratio):
    return Process(B_CLD, ratio * KAPPA, s0=S_MIN, sT=S_MAX)   # T(r) = ln(S_MAX/S_MIN)/r

# ------------------------------------------------------------------ exact score
class ExactScore:
    """Forward marginal of mixture data: components N(M mu_k, M S_k M^T + sigma^2 I).  Exact score.
    All components share one covariance here, which makes the score -P (z - sum_k r_k m_k)."""
    def __init__(self, proc, w, mu, S):
        assert np.allclose(S, S[0][None])
        self.p, self.lw, self.mu, self.S0 = proc, np.log(w), mu, S[0]
        self.cache = {}
    def params(self, t):
        key = round(float(t), 13)
        if key not in self.cache:
            M = expm(self.p.B * t)
            C = M @ self.S0 @ M.T + self.p.sigma(t)**2 * np.eye(4)
            self.cache[key] = (self.mu @ M.T, np.linalg.inv(C))
        return self.cache[key]
    def __call__(self, z, t):
        m, P = self.params(t)
        Pz, Pm = z @ P, m @ P                                     # P symmetric
        logit = self.lw[None] + Pz @ m.T - 0.5 * (m * Pm).sum(1)[None]   # z-quadratic term cancels
        logit -= logit.max(1, keepdims=True); r = np.exp(logit); r /= r.sum(1, keepdims=True)
        return -(Pz - r @ Pm)

# ------------------------------------------------------------------ samplers (paper definitions)
def drift_fn(proc, mode):
    B = proc.B
    def f(z, t, s):
        G = proc.G(t)
        if mode == 'ode':   return z @ B.T - 0.5 * s @ G.T
        Gp = proc.Gplus(t)
        if mode == 'naive': return z @ B.T - s @ Gp.T
        if mode == 'fixed': return z @ B.T - 0.5 * s @ (G + Gp).T
        raise ValueError(mode)
    return f

def run_sampler(score, proc, z, mode, K, noise_rng=None, sub=1):
    """Integrate from T to 0 with K uniform steps.  SDE: Euler-Maruyama; ODE: Heun.
    For the SDEs, each step's Gaussian is the normalised sum of `sub` consecutive draws from noise_rng,
    so runs with (K, sub) and (2K, sub/2) on the same seed share one Brownian path (coupled refinement)."""
    ts = np.linspace(proc.T, 0.0, K + 1)
    f = drift_fn(proc, mode)
    z = z.copy()
    for k in range(K):
        t, t2 = ts[k], ts[k + 1]; h = t - t2
        if mode == 'ode':
            f1 = f(z, t, score(z, t)); zp = z - h * f1
            f2 = f(zp, t2, score(zp, t2)); z = z - 0.5 * h * (f1 + f2)
        else:
            xi = noise_rng.standard_normal((sub,) + z.shape).sum(0) / np.sqrt(sub)
            z = z - h * f(z, t, score(z, t)) + np.sqrt(h) * xi @ proc.sqrtGplus(t).T
    return z

def prior_exact(proc, w, mu, S, n, rng):
    """Exact sample of p_T: data pushed by M(T) plus sigma(T) eps."""
    z0, _ = sample_mixture(w, mu, S, n, rng)
    return z0 @ expm(proc.B * proc.T).T + proc.sigma(proc.T) * rng.standard_normal(z0.shape)

# ------------------------------------------------------------------ measurements (positions)
def mode_stats(z, mu):
    x = z[:, POS]; c = mu[:, POS]
    lab = np.argmin(((x[:, None, :] - c[None])**2).sum(-1), 1)
    counts = np.bincount(lab, minlength=len(c))
    var = np.array([x[lab == k].var(0).mean() if counts[k] > 1 else np.nan for k in range(len(c))])
    within = np.nansum(var * counts) / counts.sum()               # count-weighted mean per-coordinate variance
    vvar = z[:, VEL].var(0).mean()
    return dict(within_var=float(within), mode_var=var.tolist(), weights=(counts / counts.sum()).tolist(),
                vel_var=float(vvar))

def mmd2(a, b, scales=(0.05, 0.1, 0.2, 0.5)):
    def k(x, y):
        d2 = (x**2).sum(1)[:, None] + (y**2).sum(1)[None] - 2 * x @ y.T
        return sum(np.exp(-np.maximum(d2, 0) / (2 * s * s)) for s in scales)
    n, m = len(a), len(b)
    kaa, kbb, kab = k(a, a), k(b, b), k(a, b)
    return float((kaa.sum() - np.trace(kaa)) / (n * (n - 1)) + (kbb.sum() - np.trace(kbb)) / (m * (m - 1)) - 2 * kab.mean())

# ------------------------------------------------------------------ Gaussian-moment theory (paper Prop. 6)
def gauss_output(proc, m0, S0, mode):
    """Output mean/covariance of a sampler run on Gaussian data N(m0,S0) with its exact (Gaussian) score:
    moment ODEs integrated from T to 0 (as verification/run_all.py section 9 / 15b of the paper code)."""
    B, I4 = proc.B, np.eye(4)
    def mom(t):
        M = expm(B * t); return M @ m0, M @ S0 @ M.T + proc.sigma(t)**2 * I4
    def rhs(tau, y):
        t = proc.T - tau; mu = y[:4]; Sg = y[4:].reshape(4, 4); m, C = mom(t); P = np.linalg.inv(C)
        G, Gp = proc.G(t), proc.Gplus(t)
        Kd = {'ode': 0.5 * G, 'naive': Gp, 'fixed': 0.5 * (G + Gp)}[mode]
        D = np.zeros((4, 4)) if mode == 'ode' else Gp
        A = -B - Kd @ P
        return np.concatenate([A @ mu + Kd @ P @ m, (A @ Sg + Sg @ A.T + D).ravel()])
    m, C = mom(proc.T)
    y = solve_ivp(rhs, [0, proc.T], np.concatenate([m, C.ravel()]), rtol=1e-10, atol=1e-13, method='DOP853').y[:, -1]
    Sg = y[4:].reshape(4, 4)
    return y[:4], (Sg + Sg.T) / 2, mom(0)

def theory_within_ratio(ratio, mode='naive'):
    """Predicted within-mode position-variance ratio (sampler output / sigma_min-blurred data) for one mixture
    component treated as Gaussian data on its own (all components are identical up to a shift)."""
    w, mu, S = mixture(); p = process(ratio)
    _, Sg, (_, Ct) = gauss_output(p, mu[0], S[0], mode)
    return float(Sg[POS, POS].mean() / Ct[POS, POS].mean()), float(Sg[VEL, VEL].mean() / Ct[VEL, VEL].mean())
