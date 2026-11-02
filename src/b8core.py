"""Stage B8: isotropic noise in a chosen metric Y = P Z on the CLD flow.

Forward law in the original coordinates Z = (x1, v1, x2, v2):
    Z_t = e^{Bt} Z_0 + sigma(t) P^{-1} eps,   P = kron(I_2, p),  sigma(t) = sigma_min e^{rt},
i.e. the blur is isotropic in Y = P Z.  In Y this is exactly the setting of the threshold paper with the flow
B_P = P B P^{-1} and data Y_0 = P Z_0, so every sampler is run in Y with the unchanged code of the earlier stages
(cld_mixture.py, psdt.py: copied from cld-corrected-sampler v1.0.0, src/) and the samples are mapped back to Z.
Clipping G_+ is done in Y (the metric in which the blur is isotropic).

Normalization: p is scaled so that (p^{-1} p^{-T})_xx = 1, so the position blur in Z is sigma for every metric and
the target (data blurred by sigma_min) has the same position variance s^2 + sigma_min^2 as before."""
import numpy as np
from scipy.linalg import expm
from psdt import cld_B, kappa_of, Process
from cld_mixture import (mixture, sample_mixture, ExactScore, run_sampler, mode_stats, gauss_output,
                         B_CLD, KAPPA, S_MIN, S_MAX, S_POS, POS, VEL, BETA, M_INV, GAMMA_DAMP)

b2 = B_CLD[:2, :2]                                   # beta [[0, 1/M], [-1, -Gamma/M]] = [[0, 16], [-4, -16]]
ALPHA = float(np.linalg.eigvals(b2).real.max())      # spectral abscissa (-8, double eigenvalue, Jordan block)

def rot(a): return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
def kappa2(p, b=b2):
    bp = p @ b @ np.linalg.inv(p); return float(np.linalg.eigvalsh((bp + bp.T) / 2).max())
def cond(p): s = np.linalg.svd(p, compute_uv=False); return float(s[0] / s[-1])
def normalize(p):
    q = np.linalg.inv(p); return p * np.sqrt((q @ q.T)[0, 0])

def p_optimal(c, b=b2):
    """Metric of condition number c with the smallest kappa (critical damping; Proposition 2):
    B - alpha I = u v^T;  p = diag(c, 1) [v_hat, u_hat]^T-type map that sends u to the small and v to the large
    singular direction.  Returns the normalized p."""
    a = float(np.linalg.eigvals(b).real.max()); N = b - a * np.eye(2)
    U, S, Vt = np.linalg.svd(N); u = U[:, 0] * S[0]; v = Vt[0]          # N = u v^T
    uh, vh = u / np.linalg.norm(u), v / np.linalg.norm(v)              # u ⟂ v (N nilpotent)
    R = np.stack([vh, uh])                                             # rows: right singular vectors of p
    p = np.diag([c, 1.0]) @ R                                          # sigma_max = c along v, sigma_min = 1 along u
    return normalize(p)

METRICS = {                       # name: p (2x2, before normalization)
    'I':       np.eye(2),
    'diag2':   np.diag([1.0, 2.0]),
    'diag1.5': np.diag([1.0, 1.5]),
    'diag0.5': np.diag([1.0, 0.5]),
    'opt1.25': None, 'opt2': None, 'opt5': None, 'opt20': None,
}
def metric(name):
    p = METRICS[name] if not name.startswith('opt') else p_optimal(float(name[3:]))
    p = normalize(np.asarray(p, float)); return p, np.kron(np.eye(2), p)

class MetricSetup:
    def __init__(self, name, r, smin=S_MIN, smax=S_MAX):
        self.name = name; self.p, self.P = metric(name); self.Pi = np.linalg.inv(self.P)
        self.BP = self.P @ B_CLD @ self.Pi
        self.kappa = kappa_of(self.BP); self.cond = cond(self.p)
        self.proc = Process(self.BP, r, s0=smin, sT=smax)
        w, mu, S = mixture(); self.w, self.mu, self.S = w, mu, S
        self.muY = mu @ self.P.T; self.SY = np.einsum('ij,kjl,ml->kim', self.P, S, self.P)
    def score(self): return ExactScore(self.proc, self.w, self.muY, self.SY)
    def prior(self, n, rng):
        z0, _ = sample_mixture(self.w, self.muY, self.SY, n, rng)
        return z0 @ expm(self.BP * self.proc.T).T + self.proc.sigma(self.proc.T) * rng.standard_normal(z0.shape)
    def to_Z(self, y): return y @ self.Pi.T
    def truth(self, n, rng):
        """Data blurred by sigma_min in the metric: Z_0 + sigma_min P^{-1} eps."""
        z0, _ = sample_mixture(self.w, self.mu, self.S, n, rng)
        return z0 + self.proc.s0 * rng.standard_normal(z0.shape) @ self.Pi.T
    def theory(self, mode='naive'):
        """One-mode Gaussian theory (paper G, Eq. (4) / Prop. 6 of the threshold paper) in Y, mapped to Z positions:
        returns log(var_x(sampler) / var_x(target)) for one mixture component."""
        _, SgY, (_, CtY) = gauss_output(self.proc, self.muY[0], self.SY[0], mode)
        Sg = self.Pi @ SgY @ self.Pi.T; Ct = self.Pi @ CtY @ self.Pi.T
        return float(np.log(Sg[POS, POS].mean() / Ct[POS, POS].mean())), float(np.log(Sg[VEL, VEL].mean() / Ct[VEL, VEL].mean()))
