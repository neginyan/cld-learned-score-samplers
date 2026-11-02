"""Stage B3: learned score for the B2 settings.  Network and optimiser from paper D (mlp.py = mlp.py, copied verbatim from
neginyan/learned-score-arrow-of-time @ c304639: float32 MLP, SiLU, Adam, cosine decay, EMA 0.9995).

Settings (kappa = 2 in every one; same as B1/B2):
  'D'  B1 default : M = 1/4, sigma_min = s/16 = 0.005, r/kappa = 0.5
  'B'  B2 setting B: M = 1/4, sigma_min = 1.78 s,     r/kappa = 0.3
  'C'  B2 setting C: M = 10,  sigma_min = s/2,        r/kappa = 0.5
Score model: eps-prediction, s(z, sigma) = -eps_hat / sigma.
  input  : c_in * (z - m(t)) per coordinate, c_in = 1/sqrt(diag(M Sigma_data M^T) + sigma^2)   (data moments only),
           plus Fourier features of u = log(sigma/sigma_min)/log(sigma_max/sigma_min) in [0, 1].
  loss   : mean |eps_hat - eps|^2 (weight 1 = sigma^2-weighted score matching, paper D 'eps').
  batch  : half uniform in log sigma over [sigma_min, sigma_max], half uniform in log sigma over [sigma_min, 10 sigma_min].
Data: a fixed training set of N_TRAIN = 200 000 samples of the 8-mode mixture (seed fixed)."""
import numpy as np
from scipy.linalg import expm
from mlp import MLP
from psdt import cld_B, kappa_of, Process
import cld_mixture
from cld_mixture import mixture, sample_mixture, ExactScore, prior_exact, mode_stats, POS, VEL, S_POS

S_MAX = 40.0
SET = {'D': dict(M=0.25, smin=0.005, rk=0.5), 'B': dict(M=0.25, smin=1.778 * S_POS, rk=0.3), 'C': dict(M=10.0, smin=0.5 * S_POS, rk=0.5)}
N_TRAIN, H, NF = 200_000, 128, 4

def setting(name):
    st = SET[name]; Gm = 2 * np.sqrt(st['M'])
    beta = 2.0 / kappa_of(cld_B(beta=1.0, M=st['M'], Gamma=Gm, d=1))
    B = cld_B(beta=beta, M=st['M'], Gamma=Gm)
    return Process(B, st['rk'] * kappa_of(B), s0=st['smin'], sT=S_MAX)

def train_data(seed=0):
    w, mu, S = mixture(); return sample_mixture(w, mu, S, N_TRAIN, np.random.default_rng(seed))[0]

class Precond:
    """Data moments propagated by the flow, tabulated on a fine log-sigma grid (interpolated)."""
    def __init__(self, p, data):
        self.p = p; m0, S0 = data.mean(0), np.cov(data.T)
        self.lg = np.linspace(np.log(p.s0), np.log(p.sT), 4001); t = (self.lg - np.log(p.s0)) / p.r
        Ms = expm(t[:, None, None] * p.B[None]); self.Ms = Ms
        self.mt = Ms @ m0; self.vt = np.einsum('kij,jl,kil->ki', Ms, S0, Ms)
    def t_of(self, sig): return np.log(sig / self.p.s0) / self.p.r
    def interp(self, tab, sig):
        x = (np.log(sig) - self.lg[0]) / (self.lg[1] - self.lg[0]); i = np.clip(np.floor(x).astype(int), 0, len(self.lg) - 2); f = (x - i)[:, None]
        return (1 - f) * tab[i] + f * tab[i + 1]
    def M_of(self, sig):
        x = (np.log(sig) - self.lg[0]) / (self.lg[1] - self.lg[0]); i = np.clip(np.floor(x).astype(int), 0, len(self.lg) - 2); f = (x - i)[:, None, None]
        return (1 - f) * self.Ms[i] + f * self.Ms[i + 1]
    def features(self, z, sig):
        m = self.interp(self.mt, sig); v = self.interp(self.vt, sig) + sig[:, None]**2
        u = (np.log(sig) - self.lg[0]) / (self.lg[-1] - self.lg[0])
        ff = [2 * u - 1] + [f(np.pi * k * u) for k in range(1, NF + 1) for f in (np.sin, np.cos)]
        return np.concatenate([(z - m) / np.sqrt(v), np.stack(ff, 1)], 1).astype(np.float32)

def new_net(seed): return MLP([4 + 1 + 2 * NF, H, H, H, 4], seed)

class LearnedScore:
    def __init__(self, net, pc, params=None): self.net, self.pc, self.P = net, pc, (net.E if params is None else params)
    def eps(self, z, sig_arr): return self.net.forward(self.pc.features(z, sig_arr), params=self.P).astype(np.float64)
    def __call__(self, z, t):
        sig = np.full(len(z), self.pc.p.sigma(t)); return -self.eps(z, sig) / sig[:, None]
