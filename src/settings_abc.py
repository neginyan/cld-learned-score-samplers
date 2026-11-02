"""Stage B2: settings for the mixture checks.  Same definitions as B1 (cld_mixture.py / psdt.py), with the CLD mass M,
sigma_min and r/kappa changed.  beta is chosen so that kappa = 2 in every setting (beta does not change the result at
fixed r/kappa; checked in verification).  Gamma = 2 sqrt(M) (critical, as in CLD-SGM)."""
import numpy as np
from psdt import cld_B, kappa_of, Process
from cld_mixture import (mixture, sample_mixture, ExactScore, run_sampler, prior_exact, mode_stats, mmd2,
                    gauss_output, S_POS, V0_VAR, POS, VEL)

S_MAX = 40.0                      # data units (= 500 s), as in B1
SETTINGS = {
    'A': dict(M=0.01, smin_rel=0.0625, rk=0.2),     # large kappa/beta (light mass)
    'B': dict(M=0.25, smin_rel=1.778, rk=0.3),      # sigma_min comparable to the mode width
    'C': dict(M=10.0, smin_rel=0.5, rk=0.5),        # heavy mass + sigma_min = s/2
}

def setting(name):
    st = SETTINGS[name]
    beta = 2.0 / kappa_of(cld_B(beta=1.0, M=st['M'], Gamma=2 * np.sqrt(st['M']), d=1))
    B = cld_B(beta=beta, M=st['M'], Gamma=2 * np.sqrt(st['M']))
    kap = kappa_of(B)
    p = Process(B, st['rk'] * kap, s0=st['smin_rel'] * S_POS, sT=S_MAX)
    return p, kap, beta

def theory_ratio(p, mode='naive'):
    w, mu, S = mixture()
    _, Sg, (_, Ct) = gauss_output(p, mu[0], S[0], mode)
    return float(Sg[POS, POS].mean() / Ct[POS, POS].mean()), float(Sg[VEL, VEL].mean() / Ct[VEL, VEL].mean())
