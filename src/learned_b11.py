"""Stage B11 (copy of B10's learned_b10.py, itself a copy of B9's learned_b9.py; only the locations of other stages' files differ, plus the commands 'estar' and 'ode' for H7).
Stage B9: learned score in a metric (setting D: B1 defaults, r = 1).  Same recipe as B3 / B8 Part C (learned_b8.py),
with the metric as an argument.  The network is trained and sampled entirely in Y = P Z; results are measured in Z.
  python learned_b10.py train  <metric> <net_seed> [steps]          -> results/nets/<metric>_D_seed<n>_step<k>.npz/.json (every 10k)
  python learned_b10.py err    <metric> <net_seed>                  -> results/err_<metric>_n<n>.json
  python learned_b10.py sample <metric> <net_seed> <samp_seed>      -> results/samp_<metric>_n<n>_s<s>.json
Sampling: h ~ 0.002 * 2^k (k = 0..4) on one Brownian path (sub-sums of the finest increments, as in B8 Part B);
prior draws and noise seeds (100 + s, 200 + s) are those of B8 Part B, so the exact-score values there are paired."""
import sys, os, json, time
import numpy as np
from b8core import MetricSetup
from learned import Precond, LearnedScore, new_net, train_data
from cld_mixture import run_sampler, mode_stats, sample_mixture, POS, VEL
from scipy.linalg import expm

R = 1.0; N = int(os.environ.get('B9_N', 20000)); SUBS = [16, 8, 4, 2, 1]; FAR = 0.5
NETDIR = os.environ.get('B11_NETS', 'results/nets')
B3_DIR = os.environ.get('B3_DIR', '/home/claude/B3'); B8_DIR = os.environ.get('B8_DIR', '/home/claude/B8'); B9_DIR = os.environ.get('B9_DIR', '/home/claude/B9'); B10_DIR = os.environ.get('B10_DIR', '/home/claude/B10')

def net_path(metric, ns, step=300000):
    """Networks of earlier stages are reused: I seeds 1-2 from B3, diag2 seeds 1-2 from B8, I/diag2 3-4 and opt* 1-4 from B9,
    I/diag2 5-8 from B10; I seeds 9-12 are trained in B11."""
    if os.environ.get('B11_LOCAL_ONLY') is None:
        if metric in ('I', 'diag2') and 5 <= ns <= 8: return f'{B10_DIR}/results/nets/{metric}_D_seed{ns}_step{step}'   # B10
        if metric == 'diag2' and ns in (1, 2): return f'{B8_DIR}/results/nets/diag2_D_seed{ns}_step{step}'   # B8 Part C (as in B9)
        if metric == 'I' and ns in (1, 2): return f'{B3_DIR}/results/nets/D_seed{ns}_step{step}'             # B3, bitwise-checked in B9
        if (metric in ('I', 'diag2') and ns in (3, 4)) or metric in ('opt1.25', 'opt5'):
            return f'{B9_DIR}/results/nets/{metric}_D_seed{ns}_step{step}'                                   # B9
    return f'{NETDIR}/{metric}_D_seed{ns}_step{step}'

def setup(metric):
    m = MetricSetup(metric, R); dataY = train_data(0) @ m.P.T; pc = Precond(m.proc, dataY); return m, dataY, pc

def load(metric, ns):
    m, dataY, pc = setup(metric); net = new_net(ns); net.load(net_path(metric, ns)); return m, LearnedScore(net, pc)

def far_mask(z, mu):
    return np.sqrt(((z[:, None, POS] - mu[None][:, :, POS])**2).sum(-1)).min(1) > FAR

def stats(z, m, ref):
    ok = bool(np.isfinite(z).all())
    if not ok:
        fin = np.isfinite(z).all(1); return {'finite': False, 'n_nonfinite': int((~fin).sum()), 'L': None, 'L_vel': None, 'n_far': None, 'max_weight_dev': None}
    st = mode_stats(z, m.mu); far = far_mask(z, m.mu)
    return {'finite': True, 'L': float(np.log(st['within_var'] / ref['within_var'])), 'L_vel': float(np.log(st['vel_var'] / ref['vel_var'])),
            'n_far': int(far.sum()), 'max_weight_dev': float(np.max(np.abs(np.array(st['weights']) - 1 / 8)))}

if __name__ == '__main__':
    cmd, metric, ns = sys.argv[1], sys.argv[2], int(sys.argv[3])
    if cmd == 'train':
        steps = int(sys.argv[4]) if len(sys.argv) > 4 else 300000; BS = 1024
        STOP = int(os.environ.get('B11_STOP', 10**12)); os.makedirs(NETDIR, exist_ok=True)
        m, dataY, pc = setup(metric); p = m.proc; net = new_net(ns)
        # resume from the latest checkpoint (every 10k steps); the RNG stream is replayed so that a resumed run equals an uninterrupted one
        rng = np.random.default_rng(10_000 + ns); ls0, ls1 = np.log(p.s0), np.log(p.sT); log = []; run = None; t0 = time.time()
        ck = sorted([int(f.split('_step')[1].split('.')[0]) for f in os.listdir(NETDIR) if f.startswith(f'{metric}_D_seed{ns}_step') and f.endswith('.npz')])
        start = ck[-1] if ck else 0
        if start:
            net.load(f'{NETDIR}/{metric}_D_seed{ns}_step{start}'); meta = json.load(open(f'{NETDIR}/{metric}_D_seed{ns}_step{start}.json'))
            log = meta.get('log', []); run = log[-1][1] if log else None
            if 'rng_state' in meta: rng.bit_generator.state = meta['rng_state']
            else: raise SystemExit('checkpoint without rng state; cannot resume exactly')
            print(f'resume {metric} seed{ns} from step {start}', flush=True)
        while net.step < min(steps, STOP):
            half = BS // 2
            sig = np.exp(np.concatenate([rng.uniform(ls0, ls1, half), rng.uniform(ls0, ls0 + np.log(10.0), BS - half)]))
            z0 = dataY[rng.integers(0, len(dataY), BS)]; eps = rng.standard_normal((BS, 4))
            zt = np.einsum('nij,nj->ni', pc.M_of(sig), z0) + sig[:, None] * eps
            out, cache = net.forward(pc.features(zt, sig), keep=True); d = out.astype(np.float64) - eps
            loss = float((d**2).sum(1).mean())
            st = net.step; lr = 2e-3 * min(1.0, (st + 1) / 1000) * (1e-5 / 2e-3 + (1 - 1e-5 / 2e-3) * 0.5 * (1 + np.cos(np.pi * st / steps)))
            net.adam(net.backward(cache, (2 * d / BS).astype(np.float32)), lr)
            run = loss if run is None else 0.99 * run + 0.01 * loss
            if net.step % 1000 == 0: log.append([net.step, run])
            if net.step % 10000 == 0 or net.step == steps or net.step == STOP:
                net.save(f'{NETDIR}/{metric}_D_seed{ns}_step{net.step}', {'metric': metric, 'r': R, 'seed': ns, 'steps_total': steps, 'log': log,
                                                                          'rng_state': rng.bit_generator.state})
                print(f'{metric} seed{ns} step {net.step} loss {run:.5f} ({time.time() - t0:.0f} s)', flush=True)
        sys.exit(0)

    m, ls = load(metric, ns); p = m.proc; ex = m.score()
    if cmd == 'err':
        rng = np.random.default_rng(31_000); out = []
        for sg in np.exp(np.linspace(np.log(p.s0), np.log(p.sT), 40)):
            t = np.log(sg / p.s0) / p.r
            z0, _ = sample_mixture(m.w, m.muY, m.SY, 20000, rng); yt = z0 @ expm(p.B * t).T + sg * rng.standard_normal(z0.shape)
            s, sh = ex(yt, t), ls(yt, t); sZ, dZ = s @ m.P, (sh - s) @ m.P                  # score in Z = P^T score in Y
            out.append({'sigma': float(sg), 'e_Y': float((((sh - s)**2).sum(1)).mean() / (s**2).sum(1).mean()),
                        'e_Z': float((dZ**2).sum(1).mean() / (sZ**2).sum(1).mean()),
                        'e_Z_pos': float((dZ[:, POS]**2).sum(1).mean() / (sZ[:, POS]**2).sum(1).mean()),
                        'e_Z_vel': float((dZ[:, VEL]**2).sum(1).mean() / (sZ[:, VEL]**2).sum(1).mean())})
        near = [o for o in out if o['sigma'] <= 2 * p.s0]
        res = {'metric': metric, 'net_seed': ns, 'by_sigma': out, **{f'{k}_near_smin': float(np.mean([o[k] for o in near])) for k in ['e_Y', 'e_Z', 'e_Z_pos', 'e_Z_vel']}}
        # B10 addition: error in the tail of p_t (Mahalanobis distance to the nearest component in [3.5, 5]), where the outliers start
        rng = np.random.default_rng(32_000); tail = []
        for sg in list(np.exp(np.linspace(np.log(p.s0), np.log(2 * p.s0), 6))) + list(np.exp(np.linspace(np.log(0.01), np.log(0.03), 4))):
            t = np.log(sg / p.s0) / p.r
            z0, _ = sample_mixture(m.w, m.muY, m.SY, 200000, rng); yt = z0 @ expm(p.B * t).T + sg * rng.standard_normal(z0.shape)
            mt, Pi = ex.params(t); dd = yt[:, None] - mt[None]; dm = np.sqrt(np.einsum('nki,ij,nkj->nk', dd, Pi, dd).min(1))
            yt = yt[(dm >= 3.5) & (dm <= 5.0)]
            s, sh = ex(yt, t), ls(yt, t); sZ, dZ = s @ m.P, (sh - s) @ m.P
            tail.append({'sigma': float(sg), 'n': int(len(yt)), 'e_Z': float((dZ**2).sum(1).mean() / (sZ**2).sum(1).mean()),
                         'e_Z_pos': float((dZ[:, POS]**2).sum(1).mean() / (sZ[:, POS]**2).sum(1).mean())})
        res['tail'] = tail
        res['e_Z_tail_near_smin'] = float(np.mean([o['e_Z'] for o in tail if o['sigma'] <= 2 * p.s0 * (1 + 1e-9)]))
        res['e_Z_tail_01_03'] = float(np.mean([o['e_Z'] for o in tail if o['sigma'] >= 0.01 * (1 - 1e-9)]))
        json.dump(res, open(f'results/err_{metric}_n{ns}.json', 'w'), indent=1); print('err', metric, ns, res['e_Z_near_smin'], 'tail', res['e_Z_tail_near_smin'], res['e_Z_tail_01_03'], flush=True)
    elif cmd == 'estar':
        # B11, H7: tail error e* in the window W* = [lo, hi] (6 log-spaced sigma, Mahalanobis 3.5-5 points of p_t, seed 33 000)
        lo, hi = float(sys.argv[4]), float(sys.argv[5]); rng = np.random.default_rng(33_000); rows = []
        for sg in np.exp(np.linspace(np.log(lo), np.log(hi), 6)):
            t = np.log(sg / p.s0) / p.r
            z0, _ = sample_mixture(m.w, m.muY, m.SY, 200000, rng); yt = z0 @ expm(p.B * t).T + sg * rng.standard_normal(z0.shape)
            mt, Pi = ex.params(t); dd = yt[:, None] - mt[None]; dm = np.sqrt(np.einsum('nki,ij,nkj->nk', dd, Pi, dd).min(1))
            yt = yt[(dm >= 3.5) & (dm <= 5.0)]
            s, sh = ex(yt, t), ls(yt, t); sZ, dZ = s @ m.P, (sh - s) @ m.P
            rows.append({'sigma': float(sg), 'n': int(len(yt)), 'e_Z': float((dZ**2).sum(1).mean() / (sZ**2).sum(1).mean())})
        res = {'metric': metric, 'net_seed': ns, 'window': [lo, hi], 'rows': rows, 'e_star': float(np.mean([r['e_Z'] for r in rows]))}
        json.dump(res, open(f'results/estar_{metric}_n{ns}.json', 'w'), indent=1); print('estar', metric, ns, lo, hi, res['e_star'], flush=True)
    elif cmd == 'ode':
        # B11, H7: probability-flow ODE only, h ~ 0.002 (the level that defines 'broken'); the same prior draws as 'sample'
        sd = int(sys.argv[4]); Kf = int(np.ceil(p.T / 0.002 / 16)) * 16; t0 = time.time()
        y0 = m.prior(N, np.random.default_rng(100 + sd)); ref = mode_stats(m.truth(400000, np.random.default_rng(999)), m.mu)
        with np.errstate(all='ignore'): z = m.to_Z(run_sampler(ls, p, y0, 'ode', Kf))
        st = stats(z, m, ref); bad = ~np.isfinite(z).all(1); bad[~bad] = far_mask(z[~bad], m.mu); st['n_bad'] = int(bad.sum())
        st['broken'] = (not st['finite']) or abs(st['L']) > 1 or st['n_far'] / N > 0.01; st['far_idx'] = np.where(bad)[0].tolist()
        json.dump({'metric': metric, 'net_seed': ns, 'seed': sd, 'N': N, 'K': Kf, 'h': p.T / Kf, 'ode': st, 'sec': time.time() - t0},
                  open(f'results/ode_{metric}_n{ns}_s{sd}.json', 'w'), indent=1)
        print('ode', metric, ns, sd, 'L', st['L'], 'bad', st['n_bad'], 'broken', st['broken'], '%.0f s' % (time.time() - t0), flush=True)
    elif cmd == 'sample':
        sd = int(sys.argv[4]); Kf = int(np.ceil(p.T / 0.002 / 16)) * 16
        y0 = m.prior(N, np.random.default_rng(100 + sd)); ref = mode_stats(m.truth(400000, np.random.default_rng(999)), m.mu)
        out = {'metric': metric, 'r': R, 'kappa_P': m.kappa, 'cond': m.cond, 'net_seed': ns, 'seed': sd, 'N': N, 'levels': []}; t0 = time.time()
        for sub in SUBS:
            K = Kf // sub; row = {'K': K, 'h': p.T / K}
            for md in ['ode', 'fixed']:
                with np.errstate(all='ignore'):
                    z = m.to_Z(run_sampler(ls, p, y0, md, K, np.random.default_rng(200 + sd), sub=sub))
                row[md] = stats(z, m, ref)
                if md == 'ode' and sub == 1:                                    # keep the outliers for the mechanism study
                    bad = ~np.isfinite(z).all(1); bad[~bad] = far_mask(z[~bad], m.mu)
                    row['ode']['far_idx'] = np.where(bad)[0].tolist()
                    np.save(f'results/ode_final_{metric}_n{ns}_s{sd}.npy', z[bad].astype(np.float64))
            out['levels'].append(row)
            print(metric, ns, sd, 'h %.4f' % row['h'], 'ode L', row['ode']['L'], 'far', row['ode']['n_far'], '| ii L', row['fixed']['L'], '%.0f s' % (time.time() - t0), flush=True)
        out['sec'] = time.time() - t0
        json.dump(out, open(f'results/samp_{metric}_n{ns}_s{sd}.json', 'w'), indent=1)
