"""Every table and figure number of the paper, from the stored results in results/ (no network is run here).
  python scripts/make_paper.py  -> results/paper_numbers.json, tables/*.tex, figures/fig*.{pdf,png}"""
import os, json, glob
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(root)
R = lambda f: json.load(open(f)); inf = float('inf')
MET = ['I', 'diag2', 'opt1.25', 'opt5']; NAME = {'I': '$P=I$', 'diag2': r'$\dtwo$', 'opt1.25': r'$\optc{1.25}$', 'opt5': r'$\optc{5}$'}
KAP = {'I': 2, 'diag2': 0, 'opt1.25': 0, 'opt5': -6}
NETS = {'I': range(1, 13), 'diag2': range(1, 9), 'opt1.25': range(1, 5), 'opt5': range(1, 5)}
aL = lambda st: inf if (st.get('L') is None or not st.get('finite', True)) else abs(st['L'])
out = {}

# ------------------------------------------------------------------ result 1: Heun-type corrected SDE, learned score
B9S, B10S = R('results/B9/summary_B9.json'), R('results/B10/summary_B10.json')
heun = {}
for mt in MET:
    lv = {}
    for n in range(1, 5):
        for s in (0, 1):
            h8 = R(f'results/B11/heun8_{mt}_n{n}_s{s}.json')['levels']
            rows = [(r['h'], r['heun'], r.get('em')) for r in h8]
            if mt in ('I', 'diag2'):
                rows += [(l['h'], l, None) for l in R(f'results/B10/heunL_{mt}_n{n}_s{s}.json')['levels']]
            em9 = R(f'results/B9/samp_{mt}_n{n}_s{s}.json')['levels']
            for h, st, em in rows:
                key = '%.3f' % h; d = lv.setdefault(key, {'heun': [], 'heun_bad': [], 'em': [], 'em_bad': []})
                d['heun'].append(aL(st)); d['heun_bad'].append(st['n_bad'])
                if em is not None: d['em'].append(aL(em)); d['em_bad'].append(em['n_bad'])
            for l in em9[:3]:                       # EM from B9 at h = 0.032, 0.016, 0.008 (same nets, same Brownian path)
                key = '%.3f' % l['h']; d = lv.setdefault(key, {'heun': [], 'heun_bad': [], 'em': [], 'em_bad': []})
                d['em'].append(aL(l['fixed'])); d['em_bad'].append(l['fixed']['n_far'] if l['fixed']['finite'] else None)
    heun[mt] = {h: {'n_runs': len(d['heun']), 'heun_median_absL': float(np.median(d['heun'])), 'heun_n_nonfinite': int(np.sum(np.isinf(d['heun']))),
                    'heun_max_finite_absL': float(max([x for x in d['heun'] if np.isfinite(x)], default=np.nan)),
                    'heun_n_broken': int(sum((x > 1) for x in d['heun'])), 'heun_runs_with_outliers': int(sum(b > 0 for b in d['heun_bad'])),
                    'heun_max_outliers': int(max(d['heun_bad'])), 'heun_absL': d['heun'], 'heun_bad': d['heun_bad'],
                    'em_median_absL': float(np.median(d['em'])) if d['em'] else None, 'em_n_nonfinite': int(np.sum(np.isinf(d['em']))) if d['em'] else None,
                    'em_absL': d['em'], 'em_bad': d['em_bad']} for h, d in sorted(lv.items(), key=lambda x: -float(x[0]))}
out['heun'] = heun
ex9 = {mt: R(f'results/B9/heun_{mt}_s0.json') for mt in ('I', 'diag2')}
out['heun_exact_B9'] = B9S['heun']

# ------------------------------------------------------------------ result 2: metric and ODE breakage
runs = {}
for mt in MET:
    rr = []
    for r in B9S['runs'][mt]:
        st = r['ode'][-1]; rr.append({'net': r['net'], 'seed': r['seed'], 'stage': 'B9', 'broken': r['broken'], 'absL': aL(st), 'n_bad': r['n_bad_h0002']})
    if mt in ('I', 'diag2'):
        for r in B10S['runs'][mt]:
            if r['new']: rr.append({'net': r['net'], 'seed': r['seed'], 'stage': 'B10', 'broken': r['broken'], 'absL': aL(r), 'n_bad': r['n_bad']})
    if mt == 'I':
        for n in range(9, 13):
            for s in (0, 1):
                o = R(f'results/B11/ode_I_n{n}_s{s}.json')['ode']; rr.append({'net': n, 'seed': s, 'stage': 'B11', 'broken': o['broken'], 'absL': aL(o), 'n_bad': o['n_bad']})
    runs[mt] = rr
nb = lambda mt, nets=None: [any(r['broken'] for r in runs[mt] if r['net'] == n) for n in sorted({r['net'] for r in runs[mt]}) if nets is None or n in nets]
def fish(a, b): return float(fisher_exact([[sum(a), len(a) - sum(a)], [sum(b), len(b) - sum(b)]], alternative='greater')[1])
I4, I8 = nb('I', range(1, 5)), nb('I', range(1, 9))
t = {'B9_net': {mt: {'broken': sum(nb(mt, range(1, 5))), 'n': 4, 'p': fish(I4, nb(mt, range(1, 5)))} for mt in MET[1:]},
     'B9_run': {mt: {'broken': sum(r['broken'] for r in runs[mt] if r['stage'] == 'B9'), 'n': 8,
                     'p': fish([r['broken'] for r in runs['I'] if r['stage'] == 'B9'], [r['broken'] for r in runs[mt] if r['stage'] == 'B9'])} for mt in MET[1:]},
     'B10_net_8v8': {'I': sum(I8), 'diag2': sum(nb('diag2')), 'p': fish(I8, nb('diag2'))},
     'all_nets': {mt: {'broken': sum(nb(mt)), 'n': len(nb(mt))} for mt in MET},
     'posthoc_I12_vs_diag2_8_p': fish(nb('I'), nb('diag2')),
     'posthoc_I12_vs_kappa_le0_16_p': fish(nb('I'), nb('diag2') + nb('opt1.25') + nb('opt5'))}
ps = sorted([(t['B9_net'][m]['p'], m) for m in ('opt1.25', 'opt5')]); hol = {}; run_max = 0
for i, (p_, m) in enumerate(ps): run_max = max(run_max, min(1, p_ * (2 - i))); hol[m] = run_max
for m in hol: t['B9_net'][m]['p_holm'] = hol[m]
ps = sorted([(t['B9_run'][m]['p'], m) for m in ('opt1.25', 'opt5')]); run_max = 0
for i, (p_, m) in enumerate(ps): run_max = max(run_max, min(1, p_ * (2 - i))); t['B9_run'][m]['p_holm'] = run_max
for mt in MET:
    ok = [r for r in runs[mt] if not r['broken']]; br = [r for r in runs[mt] if r['broken']]
    t['all_nets'][mt].update({'runs': len(runs[mt]), 'runs_broken': len(br), 'outliers_range': [min(r['n_bad'] for r in runs[mt]), max(r['n_bad'] for r in runs[mt])],
                              'outliers_broken_range': [min((r['n_bad'] for r in br), default=None), max((r['n_bad'] for r in br), default=None)],
                              'median_absL_unbroken': float(np.median([r['absL'] for r in ok])) if ok else None,
                              'range_absL_unbroken': [float(min(r['absL'] for r in ok)), float(max(r['absL'] for r in ok))] if ok else None,
                              'n_nonfinite_runs': int(sum(np.isinf(r['absL']) for r in runs[mt]))})
out['ode'] = t; out['ode_runs'] = runs

# ------------------------------------------------------------------ mechanism
sw = {}
for f in sorted(glob.glob('results/B10/swap_I_n*_s*.json')):
    x = R(f)
    if x['net_seed'] > 4: continue
    for c, v in x['A_learned_then_exact'].items(): sw.setdefault(('A', c), []).append(bool(v['broken']))
    for c, v in x['B_exact_then_learned'].items(): sw.setdefault(('B', c), []).append(bool(v['broken']))
out['swap_B9_I'] = {f'{a}_{c}': [sum(v), len(v)] for (a, c), v in sw.items()}
W = {}
for f in sorted(glob.glob('results/B11/win_I_n*_s*.json')):
    for k, v in R(f)['windows'].items(): W[k] = W.get(k, 0) + bool(v['broken'])
out['windows_I'] = W
trk = {'outlier': {}, 'tail_ref': {}, 'bulk_ref': {}}; pm = []
for f in sorted(glob.glob('results/B11/track_I_n*_s*.json')):
    x = R(f); pm += x['groups']['outlier']['prior_mahalanobis']
    for g in trk:
        for r in x['groups'][g]['rows']: trk[g].setdefault(str(r['sigma']), []).append(r['median_dev'])
out['tracking'] = {g: {s: [float(np.nanmin([np.nan if v is None else v for v in vs])), float(np.nanmax([np.nan if v is None else v for v in vs]))] for s, vs in d.items()} for g, d in trk.items()}
out['outlier_prior_mahalanobis'] = [float(min(pm)), float(max(pm)), len(pm)]
inj = {mt: R(f'results/B11/inject_{mt}.json') for mt in MET}
out['h6'] = {mt: {c: inj[mt]['by_sigma'][c]['tail_pool']['A_median'] for c in inj[mt]['by_sigma']} for mt in MET}
h9 = {}
for mt in MET:
    for n in range(1, 5):
        x = R(f'results/B12/h9_{mt}_n{n}.json')['by_sigma']
        h9[f'{mt}_n{n}'] = {c: {'A_E': x[c]['tail_pool']['exact']['A_median'], 'A_L': x[c]['tail_pool']['learned']['A_median'], 'R': x[c]['tail_pool']['R'],
                                'R90': x[c]['tail_pool']['R90'], 'q90_E': x[c]['tail_pool']['exact']['A_q90'], 'q90_L': x[c]['tail_pool']['learned']['A_q90'],
                                'far_E': x[c]['tail_pool']['exact']['far_fraction'], 'far_L': x[c]['tail_pool']['learned']['far_fraction'],
                                'nonfinite_L': x[c]['tail_pool']['learned']['n_nonfinite'], 'R_iso': x[c]['tail_iso']['R'], 'R_bulk': x[c]['bulk_pool']['R']}
                            for c in ('0.03', '0.01')}
Rv = {k: v['0.01']['R'] for k, v in h9.items()}; R90 = {k: v['0.01']['R90'] for k, v in h9.items()}
hi = [Rv[f'{m}_n{n}'] for m in ('I', 'diag2') for n in range(1, 5)]; lo = [Rv[f'{m}_n{n}'] for m in ('opt1.25', 'opt5') for n in range(1, 5)]
out['h9'] = {'per_net': h9, 'H9a_count_R_ge_10': sum(v >= 10 for v in Rv.values()), 'H9a_supported': sum(v >= 10 for v in Rv.values()) >= 12,
             'H9b_p': float(mannwhitneyu(hi, lo, alternative='greater').pvalue), 'H9c_count_R90_ge_10': sum(v >= 10 for v in R90.values()),
             'R_median_by_metric': {m: float(np.median([Rv[f'{m}_n{n}'] for n in range(1, 5)])) for m in MET},
             'R_range_by_metric': {m: [float(min(Rv[f'{m}_n{n}'] for n in range(1, 5))), float(max(Rv[f'{m}_n{n}'] for n in range(1, 5)))] for m in MET}}
out['h9']['H9b_supported'] = out['h9']['H9b_p'] <= 0.05; out['h9']['H9c_supported'] = out['h9']['H9c_count_R90_ge_10'] >= 12
band = {'outlier': [], 'tail_ref': [], 'bulk_ref': []}
for f in sorted(glob.glob('results/B12/band_I_n*_s*.json')):
    x = R(f)
    for g in band: band[g].append(x['groups'][g]['median_made'])
out['band_made_median_over_runs'] = {g: np.median(np.array(v), 0).tolist() for g, v in band.items()}
fit = R('results/B11/h7_fit.json'); out['h7'] = {'t': fit['I_1_8']['t'], 'correct_I_1_8': fit['I_1_8']['correct'], 'all24': fit['ref_all24']}
out['nets'] = []
for mt in MET:
    br = dict(zip(sorted({r['net'] for r in runs[mt]}), nb(mt)))
    for n in NETS[mt]:
        e = R(f'results/B10/err_{mt}_n{n}.json') if os.path.exists(f'results/B10/err_{mt}_n{n}.json') else R(f'results/B11/err_{mt}_n{n}.json')
        out['nets'].append({'metric': mt, 'net': n, 'e_Z': e['e_Z_near_smin'], 'e_Z_tail': e['e_Z_tail_near_smin'], 'broken': br[n],
                            'max_absL': max(r['absL'] for r in runs[mt] if r['net'] == n), 'max_outliers': max(r['n_bad'] for r in runs[mt] if r['net'] == n)})
os.makedirs('results', exist_ok=True)
json.dump(out, open('results/paper_numbers.json', 'w'), indent=1, default=lambda v: None)

# ------------------------------------------------------------------ tables
os.makedirs('tables', exist_ok=True)
fmt = lambda x: r'$\infty$' if not np.isfinite(x) else ('%.3f' % x if x < 10 else '%.0f' % x)
HS = ['0.064', '0.032', '0.016', '0.008']; KS = {'0.064': 140, '0.032': 281, '0.016': 562, '0.008': 1124}
L = [r'\begin{table}[t]', r'\caption{Corrected SDE \Sb{} with a learned score: median $|L|$ over 8 runs (networks 1--4 $\times$ seeds 0, 1), Heun-type integrator [Eq.~\eqref{eq:heun}] and Euler--Maruyama (EM) on the same Brownian path. $\infty$: at least half of the runs non-finite. $n_{\rm nf}$: non-finite Heun runs; $n_{\rm o}$: Heun runs with at least one outlier.}',
     r'\label{tab:heun}', r'\begin{ruledtabular}', r'\begin{tabular}{lcccc}', r' & $K=140$ & $281$ & $562$ & $1124$ \\', r' & $h\approx0.064$ & $0.032$ & $0.016$ & $0.008$ \\', r'\hline']
for mt in MET:
    L.append(NAME[mt] + ' Heun & ' + ' & '.join(fmt(heun[mt][h]['heun_median_absL']) for h in HS) + r' \\')
    L.append(r'\quad EM & ' + ' & '.join(fmt(heun[mt][h]['em_median_absL']) for h in HS) + r' \\')
    L.append(r'\quad $n_{\rm nf}$, $n_{\rm o}$ & ' + ' & '.join(f"{heun[mt][h]['heun_n_nonfinite']}, {heun[mt][h]['heun_runs_with_outliers']}" for h in HS) + r' \\')
L += [r'\end{tabular}', r'\end{ruledtabular}', r'\end{table}']
open('tables/tab_heun.tex', 'w').write('\n'.join(L) + '\n')
L = [r'\begin{table}[t]', r'\caption{Probability-flow ODE \Sc{} with learned scores ($h\approx0.002$, $N=20\,000$, two seeds per network). A network is broken if either run is. Outliers: per run, out of $20\,000$. $|L|$: median over the runs that did not break.}',
     r'\label{tab:ode}', r'\begin{ruledtabular}', r'\begin{tabular}{lccccc}', r' & $\kP$ & networks & runs & outliers & $|L|$ \\', r' & & broken & broken & per run & unbroken \\', r'\hline']
for mt in MET:
    a = t['all_nets'][mt]
    L.append(f"{NAME[mt]} & ${KAP[mt]}$ & {a['broken']}/{a['n']} & {a['runs_broken']}/{a['runs']} & {a['outliers_range'][0]}--{a['outliers_range'][1]} & "
             + (fmt(a['median_absL_unbroken']) if a['median_absL_unbroken'] is not None else '--') + r' \\')
L += [r'\end{tabular}', r'\end{ruledtabular}', r'\end{table}']
open('tables/tab_ode.tex', 'w').write('\n'.join(L) + '\n')
L = [r'\begin{table}[h]', r'\caption{All 28 networks. $e_Z$: relative mean-square score error near $\smin$ on typical points (tail: Mahalanobis distance 3.5--5). Broken: probability-flow ODE, either seed. Max $|L|$ and outliers over the two runs.}',
     r'\label{tab:nets}', r'\footnotesize', r'\begin{ruledtabular}', r'\begin{tabular}{lccccc}', r'network & $e_Z$ & tail & broken & max $|L|$ & outliers \\', r'\hline']
for x in out['nets']:
    L.append(f"{NAME[x['metric']] + ' ' if x['net'] == 1 else ''}{x['net']} & {x['e_Z']:.3f} & {x['e_Z_tail']:.3f} & {'yes' if x['broken'] else 'no'} & {fmt(x['max_absL'])} & {x['max_outliers']}" + r' \\')
L += [r'\end{tabular}', r'\end{ruledtabular}', r'\end{table}']
open('tables/tab_nets.tex', 'w').write('\n'.join(L) + '\n')

# ------------------------------------------------------------------ figures
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('figures', exist_ok=True)
C = {'I': '#2a6fdb', 'diag2': '#e07b24', 'opt1.25': '#2f9e6e', 'opt5': '#8e5bd6'}; LAB = {'I': r'$P=I$', 'diag2': r'diag$(1,2)$', 'opt1.25': r'opt$_{1.25}$', 'opt5': r'opt$_5$'}
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
TOP = 30
# Fig. 1: Heun vs EM
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
for i, mt in enumerate(MET):
    hs = [float(h) for h in HS]; off = 1 + 0.035 * (i - 1.5)
    for a, key, ls_ in ((ax[0], 'heun', '-'), (ax[1], 'em', '--')):
        med = [min(heun[mt][h][f'{key}_median_absL'], TOP) for h in HS]
        a.plot(hs, med, ls_, color=C[mt], lw=1.8, marker='o', ms=4, label=LAB[mt])
        for h in HS:
            v = np.minimum(np.array(heun[mt][h][f'{key}_absL'], float), TOP)
            a.scatter(np.full(len(v), float(h) * off), v, s=6, color=C[mt], alpha=0.35, lw=0)
for a, tt in ((ax[0], '(a) Heun-type, learned score'), (ax[1], '(b) Euler–Maruyama, learned score')):
    a.axhline(0.1, color='#999', lw=0.8, ls=':'); a.set_xscale('log'); a.set_yscale('log'); a.set_xlabel('step $h$'); a.set_title(tt, fontsize=8)
    a.set_xticks([0.008, 0.016, 0.032, 0.064]); a.set_xticklabels(['0.008', '0.016', '0.032', '0.064']); a.minorticks_off()
ax[0].set_ylabel(r'$|L|$ (runs; line: median; top: $\geq30$ or non-finite)'); ax[0].legend(fontsize=7, loc='upper left')
fig.tight_layout(); fig.savefig('figures/fig1_heun.pdf'); fig.savefig('figures/fig1_heun.png', dpi=200); plt.close(fig)
# Fig. 2: ODE breakage per network
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [1.6, 1]})
x0 = 0
for mt in MET:
    nets = sorted({r['net'] for r in runs[mt]})
    for j, n in enumerate(nets):
        for r in runs[mt]:
            if r['net'] != n: continue
            y = min(r['absL'], 300)
            ax[0].scatter(x0 + j, y, s=14, color=C[mt] if r['broken'] else 'white', edgecolor=C[mt], lw=1, marker='^' if np.isinf(r['absL']) else 'o', zorder=3)
    ax[0].text(x0 + (len(nets) - 1) / 2, 1500, LAB[mt], ha='center', fontsize=7, color=C[mt])
    x0 += len(nets) + 1.5
ax[0].axhline(1, color='#999', lw=0.8, ls=':'); ax[0].set_yscale('log'); ax[0].set_ylim(1e-3, 4000); ax[0].set_xticks([])
ax[0].set_ylabel(r'$|L|$ of the ODE, $h\approx0.002$'); ax[0].set_xlabel('networks (filled: broken run; triangle: non-finite, drawn at 300)')
ax[0].set_title('(a) runs, grouped by network', fontsize=8)
for i, mt in enumerate(MET):
    v = [r['n_bad'] for r in runs[mt]]; b = [r['broken'] for r in runs[mt]]
    xs = i + np.random.default_rng(i).uniform(-0.18, 0.18, len(v))
    ax[1].scatter(xs, v, s=12, c=[C[mt] if bb else 'white' for bb in b], edgecolors=C[mt], lw=1)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels([LAB[m] for m in MET]); ax[1].set_ylabel('outliers per run (of 20 000)'); ax[1].axhline(200, color='#999', lw=0.8, ls=':')
ax[1].set_ylim(-1, 30); ax[1].set_title('(b) outliers (1% would be 200)', fontsize=8)
fig.tight_layout(); fig.savefig('figures/fig2_ode.pdf'); fig.savefig('figures/fig2_ode.png', dpi=200); plt.close(fig)
# Fig. 3: mechanism
EDG = [40.0, 1.0, 0.3, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw={'width_ratios': [1.05, 1, 1]})
M = np.full((8, 8), np.nan)
for k, c in W.items():
    lo_, hi_ = map(float, k.split('-')); M[EDG.index(lo_) - 1, EDG.index(hi_)] = c
ax[0].imshow(M, cmap='Blues', vmin=0, vmax=12)
for i in range(8):
    for j in range(8):
        if not np.isnan(M[i, j]): ax[0].text(j, i, int(M[i, j]), ha='center', va='center', fontsize=6, color='white' if M[i, j] > 7 else '#222')
ax[0].set_xticks(range(8)); ax[0].set_xticklabels(['%g' % e for e in EDG[:-1]], fontsize=6); ax[0].set_yticks(range(8)); ax[0].set_yticklabels(['%g' % e for e in EDG[1:]], fontsize=6)
ax[0].set_xlabel(r'window top $\sigma_2$'); ax[0].set_ylabel(r'window bottom $\sigma_1$'); ax[0].set_title('(a) runs broken (of 12)', fontsize=8)
for spine in ax[0].spines.values(): spine.set_visible(False)
sg = ['1.0', '0.3', '0.1', '0.05', '0.03', '0.02', '0.01', '0.005']
for f in sorted(glob.glob('results/B11/track_I_n*_s*.json')):
    x = R(f)
    for g, col in (('outlier', '#c0392b'), ('tail_ref', '#2a6fdb'), ('bulk_ref', '#888888')):
        y = [r['median_dev'] if r['median_dev'] is not None else np.nan for r in x['groups'][g]['rows']]
        ax[1].plot([float(s) for s in sg], np.minimum(y, 1e3), color=col, lw=0.8, alpha=0.6)
ax[1].plot([], [], color='#c0392b', label='final outliers'); ax[1].plot([], [], color='#2a6fdb', label='tail (3.5–5)'); ax[1].plot([], [], color='#888888', label='bulk (<2)')
ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].invert_xaxis(); ax[1].set_xlabel(r'$\sigma$'); ax[1].set_ylabel(r'median $|\Delta z|$, learned vs exact path')
ax[1].set_title('(b) deviation (12 runs, $P=I$)', fontsize=8); ax[1].legend(fontsize=6, loc='upper left')
for i, mt in enumerate(MET):
    AE = inj[mt]['by_sigma']['0.01']['tail_pool']['A_median']
    ax[2].scatter(i - 0.15, AE, marker='s', s=22, color=C[mt], edgecolor='k', lw=0.5, zorder=3)
    AL = [h9[f'{mt}_n{n}']['0.01']['A_L'] for n in range(1, 5)]
    ax[2].scatter(i + 0.15 + np.linspace(-0.06, 0.06, 4), AL, marker='o', s=16, color=C[mt], zorder=3)
ax[2].scatter([], [], marker='s', color='#777', edgecolor='k', lw=0.5, label='exact score'); ax[2].scatter([], [], marker='o', color='#777', label='learned (4 networks)')
ax[2].set_xticks(range(4)); ax[2].set_xticklabels([LAB[m] for m in MET], fontsize=7); ax[2].set_yscale('log'); ax[2].set_ylabel(r'median amplification $A$')
ax[2].set_title(r'(c) amplification from $\sigma=0.01$', fontsize=8); ax[2].legend(fontsize=6, loc='upper left')
fig.tight_layout(); fig.savefig('figures/fig3_mechanism.pdf'); fig.savefig('figures/fig3_mechanism.png', dpi=200); plt.close(fig)
print(json.dumps({'ode': t, 'h9': {k: v for k, v in out['h9'].items() if k != 'per_net'}, 'swap': out['swap_B9_I']}, indent=1, default=lambda v: None))
