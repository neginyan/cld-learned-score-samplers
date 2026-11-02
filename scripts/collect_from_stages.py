"""Copy the stored results and the trained networks of the lab stages B3-B12 into this repository (run once, by the author).
Locations via environment variables B3_DIR, B8_DIR, B9_DIR, B10_DIR, B11_DIR, B12_DIR (defaults /home/claude/B3 ...).
Network files are renamed to <metric>_D_seed<n>_step300000.{npz,json}; nothing is changed inside them."""
import os, glob, shutil, hashlib, json
D = {k: os.environ.get(f'{k}_DIR', f'/home/claude/{k}') for k in ('B3', 'B8', 'B9', 'B10', 'B11', 'B12')}
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); manifest = {}
def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.copy2(src, dst)
    manifest[os.path.relpath(dst, root)] = {'from': src, 'sha256': hashlib.sha256(open(dst, 'rb').read()).hexdigest()}
# networks (30 万 steps only)
src_net = {}
for n in (1, 2): src_net[('I', n)] = f"{D['B3']}/results/nets/D_seed{n}_step300000"
for n in (1, 2): src_net[('diag2', n)] = f"{D['B8']}/results/nets/diag2_D_seed{n}_step300000"
for mt in ('I', 'diag2'):
    for n in (3, 4): src_net[(mt, n)] = f"{D['B9']}/results/nets/{mt}_D_seed{n}_step300000"
    for n in (5, 6, 7, 8): src_net[(mt, n)] = f"{D['B10']}/results/nets/{mt}_D_seed{n}_step300000"
for mt in ('opt1.25', 'opt5'):
    for n in (1, 2, 3, 4): src_net[(mt, n)] = f"{D['B9']}/results/nets/{mt}_D_seed{n}_step300000"
for n in (9, 10, 11, 12): src_net[('I', n)] = f"{D['B11']}/results/nets/I_D_seed{n}_step300000"
for (mt, n), s in src_net.items():
    for ext in ('npz', 'json'): cp(f'{s}.{ext}', f'{root}/nets/{mt}_D_seed{n}_step300000.{ext}')
pats = {'B9': ['samp_*_n*_s*.json', 'heun_*_s*.json', 'err_*.json'],
        'B10': ['samp_*_n*_s*.json', 'heunL_*.json', 'swap_*.json', 'err_*.json'],
        'B11': ['heun8_*.json', 'win_*.json', 'track_I_*.json', 'inject_*.json', 'estar_*.json', 'ode_*.json', 'h7_fit.json', 'err_I_n*.json', 'pool_I_n*_s*.npz'],
        'B12': ['h9_*.json', 'band_*.json', 'bandm_*.json']}
for st, ps in pats.items():
    for p in ps:
        for f in sorted(glob.glob(f"{D[st]}/results/{p}")): cp(f, f'{root}/results/{st}/{os.path.basename(f)}')
for st in ('B9', 'B10'): cp(f"{D[st]}/results/summary_{st}.json", f'{root}/results/{st}/summary_{st}.json')
for mt in ('I', 'diag2', 'opt1.25', 'opt5'):
    for s in (0, 1): cp(f"{D['B8']}/results/B_{mt}_r1_s{s}.json", f'{root}/results/B8/B_{mt}_r1_s{s}.json')
json.dump(manifest, open(f'{root}/results/MANIFEST.json', 'w'), indent=1)
print(len(manifest), 'files copied')
