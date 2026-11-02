# cld-learned-score-samplers

**Learned scores on the critically damped Langevin flow: a corrected sampler in 281 steps, and a metric-dependent fragility of the probability flow**

Code, stored results, the 28 trained networks and a verification script for the paper by Taishi Namba (2026).

**Paper:** T. Namba, *Learned scores on the critically damped Langevin flow: a corrected sampler in 281 steps, and a metric-dependent fragility of the probability flow*, preprint, Zenodo (2026).

[![verify](https://github.com/neginyan/cld-learned-score-samplers/actions/workflows/verify.yml/badge.svg)](https://github.com/neginyan/cld-learned-score-samplers/actions/workflows/verify.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## The result in brief

This is the follow-up to [paper G](https://doi.org/10.5281/zenodo.23689912) (code: [cld-corrected-sampler](https://github.com/neginyan/cld-corrected-sampler)). There, the critically damped Langevin (CLD) flow was combined with an isotropic noise schedule specified by its marginals, σ(t) = σ_min e^{rt}, and a corrected reverse SDE was shown to stay exact where the usual repair is biased. Here the same setting is run with **learned scores**, and the isotropic blur is added in a rescaled metric Y = PZ.

**The standard CLD, whose noise enters through the velocity only and is given by its SDE, does not show the failures studied here.** All results are for one four-dimensional Gaussian mixture (setting D: σ_min = 0.005, σ_max = 40, r = 1).

- **A corrected sampler in 281 steps.** A Heun-type integrator of the corrected reverse SDE, driven by a learned score and with no retraining, reaches a median |L| ≤ 0.083 at 281 steps in all four metrics tested. Euler–Maruyama on the same Brownian path gives 0.42–0.55. At 140 steps the Heun-type sampler no longer suffices.
- **A metric-dependent fragility of the probability flow.** With P = I (κ_P = 2 > r = 1) the learned probability-flow ODE broke for 10 of 12 networks; with metrics of κ_P ≤ 0, for 3 of 16. At the network level only one pre-registered comparison was significant (opt1.25, 0 of 4).
- **What breaks, and what we could not explain.** Broken runs contain at most 0.12% outliers; a few tail trajectories fly far. They start in the tail of the prior, pick up most of their deviation while σ is large, and jump at the end. The learned score amplifies perturbations of part of the tail in the last stretch in every metric, so how the metric acts remains open.
- **Thirteen of nineteen pre-registered predictions failed.** All of them are reported in the paper (Table 3), and the original pre-registration files are in `preregistration/`.

Every statement in the paper carries its status: a *Proposition* (proved in an earlier paper, cited), a *Pre-registered test* (criterion fixed and time-stamped before the runs), an *Observation*, or a *Conjecture*. There are no new proofs in this paper.

```diff
 # one step of the corrected reverse SDE, drift f = B_P y - 0.5 (G + G_plus) s_hat, noise dW ~ N(0, h G_plus)
-y_next = y - h * f(y, t) + dW                                  # Euler-Maruyama
+y_pred = y - h * f(y, t) + dW
+y_next = y - 0.5 * h * (f(y, t) + f(y_pred, t - h)) + dW       # Heun-type (same dW in both stages)
```

## Verify

```bash
pip install -r requirements.txt
python verification/run_all.py          # about 15 minutes on two CPU cores
python verification/run_all.py --full   # also re-runs all 32 Heun-type runs at 281 steps
```

`run_all.py` checks:
- every table and figure number, recomputed from the stored results by `scripts/make_paper.py`, against `verification/expected_numbers.json`;
- the tables of the paper (`paper/tables/`) against the regenerated ones;
- the SHA-256 of every stored result against `results/MANIFEST.json`;
- re-runs from the trained networks: the Heun-type sampler in all four metrics, Euler–Maruyama, the probability-flow ODE for P = I and opt1.25, and the injection experiment of Test H9.

No GPU is used. The score networks are a small NumPy MLP. On the machine that produced the results the re-runs agree bit for bit. On other CPUs (for example GitHub's runners) float32 sums differ slightly, so the check then requires a relative difference of at most 1e-3 for L (for a broken run, that it is broken again), and compares the medians of the amplification in Test H9.

## Repository layout

| folder | content |
|---|---|
| `src/` | samplers, metric set-up, networks, and the scripts of the window, Heun-type and injection experiments |
| `nets/` | the 28 trained networks (300 000 steps): P = I seeds 1–12, diag(1,2) seeds 1–8, opt1.25 and opt5 seeds 1–4 |
| `results/B8 … B12/` | stored results of the lab stages, copied unchanged (`results/MANIFEST.json` lists their origin and SHA-256) |
| `scripts/` | `make_paper.py` (every table and figure from the stored results); `collect_from_stages.py` (how the results were collected) |
| `tables/`, `figures/` | regenerated tables and figures |
| `verification/` | `run_all.py` and the expected numbers |
| `paper/` | LaTeX source of the paper (REVTeX 4.2) |
| `preregistration/` | pre-registration files written before the runs (Japanese originals, unmodified) and the hashes recorded before sampling; Table 3 of the paper lists every prediction in English |

The networks for P = I seeds 1–2 were trained in an earlier stage and reused unchanged; diag(1,2) seeds 1–2 come from a later stage, the others from the stages reported here. Resumed training reproduces an uninterrupted run bit for bit.

## Related work by the author

- T. Namba, *When the blur is slower than the flow: clipped and corrected samplers on the critically damped Langevin flow*, Zenodo (2026), [doi:10.5281/zenodo.23689912](https://doi.org/10.5281/zenodo.23689912) (paper G; code [cld-corrected-sampler](https://github.com/neginyan/cld-corrected-sampler)).
- T. Namba, *A stretching-rate threshold for phase-space diffusion models with isotropic marginal noise schedules*, Zenodo (2026), [doi:10.5281/zenodo.23181225](https://doi.org/10.5281/zenodo.23181225). The threshold and the exactness of the corrected sampler are proved there.

## Use of AI assistants

The study was carried out with AI assistants: Anthropic Claude (planning, training, computations, code, verification and drafting), Google Gemini and OpenAI ChatGPT (review of the manuscript). Their roles are described in the acknowledgments of the paper. The author directed the research and takes full responsibility for it.

## Citation

Please cite the paper and this repository; see `CITATION.cff`.

## License

MIT © 2026 Taishi Namba
