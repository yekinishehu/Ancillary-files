# Ancillary files

Companion code for

> Y. Shehu, *Adaptivity, Anchoring, and the Exact Oracle Complexity of Stochastic
> Fixed-Point Iterations*.

This repository reproduces the numerical results of **Section 6 (Numerical validation)**
of the manuscript. All stochastic experiments use the Gaussian-noise single-point oracle of
Definition 2.1 with fixed seeds; the numbers below were produced with Python 3 + NumPy/SciPy
(current CI run) and match the manuscript up to the seed/build tolerances stated per item.

## Contents

| File | Reproduces |
|---|---|
| `generate_figures.py` | Experiment 1 -> Figure 1; Experiment 3 -> Figure 4; Experiment 5 -> **Table 1, checked to the exact integer** |
| `superseded/` | archival verification code from a pre-revision appendix; not needed for the current manuscript |

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`

## Usage

```bash
python generate_figures.py
```

This writes `fig1_affine_estimator.pdf` and `fig4_A1_nonexpansive.pdf` to the working
directory and prints the Table 1 check, one line per grid point:

```
[table1] phi=0.005 eps=0.1: N=10961 (table: 10961) OK
...
[table1] all reproduced: True
```

## What is verified, and the expected output

**Figure 1 (Experiment 1; Theorem 5.2, two-point affine estimator, sigma = 0.1, z = 0.37, D = 1).**
- Success probability at batch m = 400 over 40,000 replicates: **0.6586**
  (manuscript: 0.66 with 95% CI [0.655, 0.664]; probit crossing m* ~ 394 +- 30,
  manuscript: m ~ 400 +- 30).
- Log-log slopes of the required batch: **2.00** (vs 1/(1-gamma)) and **1.91** (vs 1/eps)
  (manuscript: 1.99 and 1.90; deviations of order 0.01 reflect the RNG/BLAS build and are
  within the +-1 standard-error bands plotted in the figure).

**Figure 4 (Experiment 3; Theorem 4.2 on the rotation R_0.3, D = 1, sigma = 1, 20 runs).**
- Certified horizons N = 28 / 48 / 88 at eps = 0.2 / 0.1 / 0.05.
- Query counts per run: **81,200 / 940,800 / 12,531,200**
  (manuscript: 8.1e4 / 9.4e5 / 1.3e7).
- Mean terminal residuals: **0.063 / 0.038 / 0.016**
  (manuscript: 0.06 / 0.04 / 0.02).

**Table 1 (Experiment 5; Theorem 3.8(ii), oscillatory escape on R_phi, exact arithmetic,
c' = eps/(96D)).** All twelve hitting times N_phi(eps) on the grid
(phi, eps) in {0.005, 0.01, 0.02} x {0.1, 0.05, 0.02, 0.01} are reproduced **to the exact
integer** (10961, 40033, 179507, 644482 / 5481, 20017, 89754, 322243 / 2742, 10009, 44879,
161428); the ratios to the reference scale phi^{-1} eps^{-1/alpha_0}, alpha_0 = log2(3/2),
all lie in [1.07, 1.23].

## Scope

Experiments 2 and 4 (Figures 3 and 5) are deterministic schedule simulations whose
defining parameters (batch schedules, window lengths) are stated in full in Section 6 and
in Theorems 4.2 and 3.11, respectively; Experiment 3's rotation-law panel (Figure 2) is the
closed-form envelope of Lemma 3.4. The stochastic content of Section 6 (Figures 1 and 4)
and the exact Table 1 are reproduced by `generate_figures.py` as described above.

## Citation

If you use this code, please cite the manuscript. (arXiv link to be added upon posting.)
