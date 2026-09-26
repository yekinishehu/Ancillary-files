# Ancillary files

Companion code for

> Y. Shehu, *Adaptivity, Anchoring, and the Exact Oracle Complexity of Stochastic
> Fixed-Point Iterations*.

This repository reproduces the numerical results of **Section 6 (Numerical validation)**
of the manuscript. All stochastic experiments use the Gaussian-noise single-point oracle of
Definition 2.1 with fixed seeds (stated per function in the script); Experiment 4, the
rotation-law figure, the Experiment 2 query counts, and Table 1 are deterministic
(exact arithmetic / closed forms), as in the paper.

## Contents

| File | Reproduces |
|---|---|
| `generate_figures.py` | Experiment 1 -> Figure 1; Experiment 2 -> Figure 2; Experiment 3 -> Figure 3; rotation law (Lemma 3.6) -> Figure 4; Experiment 4 -> Figure 5; Experiment 5 -> Table 1 (exact integer check) |
| `superseded/` | archival verification code from a pre-revision appendix; not needed for the current manuscript |

(The manuscript's figure files are named `fig1_affine_estimator.pdf`,
`fig3_nonlinear_contraction.pdf`, `fig4_A1_nonexpansive.pdf`, `fig2_rotation_law.pdf`,
`fig5_doubling_window.pdf`, matching its `\includegraphics` calls.)

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`

## Usage

```bash
python generate_figures.py
```

This writes the five figure PDFs to the working directory and prints the Table 1 check:

```
[table1] phi=0.005 eps=0.1: N=10961 (table: 10961) OK
...
[table1] all reproduced: True
```

## What is verified, and the expected output

**Figure 1 / Experiment 1 (Theorem 5.3, two-point affine estimator; sigma = 0.1, z = 0.37, D = 1).**
- Success probability at batch m = 400 over 40,000 replicates: **0.6586**
  (manuscript: 0.66, 95% CI [0.655, 0.664]); probit crossing m* ~ 394 +- 30 (manuscript: m ~ 400 +- 30).
- Log-log slopes of the required batch: **2.00** (vs 1/(1-gamma)) and **1.91** (vs 1/eps)
  (manuscript: 1.99 and 1.90; deviations of order 0.01 reflect the RNG/BLAS build and are
  within the +-1 standard-error bands plotted in the figure).

**Figure 2 / Experiment 2 (Theorem 5.5, geometric batching on T(x) = gamma x + (1-gamma)/2 sin x;
sigma = 1, D = 1).** The schedule's total query count is deterministic in (rho, eps):
- (1-rho, eps) = (0.10, 0.1): **3.43e4** queries (manuscript figure label 3.4e4)
- (1-rho, eps) = (0.05, 0.1): **1.45e5** (manuscript 1.4e5; text 1.5e5)
- (1-rho, eps) = (0.05, 0.05): **5.93e5** (manuscript 5.9e5; text 6.0e5)
- Doubling ratios 4.21 and 4.10 (manuscript 4.2 and 4.1); measured bars at 3.4-3.7x the
  reference sigma^2/(eps^2(1-rho)^2) (manuscript: ~3.5-3.8x).

**Figure 3 / Experiment 3 (Theorem 4.3 on the rotation R_0.3; sigma = 1, D = 1, 20 runs).**
- Certified horizons N = 28 / 48 / 88 at eps = 0.2 / 0.1 / 0.05.
- Query counts per run: **81,200 / 940,800 / 12,531,200**
  (manuscript: 8.1e4 / 9.4e5 / 1.3e7).
- Mean terminal residuals: **0.063 / 0.038 / 0.016** (manuscript: 0.06 / 0.04 / 0.02).

**Figure 4 (rotation decay law, Lemma 3.6; deterministic).** For c in {0.25, 0.5} the
measured ||x_n|| at phi = 0.01 follows Gamma(c+1)(n phi)^{-c} up to a constant factor
(log-log slope -c); for c = 1 the trajectory oscillates within 2|sin(n phi/2)|/(n phi)
with dips at n phi in 2 pi Z reaching ~1/(n+1).

**Figure 5 / Experiment 4 (Theorems 3.1, 3.30, 3.25; deterministic hitting times).**
- On T(x) = 0.95x at eps = 0.05: classical Halpern hits at **399**, the doubling-window
  scheme at **142** (first iterate inside a window; window-end 248), the small-anchor
  track at **59** (manuscript: 399 / 142 / 248 / 59; classical error at step 59 is
  e_59 = 0.318).
- Parallel two-track scheme stops at **59** (contraction) and **598** (rotation R_0.01)
  (manuscript: 59 and 598); the small-anchor track on R_0.01 is at distance ~0.996 after
  1e5 steps (envelope of Lemma 3.6(ii)).
- Panel (b) log-log slopes depend on the eps grid; the script prints its fitted values
  (classical 1.00) and notes the paper's 0.39 / 0.33 / 0.05 / 0.11 were fit over its grid.

**Table 1 / Experiment 5 (Theorem 3.15(ii), oscillatory escape; exact arithmetic).**
All twelve hitting times N_phi(eps) on the grid
(phi, eps) in {0.005, 0.01, 0.02} x {0.1, 0.05, 0.02, 0.01} are reproduced **to the exact
integer** (10961, 40033, 179507, 644482 / 5481, 20017, 89754, 322243 / 2742, 10009, 44879,
161428); the ratios to the reference scale phi^{-1} eps^{-1/alpha_0}, alpha_0 = log2(3/2),
all lie in [1.07, 1.23].

## Citation

If you use this code, please cite the manuscript.
