# Ancillary files — Adaptivity, Anchoring, and the Exact Oracle Complexity of Stochastic Fixed-Point Iterations

This repository contains the certified computational verification accompanying
the manuscript *Adaptivity, Anchoring, and the Exact Oracle Complexity of
Stochastic Fixed-Point Iterations* (Y. Shehu), specifically **Appendix A: The
certified non-degeneracy check** used in Step 3 of the proof of the *Uniform
orbit small-ball* lemma (Section 3.5, metric lacunarity law).

## Files

| File | Description |
|---|---|
| `verify_certificate.py` | Standalone verification script (Python ≥ 3.9; numpy for Part 1). Part 1 runs the grid scan with exact edge-wise parallelogram interval bounds and reports the certified margin (≥ 0.71) and the binding point; Part 2 re-verifies the binding point by rigorous interval arithmetic with outward rounding and Taylor remainders. Run: `python3 verify_certificate.py` — expected output: `CERTIFICATE: PASS`. |
| `README.txt` | Detailed description of what the script certifies and how. |
| `anc_lacunarity_certificate.tar.gz` | The same files, packaged for upload as an arXiv ancillary file. |

## What is certified

The three equations `P_t = ∂_ω P_t = ∂_t P_t = 0` defining a persistent
double zero of the universal trig-polynomial family `P_t` have **no common
solution** over the schedule's parameter box; equivalently, the two-equation
condition `|R(ω)| = c₀`, `|R'(ω)| = 5c₀` has no solution. Certified uniform
margin: **≥ 0.71** (grid margin ≈ 0.7323 minus the explicit Lipschitz slack
0.018). See Appendix A of the manuscript for the mathematical context and the
exact interval-arithmetic construction.

## Citation

If you use this verification, please cite the manuscript (arXiv link to be
added upon posting).
