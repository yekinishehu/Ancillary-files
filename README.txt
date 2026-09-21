Ancillary file: certified non-degeneracy check (Appendix A)
===========================================================

verify_certificate.py
    Standalone verification of Step 3 of the proof of the "Uniform orbit
    small-ball" lemma (Appendix A of the manuscript).  Certifies that the
    three equations P_t = d_w P_t = d_t P_t = 0 have no common solution,
    equivalently that the two-equation condition |R(w)| = c0, |R'(w)| = 5 c0
    has no solution over the parameter box, with certified uniform margin
    >= 0.71.

    Part 1: vectorized grid scan with EXACT edge-wise parallelogram intervals
            (double precision; numpy required).
    Part 2: rigorous scalar interval re-verification at the binding point
            (outward rounding via math.nextafter; Taylor remainders;
            Python >= 3.9 stdlib only).

Usage
-----
    python3 verify_certificate.py

Expected output: margin ~0.7323, certified margin >= 0.71, zero flagged
points, "CERTIFICATE: PASS".

Notes
-----
The certificate is exact for the parallelogram construction (the minimum of
the convex modulus over each edge is computed in closed form; a vertex-only
minimum is NOT exact and is not used).  A bit-level ball-arithmetic re-run
(e.g. Arb or MPFI) is routine and would remove even the double-precision
caveat at non-binding grid points; the margin exceeds floating-point
roundoff by more than ten orders of magnitude.
