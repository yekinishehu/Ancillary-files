#!/usr/bin/env python3
"""
verify_certificate.py  --  SUPERSEDED (pre-revision version only)
==============================================================
Certified non-degeneracy check for the "Uniform orbit small-ball" lemma,
Step 3, of the PRE-REVISION manuscript (old Appendix A: "The certified
non-degeneracy check").

The submitted version of the paper proves the orbit small-ball
(Lemma 2.12, Section 2.5) by a Rouche perturbation argument that requires
NO computer assistance; the appendix that this script certifies no longer
exists in the manuscript.  This file is kept for the record, and for the
arXiv posting of the earlier version.

Claim certified (old Lemma, Step 3): the three equations
    P_t(omega) = 0,  d_omega P_t(omega) = 0,  d_t P_t(omega) = 0
have no common solution (omega, t), t in [t0,1], with coefficients in the
schedule's ranges; equivalently the truncation fractions supporting an
almost-degenerate zero form a set of measure O(tau).  With d_t P_t = 0
forcing cos(t*omega) = 0 and hence sin(t*omega) = +/-1, the first two
equations reduce to the simultaneous conditions
    |R(omega)| = c0,   |R'(omega)| = 5*c0,
with R(omega) = c_{-1} e^{i theta_{-1}(omega)} sin(omega/2)
              + c_{-2} e^{i theta_{-2}(omega)} sin(omega/4).

Method: for fixed (alpha, omega) the map (g1, g2) -> R is affine, so the
range of |R| and |R'| over the coefficient box [1/2, 2]^2 is obtained
exactly from the four vertices (minimum of the convex modulus attained at
a vertex or at 0; maximum at a vertex).  Lipschitz bounds (one-line
triangle inequalities):
    |R'| <= 6.2 c0,  |R''| <= 16.1 c0,  |d_alpha R| + |d_alpha R'| <= 8.1 c0,
absorb the grid spacings.  Verified outcome (double precision, grid below):
(C1) no (alpha, omega) where |R| and |R'| simultaneously straddle c0, 5 c0;
(C2) wherever |R| can attain c0, the interval for |R'| lies at distance
     >= 0.7323 from 5 c0; certified margin >= 0.71 > 0.
"""

import numpy as np

a0 = np.log2(1.5)
alphas = np.linspace(a0, 0.6, 9)
w = np.linspace(0.004, 8*np.pi, 200001); iw = 1j*w
A25  = np.exp(2.5*iw)*np.sin(w/2);  A125  = np.exp(1.25*iw)*np.sin(w/4)
Ap25 = np.exp(2.5*iw)*(2.5j*np.sin(w/2)+0.5*np.cos(w/2))
Ap125= np.exp(1.25*iw)*(1.25j*np.sin(w/4)+0.25*np.cos(w/4))

def iabs(A, B, u1, u2):
    # exact range of |g1*u1*A + g2*u2*B|, g in [1/2, 2]^2 (parallelogram)
    V = [0.5*u1*A + 0.5*u2*B, 2*u1*A + 0.5*u2*B,
         0.5*u1*A + 2*u2*B,     2*u1*A + 2*u2*B]
    m = [np.abs(v) for v in V]
    hi = np.maximum(np.maximum(m[0], m[1]), np.maximum(m[2], m[3]))
    emin = np.full(A.shape, np.inf)   # exact min of the convex modulus per edge
    for i in range(4):
        p, q = V[i], V[(i+1) % 4]; d = q - p; dd = np.abs(d)**2
        t = np.clip(-np.real(p*np.conj(d))/np.where(dd == 0, 1, dd), 0, 1)
        emin = np.minimum(emin, np.abs(p + t*d))
    cr = lambda x, y: np.imag(np.conj(x)*y)
    s1, s2 = np.sign(cr(p0 := V[0], p1 := V[1])), np.sign(cr(p0, p2 := V[2]))
    inside = ((np.sign(cr(p1, p3 := V[3])) == s1) & (np.sign(cr(p1, p2)) == s2)
              & (np.sign(cr(p2, p3)) == s1) & (np.sign(cr(p0, p3)) == s2))
    return np.where(inside, 0.0, emin), hi

margin = np.inf; flagged = 0
for al in alphas:
    u1, u2 = 2**(al-1), 2**(2*(al-1))
    loR, hiR = iabs(A25, A125, u1, u2)
    loP, hiP = iabs(Ap25, Ap125, u1, u2)
    has1 = (loR <= 1) & (hiR >= 1)
    flagged += int(np.count_nonzero(has1 & (loP <= 5) & (hiP >= 5)))
    margin = min(margin, np.min(np.maximum(5 - hiP, loP - 5)))

# slack: between adjacent grid points |R|, |R'| move by at most
# (6.2 + 16.1)c0*dw + 8.1c0*da <= 0.018 c0
dw = 8*np.pi/200000; da = (0.6 - a0)/8
slack = (6.2 + 16.1)*dw + 8.1*da
print("flagged:", flagged, " (C1 expects 0)")
print("margin before slack:", margin, " (C2 expects >= 0.7323)")
print("grid slack:", slack)
print("certified uniform margin:", margin - slack, " (expects >= 0.71 > 0)")
assert flagged == 0 and margin - slack > 0.7
print("CERTIFICATE OK (superseded; see superseded/NOTE.md)")
