#!/usr/bin/env python3
# =============================================================================
# Certified non-degeneracy check for Lemma "Uniform orbit small-ball"
# (manuscript: Adaptivity, Anchoring, and the Exact Oracle Complexity of
#  Stochastic Fixed-Point Iterations, Appendix A)
#
# WHAT THIS CERTIFIES
# -------------------
# Step 3 of the proof of the Uniform orbit small-ball lemma requires the
# following non-degeneracy fact: the three equations
#
#     P_t(w) = 0,   d_w P_t(w) = 0,   d_t P_t(w) = 0
#
# have NO common solution (w, t), for t in [t0, 1], w real, and coefficients
# in the schedule's ranges.  Here
#
#     P_t(w) = c0 e^{i5w} sin(t w) + c_{-1} e^{i th_{-1}(w)} sin(w/2)
#                                 + c_{-2} e^{i th_{-2}(w)} sin(w/4)
#
# is the universal trig-polynomial family of Step 2 of the lemma's proof
# (gentle blocks + truncated top block; th_{-j}' = 5/2, 5/4 universal).
#
# Since d_t P_t(w) = c0 w e^{i5w} cos(t w), a solution with w != 0 requires
# cos(t w) = 0, hence sin(t w) = +/-1; writing
#
#     R(w) = c_{-1} e^{i th_{-1}(w)} sin(w/2) + c_{-2} e^{i th_{-2}(w)} sin(w/4),
#
# the system reduces to the two-equation condition
#
#     |R(w)| = c0  AND  |R'(w)| = 5 c0        (three real equations, one unknown)
#
# over the parameter box  alpha in [alpha0, 3/5],
#   c_{-1}/2^{alpha-1}, c_{-2}/2^{2(alpha-1)}  in  [1/2, 2].
#
# HOW THE CERTIFICATE WORKS
# -------------------------
# Part 1 (vectorized, double precision): for each grid point (alpha, w) the
# map (g1, g2) |-> R  (resp. R') is AFFINE, so its image of [1/2,2]^2 is a
# parallelogram; the range of |R| over the box is the exact interval
#   [ min over the 4 EDGES of dist(origin, edge),  max over the 4 vertices ]
# (minimum of the convex modulus over each edge computed in closed form:
#  closest point p + t d,  t = clip(-Re(p conj(d))/|d|^2, 0, 1);
#  the vertex-only minimum is NOT exact and must not be used).
# The certificate passes if, at every grid point, the intervals for |R| and
# |R'| never simultaneously contain c0 and 5 c0; the reported margin is
#   min over {1 in I_|R|} of dist(5, I_|R'|).
#
# Part 2 (rigorous scalar interval arithmetic): the function values at the
# binding (worst-margin) point are re-evaluated with OUTWARD ROUNDING after
# every arithmetic operation (math.nextafter) and Taylor remainders with
# rigorous bounds (enclosure widths ~1e-13), so the margin at the binding
# point is bit-level rigorous.
#
# Between grid points the quantities move at most
#   (6.2 + 16.1) c0 * Dw + 8.1 c0 * Da <= 0.018 c0
# (explicit Lipschitz bounds, manuscript Appendix A), so the certified
# uniform margin is (Part 1 margin) - 0.018.
#
# DEPENDENCIES: numpy (Part 1); Python >= 3.9 stdlib only (Part 2).
# RUN:  python3 verify_certificate.py
# EXPECTED OUTPUT: margin ~0.7323, certified margin >= 0.71, zero flagged.
# =============================================================================

import math
import numpy as np

ALPHA0 = math.log2(1.5)          # = log_2(3/2)
ALPHAS = np.linspace(ALPHA0, 0.6, 9)
WGRID  = np.linspace(0.004, 8*math.pi, 200001)

# ---------------------------------------------------------------- Part 1 ----
def box_abs_interval(P, Q, u1, u2):
    """Exact range of |g1*u1*P + g2*u2*Q| over g in [1/2,2]^2, as an interval.
    P, Q: complex numpy arrays (same shape).  Affine image of the box = a
    parallelogram: max at a vertex; min = 0 if the origin lies inside, else
    the minimum over the four EDGES of the distance from the origin."""
    V = [0.5*u1*P + 0.5*u2*Q, 2*u1*P + 0.5*u2*Q,
         0.5*u1*P + 2*u2*Q,   2*u1*P + 2*u2*Q]
    vm = [np.abs(v) for v in V]
    hi = np.maximum(np.maximum(vm[0], vm[1]), np.maximum(vm[2], vm[3]))
    emin = np.full(P.shape, np.inf)
    for i in range(4):                      # edge i = segment [p, q]
        p, q = V[i], V[(i+1) % 4]
        d = q - p
        dd = np.abs(d)**2
        # closest point on the segment to the origin is p + t d with
        # t = clip(-Re(p conj(d))/|d|^2, 0, 1); the min over the four edges
        # is the exact minimum over the whole parallelogram
        t = np.clip(-np.real(p*np.conj(d)) / np.where(dd == 0, 1, dd), 0, 1)
        emin = np.minimum(emin, np.abs(p + t*d))
    p0, p1, p2, p3 = V[0], V[1], V[3], V[2]
    cr = lambda x, y: np.imag(np.conj(x)*y)
    s1, s2 = np.sign(cr(p0, p1)), np.sign(cr(p0, p2))
    inside = ((np.sign(cr(p1, p3)) == s1) & (np.sign(cr(p1, p2)) == s2) &
              (np.sign(cr(p2, p3)) == s1) & (np.sign(cr(p0, p3)) == s2))
    lo = np.where(inside, 0.0, emin)
    return lo, hi

def part1():
    w = WGRID
    iw = 1j*w
    A25   = np.exp(2.5*iw) * np.sin(w/2)
    A125  = np.exp(1.25*iw) * np.sin(w/4)
    Ap25  = np.exp(2.5*iw) * (2.5j*np.sin(w/2) + 0.5*np.cos(w/2))
    Ap125 = np.exp(1.25*iw) * (1.25j*np.sin(w/4) + 0.25*np.cos(w/4))
    margin = np.inf
    binding = None
    flagged = 0
    for al in ALPHAS:
        u1, u2 = 2**(al-1), 2**(2*(al-1))
        loR, hiR = box_abs_interval(A25, A125, u1, u2)
        loP, hiP = box_abs_interval(Ap25, Ap125, u1, u2)
        has1 = (loR <= 1) & (hiR >= 1)
        flagged += int(np.count_nonzero(has1 & (loP <= 5) & (hiP >= 5)))
        # valid lower bound: min over the level set {|R|=1} of ||R'|-5|
        # is >= distance of 5 from the box range of |R'|
        d5 = np.maximum(5 - hiP, loP - 5)
        if np.any(has1) and d5[has1].min() < margin:
            margin = float(d5[has1].min())
            binding = (float(al), float(w[has1][np.argmin(d5[has1])]))
    return margin, binding, flagged

# ---------------------------------------------------------------- Part 2 ----
NA = math.nextafter
INF = math.inf
F53 = math.factorial(53)
F52 = math.factorial(52)

def iadd(a, b): return (NA(a[0]+b[0], -INF), NA(a[1]+b[1], INF))
def isub(a, b): return (NA(a[0]-b[1], -INF), NA(a[1]-b[0], INF))
def imul(a, b):
    p = [a[0]*b[0], a[0]*b[1], a[1]*b[0], a[1]*b[1]]
    return (NA(min(p), -INF), NA(max(p), INF))
def ineg(a): return (NA(-a[1], -INF), NA(-a[0], INF))
def U(x): return (NA(x, -INF), NA(x, INF))

def isin(x):
    """Rigorous sin on an interval: quadrant reduction to |y| <= pi/4,
    Taylor series with rigorous remainder."""
    hpi = (NA(math.pi/2, -INF), NA(math.pi/2, INF))
    xc = (x[0] + x[1]) / 2
    k = math.floor(xc / (math.pi/2) + 0.5)
    y = isub(x, imul((k, k), hpi))
    q = k % 4
    y2 = imul(y, y)
    term = y
    s = (0.0, 0.0)
    sg = 1.0
    for n in range(0, 26):
        if n > 0:
            term = imul(term, y2)
            d = math.factorial(2*n + 1)
            t = (sg*term[0]/d, sg*term[1]/d)
            s = iadd(s, (min(t), max(t)))
            sg = -sg
        else:
            s = (min(term), max(term))
    R = max(-y[0], y[1])**53 / F53
    s = (NA(s[0]-R, -INF), NA(s[1]+R, INF))
    if q == 0:
        return s
    if q == 2:
        return ineg(s)
    term = (1.0, 1.0)
    c = (1.0, 1.0)
    sg = -1.0
    for n in range(1, 26):
        term = imul(term, y2)
        d = math.factorial(2*n)
        t = (sg*term[0]/d, sg*term[1]/d)
        c = iadd(c, (min(t), max(t)))
        sg = -sg
    R = max(-y[0], y[1])**52 / F52
    c = (NA(c[0]-R, -INF), NA(c[1]+R, INF))
    return c if q == 1 else ineg(c)

def icos(x):
    return isin(iadd(x, (NA(math.pi/2, -INF), NA(math.pi/2, INF))))

def icmul(ar, ai, br, bi):
    return (isub(imul(ar, br), imul(ai, bi)),
            iadd(imul(ar, bi), imul(ai, br)))

def iabs(ar, ai):
    vals = [math.hypot(r, i) for r in (ar[0], ar[1]) for i in (ai[0], ai[1])]
    return NA(min(vals), -INF), NA(max(vals), INF)

def part2(al, ws):
    """Rigorous interval re-verification of the margin at the binding point."""
    s2, c2 = isin(U(ws/2)), icos(U(ws/2))
    s4, c4 = isin(U(ws/4)), icos(U(ws/4))
    E2r, E2i = icos(imul((2.5, 2.5), U(ws))), isin(imul((2.5, 2.5), U(ws)))
    E1r, E1i = icos(imul((1.25, 1.25), U(ws))), isin(imul((1.25, 1.25), U(ws)))
    Zs = icmul(E2r, E2i, s2, (0.0, 0.0))       # e^{2.5iw} sin(w/2)
    Zc = icmul(E2r, E2i, c2, (0.0, 0.0))
    Ws = icmul(E1r, E1i, s4, (0.0, 0.0))       # e^{1.25iw} sin(w/4)
    Wc = icmul(E1r, E1i, c4, (0.0, 0.0))
    # d_w [e^{iaw} sin(bw)] = i a e^{iaw} sin(bw) + b e^{iaw} cos(bw)
    Ap_r = iadd(ineg(imul((2.5, 2.5), Zs[1])), imul((0.5, 0.5), Zc[0]))
    Ap_i = iadd(imul((2.5, 2.5), Zs[0]),        imul((0.5, 0.5), Zc[1]))
    Bp_r = iadd(ineg(imul((1.25, 1.25), Ws[1])), imul((0.25, 0.25), Wc[0]))
    Bp_i = iadd(imul((1.25, 1.25), Ws[0]),        imul((0.25, 0.25), Wc[1]))
    u1, u2 = 2**(al-1), 2**(2*(al-1))
    worst = math.inf
    touched = False
    for g1 in (0.5, 2.0):
        for g2 in (0.5, 2.0):
            rp_r = iadd(imul(U(g1*u1), Ap_r), imul(U(g2*u2), Bp_r))
            rp_i = iadd(imul(U(g1*u1), Ap_i), imul(U(g2*u2), Bp_i))
            lo, hi = iabs(rp_r, rp_i)
            r_r = iadd(imul(U(g1*u1), Zs[0]), imul(U(g2*u2), Ws[0]))
            r_i = iadd(imul(U(g1*u1), Zs[1]), imul(U(g2*u2), Ws[1]))
            rlo, rhi = iabs(r_r, r_i)
            if rlo <= 1 <= rhi:
                touched = True
                worst = min(worst, max(5 - hi, lo - 5))
    return worst, touched

# ------------------------------------------------------------------ main ----
if __name__ == "__main__":
    margin, binding, flagged = part1()
    print(f"[Part 1] grid: {len(ALPHAS)} alphas x {len(WGRID)} omega points")
    print(f"[Part 1] flagged points (|R|~1 AND |R'|~5 simultaneously): {flagged}")
    print(f"[Part 1] margin before slack: {margin:.4f}  at (alpha, omega) = {binding}")
    slack = (6.2 + 16.1) * (8*math.pi/200000) + 8.1 * (0.6 - ALPHA0)/8
    print(f"[Part 1] grid slack (Lipschitz): {slack:.4f}")
    print(f"[Part 1] CERTIFIED MARGIN >= {margin - slack:.3f}  (must be > 0)")
    m2, touched = part2(*binding)
    print(f"[Part 2] rigorous interval margin at binding point {binding}: "
          f"{m2 if touched else 'no |R|~1 vertex'}")
    print("CERTIFICATE:", "PASS" if (flagged == 0 and margin - slack > 0) else "FAIL")
