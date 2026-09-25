#!/usr/bin/env python3
"""generate_figures.py -- reproduces Figures 1 and 4 of
"Adaptivity, Anchoring, and the Exact Oracle Complexity of Stochastic Fixed-Point Iterations"
(Experiment 1: affine two-point estimator, Theorem 5.2; Experiment 3: Theorem 4.2).

Figure 2 (rotation law), Figure 3 (geometric batching), Figure 5 (adaptive escape)
and Table 1 (oscillatory escape) are deterministic exact-arithmetic simulations;
Table 1 is reproduced exactly by the check at the bottom of this script.
Run:  python generate_figures.py     (writes fig1_affine_estimator.pdf, fig4_A1_nonexpansive.pdf)
"""
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIG, Z, D = 0.1, 0.37, 1.0   # Experiment 1 configuration

def batch_estimator(m, g, ntrials, gen):
    """two-point estimator of Theorem 5.2; returns signed errors (ntrials,)"""
    h = m // 2
    z0 = (1-g)*Z + SIG*gen.standard_normal((ntrials, h)).mean(axis=1)
    zD = g*D + (1-g)*Z + SIG*gen.standard_normal((ntrials, h)).mean(axis=1)
    return z0/(1-(zD-z0)/D) - Z

def probit_crossing(g, eps, guess, gen, ntrials=8000):
    """smallest m with empirical success probability >= 2/3, probit fit on a geometric grid"""
    ms = np.unique(np.round(guess*np.array([0.45, 0.7, 1.4, 2.2])).astype(int))
    ps = np.array([np.mean(np.abs(batch_estimator(m, g, ntrials, gen)) <= eps) for m in ms])
    w = norm.ppf(np.clip(ps, 1e-4, 1-1e-4))
    A = np.vstack([np.ones(len(ms)), np.log(ms)]).T
    c, res, *_ = np.linalg.lstsq(A, w, rcond=None)
    sig_w = np.sqrt(res[0]/(len(ms)-2)) if len(res) and res[0] > 0 else 0.05
    return float(np.exp((norm.ppf(2/3)-c[0])/c[1])), sig_w/abs(c[1])

def fig1():
    rng = np.random.default_rng(20260925)
    # panel (a): mean |error| +- std vs m at (gamma, eps) = (0.95, 0.1)
    grid = np.logspace(1.7, 4.3, 18)
    mean_abs, std_abs = [], []
    for m in grid.astype(int):
        e = batch_estimator(m, 0.95, 3000, rng)
        mean_abs.append(np.abs(e).mean()); std_abs.append(np.abs(e).std())
    mean_abs, std_abs = np.array(mean_abs), np.array(std_abs)
    # panels (b),(c): required batch vs 1/(1-gamma) and vs eps
    gm_star, gm_err, xs_b = [], [], []
    for gm in [0.80, 0.90, 0.95, 0.975]:
        m_, dm = probit_crossing(gm, 0.1, 399*(0.05/(1-gm))**2, rng)
        gm_star.append(m_); gm_err.append(dm); xs_b.append(1/(1-gm))
    ep_star, ep_err, xs_c = [], [], []
    for ep in [0.05, 0.10, 0.20]:
        m_, dm = probit_crossing(0.95, ep, 399*(0.1/ep)**2, rng)
        ep_star.append(m_); ep_err.append(dm); xs_c.append(ep)
    slope_b = np.polyfit(np.log(xs_b), np.log(gm_star), 1)[0]
    slope_c = np.polyfit(np.log(xs_c), np.log(ep_star), 1)[0]
    p400 = np.mean(np.abs(batch_estimator(400, 0.95, 40000, rng)) <= 0.1)
    print(f"[fig1] P(success) at m=400 (40000 reps): {p400:.4f}; slopes {slope_b:.2f} / {slope_c:.2f}; "
          f"m* (b): {np.round(gm_star,0)}; m* (c): {np.round(ep_star,0)}")

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13.5, 3.8), layout="constrained")
    a1.loglog(grid, mean_abs, "o-", ms=3, lw=1, color="C0", label="mean |error|")
    a1.fill_between(grid, np.maximum(mean_abs-std_abs, 1e-6), mean_abs+std_abs,
                    color="C0", alpha=0.3, label="+-1 std band")
    a1.axhline(0.1, color="k", ls="--", lw=1, label="target eps = 0.1")
    a1.axvline(400, color="r", ls=":", lw=1.2, label="measured required batch (P >= 2/3)")
    a1.axvline(192000, color="gray", ls="-.", lw=1, label="theoretical threshold 192000")
    a1.set_xlabel("batch size m"); a1.set_ylabel("estimation error |z_hat - z|")
    a1.legend(fontsize=6.5, loc="lower left"); a1.set_title("(a) (gamma, eps) = (0.95, 0.1)", fontsize=10)

    a2.loglog(xs_b, gm_star, "o", color="C0", label="measured")
    a2.errorbar(xs_b, gm_star, yerr=np.array(gm_star)*np.array(gm_err), fmt="none", ecolor="C0", capsize=3)
    xx = np.linspace(4.5, 42, 50)
    a2.loglog(xx, 480*SIG**2/((1/xx)**2*0.1**2), "--", color="k", lw=1, label="theory 480 sigma^2/(eps^2 (1-gamma)^2)")
    a2.loglog(xx, 0.99*SIG**2/((1/xx)**2*0.1**2), "-", color="C0", lw=0.8, alpha=0.6, label=f"fit (slope {slope_b:.2f})")
    for x_, y_ in zip(xs_b, gm_star): a2.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(4, 4), fontsize=7)
    a2.set_xlabel("1/(1-gamma)"); a2.set_ylabel("required batch m")
    a2.legend(fontsize=6.5, loc="upper left"); a2.set_title("(b) eps = 0.1", fontsize=10)

    a3.loglog(xs_c, ep_star, "o", color="C0", label="measured")
    a3.errorbar(xs_c, ep_star, yerr=np.array(ep_star)*np.array(ep_err), fmt="none", ecolor="C0", capsize=3)
    xe = np.linspace(0.045, 0.22, 50)
    a3.loglog(xe, 480*SIG**2/(0.05**2*xe**2), "--", color="k", lw=1, label="theory 480 sigma^2/(eps^2 (1-gamma)^2)")
    a3.loglog(xe, 0.99*SIG**2/(0.05**2*xe**2), "-", color="C0", lw=0.8, alpha=0.6, label=f"fit (slope {slope_c:.2f})")
    for x_, y_ in zip(xs_c, ep_star): a3.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(4, 4), fontsize=7)
    a3.set_xlabel("eps"); a3.set_ylabel("required batch m")
    a3.legend(fontsize=6.5, loc="upper right"); a3.set_title("(c) gamma = 0.95", fontsize=10)
    fig.savefig("fig1_affine_estimator.pdf")
    print("[fig1] wrote fig1_affine_estimator.pdf")

def fig4():
    rng = np.random.default_rng(2026)
    phi, D, sig, nruns = 0.3, 1.0, 1.0, 20
    c, s = np.cos(phi), np.sin(phi)
    def run(eps):
        N = int(np.ceil(4*D/eps)) + 8
        Q = sum(int(np.ceil(8*sig**2*k/eps**2)) for k in range(1, N+1))
        x = np.tile(np.array([D, 0.0]), (nruns, 1))
        hist = np.empty((N+1, nruns)); hist[0] = np.linalg.norm(x, axis=1)
        for n in range(N):
            th = 1.0/(n+2); m = int(np.ceil(8*sig**2*max(n, 1)/eps**2))
            xi = sig*rng.standard_normal((nruns, 2))/np.sqrt(m)
            Rx = np.stack([c*x[:, 0]-s*x[:, 1], s*x[:, 0]+c*x[:, 1]], axis=1)
            x = th*np.array([D, 0.0]) + (1-th)*(Rx + xi)
            hist[n+1] = 2*np.sin(phi/2)*np.linalg.norm(x, axis=1)
        return N, Q, hist
    res, hist01 = {}, None
    for eps in [0.2, 0.1, 0.05]:
        N, Q, hist = run(eps)
        res[eps] = (N, Q, hist[-1].mean())
        print(f"[fig4] eps={eps}: N={N}, queries={Q}, mean residual={hist[-1].mean():.4f}")
        if eps == 0.1: hist01 = hist
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ns = np.arange(hist01.shape[0])
    a1.plot(ns, hist01.mean(axis=1), color="C0", lw=1.5, label="mean residual")
    a1.fill_between(ns, hist01.mean(1)-hist01.std(1), hist01.mean(1)+hist01.std(1), color="C0", alpha=0.3, label="+-1 std")
    a1.axhline(0.1, color="k", ls="--", lw=1, label="target eps = 0.1")
    a1.axvline(48, color="r", ls=":", lw=1, label="N = ceil(4D/eps)+8 = 48")
    a1.set_xlabel("iteration n"); a1.set_ylabel("||x_n - T x_n||"); a1.set_yscale("log")
    a1.legend(fontsize=8); a1.set_title("(a) Theorem 4.2 on R_0.3, eps=0.1", fontsize=10)
    es = np.array([0.2, 0.1, 0.05]); Qs = np.array([res[e][1] for e in es])
    a2.loglog(es, Qs, "o-", color="C0", label="measured queries per run")
    eref = np.linspace(0.04, 0.25, 50)
    a2.loglog(eref, 64*sig**2*D**2*eref**-4, "--", color="k", lw=1, label="reference 64 sigma^2 D^2 eps^-4")
    for e, q in zip(es, Qs):
        a2.annotate(f"{q/(64*sig**2*D**2*e**-4):.2f}", (e, q), textcoords="offset points", xytext=(6, 4), fontsize=8)
    a2.set_xlabel("eps"); a2.set_ylabel("oracle queries per run")
    a2.legend(fontsize=8); a2.set_title("(b) query scaling vs eps^-4", fontsize=10)
    fig.tight_layout(); fig.savefig("fig4_A1_nonexpansive.pdf")
    print("[fig4] wrote fig4_A1_nonexpansive.pdf")

def table1_check():
    """exact-arithmetic reproduction of Table 1 (oscillatory escape, Theorem 3.8(ii))"""
    alpha0 = np.log2(1.5)
    def theta(n, cp):
        if n >= 2:
            j = int(np.floor(np.log2(n)))
            return 1.0/(n+2) if n < 3*2**(j-1) else cp/(n+2)
        return 1.0/(n+2)
    expected = {(0.005,0.1):10961,(0.005,0.05):40033,(0.005,0.02):179507,(0.005,0.01):644482,
                (0.01,0.1):5481,(0.01,0.05):20017,(0.01,0.02):89754,(0.01,0.01):322243,
                (0.02,0.1):2742,(0.02,0.05):10009,(0.02,0.02):44879,(0.02,0.01):161428}
    ok = True
    for (phi, eps), tab in expected.items():
        cp = eps/96.0
        x = np.array([1.0, 0.0]); c, s = np.cos(phi), np.sin(phi)
        N = int(4*(1/phi)*eps**(-1/alpha0)); hit = None
        for n in range(N+1):
            if np.linalg.norm(x) <= eps: hit = n; break
            th = theta(n, cp)
            x = th*np.array([1.0, 0.0]) + (1-th)*np.array([c*x[0]-s*x[1], s*x[0]+c*x[1]])
        ok &= (hit == tab)
        print(f"[table1] phi={phi} eps={eps}: N={hit} (table: {tab}) {'OK' if hit==tab else 'MISMATCH'}")
    print("[table1] all reproduced:", ok)

if __name__ == "__main__":
    fig1(); fig4(); table1_check()
