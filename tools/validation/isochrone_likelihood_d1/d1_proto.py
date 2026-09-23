"""D1: binned per-evaluation deposit (1) vs unbinned per-star (2, 2') on identical EEP interpolation.

Pre-registered in ~/phd/agent-findings/isochrone-nuts-convergence-2026-09.md §9.3.
"""

import json
import sys
import time

import numpy as np

HERE = __import__("pathlib").Path(__file__).resolve().parent
from scipy.optimize import minimize
from scipy.special import erf

sys.path.insert(0, "/Users/notluquis/erotica/tools/validation")
from isochrone_nuts_convergence import MIST, synthetic_cluster  # noqa: E402

from erotica.analysis._isochrone import (  # noqa: E402
    _ccm89,
    _chabrier2014_weights,
    _dk_mean_q,
    _fit_error_model,
    _mag_combine,
)

KG, KBP, KRP = (_ccm89(lam, 3.1) for lam in (6390.7, 5182.6, 7825.1))
KC = KBP - KRP


# ------------------------------------------------------------------ EEP-aware reader
def read_nodes(age_lo=5.9, age_hi=7.1):
    nodes = {}
    for f in sorted(MIST.glob("*.iso.cmd")):
        Z = None
        with open(f) as fh:
            lines = [next(fh) for _ in range(12)]
        for i, ln in enumerate(lines):
            if "Zinit" in ln:
                toks = ln.lstrip("#").split()
                Z = float(lines[i + 1].lstrip("#").split()[toks.index("Zinit")])
        Z = round(Z, 6)
        d = np.loadtxt(f, comments="#", usecols=(0, 1, 2, 30, 31, 32))
        for a in np.unique(np.round(d[:, 1], 4)):
            if not (age_lo <= a <= age_hi):
                continue
            m = np.round(d[:, 1], 4) == a
            eep, mass, G, BP, RP = d[m, 0], d[m, 2], d[m, 3], d[m, 4], d[m, 5]
            o = np.argsort(eep)
            nodes[(Z, float(a))] = (eep[o].astype(int), mass[o], G[o], BP[o] - RP[o], BP[o], RP[o])
    return nodes


NODES = read_nodes()
ZS = np.array(sorted({k[0] for k in NODES}))
AGES = np.array(sorted({k[1] for k in NODES}))


def node_binary(key):
    eep, mass, G, col, BP, RP = NODES[key]
    q = _dk_mean_q(mass)
    M2 = q * mass
    G2, BP2, RP2 = (np.interp(M2, mass, a, left=a[0], right=a[-1]) for a in (G, BP, RP))
    Gb = _mag_combine(G, G2)
    cb = _mag_combine(BP, BP2) - _mag_combine(RP, RP2)
    return Gb, cb


_cache = {}


def node_on(key, e_lo, e_hi):
    """Node arrays on a common EEP axis, ends clamped (mass step -> 0 there)."""
    ck = (key, e_lo, e_hi)
    if ck not in _cache:
        eep, mass, G, col, _, _ = NODES[key]
        Gb, cb = node_binary(key)
        ax = np.arange(e_lo, e_hi + 1)
        _cache[ck] = np.stack([np.interp(ax, eep, a) for a in (mass, G, col, Gb, cb)])
    return _cache[ck]


def bracket(nodes, x):
    k = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2))
    return k, float((x - nodes[k]) / (nodes[k + 1] - nodes[k]))


def isochrone(met, loga, zs=ZS, ages=AGES, binaries=True):
    lz = np.log10(zs)
    i, wz = bracket(lz, np.log10(met))
    j, wa = bracket(ages, loga)
    keys = [(zs[i], ages[j]), (zs[i + 1], ages[j]), (zs[i], ages[j + 1]), (zs[i + 1], ages[j + 1])]
    e_lo = min(NODES[k][0][0] for k in keys)
    e_hi = max(NODES[k][0][-1] for k in keys)
    A = [node_on(k, e_lo, e_hi) for k in keys]
    X = (1 - wz) * (1 - wa) * A[0] + wz * (1 - wa) * A[1] + (1 - wz) * wa * A[2] + wz * wa * A[3]
    mass, G, col, Gb, cb = X
    w = _chabrier2014_weights(mass)
    bp = np.clip(0.09 + 0.94 / (1 + 1.4 / mass), 0, 1) * float(binaries)
    return mass, w, bp, G, col, Gb, cb


# ------------------------------------------------------------------ data summary
class Data:
    def __init__(self, t, nbright=2):
        from astropy.stats import knuth_bin_width

        self.G = np.asarray(t["Gmag"], float)
        self.c = np.asarray(t["G_BPmag"], float) - np.asarray(t["G_RPmag"], float)
        eG = np.asarray(t["e_Gmag"], float)
        eC = np.hypot(np.asarray(t["e_G_BPmag"], float), np.asarray(t["e_G_RPmag"], float))
        self.eG, self.eC = eG, eC
        self.fe_m, self.fe_c = _fit_error_model(self.G, eG, eC)
        _, bm = knuth_bin_width(self.G, return_bins=True)
        _, bc = knuth_bin_width(self.c, return_bins=True)
        nm, nc = max(len(bm) - 1, 5), max(len(bc) - 1, 5)
        g0, g1 = self.G.min(), self.G.max() * (1 + 1e-12)
        c0, c1 = self.c.min(), self.c.max() * (1 + 1e-12)
        self.bw_m, self.bw_c = (g1 - g0) / nm, (c1 - c0) / nc
        self.em = g0 + self.bw_m * np.arange(-nbright, nm + 1)
        self.ec = c0 + self.bw_c * np.arange(-1, nc + 2)
        self.H, _, _ = np.histogram2d(self.G, self.c, bins=[self.em, self.ec])
        self.Glim = g1

    def sig(self, Gapp):
        Gc = np.clip(Gapp, self.G.min(), self.G.max())
        return self.fe_m(Gc), self.fe_c(Gc)


def Phi(x):
    return 0.5 * (1 + erf(x / np.sqrt(2)))


def loglike_binned(D, th, zs=ZS, ages=AGES, bg=0.01, binaries=True):
    met, loga, dm, Av = th
    mass, w, bp, G, col, Gb, cb = isochrone(met, loga, zs, ages, binaries)
    lam = 0
    for GG, cc, ww in ((G, col, w * (1 - bp)), (Gb, cb, w * bp)):
        Ga, ca = GG + dm + KG * Av, cc + KC * Av
        em, ec = D.sig(Ga)
        sm = np.sqrt(em**2 + 0.5 * D.bw_m**2)
        sc = np.sqrt(ec**2 + 0.5 * D.bw_c**2)
        Pm = np.diff(Phi((D.em[:, None] - Ga[None]) / sm[None]), axis=0)
        Pc = np.diff(Phi((D.ec[:, None] - ca[None]) / sc[None]), axis=0)
        lam = lam + (Pm * ww) @ Pc.T
    lam = np.maximum(lam, 1e-12)
    n = D.H
    s = n.sum() / lam.sum()
    for _ in range(20):  # profile s with bg fixed
        mu = s * lam + bg
        s *= np.sum(n * lam / mu) / lam.sum()
    mu = s * lam + bg
    return float(np.sum(n * np.log(mu) - mu))


def _seg_density(xg, xc, sg, sc, Ag, Ac, Bg, Bc):
    """Mean over t in [0,1] of N2(x; A + t(B-A), diag(sg^2, sc^2)), exact (erf)."""
    ag, ac = (Ag[None] - xg[:, None]) / sg[:, None], (Ac[None] - xc[:, None]) / sc[:, None]
    dg, dc = (Bg - Ag)[None] / sg[:, None], (Bc - Ac)[None] / sc[:, None]
    L2 = dg**2 + dc**2
    L = np.sqrt(np.maximum(L2, 1e-24))
    t0 = -(ag * dg + ac * dc) / np.maximum(L2, 1e-24)  # closest approach parameter
    perp2 = np.maximum((ag + t0 * dg) ** 2 + (ac + t0 * dc) ** 2, 0)
    seg = np.sqrt(2 * np.pi) / L * (Phi(L * (1 - t0)) - Phi(-L * t0))
    point = np.exp(-0.5 * (ag**2 + ac**2))
    val = np.where(L > 1e-6, np.exp(-0.5 * perp2) * seg, point)
    return val / (2 * np.pi * sg[:, None] * sc[:, None])


def loglike_unbinned(D, th, jitter=0.0, zs=ZS, ages=AGES, binaries=True):
    met, loga, dm, Av = th
    mass, w, bp, G, col, Gb, cb = isochrone(met, loga, zs, ages, binaries)
    sG = np.sqrt(D.eG**2 + jitter**2)
    sC = np.sqrt(D.eC**2 + jitter**2)
    f = 0
    F = 0
    for GG, cc, bw in ((G, col, 1 - bp), (Gb, cb, bp)):
        Ga, ca = GG + dm + KG * Av, cc + KC * Av
        # segment k joins points k, k+1; its IMF weight is the IMF mass in [m_k, m_k+1]
        wm = _chabrier2014_weights(mass)  # xi(m)|grad m|, normalised
        ws = 0.5 * (wm[1:] * bw[1:] + wm[:-1] * bw[:-1])
        f = f + _seg_density(D.G, D.c, sG, sC, Ga[:-1], ca[:-1], Ga[1:], ca[1:]) @ ws
        em, _ = D.sig(Ga)
        pin = Phi((D.Glim - Ga) / np.sqrt(em**2 + jitter**2))
        F = F + np.sum(ws * 0.5 * (pin[1:] + pin[:-1]))
    return float(np.sum(np.log(np.maximum(f, 1e-300))) - len(D.G) * np.log(F))


ARMS = {
    "binned(1)": lambda D, th, **k: loglike_binned(D, th, **k),
    "unbinned(2)": lambda D, th, **k: loglike_unbinned(D, th, 0.0, **k),
    "unbinned(2')j0.03": lambda D, th, **k: loglike_unbinned(D, th, 0.03, **k),
}
LO = np.array([ZS[0] * 1.001, 6.0, 9.5, 0.5])
HI = np.array([ZS[-1] * 0.999, 7.0, 10.7, 2.0])


def mle(fn, D, x0, **k):
    def nll(x):
        if np.any(x < LO) or np.any(x > HI):
            return 1e30
        return -fn(D, x, **k)

    best = None
    for start in x0:
        r = minimize(
            nll,
            start,
            method="Nelder-Mead",
            options=dict(xatol=1e-5, fatol=1e-4, maxiter=4000, initial_simplex=None),
        )
        r = minimize(
            nll, r.x, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-5, maxiter=4000)
        )
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def profile_dm(fn, D, xhat, grid, **k):
    out = []
    for dmv in grid:

        def nll(y, dmv=dmv):
            x = np.array([y[0], y[1], dmv, y[2]])
            if np.any(x < LO) or np.any(x > HI):
                return 1e30
            return -fn(D, x, **k)

        r = minimize(
            nll, xhat[[0, 1, 3]], method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-4)
        )
        out.append(-r.fun)
    return np.array(out)


if __name__ == "__main__":
    which = sys.argv[1]  # arm key index
    arm = list(ARMS)[int(which)]
    fn = ARMS[arm]
    configs = [("singles", 10.47, 1.10, False, s) for s in (1, 2, 3)] + [
        ("binaries", 10.45, 1.24, True, s) for s in (1, 2, 3)
    ]
    ages_holdout = AGES[~np.isclose(AGES, 6.55)]
    res = {"arm": arm, "runs": []}
    for name, dm, Av, binaries, seed in configs:
        t0 = time.time()
        truth = np.array([0.014286, 6.55, dm, Av])
        D = Data(synthetic_cluster(0.014286, 6.55, dm, Av, seed, binaries=binaries))
        starts = [truth, np.array([0.0143, 6.5, 10.3, 1.25])]
        row = {"batch": name, "seed": seed, "truth": truth.tolist()}
        for tag, kw in (("full", {}), ("age_node_out", {"ages": ages_holdout})):
            kw = {**kw, "binaries": binaries}
            xh, lh = mle(fn, D, starts, **kw)
            g = np.round(np.arange(xh[2] - 0.2, xh[2] + 0.2001, 0.005), 4)
            pr = profile_dm(fn, D, xh, g, **kw)
            d2 = np.abs(np.diff(pr, 2))
            imax = int(np.argmax(pr))
            # curvature sd from a quadratic fit within 2 log-units of the max
            sel = pr > pr.max() - 2
            if sel.sum() >= 3:
                a = np.polyfit(g[sel], pr[sel], 2)[0]
                sd = float(np.sqrt(-1 / (2 * a))) if a < 0 else float("inf")
            else:
                sd = float("nan")
            row[tag] = {
                "mle": np.round(xh, 5).tolist(),
                "loglike": round(lh, 3),
                "dm_profile_argmax": float(g[imax]),
                "dm_sd_curv": round(sd, 4) if np.isfinite(sd) else None,
                "d2_max_over_median": round(float(d2.max() / max(np.median(d2), 1e-12)), 2),
                "profile": np.round(pr - pr.max(), 3).tolist(),
                "grid": g.tolist(),
            }
        row["seconds"] = round(time.time() - t0, 1)
        print(
            arm,
            name,
            seed,
            {
                k: (row[k]["mle"], row[k]["dm_sd_curv"], row[k]["d2_max_over_median"])
                for k in ("full", "age_node_out")
            },
            row["seconds"],
            flush=True,
        )
        res["runs"].append(row)
        (HERE / f"d1_{which}.json").write_text(json.dumps(res, indent=1, allow_nan=False))
