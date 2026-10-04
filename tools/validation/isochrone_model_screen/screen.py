#!/usr/bin/env python3
r"""Screen of seven changes to the isochrone model of NGC 6383, as one fractional factorial.

WHY THIS EXISTS
---------------
The hub finding ``isochrone-model-screen.md`` (pre-registration in its §0, committed before any
stage here ran) designs this: seven two-level factors (``model.py``) in a 2^(7-3) resolution-IV
design, 16 configurations, each evaluated (1) at the MAP on the real 254 members and (2) on synthetic
clusters with known truth that carry the effects the factors model (differential reddening,
colour-dependent extinction, BP/RP-contaminated colours, a Gaia parallax with its systematic floor),
at two pre-registered truths. Only the 2-3 configurations the pre-registered rule selects go to NUTS.

WHAT WOULD FALSIFY ITS CONCLUSIONS
----------------------------------
The rule and the gates are in the finding's §0. In short: a configuration is acceptable only if on
BOTH truths |bias(loga)| <= 0.05 dex and |bias(dm)| <= 0.05 mag (mean over replicates, with its SE
reported) and the profile 68 % interval of loga covers the truth at a rate inside the binomial 95 %
acceptance band of 0.68 for the pooled n. A factor "matters" if its main effect exceeds 2 SE.

STAGES
------
``check``     oracles: ScreenFitter with switches off == package log L; Fitz19 k at the CCM
              wavelengths' colour; parallax summary.
``real``      the 16 configurations (+ C1's own A_V bound as a reproduction arm) at the MAP on the
              real data, 3 fixed seeds, Laplace and loga-profile intervals -> ``real.json``.
``gencheck``  generator predictive check against the real CMD at the base-config real MAP.
``synth``     ``--truth T1|T2 --reps R --part k/n``: replicates x 16 configs -> ``synth_*.json``.
``summary``   main effects, gates, ranking -> ``summary.json``.

Resources: one process each, ``OMP_NUM_THREADS=1 NPROC=1 XLA_FLAGS=--xla_force_host_platform_device_count=1
nice -n 19``; ~0.25 s per log-posterior gradient (JAX), ~400-600 MB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))

from astropy.table import QTable, Table  # noqa: E402
from isochrone_nuts_convergence import MIST, SAMPLE  # noqa: E402
from model import Z_SUN, ScreenFitter, fitz19_k  # noqa: E402

FACTORS = ["zprior", "diffred", "loga_lo", "plx", "imf", "ext", "cstar"]
# 2^(7-3) resolution IV: A..D full factorial, E = ABC, F = BCD, G = ACD (Box, Hunter & Hunter)
ZPRIOR = (0.08, 0.10)  # [M/H] mean, sd: OC radial gradients at R_gc 7.19 kpc (finding §0)
PLX_FLOOR = 0.0103  # mas, Maíz Apellániz+2021 (methodology §J.2)
BASE_PRI = dict(Av_range=(0.0, 3.0), dm_mu=10.3, dm_sigma=0.2, dm_range=(9.5, 10.7))
SEEDS = [  # fixed, declared; none is a synthetic truth
    dict(met=0.0050, loga=6.07, dm=10.17, Av=0.64, sigma_int=0.15, f_bg=0.15, sigma_Av=0.2),
    dict(met=0.0175, loga=6.60, dm=10.25, Av=1.25, sigma_int=0.10, f_bg=0.15, sigma_Av=0.2),
    dict(met=0.0100, loga=6.30, dm=10.20, Av=0.95, sigma_int=0.12, f_bg=0.15, sigma_Av=0.2),
]
TRUTHS = {
    "T1": dict(met=0.0060, loga=6.15, dm=10.20, Av=0.80, f_bg=0.15),
    "T2": dict(met=0.0175, loga=6.50, dm=10.25, Av=1.15, f_bg=0.15),
}


def design():
    rows = []
    for i in range(16):
        a, b, c, d = [1 if (i >> k) & 1 else -1 for k in (3, 2, 1, 0)]
        rows.append(dict(zip(FACTORS, (a, b, c, d, a * b * c, b * c * d, a * c * d), strict=True)))
    return rows


def cstar(tab):
    C = (np.asarray(tab["phot_bp_mean_flux"]) + np.asarray(tab["phot_rp_mean_flux"])) / np.asarray(
        tab["phot_g_mean_flux"]
    )
    x = np.asarray(tab["G_BPmag"]) - np.asarray(tab["G_RPmag"])
    f = np.where(
        x < 0.5,
        1.154360 + 0.033772 * x + 0.032277 * x**2,
        np.where(
            x < 4.0,
            1.162004 + 0.011464 * x + 0.049255 * x**2 - 0.005879 * x**3,
            1.057572 + 0.140537 * x,
        ),
    )
    G = np.asarray(tab["Gmag"])
    return C - f, 0.0059898 + 8.817481e-12 * G**7.618399


def parallax_summary(tab):
    plx = np.asarray(tab["parallax"], float)
    e = np.asarray(tab["parallax_error"], float)
    ok = np.isfinite(plx) & np.isfinite(e)
    w = 1 / e[ok] ** 2
    mu = float(np.sum(w * plx[ok]) / np.sum(w))
    chi2 = float(np.sum(w * (plx[ok] - mu) ** 2) / (ok.sum() - 1))
    stat = float(1 / np.sqrt(np.sum(w)) * np.sqrt(max(chi2, 1.0)))
    return dict(
        plx=mu, stat=stat, chi2red=chi2, sigma=float(np.hypot(stat, PLX_FLOOR)), N=int(ok.sum())
    )


def real_table():
    return Table.read(SAMPLE)


def make_fitter(cfg, tab):
    pri = dict(BASE_PRI)
    pri["loga_range"] = (5.5, 7.0) if cfg["loga_lo"] > 0 else (6.0, 7.0)
    if cfg.get("Av_lo") is not None:
        pri["Av_range"] = (cfg["Av_lo"], 2.0)
    f = ScreenFitter(
        isochs_path=MIST,
        imf="chc14" if cfg["imf"] > 0 else "c05",
        ext="fitz19" if cfg["ext"] > 0 else "ccm",
        diffred=cfg["diffred"] > 0,
        **pri,
    )
    t = tab
    if cfg["cstar"] > 0:
        cs, sig = cstar(tab)
        t = tab[np.abs(cs) <= 3 * sig]
    f.setup(QTable(t), prob_threshold=0.0, precompute_grid=False)
    cache = Path(os.path.expanduser("~/.cache/erotica-screen"))
    cache.mkdir(parents=True, exist_ok=True)
    f.build_grid(grid_cache=cache / f"nodes_{pri['loga_range'][0]}.npz")
    f._plx = parallax_summary(t)
    return f


# ------------------------------------------------------------------ log posterior
NAMES = ["lz", "loga", "dm", "Av", "ls", "lf", "lsa"]


def compile_logpost(f, cfg, with_prior=True):
    """Compiled (value, grad) of log post (or log L) in u = (log10 Z, loga, dm, Av, ln sigma_int,
    logit f_bg, ln sigma_Av) -- densities in the natural parameters (no Jacobian of u), as the
    profiles of the C1 review."""
    import pytensor
    import pytensor.tensor as pt

    u = pt.dvector("u")
    met = 10.0 ** u[0]
    loga, dm, Av = u[1], u[2], u[3]
    s = pt.exp(u[4])
    fb = pt.sigmoid(u[5])
    sa = pt.exp(u[6]) if cfg["diffred"] > 0 else 0.0
    ll = pt.sum(f._star_weights * f.star_ll(met, loga, dm, Av, s, fb, sa, pt))
    lp = 0.0
    if with_prior:
        # sigma_int HalfNormal(0.05), f_bg Beta(1, 19) -- as build_model
        lp = lp - 0.5 * (s / 0.05) ** 2 + 18.0 * pt.log(1 - fb)
        if cfg["diffred"] > 0:
            lp = lp - 0.5 * (sa / 0.5) ** 2  # HalfNormal(0.5) on sigma_Av [mag]
        if cfg["zprior"] > 0:  # Normal in [M/H]; met is the sampled variable, Jacobian 1/Z
            mh = u[0] - np.log10(Z_SUN)
            lp = lp - 0.5 * ((mh - ZPRIOR[0]) / ZPRIOR[1]) ** 2 - pt.log(met)
        if cfg["plx"] > 0:
            p = f._plx
            lp = lp - 0.5 * ((p["plx"] - 10.0 ** (2.0 - dm / 5.0)) / p["sigma"]) ** 2
        else:
            lp = lp - 0.5 * ((dm - f.dm_mu) / f.dm_sigma) ** 2
    obj = ll + lp
    fn = pytensor.function([u], [obj, pytensor.grad(obj, u)], mode="JAX")
    return lambda x: (lambda o: (float(o[0]), np.asarray(o[1], float)))(fn(np.asarray(x, float)))


def bounds(f, cfg):
    zlo, zhi = 10 ** float(f._node_logz[0]), 10 ** float(f._node_logz[-1])
    b = [
        (np.log10(max(zlo, 1e-4)), np.log10(zhi) - 1e-6),
        tuple(f.loga_range),
        tuple(f.dm_range),
        tuple(f.Av_range),
        (np.log(2e-3), np.log(0.5)),
        (-8.0, 0.0),
        (np.log(2e-3), np.log(1.5)),
    ]
    if cfg["diffred"] <= 0:
        b[6] = (0.0, 0.0)
    return b


def to_u(p):
    return [
        np.log10(p["met"]),
        p["loga"],
        p["dm"],
        p["Av"],
        np.log(p.get("sigma_int", 0.1)),
        np.log(p.get("f_bg", 0.15) / (1 - p.get("f_bg", 0.15))),
        np.log(p.get("sigma_Av", 0.2)),
    ]


def maximise(fn, x0, bnd, fixed=None, maxiter=300):
    """L-BFGS-B on the free coordinates; ``fixed`` = {index: value}."""
    from scipy.optimize import minimize

    fixed = dict(fixed or {})
    for i, (lo, hi) in enumerate(bnd):
        if lo == hi:
            fixed[i] = lo
    free = [i for i in range(len(bnd)) if i not in fixed]
    full = np.array(x0, float)
    for i, v in fixed.items():
        full[i] = v

    def nf(z):
        x = full.copy()
        x[free] = z
        v, g = fn(x)
        if not np.isfinite(v):
            return 1e10, np.zeros(len(free))
        return -v, -g[free]

    z0 = np.clip(full[free], [bnd[i][0] + 1e-7 for i in free], [bnd[i][1] - 1e-7 for i in free])
    r = minimize(
        nf,
        z0,
        jac=True,
        method="L-BFGS-B",
        bounds=[bnd[i] for i in free],
        options=dict(maxiter=maxiter),
    )
    x = full.copy()
    x[free] = r.x
    return -float(r.fun), x, int(r.nfev)


def laplace(fn, x, bnd, h=1e-4):
    free = [i for i in range(len(x)) if bnd[i][0] != bnd[i][1]]
    H = np.empty((len(free), len(free)))
    for a, i in enumerate(free):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        H[a] = (fn(xp)[1][free] - fn(xm)[1][free]) / (2 * h)
    H = 0.5 * (H + H.T)
    try:
        cov = np.linalg.inv(-H)
        ok = bool(np.all(np.linalg.eigvalsh(-H) > 0))
    except np.linalg.LinAlgError:
        return None, False
    sd = np.full(len(x), np.nan)
    sd[free] = np.sqrt(np.clip(np.diag(cov), 0, None))
    logdet = float(np.linalg.slogdet(cov)[1]) if ok else np.nan
    return dict(sd=sd.tolist(), logdet_cov=logdet, n_free=len(free)), ok


def profile_interval(fn, x, best, bnd, sd0, k=1, level=0.5):
    """Profile 68 % interval of coordinate k: walk out in steps of ~sd until the profile drops by
    ``level``, linear interpolation at the crossing; a bound reached is reported as the bound."""
    out = []
    step0 = float(np.clip(sd0 if np.isfinite(sd0) and sd0 > 0 else 0.05, 0.01, 0.15))
    for sgn in (-1, 1):
        prev_v, prev_t, xw = best, x[k], x.copy()
        t = x[k]
        edge = None
        for _ in range(8):
            t = t + sgn * step0
            if t <= bnd[k][0] or t >= bnd[k][1]:
                edge = bnd[k][0] if sgn < 0 else bnd[k][1]
                break
            v, xw, _ = maximise(fn, xw, bnd, fixed={k: t}, maxiter=150)
            if best - v >= level:
                frac = (best - level - prev_v) / (v - prev_v) if v != prev_v else 1.0
                edge = prev_t + frac * (t - prev_t)
                break
            prev_v, prev_t = v, t
            step0 *= 1.5
        out.append(float(edge) if edge is not None else float(t))
    return out


def fit(f, cfg, seeds=SEEDS, intervals=True, fn=None):
    fn = fn or compile_logpost(f, cfg)
    bnd = bounds(f, cfg)
    res = []
    for sd in seeds:
        x0 = np.clip(to_u(sd), [b[0] for b in bnd], [b[1] for b in bnd])
        v, x, nfev = maximise(fn, x0, bnd)
        res.append((v, x, nfev))
    res.sort(key=lambda r: -r[0])
    best, x, _ = res[0]
    out = dict(
        logpost=best,
        u=x.tolist(),
        seeds=[[float(r[0]), r[1].tolist()] for r in res],
        nfev=int(sum(r[2] for r in res)),
    )
    out["params"] = dict(
        met=10 ** x[0],
        mh=x[0] - np.log10(Z_SUN),
        loga=x[1],
        dm=x[2],
        Av=x[3],
        sigma_int=float(np.exp(x[4])),
        f_bg=float(1 / (1 + np.exp(-x[5]))),
        sigma_Av=float(np.exp(x[6])) if cfg["diffred"] > 0 else 0.0,
    )
    out["at_bound"] = {
        NAMES[i]: bool(min(abs(x[i] - bnd[i][0]), abs(x[i] - bnd[i][1])) < 1e-3) for i in range(4)
    }
    if intervals:
        lap, ok = laplace(fn, x, bnd)
        out["laplace"] = lap
        out["laplace_ok"] = ok
        sd1 = lap["sd"][1] if lap else 0.05
        out["loga_profile68"] = profile_interval(fn, x, best, bnd, sd1, k=1)
    return out


# ------------------------------------------------------------------ synthetic generator
def error_models(tab):
    from erotica.analysis._isochrone import _fit_error_model

    G = np.asarray(tab["Gmag"], float)
    eG = np.asarray(tab["e_Gmag"], float)
    ec = np.asarray(tab["e_BP_RP"], float)
    return _fit_error_model(G, eG, ec)


def generate(truth, gen, tab, rng):
    """One synthetic member list, same N as the real one.

    Process (each piece with its source, finding §0): cluster stars from the package's forward
    model at the truth with the *c05* IMF and Offner/D&K binaries, intrinsic photometry; per-star
    A_V,i = A_V + sigma_Av * N(0,1) (clipped >= 0) with Fitz19 colour-dependent k at the star's
    intrinsic colour; + dm; Gaussian errors from the real error model at each star's G; cut at the
    real G_lim; field stars uniform in the real CMD box, fraction f_bg; BP/RP-contaminated colours:
    each star is flagged with the real flagged fraction of its G bin, a flagged star gets a real
    flagged C* from the same bin and a colour shift ``slope * C*`` (slope measured on the real
    faint stars, frozen); parallaxes from the truth distance + one common offset N(0, 10.3 µas)
    + per-star errors resampled from real stars of similar G."""
    N = len(tab)
    Gr = np.asarray(tab["Gmag"], float)
    colr = np.asarray(tab["G_BPmag"], float) - np.asarray(tab["G_RPmag"], float)
    Glim = float(Gr.max())
    eGf, ecf = error_models(tab)
    g = gen["fitter"]  # ScreenFitter, c05, ccm, nodes built: used for intrinsic draws only
    d = g._deposit(truth["met"], truth["loga"], 0.0, 0.0, np)
    w_single = d["w_single"]
    w_bin = sum(d["w_bin"])
    p = np.concatenate([w_single, w_bin])
    p = np.clip(p, 0, None) / np.clip(p, 0, None).sum()
    from erotica.analysis._isochrone import _Q_NODES, _dk_gamma

    g1 = _dk_gamma(d["m_b"]) + 1.0
    n_cl = rng.binomial(N, 1 - truth["f_bg"])
    Gs, Cs = [], []
    while sum(len(x) for x in Gs) < n_cl:
        k = rng.choice(len(p), size=4 * N, p=p)
        single = k < len(w_single)
        G0 = np.empty(k.size)
        C0 = np.empty(k.size)
        ks, t = k[single], rng.uniform(size=int(single.sum()))
        G0[single] = d["G"][ks] + t * (d["G"][ks + 1] - d["G"][ks])
        C0[single] = d["col"][ks] + t * (d["col"][ks + 1] - d["col"][ks])
        kb = k[~single] - len(w_single)
        q = rng.uniform(size=kb.size) ** (1.0 / g1[kb])
        j = np.clip(np.searchsorted(_Q_NODES, q, side="right") - 1, 0, len(_Q_NODES) - 2)
        fr = (q - _Q_NODES[j]) / (_Q_NODES[j + 1] - _Q_NODES[j])
        G0[~single] = d["Gq"][j, kb] + fr * (d["Gq"][j + 1, kb] - d["Gq"][j, kb])
        C0[~single] = d["cq"][j, kb] + fr * (d["cq"][j + 1, kb] - d["cq"][j, kb])
        Av_i = np.clip(truth["Av"] + gen["sigma_Av"] * rng.normal(size=k.size), 0.0, None)
        kG = fitz19_k("G", C0, Av_i)
        kc = fitz19_k("BP", C0, Av_i) - fitz19_k("RP", C0, Av_i)
        G = G0 + truth["dm"] + kG * Av_i
        C = C0 + kc * Av_i
        eg = np.sqrt(eGf(np.clip(G, Gr.min(), Gr.max())) ** 2 + 0.01**2)
        ec = np.sqrt(ecf(np.clip(G, Gr.min(), Gr.max())) ** 2 + 0.01**2)
        Go = G + rng.normal(size=G.size) * eg
        Co = C + rng.normal(size=G.size) * ec
        keep = Go <= Glim
        Gs.append(Go[keep])
        Cs.append(Co[keep])
    Gc = np.concatenate(Gs)[:n_cl]
    Cc = np.concatenate(Cs)[:n_cl]
    n_bg = N - n_cl
    Gb = rng.uniform(Gr.min(), Gr.max(), n_bg)
    Cb = rng.uniform(colr.min(), colr.max(), n_bg)
    G = np.r_[Gc, Gb]
    C = np.r_[Cc, Cb]
    # BP/RP contamination, sized from the real stars (frozen in gen)
    cs_real, sig_real = cstar(tab)
    flag_real = np.abs(cs_real) > 3 * sig_real
    cs = np.empty(N)
    sig = 0.0059898 + 8.817481e-12 * G**7.618399
    edges = gen["G_edges"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (G >= lo) & (G < hi)
        mr = (Gr >= lo) & (Gr < hi)
        frac = flag_real[mr].mean() if mr.sum() else 0.0
        fl = m & (rng.uniform(size=N) < frac)
        pool = cs_real[mr & flag_real]
        cs[fl] = rng.choice(pool, fl.sum()) if pool.size else 0.0
        cs[m & ~fl] = rng.normal(size=int((m & ~fl).sum())) * sig[m & ~fl]
    C = C + gen["cstar_slope"] * np.where(np.abs(cs) > 3 * sig, cs, 0.0)
    # parallaxes
    plx_true = 10.0 ** (2.0 - truth["dm"] / 5.0) + rng.normal() * PLX_FLOOR
    er = np.asarray(tab["parallax_error"], float)
    order = np.argsort(Gr)
    idx = np.clip(np.searchsorted(Gr[order], G), 0, N - 1)
    e_plx = er[order][idx]
    plx = plx_true + rng.normal(size=N) * e_plx
    # a table with the columns setup() and cstar() read; fluxes rebuilt so cstar() returns cs
    t = Table()
    t["Gmag"] = G
    bp = G + 0.5 * C  # BP-RP = C; G not on the BP/RP mean (irrelevant to the likelihood)
    t["G_BPmag"] = bp
    t["G_RPmag"] = bp - C
    t["e_Gmag"] = eGf(np.clip(G, Gr.min(), Gr.max()))
    t["e_BP_RP"] = ecf(np.clip(G, Gr.min(), Gr.max()))
    x = C
    fx = np.where(
        x < 0.5,
        1.154360 + 0.033772 * x + 0.032277 * x**2,
        np.where(
            x < 4.0,
            1.162004 + 0.011464 * x + 0.049255 * x**2 - 0.005879 * x**3,
            1.057572 + 0.140537 * x,
        ),
    )
    t["phot_g_mean_flux"] = np.ones(N)
    t["phot_bp_mean_flux"] = 0.5 * (cs + fx)
    t["phot_rp_mean_flux"] = 0.5 * (cs + fx)
    t["parallax"] = plx
    t["parallax_error"] = e_plx
    t["probability_hdbscan"] = np.ones(N)
    t["is_cluster"] = np.r_[np.ones(n_cl, bool), np.zeros(n_bg, bool)]
    return t


def gen_config(tab, real):
    """Frozen generator sizes (finding §0): sigma_Av from the real-data MAP of the config that
    differs from the base only in ``diffred``; cstar slope from the audit OLS on faint stars."""
    audit = json.loads((HERE / "audit.json").read_text())
    slope = float(audit["residuals"]["ols_resid_col_on_cstar_faint"][0])
    base = {k: -1 for k in FACTORS}
    dr = real["arms"]["diffred_only"]["params"]["sigma_Av"]
    g = make_fitter(dict(base), tab)
    return dict(
        fitter=g,
        sigma_Av=float(dr),
        cstar_slope=slope,
        G_edges=[8.0, 14.0, 17.0, 19.0, 21.0],
        sources=dict(sigma_Av="real.json arms.diffred_only MAP", cstar_slope="audit.json OLS"),
    )


# ------------------------------------------------------------------ stages
def stage_check(args):
    from erotica.analysis._isochrone import IsochroneFitter

    tab = real_table()
    base = {k: -1 for k in FACTORS}
    f = make_fitter(base, tab)
    pri = dict(BASE_PRI, loga_range=(6.0, 7.0))
    g = IsochroneFitter(isochs_path=MIST, **pri)
    g.setup(QTable(tab), prob_threshold=0.0)
    p = (0.00505, 6.0713, 10.167, 0.643, 0.148, 0.157)
    a = float(np.sum(f.star_ll(*p, 0.0, np)))
    b = g.loglike(*p)
    out = dict(screen_switches_off=a, package=b, diff=a - b)
    fe = make_fitter(dict(base, ext=1), tab)
    out["fitz19_ll_same_point"] = float(np.sum(fe.star_ll(*p, 0.0, np)))
    out["ccm_k"] = dict(kG=f._kG, kcol=f._k_col1)
    out["fitz19_k_at_A1"] = {
        str(X): dict(
            kG=float(fitz19_k("G", X, 1.0)),
            kcol=float(fitz19_k("BP", X, 1.0) - fitz19_k("RP", X, 1.0)),
        )
        for X in (-0.06, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
    }
    fd = make_fitter(dict(base, diffred=1), tab)
    out["diffred_zero_equals_base"] = float(np.sum(fd.star_ll(*p, 0.0, np))) - a
    out["parallax"] = f._plx
    out["parallax_dm"] = float(5 * (2 - np.log10(f._plx["plx"])))
    out["parallax_sigma_dm"] = float(5 / np.log(10) * f._plx["sigma"] / f._plx["plx"])
    (HERE / "check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


def stage_real(args):
    tab = real_table()
    out = {"design": design(), "configs": [], "arms": {}}
    path = HERE / "real.json"
    if path.exists():
        out = json.loads(path.read_text())
    base = {k: -1 for k in FACTORS}
    arms = {"diffred_only": dict(base, diffred=1), "c1_Av_bound": dict(base, Av_lo=0.5)}
    for name, cfg in arms.items():
        if name in out["arms"]:
            continue
        t = time.time()
        r = fit(make_fitter(cfg, tab), cfg)
        r["cfg"], r["seconds"] = cfg, round(time.time() - t, 1)
        out["arms"][name] = r
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(name, json.dumps(r["params"]), r["seconds"], flush=True)
    done = {json.dumps(c["cfg"], sort_keys=True) for c in out["configs"]}
    for i, cfg in enumerate(design()):
        if json.dumps(cfg, sort_keys=True) in done:
            continue
        t = time.time()
        r = fit(make_fitter(cfg, tab), cfg)
        r["cfg"], r["run"], r["seconds"] = cfg, i, round(time.time() - t, 1)
        out["configs"].append(r)
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")
        print(i, cfg, json.dumps(r["params"]), r["loga_profile68"], r["seconds"], flush=True)


def stage_gencheck(args):
    """Generator predictive check (methodology §A.1.3) at the real MAP of the base config."""
    from scipy.stats import ks_2samp

    tab = real_table()
    real = json.loads((HERE / "real.json").read_text())
    gen = gen_config(tab, real)
    base = next(c for c in real["configs"] if all(v == -1 for v in c["cfg"].values()))
    tr = {k: base["params"][k] for k in ("met", "loga", "dm", "Av", "f_bg")}
    rng = np.random.default_rng(7)
    Gr = np.asarray(tab["Gmag"], float)
    cr = np.asarray(tab["G_BPmag"], float) - np.asarray(tab["G_RPmag"], float)
    cs_r, sg_r = cstar(tab)
    rows = []
    for _ in range(20):
        t = generate(tr, gen, tab, rng)
        G = np.asarray(t["Gmag"])
        c = np.asarray(t["G_BPmag"]) - np.asarray(t["G_RPmag"])
        cs, sg = cstar(t)
        rows.append(
            dict(
                ks_G=float(ks_2samp(G, Gr).pvalue),
                ks_col=float(ks_2samp(c, cr).pvalue),
                frac_flag=float(np.mean(np.abs(cs) > 3 * sg)),
                n_G_lt14=int((G < 14).sum()),
                n_G_ge19=int((G >= 19).sum()),
            )
        )
    out = dict(
        truth_from="base real MAP",
        truth=tr,
        gen={k: v for k, v in gen.items() if k != "fitter"},
        real=dict(
            frac_flag=float(np.mean(np.abs(cs_r) > 3 * sg_r)),
            n_G_lt14=int((Gr < 14).sum()),
            n_G_ge19=int((Gr >= 19).sum()),
        ),
        synthetic_median={k: float(np.median([r[k] for r in rows])) for k in rows[0]},
        synthetic_range={
            k: [float(np.min([r[k] for r in rows])), float(np.max([r[k] for r in rows]))]
            for k in rows[0]
        },
    )
    (HERE / "gencheck.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps(out, indent=1, default=float))


def stage_synth(args):
    tab = real_table()
    real = json.loads((HERE / "real.json").read_text())
    gen = gen_config(tab, real)
    truth = TRUTHS[args.truth]
    k, n = map(int, args.part.split("/"))
    path = HERE / f"synth_{args.truth}_{k}of{n}.json"
    out = json.loads(path.read_text()) if path.exists() else {"truth": truth, "reps": []}
    have = {(r["rep"], json.dumps(r["cfg"], sort_keys=True)) for r in out["reps"]}
    cfgs = design()
    # truth-evaluation arm (methodology §A.1.3, astro-inference): log post at the truth vs MAP
    for rep in range(args.reps):
        if rep % n != k:
            continue
        rng = np.random.default_rng(1000 * (1 if args.truth == "T1" else 2) + rep)
        t = generate(truth, gen, tab, rng)
        for cfg in cfgs:
            key = (rep, json.dumps(cfg, sort_keys=True))
            if key in have:
                continue
            t0 = time.time()
            f = make_fitter(cfg, t)
            fn = compile_logpost(f, cfg)
            r = fit(f, cfg, fn=fn)
            ut = to_u(
                dict(
                    truth,
                    sigma_int=r["params"]["sigma_int"],
                    sigma_Av=max(r["params"]["sigma_Av"], 2e-3),
                )
            )
            ut[4], ut[5] = r["u"][4], r["u"][5]
            r["logpost_at_truth_other_at_map"] = fn(ut)[0]
            r.update(rep=rep, cfg=cfg, seconds=round(time.time() - t0, 1), N=len(t))
            out["reps"].append(r)
            path.write_text(json.dumps(out, default=float) + "\n")
            print(
                args.truth,
                rep,
                cfg,
                round(r["params"]["loga"], 3),
                r["loga_profile68"],
                r["seconds"],
                flush=True,
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["check", "real", "gencheck", "synth"])
    ap.add_argument("--truth", default="T1")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    dict(check=stage_check, real=stage_real, gencheck=stage_gencheck, synth=stage_synth)[a.stage](a)


if __name__ == "__main__":
    main()
