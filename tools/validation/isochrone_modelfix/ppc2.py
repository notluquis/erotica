#!/usr/bin/env python3
"""PPC2 of ``isochrone-model-fix.md`` §0.4 at an arm's refitted mode:
``python ppc2.py <arm> <win> [R]`` -> ``ppc/<arm>_<win>.json`` and ``ppc/<arm>_<win>.png``.

Family (per arm): global MAD and |r| > 0.3 fraction of the running-median colour residual
(gencheck's statistics); per G bin inside the window: N, running-median MAD, running-median tail
fraction, and ridge-residual MAD (colour minus the arm's single-star ridge at the same G) when >= 5
stars have a ridge. Two-sided predictive p = 2 min(P(T <= t), P(T >= t)) over R synthetic draws;
the arm passes if every p >= 0.05 / K. Gate noise: 50 of the synthetic draws taken as the data
against the other R - 1, same rule (false-fail rate of a perfect generator). The old gencheck
criterion (range of 8 draws widened 25 %, KS p < 0.01 in <= 2 of 8) is reported as it stands, with
its own false-fail rate measured the same way.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402

BINS_ALL = [(8.0, 12.0), (12.0, 14.0), (14.0, 16.0), (16.0, 17.5), (17.5, 19.0), (19.0, 21.0)]


def running_resid(G, Cc):
    edges = np.arange(np.floor(G.min()), np.ceil(G.max()) + 1, 1.0)
    med = np.full(G.size, np.nan)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (G >= lo) & (G < hi)
        if m.sum() >= 3:
            med[m] = np.median(Cc[m])
    return Cc - med


def ridge_resid(rg, rc, G, Cc):
    """Vectorised: colour minus the ridge colour at the same G, nearest branch in colour."""
    a, b = rg[:-1], rg[1:]
    ca, cb = rc[:-1], rc[1:]
    ok = (a[None, :] - G[:, None]) * (b[None, :] - G[:, None]) <= 0
    ok &= (a != b)[None, :]
    t = (G[:, None] - a[None, :]) / np.where(a != b, b - a, 1.0)[None, :]
    cc = ca[None, :] + t * (cb - ca)[None, :]
    d = np.where(ok, Cc[:, None] - cc, np.inf)
    j = np.argmin(np.abs(d), axis=1)
    out = d[np.arange(G.size), j]
    return np.where(np.isfinite(out), out, np.nan)


def mad(x):
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if x.size else np.nan


def stats(G, Cc, rg, rc, bins) -> dict:
    rr = running_resid(G, Cc)
    rd = ridge_resid(rg, rc, G, Cc)
    ok = np.isfinite(rr)
    s = {"glob_mad": mad(rr), "glob_tail": float(np.mean(np.abs(rr[ok]) > 0.3))}
    for lo, hi in bins:
        m = (G >= lo) & (G < hi)
        k = f"{lo:g}-{hi:g}"
        s[f"n_{k}"] = float(m.sum())
        mm = m & ok
        s[f"rmad_{k}"] = mad(rr[mm]) if mm.sum() >= 3 else np.nan
        s[f"rtail_{k}"] = float(np.mean(np.abs(rr[mm]) > 0.3)) if mm.sum() >= 3 else np.nan
        md = m & np.isfinite(rd)
        s[f"ridgemad_{k}"] = mad(rd[md]) if md.sum() >= 5 else np.nan
    return s


def pvals(t: dict, S: list[dict]) -> dict:
    out = {}
    for k, v in t.items():
        arr = np.array([s[k] for s in S], float)
        arr = arr[np.isfinite(arr)]
        if not np.isfinite(v) or arr.size < 20:
            continue
        out[k] = float(min(1.0, 2 * min(np.mean(arr <= v), np.mean(arr >= v))))
    return out


def old_gencheck(t: dict, S8: list[dict], ksG: list, ksC: list) -> bool:
    ok = True
    for k in ("glob_mad", "glob_tail"):
        v = [s[k] for s in S8]
        lo, hi = min(v), max(v)
        pad = 0.25 * (hi - lo)
        ok &= bool(lo - pad <= t[k] <= hi + pad)
    ok &= sum(p < 0.01 for p in ksG) <= 2 and sum(p < 0.01 for p in ksC) <= 2
    return bool(ok)


def main(arm: str, win: str, R: int = 200) -> dict:
    from scipy.stats import ks_2samp

    real = A.C.real_table()
    fit = json.loads((HERE / "fits" / f"{arm}_{win}.json").read_text())
    th = fit["mode"]
    f = A.make_fitter(real, win)
    gen = A.make_fitter(real, "F") if win != "F" else f
    window = A.WINDOW[win]
    Gr = np.asarray(real["Gmag"], float)
    Cr = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    if window is not None:
        m = (Gr >= window[0]) & (Gr <= window[1])
        Gr, Cr = Gr[m], Cr[m]
    n = Gr.size
    bins = [b for b in BINS_ALL if window is None or (b[0] >= window[0] and b[1] <= window[1])]
    d = f._deposit(th["feh"], th["loga"], th["dm"], th["Av"], np)
    rg, rc = np.asarray(d["G"], float), np.asarray(d["col"], float)
    t = stats(Gr, Cr, rg, rc, bins)
    S, cats = [], []
    for r in range(R):
        rng = np.random.default_rng(10_000 + r)
        G, Cc, ncl = A.draw(f, gen, th, n, rng, real, window)
        S.append(stats(G, Cc, rg, rc, bins))
        if r < 60:
            cats.append((G, Cc, ncl))
    p = pvals(t, S)
    K = len(p)
    thr = 0.05 / K
    fails = {k: v for k, v in p.items() if v < thr}
    # gate noise: synthetic j as the data against the rest
    ff = []
    for j in range(50):
        pj = pvals(S[j], S[:j] + S[j + 1:])
        ff.append(any(v < 0.05 / len(pj) for v in pj.values()))
    # old criterion on the data (8 draws) and its false-fail rate (synthetic j vs 8 others)
    ksG = [ks_2samp(c[0], Gr).pvalue for c in cats[:8]]
    ksC = [ks_2samp(c[1], Cr).pvalue for c in cats[:8]]
    old = old_gencheck(t, S[:8], ksG, ksC)
    old_ff = []
    for j in range(8, 58):
        Gj, Cj, _ = cats[j]
        kG = [ks_2samp(c[0], Gj).pvalue for c in cats[:8]]
        kC = [ks_2samp(c[1], Cj).pvalue for c in cats[:8]]
        old_ff.append(not old_gencheck(S[j], S[:8], kG, kC))
    q = {k: [float(np.nanpercentile([s[k] for s in S], x)) for x in (2.5, 50, 97.5)] for k in t}
    out = {"arm": arm, "window": win, "mode": th, "n": n, "R": R, "K": K, "threshold": thr,
           "data": t, "synth_2p5_50_97p5": q, "p": p, "fails": fails, "passes": not fails,
           "gate_false_fail": float(np.mean(ff)), "old_gencheck_pass": old,
           "old_gencheck_false_fail": float(np.mean(old_ff)), "ks_p_G": ksG, "ks_p_col": ksC}
    # tail asymmetry of the ridge residual in the PMS body (binaries, §0.1.7)
    rd = ridge_resid(rg, rc, Gr, Cr)
    body = (Gr >= 14) & (Gr < 19) & np.isfinite(rd)
    out["ridge_tail_red_blue_data"] = [float(np.mean(rd[body] - np.median(rd[body]) > 0.3)),
                                       float(np.mean(rd[body] - np.median(rd[body]) < -0.3))]
    asym = []
    for G, Cc, _ in cats:
        r2 = ridge_resid(rg, rc, G, Cc)
        b2 = (G >= 14) & (G < 19) & np.isfinite(r2)
        md = np.median(r2[b2])
        asym.append([float(np.mean(r2[b2] - md > 0.3)), float(np.mean(r2[b2] - md < -0.3))])
    out["ridge_tail_red_blue_synth_median"] = np.median(np.array(asym), axis=0).tolist()
    (HERE / "ppc").mkdir(exist_ok=True)
    A.C.dump(HERE / "ppc" / f"{arm}_{win}.json", out)
    figure(arm, win, Gr, Cr, cats[0], rg, rc, t, S, bins)
    print(json.dumps({k: out[k] for k in ("passes", "K", "fails", "gate_false_fail",
                                          "old_gencheck_pass", "old_gencheck_false_fail")}))
    return out


def figure(arm, win, Gr, Cr, cat, rg, rc, t, S, bins):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    G, Cc, ncl = cat
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.5))
    ax[0].plot(Cr, Gr, "k.", ms=3)
    ax[0].plot(rc, rg, "r-", lw=0.8)
    ax[0].set_title(f"real, ridge of {arm}x{win}")
    ax[1].plot(Cc[:ncl], G[:ncl], "b.", ms=3)
    ax[1].plot(Cc[ncl:], G[ncl:], "x", color="orange", ms=4)
    ax[1].plot(rc, rg, "r-", lw=0.8)
    ax[1].set_title("one synthetic (orange = field)")
    for a in ax[:2]:
        a.set_xlim(-0.6, 3.6)
        a.set_ylim(21, 8)
        a.set_xlabel("BP-RP")
    ax[0].set_ylabel("G")
    ks = [f"{b[0]:g}-{b[1]:g}" for b in bins]
    x = np.arange(len(ks))
    for j, (pref, col) in enumerate((("ridgemad_", "C0"), ("rmad_", "C1"))):
        lo = [np.nanpercentile([s[pref + k] for s in S], 2.5) for k in ks]
        hi = [np.nanpercentile([s[pref + k] for s in S], 97.5) for k in ks]
        ax[2].fill_between(x + 0.1 * j, lo, hi, color=col, alpha=0.3, label=f"synth 95% {pref[:-1]}")
        ax[2].plot(x + 0.1 * j, [t[pref + k] for k in ks], "o", color=col, label=f"data {pref[:-1]}")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(ks)
    ax[2].set_xlabel("G bin")
    ax[2].set_ylabel("MAD of colour residual")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / "ppc" / f"{arm}_{win}.png", dpi=90)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 200)
