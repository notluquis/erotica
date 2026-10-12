#!/usr/bin/env python3
"""Stage 1 diagnostic, per model component, of why the X1 generator is wider than the data
(hub finding ``isochrone-model-fix.md``). No fit: everything is evaluated at X1 (the data's own mode).

For data and for the 8 gencheck synthetic draws (seeds 900-907, field stars labelled), per G bin:
  * N (the luminosity function the model predicts at X1 vs the observed one);
  * colour residual about the running median (gencheck's statistic): MAD and |r| > 0.3 fraction;
  * colour residual about the model's single-star RIDGE at X1 at the star's own G: median (shape
    offset) and MAD (width about the model, which is what the likelihood sees);
  * the per-star field responsibility r_bg at X1 (data only), and how many stars have r_bg > 0.5;
  * median e_col (Gaia photometric error).
Writes ``diag.json`` and ``diag.png``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "isochrone_intervals"))
import iv_common as C  # noqa: E402
from gencheck import widths  # noqa: E402

BINS = [(8.0, 12.0), (12.0, 14.0), (14.0, 16.0), (16.0, 17.5), (17.5, 19.0), (19.0, 21.0)]


def running_resid(G, Cc):
    edges = np.arange(np.floor(G.min()), np.ceil(G.max()) + 1, 1.0)
    med = np.full(G.size, np.nan)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (G >= lo) & (G < hi)
        if m.sum() >= 3:
            med[m] = np.median(Cc[m])
    return Cc - med


def ridge_resid(gen, th, G, Cc):
    """Colour minus the single-star ridge colour at the same G; where the ridge is multivalued in G
    (Henyey hook), the branch nearest in colour."""
    d = gen._deposit(th["feh"], th["loga"], th["dm"], th["Av"], np)
    rg, rc = np.asarray(d["G"]), np.asarray(d["col"])
    out = np.full(G.size, np.nan)
    for i, (g, c) in enumerate(zip(G, Cc, strict=True)):
        best = np.inf
        for k in range(rg.size - 1):
            a, b = rg[k], rg[k + 1]
            if (a - g) * (b - g) <= 0 and a != b:
                t = (g - a) / (b - a)
                cc = rc[k] + t * (rc[k + 1] - rc[k])
                if abs(c - cc) < abs(best):
                    best = c - cc
        out[i] = best
    return out


def mad(x):
    x = x[np.isfinite(x)]
    return float(np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def binstats(G, Cc, rr, rd, eC=None, rbg=None):
    out = []
    for lo, hi in BINS:
        m = (G >= lo) & (G < hi)
        rec = {"G": [lo, hi], "n": int(m.sum())}
        if m.sum() >= 3:
            rec.update(
                run_mad=mad(rr[m]), run_tail=float(np.mean(np.abs(rr[m][np.isfinite(rr[m])]) > 0.3)),
                ridge_med=float(np.nanmedian(rd[m])), ridge_mad=mad(rd[m]),
                ridge_tail=float(np.mean(np.abs(rd[m][np.isfinite(rd[m])] - np.nanmedian(rd[m])) > 0.3)),
                ridge_nan=int(np.sum(~np.isfinite(rd[m]))),
            )
            if eC is not None:
                rec["eC_med"] = float(np.median(eC[m]))
            if rbg is not None:
                rec["rbg_sum"] = float(np.sum(rbg[m]))
                rec["n_rbg_gt_0p5"] = int(np.sum(rbg[m] > 0.5))
        out.append(rec)
    return out


def responsibilities(gen, th):
    s = [th[k] for k in C.PARAMS]
    ll = gen._star_loglike(*s, xp=np)
    field = np.log(th["f_bg"] / gen._box_area)
    return np.exp(field - ll)


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real = C.real_table()
    Gr = np.asarray(real["Gmag"], float)
    Cr = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    eCr = np.asarray(real["e_BP_RP"], float)
    gen = C.fitter(real)
    th = dict(C.X1)
    rbg = responsibilities(gen, th)
    rr = running_resid(Gr, Cr)
    rd = ridge_resid(gen, th, Gr, Cr)
    out = {"truth": th, "real": {"global": widths(Gr, Cr), "ridge_med": float(np.nanmedian(rd)),
                                 "ridge_mad": mad(rd), "n_rbg_gt_0p5": int(np.sum(rbg > 0.5)),
                                 "rbg_sum": float(rbg.sum()),
                                 "bins": binstats(Gr, Cr, rr, rd, eCr, rbg)},
           "synth": []}
    keep = None
    for s in range(900, 908):
        rng = np.random.default_rng(s)
        t, gi = C.draw_synthetic(gen, th, 254, rng, real)
        G = np.asarray(t["Gmag"], float)
        Cs = np.asarray(t["G_BPmag"], float)
        ncl = gi["n_cluster"]
        rrs = running_resid(G, Cs)
        rds = ridge_resid(gen, th, G, Cs)
        rec = {"seed": s, **gi, "all": binstats(G, Cs, rrs, rds),
               "cluster": binstats(G[:ncl], Cs[:ncl], rrs[:ncl], rds[:ncl]),
               "ridge_mad_all": mad(rds), "ridge_mad_cluster": mad(rds[:ncl])}
        out["synth"].append(rec)
        if keep is None:
            keep = (G, Cs, ncl)
        print(s, "done", flush=True)
    C.dump(HERE / "diag.json", out)

    d = gen._deposit(th["feh"], th["loga"], th["dm"], th["Av"], np)
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.5), sharey=True)
    sc = ax[0].scatter(Cr, Gr, c=rbg, s=8, cmap="viridis", vmin=0, vmax=1)
    ax[0].plot(d["col"], d["G"], "r-", lw=0.8)
    ax[0].set_title("real, colour = field responsibility at X1")
    fig.colorbar(sc, ax=ax[0])
    G, Cs, ncl = keep
    ax[1].plot(Cs[:ncl], G[:ncl], "b.", ms=3, label="cluster")
    ax[1].plot(Cs[ncl:], G[ncl:], "x", color="orange", ms=4, label="field")
    ax[1].plot(d["col"], d["G"], "r-", lw=0.8)
    ax[1].legend()
    ax[1].set_title("synthetic at X1, seed 900")
    ax[2].plot(rd, Gr, "k.", ms=3)
    ax[2].axvline(0, color="r", lw=0.8)
    ax[2].set_xlim(-1.5, 1.5)
    ax[2].set_title("real: colour minus X1 ridge at same G")
    ax[0].invert_yaxis()
    ax[0].set_xlim(-0.6, 3.6)
    ax[1].set_xlim(-0.6, 3.6)
    for a in ax:
        a.set_xlabel("BP-RP" if a is not ax[2] else "delta colour")
    ax[0].set_ylabel("G")
    fig.tight_layout()
    fig.savefig(HERE / "diag.png", dpi=90)


if __name__ == "__main__":
    main()
