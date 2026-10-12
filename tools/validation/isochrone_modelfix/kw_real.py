#!/usr/bin/env python3
"""The K x W fit to the data (hub finding ``isochrone-intervals-nuts.md`` §2.6), after NUTS:

    kw_real.py ppc  kw_runs/real_0 [R]   posterior predictive check on NUTS draws -> ppc_draws.json/.png
    kw_real.py comb kw_runs/real_0       comb on the MIST log t nodes             -> comb.json/.png
    kw_real.py age  kw_runs/real_0       log t quantiles and Myr                  -> age.json

PPC on draws (answers m5 of R42-04: PPC2 certified the MAP, without the priors): R draws evenly
spaced over the four chains; for each draw s a catalogue y_s at theta_s (``aniso.draw``) and the 14
statistics of PPC2 (``ppc2.stats``), the ridge residual taken against the isochrone OF THAT DRAW for
both the data and y_s (realised discrepancy). Two-sided p = 2 min(P(T(y_s) <= T(y)), P(T(y_s) >=
T(y))) over s; passes if every p >= 0.05 / K (Bonferroni over the family). Gate noise: 50 replicates
y_j taken as the data against the other draws, same rule.

Comb: fraction of log t draws within +-0.0025 dex of a MIST node against the same fraction under a
Gaussian-KDE smoothing (bandwidth 0.015) of the draws; ratio > 1.5 is declared a comb.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402
import ppc2 as P2  # noqa: E402

WIN = "W"
ZFIX = os.environ.get("ZFIX")


def _draws(run: Path, R: int):
    d = np.load(run / "draws.npz")
    nch, nd = d["loga"].shape
    flat = {k: d[k].reshape(-1) for k in d.files}
    idx = np.linspace(0, nch * nd - 1, R).round().astype(int)
    out = []
    for i in idx:
        feh = float(ZFIX) if ZFIX else float(flat["feh"][i])
        th = {"feh": feh, "loga": float(flat["loga"][i]), "dm": float(flat["dm"][i]),
              "Av": float(flat["Av"][i]), "sG": float(flat["sigma_int"][i]),
              "k": float(flat["k"][i]), "f_bg": float(flat["f_bg"][i])}
        out.append(A._fill("K", th, 0.0))
    return out


def ppc(run: str, R: int = 1000) -> dict:
    run = Path(run)
    real = A.C.real_table()
    f = A.make_fitter(real, WIN)
    gen = A.make_fitter(real, "F")
    window = A.WINDOW[WIN]
    Gr = np.asarray(real["Gmag"], float)
    Cr = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    m = (Gr >= window[0]) & (Gr <= window[1])
    Gr, Cr = Gr[m], Cr[m]
    n = Gr.size
    bins = [b for b in P2.BINS_ALL if b[0] >= window[0] and b[1] <= window[1]]
    ths = _draws(run, R)
    T_obs, T_rep, ridges, cats = [], [], [], []
    for s, th in enumerate(ths):
        dep = f._deposit(th["feh"], th["loga"], th["dm"], th["Av"], np)
        rg, rc = np.asarray(dep["G"], float), np.asarray(dep["col"], float)
        G, Cc, _ = A.draw(f, gen, th, n, np.random.default_rng(20_000 + s), real, window)
        T_obs.append(P2.stats(Gr, Cr, rg, rc, bins))
        T_rep.append(P2.stats(G, Cc, rg, rc, bins))
        ridges.append((rg, rc))
        if s < 50:
            cats.append((G, Cc))
    keys = list(T_obs[0])

    def pv(obs, rep):
        out = {}
        for k in keys:
            o = np.array([x[k] for x in obs], float)
            r = np.array([x[k] for x in rep], float)
            ok = np.isfinite(o) & np.isfinite(r)
            if ok.sum() < 20:
                continue
            out[k] = float(min(1.0, 2 * min(np.mean(r[ok] <= o[ok]), np.mean(r[ok] >= o[ok]))))
        return out

    p = pv(T_obs, T_rep)
    K = len(p)
    thr = 0.05 / K
    fails = {k: v for k, v in p.items() if v < thr}
    ff = []
    for j in range(50):
        Gj, Cj = cats[j]
        oj = [P2.stats(Gj, Cj, ridges[s][0], ridges[s][1], bins) for s in range(R) if s != j]
        rj = [T_rep[s] for s in range(R) if s != j]
        pj = pv(oj, rj)
        ff.append(any(v < 0.05 / len(pj) for v in pj.values()))
    q = {k: {"data_median": float(np.nanmedian([x[k] for x in T_obs])),
             "rep_2p5_50_97p5": [float(np.nanpercentile([x[k] for x in T_rep], z)) for z in (2.5, 50, 97.5)]}
         for k in keys}
    out = {"run": str(run), "R": R, "K": K, "threshold": thr, "p": p, "fails": fails,
           "passes": not fails, "gate_false_fail": float(np.mean(ff)),
           "gate_false_fail_k": int(sum(ff)), "gate_n": 50, "stats": q, "zfix": ZFIX}
    (run / "ppc_draws.json").write_text(json.dumps(out, indent=1) + "\n")
    _ppcfig(run, q, p, thr)
    print(json.dumps({k: out[k] for k in ("R", "K", "threshold", "fails", "passes", "gate_false_fail")}))
    print({k: round(v, 3) for k, v in sorted(p.items(), key=lambda kv: kv[1])[:5]})
    return out


def _ppcfig(run, q, p, thr):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [k for k in q if k in p]
    fig, ax = plt.subplots(figsize=(9, 0.35 * len(ks) + 1.5))
    for i, k in enumerate(ks):
        lo, mid, hi = q[k]["rep_2p5_50_97p5"]
        sc = max(abs(hi - lo), 1e-9)
        ax.plot([(lo - mid) / sc, (hi - mid) / sc], [i, i], color="0.5", lw=3)
        ax.plot((q[k]["data_median"] - mid) / sc, i, "o", color="C3" if p[k] < thr else "C0")
        ax.text(0.62, i, f"p = {p[k]:.3f}", va="center", fontsize=7)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks, fontsize=7)
    ax.set_xlim(-1.2, 0.9)
    ax.set_xlabel("(T - median of replicates) / 95 % width  (grey: replicates 2.5-97.5 %; dot: data, median over draws)")
    ax.set_title(f"PPC on NUTS draws, {run.name}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "ppc_draws.png", dpi=90)
    plt.close(fig)


def comb(run: str) -> dict:
    from scipy.stats import gaussian_kde

    run = Path(run)
    x = np.load(run / "draws.npz")["loga"].ravel()
    nodes = np.round(np.arange(5.70, 7.0001, 0.05), 4)
    near = np.min(np.abs(x[:, None] - nodes[None, :]), axis=1) <= 0.0025
    kde = gaussian_kde(x, bw_method=0.015 / x.std(ddof=1))
    grid = np.linspace(x.min() - 0.05, x.max() + 0.05, 4001)
    dens = kde(grid)
    dens /= np.trapezoid(dens, grid)
    gnear = np.min(np.abs(grid[:, None] - nodes[None, :]), axis=1) <= 0.0025
    smooth = float(np.trapezoid(np.where(gnear, dens, 0.0), grid))
    out = {"frac_near_node": float(near.mean()), "frac_near_node_smooth": smooth,
           "ratio": float(near.mean() / smooth) if smooth > 0 else None,
           "comb": bool(smooth > 0 and near.mean() / smooth > 1.5),
           "nodes_in_range": [float(v) for v in nodes if x.min() <= v <= x.max()]}
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(x, bins=np.arange(x.min(), x.max() + 0.0025, 0.0025), density=True, color="0.6")
    ax.plot(grid, dens, "C0")
    for v in out["nodes_in_range"]:
        ax.axvline(v, color="C3", lw=0.8)
    ax.set_xlabel("log t (draws; red = MIST nodes)")
    ax.set_title(f"comb check {run.name}: ratio {out['ratio']:.2f}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run / "comb.png", dpi=90)
    plt.close(fig)
    (run / "comb.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out)
    return out


def age(run: str) -> dict:
    run = Path(run)
    x = np.load(run / "draws.npz")["loga"].ravel()
    q = np.quantile(x, [0.05, 0.16, 0.5, 0.84, 0.95])
    out = {"loga_q05_16_50_84_95": q.tolist(), "Myr_q05_16_50_84_95": (10 ** (q - 6)).tolist(),
           "n_draws": int(x.size)}
    (run / "age.json").write_text(json.dumps(out, indent=1) + "\n")
    print(out)
    return out


if __name__ == "__main__":
    stage, run = sys.argv[1], sys.argv[2]
    if stage == "ppc":
        ppc(run, int(sys.argv[3]) if len(sys.argv) > 3 else 1000)
    elif stage == "comb":
        comb(run)
    else:
        age(run)
