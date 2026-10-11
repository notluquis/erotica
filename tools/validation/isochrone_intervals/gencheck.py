#!/usr/bin/env python3
"""Predictive check of the generator (methodology §A.1.3) BEFORE any coverage is read: synthetic
catalogues drawn at X1's best fit (the real data's own mode) against the 254 real members.

Statistics (each over 8 synthetic draws, seeds 900-907): two-sample KS p on G and on BP-RP;
the colour residual about a running median in G (bins of 1 mag): its MAD and the fraction of stars
more than 0.3 mag from it; and the G-rank correlation of the errors. Figure: real vs one synthetic.
The generator is accepted if the real value of each width statistic lies inside the range of the
8 synthetic ones widened by 25 %, and no KS p < 0.01 in more than 2 of 8 draws.
Writes ``gencheck.json`` and ``gencheck.png``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iv_common as C  # noqa: E402


def widths(G, Cc):
    edges = np.arange(np.floor(G.min()), np.ceil(G.max()) + 1, 1.0)
    med = np.full(G.size, np.nan)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        m = (G >= lo) & (G < hi)
        if m.sum() >= 3:
            med[m] = np.median(Cc[m])
    r = Cc - med
    ok = np.isfinite(r)
    return {
        "mad": float(np.median(np.abs(r[ok] - np.median(r[ok])))),
        "frac_gt_0p3": float(np.mean(np.abs(r[ok]) > 0.3)),
    }


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    real = C.real_table()
    Gr = np.asarray(real["Gmag"], float)
    Cr = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    gen = C.fitter(real)
    truth = dict(C.X1)
    out = {"truth": truth, "real": widths(Gr, Cr), "synth": []}
    keep = None
    for s in range(900, 908):
        rng = np.random.default_rng(s)
        t, gi = C.draw_synthetic(gen, truth, 254, rng, real)
        G = np.asarray(t["Gmag"], float)
        Cs = np.asarray(t["G_BPmag"], float)
        rec = {
            "seed": s, **gi, **widths(G, Cs),
            "ks_p_G": float(ks_2samp(G, Gr).pvalue),
            "ks_p_col": float(ks_2samp(Cs, Cr).pvalue),
        }
        out["synth"].append(rec)
        print(rec, flush=True)
        if keep is None:
            keep = (G, Cs)
    syn = out["synth"]
    verdict = {}
    for k in ("mad", "frac_gt_0p3"):
        v = [r[k] for r in syn]
        lo, hi = min(v), max(v)
        pad = 0.25 * (hi - lo)
        verdict[k] = {"real": out["real"][k], "synth_min": lo, "synth_max": hi,
                      "inside": bool(lo - pad <= out["real"][k] <= hi + pad)}
    for k in ("ks_p_G", "ks_p_col"):
        n = sum(r[k] < 0.01 for r in syn)
        verdict[k] = {"n_below_0.01": int(n), "ok": bool(n <= 2)}
    out["verdict"] = verdict
    out["accepted"] = bool(all(v.get("inside", v.get("ok")) for v in verdict.values()))
    C.dump(HERE / "gencheck.json", out)
    fig, ax = plt.subplots(1, 2, figsize=(8, 5), sharex=True, sharey=True)
    ax[0].plot(Cr, Gr, "k.", ms=3)
    ax[0].set_title("real (254)")
    ax[1].plot(keep[1], keep[0], "b.", ms=3)
    ax[1].set_title("synthetic at X1, seed 900")
    ax[0].invert_yaxis()
    for a in ax:
        a.set_xlabel("BP-RP")
    ax[0].set_ylabel("G")
    fig.tight_layout()
    fig.savefig(HERE / "gencheck.png", dpi=90)
    print(verdict, "accepted", out["accepted"])


if __name__ == "__main__":
    main()
