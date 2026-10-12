#!/usr/bin/env python3
"""Tables and figures of the hub finding ``isochrone-ms-pms-mechanism.md`` from ``out/*.json``.

* ``fig_where.png``: (left) CMD of the 254 coloured by Delta l_i = l_i(6.75) - l_i(6.00) (per-star
  log-likelihood at the global mode; positive = the star prefers the turn-on age); (right) log t of
  each fine G slice against the slice's median mass at its own age, with the T1-null bias and SD.
* ``summary.json``: every number the finding quotes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "out"
S0, MU0 = 0.041, -0.001  # T1 null of Delta (ms - pms), isochrone-ms-vs-pms.md §2.0
DELTA_A = 0.755


def loadj(name):
    p = OUT / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def slice_masses(where):
    """Median mass of the members of each slice, read off the MIST isochrone at the slice's own
    best log t (shared parameters of the global mode): nearest isochrone point in (G, colour)."""
    from run_arms import GLOBAL, fitter, real_data
    from phases import mist

    data = real_data()
    f = fitter(mist(), data)
    G = np.asarray(where["per_star"]["G"])
    C = np.asarray(where["per_star"]["bp_rp"])
    out = {}
    for key, v in where["slice_profiles"].items():
        a = v["summary"]["loga_hat"]
        X = f._interp_isochrone(GLOBAL["feh"], a, np)
        m = X[0]
        gi = X[1] + GLOBAL["dm"] + f._kG * GLOBAL["Av"]
        ci = X[2] + f._k_col1 * GLOBAL["Av"]
        lo, hi = key.split("-")
        lo = float(lo)
        hi = 99 if hi == "faint" else float(hi)
        s = (G >= lo) & (G < hi)
        d2 = (G[s, None] - gi[None, :]) ** 2 + (C[s, None] - ci[None, :]) ** 2
        ms = m[np.argmin(d2, axis=1)]
        out[key] = {"median_mass": float(np.median(ms)), "p16": float(np.percentile(ms, 16)), "p84": float(np.percentile(ms, 84)), "n": int(s.sum())}
    return out


def null_stats():
    reps = []
    for p in sorted(OUT.glob("where_null_*.json")):
        reps += json.loads(p.read_text())["reps"]
    if not reps:
        return None
    keys = [k for k in reps[0]["slices"]]
    out = {}
    for k in keys:
        d = [r["slices"][k]["summary"]["loga_hat"] - 6.02 for r in reps if r["slices"].get(k)]
        if d:
            out[k] = {"n_reps": len(d), "bias_median": float(np.median(d)), "sd": float(np.std(d, ddof=1)) if len(d) > 1 else None}
    return out


def e3_table():
    rows = {}
    for p in sorted(OUT.glob("e3_*.json")):
        d = json.loads(p.read_text())
        r = {w: d[w]["summary"]["loga_hat"] for w in ("pms", "ms") if w in d}
        r["stats"] = d.get("stats")
        r["shared"] = d.get("shared")
        if "global" in d:
            r["global"] = d["global"]["mode"]
        if "pms" in r and "ms" in r:
            r["delta"] = r["ms"] - r["pms"]
            r["z"] = (r["delta"] - MU0) / S0
        rows[p.stem] = r
    return rows


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = {"e3": e3_table()}
    for m in ("rv", "bin", "bg"):
        d = loadj(f"mech_{m}")
        if d:
            summ[f"mech_{m}"] = {w: d[w]["summary"]["loga_hat"] for w in ("ms", "pms") if w in d}
            if "global" in d:
                summ[f"mech_{m}"]["global"] = d["global"]["mode"]
    where = loadj("where")
    if where and "slice_profiles" in where:
        L = np.asarray(where["per_star"]["loglike"])  # (N, n_age)
        ages = np.asarray(where["per_star"]["loga"])
        i6, i675 = int(np.argmin(abs(ages - 6.00))), int(np.argmin(abs(ages - 6.75)))
        dl = L[:, i675] - L[:, i6]
        info = L.max(1) - L.min(1) >= 0.5
        G = np.asarray(where["per_star"]["G"])
        C = np.asarray(where["per_star"]["bp_rp"])
        best = ages[np.argmax(L, axis=1)]
        bins = [8.8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]
        per_bin = {}
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = (G >= lo) & (G < hi)
            per_bin[f"{lo:g}-{hi:g}"] = {
                "n": int(s.sum()),
                "sum_dl": float(dl[s].sum()),
                "n_prefer_old(dl>+1)": int((dl[s] > 1).sum()),
                "n_prefer_young(dl<-1)": int((dl[s] < -1).sum()),
                "n_no_info": int((~info[s]).sum()),
                "median_best_loga_informative": float(np.median(best[s & info])) if (s & info).any() else None,
            }
        top = np.argsort(-dl)[:12]
        sid = where["per_star"].get("source_id") or [None] * len(G)
        ex = {k: where["per_star"].get(k) for k in ("probability_hdbscan", "parallax", "ruwe")}
        summ["where_per_star"] = {
            "sum_dl_all": float(dl.sum()),
            "per_bin": per_bin,
            "top_old_pullers": [
                {"source_id": sid[i], "G": G[i], "bp_rp": C[i], "dl": float(dl[i]), "best_loga": float(best[i]),
                 **{k: (v[i] if v else None) for k, v in ex.items()}}
                for i in top
            ],
        }
        summ["where_slices"] = {k: v["summary"]["loga_hat"] for k, v in where["slice_profiles"].items()}
        summ["where_slices_n"] = {k: v["n"] for k, v in where["slice_profiles"].items()}
        nul = null_stats()
        summ["where_null"] = nul
        try:
            summ["slice_masses"] = slice_masses(where)
        except Exception as e:  # noqa: BLE001
            summ["slice_masses_error"] = repr(e)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
        v = np.clip(dl, -6, 6)
        sc = ax[0].scatter(C[info], G[info], c=v[info], cmap="coolwarm", s=14, vmin=-6, vmax=6)
        ax[0].scatter(C[~info], G[~info], facecolors="none", edgecolors="0.6", s=14, label="no age information")
        ax[0].invert_yaxis()
        ax[0].set_xlabel("BP − RP")
        ax[0].set_ylabel("G")
        ax[0].axhline(14, color="k", lw=0.6, ls=":")
        ax[0].legend(loc="lower left", fontsize=8)
        plt.colorbar(sc, ax=ax[0], label="ℓ(log t 6.75) − ℓ(log t 6.00), clipped ±6")
        ax[0].set_title("Which stars pull toward the turn-on age (red) or the PMS age (blue)")
        keys = list(summ["where_slices"])
        sm = summ.get("slice_masses", {})
        for k in keys:
            y = summ["where_slices"][k]
            x = sm.get(k, {}).get("median_mass", np.nan)
            nb = (nul or {}).get(k) or {}
            ok = nb.get("sd") is not None and abs(nb.get("bias_median", 9)) <= 0.10 and nb["sd"] <= 0.15
            col = "C0" if ok else "0.6"
            err = nb.get("sd") or 0
            ax[1].errorbar(x, y, yerr=err, fmt="o", color=col, capsize=3)
            ax[1].annotate(k, (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax[1].axhline(5.98, color="C1", ls="--", lw=0.8, label="window pms 5.98")
        ax[1].axhline(6.74, color="C3", ls="--", lw=0.8, label="window ms 6.74")
        ax[1].axhspan(np.log10(2.3e6), np.log10(2.8e6), color="0.85", label="HD 159176 literature 2.3–2.8 Myr")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("median mass of the slice at its own age (M☉, MIST v1.2)")
        ax[1].set_ylabel("log t (profile maximum, arm a)")
        ax[1].set_title("Age by G slice; error bar = SD of the T1 null (grey: not readable)")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(HERE / "fig_where.png", dpi=130)
    (HERE / "summary.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    print(json.dumps(summ, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
