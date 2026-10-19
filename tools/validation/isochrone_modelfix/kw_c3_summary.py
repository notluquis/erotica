#!/usr/bin/env python3
"""Rules 1 and 2 of hub ``isochrone-intervals-nuts.md`` §2.17 (+ amendment 1) on ``kw_runs/{ISL,IBL}_k/
global.json`` -> ``kw_c3_summary.json``. err = mode of ``search_global`` in log t minus the truth."""
import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
R = HERE / "kw_runs"


def load(arm, reps=range(4)):
    return {k: json.loads((R / f"{arm}_{k}" / "global.json").read_text()) for k in reps
            if (R / f"{arm}_{k}" / "global.json").exists()}


def main():
    S, B = load("ISL"), load("IBL")
    reps = sorted(set(S) & set(B))
    eS = np.array([S[k]["err_loga"] for k in reps])
    eB = np.array([B[k]["err_loga"] for k in reps])
    d = eB - eS
    n = len(d)
    half = float(stats.t.ppf(0.95, n - 1) * d.std(ddof=1) / np.sqrt(n))
    out = {"reps": reps, "err_ISL": eS.tolist(), "err_IBL": eB.tolist(), "diff": d.tolist(),
           "rule1_median_abs_err_ISL": float(abs(np.median(eS))), "rule1_pass": bool(abs(np.median(eS)) <= 0.010),
           "D_N": float(d.mean()), "half_width90": half, "interval90": [float(d.mean() - half), float(d.mean() + half)],
           "flags": {f"{a}_{k}": v["flag_search_missed"] for a, dd in (("ISL", S), ("IBL", B)) for k, v in dd.items()},
           "mean_err_ISL": float(eS.mean()), "mean_err_IBL": float(eB.mean()),
           "sd_err_ISL": float(eS.std(ddof=1)), "sd_err_IBL": float(eB.std(ddof=1)),
           "modes_loga": {f"{a}_{k}": v["mode"]["loga"] for a, dd in (("ISL", S), ("IBL", B)) for k, v in dd.items()}}
    lo, hi = out["interval90"]
    if not out["rule1_pass"]:
        out["verdict"] = "control fails: nothing attributed to binaries"
    elif lo > 0 or hi < 0:
        out["verdict"] = ("sistematico medido" if abs(out["D_N"]) >= 0.010 else "no decidido")
    elif -0.015 <= lo and hi <= 0.015:
        out["verdict"] = "sin sistematico detectado, <= 0.015 dex"
    else:
        out["verdict"] = "no decidido"
    (HERE / "kw_c3_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()


def figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = json.loads((HERE / "kw_c3_summary.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    r = np.array(s["reps"])
    ax[0].plot(r - 0.05, s["err_ISL"], "o", label="ISL (singles / singles)")
    ax[0].plot(r + 0.05, s["err_IBL"], "s", label="IBL (binaries / binaries)")
    for k in r:
        ax[0].plot([k - 0.05, k + 0.05], [s["err_ISL"][k], s["err_IBL"][k]], color="0.7", lw=0.8)
    ax[0].axhline(0, color="k", lw=0.8)
    for y in (-0.010, 0.010):
        ax[0].axhline(y, color="C3", lw=0.6, ls=":")
    ax[0].set_xlabel("replicate (truth of cell A_k)")
    ax[0].set_ylabel("mode of log t - truth (N = 1608)")
    ax[0].legend(fontsize=8)
    ax[1].errorbar([0], [s["D_N"]], yerr=[[s["D_N"] - s["interval90"][0]], [s["interval90"][1] - s["D_N"]]], fmt="o")
    ax[1].axhspan(-0.015, 0.015, color="C2", alpha=0.15)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks([0], ["IBL - ISL, paired"])
    ax[1].set_ylabel("D_N, 90 % interval (t, 3 dof)")
    fig.tight_layout()
    fig.savefig(HERE / "kw_c3_offsets.png", dpi=90)


if __name__ == "__main__":
    figure()
