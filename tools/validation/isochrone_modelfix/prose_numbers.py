#!/usr/bin/env python3
"""The prose numbers of ``isochrone-model-fix.md`` that no other script prints (review of 2026-10-12,
m1): §0.1.5 ridge slopes at X1, §0.1.6 C*/beta counts and Spearman, §1.2 16-84 % half-widths of the
V x W ridge residual, §2 partial Spearman of |residual| on e_col at fixed G (data) and its
distribution under the K x W model itself (R synthetic catalogues), and the blue-minus-red tail
difference in the body against K x W. -> ``prose_numbers.json``."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402
from ppc2 import ridge_resid, running_resid  # noqa: E402


def partial_rho(r, e, g):
    from scipy.stats import pearsonr, rankdata

    R = [rankdata(x) for x in (np.abs(r), e, g)]

    def res(a, b):
        return a - np.polyval(np.polyfit(b, a, 1), b)

    return pearsonr(res(R[0], R[2]), res(R[1], R[2]))


def main(R=200):
    import pandas as pd
    from scipy.stats import spearmanr

    real = A.C.real_table()
    G = np.asarray(real["Gmag"], float)
    Cc = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    eC = np.asarray(real["e_BP_RP"], float)
    out = {}
    # §0.1.5 slopes at X1
    fF = A.make_fitter(real, "F")
    x1 = A.C.X1
    d = fF._deposit(x1["feh"], x1["loga"], x1["dm"], x1["Av"], np)
    rg, rc = np.asarray(d["G"]), np.asarray(d["col"])
    sl = {}
    for g0 in (14.5, 15.5, 16.5, 17.5, 18.25):
        k = [i for i in range(len(rg) - 1) if (rg[i] - g0) * (rg[i + 1] - g0) <= 0][0]
        j = slice(max(k - 3, 0), k + 4)
        s = float(np.polyfit(rc[j], rg[j], 1)[0])
        sl[str(g0)] = {"dG_dcol": s, "per_vertical": 1 / abs(s),
                       "per_dAV": abs(fF._k_col1 - fF._kG / s)}
    out["slopes_X1"] = sl
    # §0.1.6 C* and beta (E6 table) on the running-median residual
    e = pd.read_csv(HERE / "e6_members_computed.csv", dtype={"source_id": str}).set_index("source_id")
    e = e.reindex([str(s) for s in real["source_id"]])
    rr = running_resid(G, Cc)
    m = (G >= 17.5) & (G < 19)
    tail = m & (np.abs(rr) > 0.3)
    ns, beta, cs = e["Cstar_nsigma"].values, e["beta"].values, e["Cstar"].values
    rho = spearmanr(np.abs(rr[m]), np.abs(cs[m]))
    out["cstar_17p5_19"] = {"n_tail": int(tail.sum()), "tail_Cstar_gt3s": int(np.sum(np.abs(ns[tail]) > 3)),
                            "tail_beta_gt_0p1": int(np.sum(beta[tail] > 0.1)),
                            "spearman_absres_absCstar": [float(rho.statistic), float(rho.pvalue)]}
    # §1.2 half-widths (V x W) and §2 partial rho (I x W ridge, as first measured; and K x W)
    fW = A.make_fitter(real, "W")
    for arm in ("V", "I", "K"):
        th = json.loads((HERE / "fits" / f"{arm}_W.json").read_text())["mode"]
        dd = fW._deposit(th["feh"], th["loga"], th["dm"], th["Av"], np)
        rd = ridge_resid(np.asarray(dd["G"]), np.asarray(dd["col"]), G, Cc)
        rec = {}
        for lo, hi in ((14, 16), (16, 17.5), (17.5, 18.25), (18.25, 19)):
            mm = (G >= lo) & (G < hi) & np.isfinite(rd)
            q = np.percentile(rd[mm] - np.median(rd[mm]), [16, 84])
            rec[f"{lo}-{hi}"] = {"half_16_84": float((q[1] - q[0]) / 2), "eC_med": float(np.median(eC[mm]))}
        mm = (G >= 17.5) & (G < 19) & np.isfinite(rd)
        pr = partial_rho(rd[mm] - np.median(rd[mm]), eC[mm], G[mm])
        rec["partial_rho_17p5_19"] = [float(pr.statistic), float(pr.pvalue), int(mm.sum())]
        out[f"ridge_{arm}_W"] = rec
    # the same partial rho, and the blue-red tail difference, under K x W itself
    th = json.loads((HERE / "fits" / "K_W.json").read_text())["mode"]
    gen = fF
    dd = fW._deposit(th["feh"], th["loga"], th["dm"], th["Av"], np)
    rg, rc = np.asarray(dd["G"]), np.asarray(dd["col"])
    win = A.WINDOW["W"]
    Gr, Cr = G[(G >= win[0]) & (G <= win[1])], Cc[(G >= win[0]) & (G <= win[1])]
    rho_s, asym_s = [], []
    for r in range(R):
        rng = np.random.default_rng(20_000 + r)
        Gs, Cs, _, eCs = A.draw(fW, gen, th, Gr.size, rng, real, win, return_errors=True)
        rds = ridge_resid(rg, rc, Gs, Cs)
        mm = (Gs >= 17.5) & (Gs < 19) & np.isfinite(rds)
        rho_s.append(partial_rho(rds[mm] - np.median(rds[mm]), eCs[mm], Gs[mm]).statistic)
        b = (Gs >= 14) & (Gs < 19) & np.isfinite(rds)
        x = rds[b] - np.median(rds[b])
        asym_s.append(float(np.mean(x < -0.3) - np.mean(x > 0.3)))
    rho_d = out["ridge_K_W"]["partial_rho_17p5_19"][0]
    rdK = ridge_resid(rg, rc, Gr, Cr)
    b = np.isfinite(rdK)
    x = rdK[b] - np.median(rdK[b])
    asym_d = float(np.mean(x < -0.3) - np.mean(x > 0.3))
    rho_s, asym_s = np.array(rho_s), np.array(asym_s)
    out["under_K_W"] = {
        "R": R,
        "partial_rho_quantiles_2p5_50_97p5": np.percentile(rho_s, [2.5, 50, 97.5]).tolist(),
        "P_rho_le_data": float(np.mean(rho_s <= rho_d)),
        "asym_blue_minus_red_data": asym_d,
        "asym_quantiles_2p5_50_97p5": np.percentile(asym_s, [2.5, 50, 97.5]).tolist(),
        "asym_p_two_sided": float(min(1, 2 * min(np.mean(asym_s <= asym_d), np.mean(asym_s >= asym_d)))),
    }
    A.C.dump(HERE / "prose_numbers.json", out)
    print(json.dumps(out["under_K_W"]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)
