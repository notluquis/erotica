#!/usr/bin/env python3
"""Coverage of the K x W interval phase with the pre-registered rules (hub finding
``isochrone-intervals-nuts.md`` §2.5). Reads ``kw_runs/<cell>_<rep>/{prep,nuts,chain_z}.json`` and
``draws.npz``; writes ``kw_coverage.json`` and ``kw_coverage_<cell>.png``.

* NUTS intervals: equal-tailed sample quantiles 16-84 and 5-95. Control: search8 mode +- 1 and
  +- 1.645 Laplace sd (NaN sd = not covered, counted).
* Converged = R-hat <= 1.01, ESS bulk and tail >= 400, 0 divergences, chain_z pass (28 tests).
* Rule (log t primary): dies if 68 % <= 2/8 or 90 % <= 5/8; > 2 of 8 not converged -> "NUTS does
  not converge in this model" for that cell.
* Cell C is "hard" only if the control's log t mode scatter >= 3 x its median Laplace sd.
* R32-21 (cell C): control modes with the lattice's A_V start at 1.0 and 0.5 agree if
  |d log t| <= 0.005 and |d log post| <= 0.05.
* Secondary (post hoc in X1, declared here): draws reweighted to flat sigma_int and f_bg priors
  (w = 1 / (p_HN(0.05) p_Beta(1, 19))), Kish ESS, weighted 68 % coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
RUNS = HERE / "kw_runs"
PS = ("loga", "feh", "Av", "sigma_int", "k", "f_bg")


def cp(k, n):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def count(flags):
    k, n = int(sum(flags)), len(flags)
    return {"k": k, "n": n, "cp95": cp(k, n)}


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, c, x[o]))


def converged(nu, cz):
    g = nu["gate"]
    return bool(g["rhat_max"] <= 1.01 and g["ess_bulk_min"] >= 400 and g["ess_tail_min"] >= 400
                and g["divergences"] == 0 and cz is not None and cz["pass"])


def row_for(d: Path) -> dict | None:
    pj = d / "prep.json"
    if not pj.exists():
        return None
    prep = json.loads(pj.read_text())
    t = prep["truth_pymc"]
    ctrl = prep["control"]
    r = {"rep": prep["rep"], "truth": t, "truth_vs_mode": prep["truth_vs_mode"],
         "ctrl_at_bound": ctrl["at_bound"]}
    for p in PS:
        m, s = ctrl["mode"][p], ctrl["laplace_sd"].get(p, np.nan)
        ok = s is not None and np.isfinite(s)
        r[f"ctrl_{p}"] = {"mode": m, "sd": s, "err": m - t[p],
                          "in68": bool(ok and abs(m - t[p]) <= s),
                          "in90": bool(ok and abs(m - t[p]) <= 1.645 * s)}
    if "control_startav05" in prep:
        a = prep["control_startav05"]
        r["r32_21"] = {"dloga": a["mode"]["loga"] - ctrl["mode"]["loga"],
                       "dlogpost": a["logpost"] - ctrl["logpost"]}
    nj = d / "nuts.json"
    if nj.exists():
        nu = json.loads(nj.read_text())
        czp = d / "chain_z.json"
        cz = json.loads(czp.read_text()) if czp.exists() else None
        # nuts_chain's gate["converged"] includes the per-chain split R-hat clause (pre-registered in X1,
        # shown unable to pass, §1.4): renamed here so it is not mistaken for §2.5's "converged" below
        r["gate"] = {("converged_prereg_clause" if k == "converged" else k): v for k, v in nu["gate"].items()}
        r["chain_z"] = None if cz is None else {"z_abs_max": cz["z_abs_max"], "z_thr": cz["z_thr"],
                                               "pass": cz["pass"]}
        r["converged"] = converged(nu, cz)
        r["divergences_per_chain"] = nu["chains"]["divergences"]
        for p in PS:
            q = nu["params"][p]
            r[f"nuts_{p}"] = {"median": q["median"], "sd": q["sd"], "err": q["median"] - t[p],
                              "q05": q["q05"], "q16": q["q16"], "q84": q["q84"], "q95": q["q95"],
                              "in68": bool(q["q16"] <= t[p] <= q["q84"]),
                              "in90": bool(q["q05"] <= t[p] <= q["q95"]),
                              "chain_mean": q["chain_mean"]}
        dr = np.load(d / "draws.npz")
        s_, f_ = dr["sigma_int"].ravel(), dr["f_bg"].ravel()
        lw = s_**2 / (2 * 0.05**2) - 18 * np.log1p(-f_)
        w = np.exp(lw - lw.max())
        x = dr["loga"].ravel()
        r["flat_prior"] = {"kish_ess": float(w.sum() ** 2 / (w**2).sum()), "n": int(w.size),
                           "loga_median": wq(x, w, 0.5),
                           "in68": bool(wq(x, w, 0.16) <= t["loga"] <= wq(x, w, 0.84))}
    return r


def summarise(rows: list[dict], cell: str) -> dict:
    res = {"n_prep": len(rows), "rows": rows}
    for p in PS:
        cr = [r[f"ctrl_{p}"] for r in rows]
        res[f"ctrl_{p}"] = {
            "cov68": count([c["in68"] for c in cr]), "cov90": count([c["in90"] for c in cr]),
            "err_median": float(np.median([c["err"] for c in cr])),
            "err_range": [float(min(c["err"] for c in cr)), float(max(c["err"] for c in cr))],
            "scatter_of_mode": float(np.std([c["err"] for c in cr], ddof=1)) if len(cr) > 1 else None,
            "sd_median": float(np.nanmedian([c["sd"] if c["sd"] is not None else np.nan for c in cr])),
            "n_sd_nan": int(sum(1 for c in cr if c["sd"] is None or not np.isfinite(c["sd"]))),
        }
    res["ctrl_n_at_bound"] = sum(1 for r in rows if r["ctrl_at_bound"])
    lc = res["ctrl_loga"]
    if lc["scatter_of_mode"] is not None:
        res["hard"] = bool(lc["scatter_of_mode"] >= 3 * lc["sd_median"])
        res["hard_ratio"] = lc["scatter_of_mode"] / lc["sd_median"]
    r21 = [r["r32_21"] for r in rows if "r32_21" in r]
    if r21:
        agree = [abs(x["dloga"]) <= 0.005 and abs(x["dlogpost"]) <= 0.05 for x in r21]
        res["r32_21"] = {"n": len(r21), "n_agree": int(sum(agree)),
                         "max_abs_dloga": float(max(abs(x["dloga"]) for x in r21)),
                         "max_abs_dlogpost": float(max(abs(x["dlogpost"]) for x in r21))}
    nr = [r for r in rows if "gate" in r]
    if nr:
        conv = [r for r in nr if r["converged"]]
        for p in PS:
            errs = [r[f"nuts_{p}"]["err"] for r in nr]
            res[f"nuts_{p}"] = {
                "cov68_all": count([r[f"nuts_{p}"]["in68"] for r in nr]),
                "cov90_all": count([r[f"nuts_{p}"]["in90"] for r in nr]),
                "cov68_converged": count([r[f"nuts_{p}"]["in68"] for r in conv]),
                "cov90_converged": count([r[f"nuts_{p}"]["in90"] for r in conv]),
                "err_median": float(np.median(errs)),
                "err_range": [float(min(errs)), float(max(errs))],
                "scatter_of_median": float(np.std(errs, ddof=1)) if len(errs) > 1 else None,
                "sd_median": float(np.median([r[f"nuts_{p}"]["sd"] for r in nr])),
                "z_truth": [float(r[f"nuts_{p}"]["err"] / r[f"nuts_{p}"]["sd"]) for r in nr],
            }
        res["n_nuts"] = len(nr)
        res["n_converged"] = len(conv)
        res["flat_prior_cov68_loga"] = count([r["flat_prior"]["in68"] for r in nr])
        c68, c90 = res["nuts_loga"]["cov68_all"], res["nuts_loga"]["cov90_all"]
        res["rule"] = {"complete": c68["n"] == 8,
                       "dies": bool(c68["k"] <= 2 or c90["k"] <= 5) if c68["n"] == 8 else None,
                       "nonconverged_gt2": bool(len(nr) - len(conv) > 2)}
    return res


def figure(res: dict, cell: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, p in zip(ax, ("loga", "feh"), strict=True):
        for i, r in enumerate(res["rows"]):
            c = r[f"ctrl_{p}"]
            if c["sd"] is not None and np.isfinite(c["sd"]):
                a.errorbar(i - 0.12, c["err"], yerr=1.645 * c["sd"], fmt="none", ecolor="0.6", lw=1)
                a.errorbar(i - 0.12, c["err"], yerr=c["sd"], fmt="o", color="0.3", ms=4, lw=2.5,
                           label="search8 + Laplace" if i == 0 else None)
            if f"nuts_{p}" in r:
                n = r[f"nuts_{p}"]
                t = n["median"] - n["err"]
                col = "C0" if r["converged"] else "C3"
                a.plot([i + 0.12] * 2, [n["q05"] - t, n["q95"] - t], color=col, lw=1, alpha=0.6)
                a.plot([i + 0.12] * 2, [n["q16"] - t, n["q84"] - t], color=col, lw=2.5)
                a.plot(i + 0.12, n["err"], "s", color=col, ms=4,
                       label=("NUTS (converged)" if r["converged"] else "NUTS (not converged)")
                       if i == 0 else None)
        a.axhline(0, color="k", lw=0.8)
        a.set_xlabel("replicate")
        a.set_ylabel(f"estimate - truth, {p}")
        a.set_title(f"K x W cell {cell}: {p} (thick 68 %, thin 90 %)", fontsize=9)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / f"kw_coverage_{cell}.png", dpi=90)
    plt.close(fig)


def main(cells) -> None:
    out = {}
    for cell in cells:
        rows = [row_for(d) for d in sorted(RUNS.glob(f"{cell}_*"), key=lambda p: int(p.name.split("_")[1]))]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        out[cell] = summarise(rows, cell)
        figure(out[cell], cell)
        r = out[cell]
        print(cell, "n", r["n_prep"], "ctrl loga 68/90:", r["ctrl_loga"]["cov68"]["k"], r["ctrl_loga"]["cov90"]["k"],
              "hard_ratio", round(r.get("hard_ratio", float("nan")), 2), "r32_21", r.get("r32_21"),
              "| nuts n", r.get("n_nuts"), "conv", r.get("n_converged"),
              "loga 68/90", (r["nuts_loga"]["cov68_all"]["k"], r["nuts_loga"]["cov90_all"]["k"]) if "nuts_loga" in r else None,
              "rule", r.get("rule"))
    (HERE / "kw_coverage.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:] or ["A", "C"])
