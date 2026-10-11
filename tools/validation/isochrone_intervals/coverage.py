#!/usr/bin/env python3
"""Coverage of the two estimators per cell, with the pre-registered rules (hub finding
``isochrone-intervals-nuts.md`` §0.5). Reads ``runs/<cell>_<rep>/{prep,nuts}.json``; writes
``coverage.json``. Intervals: NUTS = sample quantiles 16-84 and 5-95; control = mode +- 1 and
+- 1.645 Laplace sd. Coverage is counted on every fit and on the converged ones only; each count
carries its exact (Clopper-Pearson) 95 % interval and n.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iv_common as C  # noqa: E402


def cp(k, n):
    if n == 0:
        return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(0.025, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.975, k + 1, n - k))
    return [lo, hi]


def count(flags):
    k, n = int(sum(flags)), len(flags)
    return {"k": k, "n": n, "frac": (k / n if n else None), "cp95": cp(k, n)}


def main() -> None:
    out = {}
    for cell in ("A", "B", "C"):
        rows = []
        for d in sorted((HERE / "runs").glob(f"{cell}_*"), key=lambda p: int(p.name.split("_")[1])):
            pj = d / "prep.json"
            if not pj.exists():
                continue
            prep = json.loads(pj.read_text())
            t = prep["truth"]
            ctrl = prep["control"]
            row = {"rep": prep["rep"], "truth": {k: t[k] for k in ("feh", "loga")},
                   "truth_vs_mode": prep["truth_vs_mode"]}
            for p in ("feh", "loga"):
                m, s = ctrl["mode"][p], ctrl["laplace_sd"][p]
                row[f"ctrl_{p}"] = {
                    "mode": m, "sd": s, "err": m - t[p],
                    "in68": bool(np.isfinite(s) and abs(m - t[p]) <= s),
                    "in90": bool(np.isfinite(s) and abs(m - t[p]) <= 1.645 * s),
                }
            nj = d / "nuts.json"
            if nj.exists():
                nu = json.loads(nj.read_text())
                row["nuts_gate"] = nu["gate"]
                row["nuts_chains"] = nu["chains"]
                for p in ("feh", "loga"):
                    q = nu["params"][p]
                    row[f"nuts_{p}"] = {
                        "median": q["median"], "sd": q["sd"], "err": q["median"] - t[p],
                        "q05": q["q05"], "q16": q["q16"], "q84": q["q84"], "q95": q["q95"],
                        "in68": bool(q["q16"] <= t[p] <= q["q84"]),
                        "in90": bool(q["q05"] <= t[p] <= q["q95"]),
                        "chain_mean": q["chain_mean"],
                    }
            rows.append(row)
        if not rows:
            continue
        res = {"rows": rows, "n_prep": len(rows)}
        for p in ("feh", "loga"):
            cr = [r[f"ctrl_{p}"] for r in rows]
            res[f"ctrl_{p}"] = {
                "cov68": count([c["in68"] for c in cr]), "cov90": count([c["in90"] for c in cr]),
                "err_median": float(np.median([c["err"] for c in cr])),
                "err_range": [float(min(c["err"] for c in cr)), float(max(c["err"] for c in cr))],
                "scatter_of_mode": float(np.std([c["err"] for c in cr], ddof=1)) if len(cr) > 1 else None,
                "sd_median": float(np.median([c["sd"] for c in cr])),
            }
            nr = [r for r in rows if f"nuts_{p}" in r]
            if nr:
                conv = [r for r in nr if r["nuts_gate"]["converged"]]
                errs = [r[f"nuts_{p}"]["err"] for r in nr]
                res[f"nuts_{p}"] = {
                    "cov68_all": count([r[f"nuts_{p}"]["in68"] for r in nr]),
                    "cov90_all": count([r[f"nuts_{p}"]["in90"] for r in nr]),
                    "cov68_converged": count([r[f"nuts_{p}"]["in68"] for r in conv]),
                    "cov90_converged": count([r[f"nuts_{p}"]["in90"] for r in conv]),
                    "n_converged": len(conv),
                    "err_median": float(np.median(errs)),
                    "err_range": [float(min(errs)), float(max(errs))],
                    "scatter_of_median": float(np.std(errs, ddof=1)) if len(errs) > 1 else None,
                    "sd_median": float(np.median([r[f"nuts_{p}"]["sd"] for r in nr])),
                }
                if p == "loga":
                    c68, c90 = res[f"nuts_{p}"]["cov68_all"], res[f"nuts_{p}"]["cov90_all"]
                    res["rule"] = {
                        "dies": bool(c68["k"] <= 2 or c90["k"] <= 5) if c68["n"] == 8 else None,
                        "nonconverged_gt2": bool(len(nr) - len(conv) > 2),
                        "complete": c68["n"] == 8,
                    }
        out[cell] = res
    C.dump(HERE / "coverage.json", out)
    for cell, r in out.items():
        line = {k: (v["cov68"]["k"], v["cov90"]["k"]) for k, v in r.items() if k.startswith("ctrl_")}
        nline = {k: (v["cov68_all"]["k"], v["cov90_all"]["k"], v["cov68_all"]["n"], v["n_converged"])
                 for k, v in r.items() if k.startswith("nuts_")}
        print(cell, "n", r["n_prep"], "ctrl(68,90):", line, "nuts(68,90,n,conv):", nline, r.get("rule"))


if __name__ == "__main__":
    main()
