#!/usr/bin/env python3
"""Paired comparison of the cited NUTS fits against their re-runs after R113-03/04 (pre-registration:
hub ``agent-findings/r113-nutpie-rerun-prereg-2026-11-12.md``).

    r113_compare.py --cited CITED_ROOT --new NEW_ROOT --control CONTROL_ROOT --out OUT.json

Each root holds ``iv_runs/<fit>/nuts.json`` and ``kw_runs/<fit>/nuts.json`` (plus ``chain_z.json``).
For every fit present in both, per parameter (log t, [Fe/H]) and quantile (5, 50, 95 %): the
difference new - cited and its Monte Carlo standard error, MCSE_q = sd sqrt(q (1 - q)) /
(phi(z_q) sqrt(ESS)), ESS tail for 5 / 95 % and bulk for 50 %, combined in quadrature over the two
fits; the flag ``beyond_3mcse``; and the convergence of both (the cited gate: R-hat <= 1.01,
ESS >= 400, no divergences, ``chain_z`` pass). ``--control`` adds the engine control: the same A_0
data sampled with numpyro (``nuts_chain.py``) against nutpie (this re-run). The quantile MCSE assumes
a locally normal posterior; it is the rule fixed in the pre-registration, not a calibrated error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm

QS = (("q05", 0.05, "ess_tail"), ("median", 0.50, "ess_bulk"), ("q95", 0.95, "ess_tail"))
PARS = ("loga", "feh")


def mcse(p: dict, q: float, ess_key: str) -> float:
    z = norm.ppf(q)
    ess = max(float(p[ess_key]), 1.0)
    return float(p["sd"] * np.sqrt(q * (1 - q)) / (norm.pdf(z) * np.sqrt(ess)))


def converged(fit: Path, nuts: dict) -> bool:
    g = nuts["gate"]
    cz = fit / "chain_z.json"
    zpass = json.loads(cz.read_text())["pass"] if cz.exists() else None
    return bool(
        g.get("core_finite", True)
        and g["rhat_max"] <= 1.01
        and g["ess_bulk_min"] >= 400
        and g["ess_tail_min"] >= 400
        and g["divergences"] == 0
        and zpass is not False
    )


def zmax(fit: Path):
    cz = fit / "chain_z.json"
    return json.loads(cz.read_text())["z_abs_max"] if cz.exists() else None


def compare(a_dir: Path, b_dir: Path) -> dict:
    a = json.loads((a_dir / "nuts.json").read_text())
    b = json.loads((b_dir / "nuts.json").read_text())
    row = {"a_converged": converged(a_dir, a), "b_converged": converged(b_dir, b)}
    for side, d, fd in (("a", a, a_dir), ("b", b, b_dir)):
        g = d["gate"]
        row[f"{side}_gate"] = {
            "rhat_max": g["rhat_max"],
            "ess_bulk_min": g["ess_bulk_min"],
            "ess_tail_min": g["ess_tail_min"],
            "divergences": g["divergences"],
            "z_abs_max": zmax(fd),
        }
    for par in PARS:
        pa, pb = a["params"][par], b["params"][par]
        for key, q, ek in QS:
            m = float(np.hypot(mcse(pa, q, ek), mcse(pb, q, ek)))
            dq = float(pb[key] - pa[key])
            row[f"{par}_{key}"] = {
                "a": pa[key],
                "b": pb[key],
                "diff": dq,
                "mcse": m,
                "beyond_3mcse": bool(abs(dq) > 3 * m),
            }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cited", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--control", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {"pairs": {}, "engine_control": {}}
    for sub in ("iv_runs", "kw_runs"):
        for fit in sorted((Path(a.new) / sub).glob("*")):
            c = Path(a.cited) / sub / fit.name
            if (fit / "nuts.json").exists() and (c / "nuts.json").exists():
                out["pairs"][f"{sub}/{fit.name}"] = compare(c, fit)
        if a.control:
            for fit in sorted((Path(a.control) / sub).glob("*")):
                n = Path(a.new) / sub / fit.name
                if (fit / "nuts.json").exists() and (n / "nuts.json").exists():
                    out["engine_control"][f"{sub}/{fit.name}"] = compare(fit, n)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for name, rows in (
        ("pairs (a = cited, b = re-run)", out["pairs"]),
        ("engine control (a = numpyro, b = nutpie)", out["engine_control"]),
    ):
        print(f"## {name}")
        for k, r in rows.items():
            flags = [f"{p}" for p in r if isinstance(r[p], dict) and r[p].get("beyond_3mcse")]
            lt = r["loga_median"]
            print(
                f"{k:18s} conv {r['a_converged']!s:5s}->{r['b_converged']!s:5s} "
                f"log t q50 {lt['a']:.3f}->{lt['b']:.3f} (d {lt['diff']:+.4f}, 3mcse {3 * lt['mcse']:.4f}) "
                f"beyond: {','.join(flags) or '-'}"
            )


if __name__ == "__main__":
    main()
