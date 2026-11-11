#!/usr/bin/env python3
"""Summary of ``av_synth_<arm>.json`` and ``hr_nspec_<mode>.json`` against the pre-registered rules
(hub finding ``isochrone-external-constraints.md`` §0.5), written to ``external_summary.json``.
Also rewrites every JSON of this directory with NaN -> null (the repo's strict-JSON hook).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ["ctl", "map-ign", "map-cor", "grad-ign", "grad-map", "pri", "pri2", "ctl-off"]
# replicates whose mode lands exactly on the lattice node that is the truth (hub finding §4)
NODE_TOL = 1e-6


def clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o


def med(x):
    x = np.asarray(x, float)
    return [float(np.median(x)), float(x.min()), float(x.max())]


def main():
    out = {"arms": {}, "hr_nspec": {}}
    for a in ARMS:
        p = HERE / f"av_synth_{a}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        r = d["reps"]
        s = {
            "R": len(r),
            "d_feh": med([x["d_feh"] for x in r]),
            "d_loga": med([x["d_loga"] for x in r]),
            "d_Av": med([x["d_Av"] for x in r]),
            "sigma_int": float(np.median([x["mode"]["sigma_int"] for x in r])),
            "cover_feh": sum(bool(x["cover_feh"]) for x in r),
            "cover_loga": sum(bool(x["cover_loga"]) for x in r),
            "laplace_sd_loga_median": float(
                np.nanmedian(
                    [
                        np.nan if x["laplace_sd"]["loga"] is None else x["laplace_sd"]["loga"]
                        for x in r
                    ]
                )
            ),
            "laplace_sd_feh_median": float(
                np.nanmedian(
                    [
                        np.nan if x["laplace_sd"]["feh"] is None else x["laplace_sd"]["feh"]
                        for x in r
                    ]
                )
            ),
            "rmse_loga": float(np.sqrt(np.mean([x["d_loga"] ** 2 for x in r]))),
            "n_at_bound": sum(bool(x["at_bound"]) for x in r),
            "prior": d.get("prior"),
            "main_map": d.get("main_map"),
        }
        on_node = [abs(x["d_loga"]) < NODE_TOL for x in r]
        off = [x for x, n in zip(r, on_node, strict=True) if not n]
        s["n_on_truth_node"] = sum(on_node)
        s["sd_loga_between_reps"] = float(np.std([x["d_loga"] for x in r], ddof=1))
        if off:
            s["off_node"] = {
                "R": len(off),
                "d_loga": med([x["d_loga"] for x in off]),
                "cover_loga": sum(bool(x["cover_loga"]) for x in off),
                "cover_feh": sum(bool(x["cover_feh"]) for x in off),
            }
        if a != "ctl" and (HERE / "av_synth_ctl.json").exists() and a != "ctl-off":
            c = json.loads((HERE / "av_synth_ctl.json").read_text())["reps"]
            n = min(len(c), len(r))
            s["paired_minus_ctl"] = {
                q: med([r[i][f"d_{q}"] - c[i][f"d_{q}"] for i in range(n)])
                for q in ("feh", "loga", "Av")
            }
        bad = abs(s["d_feh"][0]) > 0.10 or abs(s["d_loga"][0]) > 0.10
        if a in ("ctl", "map-cor", "ctl-off"):
            s["gate_recovers"] = not bad
        if a == "grad-ign":
            df, dl = s["d_feh"][0], s["d_loga"][0]
            s["rule"] = (
                "reproduces"
                if (df <= -0.20 and dl <= -0.15)
                else "dies"
                if (df > -0.10 or dl > -0.05)
                else "partial"
            )
        out["arms"][a] = s
    for m in ("free", "prior"):
        p = HERE / f"hr_nspec_{m}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        by = {}
        for x in d["runs"]:
            by.setdefault(x["N"], []).append(x)
        out["hr_nspec"][m] = {
            str(n): {
                "R": len(v),
                "sd_loga_between_reps": float(np.std([x["d_loga"] for x in v], ddof=1))
                if len(v) > 1
                else None,
                "rmse_loga": float(np.sqrt(np.mean([x["d_loga"] ** 2 for x in v]))),
                "median_d_loga": float(np.median([x["d_loga"] for x in v])),
                "median_d_feh": float(np.median([x["d_feh"] for x in v])),
                "n_loga_at_bound": sum("loga" in x["at_bound"] for x in v),
            }
            for n, v in sorted(by.items())
        }
    for p in HERE.glob("*.json"):
        p.write_text(json.dumps(clean(json.loads(p.read_text())), indent=1) + "\n")
    (HERE / "external_summary.json").write_text(json.dumps(clean(out), indent=1) + "\n")
    print(json.dumps(clean(out), indent=1))


if __name__ == "__main__":
    main()
