#!/usr/bin/env python3
"""Applies the pre-registered rules of ``isochrone-colour-mechanism.md`` §0.2-§0.3 by code, from
``colour_synth_<arm>.json`` and ``hr_fit_<run>.json``. Output ``colour_summary.json``.

Experiment 1 (medians over the R paired replicates, ``lori`` primary):
reproduces iff d_feh <= -0.20 and d_loga <= -0.15; dies iff d_feh > -0.10 or d_loga > -0.05;
uninterpretable iff ``control`` has |d_feh| > 0.10 or |d_loga| > 0.10.
Experiment 2: dies iff R4 feh <= -0.30 and |loga(R4) - loga(R2)| < 0.10; undecided iff R3 and R4
differ in log t by more than R2 differs from either.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = ("control", "lori", "bhac15", "spots0")


def main() -> None:
    out = {"exp1": {}, "exp2": {}}
    reps = {}
    for arm in ARMS:
        p = HERE / f"colour_synth_{arm}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())["reps"]
        reps[arm] = r
        row = {"n": len(r)}
        for k in ("d_feh", "d_loga", "d_Av"):
            x = np.array([e[k] for e in r])
            row[k] = {"median": float(np.median(x)), "min": float(x.min()), "max": float(x.max())}
        s = np.array([e["mode"]["sigma_int"] for e in r])
        row["sigma_int_median"] = float(np.median(s))
        row["n_at_bound"] = int(sum(bool(e["at_bound"]) for e in r))
        out["exp1"][arm] = row
    c = out["exp1"].get("control")
    if c:
        out["exp1"]["control_recovers"] = (
            abs(c["d_feh"]["median"]) <= 0.10 and abs(c["d_loga"]["median"]) <= 0.10
        )
    lo = out["exp1"].get("lori")
    if lo:
        f, a = lo["d_feh"]["median"], lo["d_loga"]["median"]
        out["exp1"]["lori_verdict"] = (
            "reproduces"
            if (f <= -0.20 and a <= -0.15)
            else "dies"
            if (f > -0.10 or a > -0.05)
            else "partial"
        )
    for arm in ("lori", "bhac15", "spots0"):
        if arm in reps and "control" in reps:
            d = [e["d_feh"] - k["d_feh"] for e, k in zip(reps[arm], reps["control"], strict=True)]
            out["exp1"][arm]["paired_d_feh_minus_control_median"] = float(np.median(d))
    runs = {}
    for r in ("R1", "R2", "R3", "R4", "R3b"):
        p = HERE / f"hr_fit_{r}.json"
        if p.exists():
            j = json.loads(p.read_text())
            runs[r] = {
                "mode": j["mode"],
                "age_Myr": j["age_Myr"],
                "at_bound": j["at_bound"],
                "n": j["n"],
            }
    out["exp2"]["runs"] = runs
    if all(k in runs for k in ("R2", "R3", "R4")):
        la = {k: runs[k]["mode"]["loga"] for k in ("R2", "R3", "R4")}
        d34 = abs(la["R3"] - la["R4"])
        out["exp2"]["dies"] = runs["R4"]["mode"]["feh"] <= -0.30 and abs(la["R4"] - la["R2"]) < 0.10
        out["exp2"]["undecided"] = d34 > abs(la["R2"] - la["R4"]) and d34 > abs(la["R2"] - la["R3"])
        out["exp2"]["dloga"] = {
            "R3-R4": la["R3"] - la["R4"],
            "R4-R2": la["R4"] - la["R2"],
            "R3-R2": la["R3"] - la["R2"],
        }
    print(json.dumps(out, indent=1))
    (HERE / "colour_summary.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
