#!/usr/bin/env python3
"""Applies the pre-registered rule of ``colour_synth_mb.py`` (hub finding
``isochrone-multiband-implementation.md`` §2.2) to ``colour_synth_mb_<arm>.json`` and writes
``colour_summary_mb.json``. The verdict is computed here, not written by hand.

Statistic: b_X(set) = median over paired replicates of Delta_X(lori) - Delta_X(control); rho_X =
|b_X(gaia)| / |b_X(nir)|. Primary [Fe/H]: rho >= 3 confirmed, rho < 2 dies, else neither. log t read
only if the 95 % bootstrap interval of b_loga(gaia) excludes 0. Gate noise: bootstrap of rho over
replicates (10 000, seed 0). Interpretability: control |median| <= 0.10 in both sets and both
parameters; b_feh(gaia) within 0.05 of -0.173 (colour experiment). nirJ: breaks if the paired
|median| of Delta[Fe/H] or Delta log t against control is >= 0.05.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COLOUR_PAIRED_FEH = (
    -0.17280062194416757
)  # colour_summary.json exp1.lori.paired_d_feh_minus_control_median


def load(arm: str) -> dict:
    p = HERE / f"colour_synth_mb_{arm}.json"
    return json.loads(p.read_text())["reps"] if p.exists() else {}


def paired(a: dict, b: dict, s: str, x: str) -> np.ndarray:
    keys = sorted(k for k in a if k in b and s in a[k] and s in b[k])
    return np.array([a[k][s][x] - b[k][s][x] for k in keys])


def boot_median(d: np.ndarray, rng, n=10000) -> np.ndarray:
    idx = rng.integers(0, len(d), (n, len(d)))
    return np.median(d[idx], axis=1)


def main() -> None:
    ctl, lori, nirj = load("control"), load("lori"), load("nirJ")
    rng = np.random.default_rng(0)
    out: dict = {"n_pairs": {}, "control": {}, "b": {}, "ci95_b": {}}
    for s in ("gaia", "nir"):
        out["control"][s] = {
            x: float(np.median([ctl[k][s][x] for k in ctl if s in ctl[k]]))
            for x in ("d_feh", "d_loga", "d_Av")
        }
        out["n_pairs"][s] = int(len(paired(lori, ctl, s, "d_feh")))
        for x in ("d_feh", "d_loga", "d_Av"):
            d = paired(lori, ctl, s, x)
            out["b"][f"{x}_{s}"] = float(np.median(d))
            bm = boot_median(d, rng)
            out["ci95_b"][f"{x}_{s}"] = [
                float(np.percentile(bm, 2.5)),
                float(np.percentile(bm, 97.5)),
            ]
    # paired over replicates present in both sets
    keys = sorted(
        k for k in lori if k in ctl and all(s in lori[k] and s in ctl[k] for s in ("gaia", "nir"))
    )
    dg = np.array([lori[k]["gaia"]["d_feh"] - ctl[k]["gaia"]["d_feh"] for k in keys])
    dn = np.array([lori[k]["nir"]["d_feh"] - ctl[k]["nir"]["d_feh"] for k in keys])
    rho = abs(np.median(dg)) / max(abs(np.median(dn)), 1e-12)
    idx = rng.integers(0, len(keys), (10000, len(keys)))
    rho_b = np.abs(np.median(dg[idx], axis=1)) / np.maximum(
        np.abs(np.median(dn[idx], axis=1)), 1e-12
    )
    verdict = "confirmed" if rho >= 3 else ("dies" if rho < 2 else "neither")
    lg = np.array([lori[k]["gaia"]["d_loga"] - ctl[k]["gaia"]["d_loga"] for k in keys])
    ln = np.array([lori[k]["nir"]["d_loga"] - ctl[k]["nir"]["d_loga"] for k in keys])
    ci_lg = out["ci95_b"]["d_loga_gaia"]
    loga_resolved = not (ci_lg[0] <= 0.0 <= ci_lg[1])
    rho_loga = abs(np.median(lg)) / max(abs(np.median(ln)), 1e-12)
    ctl_ok = all(
        abs(out["control"][s][x]) <= 0.10 for s in ("gaia", "nir") for x in ("d_feh", "d_loga")
    )
    gen_ok = abs(float(np.median(dg)) - COLOUR_PAIRED_FEH) <= 0.05
    out["primary"] = {
        "pairs": keys,
        "b_feh_gaia": float(np.median(dg)),
        "b_feh_nir": float(np.median(dn)),
        "rho_feh": float(rho),
        "verdict": verdict,
        "boot_P_rho_lt_2": float(np.mean(rho_b < 2)),
        "boot_P_rho_ge_3": float(np.mean(rho_b >= 3)),
        "boot_rho_p5_p50_p95": [float(np.percentile(rho_b, q)) for q in (5, 50, 95)],
    }
    out["secondary_loga"] = {
        "b_loga_gaia": float(np.median(lg)),
        "b_loga_nir": float(np.median(ln)),
        "ci95_b_loga_gaia": ci_lg,
        "resolved": bool(loga_resolved),
        "rho_loga": float(rho_loga) if loga_resolved else None,
    }
    out["interpretable"] = {
        "control_within_0.10": bool(ctl_ok),
        "gaia_b_within_0.05_of_colour": bool(gen_ok),
    }
    if nirj:
        dj = {x: paired(nirj, ctl, "nir", x) for x in ("d_feh", "d_loga", "d_Av")}
        zp = [nirj[k]["nir"]["mode"].get("zp_2MASS_J") for k in sorted(nirj) if "nir" in nirj[k]]
        out["nirJ"] = {
            "n": int(len(dj["d_feh"])),
            "b": {x: float(np.median(v)) for x, v in dj.items()},
            "zp_J_median": float(np.median(zp)) if zp else None,
            "breaks": bool(
                abs(np.median(dj["d_feh"])) >= 0.05 or abs(np.median(dj["d_loga"])) >= 0.05
            ),
        }
    (HERE / "colour_summary_mb.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
