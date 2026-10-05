#!/usr/bin/env python3
"""Phase-by-phase fits on synthetic clusters of ONE age (hub finding ``isochrone-ms-vs-pms.md`` §0.4):
per-phase bias and coverage of log t, and the NULL distribution of Delta log t (MS - PMS) and of the
one-age-vs-two likelihood ratio, through exactly the pipeline the real data go through.

Generator: the grid layer's own forward model (``IsochroneFitter._draw_stars``: IMF, Offner binaries,
the real sample's error model and completeness cut), MIST v1.2, N = 254. The truth has a single age, so
any Delta log t between windows is estimator noise + estimator bias: that is the null.

Per replicate: (1) global fit on the 254 stars, ``common.search`` (coarse lattice + polish, parallax dm
prior, never seeded at the truth, §A.1.3); (2) per window, profile of log t with [Fe/H], dm, A_V fixed
at the global mode (arm ``a``) and with A_V free per window (arm ``b``); (3) the same profiles with the
shared parameters at the TRUTH (arm ``a_truth``: evaluating at the truth separates the estimator from
the global-fit error that feeds it; it is a diagnostic, not a seeded search).

Truths (pre-registered): ``T1`` = the real global fit R1 rounded (feh -0.47, log t 6.02, A_V 0.66,
sigma_int 0.15) -- the world the data look like; ``T2`` = the colour experiment's truth (feh 0, log t
6.40, A_V 1.0, sigma_int 0.03).

WHAT WOULD FALSIFY ITS USE: per-phase |median bias| > 0.10 dex in log t on arm ``a``, or a null Delta
log t median off zero by more than 0.10: then the phase fit is biased by construction and a real
Delta cannot be read.

Output: ``synth_<truth>.json`` (rewritten after every replicate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phases import (  # noqa: E402
    PRI_PLX,
    SAMPLE,
    WINDOWS,
    full_fitter,
    mist,
    one_age_vs_two,
    profile_loga,
    profile_summary,
    search,
    window_fitter,
)

TRUTHS = {
    "T1": {"feh": -0.47, "loga": 6.02, "Av": 0.66, "sigma_int": 0.15, "f_bg": 0.0},
    "T2": {"feh": 0.0, "loga": 6.40, "Av": 1.0, "sigma_int": 0.03, "f_bg": 0.0},
}
DM_MU, DM_SD, N_STARS = 10.223, 0.026, 254


def synth_table(gen, truth, dm, rng):
    from astropy.table import QTable

    G, C = gen._draw_stars(
        truth["feh"], truth["loga"], dm, truth["Av"], truth["sigma_int"], N_STARS, rng
    )
    return QTable(
        {
            "Gmag": G,
            "G_BPmag": C,
            "G_RPmag": np.zeros_like(G),
            "e_Gmag": gen._e_mag_fn(G),
            "e_G_BPmag": gen._e_col_fn(G),
            "e_G_RPmag": np.zeros_like(G),
            "e_BP_RP": gen._e_col_fn(G),
            "probability_hdbscan": np.ones_like(G),
        }
    )


def phase_profiles(grid, data, shared: dict, windows, arms=("a", "b")) -> dict:
    out = {}
    for w in windows:
        f = window_fitter(grid, data, WINDOWS[w])
        out[w] = {"n": int(f._N_obs)}
        for arm in arms:
            free = ("sigma_int", "f_bg") if arm.startswith("a") else ("Av", "sigma_int", "f_bg")
            p = profile_loga(f, shared, free=free)
            out[w][arm] = {"profile": p, "summary": profile_summary(p)}
        del f
    return out


def main() -> None:
    from astropy.table import QTable, Table

    import erotica

    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, choices=sorted(TRUTHS))
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--first", type=int, default=0)
    a = ap.parse_args()
    truth = TRUTHS[a.truth]
    grid = mist()
    real = QTable(Table.read(SAMPLE))
    gen = full_fitter(grid, real)  # the real sample's error model and completeness cut
    path = HERE / f"synth_{a.truth}.json"
    out = (
        json.loads(path.read_text())
        if path.exists() and a.first > 0
        else {
            "erotica_file": erotica.__file__,
            "truth": truth,
            "dm": [DM_MU, DM_SD],
            "windows": {k: list(v) for k, v in WINDOWS.items()},
            "priors": {k: list(v) if isinstance(v, tuple) else v for k, v in PRI_PLX.items()},
            "reps": [],
        }
    )
    for r in range(a.first, a.first + a.reps):
        rng = np.random.default_rng(5000 + r)  # same seeds for both truths
        dm = float(rng.normal(DM_MU, DM_SD))
        syn = synth_table(gen, truth, dm, rng)
        f = full_fitter(grid, syn)
        g = search(f, verbose=False)
        del f
        shared = {k: g["mode"][k] for k in ("feh", "dm", "Av")}
        at_truth = {"feh": truth["feh"], "dm": dm, "Av": truth["Av"]}
        rep = {
            "rep": r,
            "truth_dm": dm,
            "global": g,
            "n_window": {
                w: int(((syn["Gmag"] >= lo) & (syn["Gmag"] < (hi or 99))).sum())
                for w, (lo, hi) in WINDOWS.items()
            },
            "phases": phase_profiles(grid, syn, shared, ("ms", "pms", "ums", "to")),
            "phases_truth": phase_profiles(grid, syn, at_truth, ("ms", "pms"), arms=("a",)),
        }
        for key in ("phases", "phases_truth"):
            for arm in ("a", "b") if key == "phases" else ("a",):
                ph = rep[key]
                rep.setdefault("tension", {})[f"{key}:{arm}"] = one_age_vs_two(
                    ph["ms"][arm]["profile"], ph["pms"][arm]["profile"]
                )
        out["reps"].append(rep)
        s = {
            w: round(rep["phases"][w]["a"]["summary"]["loga_hat"] - truth["loga"], 3)
            for w in ("ms", "pms", "ums", "to")
        }
        print(
            a.truth,
            r,
            "global d_loga",
            round(g["mode"]["loga"] - truth["loga"], 3),
            "phase d_loga(a)",
            s,
            "tension",
            {k: round(v["delta_loga"], 3) for k, v in rep["tension"].items()},
            flush=True,
        )
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
