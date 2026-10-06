#!/usr/bin/env python3
"""NGC 6383, the 254 C1 members, CMD fits with and without the 3D-map A_V (hub finding
``isochrone-external-constraints.md`` §0.6, pre-registered; run only after ``av_synth.py``).

* ``X1``: the colour experiment's R1 (MIST v1.2, Offner binaries, parallax dm prior), re-run here to
  carry a Laplace width.
* ``X2``: the same, photometry corrected per star by the main map's Delta A_V (A_map,i minus its
  median), the map's sd (if any) added in quadrature to the errors; A_V level free.
* ``X3``: X2 + a Gaussian prior on the A_V level: N(median A_map, hypot(median sd_map, 0.10)).
  Reported as NOT APPLICABLE if the median map A_V is below X2's fitted level (the map sees only
  foreground dust: a lower bound used as a value).

Output: ``real_fit_<run>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from av_synth import fit_table, laplace, main_map, with_av_prior  # noqa: E402
from common import EDR3, PRI_PLX, SAMPLE, mist, search  # noqa: E402


def build(run: str):
    from astropy.table import QTable, Table

    import erotica
    from erotica.analysis._isochrone import IsochroneFitter

    real = QTable(Table.read(SAMPLE))
    per = Table.read(HERE / "av_maps_perstar.ecsv")
    assert np.array_equal(np.asarray(per["source_id"]), np.asarray(real["source_id"]))
    col, scol, mname = main_map(per)
    A = np.asarray(per[col], float)
    sA = np.asarray(per[scol], float) if scol else np.zeros_like(A)
    out = {
        "erotica_file": erotica.__file__,
        "run": run,
        "main_map": mname,
        "map_column": col,
        "sigma_column": scol,
        "priors": {k: list(v) if isinstance(v, tuple) else v for k, v in PRI_PLX.items()},
    }
    f0 = IsochroneFitter(grid=mist(), magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
    f0.setup(real, prob_threshold=0.0)
    if run == "X1":
        data, f = real, f0
    else:
        G = np.asarray(real["Gmag"], float)
        C = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
        data = fit_table(
            G, C, f0._e_obs_mag, f0._e_obs_col, A - np.median(A), sA, f0._kG, f0._k_col1, True
        )
        data["source_id"] = real["source_id"]
        f = IsochroneFitter(grid=mist(), magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
        f.setup(data, prob_threshold=0.0)
        out["dA_applied_p16_p84"] = np.percentile(A - np.median(A), [16, 84]).tolist()
        if run == "X3":
            mu, sd = float(np.median(A)), float(np.hypot(np.median(sA), 0.10))
            out["av_prior"] = {"mu": mu, "sd": sd}
            with_av_prior(f, mu, sd)
    return f, data, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["X1", "X2", "X3"])
    a = ap.parse_args()
    f, data, out = build(a.run)
    res = search(f, verbose=False)
    mo = res["mode"]
    res["laplace_sd"] = laplace(
        f, [mo[k] for k in ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")]
    )
    res["age_Myr"] = 10 ** mo["loga"] / 1e6
    # Mode and runner-ups with log-likelihood and log-posterior apart (§A.1.1, C1 §4): a small
    # loglike gap with a large logpost gap means the dm prior, not the photometry, picks the mode.
    # In X3 ``_compiled_loglike`` already carries the A_V prior (``with_av_prior`` wraps it).
    fl = f._compiled_loglike(None)
    best = [mo[k] for k in ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")]
    res["modes_loglike_logpost"] = [
        {"logpost": float(lp), "loglike": float(fl(p)[0]), "params": [float(v) for v in p]}
        for lp, p in [(res["logpost"], best), *[tuple(x) for x in res["runner_up"]]]
    ]
    out.update(res)
    if a.run == "X3":
        x2 = json.loads((HERE / "real_fit_X2.json").read_text())
        out["applicable"] = bool(out["av_prior"]["mu"] >= x2["mode"]["Av"])
    print(
        a.run,
        json.dumps(
            {k: out[k] for k in ("mode", "age_Myr", "at_bound", "laplace_sd", "seconds")},
            default=float,
        ),
        flush=True,
    )
    (HERE / f"real_fit_{a.run}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
