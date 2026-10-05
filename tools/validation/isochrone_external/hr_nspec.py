#!/usr/bin/env python3
"""How many PMS members with a spectroscopic T_eff would an HR-plane fit need? (hub finding
``isochrone-external-constraints.md`` §0.5, last paragraph; pre-registered.)

The (G, -10 log T_eff) path of ``isochrone_colour/hr_fit.py`` (MIST v1.2 G column, T_eff without a
colour table, no binaries), on synthetic stars drawn by that same fitter in the window
G 14-18 (PMS stars bright enough for R ~ 10^4 spectroscopy [I]), N = 10 / 20 / 40 / 80,
sigma_T = 100 K per star, [Fe/H] free, R = 8 replicates per N. Truth: [Fe/H] 0, log t 6.40,
A_V 1.0, dm ~ N(10.223, 0.026).

Two A_V treatments, because in the HR plane A_V only shifts G (k_G A_V), i.e. log L, and nothing
else constrains it inside a PMS-only window: ``free`` (A_V in [0, 3]) and ``prior`` (N(1.0, 0.1):
the level known from elsewhere, e.g. the upper MS or a map).

The generator's error model in T is fitted on the 202 real members with GSP-Phot T_eff, with
e_T = 10 * 100 K / (T ln 10) at each star's GSP-Phot T (only the G dependence of the error is taken
from the data). Output: ``hr_nspec_<mode>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent / "isochrone_colour"))
sys.path.insert(0, str(HERE))
from av_synth import laplace, with_av_prior  # noqa: E402
from common import PRI_PLX, SAMPLE, mist, search  # noqa: E402
from hr_fit import _f, hr_view  # noqa: E402

TRUTH = {"feh": 0.0, "loga": 6.40, "Av": 1.0, "dm_mu": 10.223, "dm_sd": 0.026}
WINDOW = (14.0, 18.0)
SIGMA_T = 100.0


def fitter(data):
    from erotica.analysis._isochrone import IsochroneFitter

    f = IsochroneFitter(
        grid=hr_view(mist()),
        obs_columns=("Gmag", "T", "Z"),
        obs_error_columns=("e_Gmag", "e_T", "e_Z"),
        magnitude="G",
        magnitude_effl=6390.7,
        color=("T", "zero"),
        color_effl=(6390.7, 6390.7),
        alpha=0.0,
        beta=0.0,
        **PRI_PLX,
    )
    f.setup(data, prob_threshold=0.0, mag_window=WINDOW)
    return f


def table(G, T, eG, eT):
    from astropy.table import QTable

    return QTable(
        {
            "Gmag": G,
            "T": T,
            "Z": np.zeros_like(G),
            "e_Gmag": eG,
            "e_T": eT,
            "e_Z": np.zeros_like(G),
            "e_BP_RP": eT,
            "probability_hdbscan": np.ones_like(G),
        }
    )


def main():
    from astropy.table import Table, join

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["free", "prior"])
    ap.add_argument("--ns", default="10,20,40,80")
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()

    real = Table.read(SAMPLE)
    gs = Table.read(HERE.parent / "isochrone_colour/gspphot_ngc6383.ecsv")
    gs.remove_column("source_id")
    gs.rename_column("sid", "source_id")
    t = join(real, gs, keys="source_id")
    Tk = _f(t["teff_gspphot"])
    ok = np.isfinite(Tk)
    t, Tk = t[ok], Tk[ok]
    seed = table(
        _f(t["Gmag"]), -10 * np.log10(Tk), _f(t["e_Gmag"]), 10 * SIGMA_T / (Tk * np.log(10))
    )
    gen = fitter(seed)
    out = {
        "truth": TRUTH,
        "window": WINDOW,
        "sigma_T": SIGMA_T,
        "mode": a.mode,
        "n_seed": int(len(seed)),
        "runs": [],
    }
    path = HERE / f"hr_nspec_{a.mode}.json"
    for n in map(int, a.ns.split(",")):
        for k in range(a.reps):
            rng = np.random.default_rng(3000 + 100 * n + k)
            dm = float(rng.normal(TRUTH["dm_mu"], TRUTH["dm_sd"]))
            G, T = gen._draw_stars(TRUTH["feh"], TRUTH["loga"], dm, TRUTH["Av"], 0.0, n, rng)
            f = fitter(table(G, T, gen._e_mag_fn(G), gen._e_col_fn(G)))
            if a.mode == "prior":
                with_av_prior(f, TRUTH["Av"], 0.10)
            res = search(f, verbose=False)
            mo = res["mode"]
            res["laplace_sd"] = laplace(
                f, [mo[q] for q in ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")]
            )
            res.update(
                {
                    "N": n,
                    "rep": k,
                    "d_loga": mo["loga"] - TRUTH["loga"],
                    "d_feh": mo["feh"] - TRUTH["feh"],
                    "d_Av": mo["Av"] - TRUTH["Av"],
                }
            )
            out["runs"].append(res)
            print(
                a.mode,
                n,
                k,
                round(res["d_loga"], 3),
                round(res["d_feh"], 3),
                round(res["d_Av"], 3),
                res["at_bound"],
                res["seconds"],
                flush=True,
            )
            path.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
