#!/usr/bin/env python3
"""Experiment 2 of the hub finding ``isochrone-colour-mechanism.md`` (pre-registered there, §0.3):
NGC 6383 fitted in the (G, T_eff) plane, where no colour table stands between the grid and the
stars, against the CMD on the same stars.

Runs (``--run``):

* ``R1``: CMD, the 254 C1 members, Offner binaries -- the grid layer's reference fit with the
  parallax dm prior (``common.PRI_PLX``).
* ``R2``: CMD, the members with ``teff_gspphot`` (202), no binaries -- the like-for-like control of
  R3/R4.
* ``R3``: (G, -10 log T_eff) with raw GSP-Phot T_eff, same 202, no binaries.
* ``R4``: the same with GSP-Phot minus its bias measured on lambda Ori (Cao+22 spectroscopic T_eff),
  binned in log T_gsp, the lambda Ori robust scatter added to the error. lambda Ori has A_V ~ 0.3 and
  NGC 6383 ~ 1-1.5: the calibration's transfer in extinction is a declared limit, not measured.

Process behind each piece: G carries dm + k_G A_V (one A_V, anchored by the upper-MS stars once dm
is fixed by the parallax); the "colour" column is -10 log T_eff with zero extinction coefficient
(both colour wavelengths equal, so k_c1 - k_c2 = 0); the grid's BC_G is still used (G column of
MIST). Binaries are off in R2-R4 because adding -10 log T of two stars as magnitudes is not a
physical combination (the lambda Ori HR diagnostic D2 had them off too).

Output: ``hr_fit_<run>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import CACHE, EDR3, PRI_PLX, SAMPLE, mist, search  # noqa: E402

LT_BINS = np.log10([3000, 3400, 3700, 4000, 4400, 5000, 6500])


def _f(c):
    return np.asarray(c.filled(np.nan) if hasattr(c, "filled") else c, float)


def lori_calibration() -> dict:
    """Median and robust scatter of log T_gsp - log T_Cao in bins of log T_gsp (what is observed
    in NGC 6383), all Cao+22 stars with a GSP-Phot value."""
    from astropy.table import Table, join

    lo = Table.read(CACHE / "oracles/lambda_ori/lambda_ori_2mass_dr3.ecsv")
    g = Table.read(HERE / "gspphot_lambda_ori.ecsv")
    g.rename_column("sid", "Source")
    j = join(lo, g, keys="Source")
    tg, tc = np.log10(_f(j["teff_gspphot"])), _f(j["logTeff"])
    ok = np.isfinite(tg) & np.isfinite(tc)
    tg, d = tg[ok], (tg - tc)[ok]
    rows = []
    for a, b in zip(LT_BINS[:-1], LT_BINS[1:], strict=True):
        s = (tg >= a) & (tg < b)
        if s.sum() >= 5:
            med = float(np.median(d[s]))
            rows.append(
                {
                    "logt_gsp_mid": float(np.median(tg[s])),
                    "n": int(s.sum()),
                    "bias": med,
                    "scatter": float(1.4826 * np.median(np.abs(d[s] - med))),
                }
            )
    return {"n": int(ok.sum()), "bins": rows}


def hr_view(g):
    """The grid with bands (G, T, zero): T = -10 log T_eff, so the fitter's colour is T - 0."""
    import copy

    from erotica.analysis.grids.base import GridNode

    h = copy.copy(g)
    h._nodes = {
        k: GridNode(
            n.eep,
            n.mass,
            {"G": n.mags[EDR3[0]], "T": -10.0 * n.extra["logte"], "zero": 0.0 * n.mass},
            n.extra,
        )
        for k, n in g._nodes.items()
    }
    h.bands = h.default_bands = ("G", "T", "zero")
    h.name = g.name + " [G-Teff]"
    return h


def build(run: str):
    """The fitter, its data table and the run metadata for ``run`` (R1-R4), set up and ready;
    ``export_run.py colour`` rebuilds the same fitter from here."""
    from astropy.table import QTable, Table, join

    import erotica
    from erotica.analysis._isochrone import IsochroneFitter

    real = Table.read(SAMPLE)
    gs = Table.read(HERE / "gspphot_ngc6383.ecsv")
    gs.remove_column("source_id")  # null where DR3 has no AP row; the uploaded id is `sid`
    gs.rename_column("sid", "source_id")
    t = join(real, gs, keys="source_id", join_type="left")
    assert len(t) == len(real)
    lt = np.log10(_f(t["teff_gspphot"]))
    has = np.isfinite(lt)
    out = {
        "erotica_file": erotica.__file__,
        "run": run,
        "priors": {k: list(v) if isinstance(v, tuple) else v for k, v in PRI_PLX.items()},
    }
    g = mist()
    if run in ("R1", "R2"):
        data = QTable(t if run == "R1" else t[has])
        kw = {} if run == "R1" else {"alpha": 0.0, "beta": 0.0}
        f = IsochroneFitter(grid=g, magnitude=EDR3[0], color=EDR3[1:], **kw, **PRI_PLX)
    else:
        s = t[has]
        lt_s = lt[has]
        e_lt = 0.5 * (np.log10(_f(s["teff_gspphot_upper"])) - np.log10(_f(s["teff_gspphot_lower"])))
        if run == "R4":
            cal = lori_calibration()
            x = np.array([b["logt_gsp_mid"] for b in cal["bins"]])
            bias = np.interp(lt_s, x, [b["bias"] for b in cal["bins"]])
            sc = np.interp(lt_s, x, [b["scatter"] for b in cal["bins"]])
            hot = lt_s > x[-1]  # no lambda Ori calibration above its hottest bin: left raw
            bias[hot], sc[hot] = 0.0, 0.0
            lt_s = lt_s - bias
            e_lt = np.hypot(e_lt, sc)
            out["calibration"] = cal
            out["n_uncalibrated_hot"] = int(hot.sum())
        data = QTable(
            {
                "source_id": s["source_id"],
                "Gmag": _f(s["Gmag"]),
                "T": -10.0 * lt_s,
                "Z": np.zeros(len(s)),
                "e_Gmag": _f(s["e_Gmag"]),
                "e_T": 10.0 * e_lt,
                "e_Z": np.zeros(len(s)),
                "e_BP_RP": 10.0 * e_lt,
                "probability_hdbscan": np.ones(len(s)),
            }
        )
        out["logteff_used"] = lt_s.tolist()
        f = IsochroneFitter(
            grid=hr_view(g),
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
    f.setup(data, prob_threshold=0.0)
    return f, data, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["R1", "R2", "R3", "R4"])
    a = ap.parse_args()
    f, data, out = build(a.run)
    assert a.run in ("R1", "R2") or abs(f._k_col1) < 1e-12
    res = search(f, verbose=False)
    res["age_Myr"] = 10 ** res["mode"]["loga"] / 1e6
    out.update(res)
    out["source_id"] = [int(x) for x in np.asarray(data["source_id"])]
    print(
        a.run,
        json.dumps(
            {k: out[k] for k in ("mode", "age_Myr", "at_bound", "n", "seconds")}, default=float
        ),
        flush=True,
    )
    (HERE / f"hr_fit_{a.run}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
