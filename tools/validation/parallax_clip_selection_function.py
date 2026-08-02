#!/usr/bin/env python3
r"""Measure the parallax sigma-clip as a magnitude-dependent selection function.

WHY THIS EXISTS
---------------
``analysis/_clipping.sigma_clip_parallax`` applies a robust sigma clip to the *raw*
parallax column. Gaia parallax uncertainty is strongly magnitude dependent
(sigma_plx grows by an order of magnitude from G~14 to G~20), so a cut placed at a
fixed multiple of the *sample* scatter necessarily rejects faint stars
preferentially -- regardless of whether they are members. That makes it a
selection function, not only an outlier test.

This matters because P01 attributes a faint-quartile luminosity-function / KS
signal to *Gaia* incompleteness. If the clip itself removes faint members at a
different rate than bright ones, that attribution needs a control.

CORRECTION 2026-07-27 -- READ BEFORE QUOTING THE GRADIENT
---------------------------------------------------------
This script's oracle draws synthetic member parallaxes with **no bound**, but the
real cone was queried with a parallax constraint: the raw file spans 0.750-1.100
mas and every one of the 78893 raw sources lies inside it. A faint-quartile member
(median e_Plx = 0.293 mas) scatters far outside that window, so in reality it never
enters the catalogue at all and the clip never sees it.

The +72.4 pp gradient this script reports is therefore **the query window plus the
clip**, not the clip. Measured separately over 300 realizations:

    stage                              overall   Q1     Q4     gradient
    clip alone, unbounded (this script)  67.6%  99.1%  26.7%   +72.4 pp
    ADQL query window alone              80.6%  99.8%  43.0%   +56.7 pp
    query window THEN clip (reality)     64.6%  98.6%  24.1%   +74.5 pp

The query window alone is 78% of the effect. The total is real and slightly larger
than reported here, but it is dominated by a data-acquisition choice, not by the
clip. See the decision log.

THE ORACLE
----------
By construction: a synthetic cluster is generated in which **every star is a true
member**. There is no contamination and no intrinsic depth beyond an optional
common spread. Therefore *any* rejection is a false rejection, and the retention
rate per magnitude quartile measures the induced selection function directly. No
golden numbers are involved.

Per-star uncertainties are the **real** ``e_Plx`` values from the published NGC 6383
member catalogue, so the magnitude-dependence of the errors is empirical rather
than modelled.

USAGE
-----
    python tools/validation/parallax_clip_selection_function.py
    python tools/validation/parallax_clip_selection_function.py --n-realizations 400

Writes a JSON summary next to the script for downstream citation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.stats import biweight_location, biweight_scale
from astropy.table import QTable

from erotica.analysis._clipping import sigma_clip_parallax

MEMBERS = Path(
    "/Users/notluquis/erotica/data/test/NGC6383/comments_paper/cds_final/ngc6383_members.ecsv"
)
TRUE_PARALLAX = 0.90  # mas; ~1.11 kpc, the published NGC 6383 distance regime


def _load_real_errors():
    """Real per-star (e_Plx, Gmag) from the published member table."""
    t = QTable.read(MEMBERS)
    err = np.asarray(t["e_Plx"], dtype=float)
    mag = np.asarray(t["Gmag"], dtype=float)
    ok = np.isfinite(err) & np.isfinite(mag) & (err > 0)
    return err[ok], mag[ok]


def _clip_raw(plx, mag, sigma):
    """The pipeline's actual cut: sigma clip on the raw parallax column."""
    tab = QTable({"parallax": plx, "cluster": np.zeros(len(plx), dtype=np.int64)})
    _lo, _hi, _copy, keep, _noise = sigma_clip_parallax(
        tab, cluster=0, sigma=sigma, use_biweight=True, in_place=False,
        mark_label=-1, print_results=False, return_mask=True, preselector_mask=None,
    )
    return np.asarray(keep, dtype=bool)


def _clip_normalized(plx, err, sigma):
    """Clip on the normalized residual (plx - centre) / err_i.

    This is the error-aware alternative: a star is an outlier only if it is
    discrepant *relative to its own uncertainty*, so a large error buys tolerance
    instead of a rejection.
    """
    centre = biweight_location(plx)
    z = (plx - centre) / err
    scale = biweight_scale(z)
    return np.abs(z - biweight_location(z)) <= sigma * scale


def run(n_real=400, sigma=2.0, seed=20260727, intrinsic=0.0):
    err, mag = _load_real_errors()
    n = len(err)
    rng = np.random.default_rng(seed)
    q = np.digitize(mag, np.quantile(mag, [0.25, 0.5, 0.75]))  # 0=brightest .. 3=faintest

    keep_raw = np.zeros((n_real, n), dtype=bool)
    keep_norm = np.zeros((n_real, n), dtype=bool)
    for i in range(n_real):
        truth = TRUE_PARALLAX + (rng.normal(0, intrinsic, n) if intrinsic else 0.0)
        obs = truth + rng.normal(0, err)          # every star IS a member
        keep_raw[i] = _clip_raw(obs, mag, sigma)
        keep_norm[i] = _clip_normalized(obs, err, sigma)

    def summarize(keep):
        overall = float(keep.mean())
        per_q = [float(keep[:, q == k].mean()) for k in range(4)]
        return overall, per_q

    raw_overall, raw_q = summarize(keep_raw)
    norm_overall, norm_q = summarize(keep_norm)

    return {
        "n_stars": int(n),
        "n_realizations": int(n_real),
        "sigma": sigma,
        "seed": seed,
        "intrinsic_spread_mas": intrinsic,
        "median_e_plx_by_quartile": [float(np.median(err[q == k])) for k in range(4)],
        "median_gmag_by_quartile": [float(np.median(mag[q == k])) for k in range(4)],
        "raw_clip": {"overall_retention": raw_overall, "retention_by_gmag_quartile": raw_q},
        "normalized_clip": {"overall_retention": norm_overall, "retention_by_gmag_quartile": norm_q},
        "gradient_raw": float(raw_q[0] - raw_q[3]),
        "gradient_normalized": float(norm_q[0] - norm_q[3]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-realizations", type=int, default=400)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--intrinsic", type=float, default=0.0, help="intrinsic depth, mas")
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()

    res = run(n_real=args.n_realizations, sigma=args.sigma, intrinsic=args.intrinsic)

    print(f"NGC 6383 members: {res['n_stars']} stars, {res['n_realizations']} realizations")
    print("Every simulated star is a TRUE MEMBER -> any rejection is a false rejection.\n")
    print(f"{'':22s} {'overall':>8s} {'Q1 bright':>10s} {'Q2':>7s} {'Q3':>7s} {'Q4 faint':>9s}")
    for label, key in (("raw parallax (current)", "raw_clip"), ("normalized residual", "normalized_clip")):
        r = res[key]
        qs = r["retention_by_gmag_quartile"]
        print(
            f"{label:22s} {r['overall_retention']:7.1%} "
            f"{qs[0]:9.1%} {qs[1]:6.1%} {qs[2]:6.1%} {qs[3]:8.1%}"
        )
    print(f"\nmedian e_Plx by quartile (mas): "
          f"{['%.3f' % v for v in res['median_e_plx_by_quartile']]}")
    print(f"bright->faint retention gradient: raw {res['gradient_raw']:+.1%}, "
          f"normalized {res['gradient_normalized']:+.1%}")

    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
