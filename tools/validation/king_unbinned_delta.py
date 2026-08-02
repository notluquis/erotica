#!/usr/bin/env python3
r"""Delta report: published binned King fit vs the unbinned point-process fit.

WHY THIS EXISTS
---------------
P01 reports ``R_c`` and ``R_t`` for NGC 6383 from ``structure.RDP_bayesian``: a
Normal likelihood on *equal-count* binned densities, with a single shared scatter
parameter and all four prior bounds derived from the data being fit. Three
separate defects, each independently a referee target. ``structure.king_unbinned``
replaces all of them at once. This script measures what that costs in published
numbers, before anything is rewritten.

TWO SAMPLES, AND WHY BOTH
-------------------------
1. **The published sample** -- ``paperfaithful_reference_p06.ecsv``, the p > 0.6
   member list the paper actually fitted (N = 628 inside 70'). Apples-to-apples
   with the published number.
2. **The full cone** -- every Gaia source inside the same 70' radius. This is the
   sample the model is *for*: the King background ``b`` only has something
   physical to absorb if field stars are present, and the unbinned normalisation
   ``\\int lambda dA`` is only correct if the footprint really is that disc.

Fitting a free background to an already-selected member list is unsound
regardless of likelihood -- ``b`` soaks up membership leakage and is degenerate
with ``k`` and ``R_t``. Running both quantifies how much that matters.

USAGE
-----
    python tools/validation/king_unbinned_delta.py
    python tools/validation/king_unbinned_delta.py --skip-cone   # members only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import king_unbinned

B = Path("/Users/notluquis/erotica/data/test/NGC6383")
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
FIELD = 70.0  # arcmin
MEMBERS = B / "comments_paper/radius_robustness/generated/70/paperfaithful_reference_p06.ecsv"
CONE = B / "data/70/NGC_6383_70-result.ecsv"
PUBLISHED_TRACE = B / "comments_paper/review_repo/idata_king_cone70_modpriors.nc"


def radii_arcmin(table, ra="ra", dec="dec"):
    for a, d in ((ra, dec), ("RA_ICRS", "DE_ICRS"), ("ra_epoch2000", "dec_epoch2000")):
        if a in table.colnames and d in table.colnames:
            sky = SkyCoord(np.asarray(table[a], float) * u.deg, np.asarray(table[d], float) * u.deg)
            return CENTER.separation(sky).to(u.arcmin).value
    raise KeyError(f"no sky columns found in {table.colnames[:12]}")


def published():
    """Posterior medians from the trace the paper reports."""
    import arviz as az

    post = az.from_netcdf(PUBLISHED_TRACE).posterior
    return {
        p: (float(np.median(post[p].values)), float(np.std(post[p].values)))
        for p in ("k", "b", "R_c", "R_t")
    }


def fit(radii, label, *, draws=2000, seed=20260727):
    keep = radii[np.isfinite(radii) & (radii > 0) & (radii <= FIELD)]
    dropped = len(radii) - len(keep)
    t0 = time.perf_counter()
    res = king_unbinned(
        keep,
        field_radius=FIELD,
        sampling=SamplingConfig(
            draws=draws, tune=1000, chains=4, random_seed=seed, progressbar=False
        ),
    )
    import arviz as az
    import pandas as pd

    smry = az.summary(res["king_trace"], var_names=["k", "b", "R_c", "R_t"])
    out = {
        "label": label,
        "n_stars": int(len(keep)),
        "dropped_outside_field": int(dropped),
        "seconds": round(time.perf_counter() - t0, 1),
        "r_hat_max": float(pd.to_numeric(smry["r_hat"], errors="coerce").max()),
        "ess_bulk_min": float(pd.to_numeric(smry["ess_bulk"], errors="coerce").min()),
        "divergences": int(res["king_trace"].sample_stats["diverging"].values.sum()),
    }
    for p in ("k", "b", "R_c", "R_t"):
        m, s = res[f"{p}_median"], res[f"{p}_std"]
        out[p] = (float(getattr(m, "value", m)), float(getattr(s, "value", s)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-cone", action="store_true")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()

    pub = published()
    results = [fit(radii_arcmin(Table.read(MEMBERS)), "unbinned / published member list",
                   draws=args.draws)]
    if not args.skip_cone:
        print(f"reading the full cone ({CONE.stat().st_size / 1e6:.0f} MB) ...", flush=True)
        results.append(fit(radii_arcmin(Table.read(CONE)), "unbinned / full 70' cone",
                           draws=args.draws))

    print(f"\n{'fit':34s} {'N':>7s} {'R_c (arcmin)':>18s} {'R_t (arcmin)':>18s} {'b':>12s}")
    print(f"{'published (binned, equal-count)':34s} {628:>7d} "
          f"{pub['R_c'][0]:11.3f}+/-{pub['R_c'][1]:5.3f} "
          f"{pub['R_t'][0]:11.3f}+/-{pub['R_t'][1]:5.3f} {pub['b'][0]:12.4f}")
    for r in results:
        print(f"{r['label']:34s} {r['n_stars']:>7d} "
              f"{r['R_c'][0]:11.3f}+/-{r['R_c'][1]:5.3f} "
              f"{r['R_t'][0]:11.3f}+/-{r['R_t'][1]:5.3f} {r['b'][0]:12.4f}")

    print(f"\n{'fit':34s} {'dR_c':>10s} {'dR_t':>10s}   convergence")
    for r in results:
        d_rc = (r["R_c"][0] - pub["R_c"][0]) / pub["R_c"][0]
        d_rt = (r["R_t"][0] - pub["R_t"][0]) / pub["R_t"][0]
        print(f"{r['label']:34s} {d_rc:+9.1%} {d_rt:+9.1%}   "
              f"r_hat<={r['r_hat_max']:.3f} ess>={r['ess_bulk_min']:.0f} "
              f"div={r['divergences']} ({r['seconds']}s)")

    args.out.write_text(json.dumps({"published": pub, "unbinned": results}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
