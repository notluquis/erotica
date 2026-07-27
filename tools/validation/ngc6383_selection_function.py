#!/usr/bin/env python3
"""Build the radial completeness profile S̄(r) for NGC 6383 from the Gaia DR3
selection function, for use as ``king_unbinned(completeness=...)``.

WHY THIS EXISTS
---------------
``structure.king_unbinned`` can fold a radial detection probability into the
point-process normalisation, and on synthetic data ignoring one inflates ``R_c``
by 50% and halves the central density. PART J notes that a search for
selection-function-corrected open-cluster radial density profiles returns **zero
papers**. This script produces the real S̄(r) for NGC 6383 so the correction can
be applied rather than assumed.

READ THIS BEFORE TRUSTING A FLAT ANSWER
---------------------------------------
``mode='hpx7'`` **cannot answer this question**. Its healpix pixels are ~27.5
arcmin across; NGC 6383's core radius is ~1.38 arcmin, so the core is 0.25% of a
single pixel and the whole 70 arcmin field spans 2.55 pixels. Radial structure on
cluster scales is below the map's resolution *by construction*, so a flat S̄(r)
from hpx7 means **"invisible to this map"**, not "absent". Running it that way
first, and only then checking the pixel size, is how one publishes a
completeness correction that corrects nothing.

``mode='patch'`` builds a local map at healpix order 6-12 (down to ~0.86 arcmin),
which does resolve the core. It requires a live query to the ESA Gaia archive.

WHAT THE hpx7 PASS DID ESTABLISH (resolution-independent)
---------------------------------------------------------
The selection function is correctly magnitude-sensitive at this position:

    G      10     14     17     19     20    20.7    21    21.5    22
    S    1.000  1.000  1.000  1.000  0.996  0.852  0.376  0.0002  0.000

NGC 6383's members have median G = 17.3 and a 98th percentile near G = 20.4, so
they sit where Gaia DR3 source completeness is essentially unity. **Magnitude
incompleteness is not the dominant selection acting on this sample.** By
comparison, the pipeline's own 2-sigma parallax clip retains 99% of the brightest
Gmag quartile and 27% of the faintest (see
``parallax_clip_selection_function.py``) -- an effect an order of magnitude
larger than anything Gaia does in this field.

USAGE
-----
    python tools/validation/ngc6383_selection_function.py --mode patch
    python tools/validation/ngc6383_selection_function.py --mode hpx7  # diagnostic only

Writes ``ngc6383_selection_function_<mode>.npz`` with ``r_arcmin`` aligned to the
256-node Gauss-Legendre grid that ``king_expected_count_weighted`` uses, so the
array can be passed straight through as ``completeness=``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

B = Path("/Users/notluquis/erotica/data/test/NGC6383")
MEMBERS = B / "comments_paper/radius_robustness/generated/70/paperfaithful_reference_p06.ecsv"
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
FIELD_ARCMIN = 70.0
NODES = 256  # must match king_expected_count_weighted


def quadrature_radii(field_radius=FIELD_ARCMIN, nodes=NODES):
    """The Gauss-Legendre nodes the weighted normalisation evaluates on."""
    x, _ = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * field_radius * (x + 1.0)


def build(mode="patch", n_az=96, n_mag=24, min_points=20, patch_radius_deg=1.4):
    from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG

    table = Table.read(MEMBERS)
    gmag = np.asarray(table["Gmag"], dtype=float)
    gmag = gmag[np.isfinite(gmag)]
    print(f"members N={gmag.size}  Gmag median {np.median(gmag):.2f}  "
          f"p98 {np.percentile(gmag, 98):.2f}", flush=True)

    if mode == "patch":
        print(f"building patch-mode SF (order 6-12, min_points={min_points}) -- "
              "this queries the ESA Gaia archive", flush=True)
        sf = DR3SelectionFunctionTCG(
            mode="patch", coord=CENTER, radius=patch_radius_deg, min_points=min_points
        )
    else:
        print(f"building {mode}-mode SF (DIAGNOSTIC ONLY -- see module docstring)", flush=True)
        sf = DR3SelectionFunctionTCG(mode=mode)

    # S̄(r): mean over azimuth at each radius, marginalised over the observed
    # member magnitude distribution via an equally-weighted quantile grid.
    mag_grid = np.quantile(gmag, np.linspace(0.02, 0.98, n_mag))
    radii = quadrature_radii()
    azimuth = np.linspace(0.0, 2 * np.pi, n_az, endpoint=False)
    completeness = np.empty(radii.size)
    for i, r in enumerate(radii):
        points = CENTER.directional_offset_by(azimuth * u.rad, (r * u.arcmin).to(u.deg))
        per_mag = [
            np.asarray(sf.query(points, np.full(n_az, m)), dtype=float) for m in mag_grid
        ]
        completeness[i] = float(np.nanmean(per_mag))
        if i % 64 == 0:
            print(f"  r={r:6.2f}'  S={completeness[i]:.4f}", flush=True)
    return radii, completeness, mag_grid


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", default="patch", choices=["patch", "multi", "hpx7"])
    ap.add_argument("--n-az", type=int, default=96)
    ap.add_argument("--n-mag", type=int, default=24)
    ap.add_argument("--min-points", type=int, default=20)
    args = ap.parse_args()

    radii, completeness, mag_grid = build(
        mode=args.mode, n_az=args.n_az, n_mag=args.n_mag, min_points=args.min_points
    )

    inner = completeness[radii <= 5.0]
    outer = completeness[radii >= 50.0]
    print(f"\nS(r): min={completeness.min():.4f} max={completeness.max():.4f}")
    print(f"  mean inside  5' = {inner.mean():.4f}")
    print(f"  mean beyond 50' = {outer.mean():.4f}")
    print(f"  core suppression = {1 - inner.mean() / outer.mean():+.2%}")
    if args.mode != "patch":
        print("\n  *** hpx7/multi cannot resolve the cluster: pixels are ~27.5 arcmin")
        print("  *** against R_c = 1.38 arcmin. A flat result here is uninformative. ***")

    out = Path(__file__).with_name(f"ngc6383_selection_function_{args.mode}.npz")
    np.savez(out, r_arcmin=radii, completeness=completeness, mag_grid=mag_grid,
             n_az=args.n_az, field_radius=FIELD_ARCMIN, mode=args.mode,
             min_points=args.min_points)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
