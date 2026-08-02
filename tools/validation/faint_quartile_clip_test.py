#!/usr/bin/env python3
r"""Is the faint-quartile radial deficit Gaia incompleteness, or the pipeline's own parallax clip?

WHY THIS EXISTS — IT IS A REFEREE-GRADE OBJECTION TO A PUBLISHED RESULT
-----------------------------------------------------------------------
P01 reports that the faintest magnitude quartile of NGC 6383's members is less centrally
concentrated, tests it with two-sample KS against the three brighter quartiles, and concludes:

    *"We therefore attribute the apparent deficit of faint stars toward the crowded cluster centre to
    Gaia incompleteness at G ≳ 18 mag rather than to a dynamical effect."*

The supporting argument is that the contrast is **not** strongest against the brightest quartile,
which is what dynamical mass segregation would produce.

**But the pipeline supplies a competing explanation of the same signal, and the paper does not
address it.** Membership selection applies a 2σ clip on the *raw* parallax, and that clip is strongly
magnitude-dependent: measured in ``parallax_clip_selection_function.py``, it retains 99% of the
brightest magnitude quartile and 27% of the faintest, with a retention gradient across quartiles of
**0.724**. Q4 is exactly the quartile the clip mutilates. If the surviving faint stars are also
radially selected, the deficit is manufactured by the pipeline rather than by the survey.

This was recorded in ``docs/design-notes/decisions.md`` as *"Estimated ~1 h. Not yet run"* and left
un-run — see ``methodology.md`` K.1.8 on why a TODO written in prose goes unseen.

THE DISCRIMINATING TEST
-----------------------
Re-run the same KS comparisons on a sample selected by a **normalised-residual** clip,
:math:`|\varpi - \varpi_0| / \sigma_\varpi < 2`, instead of a clip on the raw parallax. The
normalised clip is magnitude-blind by construction: a faint star with a large ``parallax_error`` is
no longer cut for being faint. Measured retention gradient across magnitude quartiles: **−0.002**,
against 0.724 for the raw clip.

**How to read the outcome — decided before running, because both directions are informative:**

* **Signal survives** → the incompleteness attribution is *strengthened*. The deficit is not an
  artefact of the clip, and P01's conclusion stands on firmer ground than it did.
* **Signal disappears** → the deficit is the pipeline's own selection, the attribution to Gaia
  incompleteness is unsupported, and P01 needs a correction.

Neither claim may be made without this test, which is why it blocks.

WHAT IS VERIFIED FIRST
----------------------
The published numbers are reproduced exactly before anything is changed — quartile edges
8.80 / 15.44 / 16.99 / 17.93 / 20.66 mag and ``D`` = 0.156, 0.230, 0.278 with ``p`` = 0.418, 0.052,
0.0097 against the paper's 0.16, 0.23, 0.28 and 0.42, 0.052, 0.010. Without that, a change in the
result could not be attributed to the change in the clip.

USAGE
-----
    python tools/validation/faint_quartile_clip_test.py

Writes ``faint_quartile_clip_test.json``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

BASE = Path("/Users/notluquis/erotica/data/test/NGC6383")
TABLE = BASE / "comments_paper/radius_robustness/generated/40/paperfaithful_with_clip_flags.ecsv"
CENTRE = (263.6826, -32.5838)  # deg, ICRS
SIGMA = 2.0
PROBABILITY_CUT = 0.6

# The published values, for the reproduction check. Order: faintest vs brightest, then inward.
PUBLISHED = [
    dict(against="8.80-15.44", D=0.16, p=0.42),
    dict(against="15.44-16.99", D=0.23, p=0.052),
    dict(against="16.99-17.93", D=0.28, p=0.010),
]


def radii(table):
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    centre = SkyCoord(CENTRE[0] * u.deg, CENTRE[1] * u.deg)
    return centre.separation(
        SkyCoord(np.asarray(table["ra"]) * u.deg, np.asarray(table["dec"]) * u.deg)
    ).to(u.arcmin).value


def quartile_ks(gmag, radius):
    """Faintest quartile against each brighter one, exactly as P01 does it."""
    from scipy.stats import ks_2samp

    edges = np.percentile(gmag, [25, 50, 75])
    masks = [
        gmag <= edges[0],
        (gmag > edges[0]) & (gmag <= edges[1]),
        (gmag > edges[1]) & (gmag <= edges[2]),
        gmag > edges[2],
    ]
    faint = radius[masks[3]]
    out = []
    for mask in masks[:3]:
        test = ks_2samp(faint, radius[mask])
        out.append(
            dict(
                against=f"{gmag[mask].min():.2f}-{gmag[mask].max():.2f}",
                n=int(mask.sum()),
                D=float(test.statistic),
                p=float(test.pvalue),
            )
        )
    return dict(
        edges=[float(gmag.min()), *map(float, edges), float(gmag.max())],
        n_faint=int(masks[3].sum()),
        median_radius_faint=float(np.median(faint)),
        comparisons=out,
    )


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values, which is the correction P01 applies."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main():
    warnings.filterwarnings("ignore")
    from astropy.table import Table

    table = Table.read(TABLE)
    gmag_all = np.asarray(table["Gmag"], dtype=float)
    radius_all = radii(table)
    probability = np.asarray(table["probability"], dtype=float)
    parallax = np.asarray(table["parallax"], dtype=float)
    parallax_error = np.asarray(table["parallax_error"], dtype=float)
    preclip = np.asarray(table["paper_reference_preclip_p05"], dtype=bool)
    published_mask = np.asarray(table["paper_reference_p06"], dtype=bool)

    results = {}

    # --- 1. reproduce ---------------------------------------------------------------------
    published = quartile_ks(gmag_all[published_mask], radius_all[published_mask])
    published["n_total"] = int(published_mask.sum())
    results["published"] = published
    print(f"PUBLISHED SAMPLE  N = {published['n_total']}")
    print(f"  quartile edges: " + " | ".join(f"{e:.2f}" for e in published["edges"]))
    print(f"  {'against':>14s} {'n':>4s} {'D':>7s} {'p':>8s}   paper")
    ok = True
    for got, want in zip(published["comparisons"], PUBLISHED):
        agrees = abs(got["D"] - want["D"]) < 0.006 and abs(got["p"] - want["p"]) < 0.002
        ok &= agrees
        print(f"  {got['against']:>14s} {got['n']:4d} {got['D']:7.3f} {got['p']:8.4f}   "
              f"D={want['D']} p={want['p']}  {'OK' if agrees else 'MISMATCH'}")
    results["reproduces_published"] = bool(ok)
    if not ok:
        print("\n  *** Reproduction failed. Do not interpret the alternative clip below. ***")

    # --- 2. the alternative clip ----------------------------------------------------------
    # Normalised residual about the same centre the raw clip uses (the parallax mode of the
    # pre-clip members), but scaled by each star's own uncertainty. Faint stars with large
    # parallax_error are no longer cut for being faint.
    from scipy import stats as sps

    centre_plx = float(sps.mode(np.round(parallax[preclip], 2), keepdims=False).mode)
    z = np.abs(parallax - centre_plx) / parallax_error
    normalised_mask = preclip & (z < SIGMA) & (probability >= PROBABILITY_CUT)

    alternative = quartile_ks(gmag_all[normalised_mask], radius_all[normalised_mask])
    alternative["n_total"] = int(normalised_mask.sum())
    alternative["parallax_centre_mas"] = centre_plx
    results["normalised_clip"] = alternative

    print(f"\nNORMALISED-RESIDUAL CLIP  N = {alternative['n_total']}  "
          f"(centre {centre_plx:.4f} mas, |z| < {SIGMA})")
    print(f"  quartile edges: " + " | ".join(f"{e:.2f}" for e in alternative["edges"]))
    print(f"  {'against':>14s} {'n':>4s} {'D':>7s} {'p':>8s}")
    for got in alternative["comparisons"]:
        print(f"  {got['against']:>14s} {got['n']:4d} {got['D']:7.3f} {got['p']:8.4f}")

    # --- 3. verdict -----------------------------------------------------------------------
    for key in ("published", "normalised_clip"):
        p = np.array([c["p"] for c in results[key]["comparisons"]])
        adj = holm(p)
        for comparison, value in zip(results[key]["comparisons"], adj):
            comparison["p_holm"] = float(value)
        results[key]["n_significant_after_holm"] = int((adj < 0.05).sum())

    before = results["published"]["n_significant_after_holm"]
    after = results["normalised_clip"]["n_significant_after_holm"]
    results["verdict"] = (
        "signal survives -- incompleteness attribution strengthened"
        if after >= 1
        else "signal disappears -- the deficit tracks the pipeline clip, not Gaia completeness"
    )
    print(f"\nSignificant comparisons after Holm correction: published {before}, "
          f"normalised clip {after}")
    print(f"VERDICT: {results['verdict']}")

    out = Path(__file__).with_name("faint_quartile_clip_test.json")
    out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
