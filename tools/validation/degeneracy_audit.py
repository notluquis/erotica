#!/usr/bin/env python3
"""Which parameters are not separately identified? A systematic posterior-correlation audit.

WHY THIS EXISTS
---------------
This programme keeps discovering, one at a time and usually late, that two things it reports as
separate numbers are the same number wearing two hats:

* ``King(R_t -> inf)`` is **exactly** ``EFF(gamma = 2)``, verified to 80 decimal places;
* a corona wider than the footprint is **exactly** a flat background, since
  ``2 delta_f sqrt(R_2^2 - r^2) -> 2 delta_f R_2`` as ``R_2`` grows -- so on NGC 6383 the two
  models sit at ``2 ln B = -2.27``, i.e. indistinguishable, with ``P(R_2 > field) = 1.000``;
* the ``r_t`` <-> background degeneracy, which **King himself** flagged in 1962 when criticising
  Wallenquist's M37 radius, and which ASteCA's source documents ("the value given to the field
  density has a *very large* influence on the final (rc, rt) values");
* Cartwright & Whitworth's ``Q`` is degenerate with contamination -- ~60% uniform dilution of a
  perfectly smooth King reproduces NGC 6383's observed ``Q = 0.833`` exactly.

Every one of those was found by accident. **This script looks for them on purpose**, before a number
is quoted, by reading the posterior correlation matrix of a fit and flagging pairs that the data do
not separate.

WHAT IT MEANS AND WHAT IT DOES NOT
----------------------------------
A high posterior correlation says the *likelihood surface* has a valley: the data constrain a
combination of the two parameters far better than either alone. That has three consequences worth
separating:

1. **The marginal uncertainties are not the whole story.** Quoting ``a = 1.65 +/- 0.38`` and
   ``gamma = 2.32 +/- 0.21`` independently, when they are correlated at 0.95, overstates what is
   known about either and understates what is known about their combination.
2. **A prior on one silently determines the other.** This is exactly the ``R_t``/``b`` story.
3. **It is not automatically a defect.** Correlated parameters can still be reported honestly, as a
   contour or as the well-constrained combination. What is not honest is reporting them as though
   they were independent measurements.

The audit therefore ranks pairs and reports the **condition number** of the correlation matrix, which
is the scale-free summary of how close the whole fit is to being rank-deficient.

USAGE
-----
    python tools/validation/degeneracy_audit.py                 # NGC 6383, all models
    python tools/validation/degeneracy_audit.py --model eff

Writes ``degeneracy_audit.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

WARN = 0.90  # |r| above which two parameters are reported as not separately identified
NOTE = 0.70


def ngc6383_radii(field_radius=70.0):
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u

    base = Path("/Users/notluquis/erotica/data/test/NGC6383")
    table = Table.read(base / "comments_paper/radius_robustness/generated/70/paperfaithful_reference_p06.ecsv")
    ra = next(c for c in table.colnames if c.lower() in ("ra", "ra_icrs"))
    dec = next(c for c in table.colnames if c.lower() in ("dec", "de_icrs"))
    centre = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
    sep = centre.separation(
        SkyCoord(np.asarray(table[ra]) * u.deg, np.asarray(table[dec]) * u.deg)
    ).to(u.arcmin).value
    return sep[(sep > 0) & (sep <= field_radius)]


def audit(trace, names):
    """Posterior correlation matrix, ranked pairs, and the condition number."""
    draws = np.column_stack(
        [np.asarray(trace.posterior[n].values).ravel() for n in names]
    )
    corr = np.corrcoef(draws, rowvar=False)
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            pairs.append(dict(a=names[i], b=names[j], r=float(corr[i, j])))
    pairs.sort(key=lambda p: -abs(p["r"]))
    return corr, pairs, float(np.linalg.cond(corr))


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="all", choices=["king", "eff", "king_corona", "all"])
    ap.add_argument("--draws", type=int, default=3000)
    args = ap.parse_args()

    import pymc as pm

    from erotica.analysis.structure import (
        CoronaPriors, EFFPriors, KingPriors, _eff_model, _king_corona_model, _king_model,
    )

    r = ngc6383_radii()
    print(f"NGC 6383, N = {r.size}, field 70 arcmin\n")

    specs = {
        "king": (lambda: _king_model(pm, r, 70.0, KingPriors(), None, None),
                 ["R_c", "R_t", "k", "b"]),
        "eff": (lambda: _eff_model(pm, r, 70.0, EFFPriors(), None, None),
                ["a", "gamma", "k", "b"]),
        "king_corona": (lambda: _king_corona_model(pm, r, 70.0, CoronaPriors(), None, None),
                        ["R_c", "R_t", "k", "R_2", "delta_f"]),
    }
    chosen = list(specs) if args.model == "all" else [args.model]

    out = {}
    for name in chosen:
        build, names = specs[name]
        with build():
            idata = pm.sample(args.draws, tune=1500, chains=4, random_seed=17,
                              progressbar=False, target_accept=0.95)
        corr, pairs, cond = audit(idata, names)
        divergences = int(np.asarray(idata.sample_stats["diverging"]).sum())

        print(f"=== {name} ===  condition number {cond:.1f}   divergences {divergences}")
        for p in pairs:
            flag = "  <-- NOT SEPARATELY IDENTIFIED" if abs(p["r"]) > WARN else (
                "  <-- strongly coupled" if abs(p["r"]) > NOTE else "")
            print(f"    corr({p['a']:>7s}, {p['b']:<7s}) = {p['r']:+.3f}{flag}")
        print()

        out[name] = dict(
            parameters=names,
            correlation=corr.tolist(),
            pairs=pairs,
            condition_number=cond,
            divergences=divergences,
            not_separately_identified=[p for p in pairs if abs(p["r"]) > WARN],
        )

    dest = Path(__file__).with_name("degeneracy_audit.json")
    dest.write_text(json.dumps(dict(n_stars=int(r.size), field_radius=70.0,
                                    warn_threshold=WARN, models=out), indent=1))
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
