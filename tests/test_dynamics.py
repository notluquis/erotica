"""Regression tests for pumps.analysis.dynamics.

The Galactic-coordinate branch of ``calculate_galactocentric_distance`` used the
non-existent ``Angle.radians`` attribute and crashed with AttributeError on every
call — proof it was untested. These pin the two call forms.
"""

import astropy.units as u
import numpy as np

from pumps.analysis.dynamics import calculate_galactocentric_distance


def test_galactocentric_distance_galactic_branch():
    # POSITIONAL (distance, l, b) form -> single radius. Regression for the
    # .radian(s) crash. NB: passing distance=/l=/b= as keywords routes to the
    # equatorial branch instead — the Galactic branch needs positional args.
    r = calculate_galactocentric_distance(1.5 * u.kpc, 355.7 * u.deg, 0.1 * u.deg)
    val = r.to(u.kpc).value
    # R_sun=8.3 kpc, cluster 1.5 kpc toward l~356/b~0 -> R_gc ~ 6.8 kpc (law of cosines).
    assert np.isfinite(val)
    assert 6.0 < val < 7.5


def test_galactocentric_distance_equatorial_returns_pair():
    # Equatorial (ra, dec, distance) form -> (radius, radius_err).
    out = calculate_galactocentric_distance(
        ra=265.0 * u.deg, dec=-32.5 * u.deg, distance=1.5 * u.kpc
    )
    assert isinstance(out, tuple) and len(out) == 2
    radius, radius_err = out
    assert np.isfinite(radius.to(u.kpc).value)
