"""Regression tests for erotica.analysis.dynamics.

The Galactic-coordinate branch of ``calculate_galactocentric_distance`` used the
non-existent ``Angle.radians`` attribute and crashed with AttributeError on every
call — proof it was untested. These pin the two call forms.
"""

import astropy.units as u
import numpy as np

from erotica.analysis.dynamics import calculate_galactocentric_distance


def test_galactocentric_distance_galactic_branch():
    # POSITIONAL (distance, l, b) form -> single radius. Regression for the
    # .radian(s) crash. NB: passing distance=/l=/b= as keywords routes to the
    # equatorial branch instead — the Galactic branch needs positional args.
    r = calculate_galactocentric_distance(1.5 * u.kpc, 355.7 * u.deg, 0.1 * u.deg)
    val = r.to(u.kpc).value
    # R_0=SOLAR_RADIUS (8.125 kpc), cluster 1.5 kpc toward l~356/b~0 -> R_gc ~ 6.6 kpc.
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


# ---------------------------------------------------------------------------
# R_0 consistency — see docs/design-notes/decisions.md (2026-07-27)
#
# Two bugs lived here: the module carried R_0 = 8.125 kpc in one function and
# 8.3 kpc in two others while all three feed the same Hill-radius chain, and
# calculate_hill_radius dropped a caller-supplied solar_radius on one branch.
# These pin both. The oracle is analytic, not a golden number.
# ---------------------------------------------------------------------------

import inspect

import pytest

from erotica.analysis.dynamics import (
    SOLAR_RADIUS,
    calculate_galactic_mass,
    calculate_hill_radius,
)


def _default_solar_radius(func):
    """The solar_radius default actually bound in a function signature."""
    return inspect.signature(func).parameters["solar_radius"].default


def test_all_functions_share_one_solar_radius():
    """The three functions in the Hill-radius chain must not disagree on R_0."""
    defaults = {
        f.__name__: _default_solar_radius(f)
        for f in (
            calculate_galactic_mass,
            calculate_galactocentric_distance,
            calculate_hill_radius,
        )
    }
    assert len(set(defaults.values())) == 1, f"R_0 disagrees across the chain: {defaults}"
    assert all(v is SOLAR_RADIUS for v in defaults.values())


def test_galactocentric_distance_matches_closed_form_toward_centre():
    """Analytic oracle: at l=0, b=0 the law of cosines collapses to |R_0 - d|."""
    for d_kpc in (0.0, 1.5, 8.125, 12.0):
        r = calculate_galactocentric_distance(d_kpc * u.kpc, 0.0 * u.deg, 0.0 * u.deg)
        expected = abs(SOLAR_RADIUS.to_value(u.kpc) - d_kpc)
        assert r.to_value(u.kpc) == pytest.approx(expected, abs=1e-6), f"d={d_kpc} kpc"


def test_galactocentric_distance_matches_closed_form_anticentre():
    """At l=180, b=0 the distances add: R_gc = R_0 + d."""
    d = 2.0
    r = calculate_galactocentric_distance(d * u.kpc, 180.0 * u.deg, 0.0 * u.deg)
    expected = SOLAR_RADIUS.to_value(u.kpc) + d
    assert r.to_value(u.kpc) == pytest.approx(expected, abs=1e-6)


def test_solar_radius_override_is_honoured_on_every_path():
    """A caller-supplied R_0 must reach the computation, not be silently dropped.

    calculate_hill_radius has two routes to the galactocentric distance -- via
    ``center`` and via ``(distance, l, b)``. The ``center`` route previously did not
    forward solar_radius.
    """
    alt = 4.0 * u.kpc  # deliberately far from the default so any effect is visible

    # direct function
    a = calculate_galactocentric_distance(1.5 * u.kpc, 0.0 * u.deg, 0.0 * u.deg)
    b = calculate_galactocentric_distance(
        1.5 * u.kpc, 0.0 * u.deg, 0.0 * u.deg, solar_radius=alt
    )
    assert not np.isclose(a.to_value(u.kpc), b.to_value(u.kpc)), "override ignored"
    assert b.to_value(u.kpc) == pytest.approx(abs(4.0 - 1.5), abs=1e-6)

    # the center branch of calculate_hill_radius
    common = dict(distance=1.5 * u.kpc, cluster_mass=500 * u.Msun, return_galdist=True)
    out_default = calculate_hill_radius(center=(265.0 * u.deg, -32.5 * u.deg), **common)
    out_alt = calculate_hill_radius(
        center=(265.0 * u.deg, -32.5 * u.deg), solar_radius=alt, **common
    )
    gd_default = out_default["galactocentric_distance"].to_value(u.kpc)
    gd_alt = out_alt["galactocentric_distance"].to_value(u.kpc)
    assert not np.isclose(gd_default, gd_alt), "solar_radius dropped on the center branch"
