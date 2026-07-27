"""Regression tests for erotica.analysis.dynamics.

The Galactic-coordinate branch of ``calculate_galactocentric_distance`` used the
non-existent ``Angle.radians`` attribute and crashed with AttributeError on every
call — proof it was untested. These pin all five call forms, the two unpacking
bugs found on 2026-07-27, and the posterior-propagation path.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from erotica.analysis.dynamics import (
    calculate_galactocentric_distance,
    crossing_time,
    half_mass_relaxation_time,
    posterior_summary,
    tidal_radius_prior,
)


def test_galactocentric_distance_galactic_branch():
    # POSITIONAL (distance, l, b) form -> single radius. Regression for the
    # .radian(s) crash.
    #
    # This comment used to add: "NB: passing distance=/l=/b= as keywords routes
    # to the equatorial branch instead". That was a BUG being documented as
    # intended behaviour -- the keyword form raised inside SkyCoord. Fixed
    # 2026-07-27; test_keyword_galactic_form_works_and_matches_the_positional_one
    # now pins the equivalence.
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


from erotica.analysis.dynamics import (
    SOLAR_RADIUS,
    calculate_galactic_mass,
    calculate_galactocentric_distance,
    calculate_hill_radius,
    crossing_time,
    half_mass_relaxation_time,
    posterior_summary,
    tidal_radius_prior,
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


# ---------------------------------------------------------------------------
# Call-form dispatch. Five forms reach calculate_galactocentric_distance, and
# before 2026-07-27 the keyword Galactic one -- the form the docstring
# advertises -- raised inside SkyCoord because the equatorial guard fired on
# ``distance is not None`` alone.
# ---------------------------------------------------------------------------

RA, DEC = 263.6826 * u.deg, -32.5838 * u.deg
DIST = 1.11 * u.kpc


def test_keyword_galactic_form_works_and_matches_the_positional_one():
    """Oracle: the positional Galactic form, which always worked."""
    positional = calculate_galactocentric_distance(DIST, 355.0 * u.deg, 0.0 * u.deg)
    keyword = calculate_galactocentric_distance(distance=DIST, l=355.0 * u.deg, b=0.0 * u.deg)
    assert keyword.to_value(u.kpc) == pytest.approx(positional.to_value(u.kpc))


def test_every_equatorial_call_form_agrees():
    """Oracle: the three equatorial spellings describe one sky position, so they
    must return one number. Form 4 is the one `calculate_hill_radius(center=...)`
    uses internally, and a guard change once broke exactly that path."""
    legacy_positional = calculate_galactocentric_distance(RA, DEC, DIST)[0]
    all_keyword = calculate_galactocentric_distance(ra=RA, dec=DEC, distance=DIST)[0]
    mixed = calculate_galactocentric_distance(RA, DEC, distance=DIST)[0]
    for got in (all_keyword, mixed):
        assert got.to_value(u.kpc) == pytest.approx(legacy_positional.to_value(u.kpc))


def test_equatorial_and_galactic_agree_for_the_same_sky_position():
    """Converting the position rather than trusting a rounded l/b."""
    gal = SkyCoord(ra=RA, dec=DEC, distance=DIST, frame="icrs").galactic
    equatorial = calculate_galactocentric_distance(ra=RA, dec=DEC, distance=DIST)[0]
    galactic = calculate_galactocentric_distance(distance=DIST, l=gal.l, b=gal.b)
    assert galactic.to_value(u.kpc) == pytest.approx(equatorial.to_value(u.kpc), rel=1e-6)


def test_tidal_radius_prior_works_on_its_default_path():
    """It unpacked a 2-tuple from a function that returns a bare Quantity when no
    error is supplied -- and no error is the DEFAULT, so the default call raised.

    This is the function PART J recommends for the King ``R_t`` prior.
    """
    out = tidal_radius_prior(900 * u.Msun, 7.2 * u.kpc, distance=1.11 * u.kpc)
    assert "angular_size" in out
    assert np.isfinite(out["angular_size"].to_value(u.arcmin))
    assert out["angular_size"].to_value(u.arcmin) > 0
    # and it still works when an error IS given, i.e. the other branch
    with_err = tidal_radius_prior(
        900 * u.Msun, 7.2 * u.kpc, 0.1 * u.kpc, distance=1.11 * u.kpc
    )
    assert np.isfinite(with_err["angular_size"].to_value(u.arcmin))


def test_jacobi_radius_scales_as_the_cube_root_of_mass():
    """Analytic oracle: r_J ∝ M^(1/3), so 8x the mass doubles the radius."""
    kw = dict(galactocentric_distance=7.2 * u.kpc, distance=1.11 * u.kpc)
    small = tidal_radius_prior(500 * u.Msun, kw["galactocentric_distance"], distance=kw["distance"])
    big = tidal_radius_prior(4000 * u.Msun, kw["galactocentric_distance"], distance=kw["distance"])
    ratio = big["angular_size"] / small["angular_size"]
    assert float(ratio) == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# Posterior propagation: the dynamics chain must accept sample arrays so derived
# quantities come out as distributions rather than as collapsed scalars.
# ---------------------------------------------------------------------------


def test_dynamics_functions_accept_posterior_samples():
    """Every function on the published-number chain must broadcast over draws."""
    n = 500
    rng = np.random.default_rng(0)
    mass = rng.normal(900, 120, n) * u.Msun
    dist = rng.normal(1.11, 0.06, n) * u.kpc
    r_half = rng.normal(1.9, 0.3, n) * u.pc
    sigma_v = rng.normal(1.2, 0.2, n) * u.km / u.s

    r_gc = calculate_galactocentric_distance(distance=dist, l=355.0 * u.deg, b=0.0 * u.deg)
    jacobi = tidal_radius_prior(mass, r_gc, distance=dist)["angular_size"]
    t_cross = crossing_time(r_half, sigma_v)
    t_rh = half_mass_relaxation_time(300, r_half, mass)

    for name, arr in (("r_gc", r_gc), ("jacobi", jacobi), ("t_cross", t_cross), ("t_rh", t_rh)):
        assert arr.shape == (n,), f"{name} collapsed to {arr.shape}"
        assert np.all(np.isfinite(arr.value)), name
        assert arr.value.std() > 0, f"{name} has no spread -- uncertainty was lost"


def test_posterior_summary_returns_a_credible_interval():
    """Oracle: percentiles of a uniform sample are known exactly."""
    s = posterior_summary(np.arange(1001.0), credible_mass=0.68)
    assert s["median"] == pytest.approx(500.0)
    assert s["lower"] == pytest.approx(160.0)
    assert s["upper"] == pytest.approx(840.0)
    assert s["minus"] == pytest.approx(340.0)
    assert s["plus"] == pytest.approx(340.0)
    assert s["n"] == 1001


def test_posterior_summary_preserves_units_and_widens_with_credible_mass():
    s68 = posterior_summary(np.arange(1001.0) * u.pc, credible_mass=0.68)
    s95 = posterior_summary(np.arange(1001.0) * u.pc, credible_mass=0.95)
    assert s68["median"].unit == u.pc
    assert s95["upper"] > s68["upper"]
    assert s95["lower"] < s68["lower"]


def test_posterior_summary_rejects_an_all_nan_sample():
    with pytest.raises(ValueError, match="finite"):
        posterior_summary(np.full(10, np.nan))
