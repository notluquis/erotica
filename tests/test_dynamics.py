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
    SOLAR_RADIUS,
    calculate_galactic_mass,
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


# ---------------------------------------------------------------------------
# Gaps found by a mutation audit on 2026-07-27. Each of these caught nothing
# before: the Jacobi prefactor could move 14.5%, posterior_summary could report
# the mean as the median, and cos(b) could be deleted entirely.
# ---------------------------------------------------------------------------


def test_jacobi_radius_matches_the_closed_form_absolutely_not_just_in_ratio():
    """Oracle: r_J = (M / 2 M_G)^(1/3) * R_gc, evaluated here in full.

    Every other test of this function checks a RATIO, so the prefactor cancels and
    a mutation from 2 to 3 in the denominator -- a 14.5% shift in the Jacobi radius,
    the physical boundary the whole R_t argument rests on -- passed unnoticed.
    """
    mass, r_gc, dist = 900 * u.Msun, 7.2 * u.kpc, 1.11 * u.kpc
    galactic_mass = calculate_galactic_mass(r_gc)
    expected_pc = ((mass / (2 * galactic_mass)) ** (1 / 3) * r_gc).to_value(u.pc)

    got = tidal_radius_prior(mass, r_gc, distance=dist)["angular_size"]
    got_pc = (got.to_value(u.arcmin) / 60 * np.pi / 180) * dist.to_value(u.pc)
    assert got_pc == pytest.approx(expected_pc, rel=1e-6)


def test_posterior_summary_distinguishes_median_from_mean():
    """A skewed sample. The old test used np.arange, which is symmetric, so mean
    and median coincide and swapping them -- or swapping `minus` and `plus` --
    could not be detected."""
    rng = np.random.default_rng(0)
    skewed = rng.lognormal(0.0, 1.0, 200_000)
    s = posterior_summary(skewed, credible_mass=0.68)
    assert s["median"] == pytest.approx(1.0, rel=0.02)          # exp(mu)
    assert s["median"] < skewed.mean() * 0.75                   # mean ~1.65: distinct
    # asymmetric, and the right way round
    assert s["plus"] > 2.0 * s["minus"]
    assert s["upper"] - s["median"] == pytest.approx(s["plus"])
    assert s["median"] - s["lower"] == pytest.approx(s["minus"])


@pytest.mark.parametrize("b_deg", [0.0, 15.0, 45.0, -30.0])
def test_galactocentric_distance_uses_galactic_latitude(b_deg):
    """Oracle: the law of cosines with the cos(b) factor written out here.

    Every earlier test used b = 0, where cos(b) = 1, so deleting cos(b) from all
    three of its occurrences changed nothing any test could see.
    """
    d, l = 1.5 * u.kpc, 40.0 * u.deg
    b = b_deg * u.deg
    R0 = SOLAR_RADIUS.to_value(u.kpc)
    dv = d.to_value(u.kpc)
    expected = np.sqrt(
        R0**2 + dv**2 - 2 * R0 * dv * np.cos(l.to_value(u.rad)) * np.cos(b.to_value(u.rad))
    )
    got = calculate_galactocentric_distance(distance=d, l=l, b=b).to_value(u.kpc)
    assert got == pytest.approx(expected, rel=1e-9)


def test_galactic_latitude_actually_changes_the_answer():
    """Guard the guard: if b had no effect the parametrized test above would be vacuous."""
    at_zero = calculate_galactocentric_distance(
        distance=1.5 * u.kpc, l=40.0 * u.deg, b=0.0 * u.deg
    ).to_value(u.kpc)
    at_forty = calculate_galactocentric_distance(
        distance=1.5 * u.kpc, l=40.0 * u.deg, b=45.0 * u.deg
    ).to_value(u.kpc)
    assert abs(at_forty - at_zero) > 0.1


# ---------------------------------------------------------------------------
# Coulomb logarithm: the pole, and which calibration applies
# ---------------------------------------------------------------------------


def test_relaxation_time_refuses_to_return_a_negative_time():
    """Oracle: a relaxation time is positive. Below N = 1/gamma the expression is not.

    This is a regression test for silent nonsense, not a hypothetical. Before the guard,
    ``half_mass_relaxation_time(5, ...)`` returned **-1.6 Myr** and ``(9, ...)`` returned
    **-173.7 Myr**, with no warning — the logarithm ``ln(gamma*N)`` goes negative below the pole
    and the sign propagates straight through.
    """
    import astropy.units as u
    import pytest as _pytest

    from erotica.analysis.dynamics import half_mass_relaxation_time

    for n in (5, 9):
        with _pytest.raises(ValueError, match="non-positive"):
            half_mass_relaxation_time(n, 2.0 * u.pc, 900 * u.Msun, lambda_value=0.11)
    # just above the pole it is finite and positive again
    assert half_mass_relaxation_time(10, 2.0 * u.pc, 900 * u.Msun, lambda_value=0.11).value > 0


def test_the_pole_moves_into_the_census_for_the_correct_coulomb_argument():
    """The Coulomb argument is not a cosmetic choice, and this pins how much it matters.

    ``gamma = 0.11`` is the **equal-mass** N-body calibration (Giersz & Heggie 1994); ``gamma = 0.02``
    is the **multi-mass** one (Giersz & Heggie 1996, *"a factor 7 smaller than the value found in
    Paper I for systems with equal masses"*). Every real cluster has a mass function, so 0.02 is the
    applicable branch — and its pole sits at N = 50, inside a census whose median is 61 and 40% of
    which lies below 50.

    At the census median the two give answers differing by nearly an order of magnitude, because the
    correct one sits close to its own pole.
    """
    import astropy.units as u
    import pytest as _pytest

    from erotica.analysis.dynamics import half_mass_relaxation_time

    args = (2.0 * u.pc, 900 * u.Msun)
    equal_mass = half_mass_relaxation_time(61, *args, lambda_value=0.11).value
    multi_mass = half_mass_relaxation_time(61, *args, lambda_value=0.02).value
    assert multi_mass / equal_mass > 5.0, (equal_mass, multi_mass)

    # N = 40 is below the multi-mass pole and must be refused, while the equal-mass value happily
    # returns a number -- which is exactly the trap.
    assert half_mass_relaxation_time(40, *args, lambda_value=0.11).value > 0
    with _pytest.raises(ValueError):
        half_mass_relaxation_time(40, *args, lambda_value=0.02)


def test_calibration_guard_catches_the_error_this_project_actually_made():
    """The failure mode is extrapolating past a calibration boundary, and it recurs.

    It is recorded twice in this project's own history: once in the completeness work (a bias law
    fitted at 50% suppression and applied at 1.2%) and once here — a commit asserted that
    ``gamma = 0.02`` "is the applicable branch" for Gaia open clusters, when Giersz & Heggie (1996)
    state in their abstract that their systems are **isolated** with **N from 250 to 1000**, while
    open clusters are tidally limited with a census median of N = 61.

    So the guard is the machine-checkable form of a mistake that documentation alone did not
    prevent. This test asserts it fires on exactly that case.
    """
    from erotica.analysis.dynamics import coulomb_calibration_warnings

    # the census median, with the "physically correct" multi-mass value
    reasons = coulomb_calibration_warnings(0.02, 61)
    assert any("outside the calibration range 250-1000" in r for r in reasons), reasons
    assert any("ISOLATED" in r for r in reasons), reasons

    # inside the N range, the isolation mismatch still stands -- it is not an N problem
    inside = coulomb_calibration_warnings(0.02, 500)
    assert not any("calibration range" in r for r in inside), inside
    assert any("ISOLATED" in r for r in inside), inside

    # the equal-mass value must be flagged as such for a cluster with a mass function
    equal_mass_value = coulomb_calibration_warnings(0.11, 500)
    assert any("EQUAL-MASS" in r for r in equal_mass_value), equal_mass_value
    # ...and not flagged when the caller says the cluster really is equal-mass
    assert not any(
        "EQUAL-MASS" in r for r in coulomb_calibration_warnings(0.11, 500, equal_mass=True)
    )

    # an undocumented value must not pass silently
    assert coulomb_calibration_warnings(0.07, 500), "unknown gamma should report missing provenance"


def test_relaxation_time_warns_rather_than_silently_extrapolating():
    """The guard has to reach the caller, not just exist."""
    import warnings as _warnings

    import astropy.units as u

    from erotica.analysis.dynamics import half_mass_relaxation_time

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        half_mass_relaxation_time(61, 2.0 * u.pc, 900 * u.Msun, lambda_value=0.02)
    messages = [str(w.message) for w in caught]
    assert any("outside its calibration" in m for m in messages), messages


def test_relaxation_time_flags_an_unphysical_implied_mean_mass():
    """``n_stars`` is ``M/<m>``, not an observed member count, and confusing them costs 4.5x.

    The function derives the mean stellar mass as ``M/N``, so passing an incomplete member list
    corrupts both the Coulomb logarithm and the mean mass. For NGC 6383 (M = 900 Msun) passing the
    254 observed members implies a mean stellar mass of 3.54 Msun -- no IMF produces that, and it is
    the only visible tell. Giersz & Heggie (1996) define N as bound stars; Cordoni et al. (2023)
    Eq. 5 writes the argument as M/<m> explicitly.
    """
    import warnings as _warnings

    import astropy.units as u

    from erotica.analysis.dynamics import half_mass_relaxation_time

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        half_mass_relaxation_time(254, 2.0 * u.pc, 900 * u.Msun)
    assert any("no IMF produces" in str(w.message) for w in caught), [str(w.message) for w in caught]

    # a physical mean mass must not warn about it
    with _warnings.catch_warnings(record=True) as clean:
        _warnings.simplefilter("always")
        half_mass_relaxation_time(1800, 2.0 * u.pc, 900 * u.Msun)
    assert not any("no IMF produces" in str(w.message) for w in clean)

    # and the size of the mistake is what makes it worth catching
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        wrong = half_mass_relaxation_time(254, 2.0 * u.pc, 900 * u.Msun).value
        right = half_mass_relaxation_time(1800, 2.0 * u.pc, 900 * u.Msun).value
    assert right / wrong > 4.0


# ---------------------------------------------------------------------------
# Henon (1975) closed-form gamma(mass function)
# ---------------------------------------------------------------------------


def test_henon_reproduces_his_own_published_tables():
    """The strongest oracle available: the author's own tabulated values.

    Hénon (1975, IAU Symp. 69, 133) Table II tabulates ``I3/I1`` for a continuous
    ``h ∝ m^-2`` spectrum over a mass ratio Q. Reproducing it to four decimals is a real check on
    the double integral, the log-measure and the mean-mass normalisation at once -- none of which
    could be verified by inspecting the output.
    """
    import numpy as np

    from erotica.analysis.dynamics import coulomb_argument_from_mass_function

    published = {2: -0.0498, 4: -0.1964, 8: -0.4322, 16: -0.7458, 32: -1.1246, 64: -1.5562}
    for ratio, expected in published.items():
        got = coulomb_argument_from_mass_function(
            lambda m: np.asarray(m, dtype=float) ** -2.0, 1.0, float(ratio)
        )["i3_over_i1"]
        assert abs(got - expected) < 2e-4, f"Q={ratio}: got {got:.4f}, Hénon {expected:.4f}"


def test_henon_equal_mass_limit_matches_his_analytic_value():
    """Parameter-free limit: as the mass range collapses, gamma must reach Hénon Eq. (19)'s 0.15.

    Note this is a **fourth** value distinct from the 0.4, 0.11 and 0.02 in circulation -- the
    analytic equal-mass result with a Maxwellian velocity distribution.
    """
    import numpy as np

    from erotica.analysis.dynamics import coulomb_argument_from_mass_function

    result = coulomb_argument_from_mass_function(
        lambda m: np.asarray(m, dtype=float) ** -2.0, 1.0, 1.0001
    )
    assert abs(result["i3_over_i1"]) < 1e-4
    assert abs(result["gamma"] - 0.15) < 0.001, result["gamma"]


def test_gamma_falls_as_the_mass_range_widens_and_that_is_age_dependent():
    """The result that matters: gamma is not a constant, and its variation tracks cluster age.

    A Kroupa IMF truncated at the turn-off gives gamma from 0.0050 at 1 Myr to 0.0686 at 1 Gyr --
    a factor of 14 -- so a single gamma imposes an age-dependent systematic on any age/t_rh trend.
    """
    import numpy as np

    from erotica.analysis.dynamics import coulomb_argument_from_mass_function

    def kroupa(m):
        m = np.asarray(m, dtype=float)
        return np.where(m < 0.5, m**-1.3, 0.5 ** (-1.3 + 2.3) * m**-2.3)

    young = coulomb_argument_from_mass_function(kroupa, 0.08, 100.0)["gamma"]
    old = coulomb_argument_from_mass_function(kroupa, 0.08, 2.0)["gamma"]
    assert young < old, (young, old)
    assert old / young > 10.0, f"expected a factor >10, got {old / young:.1f}"
    assert abs(young - 0.0050) < 0.0005 and abs(old - 0.0686) < 0.002

    # and the mean stellar mass must fall as the massive stars are removed
    assert (
        coulomb_argument_from_mass_function(kroupa, 0.08, 2.0)["mean_mass"]
        < coulomb_argument_from_mass_function(kroupa, 0.08, 100.0)["mean_mass"]
    )
