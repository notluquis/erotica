"""Tests for the photometric mass chain.

WHY THIS MATTERS MORE THAN THE MODULE NAME SUGGESTS
---------------------------------------------------
These functions produce the cluster **mass**. The mass sets the **Jacobi radius**
through :func:`~erotica.analysis.dynamics.tidal_radius_prior`, and the Jacobi
radius is the physical boundary the whole ``R_t`` argument for NGC 6383 turns on
(see ``docs/design-notes/king_model_validity.md``). A wrong mass moves a physical
boundary, not just a number in a table.

WHAT THESE ARE CHECKED AGAINST
------------------------------
* the mass-luminosity chain -> closed forms evaluated by hand in the test, and
  the defining case of a star at 10 pc where the distance modulus vanishes;
* isochrone mass assignment -> a synthetic isochrone with a known CMD-to-mass
  mapping, with stars placed exactly on it, so the assigned mass must be the
  injected one;
* the mass -> Jacobi radius link -> the analytic ``r_J ∝ M^(1/3)`` scaling.
"""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis.photometry import (
    assign_mass_nearest_isochrone_point_kdtree,
    assign_masses,
)
from erotica.analysis.units import (
    calculate_absolute_magnitude,
    estimate_cluster_mass,
    estimate_luminosity,
    estimate_mass_from_luminosity,
)

SOLAR_MAG = 4.67


# ---------------------------------------------------------------------------
# The mass-luminosity chain
# ---------------------------------------------------------------------------


def test_absolute_magnitude_vanishes_at_ten_parsecs():
    """The defining case: at 10 pc the distance modulus is zero, by definition."""
    got = calculate_absolute_magnitude(12.34, 10 * u.pc)
    assert got.to_value(u.mag) == pytest.approx(12.34)


def test_absolute_magnitude_matches_the_distance_modulus():
    """Oracle: M = m - 5 log10(d/10), computed independently here."""
    for m, d_pc in ((10.0, 1000.0), (18.5, 1110.0), (7.2, 250.0)):
        expected = m - 5 * np.log10(d_pc / 10)
        assert calculate_absolute_magnitude(m, d_pc * u.pc).to_value(u.mag) == pytest.approx(
            expected
        )


def test_a_star_of_solar_absolute_magnitude_has_unit_luminosity_and_mass():
    """Both scalings are anchored at the Sun, so this pins the zero point."""
    assert estimate_luminosity(SOLAR_MAG) == pytest.approx(1.0)
    assert estimate_mass_from_luminosity(1.0) == pytest.approx(1.0)


def test_luminosity_is_a_pogson_scale():
    """Five magnitudes brighter is exactly a factor 100 in luminosity."""
    ratio = estimate_luminosity(SOLAR_MAG - 5.0) / estimate_luminosity(SOLAR_MAG)
    assert ratio == pytest.approx(100.0)


def test_mass_luminosity_exponent_is_one_over_3_5():
    """Oracle: L = M^3.5 inverted, checked at a decade of luminosity."""
    assert estimate_mass_from_luminosity(100.0) == pytest.approx(100.0 ** (1 / 3.5))
    m = estimate_mass_from_luminosity(np.array([1.0, 10.0, 1000.0]))
    np.testing.assert_allclose(m, np.array([1.0, 10.0, 1000.0]) ** (1 / 3.5))


def test_cluster_mass_matches_the_hand_computed_chain():
    """Oracle: the full magnitude -> mass chain, recomputed in the test."""
    mags = np.array([9.0, 12.0, 15.0, 18.0])
    distance = 1.11 * u.kpc
    expected = np.sum((10 ** ((SOLAR_MAG - (mags - 5 * np.log10(1110.0 / 10))) / 2.5)) ** (1 / 3.5))
    got = estimate_cluster_mass(mags, distance)
    assert got.to_value(u.Msun) == pytest.approx(expected)


def test_cluster_mass_ignores_nan_magnitudes():
    """A missing magnitude must drop out, not poison the sum."""
    clean = np.array([10.0, 12.0, 14.0])
    dirty = np.concatenate([clean, [np.nan]])
    d = 1.0 * u.kpc
    assert estimate_cluster_mass(dirty, d).to_value(u.Msun) == pytest.approx(
        estimate_cluster_mass(clean, d).to_value(u.Msun)
    )


def test_cluster_mass_grows_with_distance_at_fixed_apparent_magnitude():
    """Same apparent brightness further away means intrinsically brighter, so heavier."""
    mags = np.full(20, 14.0)
    near = estimate_cluster_mass(mags, 0.5 * u.kpc).to_value(u.Msun)
    far = estimate_cluster_mass(mags, 2.0 * u.kpc).to_value(u.Msun)
    assert far > near
    # each factor 4 in distance is 3 magnitudes, i.e. a fixed luminosity ratio
    assert far / near == pytest.approx((4.0**2) ** (1 / 3.5))


def test_cluster_mass_accepts_a_table_and_requires_gmag():
    t = QTable({"Gmag": np.array([10.0, 12.0]) * u.mag})
    assert estimate_cluster_mass(t, 1.0 * u.kpc).to_value(u.Msun) > 0
    with pytest.raises(ValueError, match="Gmag"):
        estimate_cluster_mass(QTable({"other": [1.0]}), 1.0 * u.kpc)


# ---------------------------------------------------------------------------
# Isochrone mass assignment
# ---------------------------------------------------------------------------


def _synthetic_isochrone(n=200):
    """A monotonic CMD track with a known mass at every point."""
    mass = np.linspace(0.2, 8.0, n)
    mag = 14.0 - 2.5 * np.log10(mass**3.5)  # brighter for heavier
    color = 1.6 - 0.15 * mass  # bluer for heavier
    return mag, color, mass


def test_nearest_isochrone_point_recovers_the_injected_mass():
    """Oracle: stars placed exactly on the track must get their own mass back."""
    mag, color, mass = _synthetic_isochrone()
    pick = np.array([10, 60, 120, 199])
    stars = QTable(
        {
            "designation": np.arange(pick.size),
            "Gmag": mag[pick] * u.mag,
            "BP_RP": color[pick] * u.mag,
        }
    )
    out = assign_mass_nearest_isochrone_point_kdtree(stars, (mag, color, mass))
    np.testing.assert_allclose(out["mass"].to_value(u.Msun), mass[pick], rtol=1e-10)


def test_assign_masses_averages_the_k_nearest_and_reports_a_spread():
    """k=1 must be exact; k>1 must be close but carry a non-zero spread."""
    mag, color, mass = _synthetic_isochrone()
    iso = [(mag, color, np.zeros_like(mass), mass)]  # legacy 4-column layout
    pick = np.array([40, 100, 150])

    exact = assign_masses(iso, mag[pick], color[pick], np.arange(pick.size), k=1)
    np.testing.assert_allclose(exact["mass"].to_value(u.Msun), mass[pick], rtol=1e-10)
    assert np.all(exact["mass_std"].to_value(u.Msun) == 0.0)

    smoothed = assign_masses(iso, mag[pick], color[pick], np.arange(pick.size), k=5)
    np.testing.assert_allclose(smoothed["mass"].to_value(u.Msun), mass[pick], rtol=0.05)
    assert np.all(smoothed["mass_std"].to_value(u.Msun) > 0.0)


def test_assign_masses_rejects_isochrones_with_no_finite_points():
    with pytest.raises(ValueError, match="No finite isochrone points"):
        assign_masses([(np.array([np.nan]),) * 4], np.array([12.0]), np.array([1.0]), [0])


def test_assign_masses_drops_non_finite_isochrone_points():
    """A NaN in the track must be skipped, not propagated into a mass."""
    mag, color, mass = _synthetic_isochrone(n=50)
    mag = mag.copy()
    mag[10] = np.nan
    iso = [(mag, color, np.zeros_like(mass), mass)]
    out = assign_masses(iso, np.array([mag[20]]), np.array([color[20]]), [0], k=1)
    assert np.isfinite(out["mass"].to_value(u.Msun)).all()
    assert out["mass"].to_value(u.Msun)[0] == pytest.approx(mass[20], rel=1e-10)


# ---------------------------------------------------------------------------
# PhotometricMassEstimator -- Defect 5.
#
# Until 2026-08-04 the two ``assign*`` methods of this one class had
# incompatible contracts: ``assign_from_samples`` took three positional ARRAYS,
# used ``self.k``, returned ``mass_std`` and hardcoded its id column to
# ``source_id``; ``assign_nearest`` took a QTable plus column-NAME strings,
# ignored ``self.k``, returned no ``mass_std`` and named its id column after
# ``designation_column``. ``color_column`` meant an array in one and a string in
# the other, and the constructor accepted either isochrone layout while checking
# neither, so the wrong one surfaced inside a KDTree query.
#
# ``assign_nearest`` had zero callers and zero tests, so it was moved onto
# ``assign_from_samples``' contract. The oracle below is the same one used for
# the free functions: stars placed exactly on a synthetic track must get their
# own injected mass back, whichever method assigns it.
# ---------------------------------------------------------------------------


def test_both_assign_methods_share_one_contract():
    """Oracle: the injected mass, recovered through both entry points.

    Same call signature, same output schema, same numbers on stars that sit exactly
    on the track -- that is what makes the two interchangeable downstream, and it is
    the property that did not hold before. The comparison is between the two methods
    AND against the injected truth, so a change that broke both consistently would
    still be caught.
    """
    from erotica.analysis.photometry import PhotometricMassEstimator

    mag, color, mass = _synthetic_isochrone()
    pick = np.array([10, 60, 120, 199])
    ids = np.array([11, 22, 33, 44])

    single = PhotometricMassEstimator((mag, color, mass))
    sampled = PhotometricMassEstimator([(mag, color, np.zeros_like(mass), mass)], k=1)
    assert single.isochrone_form == "single"
    assert sampled.isochrone_form == "samples"

    from_nearest = single.assign_nearest(mag[pick], color[pick], ids)
    from_samples = sampled.assign_from_samples(mag[pick], color[pick], ids)

    assert from_nearest.colnames == from_samples.colnames == ["source_id", "mass", "mass_std"]
    for table in (from_nearest, from_samples):
        np.testing.assert_array_equal(np.asarray(table["source_id"]), ids)
        assert table["mass"].unit == u.Msun
        assert table["mass_std"].unit == u.Msun
        np.testing.assert_allclose(table["mass"].to_value(u.Msun), mass[pick], rtol=1e-10)


def test_nearest_reports_nan_spread_and_k_equals_one_reports_zero():
    """A single nearest point has no spread; ``k = 1`` over samples has a real zero.

    The distinction is the whole reason ``mass_std`` may not be 0.0 in both cases.
    ``assign_from_samples(k=1)`` averages one point drawn from a *set* of sampled
    isochrones, so zero scatter is a measurement. ``assign_nearest`` has no set to
    scatter over, so 0.0 would assert that the mass is known exactly -- the opposite
    of the truth. NaN says "unmeasured", and aggregations that use ``nan``-aware
    reductions will skip it instead of being dragged towards zero.
    """
    from erotica.analysis.photometry import PhotometricMassEstimator

    mag, color, mass = _synthetic_isochrone()
    pick = np.array([10, 60, 120])
    ids = np.arange(pick.size)

    nearest = PhotometricMassEstimator((mag, color, mass)).assign_nearest(
        mag[pick], color[pick], ids
    )
    assert np.all(np.isnan(nearest["mass_std"].to_value(u.Msun))), (
        "a single nearest point has no spread; 0.0 would claim an exact mass"
    )

    k1 = PhotometricMassEstimator([(mag, color, np.zeros_like(mass), mass)], k=1)
    spread = k1.assign_from_samples(mag[pick], color[pick], ids)["mass_std"].to_value(u.Msun)
    assert np.all(spread == 0.0)
    assert not np.any(np.isnan(spread))


@pytest.mark.parametrize(
    "isochrones, match",
    [
        ([], "empty"),
        (5, "must be a sequence"),
        # Four bare arrays: the sampled form's INNER layout leaked one level up.
        # Read as three isochrones it would take iso[3] -- a mass -- as a magnitude.
        ((np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)), "neither accepted form"),
        ((np.zeros(9), np.zeros(4), np.zeros(9)), "same points"),
        ([(np.zeros(9), np.zeros(9))], "neither accepted form"),
    ],
)
def test_the_isochrone_form_is_validated_in_the_constructor(isochrones, match):
    """The wrong layout must fail where it was passed, not inside a KDTree query.

    Nothing distinguishes the two layouts once the arrays are stacked, so the
    constructor is the last place the mistake is still legible. Each case here is a
    layout that used to be accepted silently and then produced either a crash with an
    unrelated message or -- for the four-plain-arrays case -- plausible numbers built
    from the wrong columns.
    """
    from erotica.analysis.photometry import PhotometricMassEstimator

    with pytest.raises((TypeError, ValueError), match=match):
        PhotometricMassEstimator(isochrones)


def test_calling_the_method_for_the_other_form_raises_by_name():
    """A stored form and a mismatched call is still a caller error, reported as one."""
    from erotica.analysis.photometry import PhotometricMassEstimator

    mag, color, mass = _synthetic_isochrone(n=20)
    single = PhotometricMassEstimator((mag, color, mass))
    sampled = PhotometricMassEstimator([(mag, color, np.zeros_like(mass), mass)])

    with pytest.raises(ValueError, match=r"assign_from_samples\(\) needs the 'samples'"):
        single.assign_from_samples(mag[:1], color[:1], [0])
    with pytest.raises(ValueError, match=r"assign_nearest\(\) needs the 'single'"):
        sampled.assign_nearest(mag[:1], color[:1], [0])


# ---------------------------------------------------------------------------
# The link that makes all of the above load-bearing
# ---------------------------------------------------------------------------


def test_cluster_mass_propagates_into_the_jacobi_radius():
    """A mass error moves the physical boundary as ``r_J ∝ M^(1/3)``."""
    from erotica.analysis.dynamics import tidal_radius_prior

    d, r_gc = 1.11 * u.kpc, 7.2 * u.kpc
    mags = np.full(50, 14.0)
    mass = estimate_cluster_mass(mags, d)

    base = tidal_radius_prior(mass, r_gc, distance=d)["angular_size"]
    doubled = tidal_radius_prior(2 * mass, r_gc, distance=d)["angular_size"]
    assert float(doubled / base) == pytest.approx(2 ** (1 / 3), rel=1e-6)
    assert base.to_value(u.arcmin) > 0


def test_cmd_nearest_point_assignment_is_not_scale_invariant():
    """Pins a documented defect rather than asserting correctness.

    Nearest-neighbour matching in a colour-magnitude diagram uses a Euclidean distance between two
    quantities with different units, so the result depends on their relative scaling -- on the plot's
    aspect ratio, in effect. This test asserts that the dependence is **large for individual stars
    and small for the population**, which is exactly the statement the docstring makes and the reason
    the function may be used for a mass function but not for per-star masses.

    If a future version adds error weighting or a principled metric, this test should start failing,
    and that is the intended signal.
    """
    from scipy.spatial import KDTree

    rng = np.random.default_rng(0)
    t = np.linspace(0, 1, 400)
    iso_color = np.concatenate([0.2 + 1.6 * t, 1.8 - 0.9 * t[:150]])
    iso_mag = np.concatenate([2.0 + 8.0 * t, 10.0 - 0.3 * t[:150]])
    iso_mass = np.concatenate([3.0 - 2.6 * t, 0.4 - 0.1 * t[:150]])
    stars_c = rng.uniform(0.3, 1.7, 300)
    stars_m = rng.uniform(2.5, 9.5, 300)

    def assign(scale):
        tree = KDTree(np.column_stack([iso_color * scale, iso_mag]))
        _, idx = tree.query(np.column_stack([stars_c * scale, stars_m]))
        return iso_mass[idx]

    baseline = assign(1.0)
    doubled = assign(2.0)

    changed = np.mean(~np.isclose(doubled, baseline))
    assert changed > 0.5, f"only {changed:.1%} of per-star masses moved; docstring claims ~97%"

    median_shift = abs(np.median(doubled) / np.median(baseline) - 1)
    assert median_shift < 0.05, f"population median moved {median_shift:.1%}; it should be robust"


def test_estimator_accepts_a_generator_of_isochrones():
    """A generator must survive the constructor.

    `_classify_isochrone_form` consumes its input with `list(...)` to inspect it, and `__init__`
    used to store the ORIGINAL object afterwards -- so any iterator arrived at `assign_masses`
    already exhausted, `points` stayed empty, and the call failed with "No finite isochrone
    points with masses were supplied", blaming the data for a constructor defect.
    """
    import numpy as np

    from erotica.analysis.photometry import PhotometricMassEstimator

    def _iso(offset):
        mag = np.linspace(10.0, 16.0, 25) + offset
        color = np.linspace(0.2, 1.8, 25)
        extra = np.zeros(25)
        mass = np.linspace(2.5, 0.4, 25)
        return (mag, color, extra, mass)

    est = PhotometricMassEstimator((_iso(o) for o in (0.0, 0.05, -0.05)), k=2)
    assert est.isochrone_form == "samples"
    assert len(list(est.isochrones)) == 3, "the generator was stored exhausted"

    out = est.assign_from_samples(np.array([12.0, 13.5]), np.array([0.6, 1.1]), np.array([1, 2]))
    assert len(out) == 2
    assert np.all(np.isfinite(np.asarray(out["mass"], dtype=float)))
