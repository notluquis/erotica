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
    expected = np.sum(
        (10 ** ((SOLAR_MAG - (mags - 5 * np.log10(1110.0 / 10))) / 2.5)) ** (1 / 3.5)
    )
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
    mag = 14.0 - 2.5 * np.log10(mass**3.5)     # brighter for heavier
    color = 1.6 - 0.15 * mass                   # bluer for heavier
    return mag, color, mass


def test_nearest_isochrone_point_recovers_the_injected_mass():
    """Oracle: stars placed exactly on the track must get their own mass back."""
    mag, color, mass = _synthetic_isochrone()
    pick = np.array([10, 60, 120, 199])
    stars = QTable({
        "designation": np.arange(pick.size),
        "Gmag": mag[pick] * u.mag,
        "BP_RP": color[pick] * u.mag,
    })
    out = assign_mass_nearest_isochrone_point_kdtree(stars, (mag, color, mass))
    np.testing.assert_allclose(out["mass"].to_value(u.Msun), mass[pick], rtol=1e-10)


def test_assign_masses_averages_the_k_nearest_and_reports_a_spread():
    """k=1 must be exact; k>1 must be close but carry a non-zero spread."""
    mag, color, mass = _synthetic_isochrone()
    iso = [(mag, color, np.zeros_like(mass), mass)]   # legacy 4-column layout
    pick = np.array([40, 100, 150])

    exact = assign_masses(iso, mag[pick], color[pick], np.arange(pick.size), k=1)
    np.testing.assert_allclose(exact["mass"].to_value(u.Msun), mass[pick], rtol=1e-10)
    assert np.all(exact["mass_std"].to_value(u.Msun) == 0.0)

    smoothed = assign_masses(iso, mag[pick], color[pick], np.arange(pick.size), k=5)
    np.testing.assert_allclose(
        smoothed["mass"].to_value(u.Msun), mass[pick], rtol=0.05
    )
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
