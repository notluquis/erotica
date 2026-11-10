"""Tests for erotica.metallicity.pms_bias (ASPCAP [Fe/H] bias in young K stars).

Oracles:
- the anchor and sigma_ref arguments enter by exact algebra (a shift of ``a``; a quadrature term),
  so their non-default values have closed-form consequences;
- the hub prototype ``agent-findings/scripts/apogee_pms_bias_correction_2026_11_09.py`` and its
  ``_measured.txt`` (2026-11-09): N 104 in 12 clusters, LOCO RMS 0.076 against 0.137 for a constant,
  delta = -0.238 +- 0.060 and -0.320 +- 0.063 for the two NGC 6383 members. Those are an independent
  implementation's outputs, frozen; the coefficients embedded in the module are refitted here.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from erotica.metallicity import (
    CALIBRATED_MODEL,
    PMSBiasDomainWarning,
    fit_bias_model,
    leave_one_cluster_out,
    predict_bias,
    select_calibration_sample,
)

DATA = Path(__file__).parent / "data" / "apogee_dr17_young_k_pms_bias.ecsv"
SRC = "aspcap_dr17"


@pytest.fixture(scope="module")
def table():
    return Table.read(DATA)


# --- non-default parameters first (a default-only suite cannot see these branches) -------------


def test_anchor_shifts_only_the_intercept(table):
    """y = fe_h - anchor: a different anchor moves ``a`` by exactly the difference, nothing else."""
    m0 = fit_bias_model(table)
    m1 = fit_bias_model(table, anchor=0.0)
    assert m1.beta[0] - m0.beta[0] == pytest.approx(0.021, abs=1e-12)
    np.testing.assert_allclose(m1.beta[1:], m0.beta[1:], atol=1e-12)
    np.testing.assert_allclose(m1.cov, m0.cov, atol=1e-15)
    assert (m1.sigma_e, m1.tau) == pytest.approx((m0.sigma_e, m0.tau), abs=1e-14)


def test_sigma_ref_and_anchor_err_add_in_quadrature():
    p = predict_bias(4654, 3.78, 0.04, logg_source=SRC)
    p0 = predict_bias(4654, 3.78, 0.04, logg_source=SRC, sigma_ref=0.0, anchor_err=0.0)
    p2 = predict_bias(4654, 3.78, 0.04, logg_source=SRC, sigma_ref=0.2)
    assert p0.sigma**2 == pytest.approx(p.sigma**2 - 0.05**2 - 0.005**2, abs=1e-14)
    assert p2.sigma**2 == pytest.approx(p.sigma**2 - 0.05**2 + 0.2**2, abs=1e-14)
    assert p0.delta == p.delta == p2.delta


def test_subset_fit_changes_the_prediction(table):
    """A model passed in is the one used: fitted on stars without excess only, d is unconstrained
    by excess > 0.4 and the prediction for a disc star differs from the calibrated one."""
    sub = table[np.asarray(table["ks_minus_45"]) < 0.4]
    m = fit_bias_model(sub)
    assert m.n_stars < CALIBRATED_MODEL.n_stars
    p_sub = predict_bias(4600, 3.6, 1.0, logg_source=SRC, model=m)
    p_cal = predict_bias(4600, 3.6, 1.0, logg_source=SRC)
    assert abs(p_sub.delta - p_cal.delta) > 0.02
    expected = np.array([1.0, -0.4, -0.4, 1.0]) @ m.beta
    assert p_sub.delta == pytest.approx(expected, abs=1e-12)


# --- calibration and validation ----------------------------------------------------------------


def test_selection_gives_104_stars_in_12_clusters(table):
    keep = select_calibration_sample(table)
    assert len(table) == 229
    assert keep.sum() == 104
    assert np.unique(np.asarray(table["Name"])[keep]).size == 12


def test_embedded_coefficients_reproduce_the_fit(table):
    m = fit_bias_model(table)
    assert (m.n_stars, m.n_clusters) == (104, 12)
    np.testing.assert_allclose(m.beta, CALIBRATED_MODEL.beta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(m.cov, CALIBRATED_MODEL.cov, rtol=0, atol=1e-12)
    assert m.sigma_e == pytest.approx(CALIBRATED_MODEL.sigma_e, abs=1e-10)
    assert m.tau == pytest.approx(CALIBRATED_MODEL.tau, abs=1e-10)


def test_loco_beats_a_constant(table):
    r = leave_one_cluster_out(table)
    assert r["n_clusters"] == 12 and r["n_stars"] == 104
    assert r["rms_star"] < 0.09
    assert r["rms_star"] < 0.7 * r["baseline_rms_star"]
    # frozen prototype values (hub _measured.txt, 3 decimals)
    assert r["rms_star"] == pytest.approx(0.076, abs=5e-4)
    assert r["baseline_rms_star"] == pytest.approx(0.137, abs=5e-4)
    assert r["rms_cluster"] == pytest.approx(0.047, abs=5e-4)


@pytest.mark.parametrize(
    "star, delta, sigma",
    [((4654, 3.78, 0.04), -0.238, 0.060), ((5024, 3.20, 0.09), -0.320, 0.063)],
)
def test_ngc6383_members(star, delta, sigma):
    p = predict_bias(*star, logg_source=SRC)
    assert round(float(p.delta), 3) == delta
    assert round(float(p.sigma), 3) == sigma


def test_vectorized_matches_scalar():
    v = predict_bias([4654, 5024], [3.78, 3.20], [0.04, 0.09], logg_source="aspcap_dr19")
    assert v.delta.shape == (2,)
    for i, star in enumerate([(4654, 3.78, 0.04), (5024, 3.20, 0.09)]):
        s = predict_bias(*star, logg_source=SRC)
        assert v.delta[i] == pytest.approx(float(s.delta), abs=1e-15)
        assert v.sigma[i] == pytest.approx(float(s.sigma), abs=1e-15)


# --- guards -------------------------------------------------------------------------------------


@pytest.mark.parametrize("source", ["isochrone", "physical", "aspcap", "", None])
def test_logg_source_must_be_aspcap(source):
    with pytest.raises(ValueError, match="logg_source"):
        predict_bias(4654, 3.78, 0.04, logg_source=source)


def test_logg_source_is_required():
    with pytest.raises(TypeError):
        predict_bias(4654, 3.78, 0.04)


@pytest.mark.parametrize(
    "star",
    [(3800, 4.0, 0.1), (5600, 4.0, 0.1), (4600, 2.8, 0.1), (4600, 4.9, 0.1), (4600, 4.0, 2.5)],
)
def test_out_of_domain_warns(star):
    with pytest.warns(PMSBiasDomainWarning):
        p = predict_bias(*star, logg_source=SRC)
    assert bool(p.out_of_domain)


def test_inside_domain_does_not_warn(recwarn):
    predict_bias(4654, 3.78, 0.04, logg_source=SRC)
    assert not [w for w in recwarn if issubclass(w.category, PMSBiasDomainWarning)]


@pytest.mark.parametrize(
    "star, flag",
    [
        ((5024, 3.20, 0.09), True),  # NGC 6383 member: no excess, log g below 3.45
        ((4654, 3.78, 0.04), False),  # NGC 6383 member: no excess, log g inside the support
        ((4600, 3.20, 0.80), False),  # disc star at low log g: the calibrated regime
    ],
)
def test_extrapolation_flag(star, flag):
    assert bool(predict_bias(*star, logg_source=SRC).extrapolated) is flag


def test_negative_excess_counts_as_none():
    """max(Ks - [4.5], 0): a bluer-than-photosphere colour is noise, not a negative disc."""
    p_neg = predict_bias(4654, 3.78, -0.1, logg_source=SRC)
    p_zero = predict_bias(4654, 3.78, 0.0, logg_source=SRC)
    assert p_neg.delta == p_zero.delta
