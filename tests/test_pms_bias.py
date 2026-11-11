"""Tests for erotica.metallicity.pms_bias (ASPCAP [Fe/H] bias in young K stars).

Oracles:
- the anchor and sigma_ref arguments enter by exact algebra (a shift of ``a``; a quadrature term),
  so their non-default values have closed-form consequences;
- the hub prototype ``agent-findings/scripts/apogee_pms_bias_correction_2026_11_09.py`` and its
  ``_measured.txt`` (2026-11-09): N 104 in 12 clusters, LOCO RMS 0.076 against 0.137 for a constant,
  delta = -0.238 +- 0.060 and -0.320 +- 0.063 for the two NGC 6383 members (OLS control; sigma 0.061 and
  0.065 since the cluster-aware covariance of the PR #61 review). Those are an
  independent implementation's outputs, frozen; the coefficients embedded in the module are refitted here;
- the hub hierarchical run ``agent-findings/scripts/pms_bias_hierarchical_2026_11_10.py`` and its
  ``_fit_measured.txt`` / finding ``pms-bias-hierarchical-2026-11-10.md`` (2026-11-10, hub commit 0669a745):
  posterior means, tau quantiles and the members' Monte Carlo predictive, computed there by sampling u_new
  and r; here the same moments come in closed form from the shipped draws, and a refit with the package's
  own model builder must land on the shipped posterior.
"""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from erotica.metallicity import (
    CALIBRATED_MODEL,
    HierarchicalPosterior,
    PMSBiasDomainWarning,
    fit_bias_model,
    fit_hierarchical_model,
    leave_one_cluster_out,
    load_posterior,
    predict_bias,
    predictive_draws,
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
    p_cal = predict_bias(4600, 3.6, 1.0, logg_source=SRC, model=CALIBRATED_MODEL)
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


# Hub oracle (agent-findings/pms-bias-hierarchical-2026-11-10.md sec. 0 and 2.1; _fit_measured.txt): the M1
# posterior predictive for a new cluster, delta = x beta + u_new + r, with u_new ~ N(0, tau) and r the zero-point
# shift (sigma_ref (+) anchor error). The hub drew u_new and r by Monte Carlo (8000 draws, its own seed) and
# printed 3 decimals; the package takes the same moments in closed form from the same draws, so the two agree
# within the MC error of a standard deviation (~0.0006) plus rounding: tolerance 0.002 on both.
MEMBERS_M1 = [((4654, 3.78, 0.04), -0.230, 0.073), ((5024, 3.20, 0.09), -0.323, 0.077)]


@pytest.mark.parametrize("star, delta, sigma", MEMBERS_M1)
def test_ngc6383_members_hierarchical_default(star, delta, sigma):
    """D26-64: the default is the hierarchical posterior. The OLS sigma (0.060, 0.063) fails this by 0.013."""
    p = predict_bias(*star, logg_source=SRC)
    assert float(p.sigma) == pytest.approx(sigma, abs=0.002)
    assert float(p.delta) == pytest.approx(delta, abs=0.002)


@pytest.mark.parametrize(
    "star, delta, sigma",
    [((4654, 3.78, 0.04), -0.238, 0.061), ((5024, 3.20, 0.09), -0.320, 0.065)],
)
def test_ngc6383_members_ols_control(star, delta, sigma):
    """``model=CALIBRATED_MODEL``: the prototype's delta; sigma 0.060 / 0.063 in the prototype, 0.061 /
    0.065 with the cluster-aware covariance (PR #61 review)."""
    p = predict_bias(*star, logg_source=SRC, model=CALIBRATED_MODEL)
    assert round(float(p.delta), 3) == delta
    assert round(float(p.sigma), 3) == sigma


def test_hierarchical_sigma_exceeds_ols_by_the_tau_uncertainty():
    """The point of D26-64: with tau a posterior (median 0.043) instead of a known 0.028, the
    members' sigma grows: by 0.013 against the prototype (hub 0.073 - 0.060, 0.077 - 0.063) and by
    0.011-0.012 against the least squares with the cluster-aware covariance (0.0607, 0.0653)."""
    stars = ([4654, 5024], [3.78, 3.20], [0.04, 0.09])
    h = predict_bias(*stars, logg_source=SRC)
    o = predict_bias(*stars, logg_source=SRC, model=CALIBRATED_MODEL)
    np.testing.assert_allclose(h.sigma - o.sigma, [0.0118, 0.0108], atol=0.001)


@pytest.mark.parametrize(
    "star, delta, sigma", [((4654, 3.78, 0.04), -0.255, 0.082), ((5024, 3.20, 0.09), -0.376, 0.086)]
)
def test_student_t_arm(star, delta, sigma):
    """D26-63: the Student-t arm is shipped and opt-in (hub _fit_measured.txt, B-t rows)."""
    p = predict_bias(*star, logg_source=SRC, model=load_posterior("student_t"))
    assert float(p.sigma) == pytest.approx(sigma, abs=0.002)
    assert float(p.delta) == pytest.approx(delta, abs=0.002)


def test_shipped_posterior_is_the_hub_m1():
    """Hub sec. 2.1 table and _fit_measured.txt: a -0.165 +- 0.020, b +0.208 +- 0.033, c +0.047 +- 0.014,
    d -0.143 +- 0.029; tau median 0.043, 68 % [0.031; 0.060], P(tau < 0.02) = 0.020; sigma median 0.061."""
    post = load_posterior()
    assert post.likelihood == "normal" and post.beta.shape == (8000, 4)
    np.testing.assert_allclose(post.beta_mean, [-0.165, 0.208, 0.047, -0.143], atol=6e-4)
    np.testing.assert_allclose(post.beta.std(axis=0), [0.020, 0.033, 0.014, 0.029], atol=6e-4)
    np.testing.assert_allclose(
        np.percentile(post.tau, [16, 50, 84]), [0.031, 0.043, 0.060], atol=6e-4
    )
    assert np.mean(post.tau < 0.02) == pytest.approx(0.020, abs=6e-4)
    assert np.median(post.sigma) == pytest.approx(0.061, abs=6e-4)
    with pytest.raises(ValueError, match="likelihood"):
        load_posterior("cauchy")


def test_closed_form_matches_predictive_draws():
    """predict_bias's moments are the moments of predictive_draws (same draws, sampled u_new and r).
    MC error of a sd with 8000 draws ~0.0006; of a mean ~0.0008."""
    for star in [(4654, 3.78, 0.04), (5024, 3.20, 0.09), (4300, 4.3, 0.8)]:
        p = predict_bias(*star, logg_source=SRC)
        d = predictive_draws(*star, logg_source=SRC, rng=20261111)
        assert d.shape == (8000,)
        assert d.mean() == pytest.approx(float(p.delta), abs=0.003)
        assert d.std() == pytest.approx(float(p.sigma), abs=0.003)


def test_members_share_the_cluster_offset():
    """Hub _fit_measured.txt: the mean of the two members is -0.277 +- 0.075, not sigma/sqrt(2) ~ 0.053:
    u_new and r are common to stars of one cluster. Independent draws per star would give ~0.053."""
    d = predictive_draws([4654, 5024], [3.78, 3.20], [0.04, 0.09], logg_source=SRC, rng=1)
    assert d.shape == (8000, 2)
    mean2 = d.mean(axis=1)
    assert mean2.mean() == pytest.approx(-0.277, abs=0.003)
    assert mean2.std() == pytest.approx(0.075, abs=0.003)


def test_predictive_draws_needs_draws_and_bad_models_raise():
    with pytest.raises(TypeError, match="HierarchicalPosterior"):
        predictive_draws(4654, 3.78, 0.04, logg_source=SRC, model=CALIBRATED_MODEL)
    with pytest.raises(TypeError, match="model must be"):
        predict_bias(4654, 3.78, 0.04, logg_source=SRC, model=object())
    with pytest.raises(ValueError, match="logg_source"):
        predictive_draws(4654, 3.78, 0.04, logg_source="isochrone")


def test_vendored_errors_select_with_the_sample(table):
    """fe_h_err is finite for the 104 calibration stars (the hub loader asserted the same join)."""
    keep = select_calibration_sample(table)
    e = np.asarray(table["fe_h_err"], float)[keep]
    assert np.all(np.isfinite(e)) and e.min() == pytest.approx(0.0059, abs=1e-6)
    assert e.max() == pytest.approx(0.0178, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    "kwargs, shipped, tau_median",
    [
        ({}, "normal", 0.043),
        ({"likelihood": "student_t"}, "student_t", 0.055),
        # B-tau 0.05 of the hub (sec. 2.1): b +0.207, tau median 0.041 -- the prior scale barely moves it
        ({"tau_scale": 0.05}, "normal", 0.041),
    ],
)
def test_refit_lands_on_the_shipped_posterior(table, kwargs, shipped, tau_median):
    """The package's own model builder, sampled with nutpie, reproduces the hub posterior: posterior
    means of (a, b, c, d) within 0.01 and the tau median within 0.004 (MC + seed; hub ESS >= 1528).
    Gate R-hat <= 1.01, ESS >= 400, divergences < 1 % (tests/AGENTS.md; hub M1 had 1 in 8000). With
    2 x 2000 draws the tau_scale=0.05 case gave R-hat 1.0117 on 2026-11-11: the hub saw the same edge at
    2 x 1000 (ESS ~450 on a and tau) and ran 2 x 4000 for that reason."""
    post = fit_hierarchical_model(table, **kwargs)  # the hub run: 2 x 4000 after 2000
    assert isinstance(post, HierarchicalPosterior)
    assert (post.n_stars, post.n_clusters) == (104, 12)
    dg = post.diagnostics
    assert dg["rhat_max"] <= 1.01 and dg["ess_bulk_min"] >= 400 and dg["ess_tail_min"] >= 400
    assert dg["divergences"] < 0.01 * dg["n_draws"]
    ref = load_posterior(shipped)
    np.testing.assert_allclose(post.beta_mean, ref.beta_mean, atol=0.01)
    assert np.median(post.tau) == pytest.approx(tau_median, abs=0.004)


@pytest.mark.slow
def test_refit_tau_scale_reaches_the_prior(table):
    """The hub's B-tau 0.05 moves tau's median by 0.002, inside the refit tolerance, so it cannot show
    that ``tau_scale`` is used. HalfNormal(0.01) can: P(tau > 0.03) = 0.003 under it, while the data
    alone put the median at 0.043; the posterior median must fall well below that."""
    post = fit_hierarchical_model(table, tau_scale=0.01)
    assert np.median(post.tau) < 0.03


def test_fit_requires_errors(table):
    t = table.copy()
    del t["fe_h_err"]
    with pytest.raises(KeyError):
        fit_hierarchical_model(t)
    with pytest.raises(ValueError, match="likelihood"):
        fit_hierarchical_model(table, likelihood="laplace")


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


# --- review of PR #61 (2026-11-11): shared cluster offsets, masks, degenerate samples ------------


def _replicate(table, k):
    from astropy.table import vstack

    return vstack([table] * k)


def test_ols_covariance_keeps_the_shared_cluster_offset(table):
    """Codex P2: tau is shared by the stars of a cluster, so stacking every star k times must not
    average it away. The intercept variance has a part that does not shrink with k: with the
    cluster-aware (sandwich) covariance, k = 10 keeps the intercept sd above 70 % of k = 1; with
    tau treated as per-star noise it falls to ~1/sqrt(10) = 32 %."""
    m1 = fit_bias_model(table)
    m10 = fit_bias_model(_replicate(table, 10))
    assert m10.tau > 0.02
    assert np.sqrt(m10.cov[0, 0]) > 0.7 * np.sqrt(m1.cov[0, 0])


def test_ols_covariance_is_the_cluster_sandwich(table):
    """Closed form: cov = A^-1 X^T V X A^-1, A = X^T X, V = sigma_e^2 I + tau^2 (same-cluster blocks)."""
    from erotica.metallicity.pms_bias import _design

    keep = select_calibration_sample(table)
    t = table[keep]
    m = fit_bias_model(table)
    X = _design(
        np.asarray(t["teff"], float),
        np.asarray(t["logg"], float),
        np.asarray(t["ks_minus_45"], float),
    )
    nm = np.asarray(t["Name"]).astype(str)
    V = m.sigma_e**2 * np.eye(len(t)) + m.tau**2 * (nm[:, None] == nm[None, :])
    Ai = np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(m.cov, Ai @ X.T @ V @ X @ Ai, rtol=1e-10, atol=0)


def test_hierarchical_default_never_drops_below_tau():
    """The default path models the offset as u_c: a new cluster's prediction carries E[tau^2] whatever
    the calibration size, so sigma >= sqrt(E[tau^2] + sigma_ref^2 + anchor_err^2) everywhere."""
    post = load_posterior()
    floor = np.sqrt(post.tau2_mean + 0.05**2 + 0.005**2)
    p = predict_bias(np.linspace(4000, 5500, 7), 4.0, 0.1, logg_source=SRC)
    assert np.all(p.sigma >= floor) and np.sqrt(post.tau2_mean) > 0.04


def test_masked_entries_are_missing_not_their_backing_values(table):
    """Codex P2: a masked ks_minus_45 with a finite value underneath must count as missing."""
    from astropy.table import MaskedColumn

    t = table.copy()
    keep0 = select_calibration_sample(t)
    rows = np.flatnonzero(keep0)[:5]
    ks = MaskedColumn(np.asarray(t["ks_minus_45"], float), mask=np.zeros(len(t), bool))
    ks.mask[rows] = True
    t["ks_minus_45"] = ks
    ref = table.copy()
    ref["ks_minus_45"][rows] = np.nan
    np.testing.assert_array_equal(select_calibration_sample(t), select_calibration_sample(ref))
    assert select_calibration_sample(t).sum() < keep0.sum()
    np.testing.assert_allclose(fit_bias_model(t).beta, fit_bias_model(ref).beta, rtol=0, atol=0)


def test_singleton_clusters_raise(table):
    """Gemini: with select=False and one star per cluster there is no within-cluster scatter."""
    t = table[select_calibration_sample(table)]
    _, first = np.unique(np.asarray(t["Name"]).astype(str), return_index=True)
    with pytest.raises(ValueError, match="more than one star"):
        fit_bias_model(t[np.sort(first)], select=False)


def test_rank_deficient_sample_raises(table):
    """Gemini: fewer stars than coefficients (or a design without rank 4) cannot be fitted."""
    t = table[select_calibration_sample(table)]
    with pytest.raises(ValueError, match="rank"):
        fit_bias_model(t[:3], select=False)
    flat = t.copy()
    flat["ks_minus_45"] = 0.0  # d unidentified: the excess column is all zero
    with pytest.raises(ValueError, match="rank"):
        fit_bias_model(flat, select=False)


def test_loco_needs_three_clusters(table):
    """Gemini: leaving one cluster out needs >= 2 clusters left to estimate tau, so >= 3 in all."""
    t = table[select_calibration_sample(table)]
    nm = np.asarray(t["Name"]).astype(str)
    two = t[np.isin(nm, np.unique(nm)[:2])]
    with pytest.raises(ValueError, match="clusters"):
        leave_one_cluster_out(two, select=False)
