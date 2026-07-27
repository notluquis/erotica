"""Regression tests for :mod:`erotica.analysis.structure`.

Guards backlog #8: ``_summarize_king_trace`` historically stored the posterior
*median* under keys named ``*_mean`` (a reporting-label bug). The fix adds
correctly named ``*_median`` keys and keeps ``*_mean`` as a deprecated
back-compat alias that holds the SAME median value, so paper figure notebooks
reproduce identical numbers. This test pins that contract on a deliberately
skewed posterior where the arithmetic mean and the median differ materially.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u

from erotica.analysis.structure import _summarize_king_trace


class _FakeVar:
    """Minimal stand-in for an ArviZ posterior DataArray (exposes ``.values``)."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)


def _skewed_sample(low, high, *, n=4000, power=6.0, seed=0):
    """Right-skewed positive sample on ``[low, high]`` with mean != median."""
    rng = np.random.default_rng(seed)
    # Raising U(0,1) to a power piles mass near 0 -> long right tail after scaling.
    unit = rng.uniform(0.0, 1.0, size=n) ** power
    return low + (high - low) * unit


def _make_trace():
    """Build a fake trace whose per-parameter marginals are strongly skewed."""
    posterior = {
        "k": _FakeVar(_skewed_sample(1.0, 20.0, seed=1)),
        "b": _FakeVar(_skewed_sample(0.01, 0.5, seed=2)),
        "R_c": _FakeVar(_skewed_sample(0.5, 6.0, seed=3)),
        "R_t": _FakeVar(_skewed_sample(10.0, 80.0, seed=4)),
        "sigma": _FakeVar(_skewed_sample(0.1, 2.0, seed=5)),
    }
    return SimpleNamespace(posterior=posterior)


def test_median_keys_hold_nanmedian_and_mean_aliases_match():
    trace = _make_trace()
    results = _summarize_king_trace(trace)

    for key, is_quantity in (("k", False), ("b", False), ("R_c", True), ("R_t", True)):
        arr = trace.posterior[key].values
        expected_median = float(np.nanmedian(arr))
        expected_mean = float(np.nanmean(arr))

        # Sanity: the synthetic marginal really is skewed (mean != median),
        # otherwise this test cannot distinguish the two estimators.
        assert abs(expected_mean - expected_median) > 0.05 * abs(expected_median), (
            f"{key} sample not skewed enough to demonstrate mean != median"
        )

        median_val = results[f"{key}_median"]
        mean_val = results[f"{key}_mean"]
        if is_quantity:
            assert isinstance(median_val, u.Quantity)
            assert isinstance(mean_val, u.Quantity)
            median_val = median_val.to_value(u.arcmin)
            mean_val = mean_val.to_value(u.arcmin)

        # 1. The correctly named *_median key equals np.nanmedian of the input.
        assert median_val == pytest.approx(expected_median)
        # 2. The deprecated *_mean alias holds the SAME median value (not the mean).
        assert mean_val == pytest.approx(expected_median)
        # 3. Regression guard: *_mean is the MEDIAN, and is NOT the arithmetic mean.
        assert mean_val != pytest.approx(expected_mean)


def test_king_std_uses_sigma_median():
    trace = _make_trace()
    results = _summarize_king_trace(trace)
    assert results["king_std"] == pytest.approx(float(np.nanmedian(trace.posterior["sigma"].values)))


def test_derived_quantities_computed_from_median():
    trace = _make_trace()
    results = _summarize_king_trace(trace)

    rc = float(np.nanmedian(trace.posterior["R_c"].values))
    rt = float(np.nanmedian(trace.posterior["R_t"].values))
    k = float(np.nanmedian(trace.posterior["k"].values))
    b_std = float(np.nanstd(trace.posterior["b"].values))
    b_med = float(np.nanmedian(trace.posterior["b"].values))
    bg_level = b_med + 3 * b_std

    assert results["C"] == pytest.approx(float(np.log(rt / rc)))
    assert results["bg_level"] == pytest.approx(bg_level)
    assert results["d_c"] == pytest.approx(float(1 + k / bg_level))


# ---------------------------------------------------------------------------
# Unbinned King point-process fit
#
# WHAT THESE ARE CHECKED AGAINST
# ------------------------------
# * the normalisation integral -> scipy.integrate.quad, independent quadrature
# * the fit -> parameters injected by the test into a simulated point process,
#   plus the convergence floor from ~/phd/methodology.md PART A (Vehtari+2021):
#   R-hat < 1.01, bulk-ESS > 400, zero divergences.
#
# There are no golden numbers here. Every target is either analytic or injected.
# ---------------------------------------------------------------------------

from erotica.analysis.structure import KingPriors, king_expected_count, king_unbinned  # noqa: E402

import importlib.util  # noqa: E402

requires_bayes_extra = pytest.mark.skipif(
    importlib.util.find_spec("pymc") is None, reason="requires the 'bayes' extra"
)

TRUE_KING = {"k": 6.0, "b": 0.05, "R_c": 4.0, "R_t": 30.0}
FIELD = 70.0


def _king_sigma(r, *, k, b, R_c, R_t):
    core = 1.0 / np.sqrt(1.0 + (np.asarray(r, float) / R_c) ** 2)
    edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
    return np.where(np.asarray(r, float) <= R_t, k * (core - edge) ** 2 + b, b)


def _sample_king(seed, *, field=FIELD, **params):
    """One realization of the King inhomogeneous Poisson point process."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, field, 20_001)
    intensity = 2.0 * np.pi * grid * _king_sigma(grid, **params)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (intensity[1:] + intensity[:-1]) * np.diff(grid))])
    n = rng.poisson(cdf[-1])
    return np.interp(rng.uniform(0.0, cdf[-1], n), cdf, grid)


@pytest.mark.parametrize(
    "k,b,R_c,R_t,field",
    [
        (6.0, 0.05, 4.0, 30.0, 70.0),   # the NGC 6383-like regime
        (1.0, 0.0, 1.0, 10.0, 10.0),    # no background, field edge exactly at R_t
        (20.0, 0.5, 0.7, 55.0, 70.0),   # tiny core, background-dominated
        (3.3, 0.01, 12.0, 14.0, 40.0),  # core comparable to R_t
        (5.0, 0.2, 3.0, 50.0, 20.0),    # field INSIDE R_t: integral must truncate
    ],
)
def test_king_normalisation_matches_quadrature(k, b, R_c, R_t, field):
    """Oracle: scipy.integrate.quad on the same integrand.

    A hand-derived closed form is exactly the kind of thing that is subtly wrong
    in one term and still looks plausible, so it is checked numerically rather
    than trusted -- including the case where the field stops short of R_t and the
    cluster term must truncate.
    """
    quad = pytest.importorskip("scipy.integrate").quad
    numeric = quad(
        lambda r: 2.0 * np.pi * r * _king_sigma(r, k=k, b=b, R_c=R_c, R_t=R_t),
        0.0, field, limit=400,
    )[0]
    closed = king_expected_count(k, b, R_c, R_t, field)
    assert closed == pytest.approx(numeric, rel=1e-8)


def test_normalisation_is_monotone_in_the_field_radius():
    """A larger field can only contain more stars -- a sign slip would break this."""
    fields = np.array([5.0, 15.0, 30.0, 45.0, 70.0])
    counts = [king_expected_count(**TRUE_KING, field_radius=f) for f in fields]
    assert np.all(np.diff(counts) > 0)


def test_background_only_normalisation_is_the_disc_area():
    """With k = 0 the expected count is exactly b * pi * R_f^2."""
    got = king_expected_count(0.0, 0.3, 4.0, 30.0, 70.0)
    assert got == pytest.approx(0.3 * np.pi * 70.0**2)


@requires_bayes_extra
def test_unbinned_fit_recovers_injected_parameters():
    """Oracle: the parameters the test injected, plus the PART A convergence floor."""
    import arviz as az
    import pandas as pd

    from erotica.analysis.inference import SamplingConfig

    radii = _sample_king(3, **TRUE_KING)
    res = king_unbinned(
        radii,
        field_radius=FIELD,
        sampling=SamplingConfig(draws=1000, tune=1000, chains=2, random_seed=1, progressbar=False),
    )

    for name in ("k", "b", "R_c", "R_t"):
        median = float(getattr(res[f"{name}_median"], "value", res[f"{name}_median"]))
        std = float(getattr(res[f"{name}_std"], "value", res[f"{name}_std"]))
        assert std > 0
        assert abs(median - TRUE_KING[name]) < 3 * std, (
            f"{name}: posterior {median:.3f} +/- {std:.3f} excludes injected {TRUE_KING[name]}"
        )

    summary = az.summary(res["king_trace"], var_names=["k", "b", "R_c", "R_t"])
    assert pd.to_numeric(summary["r_hat"], errors="coerce").max() < 1.01
    assert pd.to_numeric(summary["ess_bulk"], errors="coerce").min() > 400
    assert int(res["king_trace"].sample_stats["diverging"].values.sum()) == 0


@requires_bayes_extra
def test_unbinned_fit_returns_the_posterior_by_default():
    """Collapsing the posterior on exit is what stops uncertainty reaching dynamics."""
    from erotica.analysis.inference import SamplingConfig

    radii = _sample_king(5, **TRUE_KING)
    res = king_unbinned(
        radii, field_radius=FIELD,
        sampling=SamplingConfig(draws=200, tune=200, chains=1, random_seed=2, progressbar=False),
    )
    assert "king_trace" in res
    assert res["king_trace"].posterior["R_t"].values.size == 200
    assert res["n_stars"] == len(radii)


@requires_bayes_extra
def test_priors_do_not_depend_on_the_data():
    """The defect this replaces: RDP_bayesian derives every bound from the data.

    Prior-predictive draws must be identical for two completely different
    datasets. If any prior read the data, they would differ.
    """
    import pymc as pm

    def prior_draws(radii):
        with pm.Model():
            p = KingPriors()
            R_c = pm.HalfStudentT("R_c", nu=1, sigma=p.r_c_scale)
            pm.Deterministic("R_t", R_c + pm.HalfStudentT("dR", nu=1, sigma=p.r_t_scale))
            pm.HalfStudentT("k", nu=1, sigma=p.k_scale)
            pm.HalfStudentT("b", nu=1, sigma=p.b_scale)
            idata = pm.sample_prior_predictive(draws=300, random_seed=0)
        return np.asarray(idata.prior["R_t"].values).ravel()

    compact = _sample_king(11, k=50.0, b=0.001, R_c=0.5, R_t=5.0, field=FIELD)
    diffuse = _sample_king(12, **TRUE_KING)
    assert len(compact) > 0 and len(diffuse) > 0
    np.testing.assert_allclose(prior_draws(compact), prior_draws(diffuse))


@requires_bayes_extra
def test_prior_predictive_covers_the_plausible_range():
    """A prior that cannot generate the truth cannot recover it (PART A floor)."""
    import pymc as pm

    with pm.Model():
        p = KingPriors()
        R_c = pm.HalfStudentT("R_c", nu=1, sigma=p.r_c_scale)
        pm.Deterministic("R_t", R_c + pm.HalfStudentT("dR", nu=1, sigma=p.r_t_scale))
        idata = pm.sample_prior_predictive(draws=4000, random_seed=0)

    r_t = np.asarray(idata.prior["R_t"].values).ravel()
    r_c = np.asarray(idata.prior["R_c"].values).ravel()
    assert (r_t > TRUE_KING["R_t"]).mean() > 0.05
    assert (r_t < TRUE_KING["R_t"]).mean() > 0.05
    assert (r_c > TRUE_KING["R_c"]).mean() > 0.05
    assert np.all(r_t > r_c)  # ordering holds by construction, not by a bound


def test_stars_outside_the_field_are_rejected_not_silently_fitted():
    """The normalisation assumes completeness inside the disc; violating it biases R_t."""
    radii = np.concatenate([_sample_king(7, **TRUE_KING), [95.0]])
    with pytest.raises(ValueError, match="outside field_radius"):
        king_unbinned(radii, field_radius=FIELD)


def test_too_few_stars_is_an_error():
    with pytest.raises(ValueError, match="ten stars"):
        king_unbinned(np.array([1.0, 2.0, 3.0]), field_radius=FIELD)


@requires_bayes_extra
def test_half_cauchy_prior_is_built_without_the_pymc_halfcauchy_bug():
    """PyMC 6.1.0: ``HalfCauchy`` draws and logp disagree. Pin the workaround.

    ``HalfCauchy(beta)`` has the correct ``logp`` (scale = beta, matching its own
    docstring) but its random draws use ``1/beta`` as the scale. NUTS reads logp,
    so posteriors are right; ``sample_prior_predictive`` reads the draws, so every
    prior-predictive check built on it is wrong. ``HalfStudentT(nu=1, sigma)`` is
    the same distribution and is correct in both paths.

    Oracle: ``scipy.stats.halfcauchy``, for both the density and the IQR.
    """
    import pymc as pm
    import pytensor.tensor as pt
    from scipy import stats

    scale = 5.0
    ref = stats.halfcauchy(scale=scale)
    probe = np.array([0.3, 1.0, 3.0, 7.0])
    x = pt.dvector("x")

    dist = pm.HalfStudentT.dist(nu=1, sigma=scale)
    np.testing.assert_allclose(pm.logp(dist, x).eval({x: probe}), ref.logpdf(probe))

    draws = pm.draw(dist, draws=200_000, random_seed=0)
    iqr = float(np.subtract(*np.percentile(draws, [75, 25])))
    assert iqr == pytest.approx(ref.ppf(0.75) - ref.ppf(0.25), rel=0.02)

    # And the bug itself, so we notice when a PyMC upgrade fixes it and this
    # workaround can be dropped.
    buggy = pm.draw(pm.HalfCauchy.dist(scale), draws=100_000, random_seed=0)
    buggy_iqr = float(np.subtract(*np.percentile(buggy, [75, 25])))
    assert buggy_iqr == pytest.approx(iqr / scale**2, rel=0.1), (
        "pm.HalfCauchy draws no longer show the 1/beta scale bug -- "
        "re-check whether HalfStudentT is still needed"
    )
