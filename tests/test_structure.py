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

from erotica.analysis.structure import (  # noqa: E402
    KingPriors,
    king_expected_count,
    king_expected_count_weighted,
    king_unbinned,
)

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

    from erotica.analysis.structure import _king_model

    def prior_draws(radii):
        # Build the ACTUAL shipping model. An earlier version of this test
        # reimplemented the prior block inline, so it tested a copy: a mutation
        # putting `sigma=np.std(r)` back into _king_model -- the exact "data used
        # twice" defect the P01 referee raised -- left this test green.
        with _king_model(pm, np.asarray(radii, float), FIELD, KingPriors(), None, None):
            idata = pm.sample_prior_predictive(draws=400, random_seed=0)
        return np.asarray(idata.prior["R_t"].values).ravel()

    compact = _sample_king(11, k=50.0, b=0.001, R_c=0.5, R_t=5.0, field=FIELD)
    diffuse = _sample_king(12, **TRUE_KING)
    # the two datasets must differ enough that any data dependence would show
    assert len(compact) > 0 and len(diffuse) > 0
    # Guard the guard: if the two samples were similar, agreement would prove
    # nothing. They differ by 15x in count and 36x in median radius.
    assert len(diffuse) > 5 * len(compact)
    assert np.median(diffuse) > 10 * np.median(compact)
    np.testing.assert_allclose(prior_draws(compact), prior_draws(diffuse))


@requires_bayes_extra
def test_prior_predictive_covers_the_plausible_range():
    """A prior that cannot generate the truth cannot recover it (PART A floor)."""
    import pymc as pm

    from erotica.analysis.structure import _king_model

    # Again: the real model, not a reimplementation of its priors.
    radii = _sample_king(13, **TRUE_KING)
    with _king_model(pm, radii, FIELD, KingPriors(), None, None):
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

    # And the bug itself. It is FIXED in pytensor 3.2.4 (PR pytensor#2309, this project's own
    # contribution), so assert the broken behaviour below that version and the correct behaviour
    # at or above it. Without the version split this test fails on upgrade while asserting
    # something that is no longer true, which reads as a regression rather than as the fix landing.
    import pytensor
    from packaging.version import Version

    buggy = pm.draw(pm.HalfCauchy.dist(scale), draws=100_000, random_seed=0)
    buggy_iqr = float(np.subtract(*np.percentile(buggy, [75, 25])))
    if Version(pytensor.__version__) >= Version("3.2.4"):
        assert buggy_iqr == pytest.approx(iqr, rel=0.1), (
            f"pytensor {pytensor.__version__} carries the CauchyRV fix, but HalfCauchy draws still "
            "disagree with HalfStudentT -- the fix regressed"
        )
    else:
        assert buggy_iqr == pytest.approx(iqr / scale**2, rel=0.1), (
            f"pytensor {pytensor.__version__} is below 3.2.4 but no longer shows the 1/beta scale "
            "bug -- re-check whether the HalfStudentT workaround is still needed"
        )


# ---------------------------------------------------------------------------
# Selection-function-aware fitting
# ---------------------------------------------------------------------------


def _crowding_completeness(r, floor=0.35, scale=6.0):
    """Toy radial completeness rising outward, as crowding-limited surveys do.

    Gaia is *least* complete where the star density is highest, so ``S`` is
    suppressed in the core and recovers in the field. This is the direction
    Cantat-Gaudin et al. (2023) model and validate against HST in globular
    clusters; the functional form here is a stand-in, the *sign* is the point.
    """
    return floor + (1.0 - floor) * (1.0 - np.exp(-np.asarray(r, float) / scale))


def test_weighted_normalisation_reduces_to_the_closed_form_when_complete():
    """Oracle: the analytic integral. S == 1 must reproduce it exactly."""
    closed = king_expected_count(**TRUE_KING, field_radius=FIELD)
    quad = king_expected_count_weighted(
        **TRUE_KING, field_radius=FIELD, completeness=lambda r: np.ones_like(r)
    )
    assert quad == pytest.approx(closed, rel=1e-6)


def test_weighted_normalisation_matches_quadrature_for_varying_completeness():
    """Oracle: scipy.integrate.quad on the same integrand, with S(r) inside."""
    integrate = pytest.importorskip("scipy.integrate")
    numeric = integrate.quad(
        lambda r: 2 * np.pi * r * _king_sigma(r, **TRUE_KING) * _crowding_completeness(r),
        0.0, FIELD, limit=400,
    )[0]
    got = king_expected_count_weighted(
        **TRUE_KING, field_radius=FIELD, completeness=_crowding_completeness
    )
    assert got == pytest.approx(numeric, rel=1e-6)


def test_incomplete_survey_expects_fewer_stars():
    """Sanity direction: S <= 1 everywhere can only lower the expected count."""
    full = king_expected_count(**TRUE_KING, field_radius=FIELD)
    thinned = king_expected_count_weighted(
        **TRUE_KING, field_radius=FIELD, completeness=_crowding_completeness
    )
    assert 0 < thinned < full


def test_completeness_outside_zero_one_is_rejected():
    with pytest.raises(ValueError, match="probability"):
        king_expected_count_weighted(
            **TRUE_KING, field_radius=FIELD, completeness=lambda r: 1.5 * np.ones_like(r)
        )


def test_misaligned_completeness_array_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        king_expected_count_weighted(
            **TRUE_KING, field_radius=FIELD, completeness=np.ones(7)
        )


@requires_bayes_extra
def test_selection_correction_removes_the_bias_it_is_meant_to_remove():
    """The load-bearing test: does correcting for S(r) actually recover truth?

    A cluster is simulated, then **thinned** by a radially varying completeness --
    exactly what a crowding-limited survey does to it. The thinned sample is fit
    twice. The uncorrected fit sees a core that has been preferentially emptied,
    so it must be biased; the corrected fit must recover the injected ``R_c``.

    Oracle: the injected ``R_c``, and the *relative* accuracy of the two fits. A
    test that only checked the corrected fit could pass on a stopped clock -- this
    also requires that there was a bias there to remove.
    """
    from erotica.analysis.inference import SamplingConfig

    rng = np.random.default_rng(1234)
    radii = _sample_king(99, **TRUE_KING)
    keep = rng.uniform(size=radii.size) < _crowding_completeness(radii)
    observed = radii[keep]
    assert 0.4 * radii.size < observed.size < 0.95 * radii.size, "thinning did nothing useful"

    cfg = dict(draws=1500, tune=1000, chains=2, random_seed=7, progressbar=False)
    naive = king_unbinned(observed, field_radius=FIELD, sampling=SamplingConfig(**cfg))
    fixed = king_unbinned(
        observed, field_radius=FIELD, completeness=_crowding_completeness,
        sampling=SamplingConfig(**cfg),
    )

    truth = TRUE_KING["R_c"]
    naive_rc = float(naive["R_c_median"].value)
    fixed_rc = float(fixed["R_c_median"].value)

    assert fixed["completeness_corrected"] is True
    assert naive["completeness_corrected"] is False
    # the corrected fit recovers the injected core radius
    assert abs(fixed_rc - truth) < 3 * float(fixed["R_c_std"].value)
    # and it is closer to truth than the uncorrected one -- i.e. a bias existed
    assert abs(fixed_rc - truth) < abs(naive_rc - truth)
    # the uncorrected fit over-estimates R_c: the core was preferentially emptied
    assert naive_rc > truth


# ---------------------------------------------------------------------------
# EFF profile and Bayes-factor model comparison
# ---------------------------------------------------------------------------

from erotica.analysis.structure import (  # noqa: E402
    EFFPriors,
    compare_radial_profiles,
    corona_surface_density,
    eff_expected_count,
    eff_surface_density,
    eff_unbinned,
    king_profile,
)

TRUE_EFF = {"k": 8.0, "b": 0.05, "a": 3.0, "gamma": 2.8}


def _sample_profile(seed, sigma_fn, field=FIELD):
    """One realization of an inhomogeneous Poisson process with intensity 2*pi*r*Sigma(r)."""
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, field, 20_001)
    intensity = 2.0 * np.pi * grid * sigma_fn(grid)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (intensity[1:] + intensity[:-1]) * np.diff(grid))])
    n = rng.poisson(cdf[-1])
    return np.interp(rng.uniform(0.0, cdf[-1], n), cdf, grid)


@pytest.mark.parametrize(
    "k,b,a,gamma,field",
    [
        (6.0, 0.05, 4.0, 3.0, 70.0),
        (1.0, 0.0, 1.0, 4.0, 10.0),    # gamma=4 is Plummer
        (20.0, 0.5, 0.7, 2.5, 70.0),
        (3.3, 0.01, 12.0, 5.0, 40.0),
        (2.0, 0.1, 3.0, 2.0, 50.0),    # gamma=2 exactly: the logarithmic singularity
        (2.0, 0.1, 3.0, 2.0000001, 50.0),  # and just beside it
    ],
)
def test_eff_normalisation_matches_quadrature(k, b, a, gamma, field):
    """Oracle: scipy.integrate.quad. Includes gamma = 2, where the closed form is singular."""
    integrate = pytest.importorskip("scipy.integrate")
    numeric = integrate.quad(
        lambda r: 2 * np.pi * r * eff_surface_density(r, k=k, b=b, a=a, gamma=gamma),
        0.0, field, limit=400,
    )[0]
    assert eff_expected_count(k, b, a, gamma, field) == pytest.approx(numeric, rel=1e-8)


def test_eff_normalisation_is_continuous_across_gamma_two():
    """The two branches must agree in the limit, not merely each be finite."""
    vals = [eff_expected_count(4.0, 0.02, 3.0, g, 60.0)
            for g in (2.0 - 1e-4, 2.0 - 1e-8, 2.0, 2.0 + 1e-8, 2.0 + 1e-4)]
    assert np.all(np.isfinite(vals))
    assert max(vals) - min(vals) < 1e-3 * abs(vals[2])


def test_eff_has_no_tidal_cutoff():
    """The defining difference from King: density stays above background forever."""
    far = eff_surface_density([200.0, 2000.0, 20000.0], **TRUE_EFF)
    assert np.all(far > TRUE_EFF["b"])
    assert np.all(np.diff(far) < 0)  # monotonically declining, never truncating


@requires_bayes_extra
def test_eff_recovers_injected_parameters():
    """Oracle: the parameters the test injected, plus the PART A convergence floor."""
    import arviz as az
    import pandas as pd

    from erotica.analysis.inference import SamplingConfig

    # EFF has a genuine k / a / gamma degeneracy -- amplitude, scale radius and
    # slope trade off against one another -- so it needs more adaptation than the
    # King fit to clear R-hat < 1.01. At tune=1000, chains=2 it lands on exactly
    # 1.01; at these settings it is 1.0000 with bulk-ESS ~1600.
    radii = _sample_profile(21, lambda r: eff_surface_density(r, **TRUE_EFF))
    res = eff_unbinned(
        radii, field_radius=FIELD,
        sampling=SamplingConfig(draws=1500, tune=2000, chains=4, target_accept=0.95,
                                random_seed=5, progressbar=False),
    )
    for name in ("k", "b", "a", "gamma"):
        med = float(getattr(res[f"{name}_median"], "value", res[f"{name}_median"]))
        sd = float(getattr(res[f"{name}_std"], "value", res[f"{name}_std"]))
        assert abs(med - TRUE_EFF[name]) < 3 * sd, f"{name}: {med:.3f}+/-{sd:.3f} vs {TRUE_EFF[name]}"

    summary = az.summary(res["eff_trace"], var_names=["k", "b", "a", "gamma"])
    assert pd.to_numeric(summary["r_hat"], errors="coerce").max() < 1.01
    assert int(res["eff_trace"].sample_stats["diverging"].values.sum()) == 0


@requires_bayes_extra
def test_plummer_is_eff_with_gamma_fixed_to_four():
    from erotica.analysis.inference import SamplingConfig

    radii = _sample_profile(22, lambda r: eff_surface_density(r, k=5.0, b=0.02, a=4.0, gamma=4.0))
    res = eff_unbinned(
        radii, field_radius=FIELD, gamma=4.0,
        sampling=SamplingConfig(draws=400, tune=400, chains=1, random_seed=6, progressbar=False),
    )
    assert res["model"] == "plummer"
    assert res["gamma_median"] == pytest.approx(4.0)   # held fixed, not fitted
    assert res["gamma_std"] == pytest.approx(0.0)


def test_king_at_infinite_tidal_radius_is_eff_with_gamma_two():
    """The two families genuinely overlap, which bounds what a Bayes factor can do.

    As ``R_t -> inf`` the King edge term ``c = (1+(R_t/R_c)^2)^(-1/2)`` goes to
    zero and ``Sigma -> k/(1+(r/R_c)^2)``, which is exactly EFF with ``gamma=2``.
    So no amount of data separates King from EFF near that corner, and a model
    comparison must be posed away from it to mean anything.
    """
    r = np.linspace(0.1, 60.0, 25)
    king = _king_sigma(r, k=5.0, b=0.0, R_c=4.0, R_t=1e6)
    eff = eff_surface_density(r, k=5.0, b=0.0, a=4.0, gamma=2.0)
    np.testing.assert_allclose(king, eff, rtol=1e-3)


@requires_bayes_extra
@pytest.mark.slow
def test_bayes_factor_identifies_the_generating_model():
    """The load-bearing test: does the comparison pick the model the data came from?

    Run in both directions. A comparison that always preferred the more flexible
    model, or always the simpler one, would fail one of the two.

    The parameters are chosen *away from the overlap* pinned by
    ``test_king_at_infinite_tidal_radius_is_eff_with_gamma_two``: the King case
    has a sharp truncation well inside the field and essentially no background,
    and the EFF case has ``gamma=5``, far from the ``gamma=2`` limit where King
    can imitate it. With `gamma=2.8` and a large `R_t` the two are genuinely
    indistinguishable and the Bayes factor correctly returns ~0.
    """
    king_truth = {"k": 6.0, "b": 0.002, "R_c": 4.0, "R_t": 25.0}
    eff_truth = {"k": 8.0, "b": 0.002, "a": 3.0, "gamma": 5.0}
    king_data = _sample_king(41, **king_truth)
    eff_data = _sample_profile(42, lambda r: eff_surface_density(r, **eff_truth))

    opts = dict(field_radius=FIELD, models=("king", "eff"), draws=1000, chains=4, random_seed=1)
    king_res = compare_radial_profiles(king_data, **opts)
    eff_res = compare_radial_profiles(eff_data, **opts)

    assert king_res["best"] == "king", king_res["log_marginal_likelihood"]
    assert eff_res["best"] == "eff", eff_res["log_marginal_likelihood"]
    # Each verdict must exceed the chain-to-chain scatter of the evidence itself,
    # otherwise it is sampling noise dressed as a conclusion.
    assert king_res["resolvable"]["eff"]
    assert eff_res["resolvable"]["king"]
    assert king_res["log_bayes_factor_vs_best"]["eff"] < -1.0
    assert eff_res["log_bayes_factor_vs_best"]["king"] < -1.0


def test_compare_rejects_unknown_models():
    with pytest.raises(ValueError, match="unknown models"):
        compare_radial_profiles(np.linspace(1, 50, 100), field_radius=FIELD, models=("king", "nfw"))


@requires_bayes_extra
def test_eff_accepts_a_completeness_and_reduces_to_the_plain_fit_when_flat():
    """EFF must take the same S(r) hook as King -- it did not, and the pipeline's
    own selection (~35x Gaia's for NGC 6383) could not be applied to it at all.

    Oracle: a FLAT completeness must leave every shape parameter unchanged. That
    is exact, not approximate: Sigma is homogeneous of degree 1 in (k, b), so a
    constant S is absorbed by k -> S*k, b -> S*b and cannot bias a shape.
    """
    from erotica.analysis.inference import SamplingConfig

    radii = _sample_profile(51, lambda r: eff_surface_density(r, **TRUE_EFF))
    cfg = SamplingConfig(draws=800, tune=1000, chains=2, target_accept=0.95,
                         random_seed=9, progressbar=False)
    plain = eff_unbinned(radii, field_radius=FIELD, sampling=cfg)
    flat = eff_unbinned(radii, field_radius=FIELD, completeness=np.full(256, 0.6),
                        sampling=cfg)

    assert plain["completeness_corrected"] is False
    assert flat["completeness_corrected"] is True
    # shape parameters unchanged...
    assert float(flat["gamma_median"]) == pytest.approx(float(plain["gamma_median"]), rel=0.05)
    assert float(flat["a_median"].value) == pytest.approx(float(plain["a_median"].value), rel=0.08)
    # ...while the amplitude absorbs the constant, k -> k / S
    assert float(flat["k_median"]) > 1.3 * float(plain["k_median"])


def test_eff_completeness_validation():
    with pytest.raises(ValueError, match="within"):
        eff_unbinned(np.linspace(1, 50, 40), field_radius=FIELD,
                     completeness=np.full(256, 1.5))
    with pytest.raises(ValueError, match="shape"):
        eff_unbinned(np.linspace(1, 50, 40), field_radius=FIELD, completeness=np.ones(9))


# ---------------------------------------------------------------------------
# King + uniform-sphere corona (Seleznev 2016; Danilov & Putkov 2012)
# ---------------------------------------------------------------------------


def test_corona_normalisation_matches_quadrature():
    """Oracle: numerical integration of the profile this closed form claims to integrate.

    Covers the three regimes that differ algebraically -- corona inside the field,
    corona larger than the field (so the integral truncates at the field edge), and
    the exact boundary.
    """
    from scipy.integrate import quad

    from erotica.analysis.structure import corona_expected_count, corona_surface_density

    for delta_f, R_2, field in [(0.02, 40.0, 70.0), (0.02, 90.0, 70.0),
                                (5.0, 10.0, 70.0), (0.02, 70.0, 70.0)]:
        numeric = quad(
            lambda x: 2 * np.pi * x * float(
                corona_surface_density(np.array([x]), delta_f=delta_f, R_2=R_2)[0]
            ),
            0.0, field, limit=400,
        )[0]
        closed = corona_expected_count(delta_f, R_2, field)
        assert abs(closed / numeric - 1) < 1e-8, (delta_f, R_2, field, closed, numeric)


def test_corona_enclosing_field_is_volume_times_density():
    """Oracle: when the field encloses the corona the answer is elementary geometry,
    (4/3)pi R^3 * delta -- a check on the algebra that needs no integration at all."""
    from erotica.analysis.structure import corona_expected_count

    delta_f, R_2 = 0.03, 25.0
    expected = (4.0 / 3.0) * np.pi * R_2**3 * delta_f
    assert corona_expected_count(delta_f, R_2, 70.0) == pytest.approx(expected, rel=1e-12)


def test_corona_becomes_a_flat_background_when_large():
    """The near-degeneracy the model exists to expose, stated as a limit.

    For ``R_2 >> r`` the projected density tends to the constant ``2 delta_f R_2``.
    This is why a corona wider than the footprint cannot be distinguished from a
    flat background, and why an unconstrained ``R_2`` posterior is itself the result.
    """
    from erotica.analysis.structure import corona_surface_density, king_profile

    r = np.linspace(0.1, 70.0, 50)
    sigma = corona_surface_density(r, delta_f=1e-4, R_2=1e5)
    assert np.ptp(sigma) / sigma.mean() < 1e-6
    assert sigma.mean() == pytest.approx(2 * 1e-4 * 1e5, rel=1e-5)


def test_corona_vanishes_outside_its_radius():
    from erotica.analysis.structure import corona_surface_density, king_profile

    r = np.array([1.0, 10.0, 29.9, 30.0, 30.1, 50.0])
    sigma = corona_surface_density(r, delta_f=0.02, R_2=30.0)
    assert np.all(sigma[:3] > 0)
    assert np.all(sigma[3:] == 0)


@requires_bayes_extra
@pytest.mark.slow
def test_bayes_factor_finds_a_corona_that_fits_inside_the_field():
    """The corona model must be *identifiable* when the corona fits in the footprint.

    This is the counterpart to the NGC 6383 result, where ``P(R_2 > field) = 1.000``
    and the comparison against a flat background came back inconclusive
    (``2 ln B = -2.27``). That was read as "the footprint does not contain the
    object" rather than "the corona model does not work" -- and the reading is only
    legitimate if the model *can* be distinguished when the corona is small enough
    to be seen whole. So: inject a corona at ``R_2 = 20'`` in a 70' field and require
    the comparison to prefer it over King + flat background.

    Without this test the NGC 6383 interpretation is unfalsifiable.
    """
    from erotica.analysis.structure import corona_surface_density, king_profile

    # The corona must extend BEYOND the tidal radius -- that is what a corona is, and
    # it is also the only configuration in which the model is identifiable. Nested
    # inside R_t (the first attempt used R_2 = 20 < R_t = 30) the corona is absorbed
    # by a slightly different King and the comparison correctly prefers King.
    king_truth = {"k": 8.0, "b": 0.0, "R_c": 2.0, "R_t": 10.0}
    delta_f, R_2 = 0.006, 35.0

    def surface(r):
        return king_profile(
            r, core_radius=king_truth["R_c"], tidal_radius=king_truth["R_t"],
            amplitude=king_truth["k"], background=0.0,
        ) + corona_surface_density(r, delta_f=delta_f, R_2=R_2)

    data = _sample_profile(77, surface)
    # SMC needs the draws here: the corona model has a genuine k <-> delta_f degeneracy
    # and at 1000 draws the evidence scatter swamped the verdict.
    res = compare_radial_profiles(
        data, field_radius=FIELD, models=("king", "king_corona"),
        draws=4000, chains=4, random_seed=3,
    )
    assert res["best"] == "king_corona", res["log_marginal_likelihood"]
    assert res["resolvable"]["king"], "verdict is within the chain scatter of the evidence"
    assert 2 * res["log_bayes_factor_vs_best"]["king"] < -6.0, "should be 'strong' at minimum"


@requires_bayes_extra
def test_reported_king_medians_are_medians_not_means():
    """A summary that silently returned the mean would pass every recovery test and
    quietly shift every published number.

    The fixture puts the true ``R_t`` **outside** the field, which is the situation
    that actually obtains for NGC 6383 and the one where the distinction bites: with
    no truncation inside the footprint the posterior runs away to the right and mean
    and median separate by **22%**. (With ``R_t`` inside the field the posterior is
    nearly symmetric and they differ by only 2%, so a fixture chosen that way would
    have made this test almost vacuous -- the same "is the fixture in a regime where
    the test can fail?" trap that let three earlier mutations survive.)
    """
    from erotica.analysis.inference import SamplingConfig

    radii = _sample_king(5, k=5.0, b=0.001, R_c=2.0, R_t=200.0)
    fit = king_unbinned(
        radii, field_radius=FIELD,
        sampling=SamplingConfig(draws=1500, tune=1000, chains=2, random_seed=4, progressbar=False),
    )
    draws = np.asarray(fit["king_trace"].posterior["R_t"].values).ravel()
    reported = float(fit["R_t_median"].value)
    assert reported == pytest.approx(float(np.median(draws)), rel=1e-6)
    # and the distinction is not academic on this posterior
    assert abs(np.mean(draws) - np.median(draws)) > 0.10 * np.median(draws)


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "func, kwargs",
    [
        (king_profile, dict(core_radius=1.38, tidal_radius=54.0)),
        (eff_surface_density, dict(k=1.0, b=0.0, a=1.65, gamma=2.32)),
    ],
)
def test_profiles_respect_the_unit_of_the_radius(func, kwargs):
    """Passing the same physical radii in different units must give the same answer.

    This is a regression test for a silent failure, not a hypothetical one.
    ``quantity_values(x)`` with no target unit strips whatever unit ``x`` carries, so
    handing a profile radii in degrees where arcmin was meant returned numbers that were
    wrong by up to **480x** with no error raised -- `king_profile` gave 0.896 instead of
    0.0019 at r = 20'. The same trap exists in the literature: Hunt & Reffert publish
    angular radii in degrees, and reading them as arcmin understates every radius 60-fold.

    The convention is that a plain array is taken as **arcmin**; a Quantity is converted.
    """
    r_arcmin = np.array([1.0, 5.0, 20.0]) * u.arcmin
    same = np.asarray(func(r_arcmin, **kwargs))
    in_degrees = np.asarray(func(r_arcmin.to(u.deg), **kwargs))
    in_arcsec = np.asarray(func(r_arcmin.to(u.arcsec), **kwargs))
    np.testing.assert_allclose(same, in_degrees, rtol=1e-12)
    np.testing.assert_allclose(same, in_arcsec, rtol=1e-12)
    # and a bare array is still interpreted as arcmin, not rejected
    np.testing.assert_allclose(same, np.asarray(func(np.array([1.0, 5.0, 20.0]), **kwargs)))


def test_corona_respects_the_unit_of_the_radius():
    r = np.array([1.0, 5.0, 20.0]) * u.arcmin
    np.testing.assert_allclose(
        np.asarray(corona_surface_density(r, delta_f=0.02, R_2=40.0)),
        np.asarray(corona_surface_density(r.to(u.deg), delta_f=0.02, R_2=40.0)),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Which King fitter the DEFAULT path uses -- Defect 2.
#
# ``king_unbinned`` was already here, already correct, and already tested. What
# was not tested is which fitter ``ClusterStructureAnalyzer.fit_king_profile``
# reaches, and until 2026-08-02 it reached ``RDP_bayesian``: a Normal likelihood
# on equal-count binned densities (measured Poisson dispersion index 0.045
# against the asserted 1.0) with all four prior bounds taken from the data.
#
# The load-bearing oracle below is STRUCTURAL, not numerical: a point process has
# no nuisance scatter parameter, so the default's posterior must contain no
# ``sigma``. That cannot be satisfied by accident by a binned fit.
# ---------------------------------------------------------------------------

from erotica.analysis.structure import ClusterStructureAnalyzer  # noqa: E402


def _king_table(seed, *, field=FIELD, **params):
    """Sky positions whose separations from the centre ARE the injected radii.

    Built with :meth:`~astropy.coordinates.SkyCoord.directional_offset_by`, i.e.
    exact spherical geometry, so ``profile.distances`` reproduces the sampled
    radii to machine precision and no small-angle error leaks into the recovery
    tolerance -- and no star lands outside ``field_radius`` by rounding.
    """
    from astropy.coordinates import SkyCoord
    from astropy.table import QTable as _QTable

    radii = _sample_king(seed, field=field, **params)
    rng = np.random.default_rng(seed + 1)
    centre = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
    offsets = centre.directional_offset_by(
        rng.uniform(0.0, 360.0, radii.size) * u.deg, radii * u.arcmin
    )
    return centre, _QTable(
        {
            "ra": offsets.ra.to(u.deg),
            "dec": offsets.dec.to(u.deg),
            "probability": np.full(radii.size, 0.9),
        }
    )


@requires_bayes_extra
def test_the_default_king_fit_is_a_point_process_with_no_nuisance_scatter():
    """Structural oracle: a point process has no shared ``sigma``. A binned
    Gaussian on densities necessarily does, because that is the parameter that
    absorbs the binning scatter. Asking whether the posterior contains ``sigma``
    therefore distinguishes the two likelihoods with no tolerance to tune.
    """
    from erotica.analysis.inference import SamplingConfig

    centre, table = _king_table(31, **TRUE_KING)
    analyzer = ClusterStructureAnalyzer(table)
    profile = analyzer.radial_density_profile(centre)

    unbinned = analyzer.fit_king_profile(
        profile,
        field_radius=FIELD * u.arcmin,
        return_trace=True,
        sampling=SamplingConfig(
            draws=500, tune=500, chains=2, random_seed=31, progressbar=False
        ),
    )
    variables = set(unbinned["king_trace"].posterior.data_vars)
    assert "sigma" not in variables, f"the default fit carries a nuisance sigma: {variables}"
    assert {"k", "b", "R_c", "R_t"} <= variables
    # ... and the binned path, which is what it used to call, does have one.
    with pytest.warns(UserWarning, match="king_unbinned"):
        binned = analyzer.fit_king_profile(
            profile,
            method="binned",
            return_trace=True,
            sampling=SamplingConfig(
                draws=300, tune=300, chains=2, random_seed=31, progressbar=False
            ),
        )
    assert "sigma" in set(binned["king_trace"].posterior.data_vars)


@requires_bayes_extra
def test_the_default_king_fit_recovers_the_injected_core_radius():
    """Oracle: the ``R_c`` this test injected into the point process.

    ABSOLUTE tolerance tied to the injected truth, never ``N * posterior_std`` --
    a self-widening window passes automatically whenever the fit is poor, which
    is exactly when it should fail.
    """
    import arviz as az

    from erotica.analysis.inference import SamplingConfig

    centre, table = _king_table(32, **TRUE_KING)
    analyzer = ClusterStructureAnalyzer(table)
    profile = analyzer.radial_density_profile(centre)

    res = analyzer.fit_king_profile(
        profile,
        field_radius=FIELD * u.arcmin,
        sampling=SamplingConfig(
            draws=1000, tune=1000, chains=2, random_seed=32, progressbar=False
        ),
    )
    r_c = float(res["R_c_median"].to_value(u.arcmin))
    assert abs(r_c - TRUE_KING["R_c"]) < 0.6, (
        f"R_c = {r_c:.3f}' vs injected {TRUE_KING['R_c']}'"
    )
    # az.rhat / az.ess rather than az.summary: arviz >= 1 formats the summary
    # frame for display and rounds R-hat to two decimals, so a genuine 1.0051
    # reads back as exactly 1.01 and the PART A floor cannot be applied to it.
    assert float(az.rhat(res["king_trace"], var_names=["R_c"])["R_c"]) < 1.01
    assert float(az.ess(res["king_trace"], var_names=["R_c"])["R_c"]) > 400
    assert int(res["king_trace"].sample_stats["diverging"].values.sum()) == 0


@requires_bayes_extra
def test_the_binned_fitter_warns_and_names_what_replaces_it():
    """A silent legacy path is how a superseded number keeps getting published."""
    from erotica.analysis.inference import SamplingConfig
    from erotica.analysis.structure import RDP_bayesian, RDP_bayesian_log_space

    radius = np.array([1.0, 2.0, 4.0, 8.0, 16.0]) * u.arcmin
    density = np.array([2.0, 1.2, 0.5, 0.2, 0.08])
    cfg = SamplingConfig(draws=200, tune=200, chains=1, random_seed=1, progressbar=False)

    for fitter in (RDP_bayesian, RDP_bayesian_log_space):
        with pytest.warns(UserWarning) as record:
            fitter(density, radius, sampling=cfg)
        message = str(record[0].message)
        assert "king_unbinned" in message, f"{fitter.__name__} does not name its replacement"
        assert "binned" in message


def test_the_unbinned_default_will_not_guess_the_footprint():
    """The normalisation assumes completeness inside a disc, so the caller states it.

    Inferring ``field_radius`` from ``max(radii)`` would let the footprint
    assumption -- the one that is false for a membership-selected list -- be made
    silently by the code.
    """
    centre, table = _king_table(33, **TRUE_KING)
    analyzer = ClusterStructureAnalyzer(table)
    profile = analyzer.radial_density_profile(centre)

    with pytest.raises(ValueError, match="field_radius is required"):
        analyzer.fit_king_profile(profile)
    with pytest.raises(ValueError, match="log_space applies only"):
        analyzer.fit_king_profile(profile, log_space=True, field_radius=FIELD)
    with pytest.raises(ValueError, match="'unbinned' or 'binned'"):
        analyzer.fit_king_profile(profile, method="rdp")


@requires_bayes_extra
def test_the_legacy_figure_facade_still_uses_the_published_fitter():
    """``graph_king`` reproduces figures published before 2026-08-02.

    Re-plotting them from a different likelihood would move a published curve
    without saying so, so this facade is pinned while the analyzer default moved.
    Behavioural check: only the binned fit produces ``king_std``, the posterior of
    the nuisance scatter that the point process does not have.
    """
    from erotica.analysis.figures import graph_king

    centre, table = _king_table(34, **TRUE_KING)
    with pytest.warns(UserWarning, match="king_unbinned"):
        payload = graph_king(table, [centre], prob_number=(0.5,))
    assert "king_std" in payload["bayesian_results"], (
        "graph_king no longer runs the binned fit the published figures used"
    )
