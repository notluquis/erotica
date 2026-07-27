"""Regression tests for :mod:`erotica.analysis.inference`.

Focus: the fractional parallax-error selection cut in
``ClusterInferenceAnalyzer.distance_and_parallax_by_probability`` must reject
unphysical parallaxes. Historically the cut computed ``parallax_error / parallax
<= threshold``. Because ``parallax_error`` is strictly positive, that ratio is
negative for negative parallaxes and therefore passed the ``<= threshold`` test,
silently admitting negative-parallax sources; ``parallax == 0`` also caused a
division by zero. The fix requires ``parallax > 0`` and guards the division.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.table import QTable

import erotica.analysis.inference as inference
from erotica.analysis.inference import (
    ParallaxPriors,
    SamplingConfig,
    fit_parallax_model,
    ClusterInferenceAnalyzer,
    DistanceFitResult,
    ParallaxFitResult,
)


def _make_table() -> QTable:
    """Build a source table mixing positive, zero and negative parallaxes.

    The negative row is deliberately given a *small* magnitude fractional error
    (|error/parallax| = 0.5/10 = 0.05 < 0.1). That is the discriminating case:
    - old raw predicate ``error/parallax <= 0.1`` -> ``-0.05 <= 0.1`` -> INCLUDED (bug)
    - an ``abs()``-only fix -> ``0.05 <= 0.1``     -> STILL INCLUDED (insufficient)
    - the correct ``parallax > 0`` fix             -> EXCLUDED
    """
    return QTable(
        {
            # id 1: good positive source (0.1/2.0 = 0.05 <= 0.1) -> keep
            # id 2: positive but imprecise (0.5/2.0 = 0.25 > 0.1) -> drop
            # id 3: zero parallax (division by zero)              -> drop
            # id 4: negative parallax, small |ratio| = 0.05       -> drop (the bug)
            "source_id": [1, 2, 3, 4],
            "probability": [0.9, 0.9, 0.9, 0.9],
            "parallax": [2.0, 2.0, 0.0, -10.0] * u.mas,
            "parallax_error": [0.1, 0.5, 0.1, 0.5] * u.mas,
            "r_med_geo": [500.0, 500.0, 500.0, 500.0] * u.pc,
        }
    )


def _capture_useful(monkeypatch) -> dict[str, QTable]:
    """Patch the heavy PyMC model calls and capture the ``useful`` subset.

    ``distance_model`` and ``fit_parallax_model`` both receive the post-cut
    ``useful`` table as their first positional argument. We record it and return
    lightweight dataclass results so the method can finish without sampling.
    """
    captured: dict[str, QTable] = {}

    def fake_distance_model(useful, *args, **kwargs):
        captured["useful"] = useful
        return DistanceFitResult(
            mu_r_mean=0.0, std_r_mean=0.0, mu_r_std=0.0, std_r_std=0.0
        )

    def fake_fit_parallax_model(useful, *args, **kwargs):
        captured["useful_parallax"] = useful
        return ParallaxFitResult(
            mu_parallax_mean=0.0,
            sigma_parallax_mean=0.0,
            mu_parallax_std=0.0,
            sigma_parallax_std=0.0,
        )

    monkeypatch.setattr(inference, "distance_model", fake_distance_model)
    monkeypatch.setattr(inference, "fit_parallax_model", fake_fit_parallax_model)
    return captured


def test_fractional_parallax_cut_excludes_nonpositive_parallax(monkeypatch):
    """Negative and zero parallaxes must never enter the distance-posterior sample."""
    captured = _capture_useful(monkeypatch)
    analyzer = ClusterInferenceAnalyzer(_make_table(), probability_column="probability")

    analyzer.distance_and_parallax_by_probability(
        probability_thresholds=(0.5,),
        fractional_parallax_error_max=0.1,
    )

    useful = captured["useful"]
    kept_ids = set(np.asarray(useful["source_id"]).tolist())

    # Only the precise, positive-parallax source survives.
    assert kept_ids == {1}

    kept_parallax = np.asarray(useful["parallax"].to_value(u.mas))
    # The load-bearing property: nothing non-positive slips through. On the old
    # code, source_id 4 (parallax=-10, |ratio|=0.05) would be present here and
    # this assertion would fail.
    assert np.all(kept_parallax > 0)
    assert 4 not in kept_ids  # negative parallax excluded (the actual bug)
    assert 3 not in kept_ids  # zero parallax excluded (no div-by-zero passthrough)

    # Both model calls must receive the identical filtered subset.
    assert set(np.asarray(captured["useful_parallax"]["source_id"]).tolist()) == {1}


# ---------------------------------------------------------------------------
# Real sampling tests.
#
# Until 2026-07-27 NO test in this suite ever sampled a PyMC model: the tests
# above monkeypatch the sampler away, tests/test_structure.py feeds a fake trace
# and never calls RDP_bayesian, and CI installed only `.[dev]`. The Bayesian
# paths that produce every published number were therefore unexercised.
#
# The oracle here is INJECTED TRUTH, not a golden number: synthetic data is drawn
# from a known mu/sigma and the posterior must recover it. Convergence is held to
# the floor in ~/phd/methodology.md PART A (Vehtari+2021): R-hat < 1.01, bulk-ESS
# > 400, zero divergences.
# ---------------------------------------------------------------------------

import importlib.util

_missing_bayes = [m for m in ("pymc", "arviz") if importlib.util.find_spec(m) is None]
requires_bayes_extra = pytest.mark.skipif(
    bool(_missing_bayes), reason=f"needs the 'bayes' extra; missing: {', '.join(_missing_bayes)}"
)

TRUE_PARALLAX = 0.90  # mas  -> ~1.11 kpc, the NGC 6383 regime
TRUE_SPREAD = 0.05    # mas


def _synthetic_parallaxes(n=400, seed=20260727):
    rng = np.random.default_rng(seed)
    return QTable({"parallax": rng.normal(TRUE_PARALLAX, TRUE_SPREAD, n) * u.mas})


@requires_bayes_extra
def test_fit_parallax_model_recovers_injected_truth():
    """Posterior on mu must cover the injected value, and sigma must be sane."""
    from erotica.analysis.inference import SamplingConfig, fit_parallax_model

    data = _synthetic_parallaxes()
    cfg = SamplingConfig(draws=1000, tune=1000, chains=2, random_seed=11, progressbar=False)
    res = fit_parallax_model(data, return_trace=True, sampling=cfg)

    # Recovery: the injected mean must lie within ~4 posterior sd of the estimate.
    # (sd here is the *posterior* width, ~sigma/sqrt(n) ~ 0.0025 mas.)
    assert abs(res.mu_parallax_mean - TRUE_PARALLAX) < 4 * res.mu_parallax_std, (
        f"mu not recovered: got {res.mu_parallax_mean:.4f}, truth {TRUE_PARALLAX}"
    )
    # The dispersion parameter must find the injected spread, not collapse or blow up.
    assert 0.5 * TRUE_SPREAD < res.sigma_parallax_mean < 2.0 * TRUE_SPREAD


@requires_bayes_extra
def test_fit_parallax_model_meets_the_convergence_floor():
    """R-hat < 1.01, bulk-ESS > 400, zero divergences (methodology.md PART A)."""
    import arviz as az

    from erotica.analysis.inference import SamplingConfig, fit_parallax_model

    cfg = SamplingConfig(draws=2000, tune=1000, chains=4, random_seed=12, progressbar=False)
    res = fit_parallax_model(_synthetic_parallaxes(), return_trace=True, sampling=cfg)
    trace = res.trace
    assert trace is not None, "return_trace=True did not return a trace"

    summary = az.summary(trace, var_names=["mu_parallax", "sigma_parallax"])
    # arviz >=1 returns a display-formatted frame: values arrive as strings.
    r_hat = pd.to_numeric(summary["r_hat"], errors="coerce")
    ess_bulk = pd.to_numeric(summary["ess_bulk"], errors="coerce")
    assert r_hat.max() < 1.01, f"R-hat too high:\n{summary}"
    assert ess_bulk.min() > 400, f"bulk-ESS too low:\n{summary}"

    n_div = int(trace.sample_stats["diverging"].sum())
    assert n_div == 0, f"{n_div} divergent transitions"


@requires_bayes_extra
def test_return_trace_false_discards_the_posterior():
    """Characterisation: the default API contract throws the posterior away.

    Recorded deliberately -- see docs/design-notes/decisions.md. Downstream code
    therefore receives point estimates and cannot propagate uncertainty.
    """
    from erotica.analysis.inference import SamplingConfig, fit_parallax_model

    cfg = SamplingConfig(draws=200, tune=200, chains=2, random_seed=13, progressbar=False)
    res = fit_parallax_model(_synthetic_parallaxes(n=100), sampling=cfg)
    assert res.trace is None


# ---------------------------------------------------------------------------
# Per-star parallax errors in the likelihood
#
# ORACLE: an intrinsic spread the test injects, which is *deliberately much
# smaller* than the measurement errors -- the regime every Gaia open cluster is
# in. The naive model has no way to tell the two apart, so it must report the
# total scatter as if it were cluster depth. The error-aware model must recover
# the injected intrinsic value.
# ---------------------------------------------------------------------------

TRUE_INTRINSIC = 0.010  # mas; ~12 pc of depth at 1.1 kpc
# Median e_Plx per Gmag quartile of the published NGC 6383 member table, so the
# error distribution is the real one rather than a convenient one.
REAL_ERROR_SCALE = np.array([0.027, 0.065, 0.123, 0.293])


def _parallax_table(n=400, seed=5):
    rng = np.random.default_rng(seed)
    errors = np.repeat(REAL_ERROR_SCALE, n // 4)
    errors = errors * rng.uniform(0.8, 1.2, errors.size)  # spread within quartiles
    true_plx = rng.normal(TRUE_PARALLAX, TRUE_INTRINSIC, errors.size)
    observed = true_plx + rng.normal(0.0, errors)
    return QTable({"parallax": observed * u.mas, "parallax_error": errors * u.mas}), errors


@requires_bayes_extra
def test_per_star_errors_separate_cluster_depth_from_measurement_noise():
    """The naive model reports measurement scatter as cluster depth; the fix does not."""
    table, errors = _parallax_table()
    cfg = SamplingConfig(draws=1500, tune=1000, chains=2, random_seed=3, progressbar=False)

    naive = fit_parallax_model(table, sampling=cfg)
    aware = fit_parallax_model(table, parallax_error_column="parallax_error", sampling=cfg)

    assert naive.metadata["error_aware"] is False
    assert aware.metadata["error_aware"] is True

    # The error-aware fit recovers the injected intrinsic spread.
    assert abs(aware.sigma_parallax_mean - TRUE_INTRINSIC) < 4 * aware.sigma_parallax_std

    # The naive fit instead reports the total observed scatter, which is set by
    # the measurement errors and is an order of magnitude larger.
    assert naive.sigma_parallax_mean > 5 * TRUE_INTRINSIC
    assert naive.sigma_parallax_mean > 3 * aware.sigma_parallax_mean

    # Both should still find the mean parallax -- the bias is in the width only.
    for res in (naive, aware):
        assert abs(res.mu_parallax_mean - TRUE_PARALLAX) < 4 * res.mu_parallax_std


@requires_bayes_extra
def test_default_priors_do_not_depend_on_the_data():
    """Two clusters at different parallaxes must get identical prior support."""
    near = QTable({"parallax": np.random.default_rng(1).normal(5.0, 0.1, 200) * u.mas})
    far = QTable({"parallax": np.random.default_rng(2).normal(0.4, 0.1, 200) * u.mas})
    cfg = SamplingConfig(draws=200, tune=200, chains=1, random_seed=0, progressbar=False)

    a = fit_parallax_model(near, return_trace=True, sampling=cfg)
    b = fit_parallax_model(far, return_trace=True, sampling=cfg)
    assert a.metadata["prior"] == b.metadata["prior"] == "scale-free"

    # Both posteriors must lie inside the same fixed support, and each must still
    # find its own very different truth -- a fixed prior that broke recovery would
    # be no improvement.
    p = ParallaxPriors()
    for res, truth in ((a, 5.0), (b, 0.4)):
        draws = np.asarray(res.trace.posterior["mu_parallax"].values)
        assert draws.min() >= p.mu_lower and draws.max() <= p.mu_upper
        assert abs(res.mu_parallax_mean - truth) < 0.05


@requires_bayes_extra
def test_an_independent_distance_still_gives_an_informative_prior():
    """prior_distance comes from outside the sample, so it is not data reuse."""
    table, _ = _parallax_table()
    cfg = SamplingConfig(draws=300, tune=300, chains=1, random_seed=0, progressbar=False)
    res = fit_parallax_model(table, prior_distance=1.11 * u.kpc, sampling=cfg)
    assert res.metadata["prior"] == "informative"
    assert abs(res.mu_parallax_mean - TRUE_PARALLAX) < 0.05


def test_bad_parallax_errors_are_rejected():
    table = QTable({
        "parallax": np.linspace(0.8, 1.0, 10) * u.mas,
        "parallax_error": np.concatenate([[0.0], np.full(9, 0.05)]) * u.mas,
    })
    with pytest.raises(ValueError, match="finite and positive"):
        fit_parallax_model(table, parallax_error_column="parallax_error")

    short = QTable({"parallax": np.linspace(0.8, 1.0, 10) * u.mas})
    short["bad"] = np.full(10, np.nan) * u.mas
    with pytest.raises(ValueError, match="finite and positive"):
        fit_parallax_model(short, parallax_error_column="bad")


# ---------------------------------------------------------------------------
# Per-star proper-motion covariance
#
# ORACLE: an intrinsic velocity dispersion injected far below the measurement
# errors -- the regime every Gaia open cluster is in. The naive model cannot tell
# them apart and must report the total; the covariance-aware model must recover
# the injected intrinsic value.
# ---------------------------------------------------------------------------

TRUE_PM = {"mu_ra": -1.35, "mu_dec": -1.90, "sigma_int": 0.05}  # mas/yr
PM_ERROR_SCALE = np.array([0.03, 0.08, 0.18, 0.40])  # per Gmag quartile, Gaia-like


def _pm_table(n=400, seed=9, corr=0.4):
    rng = np.random.default_rng(seed)
    e_ra = np.repeat(PM_ERROR_SCALE, n // 4) * rng.uniform(0.8, 1.2, n)
    e_dec = np.repeat(PM_ERROR_SCALE, n // 4) * rng.uniform(0.8, 1.2, n)
    rho = np.full(n, corr)
    truth = rng.normal(
        [TRUE_PM["mu_ra"], TRUE_PM["mu_dec"]], TRUE_PM["sigma_int"], size=(n, 2)
    )
    obs = np.empty_like(truth)
    for i in range(n):
        c = np.array([[e_ra[i] ** 2, rho[i] * e_ra[i] * e_dec[i]],
                      [rho[i] * e_ra[i] * e_dec[i], e_dec[i] ** 2]])
        obs[i] = rng.multivariate_normal(truth[i], c)
    return obs[:, 0], obs[:, 1], e_ra, e_dec, rho


@requires_bayes_extra
def test_per_star_covariance_separates_dispersion_from_measurement_noise():
    from erotica.analysis.inference import proper_motion_2d_gaussian

    ra, dec, e_ra, e_dec, rho = _pm_table()
    # cores=1: the batched (n, 2, 2) covariance kills PyMC's multiprocess workers
    # with EOFError on this machine. Sequential sampling is correct and slower,
    # which is the right trade for a test.
    cfg = SamplingConfig(draws=800, tune=800, chains=2, random_seed=4, progressbar=False,
                         extra_kwargs={"cores": 1})

    naive = proper_motion_2d_gaussian(ra, dec, sampling=cfg)
    aware = proper_motion_2d_gaussian(
        ra, dec, pm_ra_error=e_ra, pm_dec_error=e_dec, pm_ra_dec_corr=rho, sampling=cfg
    )

    assert aware.metadata["error_aware"] is True
    assert naive.metadata["error_aware"] is False

    # the covariance-aware fit recovers the injected intrinsic dispersion
    for axis in ("RA", "Dec"):
        got = aware.results[f"sigma_{axis}_mean"]
        assert abs(got - TRUE_PM["sigma_int"]) < 4 * aware.results[f"sigma_{axis}_std"]
        # the naive one reports the measurement scatter instead, several times larger
        assert naive.results[f"sigma_{axis}_mean"] > 3 * got

    # both should still find the centroid; the bias is in the width
    for res in (naive, aware):
        assert abs(res.results["mu_RA_mean"] - TRUE_PM["mu_ra"]) < 0.05
        assert abs(res.results["mu_Dec_mean"] - TRUE_PM["mu_dec"]) < 0.05

    # The injected INTRINSIC correlation is zero; the 0.4 lives only in the
    # measurement errors. The naive fit reports it as a kinematic correlation of
    # the cluster, which it is not; the covariance-aware fit does not.
    assert abs(aware.results["corr_mean"]) < 0.15
    assert naive.results["corr_mean"] > 0.2


@requires_bayes_extra
def test_pm_priors_do_not_depend_on_the_data():
    """Two clusters at very different proper motions get identical prior support."""
    from erotica.analysis.inference import ProperMotionPriors, proper_motion_2d_gaussian

    rng = np.random.default_rng(3)
    cfg = SamplingConfig(draws=300, tune=300, chains=1, random_seed=0, progressbar=False)
    a = proper_motion_2d_gaussian(
        rng.normal(-1.4, 0.2, 200), rng.normal(-1.9, 0.2, 200), return_trace=True, sampling=cfg
    )
    b = proper_motion_2d_gaussian(
        rng.normal(12.0, 0.2, 200), rng.normal(-8.0, 0.2, 200), return_trace=True, sampling=cfg
    )
    assert a.metadata["prior"] == b.metadata["prior"] == "scale-free"
    # each still finds its own very different centroid
    assert abs(a.results["mu_RA_mean"] - (-1.4)) < 0.1
    assert abs(b.results["mu_RA_mean"] - 12.0) < 0.1
    p = ProperMotionPriors()
    assert p.mu_scale > 0 and p.sigma_scale > 0


def test_pm_error_validation():
    from erotica.analysis.inference import proper_motion_2d_gaussian

    ra, dec = np.linspace(-2, -1, 20), np.linspace(-2, -1, 20)
    with pytest.raises(ValueError, match="both"):
        proper_motion_2d_gaussian(ra, dec, pm_ra_error=np.full(20, 0.1))
    with pytest.raises(ValueError, match="positive"):
        proper_motion_2d_gaussian(
            ra, dec, pm_ra_error=np.zeros(20), pm_dec_error=np.full(20, 0.1)
        )
    with pytest.raises(ValueError, match="strictly inside"):
        proper_motion_2d_gaussian(
            ra, dec, pm_ra_error=np.full(20, 0.1), pm_dec_error=np.full(20, 0.1),
            pm_ra_dec_corr=np.full(20, 1.0),
        )


# ---------------------------------------------------------------------------
# Bailer-Jones distance uncertainties in the hierarchical distance model
#
# ORACLE: an intrinsic line-of-sight depth injected far below the catalogue
# uncertainties -- the regime every Gaia cluster is in, since a 10% distance
# error at 1.1 kpc is 110 pc while no bound open cluster is that deep.
# ---------------------------------------------------------------------------

TRUE_DIST = {"mu": 1.11, "depth": 0.020}  # kpc; 20 pc of real depth


def _distance_table(n=200, seed=13):
    rng = np.random.default_rng(seed)
    frac = rng.uniform(0.04, 0.14, n)              # Bailer-Jones fractional errors
    sigma = frac * TRUE_DIST["mu"]
    true_r = rng.normal(TRUE_DIST["mu"], TRUE_DIST["depth"], n)
    obs = true_r + rng.normal(0.0, sigma)
    return QTable({
        "r_med_geo": obs * u.kpc,
        "r_lo_geo": (obs - sigma) * u.kpc,          # 16th percentile
        "r_hi_geo": (obs + sigma) * u.kpc,          # 84th percentile
        "parallax": (1.0 / obs) * u.mas,
    }), sigma


@requires_bayes_extra
def test_bailer_jones_bounds_separate_depth_from_catalogue_error():
    from erotica.analysis.inference import distance_model

    table, sigma = _distance_table()
    cfg = SamplingConfig(draws=800, tune=800, chains=2, random_seed=8, progressbar=False,
                         extra_kwargs={"cores": 1})

    naive = distance_model(table, sampling=cfg)
    aware = distance_model(
        table, distance_lo_column="r_lo_geo", distance_hi_column="r_hi_geo", sampling=cfg
    )

    assert naive.metadata["error_aware"] is False
    assert aware.metadata["error_aware"] is True
    assert naive.metadata["prior"] == aware.metadata["prior"] == "scale-free"

    # both find the cluster distance
    for res in (naive, aware):
        assert abs(res.mu_r_mean - TRUE_DIST["mu"]) < 5 * res.mu_r_std

    # the error-aware fit recovers the injected depth; the naive one reports the
    # catalogue scatter instead, which is several times larger
    assert abs(aware.std_r_mean - TRUE_DIST["depth"]) < 4 * aware.std_r_std
    assert naive.std_r_mean > 3 * aware.std_r_mean
    assert naive.std_r_mean > 0.5 * float(np.median(sigma))


@requires_bayes_extra
def test_distance_priors_do_not_depend_on_the_data():
    """A near and a far cluster get the same prior support and both are found."""
    from erotica.analysis.inference import DistancePriors, distance_model

    rng = np.random.default_rng(2)
    cfg = SamplingConfig(draws=400, tune=400, chains=1, random_seed=0, progressbar=False)
    near = QTable({"r_med_geo": rng.normal(0.4, 0.02, 150) * u.kpc,
                   "parallax": np.full(150, 2.5) * u.mas})
    far = QTable({"r_med_geo": rng.normal(4.0, 0.20, 150) * u.kpc,
                  "parallax": np.full(150, 0.25) * u.mas})
    a = distance_model(near, sampling=cfg)
    b = distance_model(far, sampling=cfg)
    p = DistancePriors()
    assert p.mu_lower < 0.4 and p.mu_upper > 4.0     # one fixed support spans both
    assert abs(a.mu_r_mean - 0.4) < 0.05
    assert abs(b.mu_r_mean - 4.0) < 0.30


def test_distance_bound_validation():
    from erotica.analysis.inference import distance_model

    t = QTable({"r_med_geo": np.linspace(1.0, 1.2, 20) * u.kpc,
                "parallax": np.full(20, 0.9) * u.mas})
    with pytest.raises(ValueError, match="both"):
        distance_model(t, distance_lo_column="r_med_geo")
    t["lo"] = np.linspace(1.0, 1.2, 20) * u.kpc
    t["hi"] = np.linspace(0.9, 1.1, 20) * u.kpc      # hi < lo everywhere
    with pytest.raises(ValueError, match="must exceed"):
        distance_model(t, distance_lo_column="lo", distance_hi_column="hi")
