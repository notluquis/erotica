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

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from astropy import units as u
from astropy.table import QTable

import erotica.analysis.inference as inference
from erotica.analysis.inference import (
    ClusterInferenceAnalyzer,
    DistanceFitResult,
    ParallaxFitResult,
    ParallaxPriors,
    SamplingConfig,
    fit_parallax_model,
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
            "r_lo_geo": [450.0, 450.0, 450.0, 450.0] * u.pc,
            "r_hi_geo": [550.0, 550.0, 550.0, 550.0] * u.pc,
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
        # `metadata` con la familia dentro, como la trae SIEMPRE el resultado real: `distance_model`
        # la pone incondicionalmente. El stub la omitia, o sea fingia una forma de resultado que el
        # codigo no puede producir, y el llamador reventaba con KeyError en cuanto empezo a leerla.
        # Un doble que no puede representar la salida real convierte cualquier consumo nuevo de esa
        # salida en un fallo del test en vez de un fallo del codigo.
        return DistanceFitResult(
            mu_r_mean=0.0,
            std_r_mean=0.0,
            mu_r_std=0.0,
            std_r_std=0.0,
            metadata={
                "population": "normal-marginalised" if kwargs.get("distance_lo_column") else "gamma"
            },
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
TRUE_SPREAD = 0.05  # mas


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
# Median e_Plx per Gmag quartile of the published NGC 6383 member table, SCALED
# DOWN by 5x. At the real scale the per-star errors dwarf a realistic intrinsic
# spread, sigma_int is barely identified (posterior std ~ half the value), and no
# assertion on it can fail for the reason it was written -- a mutation audit beat
# the earlier fixture by dropping the square from sqrt(sigma^2 + e_i^2) and the
# recovered value moved by less than a tenth of its own uncertainty. The SHAPE of
# the real error distribution is what matters here, not its absolute scale.
REAL_ERROR_SCALE = np.array([0.027, 0.065, 0.123, 0.293]) / 5.0


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

    # ABSOLUTE tolerance, not `< N * posterior_std`. A `< N*std` window is
    # self-widening: degrading the fit inflates the very tolerance meant to catch
    # the degradation, so it cannot fail for the reason it was written. A mutation
    # audit beat the old form by dropping the square from sqrt(sigma^2 + e_i^2).
    assert abs(aware.sigma_parallax_mean - TRUE_INTRINSIC) < 0.006, (
        f"intrinsic spread {aware.sigma_parallax_mean:.5f} vs injected {TRUE_INTRINSIC}"
    )
    # and the posterior must actually be informative, not merely centred
    assert aware.sigma_parallax_std < 0.003

    # The naive fit instead reports the total observed scatter, which is set by
    # the measurement errors and is an order of magnitude larger.
    assert naive.sigma_parallax_mean > 2.5 * TRUE_INTRINSIC
    assert naive.sigma_parallax_mean > 2.5 * aware.sigma_parallax_mean

    # Both should still find the mean parallax -- the bias is in the width only.
    for res in (naive, aware):
        assert abs(res.mu_parallax_mean - TRUE_PARALLAX) < 0.03
        assert res.mu_parallax_std < 0.005


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
    table = QTable(
        {
            "parallax": np.linspace(0.8, 1.0, 10) * u.mas,
            "parallax_error": np.concatenate([[0.0], np.full(9, 0.05)]) * u.mas,
        }
    )
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
    """Synthetic PM sample.

    KNOWN GAP, recorded rather than papered over. ``e_ra`` and ``e_dec`` are drawn
    from the same scale here, which means ``rho*e_ra*e_dec`` and ``rho*e_ra*e_ra``
    are numerically indistinguishable, so **the off-diagonal of the per-star
    covariance is not tested** -- a mutation audit beat this fixture exactly that
    way. Setting ``e_dec = 3*e_ra`` makes the off-diagonal testable, but in that
    regime the fit stops recovering the intrinsic correlation (returns +0.37 for an
    injected 0.0, with sigma_Dec 30% low), and it is not yet established whether
    that is an identifiability limit or a modelling error. Shipping a test tuned to
    a regime that is not understood would repeat the failure this audit exposed.
    See ~/phd/open-threads.md C5.
    """
    rng = np.random.default_rng(seed)
    e_ra = np.repeat(PM_ERROR_SCALE, n // 4) * rng.uniform(0.8, 1.2, n)
    e_dec = np.repeat(PM_ERROR_SCALE, n // 4) * rng.uniform(0.8, 1.2, n)
    rho = np.full(n, corr)
    truth = rng.normal([TRUE_PM["mu_ra"], TRUE_PM["mu_dec"]], TRUE_PM["sigma_int"], size=(n, 2))
    obs = np.empty_like(truth)
    for i in range(n):
        c = np.array(
            [
                [e_ra[i] ** 2, rho[i] * e_ra[i] * e_dec[i]],
                [rho[i] * e_ra[i] * e_dec[i], e_dec[i] ** 2],
            ]
        )
        obs[i] = rng.multivariate_normal(truth[i], c)
    return obs[:, 0], obs[:, 1], e_ra, e_dec, rho


def _assert_no_divergences(res, label: str, *, latentes: frozenset[str] = frozenset()) -> None:
    """Gate on the divergence COUNT, which is what the RuntimeWarning filter gave up.

    ``pytest.ini`` escalates RuntimeWarning to an error, and NUTS signals a divergent
    trajectory by overflowing its kinetic-energy dot product to +inf. Escalating that
    ABORTS sampling at the first divergence, so the run dies before anyone can see whether
    it was one benign transition or a broken geometry. The warning is now scoped out for
    ``pymc.step_methods.hmc`` and the gate lives here instead: a real problem still fails,
    and fails with a number rather than a warning.
    """
    trace = getattr(res, "trace", None)
    assert trace is not None, f"{label}: return_trace=True did not return a trace"
    n_div = int(trace.sample_stats["diverging"].sum())

    # Convergence, not just the divergence counter. Measured 2026-08-24: the error-aware distance
    # fit passed this assertion on seed 8 with **R-hat 1.377 and ESS 5** on `std_r` -- the very
    # parameter the caller then asserts on, with a tolerance of +-60% of the true value. It was
    # passing on the tolerance, not on the fit, and the 8-in-100 CI failures were the same broken
    # geometry occasionally producing the symptom being checked. A divergence count of 0, 1, 32 or
    # 344 is arbitrary when the chain has not converged.
    #
    # Se exime por NOMBRE, y por defecto no se exime nada. La primera version filtraba por FORMA
    # --quedarse con lo que no tuviera dimensiones mas alla de (chain, draw)-- para saltarse un
    # latente por estrella como `r_true`, y eso tenia tres problemas: `r_true` ya no existe en
    # ninguno de los dos modelos que llegan aca (la marginalizacion lo elimino, y el modelo de
    # movimientos propios nunca lo tuvo), asi que el filtro no eximia nada; un parametro de
    # POBLACION declarado vectorial —`pm.Normal("mu", shape=2)`— quedaba exento por accidente y
    # podia pasar con R-hat 1,5; y si el filtro no seleccionaba nada, `max(())` reventaba con un
    # `ValueError: max() iterable argument is empty` en vez de decir que pasaba.
    #
    # Un latente futuro se exime nombrandolo, que obliga a justificarlo en el sitio de la llamada.
    import arviz as az

    post = trace.posterior
    poblacion = [v for v in post.data_vars if v not in latentes]
    assert poblacion, (
        f"{label}: no queda ningun parametro que revisar tras eximir {sorted(latentes)} "
        f"de {sorted(post.data_vars)}"
    )
    rhat, ess = az.rhat(post), az.ess(post)
    # `.max()` / `.min()` y no `float(...values)`: un parametro vectorial trae un array por
    # variable, y `float()` sobre el habria reventado en vez de tomar el peor de sus componentes.
    peor_rhat = max(((v, float(rhat[v].max())) for v in poblacion), key=lambda x: x[1])
    peor_ess = min(((v, float(ess[v].min())) for v in poblacion), key=lambda x: x[1])
    assert peor_rhat[1] < 1.01, (
        f"{label}: R-hat {peor_rhat[1]:.3f} on `{peor_rhat[0]}` -- la cadena no convergio, "
        f"asi que el conteo de divergencias ({n_div}) no significa nada"
    )
    assert peor_ess[1] > 400, (
        f"{label}: ESS {peor_ess[1]:.0f} on `{peor_ess[0]}` -- muestras efectivas insuficientes "
        "para creerle a una media posterior"
    )
    assert n_div == 0, f"{label}: {n_div} divergent transitions"


@requires_bayes_extra
def test_per_star_covariance_separates_dispersion_from_measurement_noise():
    from erotica.analysis.inference import proper_motion_2d_gaussian

    ra, dec, e_ra, e_dec, rho = _pm_table()
    # cores=1: the batched (n, 2, 2) covariance kills PyMC's multiprocess workers
    # with EOFError on this machine. Sequential sampling is correct and slower,
    # which is the right trade for a test.
    cfg = SamplingConfig(
        draws=800, tune=800, chains=2, random_seed=4, progressbar=False, extra_kwargs={"cores": 1}
    )

    naive = proper_motion_2d_gaussian(ra, dec, sampling=cfg, return_trace=True)
    aware = proper_motion_2d_gaussian(
        ra,
        dec,
        pm_ra_error=e_ra,
        pm_dec_error=e_dec,
        pm_ra_dec_corr=rho,
        sampling=cfg,
        return_trace=True,
    )

    assert aware.metadata["error_aware"] is True
    assert naive.metadata["error_aware"] is False
    _assert_no_divergences(naive, "proper_motion naive")
    _assert_no_divergences(aware, "proper_motion covariance-aware")

    # the covariance-aware fit recovers the injected intrinsic dispersion.
    # ABSOLUTE tolerance -- see the parallax test. The old `< 4 * std` form let a
    # wrong off-diagonal (rho*e_ra*e_ra) through.
    for axis in ("RA", "Dec"):
        got = aware.results[f"sigma_{axis}_mean"]
        assert abs(got - TRUE_PM["sigma_int"]) < 0.030, f"sigma_{axis} = {got:.4f}"
        assert aware.results[f"sigma_{axis}_std"] < 0.040
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
        proper_motion_2d_gaussian(ra, dec, pm_ra_error=np.zeros(20), pm_dec_error=np.full(20, 0.1))
    with pytest.raises(ValueError, match="strictly inside"):
        proper_motion_2d_gaussian(
            ra,
            dec,
            pm_ra_error=np.full(20, 0.1),
            pm_dec_error=np.full(20, 0.1),
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
    frac = rng.uniform(0.04, 0.14, n)  # Bailer-Jones fractional errors
    sigma = frac * TRUE_DIST["mu"]
    true_r = rng.normal(TRUE_DIST["mu"], TRUE_DIST["depth"], n)
    obs = true_r + rng.normal(0.0, sigma)
    return QTable(
        {
            "r_med_geo": obs * u.kpc,
            "r_lo_geo": (obs - sigma) * u.kpc,  # 16th percentile
            "r_hi_geo": (obs + sigma) * u.kpc,  # 84th percentile
            "parallax": (1.0 / obs) * u.mas,
        }
    ), sigma


@requires_bayes_extra
def test_bailer_jones_bounds_separate_depth_from_catalogue_error():
    from erotica.analysis.inference import distance_model

    table, sigma = _distance_table()
    # 2000 extracciones y no 800: con los latentes marginalizados el ajuste es mas barato —200
    # parametros menos— y el ESS de `std_r` pasa de 220-432 a 849-992, que es lo que hace falta
    # para creerle a una media posterior segun el umbral de 400 que este helper exige.
    cfg = SamplingConfig(
        draws=2000, tune=1000, chains=2, random_seed=8, progressbar=False, extra_kwargs={"cores": 1}
    )

    naive = distance_model(table, sampling=cfg, return_trace=True)
    aware = distance_model(
        table,
        distance_lo_column="r_lo_geo",
        distance_hi_column="r_hi_geo",
        sampling=cfg,
        return_trace=True,
    )

    assert naive.metadata["error_aware"] is False
    assert aware.metadata["error_aware"] is True
    assert naive.metadata["prior"] == aware.metadata["prior"] == "scale-free"
    _assert_no_divergences(naive, "distance naive")
    _assert_no_divergences(aware, "distance error-aware")

    # both find the cluster distance.
    # ABSOLUTA, igual que la de la profundidad tres lineas mas abajo. Era `< 5 * res.mu_r_std`, la
    # tolerancia auto-ensanchable que `tests/CLAUDE.md` prohibe por nombre: pasa sola siempre que el
    # ajuste sea incierto, o sea justo cuando importa. 0,05 kpc es ~2,5x la profundidad inyectada y
    # ~la mitad del error de catalogo mediano (0,104 kpc), asi que sigue siendo holgada para el
    # brazo naive sin volverse insensible.
    for res in (naive, aware):
        assert abs(res.mu_r_mean - TRUE_DIST["mu"]) < 0.05, (
            f"mu_r {res.mu_r_mean:.4f} kpc vs inyectado {TRUE_DIST['mu']}"
        )

    # the error-aware fit recovers the injected depth; the naive one reports the
    # catalogue scatter instead, which is several times larger.
    # ABSOLUTE tolerance -- see the note in the parallax test above. The old
    # `< 4 * std` form was beaten by using (hi - lo) instead of (hi - lo)/2, i.e.
    # assuming twice the catalogue uncertainty.
    assert abs(aware.std_r_mean - TRUE_DIST["depth"]) < 0.012, (
        f"depth {aware.std_r_mean:.4f} kpc vs injected {TRUE_DIST['depth']}"
    )
    assert aware.std_r_std < 0.020
    assert naive.std_r_mean > 3 * aware.std_r_mean
    assert naive.std_r_mean > 0.5 * float(np.median(sigma))


@requires_bayes_extra
@pytest.mark.slow
def test_distance_depth_is_identified_by_the_data_not_by_its_prior():
    """La profundidad que se reporta, ¿la decide el dato o el ``HalfNormal``?

    Al marginalizar los latentes, ``std_r`` sólo queda identificado como el **exceso** de varianza
    observada por encima de los errores conocidos. En el régimen que este modelo declara de interés
    —``std_r`` mucho menor que los errores; acá el error mediano es 5,2x la profundidad inyectada— la
    verosimilitud podría ser casi plana cerca de cero, y entonces el número publicado sería el prior
    y no el dato. Con la tolerancia del test de arriba en ±60% del valor verdadero, una profundidad
    enteramente impuesta por el prior pasaría sin que nada lo dijera.

    Medido 2026-08-24 barriendo ``sigma_scale`` 8x, de 0,025 a 0,20:

    ==============  ================  ===============
    ``sigma_scale``  media del prior   ``std_r``
    ==============  ================  ===============
    0,025            0,0199            0,0194
    0,05  (defecto)  0,0399            0,0233
    0,10             0,0798            0,0243
    0,20             0,1596            0,0253
    ==============  ================  ===============

    ``std_r`` se mueve 0,0058 kpc mientras el prior se mueve 0,1396 — el **4%**. No está dominado
    por el prior. Pero 0,0058 kpc es el **29% de la profundidad inyectada**, así que la sensibilidad
    residual no es cero y ese es el rango de validez que hay que citar junto a cualquier profundidad.

    El límite se pone en 0,010 kpc, ~1,7x lo medido: se rompe si el prior pasa a mandar, y no se
    rompe por ruido de muestreo.

    **Lo que este test NO caza, medido al mutarlo.** Volver a la jerarquía centrada
    (``r_true ~ Gamma`` muestreado + ``r ~ Normal(r_true, errors)``) lo deja **verde**: esa
    parametrización identifica ``std_r`` igual de bien, y lo que rompe es la *geometría*, no la
    identificación. Quien vigila eso son las aserciones de R-hat y ESS de
    ``_assert_no_divergences``, no esta. Las dos mutaciones que sí lo ponen en rojo son las que
    borran ``std_r`` de la verosimilitud —``sigma=errors`` (0,1343 kpc de barrido) y
    ``sigma=sqrt((std_r*0.001)**2 + errors**2)`` (0,1389)— que es exactamente el modo de falla
    que el test declara.
    """
    from erotica.analysis.inference import DistancePriors, distance_model

    table, _ = _distance_table()
    ajustes = {}
    for escala in (0.025, 0.20):
        res = distance_model(
            table,
            distance_lo_column="r_lo_geo",
            distance_hi_column="r_hi_geo",
            sampling=SamplingConfig(
                draws=2000,
                tune=1000,
                chains=2,
                random_seed=8,
                progressbar=False,
                extra_kwargs={"cores": 1},
            ),
            priors=DistancePriors(sigma_scale=escala),
        )
        ajustes[escala] = res.std_r_mean

    barrido = abs(ajustes[0.20] - ajustes[0.025])
    barrido_del_prior = (0.20 - 0.025) * math.sqrt(2 / math.pi)
    assert barrido < 0.010, (
        f"std_r se mueve {barrido:.4f} kpc cuando el prior se mueve {barrido_del_prior:.4f}: "
        f"la profundidad la esta poniendo el prior, no el dato ({ajustes})"
    )
    # Y la otra direccion: si NO se moviera nada, el rango de prior seria demasiado angosto para
    # discriminar y el test estaria pasando por construccion en vez de por medicion.
    assert barrido > 0.0005, (
        f"std_r no se movio ({barrido:.5f} kpc) con el prior barriendo 8x: el barrido no discrimina"
    )


@requires_bayes_extra
def test_distance_priors_do_not_depend_on_the_data():
    """A near and a far cluster get the same prior support and both are found."""
    from erotica.analysis.inference import DistancePriors, distance_model

    rng = np.random.default_rng(2)
    cfg = SamplingConfig(draws=400, tune=400, chains=1, random_seed=0, progressbar=False)
    near = QTable(
        {"r_med_geo": rng.normal(0.4, 0.02, 150) * u.kpc, "parallax": np.full(150, 2.5) * u.mas}
    )
    far = QTable(
        {"r_med_geo": rng.normal(4.0, 0.20, 150) * u.kpc, "parallax": np.full(150, 0.25) * u.mas}
    )
    a = distance_model(near, sampling=cfg)
    b = distance_model(far, sampling=cfg)
    p = DistancePriors()
    assert p.mu_lower < 0.4 and p.mu_upper > 4.0  # one fixed support spans both
    assert abs(a.mu_r_mean - 0.4) < 0.05
    assert abs(b.mu_r_mean - 4.0) < 0.30


def test_distance_bound_validation():
    from erotica.analysis.inference import distance_model

    t = QTable(
        {"r_med_geo": np.linspace(1.0, 1.2, 20) * u.kpc, "parallax": np.full(20, 0.9) * u.mas}
    )
    with pytest.raises(ValueError, match="both"):
        distance_model(t, distance_lo_column="r_med_geo")
    t["lo"] = np.linspace(1.0, 1.2, 20) * u.kpc
    t["hi"] = np.linspace(0.9, 1.1, 20) * u.kpc  # hi < lo everywhere
    with pytest.raises(ValueError, match="must exceed"):
        distance_model(t, distance_lo_column="lo", distance_hi_column="hi")


# ---------------------------------------------------------------------------
# Velocity model -- the fourth model, and it had all three defects too
# ---------------------------------------------------------------------------

TRUE_V = {"mu": 3.5, "sigma_int": 0.4}  # km/s; a realistic OC internal dispersion


def _velocity_sample(n=300, seed=17):
    rng = np.random.default_rng(seed)
    errors = rng.uniform(0.8, 3.0, n)  # typical Gaia-era RV errors, km/s
    truth = rng.normal(TRUE_V["mu"], TRUE_V["sigma_int"], n)
    return truth + rng.normal(0.0, errors), errors


@requires_bayes_extra
def test_velocity_errors_separate_dispersion_from_measurement_noise():
    from erotica.analysis.inference import velocity_model

    v, e = _velocity_sample()
    cfg = SamplingConfig(draws=1200, tune=1000, chains=2, random_seed=6, progressbar=False)

    naive = velocity_model(v, sampling=cfg)
    aware = velocity_model(v, errors=e, sampling=cfg)

    assert naive.metadata["error_aware"] is False
    assert aware.metadata["error_aware"] is True

    assert abs(aware.std_v_mean - TRUE_V["sigma_int"]) < 0.25  # absolute, not N*std
    assert aware.std_v_std < 0.35
    assert naive.std_v_mean > 2 * aware.std_v_mean
    for res in (naive, aware):
        assert abs(res.mu_v_mean - TRUE_V["mu"]) < 5 * res.mu_v_std


@requires_bayes_extra
def test_velocity_dispersion_prior_is_not_eighty_times_too_wide():
    """The old prior was ``Uniform(0, 40)`` km/s on a ~0.5 km/s quantity.

    In the low-dispersion regime that leaves the posterior prior-dominated. The
    replacement must both (a) put most of its mass at plausible dispersions and
    (b) still reach an unusually hot cluster through its tail.
    """
    import pymc as pm

    from erotica.analysis.inference import VelocityPriors

    p = VelocityPriors()
    with pm.Model():
        pm.HalfNormal("std_v", sigma=p.sigma_scale)
        draws = np.asarray(
            pm.sample_prior_predictive(draws=8000, random_seed=0).prior["std_v"].values
        ).ravel()

    assert (draws < 2.0).mean() > 0.6  # most mass where open clusters live
    assert (draws > 5.0).mean() > 0.005  # but a hot cluster is still reachable
    # the old Uniform(0, 40) put only ~5% of its mass below 2 km/s
    assert (np.random.default_rng(0).uniform(0, 40, 8000) < 2.0).mean() < 0.10


def test_velocity_error_validation():
    from erotica.analysis.inference import velocity_model

    v = np.linspace(1.0, 5.0, 20)
    with pytest.raises(ValueError, match="match the velocities"):
        velocity_model(v, errors=np.ones(5))
    with pytest.raises(ValueError, match="finite and positive"):
        velocity_model(v, errors=np.zeros(20))


@requires_bayes_extra
def test_zero_point_nuisance_widens_the_mean_parallax_by_the_systematic_floor():
    """Oracle: quadrature. The residual ZP is degenerate with the mean by
    construction, so enabling it must widen ``mu_parallax`` by ~the floor.

    Gaia's zero-point correction is, per Lindegren et al. (2021), "not perfect",
    and its residual is correlated across a cluster -- every member carries the
    same leftover offset, which averaging cannot remove. Vasiliev & Baumgardt
    (2021) put that floor at 10.3 uas for a compact cluster.
    """
    from erotica.analysis.inference import ParallaxPriors

    table, _ = _parallax_table()
    cfg = SamplingConfig(draws=1500, tune=1000, chains=2, random_seed=12, progressbar=False)

    plain = fit_parallax_model(table, parallax_error_column="parallax_error", sampling=cfg)
    withzp = fit_parallax_model(
        table, parallax_error_column="parallax_error", zero_point=True, sampling=cfg
    )

    assert plain.metadata["zero_point_nuisance"] is False
    assert withzp.metadata["zero_point_nuisance"] is True

    # The floor is asserted as a LITERAL, not read from the dataclass the model
    # also reads. The old form compared the model against its own constant, so
    # hardcoding a different floor inside the model passed.
    PUBLISHED_FLOOR_MAS = 0.0103  # Maiz Apellaniz+2021, A&A 649, A13
    assert ParallaxPriors().zero_point_scale == pytest.approx(PUBLISHED_FLOOR_MAS)
    expected = np.hypot(plain.mu_parallax_std, PUBLISHED_FLOOR_MAS)
    assert withzp.mu_parallax_std == pytest.approx(expected, rel=0.12)
    # it must genuinely widen, not merely differ
    assert withzp.mu_parallax_std > plain.mu_parallax_std
    # and the mean itself must not move -- this adds uncertainty, not a shift
    assert abs(withzp.mu_parallax_mean - plain.mu_parallax_mean) < 3 * plain.mu_parallax_std
    # the cluster's intrinsic depth is a different parameter and must be unaffected
    assert withzp.sigma_parallax_mean == pytest.approx(plain.sigma_parallax_mean, rel=0.35)


# ---------------------------------------------------------------------------
# The analyzer wrapper -- Defect 1.
#
# Every error-aware branch exercised above already existed. Until 2026-08-02 the
# wrapper that the pipeline actually calls reached none of them: it loaded
# ``parallax_error`` only in order to DISCARD stars and then forwarded neither
# the errors, nor the Bailer-Jones bounds, nor ``zero_point``.
#
# ORACLE: the same injected-truth design as the model-level tests. An intrinsic
# spread deliberately far below the measurement scatter, so a wrapper that drops
# the errors on the floor must report the scatter as cluster depth and cannot
# accidentally land on the injected value.
# ---------------------------------------------------------------------------

WRAPPER_TRUE_DEPTH_KPC = 0.005  # 5 pc of line-of-sight depth
WRAPPER_DISTANCE_ERR_KPC = 0.050  # 50 pc per-star Bailer-Jones error, 10x larger
WRAPPER_MU_R_KPC = 1.11


def _wrapper_table(n=400, seed=20260802):
    """One cluster, known depth, known per-star errors, all columns wired.

    ``r_lo_geo``/``r_hi_geo`` are placed symmetrically about ``r_med_geo`` at
    exactly the error used to generate the scatter, so ``(hi - lo)/2`` recovers
    the injected ``sigma_i`` with no modelling assumption.
    """
    rng = np.random.default_rng(seed)
    errors = np.repeat(REAL_ERROR_SCALE, n // 4)
    errors = errors * rng.uniform(0.8, 1.2, errors.size)
    true_plx = rng.normal(TRUE_PARALLAX, TRUE_INTRINSIC, errors.size)
    observed_plx = true_plx + rng.normal(0.0, errors)

    d_err = np.full(errors.size, WRAPPER_DISTANCE_ERR_KPC)
    true_r = rng.normal(WRAPPER_MU_R_KPC, WRAPPER_TRUE_DEPTH_KPC, errors.size)
    observed_r = true_r + rng.normal(0.0, d_err)

    return QTable(
        {
            "probability": np.full(errors.size, 0.9),
            "parallax": observed_plx * u.mas,
            "parallax_error": errors * u.mas,
            "r_med_geo": observed_r * u.kpc,
            "r_lo_geo": (observed_r - d_err) * u.kpc,
            "r_hi_geo": (observed_r + d_err) * u.kpc,
        }
    )


def _wrapper_fit(table, **kwargs):
    analyzer = ClusterInferenceAnalyzer(
        table,
        sampling=SamplingConfig(
            draws=600,
            tune=600,
            chains=2,
            random_seed=21,
            progressbar=False,
            extra_kwargs={"cores": 1},
        ),
    )
    out = analyzer.distance_and_parallax_by_probability((0.5,), **kwargs)
    return {key: values[0] for key, values in out.items()}


@requires_bayes_extra
def test_the_wrapper_default_puts_per_star_errors_into_both_likelihoods():
    """The shipped default must separate measurement scatter from cluster depth.

    Absolute tolerances tied to the injected truth -- never ``N * posterior_std``,
    which self-widens whenever the fit degrades and so cannot fail for the reason
    it was written.
    """
    table = _wrapper_table()

    aware = _wrapper_fit(table)
    naive = _wrapper_fit(
        table,
        parallax_error_column=None,
        distance_lo_column=None,
        distance_hi_column=None,
        zero_point=False,
    )

    # --- parallax: intrinsic depth, not measurement scatter ------------------
    assert abs(aware["sigma_parallax_mean"] - TRUE_INTRINSIC) < 0.004, (
        f"intrinsic parallax spread {aware['sigma_parallax_mean']:.5f} vs injected {TRUE_INTRINSIC}"
    )
    # The posterior must also be INFORMATIVE, not merely centred. A recovery test
    # in a non-identifiable regime cannot fail for the reason it was written --
    # and this exact gate is what kills the mutation that drops the square from
    # sqrt(sigma^2 + e_i^2): the mean stays within a loose band (0.0046 against an
    # injected 0.010) while the posterior width quadruples, 0.0008 -> 0.0035.
    assert aware["sigma_parallax_std"] < 0.003, (
        f"posterior on the intrinsic spread is uninformative: "
        f"sd = {aware['sigma_parallax_std']:.5f}"
    )
    assert naive["sigma_parallax_mean"] > 2.5 * TRUE_INTRINSIC
    assert naive["sigma_parallax_mean"] > 2.5 * aware["sigma_parallax_mean"]

    # --- distance: the Bailer-Jones bounds must reach distance_model ---------
    assert abs(aware["std_r_mean"] - WRAPPER_TRUE_DEPTH_KPC) < 0.010, (
        f"cluster depth {aware['std_r_mean']:.4f} kpc vs injected {WRAPPER_TRUE_DEPTH_KPC} kpc"
    )
    # Without the bounds, std_r absorbs the 50 pc per-star error as if it were depth.
    assert naive["std_r_mean"] > 3 * WRAPPER_TRUE_DEPTH_KPC
    assert naive["std_r_mean"] > 3 * aware["std_r_mean"]

    # --- both must still find the centres; the defect is in the widths -------
    assert abs(aware["mu_parallax_mean"] - TRUE_PARALLAX) < 0.03
    assert abs(aware["mu_r_mean"] - WRAPPER_MU_R_KPC) < 0.03


@requires_bayes_extra
def test_the_wrapper_default_carries_the_gaia_systematic_floor():
    """Oracle: quadrature against a literal floor, through the wrapper's default.

    The zero-point nuisance is exactly degenerate with ``mu_parallax`` for one
    cluster, so enabling it must widen the reported uncertainty by the systematic
    floor and must NOT move the mean. The floor is written as a literal here, not
    read from ``ParallaxPriors``, so a model that hardcodes a different value
    cannot agree with itself.
    """
    table = _wrapper_table(n=120)

    withzp = _wrapper_fit(table)  # default is zero_point=True
    without = _wrapper_fit(table, zero_point=False)

    PUBLISHED_FLOOR_MAS = 0.0103  # Maiz Apellaniz+2021, A&A 649, A13
    expected = float(np.hypot(without["mu_parallax_std"], PUBLISHED_FLOOR_MAS))
    assert withzp["mu_parallax_std"] == pytest.approx(expected, rel=0.15)
    assert withzp["mu_parallax_std"] > without["mu_parallax_std"]
    assert abs(withzp["mu_parallax_mean"] - without["mu_parallax_mean"]) < 3 * expected


def test_the_wrapper_no_longer_pre_cuts_the_sample(monkeypatch):
    """The default must fit every selected star, not a precision-selected subset.

    ``sigma_varpi/varpi <= 0.1`` deletes the faint, low-precision end -- a
    magnitude-dependent selection imposed on the inference sample. Luri et al.
    (2018, A&A 616, A9) state that "parallaxes with relatively large uncertainties
    still contain valuable information"; once the errors are in the likelihood
    those stars are down-weighted rather than discarded.
    """
    captured = _capture_useful(monkeypatch)
    analyzer = ClusterInferenceAnalyzer(_make_table(), probability_column="probability")

    analyzer.distance_and_parallax_by_probability(probability_thresholds=(0.5,))

    kept = set(np.asarray(captured["useful"]["source_id"]).tolist())
    assert kept == {1, 2, 3, 4}, "the default still pre-cuts the sample"
    # and the parallax model must see the identical sample
    assert set(np.asarray(captured["useful_parallax"]["source_id"]).tolist()) == kept


def test_a_missing_uncertainty_column_is_refused_not_silently_dropped():
    """Degrading to the uncertainty-free path on a missing column would be silent."""
    table = _make_table()
    table.remove_column("parallax_error")
    analyzer = ClusterInferenceAnalyzer(table, probability_column="probability")
    with pytest.raises(ValueError, match="Missing uncertainty column"):
        analyzer.distance_and_parallax_by_probability(probability_thresholds=(0.5,))


def test_input_validation_does_not_require_the_bayes_extra(monkeypatch):
    """A malformed table is malformed whether or not PyMC is installed.

    Raising ``ImportError`` first hides the real defect from anyone without the
    optional extra -- and made eight input-validation tests unrunnable in CI.
    """

    def no_pymc():
        raise ImportError("PyMC is required for erotica.analysis.inference.")

    monkeypatch.setattr(inference, "_require_pymc", no_pymc)

    bad = QTable(
        {
            "parallax": np.linspace(0.8, 1.0, 10) * u.mas,
            "parallax_error": np.concatenate([[0.0], np.full(9, 0.05)]) * u.mas,
        }
    )
    with pytest.raises(ValueError, match="finite and positive"):
        inference.fit_parallax_model(bad, parallax_error_column="parallax_error")

    with pytest.raises(ValueError, match="prior_type"):
        inference.fit_parallax_model(
            QTable({"parallax": np.linspace(0.8, 1.0, 10) * u.mas}), prior_type="lognormal"
        )

    dist = QTable(
        {
            "r_med_geo": np.linspace(1.0, 1.2, 10) * u.kpc,
            "r_lo_geo": np.linspace(1.1, 1.3, 10) * u.kpc,  # lo above hi
            "r_hi_geo": np.linspace(1.0, 1.2, 10) * u.kpc,
        }
    )
    with pytest.raises(ValueError, match="must exceed"):
        inference.distance_model(dist, distance_lo_column="r_lo_geo", distance_hi_column="r_hi_geo")


# --- D9 y D10: las dos cifras que el paquete calculaba y no devolvia -------------------------


def test_moda_predictiva_gamma_es_la_forma_cerrada():
    """La moda de la Gamma es `(k-1)*theta`, no un pico estimado por KDE.

    Se afirma contra la forma cerrada y no contra un numero guardado: una moda por KDE se aleja de
    esta 0.0152 / 0.0049 / 0.0045 kpc segun cuantas extracciones se le den, o sea **20 a 70 veces**
    la diferencia entre los dos priores que el manuscrito de NGC 6383 discute. El ultimo digito
    impreso lo decidia el estimador y no el modelo.
    """
    from erotica.analysis.inference import _moda_predictiva

    mu = np.full(500, 1.11)
    sigma = np.full(500, 0.06)
    k = (mu / sigma) ** 2
    esperado = float(np.median((k - 1.0) * (sigma**2 / mu)))
    assert _moda_predictiva(mu, sigma, gamma=True) == pytest.approx(esperado, rel=1e-12)
    # Y no depende del tamano de la muestra, que es la propiedad que el KDE no tiene.
    corto = _moda_predictiva(mu[:50], sigma[:50], gamma=True)
    assert corto == pytest.approx(esperado, rel=1e-12)


def test_moda_predictiva_normal_es_mu():
    """En la rama error-aware la poblacion es Normal marginalizada y su moda ES la media."""
    from erotica.analysis.inference import _moda_predictiva

    mu = np.linspace(1.0, 1.2, 200)
    sigma = np.full(200, 0.03)
    assert _moda_predictiva(mu, sigma, gamma=False) == pytest.approx(float(np.mean(mu)), rel=1e-12)


@requires_bayes_extra
@pytest.mark.slow
def test_distance_std_incluye_el_suelo_del_cero():
    """`zero_point=True` tiene que ENSANCHAR la incertidumbre de la distancia, no moverla.

    Es la afirmacion entera de D10: el suelo sistematico del cero residual (10.3 uas, Maiz
    Apellaniz+2021) domina la incertidumbre correcta de la media —medido sobre el ajuste publicado
    de NGC 6383, **86.5% de la varianza**— y sin el nuisance queda fuera entera. El camino ingenuo,
    el error estandar de la media a secas, se queda 2.7x corto.

    Se afirma la RELACION y no una cifra: la relacion es lo que el nuisance promete.
    """
    from erotica.analysis.inference import ParallaxPriors, SamplingConfig, fit_parallax_model

    rng = np.random.default_rng(20260826)
    n = 120
    verdad, disp, err = 0.9, 0.01, 0.04
    plx = rng.normal(verdad, np.hypot(disp, err), n)
    t = QTable()
    t["parallax"] = plx * u.mas
    t["parallax_error"] = np.full(n, err) * u.mas
    cfg = SamplingConfig(
        draws=800,
        tune=800,
        target_accept=0.9,
        chains=2,
        random_seed=7,
        nuts_sampler="pymc",
        progressbar=False,
        extra_kwargs={"cores": 1},
    )
    priors = ParallaxPriors()
    con = fit_parallax_model(
        t, parallax_error_column="parallax_error", zero_point=True, sampling=cfg, priors=priors
    )
    sin = fit_parallax_model(
        t, parallax_error_column="parallax_error", zero_point=False, sampling=cfg, priors=priors
    )

    assert math.isfinite(con.distance_mean) and math.isfinite(con.distance_std)
    # El valor central no se mueve: el nuisance es un ensanchador, NO una correccion de sesgo.
    assert con.distance_mean == pytest.approx(sin.distance_mean, rel=0.02)
    # Y la anchura crece por el suelo, que aqui domina sobre la parte estadistica.
    assert con.distance_std > sin.distance_std
    esperado = math.hypot(sin.distance_std, priors.zero_point_scale / con.mu_parallax_mean**2)
    assert con.distance_std == pytest.approx(esperado, rel=0.25)


def test_distance_model_sin_errores_avisa_de_su_defecto(monkeypatch):
    """La rama Gamma-sobre-`r_med_geo` avisa, y la marginalizada no.

    Retirada 2026-08-26 (J.0 #5) pero **no borrada**: el `1.11 +- 0.06 kpc` de P01 salio de ahi y
    tiene que seguir reproducible, igual que `RDP_bayesian` para los radios. Lo que se retira es que
    sea una ruta por defecto silenciosa, no que exista.

    El aviso nombra la SALIDA —`fit_parallax_model`— y no solo el defecto: un aviso que no dice que
    usar en su lugar deja al lector donde estaba.

    `_require_pymc` se corta a proposito: el aviso se emite ANTES, asi que el test es determinista
    con o sin el extra `bayes` instalado y no muestrea nada.
    """
    import erotica.analysis.inference as inf

    monkeypatch.setattr(inf, "_require_pymc", lambda: (_ for _ in ()).throw(RuntimeError("corte")))
    t = QTable()
    t["r_med_geo"] = np.linspace(1.0, 1.2, 40) * u.kpc
    t["r_lo_geo"] = (np.linspace(1.0, 1.2, 40) - 0.05) * u.kpc
    t["r_hi_geo"] = (np.linspace(1.0, 1.2, 40) + 0.05) * u.kpc

    with pytest.warns(UserWarning, match="fit_parallax_model"):
        with pytest.raises(RuntimeError, match="corte"):
            inf.distance_model(t)

    # La rama marginalizada NO avisa: es la que se recomienda.
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(RuntimeError, match="corte"):
            inf.distance_model(t, distance_lo_column="r_lo_geo", distance_hi_column="r_hi_geo")
