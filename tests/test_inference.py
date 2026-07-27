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
