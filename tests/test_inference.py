"""Regression tests for :mod:`pumps.analysis.inference`.

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
from astropy import units as u
from astropy.table import QTable

import pumps.analysis.inference as inference
from pumps.analysis.inference import (
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
