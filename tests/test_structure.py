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
