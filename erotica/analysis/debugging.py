"""Debugging and migration validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from astropy import units as u


@dataclass(frozen=True)
class DistancePosteriorComparison:
    """Numerical comparison between two distance posteriors."""

    mode_a: u.Quantity
    mode_b: u.Quantity
    delta: u.Quantity
    sigma_a_formal: u.Quantity
    sigma_b_16_84: u.Quantity
    sigma_combined: u.Quantity
    delta_in_sigma_combined: float
    delta_in_sigma_a_formal: float
    delta_interval_16_84: u.Quantity
    delta_prob_positive: float


def summarize_trace_vars(trace) -> dict[str, tuple[int, ...]]:
    """Return posterior variable names and shapes for quick notebook debugging."""
    if not hasattr(trace, "posterior"):
        raise ValueError("Trace must have a posterior group.")
    return {name: tuple(trace.posterior[name].shape) for name in trace.posterior.data_vars}


def _flatten_posterior_samples(trace, var_name: str):
    if trace is None or not hasattr(trace, "posterior") or var_name not in trace.posterior:
        return None
    samples = np.asarray(trace.posterior[var_name].values, dtype=float).ravel()
    return samples[np.isfinite(samples)]


def extract_distance_samples(trace_or_result, *, prefer: str = "mu_r") -> np.ndarray:
    """Extract distance samples in kpc from a trace or a result dict with traces."""
    traces = []
    if isinstance(trace_or_result, Mapping):
        if "traces" in trace_or_result:
            traces.extend(trace_or_result["traces"])
        elif "trace" in trace_or_result:
            traces.append(trace_or_result["trace"])
    else:
        traces.append(trace_or_result)

    names = [prefer, "mu_r", "mu_parallax"] if prefer != "mu_parallax" else ["mu_parallax", "mu_r"]
    for name in names:
        for trace in traces:
            samples = _flatten_posterior_samples(trace, name)
            if samples is None or len(samples) == 0:
                continue
            if name == "mu_parallax":
                samples = samples[samples > 0]
                return 1 / samples
            return samples
    raise ValueError("No distance posterior found. Expected 'mu_r' or 'mu_parallax'.")


def validate_cluster_table(table, required_columns, units: Mapping[str, u.UnitBase] | None = None) -> None:
    """Validate required columns and optional units."""
    missing = [column for column in required_columns if column not in table.colnames]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    for column, expected_unit in (units or {}).items():
        if column not in table.colnames:
            continue
        actual = getattr(table[column], "unit", None)
        if actual is None:
            raise ValueError(f"Column {column!r} has no unit; expected {expected_unit}.")
        try:
            actual.to(expected_unit)
        except Exception as exc:
            raise ValueError(
                f"Column {column!r} has unit {actual}, not convertible to {expected_unit}."
            ) from exc


def compare_legacy_output(new, legacy, tolerances: Mapping[str, float]) -> dict[str, float]:
    """Compare migrated numeric outputs against legacy values."""
    deltas = {}
    for key, tolerance in tolerances.items():
        if key not in new or key not in legacy:
            raise KeyError(f"Cannot compare missing key {key!r}.")
        new_value = getattr(new[key], "value", new[key])
        legacy_value = getattr(legacy[key], "value", legacy[key])
        delta = float(np.nanmax(np.abs(np.asarray(new_value) - np.asarray(legacy_value))))
        if delta > tolerance:
            raise AssertionError(f"{key} differs by {delta}, tolerance {tolerance}.")
        deltas[key] = delta
    return deltas


def diagnose_distance_comparison(distance_a, distance_b, *, random_seed=42):
    """Compute mode offset and combined posterior-width diagnostics."""
    from scipy.stats import gaussian_kde

    a = np.asarray(distance_a, dtype=float).ravel()
    b = np.asarray(distance_b, dtype=float).ravel()
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        raise ValueError("Both distance posteriors must contain finite samples.")

    def mode(samples):
        grid = np.linspace(*np.nanpercentile(samples, [0.5, 99.5]), 400)
        density = gaussian_kde(samples)(grid)
        return float(grid[int(np.argmax(density))])

    mode_a = mode(a)
    mode_b = mode(b)
    delta = mode_b - mode_a
    sigma_a = float(np.nanstd(a))
    b_16_84 = np.nanpercentile(b, [16, 84])
    sigma_b = float(0.5 * (b_16_84[1] - b_16_84[0]))
    sigma_combined = float(np.sqrt(sigma_a**2 + sigma_b**2))
    rng = np.random.default_rng(random_seed)
    n = min(len(a), len(b), 100_000)
    delta_samples = rng.choice(b, n, replace=False) - rng.choice(a, n, replace=False)
    return DistancePosteriorComparison(
        mode_a=mode_a * u.kpc,
        mode_b=mode_b * u.kpc,
        delta=delta * u.kpc,
        sigma_a_formal=sigma_a * u.kpc,
        sigma_b_16_84=sigma_b * u.kpc,
        sigma_combined=sigma_combined * u.kpc,
        delta_in_sigma_combined=delta / sigma_combined,
        delta_in_sigma_a_formal=delta / sigma_a,
        delta_interval_16_84=np.nanpercentile(delta_samples, [16, 84]) * u.kpc,
        delta_prob_positive=float(np.mean(delta_samples > 0)),
    )


__all__ = [
    "DistancePosteriorComparison",
    "compare_legacy_output",
    "diagnose_distance_comparison",
    "extract_distance_samples",
    "summarize_trace_vars",
    "validate_cluster_table",
]
