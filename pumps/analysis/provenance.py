"""Trace and result provenance helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pumps import __version__ as _PUMPS_VERSION


@dataclass(frozen=True)
class TraceSummary:
    """Compact summary of a posterior variable."""

    variable: str
    mean: float
    median: float
    std: float
    mode: float


def _require_arviz():
    try:
        import arviz as az
    except ImportError as exc:
        raise ImportError(
            "ArviZ is required for trace provenance helpers. Install the 'bayes' extra."
        ) from exc
    return az


def posterior_mode(values) -> float:
    """Estimate posterior mode with ArviZ KDE if available, else histogram mode."""
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError("Cannot estimate mode for empty posterior values.")
    try:
        from arviz import kde

        density, grid = kde(values)
        return float(grid[int(np.argmax(density))])
    except Exception:
        hist, edges = np.histogram(values, bins="auto")
        idx = int(np.argmax(hist))
        return float(0.5 * (edges[idx] + edges[idx + 1]))


def calculate_mode(values, bw: str = "default", circular: bool = False) -> float:
    """Legacy-compatible posterior mode estimator.

    ``bw`` and ``circular`` are accepted for compatibility with the old notebook
    helper. The implementation delegates to :func:`posterior_mode`, which uses
    ArviZ KDE when available and a histogram fallback otherwise.
    """
    del bw, circular
    return posterior_mode(values)


def summarize_trace(trace, *, excluded_parameters=("likelihood", "likelihood_unobserved")) -> list[TraceSummary]:
    """Summarize all posterior variables in an ArviZ InferenceData object."""
    if not hasattr(trace, "posterior"):
        raise ValueError("Trace must be an ArviZ InferenceData object with a posterior group.")
    summaries: list[TraceSummary] = []
    for variable in trace.posterior.data_vars:
        if variable in excluded_parameters:
            continue
        values = np.asarray(trace.posterior[variable].values, dtype=float).ravel()
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        summaries.append(
            TraceSummary(
                variable=variable,
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                std=float(np.std(values)),
                mode=posterior_mode(values),
            )
        )
    return summaries


def store_trace_results(
    trace,
    file_path,
    *,
    excluded_parameters=("likelihood", "likelihood_unobserved"),
    save_trace: bool = True,
) -> pd.DataFrame:
    """Append summarized trace statistics and optionally write the full trace to NetCDF."""
    path = Path(file_path)
    summaries = summarize_trace(trace, excluded_parameters=excluded_parameters)
    trace_index = int(datetime.now(tz=timezone.utc).timestamp())
    rows = [
        {
            "Parameter": item.variable,
            "Mean": item.mean,
            "Median": item.median,
            "Standard_Deviation": item.std,
            "Std": item.std,
            "Mode": item.mode,
            "Date_Time": datetime.now(tz=timezone.utc).isoformat(),
            "Trace_Index": trace_index,
        }
        for item in summaries
    ]
    new_data = pd.DataFrame(rows)
    if path.exists():
        updated = pd.concat([pd.read_csv(path), new_data], ignore_index=True)
    else:
        updated = new_data
    updated.to_csv(path, index=False)
    if save_trace:
        trace.to_netcdf(path.with_name(f"{path.stem}_trace_{trace_index}.nc"))
    return updated


def load_results(file_path="fit_parameters.csv", *, load_trace=False, only_last=True):
    """Load summarized CSV results and optionally the associated NetCDF trace."""
    az = _require_arviz() if load_trace else None
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No summarized results file found: {path}")
    results = pd.read_csv(path)
    trace_path = None
    if only_last:
        results["Date_Time"] = pd.to_datetime(results["Date_Time"])
        max_date = results["Date_Time"].max()
        results = results[results["Date_Time"] == max_date]
        trace_index = int(results.iloc[0]["Trace_Index"])
        trace_path = path.with_name(f"{path.stem}_trace_{trace_index}.nc")
    trace = None
    if load_trace:
        if trace_path is None or not trace_path.exists():
            raise FileNotFoundError(f"No trace file found for summarized results: {trace_path}")
        trace = az.from_netcdf(trace_path)
    return results, trace


def build_metadata(**kwargs: Any) -> dict[str, Any]:
    """Build a small provenance dictionary."""
    return {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "pumps_version": _PUMPS_VERSION,
        **kwargs,
    }


__all__ = [
    "TraceSummary",
    "build_metadata",
    "calculate_mode",
    "load_results",
    "posterior_mode",
    "store_trace_results",
    "summarize_trace",
]
