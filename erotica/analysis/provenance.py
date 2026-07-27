"""Trace and result provenance helpers.

A result is reproducible only if you can say *what code, what inputs, and what
randomness* produced it. :func:`build_metadata` records all three in a
JSON-serialisable dictionary: the git commit and whether the tree was dirty, a
content checksum of every input file, the RNG seeds, and the versions of the
packages that actually do the numerics.

See :doc:`/design-notes/decisions` for why each field is present.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from erotica import __version__ as _EROTICA_VERSION

#: Packages whose version can change a number. Recorded when installed, skipped
#: when not -- an absent optional extra is itself provenance (a run without
#: ``[bayes]`` produced no Bayesian numbers).
_TRACKED_DISTRIBUTIONS: tuple[str, ...] = (
    "numpy",
    "scipy",
    "pandas",
    "astropy",
    "scikit-learn",
    "hdbscan",
    "matplotlib",
    "pymc",
    "pytensor",
    "arviz",
    "numpyro",
    "jax",
    "blackjax",
    "optuna",
)


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
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Append summarized trace statistics and optionally write the full trace to NetCDF.

    When `save_trace` is set, a ``*_provenance_<index>.json`` sidecar is written
    next to the NetCDF recording the code version, git state and dependency
    versions that produced it (see :func:`build_metadata`). Extra fields can be
    supplied through `metadata` -- typically the input catalogue path and the
    sampler seed, which this function cannot discover on its own.
    """
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
        write_metadata(
            path.with_name(f"{path.stem}_provenance_{trace_index}.json"),
            trace_index=trace_index,
            variables=[item.variable for item in summaries],
            **(metadata or {}),
        )
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


def _jsonable(obj: Any) -> Any:
    """Coerce `obj` into something :func:`json.dumps` accepts.

    Provenance that cannot be written to disk is not provenance. NumPy scalars,
    ``Path``, dataclasses (e.g. ``SamplingConfig``) and Astropy ``Quantity`` all
    appear naturally in callers' keyword arguments and all break ``json.dumps``,
    so they are converted rather than allowed to raise at write time.
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, np.generic):  # numpy scalar -> Python scalar
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in obj]
    return str(obj)  # last resort: never lose the field, only its type


def git_provenance(path: str | os.PathLike[str] | None = None) -> dict[str, Any] | None:
    """Return the git commit and dirty flag for the tree containing `path`.

    Parameters
    ----------
    path : str or path-like, optional
        Any location inside the repository. Defaults to the installed package,
        which is the tree that produced the code being run.

    Returns
    -------
    dict or None
        ``{"commit": <40-hex>, "dirty": bool, "branch": str}``, or ``None`` when
        `path` is not inside a git repository or ``git`` is unavailable -- the
        normal case for a wheel installed from PyPI.

    Notes
    -----
    ``dirty`` is the load-bearing field. A commit hash alone is misleading if the
    working tree had uncommitted edits when the run happened, which is the usual
    state during analysis.
    """
    root = Path(path) if path is not None else Path(__file__).resolve().parent

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(root), *args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return None
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "dirty": bool(status),  # None (failed) and "" (clean) both -> False
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    }


def file_checksum(path: str | os.PathLike[str], *, algorithm: str = "blake2b") -> dict[str, Any]:
    """Content digest and size of one input file.

    Streams the file, so a multi-GB catalogue costs memory in kilobytes. The
    digest identifies the *content*: a file that is renamed, re-downloaded or
    re-sorted into a different path but is byte-identical yields the same value,
    which is what "same input" has to mean.
    """
    p = Path(path)
    with p.open("rb") as fh:
        digest = hashlib.file_digest(fh, algorithm)  # stdlib, streams in chunks
    return {
        "path": str(p),
        "bytes": p.stat().st_size,
        algorithm: digest.hexdigest(),
    }


def dependency_versions() -> dict[str, str]:
    """Installed versions of the packages that can change a number.

    Read from installed distribution metadata rather than by importing each
    package -- importing PyMC costs seconds and is a side effect a provenance
    call has no business causing.
    """
    found: dict[str, str] = {}
    for dist in _TRACKED_DISTRIBUTIONS:
        try:
            found[dist] = _dist_version(dist)
        except PackageNotFoundError:
            continue
    return found


def build_metadata(
    *,
    inputs: Any = None,
    seeds: Any = None,
    strict: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Build a reproducibility record for a result.

    Parameters
    ----------
    inputs : path-like or sequence of path-like, optional
        Input data files to checksum. Missing files are recorded as an error
        entry rather than raising, so a provenance call never destroys the
        result it was meant to describe -- unless `strict` is set.
    seeds : int or dict or sequence, optional
        RNG seed(s) used. Recorded verbatim; ``None`` is itself informative and
        is preserved, because a run with no fixed seed is not reproducible and
        the record should say so.
    strict : bool, default False
        Raise instead of recording an error when an input file cannot be read.
    **kwargs
        Anything else worth recording -- sampler configuration, cluster label,
        selection cuts. Coerced to JSON-safe values.

    Returns
    -------
    dict
        Always JSON-serialisable. Keys: ``created_at``, ``erotica_version``,
        ``git``, ``python``, ``platform``, ``dependencies``, ``seeds``,
        ``inputs``, plus the caller's keywords.

    Examples
    --------
    >>> meta = build_metadata(seeds=42, cluster=3)
    >>> sorted(meta)[:3]
    ['cluster', 'created_at', 'dependencies']
    >>> import json; isinstance(json.dumps(meta), str)
    True
    """
    if inputs is None:
        input_records: list[dict[str, Any]] = []
    else:
        paths = [inputs] if isinstance(inputs, (str, os.PathLike)) else list(inputs)
        input_records = []
        for item in paths:
            try:
                input_records.append(file_checksum(item))
            except OSError as exc:
                if strict:
                    raise
                input_records.append({"path": str(item), "error": str(exc)})

    record = {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "erotica_version": _EROTICA_VERSION,
        "git": git_provenance(),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.machine()}",
        "dependencies": dependency_versions(),
        "seeds": _jsonable(seeds),
        "inputs": input_records,
        **{k: _jsonable(v) for k, v in kwargs.items()},
    }
    json.dumps(record)  # fail here, loudly, not at write time
    return record


def write_metadata(path: str | os.PathLike[str], **kwargs: Any) -> Path:
    """Write :func:`build_metadata` to `path` as indented JSON."""
    target = Path(path)
    target.write_text(json.dumps(build_metadata(**kwargs), indent=2))
    return target


__all__ = [
    "TraceSummary",
    "build_metadata",
    "calculate_mode",
    "dependency_versions",
    "file_checksum",
    "git_provenance",
    "load_results",
    "posterior_mode",
    "store_trace_results",
    "summarize_trace",
    "write_metadata",
]
