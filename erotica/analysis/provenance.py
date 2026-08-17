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
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _dist_version
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
    """Estimate posterior mode with ArviZ KDE if available, else histogram mode.

    Parameters
    ----------
    values : array-like
        Posterior draws for one variable. Ravelled first, so chain and draw
        structure is discarded and a vector-valued variable is pooled into a
        single sample. Non-finite draws are removed before the estimate.

    Returns
    -------
    float
        The mode, as a plain float in whatever unit the caller's draws carried.
        This function is unit-agnostic: it neither reads nor attaches units.

    Raises
    ------
    ValueError
        If no finite draws remain.

    Notes
    -----
    The fallback is guarded by a bare ``except Exception``, so it is entered on
    **any** KDE failure, not only on ArviZ being absent. A degenerate posterior
    that makes :func:`arviz.kde` raise therefore returns a histogram mode with
    no warning. The two estimators do not agree in general -- the fallback
    returns a bin centre from ``bins="auto"``, so its resolution is set by the
    number of draws, not by the width of the posterior.
    """
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


def summarize_trace(
    trace, *, excluded_parameters=("likelihood", "likelihood_unobserved")
) -> list[TraceSummary]:
    """Summarize all posterior variables in an ArviZ InferenceData object.

    Parameters
    ----------
    trace : arviz.InferenceData
        Must carry a ``posterior`` group.
    excluded_parameters : sequence of str, optional
        Variable names to skip. The defaults are the log-likelihood nodes, which
        are per-draw bookkeeping rather than parameters of the model.

    Returns
    -------
    list of TraceSummary
        One entry per retained variable, in ``trace.posterior.data_vars`` order,
        each holding ``mean``, ``median``, ``std`` and ``mode`` as plain floats.
        Units are **not** carried: a radius sampled in arcmin comes back as a
        bare number, and the caller must reattach the unit.

    Raises
    ------
    ValueError
        If `trace` has no ``posterior`` attribute.

    Notes
    -----
    Each variable is **ravelled before summarising**, so chains are pooled *and
    a vector-valued variable collapses to a single row covering every element at
    once*. If a model samples one mass per star, the row named ``mass`` is the
    mean over all stars and all draws together -- not a per-star quantity. Read
    per-element summaries off the trace directly instead.

    A variable whose draws are all non-finite is dropped silently rather than
    reported as NaN, so an absent row means "nothing finite to summarise", not
    "not sampled".
    """
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


def _same_posterior(a, b) -> bool:
    """Whether two traces carry the same posterior draws.

    Deliberately strict and deliberately cheap to reason about: same variable names, same shapes,
    same values. Anything else -- sample_stats, attrs, creation metadata -- is allowed to differ,
    because it does differ between two writes of one trace and is not what identifies the fit.
    """
    try:
        pa, pb = a.posterior, b.posterior
        va, vb = set(pa.data_vars), set(pb.data_vars)
        if va != vb:
            return False
        # equal_nan=True is required, not defensive. `np.array_equal([1.0, nan], [1.0, nan])` is
        # False by default, so a posterior holding ANY non-finite draw would never compare equal
        # to itself and the content-derived identity would degrade silently to "always allocate a
        # new slot" -- duplicating the archive with no warning. Non-finite draws are an expected
        # condition here: `summarize_trace` documents dropping all-non-finite variables.
        return all(
            np.array_equal(np.asarray(pa[v].values), np.asarray(pb[v].values), equal_nan=True)
            for v in va
        )
    except Exception:  # noqa: BLE001 - "cannot tell" must read as "not the same"
        return False


def _allocate_trace_index(path: Path, trace) -> int:
    """Return an index that names a trace file without destroying an existing one.

    ``Trace_Index`` was a UTC timestamp truncated to whole seconds, used as an *identity*: it
    named the NetCDF, named the provenance sidecar, and linked the CSV rows to both. A clock
    reading is the wrong thing to identify data with, for two independent reasons, and the code
    hit both:

    * **It is not unique.** Two calls inside the same second produced the same index, and the
      second silently overwrote the first's ``.nc`` while both sets of summary rows survived in
      the CSV. Storing several fits in a loop therefore lost traces and left the CSV claiming
      they existed. Adding sub-second precision only narrows that window; it does not close it.
    * **It is not derived from the content.** Storing the *same* trace twice produced two
      indices, two files and two sidecars, so the archive could not say whether two entries were
      two fits or one fit written twice.

    The fix is to make the index an *allocated identifier* rather than a reading: seeded from the
    clock so the archive stays sortable and older files keep their meaning, then resolved against
    what is already on disk.

    * If **any already-archived trace carries the same posterior draws**, return *its* index.
      Storing one trace twice is then idempotent, whatever the clock did in between.
    * Otherwise take the clock-seeded slot, stepping forward while it is occupied. The index stops
      being a timestamp at that point, which is why ``Date_Time`` exists and is the column to read
      for *when*.

    Sameness is decided on the posterior draws (see :func:`_same_posterior`), not on the file:
    NetCDF embeds creation metadata, so two writes of one identical trace are never
    byte-identical, and comparing summaries would be weaker still because distinct posteriors can
    share a mean and a standard deviation.

    .. warning::
       The search is over **all** sibling slots by design. An earlier version checked only the
       slot the clock named and returned early when it was free, so two stores one second apart
       produced a duplicate — the precise outcome this function exists to prevent. It passed
       whenever both calls fell inside one second, which made its own idempotence test a coin
       flip on timing. Content-derived identity cannot be conditional on *when* the second write
       happens.
    """
    index = int(datetime.now(tz=UTC).timestamp())

    def slot(i: int) -> Path:
        return path.with_name(f"{path.stem}_trace_{i}.nc")

    if trace is None:
        # `save_trace=False`: there is no content to match against, so the identity cannot be
        # content-derived -- but it still must not COLLIDE. Returning the raw clock reading here
        # was a hole in exactly the guarantee this function exists to provide: a second call in
        # the same second would stamp its summary rows with an index already naming a DIFFERENT
        # trace's `.nc`, and `load_results(load_trace=True)` would then hand back one fit's
        # posterior beside another fit's summary numbers, silently. Step past occupied slots.
        while slot(index).exists():
            index += 1
        return index

    # Look for this trace among EVERY archived slot, not just the one the clock happens to name.
    #
    # This is the correction to the first version, and the failure is worth keeping because the
    # test caught it only by luck. That version checked `slot(index)` alone and returned early if
    # it was free. Since `index` is seeded from the clock, two stores of one trace one second
    # apart got two different indices, the second slot was empty, and it archived a duplicate --
    # the exact behaviour this function exists to prevent. It passed whenever both calls landed
    # inside the same second and failed when they straddled a boundary, so the idempotence test
    # was a coin flip on wall-clock timing rather than a check on identity.
    #
    # Content-derived identity cannot be conditional on when the second write happens. Scanning
    # the siblings is what makes it unconditional.
    #
    # Sameness is decided on the DRAWS, not on the file. The version before that compared blake2b
    # digests of the two .nc files -- which can never match, because NetCDF embeds creation
    # metadata, so two writes of one identical trace differ byte-for-byte. Comparing summaries
    # would be weaker still: distinct posteriors can share a mean and a standard deviation.
    try:
        import arviz as az

        # Cheap discriminators first. Deserialising every archived trace and comparing full
        # posterior arrays on every store is O(N^2) in netCDF reads over a loop of N fits -- and
        # `store_trace_results`' own docstring names "storing several fits in a loop" as the
        # motivating case. Variable names and per-variable shapes are readable from the netCDF
        # header without materialising any values, and a mismatch in either rules the slot out.
        want_vars = set(trace.posterior.data_vars)
        want_shapes = {v: tuple(trace.posterior[v].shape) for v in want_vars}
        for existing_path in sorted(path.parent.glob(f"{path.stem}_trace_*.nc")):
            try:
                candidate = az.from_netcdf(existing_path)
                if set(candidate.posterior.data_vars) != want_vars:
                    continue
                if {v: tuple(candidate.posterior[v].shape) for v in want_vars} != want_shapes:
                    continue
                if _same_posterior(candidate, trace):
                    # Recover the index this trace already owns, so the CSV rows point at it.
                    return int(existing_path.stem.rsplit("_trace_", 1)[1])
            except Exception:  # noqa: BLE001 - one unreadable neighbour must not abort the scan
                continue
    except Exception:  # noqa: BLE001 - no arviz, or an unreadable directory: fall through
        pass

    while slot(index).exists():
        index += 1
    return index


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

    Parameters
    ----------
    trace : arviz.InferenceData
        The posterior to summarise and, optionally, archive.
    file_path : str or path-like
        CSV to append to. Created if absent; if present, the existing rows are
        read and the new ones concatenated, so the file accumulates across runs
        rather than being overwritten.
    excluded_parameters : sequence of str, optional
        Forwarded to :func:`summarize_trace`.
    save_trace : bool, default True
        Also write ``<stem>_trace_<index>.nc`` and the JSON provenance sidecar
        beside the CSV. Turning this off keeps the summary rows but leaves the
        result unreproducible -- the numbers survive, the record of what made
        them does not.
    metadata : dict, optional
        Extra fields for the sidecar. ``trace_index`` and ``variables`` are
        added automatically; anything here is merged on top of them.

    Returns
    -------
    pandas.DataFrame
        The **full** table written to `file_path`, i.e. previously stored rows
        plus the new ones -- not just this call's rows. Columns: ``Parameter``,
        ``Mean``, ``Median``, ``Standard_Deviation``, ``Std`` (a duplicate of
        the previous column, kept for older readers), ``Mode``, ``Date_Time``
        and ``Trace_Index``. The statistics are plain floats with no units, for
        the reason given in :func:`summarize_trace`.

    Notes
    -----
    ``Trace_Index`` names the NetCDF and the sidecar, and is an *allocated*
    identifier rather than a clock reading: see :func:`_allocate_trace_index`.
    It is seeded from the UTC Unix second so the archive stays sortable, then
    resolved against what is already on disk -- an identical trace reuses its
    slot, and a different one steps forward past occupied slots. Storing several
    fits in a loop is therefore safe; ``Date_Time`` is the column that records
    *when*, and one call stamps one value across all of its rows.
    """
    path = Path(file_path)
    summaries = summarize_trace(trace, excluded_parameters=excluded_parameters)
    trace_index = _allocate_trace_index(path, trace if save_trace else None)
    # ONE stamp for the whole call, not one per row. `datetime.now().isoformat()` carries
    # microseconds, so evaluating it inside the comprehension below gave every parameter of a
    # single fit a DIFFERENT `Date_Time` (measured: 6 distinct values in a 6-row loop). Since
    # `load_results(only_last=True)` keeps the rows sharing the maximum `Date_Time`, a four-
    # parameter King fit was reduced to whichever parameter happened to be serialised last, with
    # the other three dropped silently -- on the path the published figures are regenerated from.
    # `Date_Time` identifies the storing EVENT; it is not a per-row measurement.
    stored_at = datetime.now(tz=UTC).isoformat()
    rows = [
        {
            "Parameter": item.variable,
            "Mean": item.mean,
            "Median": item.median,
            "Standard_Deviation": item.std,
            "Std": item.std,
            "Mode": item.mode,
            "Date_Time": stored_at,
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
        target = path.with_name(f"{path.stem}_trace_{trace_index}.nc")
        sidecar = path.with_name(f"{path.stem}_provenance_{trace_index}.json")
        reused = target.exists()  # _allocate_trace_index returned an existing, identical trace
        if not reused:
            trace.to_netcdf(target)
        # The sidecar needs the SAME protection as the NetCDF, and did not have it. On the
        # idempotent path the trace was correctly left alone while `write_metadata` overwrote its
        # provenance record unconditionally -- so re-storing one trace preserved the data and
        # destroyed the account of what produced it (`created_at`, the input checksums, the seeds,
        # any caller-supplied `metadata`). That is the one thing this module exists to keep.
        if not (reused and sidecar.exists()):
            write_metadata(
                sidecar,
                trace_index=trace_index,
                variables=[item.variable for item in summaries],
                **(metadata or {}),
            )
    return updated


def load_results(file_path="fit_parameters.csv", *, load_trace=False, only_last=True):
    """Load summarized CSV results and optionally the associated NetCDF trace.

    Parameters
    ----------
    file_path : str or path-like, default ``"fit_parameters.csv"``
        The summary CSV written by :func:`store_trace_results`.
    load_trace : bool, default False
        Also read the NetCDF trace stored alongside the CSV. Requires ArviZ and
        **requires** ``only_last=True``; see the warning below.
    only_last : bool, default True
        Keep only the rows sharing the most recent ``Date_Time``, i.e. the last
        fit stored. This is also what locates the trace: the ``Trace_Index`` of
        the surviving rows is what names the ``.nc`` file.

    Returns
    -------
    results : pandas.DataFrame
        The stored summary rows, filtered to the last fit when `only_last`.
    trace : arviz.InferenceData or None
        ``None`` unless `load_trace` is set.

    Raises
    ------
    FileNotFoundError
        If the CSV is absent, or if `load_trace` is set and the trace cannot be
        located -- including the ``only_last=False`` case described below.
    ImportError
        If `load_trace` is set and ArviZ is not installed.

    Notes
    -----
    The trace is located from the ``Trace_Index`` of the surviving rows, on both
    branches. ``load_trace=True`` therefore works with ``only_last=False`` too,
    provided those rows reference exactly ONE trace -- which holds whenever the
    CSV contains a single fit. When they reference several, this raises and says
    how many it found, because there is no rule for choosing among them.

    (Until 2026-08-04 ``trace_path`` was assigned only inside the ``only_last``
    branch, so ``load_trace=True, only_last=False`` raised unconditionally with
    ``... : None`` however many valid traces sat beside the CSV.)
    """
    az = _require_arviz() if load_trace else None
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"No summarized results file found: {path}")
    results = pd.read_csv(path)
    # Filter first. This is the ROW concern, and it is deliberately independent of the trace
    # concern below -- see the comment there for why separating them was the fix.
    if only_last and "Date_Time" in results.columns and not results.empty:
        results = results[results["Date_Time"] == results["Date_Time"].max()]

    # Locating the trace and filtering the rows are two separate concerns, and conflating them
    # is what made `load_trace=True, only_last=False` raise unconditionally: `trace_path` was
    # assigned only inside the `only_last` branch, so the other branch always reached the
    # existence check with `None` and reported "No trace file found ... : None" no matter how
    # many valid traces sat next to the CSV.
    #
    # The real precondition is not `only_last`. It is that the surviving rows refer to exactly
    # ONE trace, which `only_last=True` happens to guarantee and `only_last=False` satisfies
    # whenever the CSV holds a single fit. Resolving it from the rows makes that explicit, makes
    # the working case work, and turns the ambiguous case into an error that says what is
    # ambiguous.
    trace = None
    if load_trace:
        # `az` is already bound above by `_require_arviz()`, which runs whenever `load_trace` is
        # truthy and raises with its own message. A second `try: import arviz` here could never
        # reach its `except`, so it was dead code carrying a competing error string for one
        # condition -- two messages to keep in sync, and no way to tell which users would see.
        if "Trace_Index" not in results.columns:
            raise FileNotFoundError(
                f"{path} has no Trace_Index column, so no trace can be located. It was probably "
                "written by a version predating store_trace_results."
            )
        indices = sorted(set(results["Trace_Index"].dropna().tolist()))
        if not indices:
            raise FileNotFoundError(f"No Trace_Index values in {path}; nothing to load.")
        if len(indices) > 1:
            raise FileNotFoundError(
                f"{len(indices)} distinct traces are referenced by these rows "
                f"({indices[:5]}{'...' if len(indices) > 5 else ''}), so 'the' trace is "
                "ambiguous. Pass only_last=True for the most recent fit, or filter the frame "
                "to one Trace_Index and call again."
            )
        trace_path = path.with_name(f"{path.stem}_trace_{int(indices[0])}.nc")
        if not trace_path.exists():
            raise FileNotFoundError(
                f"Rows reference Trace_Index {int(indices[0])} but {trace_path} does not exist. "
                "The summary row outlived its trace -- see store_trace_results on why that used "
                "to happen silently."
            )
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

    Parameters
    ----------
    path : str or path-like
        File to digest. Must exist and be readable.
    algorithm : str, default ``"blake2b"``
        Any name :func:`hashlib.file_digest` accepts. Note this also **names the
        key** in the returned dict, so changing it changes the output schema.

    Returns
    -------
    dict
        ``{"path": str, "bytes": int, <algorithm>: <hex digest>}``. The third
        key is the *value of* `algorithm`, not the literal string
        ``"algorithm"`` -- with the default it is ``"blake2b"``. Callers reading
        the digest generically should take the one key that is neither ``path``
        nor ``bytes`` rather than hard-coding a hash name.

    Raises
    ------
    OSError
        If the file cannot be opened or stat-ed. :func:`build_metadata` catches
        this and records the failure instead of propagating it, unless it was
        called with ``strict=True``.
    ValueError
        If `algorithm` is not a hash :mod:`hashlib` supports ("unsupported hash
        type"). Note :func:`build_metadata` does **not** catch this one, so a
        mistyped algorithm aborts the whole provenance record rather than
        degrading it.
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

    Returns
    -------
    dict
        Distribution name to version string, restricted to
        ``_TRACKED_DISTRIBUTIONS`` and to those actually installed. A package
        that is not installed is **omitted rather than recorded as null**, which
        is itself the record: no ``pymc`` key means the run produced no
        Bayesian numbers. Insertion order follows ``_TRACKED_DISTRIBUTIONS``.

        The versions are distribution versions from installed metadata, so they
        can differ from a package's ``__version__`` attribute if the two were
        allowed to drift (``scikit-learn`` is keyed by its distribution name,
        not by ``sklearn``).
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
    >>> import json
    >>> isinstance(json.dumps(meta), str)
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
        "created_at": datetime.now(tz=UTC).isoformat(),
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
    """Write :func:`build_metadata` to `path` as indented JSON.

    Parameters
    ----------
    path : str or path-like
        Destination file, **overwritten** if it exists. Parent directories are
        not created.
    **kwargs
        Passed straight through to :func:`build_metadata` -- ``inputs``,
        ``seeds``, ``strict``, and any extra fields worth recording.

    Returns
    -------
    Path
        The file written, so the call can be chained or logged.

    Notes
    -----
    :func:`build_metadata` calls :func:`json.dumps` on the record before
    returning it, so an unserialisable field raises **before** `path` is
    touched. A failed call therefore leaves no truncated sidecar behind.
    """
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
