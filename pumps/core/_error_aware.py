"""Error-aware pseudo-probability for open-cluster membership.

Monte-Carlo over Gaia astrometric errors, layered on the ``min_cluster_size``
sweep. For each of ``n_mc`` draws every source is perturbed by its measurement
error, the sweep is re-run, and cluster membership is recorded. The error-aware
recovery frequency :math:`f_i` is the mean clustered fraction over draws x
resolutions; its spread across draws measures how much measurement error alone
moves membership.

This extends the UPMASK resampling idea (Krone-Martins & Moitinho 2014;
pyUPMASK, Pera et al. 2021) in two ways:

1. it samples the **full per-source covariance**, including Gaia's proper-motion
   / parallax correlations, instead of perturbing each observable independently;
2. it integrates the error draws with PUMPS's **multi-resolution sweep**, so
   :math:`f_i` reflects stability against both measurement error and clustering
   resolution.

The frequency it returns is frequentist (a stability frequency with a spread),
not yet a Bayesian posterior -- see the membership guide for the posterior layer.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np
from tqdm.auto import tqdm

from ._estimator import HDBSCANEstimator


def _cholesky_factors(cov: np.ndarray) -> np.ndarray:
    """Lower-Cholesky factor of each (d, d) covariance in a stacked (n, d, d) array.

    A tiny variance-scaled jitter is added to the diagonal so that covariances
    that are only numerically non-positive-definite (e.g. ``|corr| = 1``) still
    factorize.
    """
    cov = np.asarray(cov, dtype=float)
    d = cov.shape[-1]
    diag_mean = np.einsum("...ii->...i", cov).mean(axis=-1)
    jitter = 1e-10 * np.maximum(1.0, diag_mean)
    cov = cov + jitter[..., None, None] * np.eye(d)
    return np.linalg.cholesky(cov)


def _lookup_corr(
    table,
    a: str,
    b: str,
    corr_columns: Mapping[tuple[str, str], str] | None,
) -> np.ndarray:
    """Correlation coefficient array for the (a, b) axis pair, or zeros if absent.

    Honours an explicit ``corr_columns`` mapping first, then the Gaia convention
    ``"{a}_{b}_corr"`` / ``"{b}_{a}_corr"``.
    """
    n = len(table)
    if corr_columns:
        name = corr_columns.get((a, b)) or corr_columns.get((b, a))
        if name is not None:
            return np.asarray(table[name], dtype=float)
    for name in (f"{a}_{b}_corr", f"{b}_{a}_corr"):
        if name in table.colnames:
            return np.asarray(table[name], dtype=float)
    return np.zeros(n, dtype=float)


def gaia_covariance(
    table,
    columns: Sequence[str],
    error_columns: Sequence[str] | None = None,
    corr_columns: Mapping[tuple[str, str], str] | None = None,
) -> np.ndarray:
    """Assemble a per-source covariance tensor from Gaia error + correlation columns.

    Parameters
    ----------
    table : astropy.table.Table
        Source table carrying the value, error, and (optionally) correlation columns.
    columns : sequence of str
        The clustering feature columns, e.g. ``("pmra", "pmdec", "parallax")``.
    error_columns : sequence of str, optional
        Error column for each feature. Defaults to the Gaia ``"{col}_error"``.
    corr_columns : mapping, optional
        ``{(col_i, col_j): "corr_column_name"}`` overrides. Missing pairs fall back
        to the Gaia ``"{a}_{b}_corr"`` convention, then to zero correlation.

    Returns
    -------
    numpy.ndarray
        Covariance tensor of shape ``(n_sources, d, d)`` with ``d = len(columns)``.
        Missing (NaN) errors are treated as zero variance (that source is left
        unperturbed on that axis).
    """
    cols = list(columns)
    d = len(cols)
    n = len(table)
    if error_columns is None:
        error_columns = [f"{c}_error" for c in cols]
    if len(error_columns) != d:
        raise ValueError("error_columns must match columns in length.")

    sig = np.stack([np.asarray(table[e], dtype=float) for e in error_columns], axis=1)
    sig = np.nan_to_num(sig, nan=0.0)  # missing error -> unperturbed on that axis

    cov = np.zeros((n, d, d), dtype=float)
    for i in range(d):
        cov[:, i, i] = sig[:, i] ** 2
    for i in range(d):
        for j in range(i + 1, d):
            rho = np.nan_to_num(_lookup_corr(table, cols[i], cols[j], corr_columns))
            cov[:, i, j] = cov[:, j, i] = rho * sig[:, i] * sig[:, j]
    return cov


def error_aware_pseudoprobability(
    X: np.ndarray,
    *,
    cov: np.ndarray | None = None,
    errors: np.ndarray | None = None,
    min_cluster_size_samples: Iterable[int] = range(10, 300, 10),
    min_samples: int | None = None,
    n_mc: int = 100,
    random_state: int = 0,
    hdbscan_kwargs: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Error-aware recovery frequency :math:`f_i` and its Monte-Carlo spread.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix, shape ``(n_sources, d)`` (the same columns you cluster on).
    cov : numpy.ndarray, optional
        Per-source covariance, shape ``(n_sources, d, d)``. Use for correlated
        Gaia astrometry (see :func:`gaia_covariance`).
    errors : numpy.ndarray, optional
        Per-source, per-axis 1-sigma errors, shape ``(n_sources, d)``, used when
        the covariance is diagonal. Exactly one of ``cov`` / ``errors`` is required.
    min_cluster_size_samples : iterable of int, optional
        The ``min_cluster_size`` sweep run within every Monte-Carlo draw. Defaults
        to a coarse ``range(10, 300, 10)`` to keep the ``n_mc x sweep`` cost sane.
    min_samples : int, optional
        HDBSCAN ``min_samples`` (decoupled from ``min_cluster_size``).
    n_mc : int, optional
        Number of Monte-Carlo error draws. Default 100.
    random_state : int, optional
        Seed for the perturbation RNG (reproducible).
    hdbscan_kwargs : dict, optional
        Extra HDBSCAN keyword overrides.

    Returns
    -------
    f_mean : numpy.ndarray
        Error-aware recovery frequency per source, shape ``(n_sources,)``.
    f_std : numpy.ndarray
        Standard deviation of the per-draw frequency across Monte-Carlo draws --
        the error-induced uncertainty on membership.

    Notes
    -----
    Like the base ``pFreq``, "clustered" means assigned to *any* cluster
    (``labels != -1``), not a specific target cluster; in a field with several
    comoving groups, pair this frequency with the target-cluster selection.

    Cost is ``n_mc x len(min_cluster_size_samples)`` HDBSCAN fits (the default
    ``100 x 29 ~ 2900``). On a large Gaia field that is minutes to tens of minutes;
    lower ``n_mc`` or coarsen the sweep to trade precision for speed.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D (n_sources, d).")
    n, d = X.shape
    samples = [int(s) for s in min_cluster_size_samples]
    if not samples:
        raise ValueError("min_cluster_size_samples is empty.")
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1.")

    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (n, d, d):
            raise ValueError(f"cov must have shape {(n, d, d)}, got {cov.shape}.")
        L = _cholesky_factors(cov)
    elif errors is not None:
        errors = np.asarray(errors, dtype=float)
        if errors.shape != (n, d):
            raise ValueError(f"errors must have shape {(n, d)}, got {errors.shape}.")
        L = errors[:, :, None] * np.eye(d)  # diagonal Cholesky factor
    else:
        raise ValueError("Provide either `cov` (n, d, d) or `errors` (n, d).")

    kw = {
        "algorithm": "best",
        "cluster_selection_method": "eom",
        "allow_single_cluster": False,
        "metric": "euclidean",
        "match_reference_implementation": True,
        "gen_min_span_tree": False,
        **(hdbscan_kwargs or {}),
    }

    rng = np.random.default_rng(random_state)
    f_draws = np.empty((n_mc, n), dtype=float)
    for draw in tqdm(range(n_mc), desc="error-MC draws", unit="draw"):
        z = rng.standard_normal((n, d))
        Xp = X + np.einsum("nij,nj->ni", L, z)
        clustered = np.zeros(n, dtype=float)
        for mcs in samples:
            labels = (
                HDBSCANEstimator(min_cluster_size=mcs, min_samples=min_samples, **kw)
                .fit(Xp)
                .model_.labels_
            )
            clustered += np.asarray(labels) != -1
        f_draws[draw] = clustered / len(samples)

    return f_draws.mean(axis=0), f_draws.std(axis=0)
