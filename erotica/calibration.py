"""Membership-probability calibration diagnostics and recalibration.

This module evaluates whether EROTICA's per-star membership statistic

    p̃_i = f_i · p_HDBSCAN,i

behaves as a genuine probability, i.e. whether a source assigned p̃ ≈ 0.8 is a
true cluster member roughly 80% of the time. Here ``p_HDBSCAN`` is HDBSCAN's
soft-cluster membership (``model.probabilities_``) and ``f_i`` is the fraction
of ``min_cluster_size`` sweep iterations in which the source landed in *any*
cluster (``probability_times``). Both live on the clustering table produced by
:meth:`erotica.core.clustering.Clustering.search_pseudoprobability` as the
columns ``probability_hdbscan``, ``probability_times`` and ``probability``
(the last being p̃ itself).

Given p̃ together with ground-truth membership labels — obtained from synthetic
field+cluster mixtures where truth is known — this module computes:

* a **reliability diagram** (binned predicted-vs-observed frequency),
* the **Hosmer–Lemeshow** goodness-of-fit statistic,
* the **Brier score**, and
* the **expected / maximum calibration error** (ECE / MCE),

and provides **isotonic** and **Platt** recalibration that return a fitted,
callable mapping ``p̃ ↦ p̃_calibrated``.

Everything here is pure ``numpy`` / ``scipy`` / ``scikit-learn`` (no PyMC).
``scipy`` and ``scikit-learn`` are core EROTICA dependencies and are imported at
module scope, matching :mod:`erotica.core._estimator` and
:mod:`erotica.analysis.structure`. ``matplotlib`` is imported lazily inside the
plotting method, matching :meth:`Clustering.plot_mcs_sweep`.

References
----------
Brier (1950); Hosmer & Lemeshow (1980, 2000); Platt (1999);
Zadrozny & Elkan (2002); Niculescu-Mizil & Caruana (2005); Guo et al. (2017).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy.stats import chi2
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

__all__ = [
    "ReliabilityDiagram",
    "HosmerLemeshowResult",
    "CalibrationReport",
    "Recalibrator",
    "reliability_diagram",
    "brier_score",
    "expected_calibration_error",
    "hosmer_lemeshow",
    "calibration_report",
    "fit_isotonic",
    "fit_platt",
    "recalibrate",
    "calibration_report_from_clustering",
]

_EPS = 1e-6
_TINY = 1e-12
_PROB_TOL = 1e-6  # tolerance for the [0, 1] range check


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _as_probabilities(
    values: Sequence[float] | np.ndarray, *, name: str = "probabilities"
) -> np.ndarray:
    """Coerce to a finite 1-D float array constrained to [0, 1].

    Accepts plain sequences, numpy arrays, astropy ``Column`` objects, or a
    dimensionless astropy ``Quantity`` (``np.asarray`` yields its magnitude).
    """
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf).")
    if arr.min() < -_PROB_TOL or arr.max() > 1.0 + _PROB_TOL:
        raise ValueError(
            f"{name} must lie in [0, 1]; got range [{arr.min():.4g}, {arr.max():.4g}]."
        )
    return np.clip(arr, 0.0, 1.0)


def _as_labels(values: Sequence[int] | np.ndarray) -> np.ndarray:
    """Coerce to a finite 1-D float array of binary {0, 1} labels."""
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError("labels is empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("labels contains non-finite values (NaN/inf).")
    unique = np.unique(arr)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError(f"labels must be binary in {{0, 1}}; got unique values {unique.tolist()}.")
    return arr


def _validate_pair(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated, equal-length ``(probabilities, labels)`` float arrays."""
    p = _as_probabilities(probabilities)
    y = _as_labels(labels)
    if p.shape != y.shape:
        raise ValueError(f"probabilities and labels length mismatch: {p.shape} vs {y.shape}.")
    return p, y


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p) - np.log1p(-p)


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------
def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    """Return monotone bin edges for ``strategy`` in {'uniform', 'quantile'}."""
    if n_bins < 1:
        raise ValueError("n_bins must be >= 1.")
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
        edges[0], edges[-1] = 0.0, 1.0
        # Collapse duplicate edges (heavy ties, e.g. a point mass at p=0).
        return np.unique(edges)
    raise ValueError("strategy must be 'uniform' or 'quantile'.")


def _bin_indices(p: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, int]:
    """Assign each probability to a bin ``[0, n_bins)`` given ``edges``."""
    n_bins = len(edges) - 1
    if n_bins < 1:  # fully degenerate (all identical values)
        return np.zeros(p.shape, dtype=int), 1
    idx = np.digitize(p, edges[1:-1], right=False)
    return np.clip(idx, 0, n_bins - 1), n_bins


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReliabilityDiagram:
    """Binned predicted-vs-observed membership frequencies.

    Empty bins carry ``nan`` in ``mean_predicted`` / ``observed_frequency``.
    """

    bin_edges: np.ndarray
    mean_predicted: np.ndarray
    observed_frequency: np.ndarray
    counts: np.ndarray
    strategy: str
    n_bins: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def plot(self, ax=None, *, show_counts: bool = False, save_path: str | None = None):
        """Draw the reliability curve against the diagonal of perfect calibration."""
        import matplotlib.pyplot as plt  # lazy, matching Clustering.plot_mcs_sweep

        if ax is None:
            _, ax = plt.subplots(figsize=(5, 5))
        nonempty = self.counts > 0
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", label="Perfect calibration")
        ax.plot(
            self.mean_predicted[nonempty],
            self.observed_frequency[nonempty],
            marker="o",
            color="steelblue",
            label="EROTICA p̃",
        )
        if show_counts:
            for xc, yc, nc in zip(
                self.mean_predicted[nonempty],
                self.observed_frequency[nonempty],
                self.counts[nonempty],
            ):
                ax.annotate(
                    str(int(nc)), (xc, yc), textcoords="offset points", xytext=(4, -8), fontsize=7
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("Mean predicted membership probability")
        ax.set_ylabel("Observed cluster-member fraction")
        ax.legend(loc="upper left")
        if save_path:
            ax.figure.savefig(save_path, bbox_inches="tight")
        return ax


@dataclass(frozen=True)
class HosmerLemeshowResult:
    """Hosmer–Lemeshow goodness-of-fit statistic and its chi-squared p-value."""

    statistic: float
    pvalue: float
    dof: int
    n_groups: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CalibrationReport:
    """Bundle of calibration diagnostics for one ``(p̃, labels)`` pair."""

    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    hosmer_lemeshow: HosmerLemeshowResult
    reliability: ReliabilityDiagram
    n_samples: int
    n_positive: int
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def reliability_diagram(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> ReliabilityDiagram:
    """Bin predicted probabilities and measure the observed positive fraction.

    Parameters
    ----------
    probabilities, labels
        Per-star p̃ in [0, 1] and binary ground-truth membership labels.
    n_bins
        Number of bins.
    strategy
        ``"uniform"`` (equal-width, the Guo et al. 2017 convention) or
        ``"quantile"`` (equal-count; duplicate edges from ties are collapsed,
        which can reduce the effective bin count — check ``.n_bins``).
    """
    p, y = _validate_pair(probabilities, labels)
    edges = _bin_edges(p, n_bins, strategy)
    idx, n_eff = _bin_indices(p, edges)

    counts = np.bincount(idx, minlength=n_eff).astype(int)
    sum_pred = np.bincount(idx, weights=p, minlength=n_eff)
    sum_obs = np.bincount(idx, weights=y, minlength=n_eff)
    safe = np.maximum(counts, 1)
    mean_pred = np.where(counts > 0, sum_pred / safe, np.nan)
    obs_freq = np.where(counts > 0, sum_obs / safe, np.nan)

    return ReliabilityDiagram(
        bin_edges=edges,
        mean_predicted=mean_pred,
        observed_frequency=obs_freq,
        counts=counts,
        strategy=strategy,
        n_bins=n_eff,
        metadata={"n_samples": int(p.size), "requested_n_bins": int(n_bins)},
    )


def brier_score(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> float:
    """Mean squared error between p̃ and the binary label (Brier 1950).

    Equivalent to ``sklearn.metrics.brier_score_loss`` for binary labels; lower
    is better and 0 is perfect. Computed directly for transparency.
    """
    p, y = _validate_pair(probabilities, labels)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
) -> tuple[float, float]:
    """Return ``(ECE, MCE)`` — the count-weighted mean and max reliability gap.

    ECE = Σ_m (n_m / N) · |obs_m − pred_m|; MCE = max_m |obs_m − pred_m|,
    over non-empty bins (Guo et al. 2017). Uses equal-width bins by default.
    """
    diagram = reliability_diagram(probabilities, labels, n_bins=n_bins, strategy=strategy)
    nonempty = diagram.counts > 0
    if not np.any(nonempty):
        return 0.0, 0.0
    gaps = np.abs(diagram.observed_frequency[nonempty] - diagram.mean_predicted[nonempty])
    weights = diagram.counts[nonempty] / diagram.counts.sum()
    return float(np.sum(weights * gaps)), float(np.max(gaps))


def hosmer_lemeshow(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_groups: int = 10,
) -> HosmerLemeshowResult:
    r"""Hosmer–Lemeshow :math:`\hat{H}` statistic with equal-count grouping.

    Sources are sorted by p̃ and split into ``n_groups`` contiguous groups of
    near-equal size (equal-count / "deciles of risk"). For each group *k*

    .. math::
        \hat{H} = \sum_k \frac{(O_k - E_k)^2}{E_k\,(1 - E_k / n_k)},

    with :math:`O_k` observed positives and :math:`E_k = \sum_{i \in k} p_i`
    expected positives. Under good calibration :math:`\hat{H} \sim \chi^2_{g-2}`
    and the p-value is :math:`P(\chi^2_{g-2} > \hat{H})`. Large :math:`\hat{H}`
    (small p) rejects calibration.

    Caveats
    -------
    * ``dof = n_groups - 2`` is the convention Hosmer & Lemeshow calibrated for
      probabilities *fit* to the same sample by logistic regression. EROTICA's p̃
      is computed externally (not fit to these labels), so ``g-2`` is mildly
      anti-conservative here; ``n_groups - 1`` is a more conservative reference.
      The choice is documented rather than hidden — ``n_groups`` is exposed.
    * Groups whose ``E_k (1 - E_k/n_k)`` collapses to ~0 (e.g. a point mass of
      exact-zero p̃ from field stars) are skipped to avoid division blow-up;
      ``dof`` is left at ``n_groups - 2`` (slightly conservative).
    """
    p, y = _validate_pair(probabilities, labels)
    n = p.size
    if n_groups < 3:
        raise ValueError("n_groups must be >= 3 so that dof = n_groups - 2 >= 1.")
    if n_groups > n:
        raise ValueError(f"n_groups ({n_groups}) cannot exceed the sample size ({n}).")

    order = np.argsort(p, kind="mergesort")  # stable → deterministic tie handling
    statistic = 0.0
    for group in np.array_split(order, n_groups):
        n_k = group.size
        if n_k == 0:
            continue
        e_k = float(p[group].sum())
        o_k = float(y[group].sum())
        denom = e_k * (1.0 - e_k / n_k)
        if denom <= _TINY:  # degenerate group (all p̃≈0 or all p̃≈1) — skip
            continue
        statistic += (o_k - e_k) ** 2 / denom

    dof = n_groups - 2
    pvalue = float(chi2.sf(statistic, dof))
    return HosmerLemeshowResult(
        statistic=float(statistic),
        pvalue=pvalue,
        dof=int(dof),
        n_groups=int(n_groups),
        metadata={"grouping": "equal_count", "n_samples": int(n)},
    )


def calibration_report(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    n_bins: int = 10,
    strategy: str = "uniform",
    hl_groups: int = 10,
) -> CalibrationReport:
    """Compute all diagnostics for one ``(p̃, labels)`` pair in one call."""
    p, y = _validate_pair(probabilities, labels)
    diagram = reliability_diagram(p, y, n_bins=n_bins, strategy=strategy)
    nonempty = diagram.counts > 0
    if np.any(nonempty):
        gaps = np.abs(diagram.observed_frequency[nonempty] - diagram.mean_predicted[nonempty])
        weights = diagram.counts[nonempty] / diagram.counts.sum()
        ece, mce = float(np.sum(weights * gaps)), float(np.max(gaps))
    else:  # pragma: no cover - unreachable for non-empty input
        ece, mce = 0.0, 0.0

    return CalibrationReport(
        brier_score=float(np.mean((p - y) ** 2)),
        expected_calibration_error=ece,
        maximum_calibration_error=mce,
        hosmer_lemeshow=hosmer_lemeshow(p, y, n_groups=hl_groups),
        reliability=diagram,
        n_samples=int(p.size),
        n_positive=int(y.sum()),
    )


# ---------------------------------------------------------------------------
# Recalibration
# ---------------------------------------------------------------------------
class Recalibrator:
    """A fitted probability→probability recalibration map.

    Callable: ``recalibrated = recal(p̃)``. Output is clipped to [0, 1].
    Construct via :func:`fit_isotonic`, :func:`fit_platt`, or :func:`recalibrate`.
    """

    def __init__(self, method: str, model: Any, *, metadata: dict[str, Any] | None = None):
        self.method = method
        self.model = model
        self.metadata = metadata or {}

    def __call__(self, probabilities: Sequence[float] | np.ndarray) -> np.ndarray:
        p = _as_probabilities(probabilities)
        if self.method == "isotonic":
            out = self.model.predict(p)
        elif self.method == "platt":
            z = _logit(p).reshape(-1, 1)
            out = self.model.predict_proba(z)[:, 1]
        else:  # pragma: no cover - guarded at construction
            raise ValueError(f"unknown recalibration method {self.method!r}.")
        return np.clip(np.asarray(out, dtype=float), 0.0, 1.0)

    # sklearn-style alias
    predict = __call__

    def __repr__(self) -> str:
        return f"Recalibrator(method={self.method!r})"


def fit_isotonic(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> Recalibrator:
    """Fit non-parametric isotonic recalibration (Zadrozny & Elkan 2002).

    Learns a monotone step function mapping p̃ to the empirical positive rate.
    Flexible but data-hungry and prone to overfit on small samples; fit and
    evaluate on independent draws for an honest estimate.
    """
    p, y = _validate_pair(probabilities, labels)
    if np.unique(y).size < 2:
        raise ValueError("isotonic recalibration needs both classes present in the labels.")
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip", increasing=True)
    model.fit(p, y)
    return Recalibrator("isotonic", model, metadata={"n_samples": int(p.size)})


def fit_platt(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
) -> Recalibrator:
    """Fit parametric Platt (sigmoid) recalibration.

    Fits ``P = σ(A·logit(p̃) + B)`` by (essentially unregularised) logistic
    regression. Fitting on ``logit(p̃)`` rather than the raw score means an
    already-calibrated input is representable as the identity (A=1, B=0).

    This differs from Platt (1999) / sklearn's internal ``_SigmoidCalibration``
    in two ways: those fit on the classifier's decision-function scores, and
    they add a target-smoothing prior (t⁺ = (N⁺+1)/(N⁺+2)) to curb overfit. We
    use the plain MLE (``C=1e10``); for the large synthetic samples used to
    calibrate EROTICA this is negligible, but note it for small-sample use.
    """
    p, y = _validate_pair(probabilities, labels)
    if np.unique(y).size < 2:
        raise ValueError("Platt recalibration needs both classes present in the labels.")
    z = _logit(p).reshape(-1, 1)
    model = LogisticRegression(C=1e10, solver="lbfgs", max_iter=10_000)
    model.fit(z, y)
    return Recalibrator("platt", model, metadata={"n_samples": int(p.size)})


def recalibrate(
    probabilities: Sequence[float] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    *,
    method: str = "isotonic",
) -> Recalibrator:
    """Fit a recalibration map; ``method`` in {'isotonic', 'platt'}."""
    if method == "isotonic":
        return fit_isotonic(probabilities, labels)
    if method == "platt":
        return fit_platt(probabilities, labels)
    raise ValueError("method must be 'isotonic' or 'platt'.")


# ---------------------------------------------------------------------------
# Integration convenience
# ---------------------------------------------------------------------------
def calibration_report_from_clustering(
    clustering: Any,
    labels: Sequence[int] | np.ndarray,
    *,
    probability_column: str = "probability",
    **kwargs: Any,
) -> CalibrationReport:
    """Build a :class:`CalibrationReport` straight from a ``Clustering`` object.

    Pulls the p̃ column (default ``"probability"`` — i.e. f_i · p_HDBSCAN as set
    by :meth:`Clustering.search_pseudoprobability`) from ``clustering.data`` and
    pairs it with externally supplied ground-truth membership ``labels``. Duck
    typed: any object exposing a ``.data`` table with the column works, so this
    module never imports :mod:`erotica.core.clustering`.

    ``labels[i]`` must align positionally to row ``i`` of ``clustering.data``;
    the column is read in table order and zipped by position.
    """
    table = getattr(clustering, "data", None)
    if table is None or probability_column not in getattr(table, "colnames", []):
        raise ValueError(
            f"clustering.data must expose a {probability_column!r} column; "
            "run Clustering.search_pseudoprobability first."
        )
    return calibration_report(np.asarray(table[probability_column], dtype=float), labels, **kwargs)

