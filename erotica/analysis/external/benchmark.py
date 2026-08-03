"""Head-to-head membership benchmark harness: EROTICA vs ASteCA vs pyUPMASK.

This is the P02 *comparability* deliverable. **ASteCA and pyUPMASK are the reference
implementations in this field, and the baselines EROTICA has to be measured against.**
The harness compares the three pipelines along four axes:

    (a) membership AGREEMENT  -- pairwise overlap / confusion / rank AUC / kappa
    (b) parameter RECOVERY    -- fundamental parameters vs. known synthetic truth
    (c) RUNTIME               -- wall-clock per method
    (d) CALIBRATION           -- reliability curve / ECE / MCE / Brier

.. important::
   **A ranking is a legitimate output of this harness -- but only from a run.** Axes (b)
   and (d) have ground truth, so "better" is well defined there, and establishing it is
   what this file is for. What is not legitimate is asserting a win the harness has not
   produced: **no benchmark run is committed anywhere in this repository**, so any
   superiority claim in the paper is a claim about capability, not a measured result, and
   has to say so.

   An earlier version of this docstring instructed that the methods be framed as
   "complementary (not to declare a winner)" and that the framing be kept "symmetric in
   any paper text built on this harness". That was the wrong instruction: it made an
   unmeasured tie the default conclusion, which is no better supported than an unmeasured
   win. The two honest positions are "measured, and here is the number" and "not yet
   measured".

Axis (d) is the one the others do not typically *report*, though they can compute it:
ASteCA and pyUPMASK emit membership probabilities, and pyUPMASK's own Sect. 3.2 already
scores its output with six probabilistic-classification metrics including Brier, log-loss
and the H-measure (Pera et al. 2021, A&A 650, A109). So the open question is not whether
such scores exist, but whether the probabilities are **calibrated** -- reliability
conditional on magnitude, crowding and radius. Concede the first; measure the second.

Design
------
The *metrics* (``compute_calibration``, ``compute_agreement``,
``compute_parameter_recovery``, ``build_report``) operate purely on co-indexed
probability arrays plus optional ground truth. They pull in only numpy / scipy /
scikit-learn (all core EROTICA dependencies), so the harness is fully usable --
and unit-tested -- without any external clustering package installed.

The *runner* hooks (``run_erotica``, ``run_asteca_membership``,
``run_pyupmask_membership``) actually execute each pipeline. They import their
heavy / optional dependency **lazily**, matching ``erotica.analysis.external.
asteca``. The ASteCA and pyUPMASK membership call surfaces are BEST-EFFORT and
must be verified against the installed versions -- see each runner's docstring.

Important caveats (documentation, not enforced by the metrics)
--------------------------------------------------------------
1. SOURCE ALIGNMENT. Every probability vector passed to the metrics must be
   co-indexed: the i-th entry of each array must refer to the *same source*.
   Real pipelines drop / reorder rows (quality cuts, NaN photometry). Cross-match
   first (see ``data/test/NGC6383/comments_paper/review_repo/cone_crossmatch.py``)
   and pass ``ids=`` to ``build_report`` to inner-join on a shared identifier.
2. PARAMETER-NAME NORMALIZATION. ASteCA reports ``loga`` / ``dm`` / ``Av`` /
   ``met`` (as Z); EROTICA inference reports distance in pc, linear age, etc.
   ``compute_parameter_recovery`` validates the error *arithmetic* on whatever
   common schema you hand it -- the caller is responsible for mapping each tool's
   native names onto that schema first.
3. ``erotica.calibration`` DOES NOT EXIST YET. The calibration metrics live here
   inline. They are self-contained and should later be extracted into a
   dedicated ``erotica/calibration.py`` and imported from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Lazy optional-dependency guards (mirror erotica.analysis.external.asteca)
# ---------------------------------------------------------------------------
def _require_asteca():
    try:
        import asteca
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "ASteCA is required for this runner. Install the 'asteca' extra."
        ) from exc
    return asteca


def _require_pyupmask():
    """Import pyUPMASK lazily.

    pyUPMASK (Pera et al. 2021) is distributed as a repository of scripts, not as
    a PyPI package with a stable import surface. There is no guaranteed
    ``import pyupmask`` entry point. Prefer running pyUPMASK externally and
    feeding its output file to ``run_pyupmask_membership(results_path=...)``.
    """
    try:
        import pyUPMASK as pyupmask  # noqa: N813 - upstream capitalisation
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyUPMASK is not importable. It ships as scripts, not a pip package; "
            "run it externally and pass its output to run_pyupmask_membership()."
        ) from exc
    return pyupmask


# ---------------------------------------------------------------------------
# Report dataclasses (returned, never printed)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalibrationCurve:
    """Binned reliability curve. Tuples so the dataclass stays hashable/eq-safe."""

    bin_edges: tuple[float, ...]
    bin_centers: tuple[float, ...]
    mean_predicted: tuple[float, ...]      # NaN in empty bins
    empirical_frequency: tuple[float, ...]  # NaN in empty bins
    counts: tuple[int, ...]


@dataclass(frozen=True)
class CalibrationMetrics:
    n: int
    positive_fraction: float
    brier: float
    ece: float
    mce: float
    curve: CalibrationCurve


@dataclass(frozen=True)
class AgreementMetrics:
    """Pairwise agreement. Confusion uses method A as reference (rows), B as pred.

    tp = both member, tn = both non-member, fp = A non-member & B member,
    fn = A member & B non-member. AUC(a_scores_b) ranks B's member set with A's
    probabilities; single-class references yield NaN rather than raising.
    """

    method_a: str
    method_b: str
    n: int
    threshold: float
    n_members_a: int
    n_members_b: int
    jaccard: float
    overlap_coefficient: float
    cohen_kappa: float
    spearman_r: float
    pearson_r: float
    auc_a_scores_b: float
    auc_b_scores_a: float
    tp: int
    fp: int
    fn: int
    tn: int


@dataclass(frozen=True)
class ParameterRecovery:
    method: str
    param: str
    truth: float
    recovered: float
    abs_error: float
    rel_error: float   # abs_error / |truth|; NaN if truth == 0
    z_score: float     # abs_error / uncertainty; NaN if no uncertainty given


@dataclass(eq=False)  # holds numpy arrays -> element-wise __eq__ would break
class MethodResult:
    name: str
    probabilities: np.ndarray
    member_mask: np.ndarray | None = None
    runtime_s: float | None = None
    calibration: CalibrationMetrics | None = None
    recovery: tuple[ParameterRecovery, ...] = ()


@dataclass(eq=False)
class BenchmarkReport:
    dataset: str
    n_sources: int
    truth_available: bool
    methods: tuple[MethodResult, ...]
    agreements: tuple[AgreementMetrics, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    # -- convenience projections (pandas is a core EROTICA dep; import lazily) --
    def calibration_table(self):
        import pandas as pd

        rows = [
            {
                "method": m.name,
                "n": m.calibration.n,
                "positive_fraction": m.calibration.positive_fraction,
                "brier": m.calibration.brier,
                "ece": m.calibration.ece,
                "mce": m.calibration.mce,
            }
            for m in self.methods
            if m.calibration is not None
        ]
        return pd.DataFrame(rows)

    def agreement_table(self):
        import pandas as pd

        return pd.DataFrame(
            {
                "method_a": a.method_a,
                "method_b": a.method_b,
                "jaccard": a.jaccard,
                "overlap_coefficient": a.overlap_coefficient,
                "cohen_kappa": a.cohen_kappa,
                "spearman_r": a.spearman_r,
                "pearson_r": a.pearson_r,
                "auc_a_scores_b": a.auc_a_scores_b,
                "auc_b_scores_a": a.auc_b_scores_a,
                "tp": a.tp,
                "fp": a.fp,
                "fn": a.fn,
                "tn": a.tn,
            }
            for a in self.agreements
        ) if self.agreements else pd.DataFrame()

    def recovery_table(self):
        import pandas as pd

        rows = [
            {
                "method": r.method,
                "param": r.param,
                "truth": r.truth,
                "recovered": r.recovered,
                "abs_error": r.abs_error,
                "rel_error": r.rel_error,
                "z_score": r.z_score,
            }
            for m in self.methods
            for r in m.recovery
        ]
        return pd.DataFrame(rows)

    def runtime_table(self):
        import pandas as pd

        return pd.DataFrame(
            [{"method": m.name, "runtime_s": m.runtime_s} for m in self.methods]
        )


# ---------------------------------------------------------------------------
# Small numeric helpers (guarded so degenerate inputs return NaN, never raise)
# ---------------------------------------------------------------------------
def _as_member_mask(x: np.ndarray, threshold: float) -> np.ndarray:
    """Boolean membership from probabilities (>= threshold) or an existing mask."""
    x = np.asarray(x)
    if x.dtype == bool:
        return x
    return np.asarray(x, dtype=float) >= threshold


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC of ``scores`` against binary ``labels``; NaN if single-class."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    finite = np.isfinite(scores)
    labels, scores = labels[finite], scores[finite]
    if labels.size == 0 or labels.min() == labels.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, scores))


def _safe_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).astype(int)
    b = np.asarray(b).astype(int)
    # Cohen's kappa is undefined when either rater is constant AND they agree
    # perfectly on the single class; sklearn returns NaN/0 unstably -> guard.
    if a.size == 0:
        return float("nan")
    if a.min() == a.max() and b.min() == b.max():
        return 1.0 if a[0] == b[0] else 0.0
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(a, b))


def _safe_corr(a: np.ndarray, b: np.ndarray, kind: str) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    a, b = a[finite], b[finite]
    if a.size < 2 or np.all(a == a[0]) or np.all(b == b[0]):
        return float("nan")
    if kind == "spearman":
        from scipy.stats import spearmanr

        return float(spearmanr(a, b).statistic)
    from scipy.stats import pearsonr

    return float(pearsonr(a, b).statistic)


# ---------------------------------------------------------------------------
# (d) Calibration -- reliability curve, ECE, MCE, Brier
# ---------------------------------------------------------------------------
def compute_calibration(
    probabilities: np.ndarray,
    truth: np.ndarray,
    *,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Reliability of ``probabilities`` against binary ``truth`` membership.

    Uniform bins on [0, 1]. ECE = sum_b (n_b/N) |mean_pred_b - emp_freq_b|;
    MCE = max_b |...|; Brier = mean((p - y)^2). Empty bins carry NaN in the curve
    and contribute nothing to ECE/MCE.
    """
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(truth, dtype=float)
    finite = np.isfinite(p) & np.isfinite(y)
    p, y = p[finite], y[finite]
    n = int(p.size)
    if n == 0:
        edges = tuple(np.linspace(0.0, 1.0, n_bins + 1).tolist())
        empty = tuple([float("nan")] * n_bins)
        curve = CalibrationCurve(edges, empty, empty, empty, tuple([0] * n_bins))
        return CalibrationMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), curve)

    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)

    centers, mean_pred, emp_freq, counts = [], [], [], []
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        sel = idx == b
        c = int(sel.sum())
        counts.append(c)
        centers.append(float((edges[b] + edges[b + 1]) / 2.0))
        if c:
            mp = float(np.mean(p[sel]))
            ef = float(np.mean(y[sel]))
            gap = abs(mp - ef)
            ece += (c / n) * gap
            mce = max(mce, gap)
        else:
            mp = float("nan")
            ef = float("nan")
        mean_pred.append(mp)
        emp_freq.append(ef)

    curve = CalibrationCurve(
        bin_edges=tuple(edges.tolist()),
        bin_centers=tuple(centers),
        mean_predicted=tuple(mean_pred),
        empirical_frequency=tuple(emp_freq),
        counts=tuple(counts),
    )
    return CalibrationMetrics(
        n=n,
        positive_fraction=float(np.mean(y)),
        brier=brier,
        ece=float(ece),
        mce=float(mce),
        curve=curve,
    )


# ---------------------------------------------------------------------------
# (a) Membership agreement -- overlap / confusion / rank AUC / kappa / corr
# ---------------------------------------------------------------------------
def compute_agreement(
    method_a: str,
    probs_a: np.ndarray,
    method_b: str,
    probs_b: np.ndarray,
    *,
    threshold: float = 0.5,
) -> AgreementMetrics:
    """Pairwise agreement between two co-indexed membership-probability vectors."""
    pa = np.asarray(probs_a, dtype=float)
    pb = np.asarray(probs_b, dtype=float)
    if pa.shape != pb.shape:
        raise ValueError(
            f"Probability vectors must be co-indexed and equal length: "
            f"{pa.shape} vs {pb.shape}. Cross-match sources first."
        )
    ma = _as_member_mask(pa, threshold)
    mb = _as_member_mask(pb, threshold)

    inter = int(np.sum(ma & mb))
    union = int(np.sum(ma | mb))
    na = int(ma.sum())
    nb = int(mb.sum())
    jaccard = float(inter / union) if union else float("nan")
    smaller = min(na, nb)
    overlap_coeff = float(inter / smaller) if smaller else float("nan")

    tp = inter
    tn = int(np.sum(~ma & ~mb))
    fp = int(np.sum(~ma & mb))
    fn = int(np.sum(ma & ~mb))

    return AgreementMetrics(
        method_a=method_a,
        method_b=method_b,
        n=int(pa.size),
        threshold=threshold,
        n_members_a=na,
        n_members_b=nb,
        jaccard=jaccard,
        overlap_coefficient=overlap_coeff,
        cohen_kappa=_safe_kappa(ma, mb),
        spearman_r=_safe_corr(pa, pb, "spearman"),
        pearson_r=_safe_corr(pa, pb, "pearson"),
        auc_a_scores_b=_safe_auc(mb, pa),
        auc_b_scores_a=_safe_auc(ma, pb),
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


# ---------------------------------------------------------------------------
# (b) Fundamental-parameter recovery vs. known truth
# ---------------------------------------------------------------------------
def compute_parameter_recovery(
    method: str,
    recovered: Mapping[str, float],
    truth: Mapping[str, float],
    *,
    uncertainties: Mapping[str, float] | None = None,
) -> tuple[ParameterRecovery, ...]:
    """Per-parameter recovery error over the shared keys of ``recovered``/``truth``.

    NOTE: this validates the error *arithmetic* on a common schema. The caller
    must normalise each tool's native parameter names (ASteCA ``loga`` / ``dm``,
    EROTICA distance-in-pc, ...) onto that schema before calling.
    """
    uncertainties = uncertainties or {}
    out: list[ParameterRecovery] = []
    for key in truth:
        if key not in recovered:
            continue
        t = float(truth[key])
        r = float(recovered[key])
        abs_err = r - t
        rel_err = abs_err / abs(t) if t != 0 else float("nan")
        unc = uncertainties.get(key)
        z = abs_err / unc if unc not in (None, 0) else float("nan")
        out.append(
            ParameterRecovery(
                method=method,
                param=key,
                truth=t,
                recovered=r,
                abs_error=abs_err,
                rel_error=rel_err,
                z_score=float(z),
            )
        )
    return tuple(out)


# ---------------------------------------------------------------------------
# Orchestrator -- assemble a full report from co-indexed inputs
# ---------------------------------------------------------------------------
def build_report(
    dataset: str,
    method_probs: Mapping[str, np.ndarray],
    *,
    truth_membership: np.ndarray | None = None,
    member_masks: Mapping[str, np.ndarray] | None = None,
    truth_params: Mapping[str, float] | None = None,
    recovered_params: Mapping[str, Mapping[str, float]] | None = None,
    param_uncertainties: Mapping[str, Mapping[str, float]] | None = None,
    runtimes: Mapping[str, float] | None = None,
    ids: Sequence | None = None,
    threshold: float = 0.5,
    n_bins: int = 10,
    notes: Sequence[str] = (),
) -> BenchmarkReport:
    """Build a :class:`BenchmarkReport` from co-indexed membership probabilities.

    Parameters
    ----------
    method_probs : mapping name -> probability array (all co-indexed, same length)
    truth_membership : optional binary/boolean array; enables the calibration axis
    member_masks : optional per-method boolean masks; else derived at ``threshold``
    recovered_params / truth_params : enable the parameter-recovery axis
    runtimes : optional per-method wall-clock seconds (from the runner hooks)
    ids : optional identifiers; only validated for length here (see module note 1
          on cross-matching -- this harness does not itself join disparate rows)
    """
    names = list(method_probs)
    if not names:
        raise ValueError("method_probs is empty; nothing to benchmark.")
    lengths = {name: np.asarray(p).shape[0] for name, p in method_probs.items()}
    n = next(iter(lengths.values()))
    if len(set(lengths.values())) != 1:
        raise ValueError(f"All probability vectors must be co-indexed; got {lengths}.")
    if ids is not None and len(ids) != n:
        raise ValueError(f"ids length {len(ids)} != n_sources {n}.")
    if truth_membership is not None and np.asarray(truth_membership).shape[0] != n:
        raise ValueError("truth_membership length must match the probability vectors.")

    truth_available = truth_membership is not None
    member_masks = member_masks or {}
    recovered_params = recovered_params or {}
    param_uncertainties = param_uncertainties or {}
    runtimes = runtimes or {}

    methods: list[MethodResult] = []
    for name in names:
        probs = np.asarray(method_probs[name], dtype=float)
        mask = (
            _as_member_mask(member_masks[name], threshold)
            if name in member_masks
            else _as_member_mask(probs, threshold)
        )
        calib = (
            compute_calibration(probs, truth_membership, n_bins=n_bins)
            if truth_available
            else None
        )
        recov: tuple[ParameterRecovery, ...] = ()
        if truth_params is not None and name in recovered_params:
            recov = compute_parameter_recovery(
                name,
                recovered_params[name],
                truth_params,
                uncertainties=param_uncertainties.get(name),
            )
        methods.append(
            MethodResult(
                name=name,
                probabilities=probs,
                member_mask=mask,
                runtime_s=runtimes.get(name),
                calibration=calib,
                recovery=recov,
            )
        )

    agreements: list[AgreementMetrics] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            agreements.append(
                compute_agreement(
                    names[i],
                    method_probs[names[i]],
                    names[j],
                    method_probs[names[j]],
                    threshold=threshold,
                )
            )

    return BenchmarkReport(
        dataset=dataset,
        n_sources=int(n),
        truth_available=truth_available,
        methods=tuple(methods),
        agreements=tuple(agreements),
        notes=tuple(notes),
    )


# ---------------------------------------------------------------------------
# (c) Runner hooks -- lazy imports; produce (probabilities, mask, runtime_s)
# ---------------------------------------------------------------------------
def run_erotica(
    data,
    *,
    columns: Sequence[str] = ("pmra", "pmdec"),
    min_cluster_size_samples: Sequence[int] | range = range(10, 80),
    probability_threshold: float = 0.5,
    bad_data=None,
    **search_kwargs,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run EROTICA's pseudo-probability pipeline and return (p_tilde, mask, seconds).

    ``p_tilde`` is EROTICA's ``probability`` column = probability_hdbscan *
    probability_times (see ``Clustering.search_pseudoprobability``). The default
    mcs sweep in EROTICA is ``range(10, 300)`` and is slow; ``min_cluster_size_samples``
    is exposed here so smoke tests can use a tiny range.

    MUST be verified end-to-end in the real env: this wires
    ``erotica.Clustering`` but has not been executed against live Gaia data here.
    """
    from erotica import Clustering  # lazy: keep module import cheap

    clustering = Clustering(data, bad_data=bad_data)
    t0 = perf_counter()
    clustering.search_pseudoprobability(
        columns=tuple(columns),
        min_cluster_size_samples=min_cluster_size_samples,
        probability_threshold=probability_threshold,
        **search_kwargs,
    )
    runtime = perf_counter() - t0
    table = clustering.data
    probs = np.asarray(table["probability"], dtype=float)
    mask = np.asarray(table["cluster"], dtype=int) != -1
    return probs, mask, runtime


def run_asteca_membership(
    *,
    cluster_kwargs: Mapping,
    membership_fn: Callable | None = None,
) -> tuple[np.ndarray, float]:
    """Run ASteCA membership and return (probabilities, seconds).

    ASteCA 0.6+ builds an ``asteca.Cluster`` (see
    ``data/test/NGC6383/comments_paper/review_repo/run_asteca069.py`` for the
    Cluster/Isochrones/Synthetic surface actually used in this repo). The
    *membership* entry point (fastMP / Membership) is **not** exercised anywhere
    in this codebase and its exact signature is UNVERIFIED here.

    To stay honest, this runner does not guess a call chain. Pass ``membership_fn``
    -- a callable ``membership_fn(asteca_module, cluster) -> probabilities`` --
    wiring the version you actually have installed, e.g.::

        def memb(ac, clu):
            m = ac.Membership(clu)         # verify against your ASteCA version
            return m.fastmp()              # returns per-source probabilities

        probs, secs = run_asteca_membership(cluster_kwargs=..., membership_fn=memb)
    """
    asteca = _require_asteca()
    t0 = perf_counter()
    cluster = asteca.Cluster(**cluster_kwargs)
    if membership_fn is None:
        raise NotImplementedError(
            "ASteCA membership signature is version-specific and unverified in "
            "this repo. Pass membership_fn(asteca_module, cluster) -> probabilities."
        )
    probs = np.asarray(membership_fn(asteca, cluster), dtype=float)
    return probs, perf_counter() - t0


def run_pyupmask_membership(
    *,
    results_path: str | None = None,
    prob_column: str = "probability",
    membership_fn: Callable | None = None,
) -> tuple[np.ndarray, float]:
    """Return pyUPMASK membership (probabilities, seconds).

    Preferred path: run pyUPMASK externally (it ships as scripts, not a stable
    importable API) and pass its output CSV via ``results_path``; the column
    ``prob_column`` is read as the membership probability.

    Alternative: pass ``membership_fn() -> probabilities`` wiring whatever local
    pyUPMASK entry point you have. ``_require_pyupmask`` is invoked in that branch
    only, satisfying the lazy-import requirement without pretending a stable API
    exists.
    """
    t0 = perf_counter()
    if results_path is not None:
        import pandas as pd

        df = pd.read_csv(results_path)
        if prob_column not in df.columns:
            raise KeyError(
                f"Column {prob_column!r} not in pyUPMASK output {list(df.columns)}."
            )
        return np.asarray(df[prob_column], dtype=float), perf_counter() - t0
    if membership_fn is not None:
        _require_pyupmask()  # ensure the dependency is present before calling
        return np.asarray(membership_fn(), dtype=float), perf_counter() - t0
    raise ValueError(
        "Provide either results_path (a pyUPMASK output file) or membership_fn."
    )


__all__ = [
    "AgreementMetrics",
    "BenchmarkReport",
    "CalibrationCurve",
    "CalibrationMetrics",
    "MethodResult",
    "ParameterRecovery",
    "build_report",
    "compute_agreement",
    "compute_calibration",
    "compute_parameter_recovery",
    "run_asteca_membership",
    "run_erotica",
    "run_pyupmask_membership",
]

