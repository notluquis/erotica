"""Is ``tgt``'s advantage over ``p̃`` a property of the product, or of the SELECTOR?

WHY THIS EXISTS
---------------
``benchmark_ptilde_decomposition.py`` reported that the truth-free binary indicator

    tgt_i = 1 if labels_i == selected_cluster else 0

outranks EROTICA's shipped continuous score ``p̃_i = score_i * f_i`` by
**+0.1948 +- 0.0355 average precision** on held-out cells of the ``erotica_5d_soft`` arm.
That was read as a STRUCTURAL DEFECT in the product: ``score`` means "firmly inside my OWN
cluster" (a field star inside the field cluster scores ~1.0) and ``f_i`` means "clustered
into ANYTHING", so neither factor encodes *target-cluster identity* and the product can be
large for a star that was never a candidate member.

The confound: ``tgt`` is evaluated AFTER the sweep has already chosen a cluster, and the
sweep-step rule was itself chosen partly on this benchmark. So some unknown fraction of
``tgt``'s advantage could be measuring **how good the selector is at picking the right
cluster on this generator**, not a property of ``p̃``'s construction. That worry is not
hypothetical: ``selection="max_persistence"`` was adopted on 54 cells and then failed to
replicate on a held-out block, the paired difference against ``"max_members"`` landing
within one standard error of zero on every metric
(``erotica/core/clustering.py``, ``_select_pseudoprobability_result`` docstring).

THE PREMISE OF THE ORIGINAL COMPARISON WAS WRONG, AND THAT IS THE FIRST RESULT
------------------------------------------------------------------------------
``git log -L`` on ``EROTICA_CONFIGS`` shows ``erotica_5d`` has ALWAYS been
``("max_members", "hdbscan")`` and ``erotica_5d_soft`` ALWAYS ``("max_persistence",
"soft")``. So the published pair

    erotica_5d       delta_AP(tgt - p̃) = +0.1752 +- 0.0421   <- max_members  + hdbscan score
    erotica_5d_soft  delta_AP(tgt - p̃) = +0.1948 +- 0.0355   <- max_persistence + soft score

is **already** a two-selector comparison -- but a confounded one, because selector and
score type move together. This script separates them with the full 3 x 2 factorial:

    selection      in {max_members, max_persistence, max_lambda}
    score          in {hdbscan (probabilities_), soft (all_points_membership_vectors)}

and reports the PAIRED per-cell difference-of-deltas across selectors at fixed score type,
which is the statistic the pre-registered branch actually turns on. Two deltas with
independent standard errors cannot be eyeballed as "similar"; the cells are shared, so the
pairing is available and is what is used.

PRE-REGISTERED INTERPRETATION (fixed before the numbers were seen)
------------------------------------------------------------------
* ``tgt - p̃`` SURVIVES at a similar size under the other selectors -> the advantage is not
  selector-specific and the structural-defect claim is clean.
* It SHRINKS substantially -> ``tgt``'s advantage is partly selector overfit, +0.195 is
  inflated, and the honest headline is whatever survives.
* It INVERTS or VANISHES -> the defect claim as measured does not hold and must be
  withdrawn to the structural argument alone, which stands on the definitions of ``score``
  and ``f_i`` independently of any measurement.

WHAT WOULD FALSIFY THE CONCLUSION -- AND THE CHECK THAT MAKES IT MEANINGFUL
---------------------------------------------------------------------------
**A null here means nothing if the selectors pick the same cluster.** If ``max_members``
and ``max_persistence`` agree in most cells, the comparison has no power and "the delta
survives" is a statement about arithmetic, not about selectors. The disagreement rate is
therefore measured FIRST and reported next to every delta: the fraction of cells where the
selected ``min_cluster_size`` differs, the fraction where the selected member SET differs
(the one with power), and the Jaccard overlap of those sets.

``max_lambda`` is included for exactly this reason, and it is NOT a third co-equal vote.
The package docstring records that it chose ``mcs_range.start`` in 12 of 12 cells -- it
effectively ignores the sweep -- so the expectation before running was that it would
maximise disagreement and carry the test. **It did not**: see the measured Jaccard below.
It ends up a consistency check on ``max_persistence``, and ``max_members`` is the only
selector that genuinely contrasts. The expectation is left here as stated rather than
rewritten, because the gap between it and the measurement is itself the reason the
disagreement rate is reported before any delta.

The conclusion is falsified if the paired difference-of-deltas across selectors is large
and negative for the alternative selectors -- i.e. ``tgt`` only beats ``p̃`` when
``max_persistence`` picks the cluster.

HOW ONE RUN PER SELECTOR YIELDS BOTH SCORE TYPES (validated, not assumed)
-------------------------------------------------------------------------
``_annotate_pseudoprobability_results`` writes ``probability_hdbscan`` on EVERY path
(``clustering.py:704``) and additionally ``probability_soft`` on the soft path
(``clustering.py:730``). So a single ``probability_method="soft"`` run per selector carries
both scores, and both products are recoverable as ``score * probability_times``. This
halves the compute, and it is licensed by an EXTERNAL ORACLE rather than by reading the
source: every cell asserts

    max_members    + hdbscan score  ==  stored ``erotica_5d`` p̃
    max_persistence + soft score    ==  stored ``erotica_5d_soft`` p̃

against the 108-cell run in ``soft_bench.cells.jsonl``. Smoke-tested on 4 cells before the
full run: max abs error 0.0 on both, which also settles that ``prediction_data=True`` (set
only on the soft path) does not perturb ``labels_``.

THE SILENT FALLBACK THAT WOULD FAKE A NULL
------------------------------------------
``_soft_membership_column`` can return ``None`` (``clustering.py:729``). When it does,
``probability_soft`` is never written and ``probability`` stays ``hdbscan * f_i`` -- the
soft arm silently COLLAPSES onto the hdbscan arm and a difference between score types
disappears for reasons that have nothing to do with selectors. Measured on the smoke set:
it fires (seed 5001). Every cell therefore records ``has_soft`` explicitly and the counts
are reported; where it is False the soft columns are recorded as absent rather than as a
number equal to the hdbscan one.

FAILURES ARE PAIRED, AND THAT IS LOAD-BEARING
---------------------------------------------
``search_pseudoprobability`` raises when the sweep finds no candidate cluster at any step.
``results`` is built BEFORE any selection happens, so the failure set is identical for all
three selectors by construction -- asserted per cell. In a failed cell every quantity is
the all-zero vector, so ``roc = 0.5``, ``ap = base rate``, and every delta is exactly 0:
failures are ties that cancel in the paired statistics. They are kept in the primary
numbers so those stay comparable to the published table, and the count of cells that
actually CONTRIBUTE (search succeeded) is reported alongside every mean.

METRICS
-------
Average precision leads. Base rates are 0.50 / 0.20 / 0.05 at contamination 0.50 / 0.80 /
0.95, and ROC understates a binary score, which has a single operating point. Both are
reported; the verdict is read off AP. Every mean carries a standard error across cells and
a win/loss/TIE count -- never the mean alone, and never a win rate that hides how many of
its denominators are the forced ties above.

HELD-OUT SPLIT
--------------
Realisation index ``k = seed % 10``; ``k in {0,1,2}`` train, ``k in {3,4,5}`` HELD OUT.
Identical to ``benchmark_ptilde_decomposition.py``, so the numbers are comparable cell for
cell. **Every decision is made on the held-out block only**; the train block is reported
only so a sign flip between blocks is visible.

WHAT WAS MEASURED (108 cells, 2026-08-04, 1114 s; sidecar ``tgt_selector_robustness.json``)
--------------------------------------------------------------------------------------------
**The test has power, but only along one axis, and that has to be said before the numbers.**
Held-out contributing cells (37 of 54)::

    pair                              mcs differs   member SET differs   mean Jaccard
    max_members  vs max_persistence      75.7%            56.8%          0.840 +- 0.058
    max_members  vs max_lambda           70.3%            51.4%          0.830 +- 0.058
    max_persistence vs max_lambda        51.4%            29.7%          0.983 +- 0.012

All three rules agree on the selected set in only 35.1% of cells, so the comparison is not
vacuous. But ``max_lambda`` picked ``mcs_range.start`` in **37 of 37** cells (confirming the
package docstring's 12-of-12 at three times the sample) and nonetheless lands on nearly the
same member set as ``max_persistence`` -- Jaccard 0.983, disagreeing on the set in under a
third of cells. **``max_lambda`` is therefore a consistency check, not an independent third
selector**, and ``lambda - persistence ~ 0`` below is close to tautological rather than
confirmatory. An earlier draft of this docstring billed it as the strongest test because it
ignores the sweep; the measured Jaccard says otherwise. The one real contrast is
**members vs persistence**, and that is the pair the verdict rests on::

    held-out delta_AP (tgt - p̃), 54 cells, W/L/TIE      hdbscan score        soft score
    max_members                                    +0.1752 +- 0.0421   +0.1246 +- 0.0363
                                                        26/ 9/19            26/ 9/17
    max_persistence                                +0.2661 +- 0.0409   +0.1948 +- 0.0355
                                                        31/ 6/17            31/ 6/17
    max_lambda                                     +0.2642 +- 0.0430   +0.1885 +- 0.0359
                                                        31/ 6/17            31/ 6/17

Every one of the six is positive at >= 3.4 standard errors. The two entries that also exist
in the published table reproduce it exactly (+0.1752 for ``max_members`` + hdbscan,
+0.1948 for ``max_persistence`` + soft), which is the check that the factorial is measuring
the same quantity.

**The pre-registered branch is SURVIVES.** The delta holds under every selector, and the
honest headline is the RANGE, not any single cell: **+0.125 to +0.266 AP, and the published
+0.1948 sits inside it.** It is NOT a conservative number -- at fixed score type, the only
comparison this experiment licenses, +0.1948 is the LARGER of the two genuinely-disagreeing
selectors (``max_members`` + soft gives +0.1246). The +0.266 cell is larger only because it
crosses score types: ``tgt`` is identical there (AP 0.783 at fixed selector) and it is
``p̃_hdbscan`` that is the weaker baseline (AP 0.517 vs 0.588). Quoting +0.266 as "what
survives" would re-inflate the claim through the axis this experiment was built to control.

Paired difference-of-deltas, same cells, same score type (held-out, AP)::

    max_persistence - max_members      +0.0909 +- 0.0390 (hdbscan)  W/L/TIE 14/13/27
                                       +0.0573 +- 0.0293 (soft)             14/11/27
    max_lambda      - max_members      +0.0890 +- 0.0415                    15/11/28
                                       +0.0508 +- 0.0316                    15/ 7/30
    max_lambda      - max_persistence  -0.0019 +- 0.0110                    10/ 9/35
                                       -0.0062 +- 0.0098                    11/ 8/35

**The win/loss counts are the strongest statement here, and they are stronger than the
means.** Cell by cell, ``max_members`` gives the LARGER delta in 13 of 27 informative cells
against ``max_persistence`` -- a coin flip. The +0.0909 mean is carried by a minority of
large cells, not by a systematic selector dependence. There is no direction in which the
selector reliably moves the gap.

MECHANISM: p̃ CANNOT RESPOND TO A BETTER SELECTOR (held-out AP)
---------------------------------------------------------------
::

                       max_members   max_persistence   max_lambda
    tgt         AP        0.6896          0.7831          0.7751
    p̃ (hdbscan) AP        0.5144          0.5170          0.5109
    p̃ (soft)    AP        0.5896          0.5883          0.5866
    f_i alone   AP        0.5804          0.5804          0.5804   (selector-independent
                                                                    by construction)

Improving the selector buys **+0.094 AP on ``tgt`` and ~0.000 on ``p̃``, either score type**.
That is the structural claim stated as a mechanism rather than as a gap: ``p̃`` is
*structurally unable* to benefit from a better choice of cluster, because neither ``score``
(strength in whichever cluster a star landed in) nor ``f_i`` (clustered into anything) ever
references the selected cluster. The gap is therefore not a tuning artefact -- it is the
signature of a quantity that does not depend on the selection at all.

WHICH CELL IS THE SHIPPED PACKAGE
---------------------------------
``search_pseudoprobability`` defaults to ``selection="max_persistence"`` and
``probability_method="hdbscan"`` (``clustering.py:147,149``), so **what a package user
actually gets is +0.2661 +- 0.0409 -- a cell that appears in no published table.** The
published +0.1948 is ``max_persistence`` + soft, which is not the shipped default either
(soft is opt-in). Name the configuration whenever this number is quoted.

The one honest caveat is the train block, where ``max_members`` gives +0.0805 +- 0.0386
(hdbscan) against the held-out +0.1752 +- 0.0421. The *ordering* of the three selectors is
identical in both blocks and every delta is positive in both; it is the magnitude under the
weakest selector that moves. Quote the held-out numbers, and quote the range, not a point.

ENVIRONMENT
-----------
``erotica-bench`` (python 3.13, hdbscan 0.8.44, erotica -e). NOT ``cosmic``, which has no
``erotica`` installed. ``erotica/core/clustering.py`` has changed only in docstrings since
the reference run (``git diff bf194ce HEAD`` = 2 lines, both ``::`` literal-block markers),
which is what keeps the stored-p̃ reproduction check valid.

USAGE
-----
    python tools/validation/benchmark_tgt_selector_robustness.py \
        --out tgt_selector_robustness.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
from astropy.table import QTable
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

import benchmark_erotica_vs_asteca as B  # noqa: E402

# The 108-cell run this script reproduces, cell for cell. Written by
# `benchmark_erotica_vs_asteca.py --configs erotica_3d,erotica_3d_soft,erotica_5d,erotica_5d_soft`.
REFERENCE_CELLS = Path(
    "/private/tmp/claude-501/-Users-notluquis-phd/"
    "d055f28c-94b0-484a-8d8b-601af31f9812/scratchpad/soft_bench.cells.jsonl"
)

HELD_OUT_K = (3, 4, 5)
SELECTORS = ("max_members", "max_persistence", "max_lambda")
SCORES = ("hdbscan", "soft")
# The 5d feature set: the only one with feature parity against ASteCA, and the one the
# published decomposition table was computed on.
QUANTITIES = ("ra", "dec", "pmra", "pmdec", "plx")
# Which stored arm each (selector, score) pair must reproduce exactly. This is the external
# oracle that licenses reading two score types out of one run.
REFERENCE_ARM = {
    ("max_members", "hdbscan"): "erotica_5d",
    ("max_persistence", "soft"): "erotica_5d_soft",
}


def held_out(seed: int) -> bool:
    return (seed % 10) in HELD_OUT_K


def _score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


# ---------------------------------------------------------------------------
# One (cell, selector) run: every factor kept separately, both score types
# ---------------------------------------------------------------------------
def run_selector(real, *, selection: str, mcs_range: range) -> dict:
    """One ``search_pseudoprobability`` at ``selection``, with both score columns kept.

    Returns the raw per-star vectors; nothing is scored here. On a sweep failure every
    vector is all-zero -- exactly what the benchmark records -- so the arms stay comparable.
    """
    from erotica import Clustering
    from erotica.core.clustering import Clustering as _C

    n = real.truth.size
    zero = np.zeros(n)
    cols = {f"{q}_z": B._zscore(getattr(real, q)) for q in QUANTITIES}
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clu = Clustering(QTable(cols))
            clu.search_pseudoprobability(
                columns=tuple(cols),
                min_cluster_size_samples=mcs_range,
                probability_threshold=0.5,
                selection=selection,
                # Always "soft": it writes BOTH probability_hdbscan and probability_soft,
                # so one run carries both score types. Validated against the stored arms.
                probability_method="soft",
            )
    except Exception as exc:
        return {
            "selection": selection,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.perf_counter() - t0,
            "f_i": zero,
            "score": {"hdbscan": zero, "soft": zero},
            "p_tilde": {"hdbscan": zero, "soft": zero},
            "tgt": zero,
            "has_soft": False,
            "mcs": None,
            "selected_cluster": -1,
            "would": None,
        }
    secs = time.perf_counter() - t0
    tab = clu.data
    f_i = np.asarray(tab["probability_times"], dtype=float)
    score_h = np.asarray(tab["probability_hdbscan"], dtype=float)
    has_soft = "probability_soft" in tab.colnames
    score_s = np.asarray(tab["probability_soft"], dtype=float) if has_soft else None
    labels = np.asarray(tab["cluster_hdbscan"], dtype=int)
    sel_label = int((clu.pseudoprobability_selected_ or {}).get("selected_cluster", -1))
    tgt = (labels == sel_label).astype(float) if sel_label >= 0 else zero
    # What each of the three rules WOULD have picked, read off this one run's `results`
    # list through the package's own staticmethod -- not re-derived here.
    would = {
        s: int(
            _C._select_pseudoprobability_result(clu.pseudoprobability_results_, s)[
                "min_cluster_size"
            ]
        )
        for s in SELECTORS
    }
    mcs = int(clu.best_params_["min_cluster_size"])
    return {
        "selection": selection,
        "error": None,
        "runtime_s": secs,
        "f_i": f_i,
        "score": {"hdbscan": score_h, "soft": score_s if has_soft else None},
        # p_tilde = score * f_i. The identity is asserted below against `tab["probability"]`
        # for whichever score the package itself used, rather than assumed.
        "p_tilde": {
            "hdbscan": score_h * f_i,
            "soft": (score_s * f_i) if has_soft else None,
        },
        "package_probability": np.asarray(tab["probability"], dtype=float),
        "tgt": tgt,
        "has_soft": bool(has_soft),
        "mcs": mcs,
        "selected_cluster": sel_label,
        "n_selected": int((labels == sel_label).sum()) if sel_label >= 0 else 0,
        "would": would,
        "would_matches_actual": bool(would[selection] == mcs),
    }


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0
    b = b > 0
    u = int((a | b).sum())
    return float((a & b).sum() / u) if u else float("nan")


def run_cell(*, n_members, contamination, fractal_dimension, seed, mcs_range, reference) -> dict:
    real = B.generate(
        n_members=n_members,
        contamination=contamination,
        fractal_dimension=fractal_dimension,
        seed=seed,
    )
    y = real.truth.astype(int)
    out = {
        "n_members": n_members,
        "contamination": contamination,
        "fractal_dimension": fractal_dimension,
        "seed": seed,
        "held_out": held_out(seed),
        "n_sources": int(y.size),
        "base_rate": float(y.mean()),
        "generator_check": B.check_generator(real, requested_contamination=contamination),
        "sel": {},
        "checks": {},
    }
    runs = {s: run_selector(real, selection=s, mcs_range=mcs_range) for s in SELECTORS}
    out["search_failed"] = bool(runs[SELECTORS[0]]["error"] is not None)
    out["error"] = runs[SELECTORS[0]]["error"]

    for s, r in runs.items():
        blk = {
            "mcs": r["mcs"],
            "selected_cluster": r["selected_cluster"],
            "n_selected": r.get("n_selected", 0),
            "has_soft": r["has_soft"],
            "error": r["error"],
            "runtime_s": r["runtime_s"],
            "would_matches_actual": r.get("would_matches_actual"),
        }
        roc, ap = _score(y, r["f_i"])
        blk["roc_f_i"], blk["ap_f_i"] = roc, ap
        roc, ap = _score(y, r["tgt"])
        blk["roc_tgt"], blk["ap_tgt"] = roc, ap
        blk["mean_tgt"] = float(np.mean(r["tgt"]))
        for m in SCORES:
            sc, pt = r["score"][m], r["p_tilde"][m]
            if sc is None:
                # soft column unavailable: record the absence, do NOT silently reuse hdbscan
                blk[f"roc_score_{m}"] = blk[f"ap_score_{m}"] = None
                blk[f"roc_ptilde_{m}"] = blk[f"ap_ptilde_{m}"] = None
                continue
            blk[f"roc_score_{m}"], blk[f"ap_score_{m}"] = _score(y, sc)
            blk[f"roc_ptilde_{m}"], blk[f"ap_ptilde_{m}"] = _score(y, pt)
        out["sel"][s] = blk

    out["would"] = runs[SELECTORS[0]]["would"]

    # ---- checks that make this run load-bearing --------------------------
    ck = out["checks"]
    ok = [s for s in SELECTORS if runs[s]["error"] is None]
    # (a) all three selectors fail together, or none does -- the pairing depends on it
    ck["failure_is_paired"] = len({runs[s]["error"] is None for s in SELECTORS}) == 1
    # (b) f_i does not depend on the selection rule (the sweep runs before selection)
    ck["f_i_selector_independent"] = (
        bool(all(np.allclose(runs[ok[0]]["f_i"], runs[s]["f_i"]) for s in ok[1:]))
        if len(ok) > 1
        else None
    )
    # (c) the package's own `results` list, run through its own selector staticmethod,
    #     reproduces the min_cluster_size each run actually used
    ck["would_matches_actual"] = all(runs[s]["would_matches_actual"] for s in ok) if ok else None
    # (d) same selected mcs => byte-identical downstream. This is what licenses reading
    #     three selectors off one sweep, and it is asserted rather than assumed.
    same_mcs_identical = []
    for i, a in enumerate(ok):
        for b in ok[i + 1 :]:
            if runs[a]["mcs"] == runs[b]["mcs"]:
                same_mcs_identical.append(
                    bool(
                        np.array_equal(runs[a]["tgt"], runs[b]["tgt"])
                        and np.allclose(
                            runs[a]["p_tilde"]["hdbscan"], runs[b]["p_tilde"]["hdbscan"]
                        )
                    )
                )
    ck["same_mcs_gives_identical_vectors"] = all(same_mcs_identical) if same_mcs_identical else None
    # (e) product identity, against the value the package itself stored
    perr = []
    for s in ok:
        r = runs[s]
        used = r["p_tilde"]["soft"] if r["has_soft"] else r["p_tilde"]["hdbscan"]
        perr.append(float(np.max(np.abs(r["package_probability"] - used))))
    ck["product_identity_maxabs"] = max(perr) if perr else None
    # (f) EXTERNAL ORACLE: reproduce the two stored arms of the 108-cell benchmark
    ref = reference.get(seed)
    for (s, m), arm in REFERENCE_ARM.items():
        if ref is None or arm not in ref:
            ck[f"reproduces_{arm}"] = None
            continue
        stored = np.asarray(ref[arm], dtype=float)
        mine = runs[s]["p_tilde"][m]
        if mine is None:
            # soft column absent: the package fell back to hdbscan for `probability`, so
            # that is what the stored arm holds. Compare against the same fallback.
            mine = runs[s]["p_tilde"]["hdbscan"]
            ck[f"reproduces_{arm}_via_soft_fallback"] = True
        ck[f"reproduces_{arm}"] = bool(
            stored.size == mine.size and np.allclose(stored, mine, atol=1e-9)
        )
        ck[f"reproduces_{arm}_maxabs"] = (
            float(np.max(np.abs(stored - mine))) if stored.size == mine.size else None
        )

    # ---- selector disagreement: measured FIRST, because a null without it is empty ----
    dis = {}
    for i, a in enumerate(SELECTORS):
        for b in SELECTORS[i + 1 :]:
            key = f"{a}_vs_{b}"
            if runs[a]["error"] or runs[b]["error"]:
                dis[key] = {"mcs_same": None, "tgt_same": None, "jaccard": None}
                continue
            dis[key] = {
                "mcs_same": bool(runs[a]["mcs"] == runs[b]["mcs"]),
                "tgt_same": bool(np.array_equal(runs[a]["tgt"], runs[b]["tgt"])),
                "jaccard": _jaccard(runs[a]["tgt"], runs[b]["tgt"]),
            }
    dis["all_three_agree_mcs"] = (
        None if out["search_failed"] else bool(len({runs[s]["mcs"] for s in SELECTORS}) == 1)
    )
    dis["all_three_agree_set"] = (
        None
        if out["search_failed"]
        else bool(
            np.array_equal(runs["max_members"]["tgt"], runs["max_persistence"]["tgt"])
            and np.array_equal(runs["max_members"]["tgt"], runs["max_lambda"]["tgt"])
        )
    )
    out["disagreement"] = dis
    return out


# ---------------------------------------------------------------------------
def _load_reference(path: Path) -> dict:
    if not path.exists():
        print(f"[warn] reference cells not at {path}; oracle check DISABLED", file=sys.stderr)
        return {}
    ref = {}
    for line in path.read_text().splitlines():
        if line.strip():
            c = json.loads(line)
            ref[c["seed"]] = {
                k: v for k, v in c["_probs"].items() if k in ("erotica_5d", "erotica_5d_soft")
            }
    return ref


def _ms(vals) -> dict:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return {"mean": None, "sem": None, "n": 0}
    if v.size == 1:
        return {"mean": float(v[0]), "sem": None, "n": 1}
    return {
        "mean": float(v.mean()),
        "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
        "n": int(v.size),
    }


def _wlt(vals) -> dict:
    """Win / loss / TIE. Ties are not decoration here: a failed search forces delta == 0
    in every arm, so a bare win rate silently counts those forced ties as losses."""
    v = [x for x in vals if x is not None and np.isfinite(x)]
    w = sum(1 for x in v if x > 0)
    ln = sum(1 for x in v if x < 0)
    t = sum(1 for x in v if x == 0)
    return {
        "win": w,
        "loss": ln,
        "tie": t,
        "n": len(v),
        "win_rate_all": (w / len(v)) if v else None,
        "win_rate_untied": (w / (w + ln)) if (w + ln) else None,
    }


def summarise(cells: list[dict]) -> dict:
    out = {}
    for split, keep in (
        ("held_out", lambda c: c["held_out"]),
        ("train", lambda c: not c["held_out"]),
    ):
        sub = [c for c in cells if keep(c)]
        contrib = [c for c in sub if not c["search_failed"]]
        blk = {
            "n_cells": len(sub),
            "n_contributing": len(contrib),
            "n_search_failures": len(sub) - len(contrib),
            "failures_by_contamination": {
                str(x): sum(1 for c in sub if c["contamination"] == x and c["search_failed"])
                for x in sorted({c["contamination"] for c in sub})
            },
            "base_rate_by_contamination": {
                str(x): _ms([c["base_rate"] for c in sub if c["contamination"] == x])
                for x in sorted({c["contamination"] for c in sub})
            },
            "n_soft_column_missing": {
                s: sum(1 for c in contrib if not c["sel"][s]["has_soft"]) for s in SELECTORS
            },
        }

        # ---- selector disagreement, on contributing cells only ------------
        dis = {}
        for i, a in enumerate(SELECTORS):
            for b in SELECTORS[i + 1 :]:
                key = f"{a}_vs_{b}"
                rows = [c["disagreement"][key] for c in contrib]
                n = len(rows)
                dis[key] = {
                    "n": n,
                    "mcs_disagree_rate": (
                        sum(1 for r in rows if r["mcs_same"] is False) / n if n else None
                    ),
                    "set_disagree_rate": (
                        sum(1 for r in rows if r["tgt_same"] is False) / n if n else None
                    ),
                    "mean_jaccard": _ms([r["jaccard"] for r in rows]),
                }
        dis["all_three_agree_mcs_rate"] = (
            sum(1 for c in contrib if c["disagreement"]["all_three_agree_mcs"]) / len(contrib)
            if contrib
            else None
        )
        dis["all_three_agree_set_rate"] = (
            sum(1 for c in contrib if c["disagreement"]["all_three_agree_set"]) / len(contrib)
            if contrib
            else None
        )
        dis["max_lambda_picks_sweep_start_rate"] = (
            sum(1 for c in contrib if c["sel"]["max_lambda"]["mcs"] == 10) / len(contrib)
            if contrib
            else None
        )
        blk["selector_disagreement"] = dis

        # ---- per (selector, score): levels and the tgt - p_tilde delta ----
        per = {}
        for s in SELECTORS:
            per[s] = {
                "roc_f_i": _ms([c["sel"][s]["roc_f_i"] for c in sub]),
                "ap_f_i": _ms([c["sel"][s]["ap_f_i"] for c in sub]),
                "roc_tgt": _ms([c["sel"][s]["roc_tgt"] for c in sub]),
                "ap_tgt": _ms([c["sel"][s]["ap_tgt"] for c in sub]),
                "mean_selected_mcs": _ms(
                    [c["sel"][s]["mcs"] for c in sub if c["sel"][s]["mcs"] is not None]
                ),
            }
            for m in SCORES:
                # A cell where the soft column is missing contributes None and drops out of
                # the soft block; it is NOT filled with the hdbscan value.
                per[s][f"roc_score_{m}"] = _ms([c["sel"][s][f"roc_score_{m}"] for c in sub])
                per[s][f"ap_score_{m}"] = _ms([c["sel"][s][f"ap_score_{m}"] for c in sub])
                per[s][f"roc_ptilde_{m}"] = _ms([c["sel"][s][f"roc_ptilde_{m}"] for c in sub])
                per[s][f"ap_ptilde_{m}"] = _ms([c["sel"][s][f"ap_ptilde_{m}"] for c in sub])
                for metric in ("roc", "ap"):
                    d = [_delta(c, s, m, metric) for c in sub]
                    per[s][f"delta_{metric}_tgt_minus_ptilde_{m}"] = _ms(d)
                    per[s][f"delta_{metric}_tgt_minus_ptilde_{m}_wlt"] = _wlt(d)
                    dc = [_delta(c, s, m, metric) for c in contrib]
                    per[s][f"delta_{metric}_tgt_minus_ptilde_{m}_contributing"] = _ms(dc)
        blk["by_selector"] = per

        # ---- THE decisive statistic: paired difference-of-deltas ----------
        # Same cells, same score type, selector varied. Independent SEMs cannot answer
        # "similar size"; this can.
        pair = {}
        for i, a in enumerate(SELECTORS):
            for b in SELECTORS:
                if b == a:
                    continue
                if SELECTORS.index(b) < i:
                    continue
                for m in SCORES:
                    for metric in ("roc", "ap"):
                        key = f"{b}_minus_{a}__{metric}_{m}"
                        dd = [
                            (_delta(c, b, m, metric) - _delta(c, a, m, metric))
                            if (
                                _delta(c, b, m, metric) is not None
                                and _delta(c, a, m, metric) is not None
                            )
                            else None
                            for c in sub
                        ]
                        pair[key] = {"stats": _ms(dd), "wlt": _wlt(dd)}
        blk["paired_delta_difference_across_selectors"] = pair
        out[split] = blk
    return out


def _delta(cell: dict, selector: str, score: str, metric: str):
    """``tgt - p_tilde`` for one cell, or None when the soft column was unavailable."""
    b = cell["sel"][selector]
    pt = b[f"{metric}_ptilde_{score}"]
    tg = b[f"{metric}_tgt"]
    if pt is None or tg is None:
        return None
    return tg - pt


def _reduce_cell(c: dict) -> dict:
    """Per-cell record for the COMMITTED sidecar: scalars only, no per-star arrays.

    The full record (identical, plus nothing -- no arrays are kept anywhere) goes to the
    gitignored ``.cells.jsonl``; this trims the redundant generator block so the sidecar
    stays under the 500 kB pre-commit gate.
    """
    keep_sel = (
        "mcs",
        "selected_cluster",
        "n_selected",
        "has_soft",
        "roc_f_i",
        "ap_f_i",
        "roc_tgt",
        "ap_tgt",
        "roc_score_hdbscan",
        "ap_score_hdbscan",
        "roc_ptilde_hdbscan",
        "ap_ptilde_hdbscan",
        "roc_score_soft",
        "ap_score_soft",
        "roc_ptilde_soft",
        "ap_ptilde_soft",
    )
    return {
        "seed": c["seed"],
        "n_members": c["n_members"],
        "contamination": c["contamination"],
        "fractal_dimension": c["fractal_dimension"],
        "held_out": c["held_out"],
        "n_sources": c["n_sources"],
        "base_rate": round(c["base_rate"], 6),
        "search_failed": c["search_failed"],
        "would": c["would"],
        "sel": {
            s: {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in c["sel"][s].items()
                if k in keep_sel
            }
            for s in SELECTORS
        },
        "disagreement": {
            k: v for k, v in c["disagreement"].items() if not k.startswith("all_three")
        },
        "checks": c["checks"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("tgt_selector_robustness.json")))
    ap.add_argument("--mcs-lo", type=int, default=10)
    ap.add_argument("--mcs-hi", type=int, default=100)
    ap.add_argument("--n-grid", default="30,61,150")
    ap.add_argument("--cont-grid", default="0.5,0.8,0.95")
    ap.add_argument("--dim-grid", default="1.6,3.0")
    ap.add_argument("--realisations", type=int, default=6)
    ap.add_argument("--reference", default=str(REFERENCE_CELLS))
    args = ap.parse_args(argv)

    mcs_range = range(args.mcs_lo, args.mcs_hi)
    n_grid = [int(v) for v in args.n_grid.split(",")]
    cont_grid = [float(v) for v in args.cont_grid.split(",")]
    dim_grid = [float(v) for v in args.dim_grid.split(",")]
    reference = _load_reference(Path(args.reference))

    t0 = time.perf_counter()
    ckpt = Path(args.out).with_suffix(".cells.jsonl")
    ckpt.write_text("")
    cells: list[dict] = []
    total = len(n_grid) * len(cont_grid) * len(dim_grid) * args.realisations
    for n_members in n_grid:
        for cont in cont_grid:
            for dim in dim_grid:
                for k in range(args.realisations):
                    seed = (
                        1000 * n_grid.index(n_members)
                        + 100 * int(cont * 100)
                        + 10 * dim_grid.index(dim)
                        + k
                    )
                    c = run_cell(
                        n_members=n_members,
                        contamination=cont,
                        fractal_dimension=dim,
                        seed=seed,
                        mcs_range=mcs_range,
                        reference=reference,
                    )
                    cells.append(c)
                    with ckpt.open("a") as fh:
                        fh.write(json.dumps(c) + "\n")
                    d = c["disagreement"]
                    print(
                        f"[{len(cells):3d}/{total}] N={n_members} c={cont} D={dim} seed={seed} "
                        f"heldout={c['held_out']} fail={c['search_failed']} "
                        f"mcs={[c['sel'][s]['mcs'] for s in SELECTORS]} "
                        f"agree_set={d['all_three_agree_set']} "
                        f"repro5d={c['checks'].get('reproduces_erotica_5d')} "
                        f"reprosoft={c['checks'].get('reproduces_erotica_5d_soft')} "
                        f"t={time.perf_counter() - t0:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )

    n = len(cells)
    checks = {
        "reproduces_erotica_5d": f"{sum(1 for c in cells if c['checks'].get('reproduces_erotica_5d'))}/{n}",
        "reproduces_erotica_5d_soft": f"{sum(1 for c in cells if c['checks'].get('reproduces_erotica_5d_soft'))}/{n}",
        "failure_is_paired": f"{sum(1 for c in cells if c['checks']['failure_is_paired'])}/{n}",
        "f_i_selector_independent": f"{sum(1 for c in cells if c['checks']['f_i_selector_independent'])}/"
        f"{sum(1 for c in cells if c['checks']['f_i_selector_independent'] is not None)}",
        "would_matches_actual": f"{sum(1 for c in cells if c['checks']['would_matches_actual'])}/"
        f"{sum(1 for c in cells if c['checks']['would_matches_actual'] is not None)}",
        "same_mcs_gives_identical_vectors": f"{sum(1 for c in cells if c['checks']['same_mcs_gives_identical_vectors'])}/"
        f"{sum(1 for c in cells if c['checks']['same_mcs_gives_identical_vectors'] is not None)}",
        "max_product_identity_err": max(
            (c["checks"]["product_identity_maxabs"] or 0.0) for c in cells
        ),
        "n_search_failures": sum(1 for c in cells if c["search_failed"]),
        "n_soft_column_missing": {
            s: sum(1 for c in cells if not c["search_failed"] and not c["sel"][s]["has_soft"])
            for s in SELECTORS
        },
    }
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": time.perf_counter() - t0,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                p: _pkg_version(p)
                for p in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "astropy",
                    "pandas",
                    "hdbscan",
                    "erotica",
                )
            },
        },
        "config": vars(args),
        "held_out_definition": "k = seed % 10; k in {3,4,5} held out, k in {0,1,2} train",
        "checks": checks,
        "summary": summarise(cells),
        "cells": [_reduce_cell(c) for c in cells],
    }
    out_path = Path(args.out)
    text = json.dumps(payload, indent=2)
    if len(text.encode()) > 480_000:  # pre-commit check-added-large-files: --maxkb=500
        text = json.dumps(payload, separators=(",", ":"))
        print("[note] sidecar compacted to stay under the 500 kB gate", file=sys.stderr)
    out_path.write_text(text)
    print(
        f"wrote {out_path} ({n} cells, {len(text.encode()) / 1024:.0f} kB, "
        f"{payload['wall_clock_s']:.1f} s)",
        file=sys.stderr,
    )
    print(json.dumps(checks, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
