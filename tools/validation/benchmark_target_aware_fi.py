"""Make the sweep-recovery term target-aware, and measure whether that earns the default.

WHY THIS EXISTS
---------------
EROTICA's per-star score is a product, ``p_tilde_i = score_i * f_i``, and
``benchmark_ptilde_decomposition.py`` established that **neither factor encodes
target-cluster identity**:

* ``score`` is ``probabilities_``, the membership strength in *whichever* cluster a star was
  assigned. A field star sitting firmly inside the field cluster scores ~1.0.
* ``f_i`` is ``probability_times``, the fraction of swept ``min_cluster_size`` values in which
  the star was clustered into **anything**.

``benchmark_tgt_selector_robustness.py`` then showed the consequence is structural rather than
an artefact of one selector: across a 3x2 factorial of selection rule and score type, a better
selector buys **+0.094 average precision on ``tgt``** (the hard indicator ``labels == selected``)
and **~0.000 on ``p_tilde``**. The product cannot respond to a better choice of cluster because
nothing in it refers to the choice.

This script builds the fix the diagnosis calls for and measures it. ``f_i`` is the higher-leverage
factor -- alone it already scores 0.5804 average precision against the shipped product's 0.5144 --
so the change under test is to count recovery into the **target** instead of into anything:

    f_target_i = (1/n_steps) * #{ sweep steps where star i landed in that step's target cluster }

with the step's target chosen **without ground truth**, as the cluster of maximum Jaccard overlap
with the finally selected member set. Implemented as
``Clustering.search_pseudoprobability(recovery_frequency="target")``.

PRE-REGISTERED CHOICES (fixed before any held-out number was seen)
------------------------------------------------------------------
* **Matching rule: Jaccard, argmax, no floor beyond a non-empty intersection.** Jaccard rather
  than raw overlap because raw overlap is maximised by the largest cluster, which at high
  contamination is the field. No floor because any floor is a tuned quantity; requiring the
  matched cluster to share at least one source with the target is the only threshold the
  definition forces. Centroid proximity is *not* run as a co-candidate -- it would introduce a
  metric and a scale choice, and picking between two matching rules on held-out data is how a
  selector gets overfit.
* **Primary comparison: same selector, same score type, ``f_target`` against ``f_any``.** One
  factor changes and nothing else, so a difference is attributable to the factor. The headline
  comparison against the two published arms (``erotica_5d`` = max_members + hdbscan,
  ``erotica_5d_soft`` = max_persistence + soft) is reported alongside.
* **``p_tilde * tgt`` is a REFERENCE POINT, not a candidate.** It hard-zeros every non-member,
  so it cannot rank in the field and cannot be calibrated there. It is scored only to show what
  the crude gate buys, against which a principled score has to justify itself.
* **``tgt`` is not shippable** and is scored only as the ceiling of this family: it is binary,
  has one operating point, and its ROC on the hdbscan arm sits *below* ``p_tilde``'s.

WHAT WOULD FALSIFY THE CONCLUSION -- THE FOUR BARS, ALL OF WHICH MUST HOLD
--------------------------------------------------------------------------
1. **Held-out average precision**, paired per cell against the shipped product at the same
   selector and score type, with a standard error and a win/loss/**tie** count. Adoption fails
   if the paired mean is not clear of zero by a comfortable margin of its own standard error, or
   if the win rate is at or below half once forced ties are excluded.
2. **A matched control.** Any candidate that changes the sweep grid must be compared against the
   same grid: grid decimation *alone* costs 0.0385 +- 0.0114 average precision, which is the
   size of a real effect, and comparing against an undecimated baseline would misattribute it.
   Here the control holds **by construction and is asserted rather than assumed**: ``f_target``
   is computed from the same ``labels_matrix``, over the same ``min_cluster_size`` grid, with the
   same denominator ``n_steps`` as ``f_any``. The per-cell assertion is
   ``f_target <= f_any`` pointwise -- true only if the counted steps are a subset of the same
   step set -- plus equality of the two grids. Adoption fails if either assertion fails in any
   cell, because then the comparison is confounded and the number means nothing.
3. **Calibration, out of sample, decomposed.** Isotonic is fitted on the TRAIN seeds
   (``k in {0,1,2}``) and scored on the HELD-OUT seeds (``k in {3,4,5}``) -- not on the
   ``seeds[::2]`` split ``benchmark_erotica_vs_asteca.out_of_sample_recalibration`` uses, which
   would fit on rows inside the decision block. Reported as the Murphy decomposition,
   ``BS = reliability - resolution + uncertainty``, plus ECE. **Reliability is repairable by
   recalibration and resolution is not.** Adoption fails if the candidate has no more resolution
   than the shipped product after isotonic: a ranking gain that recalibration erases was never a
   gain in information. This is the argument that decided the soft-membership question.
4. **The negative control.** Field-only frames, no cluster injected, where every selected star is
   a false positive. Adoption fails if a candidate manufactures members from pure field,
   whatever its average precision. Reported at the shipped mask rule *and* at matched operating
   points, because ``f_target <= f_any`` means a candidate selects fewer stars at a fixed
   threshold for purely mechanical reasons and a fixed-threshold count would flatter it.

TWO THINGS THAT WOULD MAKE THE HEADLINE NUMBER MEAN LESS THAN IT LOOKS
-----------------------------------------------------------------------
* **34 of 108 cells return no cluster at all**, concentrated at LOW contamination. In those the
  search raises, every quantity is the all-zero vector, ROC is 0.5, average precision is the base
  rate, and every paired delta is exactly 0. They are forced ties that dilute the mean and pad
  the win-rate denominator. Every table is therefore reported twice: over all held-out cells, so
  it stays comparable to the published 0.7831 / 0.8615, and over CONTRIBUTING cells only, where
  the comparison has power. Win/loss/tie is reported with the tie count visible.
* **A large share of the gap to ASteCA is that failure mode, not the score construction.**
  ASteCA returns a cluster in cells where the sweep returns nothing. Fixing the score cannot
  recover those cells, and conflating the two would misattribute whatever ships here.

THE HAZARD IN THE MATCHING RULE, MEASURED RATHER THAN ASSUMED AWAY
-------------------------------------------------------------------
Argmax-Jaccard with no floor still matches *something* at any step where a cluster overlaps the
target. The failure case is a step whose only cluster is one giant structure that happens to
contain the target: Jaccard is small, but it is the argmax, and every field star in that
structure is counted. The diagnostics ``mean_matched_jaccard`` and ``mean_matched_size_ratio``
(matched cluster size / target size) are recorded per cell so the rate of that case is visible
instead of inferred.

REPRODUCTION CHECK, AND THE ONE PLACE IT IS EXPECTED TO BREAK
--------------------------------------------------------------
``score * probability_times`` is asserted per cell against the stored ``p_tilde`` of the 108-cell
run, for ``max_members`` + hdbscan (= ``erotica_5d``) and ``max_persistence`` + soft
(= ``erotica_5d_soft``). It must hold everywhere EXCEPT in cells where
``soft_column.remapped`` is true: ``soft_column_alignment.py`` established that
``_soft_membership_column`` used to index ``all_points_membership_vectors`` by label instead of
by resolved column, and fell back to ``probabilities_`` whenever that index was out of range.
Those cells now get the correct soft column, so their soft ``p_tilde`` legitimately differs from
the stored value. The count is reported and the two populations are separated; a mismatch in a
non-remapped cell means the re-run is not the same experiment and nothing here is comparable.

ENVIRONMENT
-----------
``erotica-bench`` (python 3.13, hdbscan 0.8.44, asteca 0.7.0, erotica -e). NOT ``cosmic``, which
has no ``erotica`` installed and ships asteca 0.6.9.

USAGE
-----
    python tools/validation/benchmark_target_aware_fi.py --out target_aware_fi.json
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

QUANTITIES = ("ra", "dec", "pmra", "pmdec", "plx")
SELECTORS = ("max_members", "max_persistence")
HELD_OUT_K = (3, 4, 5)

# Which stored arm each (selector, score) run must reproduce.
REFERENCE_ARM = {
    ("max_members", "hdbscan"): "erotica_5d",
    ("max_persistence", "soft"): "erotica_5d_soft",
}
REFERENCE_CELLS = Path(
    "/private/tmp/claude-501/-Users-notluquis-phd/"
    "d055f28c-94b0-484a-8d8b-601af31f9812/scratchpad/soft_bench.cells.jsonl"
)

# The quantities scored per (cell, selector). `shipped_*` are the products that ship today;
# `A_*` / `B_*` are the target-aware candidates; `C_*` are the crude-gate reference points;
# the rest are context.
QUANTITY_NAMES = (
    "shipped_hdbscan",  # score_hdbscan * f_any          <- erotica_5d at max_members
    "shipped_soft",  # score_soft    * f_any          <- erotica_5d_soft at max_persistence
    "A_f_target",  # f_target alone
    "A_hdbscan",  # score_hdbscan * f_target
    "B_soft",  # score_soft    * f_target       <- candidate B
    "C_gate_hdbscan",  # shipped_hdbscan * tgt          <- reference, not a candidate
    "C_gate_soft",  # shipped_soft    * tgt          <- reference, not a candidate
    "f_any",
    "score_hdbscan",
    "score_soft",
    "tgt",
)
# Which shipped product each candidate is paired against (same selector, same score type).
PAIRED_BASELINE = {
    "A_f_target": "shipped_hdbscan",
    "A_hdbscan": "shipped_hdbscan",
    "B_soft": "shipped_soft",
    "C_gate_hdbscan": "shipped_hdbscan",
    "C_gate_soft": "shipped_soft",
}


def held_out(seed: int) -> bool:
    return (seed % 10) in HELD_OUT_K


def _score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


# ---------------------------------------------------------------------------
# Murphy decomposition
# ---------------------------------------------------------------------------
def _bin_index(p: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin assignment that gives EVERY heavy atom its own bin, not just the one at zero.

    Quantile binning collapses on these scores: ``f_target``, the crude gate and every failed cell
    put a large point mass at a single value, and a quantile edge inside that mass produces bins
    differing only by tie-breaking, which turns reliability and resolution into binning artefacts.

    .. warning::
       **The first version separated the atom at exactly zero and quantile-binned the rest, and
       that was wrong in the one place it mattered most.** After isotonic recalibration the atom
       is no longer at zero — isotonic maps the whole zero group to that group's empirical rate, a
       small POSITIVE constant. So ``p > 0`` was true for ~every row, ``np.unique(np.quantile(...))``
       on a distribution where 91% of the mass shares one value returned fewer than two distinct
       edges, and the fallback assigned a SINGLE bin. Resolution is a between-bin variance, so one
       bin makes it identically ``0.0`` by construction.

       Measured consequence: the isotonic resolution of the best-performing arms read exactly
       0.0000 — and so did ``tgt``, a binary indicator with average precision 0.783, whose
       resolution cannot be zero. That impossibility is what exposed it. Read naively, the table
       would have REJECTED the winning score on the very bar chosen to decide adoption.

       The docstring above already identified the hazard and then hardcoded the wrong constant.

    Atoms are now found by mass rather than by value, so zero, an isotonic level and any tie-heavy
    score are handled by one rule. If the forecast is piecewise constant with few levels — which
    isotonic output always is — each level simply becomes its own bin, which is exact.
    """
    p = np.asarray(p, dtype=float)
    idx = np.full(p.size, -1, dtype=int)
    vals, counts = np.unique(p, return_counts=True)

    # Few distinct values (isotonic output, or a binary score): one bin per value. Exact.
    if vals.size <= n_bins:
        return np.searchsorted(vals, p)

    # Otherwise: any value carrying at least 1/n_bins of the mass is an atom and gets its own bin;
    # the remainder is quantile-binned among the bins that are left.
    atoms = vals[counts >= max(1, p.size // n_bins)]
    nxt = 0
    for a in atoms:
        idx[p == a] = nxt
        nxt += 1
    rest = idx < 0
    if not rest.any():
        return idx
    n_rest_bins = max(2, n_bins - int(atoms.size))
    edges = np.unique(np.quantile(p[rest], np.linspace(0.0, 1.0, n_rest_bins)))
    if edges.size < 2:
        idx[rest] = nxt
        return idx
    idx[rest] = nxt + np.clip(np.digitize(p[rest], edges[1:-1], right=False), 0, edges.size - 1)
    return idx


def murphy(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> dict:
    """``BS = reliability - resolution + uncertainty`` on a binned forecast.

    The identity is checked rather than asserted by construction: ``residual`` is the amount by
    which the three terms fail to reconstruct the Brier score, and it is exactly the
    within-bin variance the binning discards. A large residual means the binning is too coarse
    to interpret the split, and it is reported next to the terms rather than hidden.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    n = y.size
    ybar = float(y.mean())
    idx = _bin_index(p, n_bins)
    rel = res = 0.0
    occupancy = []
    for k in np.unique(idx):
        m = idx == k
        nk = int(m.sum())
        pk, yk = float(p[m].mean()), float(y[m].mean())
        rel += nk * (pk - yk) ** 2
        res += nk * (yk - ybar) ** 2
        occupancy.append({"bin": int(k), "n": nk, "mean_p": pk, "mean_y": yk})
    rel /= n
    res /= n
    unc = ybar * (1.0 - ybar)
    bs = float(np.mean((p - y) ** 2))
    return {
        "n": int(n),
        "brier": bs,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "residual": float(bs - (rel - res + unc)),
        "n_bins_used": int(len(occupancy)),
        "frac_exactly_zero": float((p == 0).mean()),
        "occupancy": occupancy,
    }


def ece(p: np.ndarray, y: np.ndarray, *, n_bins: int = 10) -> float:
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    y = np.asarray(y, dtype=float)
    idx = _bin_index(p, n_bins)
    tot = 0.0
    for k in np.unique(idx):
        m = idx == k
        tot += m.sum() * abs(float(p[m].mean()) - float(y[m].mean()))
    return float(tot / y.size)


# ---------------------------------------------------------------------------
# One (cell, selector) run
# ---------------------------------------------------------------------------
def run_selector(real, *, selection: str, mcs_range: range) -> dict:
    """One ``search_pseudoprobability`` run, returning every factor separately.

    ``probability_method="soft"`` is used for BOTH score types because
    ``_annotate_pseudoprobability_results`` writes ``probability_hdbscan`` on every path and
    ``probability_soft`` additionally on the soft path, so a single run carries both scores and
    both products are recoverable. This is the same halving
    ``benchmark_tgt_selector_robustness.py`` uses, and it is licensed by the reproduction check
    below rather than by reading the source.
    """
    from erotica import Clustering

    n = real.truth.size
    cols = {f"{q}_z": B._zscore(getattr(real, q)) for q in QUANTITIES}
    zeros = np.zeros(n)
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
                probability_method="soft",
                recovery_frequency="target",
            )
    except Exception as exc:
        return {
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.perf_counter() - t0,
            "q": {name: zeros for name in QUANTITY_NAMES},
        }
    secs = time.perf_counter() - t0
    tab = clu.data
    sel = int((clu.pseudoprobability_selected_ or {}).get("selected_cluster", -1))
    labels = np.asarray(tab["cluster_hdbscan"], dtype=int)
    f_any = np.asarray(tab["probability_times"], dtype=float)
    f_tgt = np.asarray(tab["probability_times_target"], dtype=float)
    score_h = np.asarray(tab["probability_hdbscan"], dtype=float)
    has_soft = "probability_soft" in tab.colnames
    score_s = np.asarray(tab["probability_soft"], dtype=float) if has_soft else score_h
    tgt = (labels == sel).astype(float) if sel >= 0 else zeros

    q = {
        "shipped_hdbscan": score_h * f_any,
        "shipped_soft": score_s * f_any,
        "A_f_target": f_tgt,
        "A_hdbscan": score_h * f_tgt,
        "B_soft": score_s * f_tgt,
        "C_gate_hdbscan": score_h * f_any * tgt,
        "C_gate_soft": score_s * f_any * tgt,
        "f_any": f_any,
        "score_hdbscan": score_h,
        "score_soft": score_s,
        "tgt": tgt,
    }
    info = (clu.pseudoprobability_selected_ or {}).get("target_recovery", {}) or {}
    soft_col = (clu.pseudoprobability_selected_ or {}).get("soft_column", {}) or {}
    return {
        "error": None,
        "runtime_s": secs,
        "q": q,
        "selected_cluster": sel,
        "best_mcs": int(clu.best_params_["min_cluster_size"]),
        "has_soft": bool(has_soft),
        "soft_column_remapped": bool(soft_col.get("remapped", False)),
        "soft_column": soft_col or None,
        "target_recovery": info,
        # ---- BAR 2, asserted rather than assumed ------------------------
        "bar2_same_grid": bool(int(info.get("n_steps", -1)) == len(list(mcs_range))),
        "bar2_ftarget_le_fany": bool(np.all(f_tgt <= f_any + 1e-12)),
        "bar2_members_recovered": bool(
            (sel < 0)
            or (not (labels == sel).any())
            or np.all(f_tgt[labels == sel] >= 1.0 / max(int(info.get("n_steps", 1)), 1) - 1e-12)
        ),
        "n_distinct_f_target": int(np.unique(f_tgt).size),
        "max_f_target": float(f_tgt.max()),
        "max_f_any": float(f_any.max()),
    }


def run_cell(*, n_members, contamination, fractal_dimension, seed, mcs_range, reference) -> dict:
    real = B.generate(
        n_members=n_members,
        contamination=contamination,
        fractal_dimension=fractal_dimension,
        seed=seed,
    )
    y = real.truth.astype(int)
    cell = {
        "n_members": n_members,
        "contamination": contamination,
        "fractal_dimension": fractal_dimension,
        "seed": seed,
        "held_out": held_out(seed),
        "n_sources": int(y.size),
        "base_rate": float(y.mean()),
        "generator_check": B.check_generator(real, requested_contamination=contamination),
        "arms": {},
    }
    probs: dict[tuple[str, str], np.ndarray] = {}
    for selection in SELECTORS:
        r = run_selector(real, selection=selection, mcs_range=mcs_range)
        arm = {k: v for k, v in r.items() if k != "q"}
        for name, p in r["q"].items():
            roc, ap = _score(y, p)
            arm[f"roc_{name}"] = roc
            arm[f"ap_{name}"] = ap
            probs[(selection, name)] = p
        # reproduction against the stored 108-cell run
        ref = (reference or {}).get(seed, {})
        for (sel_name, score_kind), stored_arm in REFERENCE_ARM.items():
            if sel_name != selection:
                continue
            key = "shipped_hdbscan" if score_kind == "hdbscan" else "shipped_soft"
            stored = ref.get(stored_arm)
            if stored is None:
                arm[f"reproduces_{stored_arm}"] = None
                continue
            s = np.asarray(stored, dtype=float)
            same = bool(s.size == r["q"][key].size and np.allclose(s, r["q"][key], atol=1e-9))
            arm[f"reproduces_{stored_arm}"] = same
            arm[f"reproduces_{stored_arm}_maxabs"] = (
                float(np.max(np.abs(s - r["q"][key]))) if s.size == r["q"][key].size else None
            )
        cell["arms"][selection] = arm
    return cell, probs, y


# ---------------------------------------------------------------------------
def _load_reference(path: Path) -> dict:
    if not path.exists():
        print(f"[warn] reference cells not at {path}; reproduction check disabled", file=sys.stderr)
        return {}
    ref = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
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


def _wlt(deltas: list[float], tol: float = 1e-12) -> dict:
    d = [x for x in deltas if x is not None and np.isfinite(x)]
    w = sum(1 for x in d if x > tol)
    losses = sum(1 for x in d if x < -tol)
    return {"win": w, "loss": losses, "tie": len(d) - w - losses, "n": len(d)}


def summarise(cells: list[dict]) -> dict:
    out: dict = {}
    for split, keep in (
        ("held_out", lambda c: c["held_out"]),
        ("train", lambda c: not c["held_out"]),
    ):
        sub = [c for c in cells if keep(c)]
        blk: dict = {"n_cells": len(sub)}
        for selection in SELECTORS:
            contributing = [c for c in sub if not c["arms"][selection]["error"]]
            sblk = {
                "n_cells": len(sub),
                "n_contributing": len(contributing),
                "n_failed_search": len(sub) - len(contributing),
                "failed_by_contamination": {
                    str(x): sum(
                        1 for c in sub if c["contamination"] == x and c["arms"][selection]["error"]
                    )
                    for x in sorted({c["contamination"] for c in sub})
                },
            }
            for pop, rows in (("all", sub), ("contributing", contributing)):
                for m in ("roc", "ap"):
                    sblk[f"{pop}_{m}"] = {
                        name: _ms([c["arms"][selection][f"{m}_{name}"] for c in rows])
                        for name in QUANTITY_NAMES
                    }
                for name, base in PAIRED_BASELINE.items():
                    for m in ("roc", "ap"):
                        d = [
                            c["arms"][selection][f"{m}_{name}"]
                            - c["arms"][selection][f"{m}_{base}"]
                            for c in rows
                        ]
                        sblk[f"{pop}_delta_{m}_{name}_minus_{base}"] = _ms(d)
                        sblk[f"{pop}_wlt_{m}_{name}_minus_{base}"] = _wlt(d)
            sblk["target_recovery"] = {
                k: _ms([c["arms"][selection]["target_recovery"].get(k) for c in contributing])
                for k in ("n_steps_matched", "mean_matched_jaccard", "mean_matched_size_ratio")
            }
            sblk["n_distinct_f_target"] = _ms(
                [c["arms"][selection]["n_distinct_f_target"] for c in contributing]
            )
            sblk["max_f_target"] = _ms([c["arms"][selection]["max_f_target"] for c in contributing])
            blk[selection] = sblk
        out[split] = blk
    return out


def bar2_checks(cells: list[dict]) -> dict:
    res = {}
    for selection in SELECTORS:
        ok = [c for c in cells if not c["arms"][selection]["error"]]
        res[selection] = {
            "same_grid": f"{sum(1 for c in ok if c['arms'][selection]['bar2_same_grid'])}/{len(ok)}",
            "f_target_le_f_any": (
                f"{sum(1 for c in ok if c['arms'][selection]['bar2_ftarget_le_fany'])}/{len(ok)}"
            ),
            "target_members_recovered": (
                f"{sum(1 for c in ok if c['arms'][selection]['bar2_members_recovered'])}/{len(ok)}"
            ),
        }
    return res


def reproduction_checks(cells: list[dict]) -> dict:
    res = {}
    for (selection, _score_kind), arm_name in REFERENCE_ARM.items():
        rows = [c for c in cells if c["arms"][selection].get(f"reproduces_{arm_name}") is not None]
        # `.get(..., False)` and not `[...]`: a cell whose run raised returns an arm dict built on
        # the error path, which carries no soft-column info at all. Absent info means "not
        # remapped" by definition -- the remap can only happen on a run that produced a cluster.
        # Indexing here crashed the whole aggregation AFTER all 108 cells had been computed, and
        # since nothing was checkpointed the entire run was lost. Hence also `--resume` below.
        remapped = [c for c in rows if c["arms"][selection].get("soft_column_remapped", False)]
        plain = [c for c in rows if not c["arms"][selection].get("soft_column_remapped", False)]
        res[arm_name] = {
            "n_compared": len(rows),
            "n_remapped_by_the_soft_column_fix": len(remapped),
            "reproduces_where_not_remapped": (
                f"{sum(1 for c in plain if c['arms'][selection][f'reproduces_{arm_name}'])}"
                f"/{len(plain)}"
            ),
            "reproduces_where_remapped": (
                f"{sum(1 for c in remapped if c['arms'][selection][f'reproduces_{arm_name}'])}"
                f"/{len(remapped)}"
            ),
        }
    return res


# ---------------------------------------------------------------------------
# BAR 3: out-of-sample calibration, split on the DECISION split
# ---------------------------------------------------------------------------
def calibration(pool: dict, *, n_bins: int) -> dict:
    from erotica.calibration import fit_isotonic

    res: dict = {}
    for selection in SELECTORS:
        blk = {}
        if not pool.get((selection, "_y", "train")) or not pool.get((selection, "_y", "held_out")):
            res[selection] = {"error": "train or held-out pool is empty"}
            continue
        y_tr = np.concatenate(pool[(selection, "_y", "train")])
        y_te = np.concatenate(pool[(selection, "_y", "held_out")])
        blk["n_train_rows"] = int(y_tr.size)
        blk["n_test_rows"] = int(y_te.size)
        for name in QUANTITY_NAMES:
            p_tr = np.concatenate(pool[(selection, name, "train")])
            p_te = np.concatenate(pool[(selection, name, "held_out")])
            entry = {
                "raw": murphy(p_te, y_te, n_bins=n_bins),
                "raw_ece": ece(p_te, y_te, n_bins=n_bins),
            }
            try:
                rec = fit_isotonic(p_tr, y_tr)
                p_cal = np.clip(np.asarray(rec(p_te), dtype=float), 0.0, 1.0)
                entry["isotonic"] = murphy(p_cal, y_te, n_bins=n_bins)
                entry["isotonic_ece"] = ece(p_cal, y_te, n_bins=n_bins)
            except Exception as exc:
                entry["isotonic_error"] = f"{type(exc).__name__}: {exc}"
            # drop the per-bin occupancy from the sidecar: informative to compute, too big to keep
            for k in ("raw", "isotonic"):
                if k in entry:
                    entry[k].pop("occupancy", None)
            blk[name] = entry
        res[selection] = blk
    return res


# ---------------------------------------------------------------------------
# BAR 4: negative control
# ---------------------------------------------------------------------------
def negative_control(*, seeds: list[int], n_field_like: int, mcs_range: range) -> list[dict]:
    """Field-only frames. Every star selected is a false positive, by construction."""
    out = []
    for seed in seeds:
        real = B.generate(
            n_members=n_field_like,
            contamination=0.9,
            fractal_dimension=1.6,
            seed=seed,
            inject_cluster=False,
        )
        row = {"seed": seed, "n_sources": int(real.truth.size), "n_true_members": 0, "arms": {}}
        for selection in SELECTORS:
            r = run_selector(real, selection=selection, mcs_range=mcs_range)
            sel = r.get("selected_cluster", -1)
            tgt = r["q"]["tgt"] > 0
            arm = {"error": r["error"], "selected_cluster": sel}
            for name in QUANTITY_NAMES:
                p = r["q"][name]
                arm[f"{name}_max"] = float(p.max()) if p.size else 0.0
                # shipped mask rule: in the selected cluster AND over the threshold
                arm[f"{name}_n_shipped_rule"] = int(np.count_nonzero(tgt & (p > 0.5)))
                arm[f"{name}_n_over_half"] = int(np.count_nonzero(p > 0.5))
                # matched operating point: the count is fixed, so a smaller-scaled score cannot
                # look better merely by being smaller
                arm[f"{name}_n_top50_positive"] = int(np.count_nonzero(np.sort(p)[::-1][:50] > 0.0))
            row["arms"][selection] = arm
        out.append(row)
    return out


# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("target_aware_fi.json")))
    ap.add_argument("--mcs-lo", type=int, default=10)
    ap.add_argument("--mcs-hi", type=int, default=100)
    ap.add_argument("--n-grid", default="30,61,150")
    ap.add_argument("--cont-grid", default="0.5,0.8,0.95")
    ap.add_argument("--dim-grid", default="1.6,3.0")
    ap.add_argument("--realisations", type=int, default=6)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--negative-control-seeds", default="70001,70002,70003,70004,70005,70006")
    ap.add_argument("--negative-control-n", type=int, default=150)
    ap.add_argument("--reference", default=str(REFERENCE_CELLS))
    args = ap.parse_args(argv)

    mcs_range = range(args.mcs_lo, args.mcs_hi)
    n_grid = [int(v) for v in args.n_grid.split(",")]
    cont_grid = [float(v) for v in args.cont_grid.split(",")]
    dim_grid = [float(v) for v in args.dim_grid.split(",")]
    reference = _load_reference(Path(args.reference))

    t0 = time.perf_counter()
    cells: list[dict] = []
    pool: dict = {}
    # Per-cell checkpoint, truncated at the start of each run so it always describes THIS run.
    # Gitignored with the other `benchmark_*.cells.jsonl` artifacts.
    cells_jsonl = Path(args.out).with_suffix("").with_suffix(".cells.jsonl")
    cells_jsonl.write_text("")

    def stash(key, arr):
        pool.setdefault(key, []).append(arr)

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
                    cell, probs, y = run_cell(
                        n_members=n_members,
                        contamination=cont,
                        fractal_dimension=dim,
                        seed=seed,
                        mcs_range=mcs_range,
                        reference=reference,
                    )
                    cells.append(cell)
                    # Checkpoint EVERY cell as it completes. The first run of this script
                    # computed all 108 cells (~40 min) and then died in aggregation on a
                    # KeyError, and because nothing had been written the whole run was lost.
                    # A sidecar written only at the end is a sidecar that records nothing
                    # about the runs that fail.
                    with open(cells_jsonl, "a") as _ck:
                        _ck.write(json.dumps(cell, default=float) + "\n")
                    split = "held_out" if cell["held_out"] else "train"
                    for selection in SELECTORS:
                        stash((selection, "_y", split), y)
                        for name in QUANTITY_NAMES:
                            stash((selection, name, split), probs[(selection, name)])
                    a = cell["arms"]
                    print(
                        f"[{len(cells):3d}/108] N={n_members} c={cont} D={dim} seed={seed} "
                        f"ho={cell['held_out']} "
                        f"mm_err={a['max_members']['error'] is not None} "
                        f"mp_err={a['max_persistence']['error'] is not None} "
                        f"repro5d={a['max_members'].get('reproduces_erotica_5d')} "
                        f"reprosoft={a['max_persistence'].get('reproduces_erotica_5d_soft')} "
                        f"apB={a['max_persistence'].get('ap_B_soft')} "
                        f"apShip={a['max_persistence'].get('ap_shipped_soft')} "
                        f"t={time.perf_counter() - t0:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )

    print("running negative control ...", file=sys.stderr, flush=True)
    neg = negative_control(
        seeds=[int(v) for v in args.negative_control_seeds.split(",")],
        n_field_like=args.negative_control_n,
        mcs_range=mcs_range,
    )

    print("running calibration ...", file=sys.stderr, flush=True)
    cal = calibration(pool, n_bins=args.n_bins)

    # per-star arrays never reach disk: the sidecar is scalars only (500 kB pre-commit ceiling)
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": time.perf_counter() - t0,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                p: _pkg_version(p)
                for p in ("numpy", "scipy", "scikit-learn", "astropy", "hdbscan", "erotica")
            },
        },
        "config": vars(args),
        "held_out_definition": "k = seed % 10; k in {3,4,5} held out, k in {0,1,2} train",
        "bar1_and_context": summarise(cells),
        "bar2_matched_control": bar2_checks(cells),
        "bar3_calibration": cal,
        "bar4_negative_control": neg,
        "reproduction": reproduction_checks(cells),
        "cells": [
            {k: v for k, v in c.items() if k not in ("arms",)}
            | {
                "arms": {
                    s: {k: v for k, v in a.items() if k not in ("soft_column",)}
                    for s, a in c["arms"].items()
                }
            }
            for c in cells
        ],
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    size_kb = Path(args.out).stat().st_size / 1024
    print(
        f"wrote {args.out} ({len(cells)} cells, {payload['wall_clock_s']:.1f} s, {size_kb:.0f} kB)",
        file=sys.stderr,
    )
    print(json.dumps(payload["bar2_matched_control"], indent=2), file=sys.stderr)
    print(json.dumps(payload["reproduction"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
