"""Is ``_soft_membership_column`` reading the SELECTED cluster's column, or another one?

WHY THIS EXISTS
---------------
``benchmark_ptilde_decomposition.py`` measured, on 54 held-out synthetic cells:

    soft membership of the selected cluster, alone   average precision  0.4559
    tgt = (labels_ == selected_cluster), alone       average precision  0.7831

``_soft_membership_column`` (``erotica/core/clustering.py:745``) returns
``all_points_membership_vectors(clusterer)[:, selected_label]``, so it is target-aware **by
construction**. A continuous, target-aware score losing to the hard binary indicator of the
*same* cluster by 0.33 average precision is not something a well-formed score does. Either
the column being read is not the selected cluster's column -- a shipped bug affecting every
``probability_method="soft"`` run -- or soft membership genuinely smears mass onto field
stars and the loss is a property of the score. The two diagnoses lead to opposite fixes, so
this is settled before anything is built on top of it.

The method's own docstring already flags the hazard, and it is a real one, traced to source:

* ``all_points_membership_vectors`` builds its columns from
  ``sorted(clusterer.condensed_tree_._select_clusters())`` (``prediction.py:658``), and
  ``PredictionData`` builds ``exemplars`` from the same sorted list (``prediction.py:174``).
* ``labels_`` are assigned by ``cluster_map = {c: n for n, c in enumerate(sorted(clusters))}``
  in ``get_clusters`` (``_hdbscan_tree.pyx:1040``), where ``clusters`` is the set of
  condensed-tree node ids that EOM actually selected.
* ``_select_clusters`` (``plots.py:235-245``) does **not** have that set. It reverse-engineers
  it from ``labels_`` as ``groups[label].min()`` -- the smallest condensed-tree node any point
  of that label fell out of -- and returns the values **in label order**.

THE EXACT TEST, AND WHY IT IS EXACT
------------------------------------
``_select_clusters`` returns one entry per label **present in** ``labels_``, in ascending
label order; ``all_points_membership_vectors`` then sorts those ids. So the mapping is

    column c  <->  present_labels[argsort(raw)[c]]

where ``present_labels = sorted(set(labels_) - {-1})``. Reading ``soft[:, selected_label]``
is therefore correct exactly when that mapping is the identity, which needs **two**
conditions, and the method's docstring names only the first:

1. ``raw`` is ascending. Node ids are assigned breadth-first, so a selected parent ``C`` at
   depth ``d`` has a smaller id than a sibling ``D``, while ``C``'s children at depth ``d+1``
   have LARGER ids than ``D``. If no point falls directly out of ``C`` before it splits,
   ``groups[C].min()`` is a child id, which can sort after ``D`` and permute the columns.
   ``cluster_selection_method="eom"`` selects parents over children routinely.
2. ``present_labels == [0, 1, ..., k-1]``, i.e. **no label is empty**. ``get_clusters``
   numbers labels from the selected node set, but ``do_labelling`` can then reassign every
   point of a cluster to noise -- and with ``match_reference_implementation=True``, which
   this package hardcodes, there is an *extra* ``-1`` assignment beyond the three documented
   effects (``_hdbscan_tree.pyx:508-512``). A label with no surviving points never appears in
   ``labels_``, ``_select_clusters`` skips it, and **every higher label's column shifts down
   by one.**

Both are tested. The second is not hypothetical: the 108-cell run in
``tgt_selector_robustness.json`` already records ``n_soft_column_missing`` = 5 / 1 / 1 across
the three selectors, which is this condition being intercepted downstream and logged as
"soft membership unavailable".

WHAT THE BOUNDS GUARD DOES AND DOES NOT CATCH
----------------------------------------------
``_soft_membership_column`` bails out when ``selected_label >= soft.shape[1]``. That is a
**bounds** check, not an identity check. With a gap at the top of the label range it fires and
the run silently falls back to ``probabilities_``. With three or more present labels and a gap
that is not at the top -- present ``[0, 2, 3]``, selected ``2`` -- the naive index is in range
and reads label ``3``'s column instead. So the guard converts most instances of the defect
into a silent downgrade and leaves a residue that is a silent wrong answer. Both outcomes are
counted separately below.

A SECOND ORACLE, WITH DIFFERENT ALGEBRA
----------------------------------------
The mapping above is read from source; a transcription error in reading it would be invisible
to itself. So each cell is also checked against the *exemplars*, which are physical data rows
rather than indices: ``prediction_data_.exemplars[c]`` holds the raw coordinates of the points
that define column ``c``. Those rows are matched back to the input by value and their
``labels_`` read. Every exemplar of column ``c`` must carry the label that the derived mapping
assigns to column ``c``. This uses the exemplar arrays, not the sorted-id argument, so it fails
independently. Value-matching requires the input rows to be distinct, which is asserted per
cell (5 continuous z-scored columns; a collision would be a generator bug).

PRE-REGISTERED FALSIFICATION CRITERIA (written before any number was seen)
--------------------------------------------------------------------------
* **BUG CONFIRMED, ACTIVE** if in one or more cells the naive index ``selected_label`` is in
  range but the derived mapping sends ``selected_label`` to a different column. The shipped
  soft arm then read the wrong column, its published numbers are contaminated, and the fix
  precedes every other question.
* **BUG CONFIRMED, LATENT** if the mapping is not the identity in one or more cells but every
  such instance is intercepted by the bounds guard. The shipped numbers are then uncontaminated
  -- the arm silently fell back to ``probabilities_`` instead -- but the read is wrong by
  construction and only accidentally safe.
* **BUG REFUTED** if the mapping is the identity in **every** cell. Then the 0.33
  average-precision deficit is a property of soft membership itself, and the smear diagnostics
  below quantify the mechanism.
* **INCONCLUSIVE** if the two oracles disagree.

A permutation of two columns is a swap, which is the easiest case to detect; a *partial*
permutation needs three or more clusters to exist at all. The count of cells with ``>= 3``
non-noise clusters is reported next to the verdict, and a verdict resting only on two-cluster
cells is labelled as such.

THE MECHANISM, IF THE BUG IS REFUTED
-------------------------------------
``all_points_membership_vectors`` ends with (``prediction.py:711-714``)::

    result = distance_vecs * outlier_vecs
    result /= result.sum(axis=1)[:, None]
    result *= in_cluster_probs[:, None]

The row normalisation makes each row a *share* of ``in_cluster_probs``. In a contaminated
frame the field is itself an extended HDBSCAN cluster whose exemplars are scattered across
the whole field of view, so almost every point -- member or not -- sits near some field
exemplar and the mass splits. The diagnostics recorded per cell are therefore the mean
selected-column value among members and among non-members, their ratio, and the average
precision of each column, which together decide whether the deficit is smear rather than
misindexing.

ENVIRONMENT
-----------
``erotica-bench`` (python 3.13, hdbscan 0.8.44, asteca 0.7.0, erotica -e). NOT ``cosmic``:
it has no ``erotica`` installed and ships asteca 0.6.9.

USAGE
-----
    python tools/validation/soft_column_alignment.py --out soft_column_alignment.json
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


def _row_key(arr: np.ndarray) -> list[bytes]:
    return [np.ascontiguousarray(r, dtype=float).tobytes() for r in arr]


def audit_cell(real, *, mcs_range: range, selection: str) -> dict:
    """Run one soft-arm fit and interrogate the column mapping two independent ways."""
    import hdbscan as _hdbscan

    from erotica import Clustering

    y = real.truth.astype(int)
    cols = {f"{q}_z": B._zscore(getattr(real, q)) for q in QUANTITIES}
    X = np.column_stack([cols[c] for c in cols])
    out: dict = {"selection": selection, "n_sources": int(y.size), "base_rate": float(y.mean())}

    keys = _row_key(X)
    out["rows_distinct"] = bool(len(set(keys)) == len(keys))

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
            )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["runtime_s"] = time.perf_counter() - t0
        return out
    out["runtime_s"] = time.perf_counter() - t0
    out["error"] = None

    labels = np.asarray(clu.data["cluster_hdbscan"], dtype=int)
    sel = int(clu.pseudoprobability_selected_["selected_cluster"])
    out["selected_cluster"] = sel
    out["best_mcs"] = int(clu.best_params_["min_cluster_size"])
    non_noise = sorted(int(v) for v in set(labels.tolist()) if v >= 0)
    out["n_clusters"] = len(non_noise)
    out["labels_present"] = non_noise
    out["cluster_sizes"] = {str(v): int((labels == v).sum()) for v in non_noise}

    # ---- ORACLE 1: the mapping derived from the two sorts ----------------
    raw = list(int(v) for v in clu.clusterer.condensed_tree_._select_clusters())
    order = [int(i) for i in np.argsort(np.asarray(raw))] if raw else []
    # column c holds the node sorted(raw)[c] = raw[order[c]], which is the (order[c])-th
    # PRESENT label. Hence:
    label_of_column = [non_noise[j] for j in order] if len(order) == len(non_noise) else None
    out["select_clusters_raw"] = raw
    out["ascending"] = bool(raw == sorted(raw))
    out["present_labels_contiguous"] = bool(non_noise == list(range(len(non_noise))))
    out["label_of_column"] = label_of_column
    out["mapping_is_identity"] = bool(
        label_of_column is not None and label_of_column == list(range(len(label_of_column)))
    )

    soft = np.atleast_2d(
        np.asarray(_hdbscan.all_points_membership_vectors(clu.clusterer), dtype=float)
    )
    out["soft_shape"] = list(soft.shape)
    out["n_columns_equals_n_clusters"] = bool(soft.shape[1] == len(non_noise))

    # What the SHIPPED code does with `selected_label`, and whether it is right.
    correct_column = (
        label_of_column.index(sel) if (label_of_column and sel in label_of_column) else None
    )
    out["correct_column_for_selected"] = correct_column
    out["naive_index_in_range"] = bool(0 <= sel < soft.shape[1])
    out["naive_reads_wrong_column"] = bool(
        out["naive_index_in_range"] and correct_column is not None and correct_column != sel
    )
    out["guard_intercepted"] = bool(
        (not out["naive_index_in_range"]) and correct_column is not None
    )

    # ---- ORACLE 2: exemplar rows carry their column's label -------------
    index_of = {k: i for i, k in enumerate(keys)}
    exemplar_ok, exemplar_detail = True, []
    for c, ex in enumerate(clu.clusterer.prediction_data_.exemplars):
        expect = label_of_column[c] if label_of_column and c < len(label_of_column) else None
        ex = np.atleast_2d(np.asarray(ex, dtype=float))
        if ex.size == 0:
            exemplar_detail.append({"column": c, "n_exemplars": 0, "labels": {}})
            exemplar_ok = False
            continue
        idx = [index_of.get(k) for k in _row_key(ex)]
        found = [i for i in idx if i is not None]
        lab = labels[np.asarray(found, dtype=int)] if found else np.empty(0, dtype=int)
        counts = {str(int(v)): int((lab == v).sum()) for v in sorted(set(lab.tolist()))}
        exemplar_detail.append(
            {
                "column": c,
                "expected_label": expect,
                "n_exemplars": int(ex.shape[0]),
                "n_matched": len(found),
                "labels": counts,
            }
        )
        if expect is None or len(found) != ex.shape[0] or counts != {str(expect): len(found)}:
            exemplar_ok = False
    out["exemplar_labels"] = exemplar_detail
    out["exemplar_oracle_ok"] = bool(exemplar_ok)

    # ---- what the two diagnoses predict differently ---------------------
    tgt = (labels == sel).astype(float)
    out["purity_selected"] = float(y[labels == sel].mean()) if (labels == sel).any() else None
    out["ap_tgt"] = float(average_precision_score(y, tgt)) if 0 < y.sum() < y.size else None
    out["ap_by_column"] = [
        float(average_precision_score(y, soft[:, c])) if 0 < y.sum() < y.size else None
        for c in range(soft.shape[1])
    ]
    out["ap_selected_column"] = out["ap_by_column"][sel] if 0 <= sel < soft.shape[1] else None
    out["argmax_ap_column"] = (
        int(np.argmax([a for a in out["ap_by_column"]]))
        if out["ap_by_column"][0] is not None
        else None
    )
    # AUC of every column against every cluster's hard indicator: the assignment this induces
    # must be the identity if the mapping is right. Scale-free, so an overall level difference
    # between columns cannot fake it.
    auc = []
    for c in range(soft.shape[1]):
        row = []
        for j in non_noise:
            ind = (labels == j).astype(int)
            row.append(
                float(roc_auc_score(ind, soft[:, c])) if 0 < ind.sum() < ind.size else float("nan")
            )
        auc.append(row)
    out["auc_column_vs_label"] = auc
    out["auc_assignment"] = [
        int(non_noise[int(np.nanargmax(r))]) if np.any(np.isfinite(r)) else -1 for r in auc
    ]
    out["auc_assignment_is_identity"] = bool(out["auc_assignment"] == non_noise)

    if 0 <= sel < soft.shape[1]:
        m = labels == sel
        out["mean_soft_sel_in_target"] = float(soft[m, sel].mean()) if m.any() else None
        out["mean_soft_sel_out_target"] = float(soft[~m, sel].mean()) if (~m).any() else None
        out["row_sum_mean"] = float(soft.sum(axis=1).mean())
        out["frac_nonmembers_over_half"] = (
            float((soft[~m, sel] > 0.5).mean()) if (~m).any() else None
        )
    return out


def summarise(cells: list[dict]) -> dict:
    ok = [c for c in cells if not c.get("error")]
    asc = [c for c in ok if c["ascending"]]
    exo = [c for c in ok if c["exemplar_oracle_ok"]]
    ident = [c for c in ok if c["mapping_is_identity"]]
    active = [c for c in ok if c["naive_reads_wrong_column"]]
    latent = [c for c in ok if c["guard_intercepted"]]
    multi = [c for c in ok if c["n_clusters"] >= 3]
    if not ok:
        verdict = "INCONCLUSIVE (no cell produced a cluster)"
    elif len(exo) < len(ok):
        verdict = "INCONCLUSIVE (oracles disagree: exemplars contradict the derived mapping)"
    elif active:
        verdict = f"BUG CONFIRMED, ACTIVE ({len(active)}/{len(ok)} cells read the wrong column)"
    elif latent:
        verdict = (
            f"BUG CONFIRMED, LATENT ({len(latent)}/{len(ok)} cells have a non-identity mapping; "
            "every one was intercepted by the bounds guard, so no wrong column was read)"
        )
    else:
        verdict = "BUG REFUTED"
    if verdict == "BUG REFUTED" and not multi:
        verdict = "BUG REFUTED (two-cluster cells only; a partial permutation could not arise)"

    def ms(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        if v.size == 0:
            return {"mean": None, "sem": None, "n": 0}
        return {
            "mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else None,
            "n": int(v.size),
        }

    return {
        "verdict": verdict,
        "n_cells": len(cells),
        "n_cells_with_cluster": len(ok),
        "n_cells_ge3_clusters": len(multi),
        "ascending": f"{len(asc)}/{len(ok)}",
        "present_labels_contiguous": f"{sum(1 for c in ok if c['present_labels_contiguous'])}/{len(ok)}",
        "mapping_is_identity": f"{len(ident)}/{len(ok)}",
        "naive_reads_wrong_column": f"{len(active)}/{len(ok)}",
        "guard_intercepted": f"{len(latent)}/{len(ok)}",
        "exemplar_oracle_ok": f"{len(exo)}/{len(ok)}",
        "auc_assignment_identity": f"{sum(1 for c in ok if c['auc_assignment_is_identity'])}/{len(ok)}",
        "n_columns_equals_n_clusters": f"{sum(1 for c in ok if c['n_columns_equals_n_clusters'])}/{len(ok)}",
        "rows_distinct": f"{sum(1 for c in cells if c['rows_distinct'])}/{len(cells)}",
        "selected_column_is_argmax_ap": (
            f"{sum(1 for c in ok if c['argmax_ap_column'] == c['selected_cluster'])}/{len(ok)}"
        ),
        "ap_selected_column": ms([c.get("ap_selected_column") for c in ok]),
        "ap_tgt": ms([c.get("ap_tgt") for c in ok]),
        "mean_soft_sel_in_target": ms([c.get("mean_soft_sel_in_target") for c in ok]),
        "mean_soft_sel_out_target": ms([c.get("mean_soft_sel_out_target") for c in ok]),
        "row_sum_mean": ms([c.get("row_sum_mean") for c in ok]),
        "purity_selected": ms([c.get("purity_selected") for c in ok]),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("soft_column_alignment.json")))
    ap.add_argument("--mcs-lo", type=int, default=10)
    ap.add_argument("--mcs-hi", type=int, default=100)
    ap.add_argument("--n-grid", default="30,61,150")
    ap.add_argument("--cont-grid", default="0.5,0.8,0.95")
    ap.add_argument("--dim-grid", default="1.6,3.0")
    ap.add_argument("--realisations", default="0,3", help="k values from the 6-realisation grid")
    args = ap.parse_args(argv)

    mcs_range = range(args.mcs_lo, args.mcs_hi)
    n_grid = [int(v) for v in args.n_grid.split(",")]
    cont_grid = [float(v) for v in args.cont_grid.split(",")]
    dim_grid = [float(v) for v in args.dim_grid.split(",")]
    ks = [int(v) for v in args.realisations.split(",")]

    t0 = time.perf_counter()
    cells: list[dict] = []
    for n_members in n_grid:
        for cont in cont_grid:
            for dim in dim_grid:
                for k in ks:
                    seed = (
                        1000 * n_grid.index(n_members)
                        + 100 * int(cont * 100)
                        + 10 * dim_grid.index(dim)
                        + k
                    )
                    real = B.generate(
                        n_members=n_members,
                        contamination=cont,
                        fractal_dimension=dim,
                        seed=seed,
                    )
                    for selection in ("max_persistence", "max_members"):
                        c = audit_cell(real, mcs_range=mcs_range, selection=selection)
                        c.update(
                            {
                                "n_members": n_members,
                                "contamination": cont,
                                "fractal_dimension": dim,
                                "seed": seed,
                                "held_out": (seed % 10) in (3, 4, 5),
                            }
                        )
                        cells.append(c)
                        print(
                            f"[{len(cells):3d}] seed={seed} {selection:16s} "
                            f"err={c['error'] is not None} "
                            f"ncl={c.get('n_clusters')} asc={c.get('ascending')} "
                            f"exo={c.get('exemplar_oracle_ok')} "
                            f"apsel={c.get('ap_selected_column')} aptgt={c.get('ap_tgt')} "
                            f"t={time.perf_counter() - t0:.0f}s",
                            file=sys.stderr,
                            flush=True,
                        )

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": time.perf_counter() - t0,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                p: _pkg_version(p) for p in ("numpy", "scikit-learn", "hdbscan", "erotica")
            },
        },
        "config": vars(args),
        "summary": summarise(cells),
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload["summary"], indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
