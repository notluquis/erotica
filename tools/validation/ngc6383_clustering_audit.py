"""Export transparent clustering audit products for NGC 6383 robustness runs.

The radius robustness script produces the final science summaries. This script
extracts the intermediate clustering state preserved in the `.dill` files:

- the selected paper-faithful configuration,
- all candidate sweep results retained by `search_pseudoprobability`,
- the full min-cluster-size sweep track,
- final HDBSCAN label summaries for every recovered final label,
- a source-level final table with every recovered final label and branch flag,
- final cluster persistence values,
- the final condensed tree table.

Important limitation: the current `Clustering.search_pseudoprobability` method
does not persist the full source-by-run labels matrix. Therefore this audit
cannot reconstruct every label assignment in each of the 290 sweep iterations
without rerunning HDBSCAN. It does preserve all sweep-level state that was
stored in the current objects.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="cosmic-mpl-"))

import dill
import numpy as np

# Imported, never re-implemented. This file used to carry its own byte-for-byte copy of
# `_cluster_label_for_size`, which meant the fix for issue #7 could not reach it and the
# audit would have kept reporting the pre-fix answer as if it were current.
#
# The audit's PRIMARY selector stays the legacy one on purpose: these `.dill` files are
# artifacts of runs published before 2026-08-03, and `algorithm_selected_final_label` is a
# statement about what those runs actually did. `tree_selected_final_label` is reported
# beside it so the effect of the fix on the archived runs is visible rather than inferred.
from erotica.core.clustering import Clustering

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NGC_ROOT = PROJECT_ROOT / "data" / "test" / "NGC6383"
DEFAULT_INPUT_DIR = NGC_ROOT / "comments_paper" / "radius_robustness" / "generated" / "dill"
DEFAULT_OUTPUT_DIR = NGC_ROOT / "comments_paper" / "clustering_audit" / "generated"
DEFAULT_REFERENCE_PM = (2.54, -1.71)


def _jsonable(value: Any) -> Any:
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _as_float_array(values) -> np.ndarray:
    if hasattr(values, "to_value"):
        return np.asarray(values.to_value(), dtype=float)
    quantity = getattr(values, "quantity", None)
    if quantity is not None and hasattr(quantity, "to_value"):
        return np.asarray(quantity.to_value(), dtype=float)
    return np.asarray(values, dtype=float)


def _masked_column(table, column: str, mask: np.ndarray) -> np.ndarray:
    return _as_float_array(table[column][mask])


def _nanmedian(table, column: str, mask: np.ndarray) -> float:
    return float(np.nanmedian(_masked_column(table, column, mask)))


def _finite(values) -> np.ndarray:
    arr = _as_float_array(values)
    return arr[np.isfinite(arr)]


def _stats(table, mask: np.ndarray, column: str) -> dict[str, float | None]:
    if column not in table.colnames or not np.any(mask):
        return {"mean": None, "median": None, "std": None}
    values = _finite(table[column][mask])
    if len(values) == 0:
        return {"mean": None, "median": None, "std": None}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: _jsonable(row.get(key)) for key in fieldnames} for row in rows])


def _candidate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in results:
        persistence = np.asarray(item.get("cluster_persistence", []), dtype=float)
        rows.append(
            {
                "min_cluster_size": item.get("min_cluster_size"),
                "desired_len": item.get("desired_len"),
                "lambda_value": item.get("lambda_value"),
                "relative_validity": item.get("relative_validity"),
                "n_persistence_values": int(len(persistence)),
                "max_cluster_persistence": float(np.nanmax(persistence))
                if len(persistence)
                else None,
                "mean_cluster_persistence": float(np.nanmean(persistence))
                if len(persistence)
                else None,
            }
        )
    return rows


def _sweep_rows(
    sweep_track: list[dict[str, Any]], candidates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_mcs = {int(item["min_cluster_size"]): item for item in candidates}
    rows = []
    for item in sweep_track:
        mcs = int(item["min_cluster_size"])
        candidate = by_mcs.get(mcs)
        rows.append(
            {
                "min_cluster_size": mcs,
                "desired_len": int(item["desired_len"]),
                "is_candidate": candidate is not None,
                "lambda_value": None if candidate is None else candidate.get("lambda_value"),
                "relative_validity": None
                if candidate is None
                else candidate.get("relative_validity"),
            }
        )
    return rows


def _ngc_like_label(
    table, labels: np.ndarray, reference_pm: tuple[float, float]
) -> tuple[int, float]:
    probability = _as_float_array(table["probability"])
    best_label = -1
    best_distance = np.inf
    for label in sorted(int(value) for value in np.unique(labels) if int(value) != -1):
        mask = labels == label
        high_mask = mask & (probability >= 0.5)
        if np.count_nonzero(high_mask) == 0:
            continue
        pmra = _nanmedian(table, "pmra", high_mask)
        pmdec = _nanmedian(table, "pmdec", high_mask)
        distance = float(np.hypot(pmra - reference_pm[0], pmdec - reference_pm[1]))
        if distance < best_distance:
            best_label = label
            best_distance = distance
    if best_label == -1:
        raise RuntimeError(
            "No non-noise label with probability >= 0.5 was available for NGC-like selection."
        )
    return best_label, best_distance


def _final_label_rows(
    clust,
    reference_pm: tuple[float, float],
    algorithm_selected_label: int,
    tree_selected_label: int,
) -> list[dict[str, Any]]:
    table = clust.data
    labels = np.asarray(table["cluster_hdbscan"], dtype=int)
    ngc_like_label, _ = _ngc_like_label(table, labels, reference_pm)
    rows = []

    for label in sorted(np.unique(labels)):
        mask = labels == label
        probability = _as_float_array(table["probability"])
        high_mask = mask & (probability >= 0.5)
        prob = probability[mask]
        prob_hdb = _as_float_array(table["probability_hdbscan"])[mask]
        prob_times = _as_float_array(table["probability_times"])[mask]
        if label != -1 and np.count_nonzero(high_mask) > 0:
            pmra_high = _nanmedian(table, "pmra", high_mask)
            pmdec_high = _nanmedian(table, "pmdec", high_mask)
            pm_distance = float(np.hypot(pmra_high - reference_pm[0], pmdec_high - reference_pm[1]))
        else:
            pm_distance = None
        row = {
            "label": int(label),
            "is_noise": bool(label == -1),
            "is_algorithm_selected_branch": bool(label == algorithm_selected_label),
            "is_tree_selected_branch": bool(label == tree_selected_label),
            "is_ngc_like_branch": bool(label == ngc_like_label),
            "pm_distance_to_reference": pm_distance,
            "n_sources": int(np.count_nonzero(mask)),
            "n_probability_ge_0p5": int(np.count_nonzero(prob >= 0.5)),
            "n_probability_ge_0p6": int(np.count_nonzero(prob >= 0.6)),
            "n_probability_ge_0p7": int(np.count_nonzero(prob >= 0.7)),
            "n_probability_ge_0p8": int(np.count_nonzero(prob >= 0.8)),
            "probability_mean": float(np.nanmean(prob)) if len(prob) else None,
            "probability_median": float(np.nanmedian(prob)) if len(prob) else None,
            "probability_hdbscan_mean": float(np.nanmean(prob_hdb)) if len(prob_hdb) else None,
            "probability_times_mean": float(np.nanmean(prob_times)) if len(prob_times) else None,
        }
        for column in ("pmra", "pmdec", "parallax", "Gmag", "ra", "dec", "fidelity_v2"):
            stats = _stats(table, mask, column)
            row[f"{column}_mean"] = stats["mean"]
            row[f"{column}_median"] = stats["median"]
            row[f"{column}_std"] = stats["std"]
        rows.append(row)

    return rows


def _persistence_rows(clust) -> list[dict[str, Any]]:
    labels = np.asarray(clust.clusterer.labels_, dtype=int)
    cluster_labels = sorted(int(label) for label in np.unique(labels) if label != -1)
    persistence = np.asarray(getattr(clust.clusterer, "cluster_persistence_", []), dtype=float)
    rows = []
    for index, value in enumerate(persistence):
        rows.append(
            {
                "cluster_index": index,
                "label_if_ordered": cluster_labels[index] if index < len(cluster_labels) else None,
                "cluster_persistence": float(value),
            }
        )
    return rows


def audit_radius(
    dill_path: Path, output_dir: Path, reference_pm: tuple[float, float]
) -> dict[str, Any]:
    radius = int(dill_path.name.split("_")[1])
    radius_dir = output_dir / str(radius)
    radius_dir.mkdir(parents=True, exist_ok=True)

    with dill_path.open("rb") as handle:
        clust = dill.load(handle)

    table = clust.data
    selected = clust.pseudoprobability_selected_ or {}
    final_labels = np.asarray(table["cluster_hdbscan"], dtype=int)
    condensed_tree = clust.clusterer.condensed_tree_.to_pandas()
    # legacy = what the published run did; tree = what the issue-#7 fix would have done.
    algorithm_selected_label = Clustering._cluster_label_for_size(
        final_labels, int(selected.get("desired_len", -1))
    )
    tree_selected_label = Clustering._cluster_label_from_tree(
        condensed_tree, final_labels, len(final_labels)
    )
    ngc_like_label, ngc_pm_distance = _ngc_like_label(table, final_labels, reference_pm)
    probability = _as_float_array(table["probability"])
    probability_times = _as_float_array(table["probability_times"])

    candidates = _candidate_rows(clust.pseudoprobability_results_ or [])
    sweep = _sweep_rows(
        clust.pseudoprobability_sweep_track_ or [], clust.pseudoprobability_results_ or []
    )
    label_rows = _final_label_rows(
        clust, reference_pm, algorithm_selected_label, tree_selected_label
    )
    persistence = _persistence_rows(clust)
    source_table = table.copy(copy_data=True)
    source_table["audit_radius_arcmin"] = radius
    source_table["is_algorithm_selected_branch"] = final_labels == algorithm_selected_label
    source_table["is_tree_selected_branch"] = final_labels == tree_selected_label
    source_table["is_ngc_like_branch"] = final_labels == ngc_like_label
    source_table["probability_ge_0p5"] = probability >= 0.5
    source_table["probability_ge_0p6"] = probability >= 0.6
    source_table["probability_ge_0p7"] = probability >= 0.7
    source_table["probability_ge_0p8"] = probability >= 0.8

    _write_csv(radius_dir / "candidate_sweep_results.csv", candidates)
    _write_csv(radius_dir / "min_cluster_size_sweep_track.csv", sweep)
    _write_csv(radius_dir / "final_label_summary.csv", label_rows)
    _write_csv(radius_dir / "final_cluster_persistence.csv", persistence)
    source_table.write(
        radius_dir / "final_sources_with_labels.ecsv", format="ascii.ecsv", overwrite=True
    )

    tree = condensed_tree
    tree.to_csv(radius_dir / "final_condensed_tree.csv", index=False)

    best_params = dict(clust.best_params_ or {})
    best_params.pop("gen_min_span_tree", None)
    summary = {
        "radius_arcmin": radius,
        "dill_path": dill_path,
        "n_sources": len(table),
        "n_bad_sources": None if clust.bad_data is None else len(clust.bad_data),
        "best_params": best_params,
        "best_score_lambda_value": clust.best_score_,
        "selected": {
            "min_cluster_size": selected.get("min_cluster_size"),
            "desired_len": selected.get("desired_len"),
            "algorithm_selected_final_label": algorithm_selected_label,
            "tree_selected_final_label": tree_selected_label,
            "ngc_like_final_label": ngc_like_label,
            "ngc_like_pm_distance_to_reference": ngc_pm_distance,
            "reference_pm": list(reference_pm),
            "lambda_value": selected.get("lambda_value"),
            "relative_validity": selected.get("relative_validity"),
            "branch_selection_note": (
                "The HDBSCAN sweep-selected branch differs from the NGC-like branch. "
                "Inspect final_label_summary.csv before interpreting membership."
                if algorithm_selected_label != ngc_like_label
                else "The HDBSCAN sweep-selected branch matches the NGC-like branch."
            ),
            "selector_fix_note": (
                "Issue #7: the legacy size-matching selector and the condensed-tree "
                "selector disagree on this run. `algorithm_selected_final_label` is what "
                "the published run reported; `tree_selected_final_label` is what the "
                "fixed selector returns."
                if algorithm_selected_label != tree_selected_label
                else "Issue #7: both selectors return the same label on this run."
            ),
        },
        "stored_state_limits": {
            "full_labels_matrix_saved": False,
            "note": "The current Clustering object stores probability_times and sweep summaries, not every source label in every sweep iteration.",
        },
        "counts": {
            "n_candidate_sweep_results": len(candidates),
            "n_sweep_track_rows": len(sweep),
            "n_final_labels_including_noise": len(label_rows),
            "n_condensed_tree_rows": int(len(tree)),
            "probability_times_unique_count": int(len(np.unique(probability_times))),
            "probability_ge_0p5_all_labels": int(np.count_nonzero(probability >= 0.5)),
            "probability_ge_0p6_all_labels": int(np.count_nonzero(probability >= 0.6)),
        },
        "outputs": {
            "candidate_sweep_results": radius_dir / "candidate_sweep_results.csv",
            "min_cluster_size_sweep_track": radius_dir / "min_cluster_size_sweep_track.csv",
            "final_label_summary": radius_dir / "final_label_summary.csv",
            "final_cluster_persistence": radius_dir / "final_cluster_persistence.csv",
            "final_condensed_tree": radius_dir / "final_condensed_tree.csv",
            "final_sources_with_labels": radius_dir / "final_sources_with_labels.ecsv",
        },
    }
    _write_json(radius_dir / "audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--radii", nargs="+", type=int, default=[40, 50, 60, 70])
    parser.add_argument("--reference-pmra", type=float, default=DEFAULT_REFERENCE_PM[0])
    parser.add_argument("--reference-pmdec", type=float, default=DEFAULT_REFERENCE_PM[1])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_pm = (args.reference_pmra, args.reference_pmdec)
    summaries = []
    for radius in args.radii:
        dill_path = args.input_dir / f"ngc6383_{radius}_paperfaithful.dill"
        if not dill_path.exists():
            raise FileNotFoundError(dill_path)
        summaries.append(audit_radius(dill_path, args.output_dir, reference_pm))

    combined_path = args.output_dir / "clustering_audit_summary.json"
    _write_json(combined_path, summaries)
    print(f"Wrote {combined_path}")
    print(
        "radius candidates sweep_rows final_labels condensed_tree_rows "
        "legacy_label tree_label ngc_label selected_mcs"
    )
    for item in summaries:
        print(
            item["radius_arcmin"],
            item["counts"]["n_candidate_sweep_results"],
            item["counts"]["n_sweep_track_rows"],
            item["counts"]["n_final_labels_including_noise"],
            item["counts"]["n_condensed_tree_rows"],
            item["selected"]["algorithm_selected_final_label"],
            item["selected"]["tree_selected_final_label"],
            item["selected"]["ngc_like_final_label"],
            item["selected"]["min_cluster_size"],
        )


if __name__ == "__main__":
    main()
