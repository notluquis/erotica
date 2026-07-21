"""Run the paper-faithful NGC 6383 radius robustness clustering check.

This script keeps the A&A referee-response run reproducible outside notebooks.
It intentionally preserves the legacy pseudo-probability setup that reproduces
the submitted 40 arcmin, 254-member sample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="cosmic-mpl-"))

import dill
import numpy as np
from astropy.stats import sigma_clip


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NGC_ROOT = PROJECT_ROOT / "data" / "test" / "NGC6383"
DEFAULT_OUTPUT_DIR = NGC_ROOT / "comments_paper" / "radius_robustness" / "generated"
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
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _as_float_array(values) -> np.ndarray:
    if hasattr(values, "to_value"):
        return np.asarray(values.to_value(), dtype=float)
    quantity = getattr(values, "quantity", None)
    if quantity is not None and hasattr(quantity, "to_value"):
        return np.asarray(quantity.to_value(), dtype=float)
    return np.asarray(values, dtype=float)


def _column_values(table, column: str) -> np.ndarray:
    return _as_float_array(table[column])


def _nanmedian(table, column: str, mask: np.ndarray) -> float:
    return float(np.nanmedian(_as_float_array(table[column][mask])))


def _as_float_scalar(value) -> float:
    return float(_as_float_array(value).reshape(-1)[0])


def _unit_of(values):
    unit = getattr(values, "unit", None)
    if unit is not None:
        return unit
    quantity = getattr(values, "quantity", None)
    return getattr(quantity, "unit", None)


def _histogram_mode_for_sigma_clip(histogram_mode_func, values, axis=None):
    if axis is not None:
        mode = np.apply_along_axis(histogram_mode_func, axis, values)
    else:
        mode = histogram_mode_func(values)
    unit = _unit_of(values)
    return mode * unit if unit is not None else mode


def _source_for_radius(radius: int) -> Path:
    return NGC_ROOT / "data" / str(radius) / f"NGC_6383_{radius}-result.ecsv"


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": path,
        "size_bytes": stat.st_size,
        "sha256": digest.hexdigest(),
    }


def _ngc_like_label(table, labels: np.ndarray, reference_pm: tuple[float, float]) -> tuple[int, float]:
    """Select the final label nearest to the reference NGC 6383 proper motion."""
    probability = _column_values(table, "probability")
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
        raise RuntimeError("No non-noise label with probability >= 0.5 was available for NGC-like selection.")
    return best_label, best_distance


def _import_cosmic_runtime():
    try:
        from pumps.analysis import histogram_mode
        from pumps.core import Clustering
        from pumps.io import DataLoader
        from pumps.preprocess import DataPreprocessor
    except ModuleNotFoundError as exc:
        if exc.name == "zero_point":
            raise SystemExit(
                "Missing dependency 'zero_point'. Activate/install the PUMPS "
                "astronomy environment before running this Gaia preprocessing "
                "check."
            ) from exc
        raise

    return DataLoader, DataPreprocessor, Clustering, histogram_mode


def preprocess_source(path: Path):
    DataLoader, DataPreprocessor, _, _ = _import_cosmic_runtime()

    loader = DataLoader(path)
    data = loader.load_data(
        systems=["Gaia", "TMASS", "WISE"],
        include_distances=["geometric"],
        include_zp_cols=True,
        include_flux_errors=True,
        fidelity="fidelity_v2",
    )

    pre = DataPreprocessor(data)
    pre.rename_columns()
    pre.drop_invalid_sources()
    pre.fill_missing_values()
    pre.apply_zero_point_correction()
    pre.correct_proper_motion()
    pre.add_photometric_errors()
    good_data, bad_data = pre.filter_data(fidelity_threshold=0.5)
    metadata = {
        "input_file": _file_metadata(path),
        "loaded_rows": len(data),
        "loaded_columns": list(data.colnames),
        "loaded_column_count": len(data.colnames),
        "good_rows": len(good_data),
        "bad_rows": len(bad_data),
        "good_columns": list(good_data.colnames),
        "good_column_count": len(good_data.colnames),
        "bad_columns": list(bad_data.colnames),
        "bad_column_count": len(bad_data.colnames),
        "steps": [
            "DataLoader.load_data(systems=['Gaia', 'TMASS', 'WISE'], include_distances=['geometric'], include_zp_cols=True, include_flux_errors=True, fidelity='fidelity_v2')",
            "DataPreprocessor.rename_columns()",
            "DataPreprocessor.drop_invalid_sources()",
            "DataPreprocessor.fill_missing_values()",
            "DataPreprocessor.apply_zero_point_correction()",
            "DataPreprocessor.correct_proper_motion()",
            "DataPreprocessor.add_photometric_errors()",
            "DataPreprocessor.filter_data(fidelity_threshold=0.5)",
        ],
    }
    return good_data, bad_data, metadata


def run_radius(
    radius: int,
    output_dir: Path,
    *,
    write_full_tables: bool,
    write_cluster_object: bool,
    reference_pm: tuple[float, float],
) -> dict[str, Any]:
    _, _, Clustering, histogram_mode = _import_cosmic_runtime()

    source = _source_for_radius(radius)
    if not source.exists():
        raise FileNotFoundError(source)

    good_data, bad_data, preprocessing = preprocess_source(source)

    clust = Clustering(good_data, bad_data)
    clust.search_pseudoprobability(
        columns=["pmra", "pmdec"],
        min_cluster_size_samples=range(10, 300),
        probability_threshold=0.5,
        min_cluster_members=200,
        max_cluster_members=1000,
        select_cluster=False,
        hdbscan_kwargs={
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
            "match_reference_implementation": False,
            "core_dist_n_jobs": 1,
        },
    )

    table = clust.data.copy(copy_data=True)
    selected = clust.pseudoprobability_selected_ or {}
    labels = np.asarray(table["cluster_hdbscan"], dtype=int)
    algorithm_label = Clustering._cluster_label_for_size(labels, int(selected["desired_len"]))
    label, pm_dist = _ngc_like_label(table, labels, reference_pm)

    probability = _column_values(table, "probability")
    branch_mask = labels == label
    preclip_mask = branch_mask & (probability >= 0.5)

    def parallax_mode(values, axis=None):
        return _histogram_mode_for_sigma_clip(histogram_mode, values, axis=axis)

    _, clip_low, clip_high = sigma_clip(
        table["parallax"][preclip_mask],
        sigma=2,
        cenfunc=parallax_mode,
        stdfunc="std",
        return_bounds=True,
    )
    parallax = table["parallax"]
    clip_mask = (parallax >= clip_low) & (parallax <= clip_high)
    postclip_p05 = branch_mask & (probability >= 0.5) & clip_mask
    reference_p06 = branch_mask & (probability >= 0.6) & clip_mask

    paper_flags = {
        "paper_reference_branch": branch_mask,
        "paper_reference_preclip_p05": preclip_mask,
        "paper_reference_postclip_p05": postclip_p05,
        "paper_reference_p06": reference_p06,
    }
    for column, values in paper_flags.items():
        table[column] = values
        clust.data[column] = values

    threshold_counts = []
    for threshold in (0.5, 0.6, 0.7, 0.8):
        mask = branch_mask & (probability >= threshold) & clip_mask
        threshold_counts.append(
            {
                "threshold": threshold,
                "n_members": int(np.count_nonzero(mask)),
                "n_g_lt_19": int(np.count_nonzero(mask & (_column_values(table, "Gmag") < 19))),
                "n_g_le_18": int(np.count_nonzero(mask & (_column_values(table, "Gmag") <= 18))),
                "n_g_le_17": int(np.count_nonzero(mask & (_column_values(table, "Gmag") <= 17))),
            }
        )

    summary = {
        "radius_arcmin": radius,
        "source": source,
        "preprocess_good": len(good_data),
        "preprocess_bad": len(bad_data),
        "preprocessing": preprocessing,
        "n_runs": 290,
        "min_cluster_size_samples": [10, 299],
        "best_mcs": int(selected["min_cluster_size"]),
        "desired_len": int(selected["desired_len"]),
        "algorithm_selected_label": int(algorithm_label),
        "ngc_like_label": int(label),
        "label": int(label),
        "branch_selection_reference_pm": list(reference_pm),
        "branch_selection_pm_dist": float(pm_dist),
        "branch_selection_note": (
            "The HDBSCAN sweep selection and the NGC-like branch selection differ. "
            "The science summary uses the branch nearest to the 40 arcmin NGC 6383 reference proper motion."
            if int(algorithm_label) != int(label)
            else "The HDBSCAN sweep-selected branch matches the NGC-like branch."
        ),
        "branch_n": int(np.count_nonzero(branch_mask)),
        "branch_p05": int(np.count_nonzero(preclip_mask)),
        "pmra": _nanmedian(table, "pmra", reference_p06),
        "pmdec": _nanmedian(table, "pmdec", reference_p06),
        "clip_low": _as_float_scalar(clip_low),
        "clip_high": _as_float_scalar(clip_high),
        "preclip_p05": int(np.count_nonzero(preclip_mask)),
        "postclip_p05": int(np.count_nonzero(postclip_p05)),
        "reference_p06": int(np.count_nonzero(reference_p06)),
        "reference_p06_g_lt_19": int(np.count_nonzero(reference_p06 & (_column_values(table, "Gmag") < 19))),
        "threshold_counts": threshold_counts,
    }

    radius_dir = output_dir / str(radius)
    radius_dir.mkdir(parents=True, exist_ok=True)

    if write_cluster_object:
        dill_dir = output_dir / "dill"
        dill_dir.mkdir(parents=True, exist_ok=True)
        dill_path = dill_dir / f"ngc6383_{radius}_paperfaithful.dill"
        with dill_path.open("wb") as handle:
            dill.dump(clust, handle)
        summary["cluster_object"] = dill_path

    (radius_dir / "summary.json").write_text(
        json.dumps(_jsonable(summary), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    if write_full_tables:
        good_data.write(radius_dir / "preprocessed_good.ecsv", format="ascii.ecsv", overwrite=True)
        bad_data.write(radius_dir / "preprocessed_bad.ecsv", format="ascii.ecsv", overwrite=True)
        table.write(radius_dir / "paperfaithful_with_clip_flags.ecsv", format="ascii.ecsv", overwrite=True)
        table[reference_p06].write(radius_dir / "paperfaithful_reference_p06.ecsv", format="ascii.ecsv", overwrite=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radii", nargs="+", type=int, default=[40, 50, 60, 70])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-pmra", type=float, default=DEFAULT_REFERENCE_PM[0])
    parser.add_argument("--reference-pmdec", type=float, default=DEFAULT_REFERENCE_PM[1])
    parser.add_argument(
        "--no-write-dill",
        action="store_true",
        help="Do not persist the Clustering objects used by the audit script.",
    )
    parser.add_argument(
        "--write-full-tables",
        action="store_true",
        help="Write preprocessed, flagged, and final ECSV tables in addition to summaries.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference_pm = (args.reference_pmra, args.reference_pmdec)
    summaries = [
        run_radius(
            radius,
            args.output_dir,
            write_full_tables=args.write_full_tables,
            write_cluster_object=not args.no_write_dill,
            reference_pm=reference_pm,
        )
        for radius in args.radii
    ]
    summary_path = args.output_dir / "paperfaithful_radius_summary.json"
    summary_path.write_text(
        json.dumps(_jsonable(summaries), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print("radius good bad best_mcs algorithm_label ngc_label branch postclip_p05 reference_p06 reference_p06_g_lt_19")
    for item in summaries:
        print(
            item["radius_arcmin"],
            item["preprocess_good"],
            item["preprocess_bad"],
            item["best_mcs"],
            item["algorithm_selected_label"],
            item["ngc_like_label"],
            item["branch_n"],
            item["postclip_p05"],
            item["reference_p06"],
            item["reference_p06_g_lt_19"],
        )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
