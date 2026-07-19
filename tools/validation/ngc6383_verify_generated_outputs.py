"""Verify consistency of generated NGC 6383 robustness and audit outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import Table


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NGC_ROOT = PROJECT_ROOT / "data" / "test" / "NGC6383"
DEFAULT_RADIUS_DIR = NGC_ROOT / "comments_paper" / "radius_robustness" / "generated"
DEFAULT_AUDIT_DIR = NGC_ROOT / "comments_paper" / "clustering_audit" / "generated"


def _jsonable(value: Any) -> Any:
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


def _load_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _load_label_summary(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bool_sum(table: Table, column: str) -> int:
    return int(np.count_nonzero(np.asarray(table[column], dtype=bool)))


def _assert_equal(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        errors.append(f"{label}: actual={actual!r} expected={expected!r}")


def verify_radius(radius_summary: dict[str, Any], audit_summary: dict[str, Any], audit_dir: Path) -> dict[str, Any]:
    radius = int(radius_summary["radius_arcmin"])
    errors: list[str] = []

    _assert_equal(errors, f"{radius} audit radius", int(audit_summary["radius_arcmin"]), radius)
    _assert_equal(errors, f"{radius} n_sources", int(audit_summary["n_sources"]), int(radius_summary["preprocess_good"]))
    _assert_equal(errors, f"{radius} n_bad_sources", int(audit_summary["n_bad_sources"]), int(radius_summary["preprocess_bad"]))
    _assert_equal(
        errors,
        f"{radius} selected min_cluster_size",
        int(audit_summary["selected"]["min_cluster_size"]),
        int(radius_summary["best_mcs"]),
    )
    _assert_equal(
        errors,
        f"{radius} algorithm label",
        int(audit_summary["selected"]["algorithm_selected_final_label"]),
        int(radius_summary["algorithm_selected_label"]),
    )
    _assert_equal(
        errors,
        f"{radius} NGC-like label",
        int(audit_summary["selected"]["ngc_like_final_label"]),
        int(radius_summary["ngc_like_label"]),
    )

    radius_audit_dir = audit_dir / str(radius)
    labels = _load_label_summary(radius_audit_dir / "final_label_summary.csv")
    ngc_rows = [row for row in labels if row["is_ngc_like_branch"] == "True"]
    _assert_equal(errors, f"{radius} one NGC-like label row", len(ngc_rows), 1)
    if ngc_rows:
        ngc_row = ngc_rows[0]
        _assert_equal(errors, f"{radius} NGC branch size", int(ngc_row["n_sources"]), int(radius_summary["branch_n"]))
        _assert_equal(
            errors,
            f"{radius} NGC branch probability >= 0.5",
            int(ngc_row["n_probability_ge_0p5"]),
            int(radius_summary["branch_p05"]),
        )

    sources = Table.read(radius_audit_dir / "final_sources_with_labels.ecsv", format="ascii.ecsv")
    _assert_equal(errors, f"{radius} source table length", len(sources), int(radius_summary["preprocess_good"]))
    _assert_equal(errors, f"{radius} source NGC branch size", _bool_sum(sources, "is_ngc_like_branch"), int(radius_summary["branch_n"]))
    _assert_equal(
        errors,
        f"{radius} source postclip p>=0.5",
        _bool_sum(sources, "paper_reference_postclip_p05"),
        int(radius_summary["postclip_p05"]),
    )
    _assert_equal(
        errors,
        f"{radius} source reference p>=0.6",
        _bool_sum(sources, "paper_reference_p06"),
        int(radius_summary["reference_p06"]),
    )
    reference_mask = np.asarray(sources["paper_reference_p06"], dtype=bool)
    gmag = np.asarray(sources["Gmag"], dtype=float)
    _assert_equal(
        errors,
        f"{radius} source reference p>=0.6 and G<19",
        int(np.count_nonzero(reference_mask & (gmag < 19))),
        int(radius_summary["reference_p06_g_lt_19"]),
    )

    return {
        "radius_arcmin": radius,
        "status": "ok" if not errors else "failed",
        "errors": errors,
        "algorithm_selected_label": radius_summary["algorithm_selected_label"],
        "ngc_like_label": radius_summary["ngc_like_label"],
        "reference_p06": radius_summary["reference_p06"],
        "reference_p06_g_lt_19": radius_summary["reference_p06_g_lt_19"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-dir", type=Path, default=DEFAULT_RADIUS_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    args = parser.parse_args()

    radius_summaries = _load_json(args.radius_dir / "paperfaithful_radius_summary.json")
    audit_summaries = _load_json(args.audit_dir / "clustering_audit_summary.json")
    audit_by_radius = {int(item["radius_arcmin"]): item for item in audit_summaries}

    report = []
    for item in radius_summaries:
        radius = int(item["radius_arcmin"])
        if radius not in audit_by_radius:
            report.append({"radius_arcmin": radius, "status": "failed", "errors": ["missing audit summary"]})
            continue
        report.append(verify_radius(item, audit_by_radius[radius], args.audit_dir))

    output_path = args.audit_dir / "verification_report.json"
    output_path.write_text(json.dumps(_jsonable(report), indent=2, allow_nan=False) + "\n", encoding="utf-8")

    failures = [item for item in report if item["status"] != "ok"]
    for item in report:
        print(
            item["radius_arcmin"],
            item["status"],
            "algorithm_label=",
            item.get("algorithm_selected_label"),
            "ngc_label=",
            item.get("ngc_like_label"),
            "reference_p06=",
            item.get("reference_p06"),
        )
    if failures:
        raise SystemExit(f"Verification failed for {len(failures)} radius run(s). See {output_path}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
