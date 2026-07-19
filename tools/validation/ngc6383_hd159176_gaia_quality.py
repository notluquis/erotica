"""Compare HD 159176 Gaia DR3 quality metrics against NGC 6383 candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table, join, vstack
from astroquery.gaia import Gaia


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMENTS_ROOT = PROJECT_ROOT / "data" / "test" / "NGC6383" / "comments_paper"
DEFAULT_MEMBERS = COMMENTS_ROOT / "cds_final" / "ngc6383_members.ecsv"
DEFAULT_OUTPUT_DIR = COMMENTS_ROOT / "hd159176_gaia_quality"

HD159176_SOURCE_ID = 4054618559611164288
QUALITY_COLUMNS = [
    "source_id",
    "designation",
    "ra",
    "dec",
    "phot_g_mean_mag",
    "bp_rp",
    "parallax",
    "parallax_error",
    "pmra",
    "pmra_error",
    "pmdec",
    "pmdec_error",
    "ruwe",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "visibility_periods_used",
    "duplicated_source",
    "astrometric_params_solved",
    "astrometric_gof_al",
    "astrometric_chi2_al",
    "astrometric_n_good_obs_al",
    "astrometric_n_bad_obs_al",
    "astrometric_sigma5d_max",
    "ipd_gof_harmonic_amplitude",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "phot_bp_rp_excess_factor",
]

METRICS = [
    "phot_g_mean_mag",
    "ruwe",
    "astrometric_excess_noise",
    "astrometric_excess_noise_sig",
    "visibility_periods_used",
    "astrometric_gof_al",
    "astrometric_chi2_al",
    "astrometric_n_good_obs_al",
    "astrometric_n_bad_obs_al",
    "astrometric_sigma5d_max",
    "ipd_gof_harmonic_amplitude",
    "ipd_frac_multi_peak",
    "ipd_frac_odd_win",
    "phot_bp_rp_excess_factor",
]


def _query_gaia_source(source_ids: list[int], chunk_size: int = 100) -> Table:
    tables: list[Table] = []
    columns = ", ".join(QUALITY_COLUMNS)
    for start in range(0, len(source_ids), chunk_size):
        ids = source_ids[start : start + chunk_size]
        query = (
            f"SELECT {columns} "
            "FROM gaiadr3.gaia_source "
            f"WHERE source_id IN ({','.join(str(int(x)) for x in ids)})"
        )
        job = Gaia.launch_job_async(query, dump_to_file=False)
        tables.append(job.get_results())
    return vstack(tables, metadata_conflicts="silent")


def _member_context(path: Path) -> Table:
    table = Table.read(path)
    out = Table()
    out["source_id"] = np.asarray(table["GaiaDR3"], dtype=np.int64)
    out["p_member"] = np.asarray(table["pMember"], dtype=float)
    out["is_reference_sample"] = np.asarray(table["Ref"], dtype=int).astype(bool)
    out["fidelity_v2"] = np.asarray(table["Fidelity"], dtype=float)
    out["pms_probability"] = np.asarray(table["PMSProb"], dtype=float)
    return out


def _percentile_of_score(values: np.ndarray, score: float) -> float | None:
    clean = values[np.isfinite(values)]
    if len(clean) == 0 or not np.isfinite(score):
        return None
    below = np.count_nonzero(clean < score)
    equal = np.count_nonzero(clean == score)
    return float(100.0 * (below + 0.5 * equal) / len(clean))


def _metric_summary(values: np.ndarray, hd_value: float) -> dict[str, float | int | None]:
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return {
            "n": 0,
            "median": None,
            "mean": None,
            "std": None,
            "p05": None,
            "p16": None,
            "p84": None,
            "p95": None,
            "min": None,
            "max": None,
            "hd_percentile": None,
            "hd_fraction_le": None,
            "hd_fraction_ge": None,
        }
    percentiles = np.percentile(clean, [5, 16, 84, 95])
    return {
        "n": int(len(clean)),
        "median": float(np.median(clean)),
        "mean": float(np.mean(clean)),
        "std": float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0,
        "p05": float(percentiles[0]),
        "p16": float(percentiles[1]),
        "p84": float(percentiles[2]),
        "p95": float(percentiles[3]),
        "min": float(np.min(clean)),
        "max": float(np.max(clean)),
        "hd_percentile": _percentile_of_score(clean, hd_value),
        "hd_fraction_le": float(100.0 * np.count_nonzero(clean <= hd_value) / len(clean)) if np.isfinite(hd_value) else None,
        "hd_fraction_ge": float(100.0 * np.count_nonzero(clean >= hd_value) / len(clean)) if np.isfinite(hd_value) else None,
    }


def _summaries(merged: Table, hd_row: Table) -> dict[str, object]:
    hd = {name: hd_row[name][0] for name in hd_row.colnames}
    likely = np.asarray(merged["is_ngc_candidate"], dtype=bool)
    reference = likely & np.asarray(merged["is_reference_sample"], dtype=bool)
    bright = likely & (np.asarray(merged["phot_g_mean_mag"], dtype=float) < 13.0)
    groups = {
        "likely_candidates_p_gt_0p5": likely,
        "reference_members_p_ge_0p6": reference,
        "bright_candidates_g_lt_13": bright,
    }
    result: dict[str, object] = {
        "hd159176": {
            name: (bool(value) if isinstance(value, (np.bool_, bool)) else float(value) if isinstance(value, (np.floating, float)) and np.isfinite(value) else int(value) if isinstance(value, (np.integer, int)) else str(value))
            for name, value in hd.items()
        },
        "groups": {},
    }
    for group_name, mask in groups.items():
        group = merged[mask]
        group_summary: dict[str, object] = {"n": int(len(group)), "metrics": {}}
        for metric in METRICS:
            group_summary["metrics"][metric] = _metric_summary(
                np.asarray(group[metric], dtype=float),
                float(hd[metric]),
            )
        group_summary["duplicated_source_true"] = int(np.count_nonzero(np.asarray(group["duplicated_source"], dtype=bool)))
        group_summary["five_parameter_solutions"] = int(np.count_nonzero(np.asarray(group["astrometric_params_solved"], dtype=int) == 31))
        result["groups"][group_name] = group_summary
    return result


def _write_plot(merged: Table, hd_row: Table, output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    likely = np.asarray(merged["is_ngc_candidate"], dtype=bool)
    reference = likely & np.asarray(merged["is_reference_sample"], dtype=bool)
    hd = hd_row[0]

    metrics = [
        ("ruwe", "RUWE"),
        ("astrometric_excess_noise_sig", "excess-noise significance"),
        ("visibility_periods_used", "visibility periods used"),
        ("ipd_frac_multi_peak", "IPD multi-peak fraction [%]"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics, strict=True):
        vals = np.asarray(merged[metric][reference], dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=24, color="#5b8db8", alpha=0.75, label="NGC 6383 ref.")
        ax.axvline(float(hd[metric]), color="#b83232", linewidth=2, label="HD 159176")
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
    fig.suptitle("HD 159176 vs NGC 6383 Gaia DR3 astrometric-quality metrics")
    fig.savefig(output_dir / "hd159176_gaia_quality_comparison.pdf")
    fig.savefig(output_dir / "hd159176_gaia_quality_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter(
        np.asarray(merged["phot_g_mean_mag"][likely], dtype=float),
        np.asarray(merged["ipd_frac_multi_peak"][likely], dtype=float),
        s=24,
        color="#4a7c59",
        alpha=0.75,
        label="likely candidates",
    )
    ax.scatter(
        [float(hd["phot_g_mean_mag"])],
        [float(hd["ipd_frac_multi_peak"])],
        marker="*",
        s=220,
        color="#b83232",
        label="HD 159176",
        zorder=10,
    )
    ax.invert_xaxis()
    ax.set_xlabel("Gaia DR3 G [mag]")
    ax.set_ylabel("IPD multi-peak fraction [%]")
    ax.legend()
    fig.savefig(output_dir / "hd159176_ipd_vs_gmag.pdf")
    fig.savefig(output_dir / "hd159176_ipd_vs_gmag.png", dpi=180)
    plt.close(fig)


def _write_report(summary: dict[str, object], output_dir: Path) -> None:
    hd = summary["hd159176"]
    groups = summary["groups"]
    ref = groups["reference_members_p_ge_0p6"]
    bright = groups["bright_candidates_g_lt_13"]

    def metric_line(metric: str) -> str:
        item = ref["metrics"][metric]
        tail = min(float(item["hd_fraction_le"]), float(item["hd_fraction_ge"]))
        return (
            f"| `{metric}` | {hd[metric]:.6g} | {item['median']:.6g} | "
            f"{item['p16']:.6g}-{item['p84']:.6g} | {item['hd_percentile']:.1f} | "
            f"{tail:.1f} |"
        )

    lines = [
        "# HD 159176 Gaia DR3 quality comparison",
        "",
        "This report compares the Gaia DR3 astrometric-quality indicators of HD 159176",
        "against the NGC 6383 likely-candidate catalogue used in the A&A revision.",
        "",
        "## Main result",
        "",
        f"HD 159176 is much brighter than the member sample (`G={hd['phot_g_mean_mag']:.3f}` mag),",
        "so the comparison must be interpreted as a quality-context check rather than",
        "a direct like-for-like member test. Its `ruwe` is not alarming, but its",
        "`astrometric_excess_noise_sig` and `ipd_frac_multi_peak` are extreme relative",
        "to the NGC 6383 reference members and are physically plausible warning signs",
        "for an O+O binary outside the magnitude range of the Gaia-selected reference",
        "sample.",
        "",
        "The percentile column below uses a midrank convention for tied values. The",
        "last column reports the smaller of the lower-tail and upper-tail fractions,",
        "which avoids over-interpreting metrics where most reference stars have the",
        "same value, such as `ipd_frac_odd_win=0`.",
        "",
        "## HD 159176 values",
        "",
        "| quantity | value |",
        "| --- | ---: |",
        f"| `source_id` | {int(hd['source_id'])} |",
        f"| `G` | {hd['phot_g_mean_mag']:.6f} |",
        f"| `parallax` | {hd['parallax']:.6f} +/- {hd['parallax_error']:.6f} mas |",
        f"| `pmra` | {hd['pmra']:.6f} +/- {hd['pmra_error']:.6f} mas/yr |",
        f"| `pmdec` | {hd['pmdec']:.6f} +/- {hd['pmdec_error']:.6f} mas/yr |",
        f"| `ruwe` | {hd['ruwe']:.6f} |",
        f"| `astrometric_excess_noise` | {hd['astrometric_excess_noise']:.6f} mas |",
        f"| `astrometric_excess_noise_sig` | {hd['astrometric_excess_noise_sig']:.6f} |",
        f"| `visibility_periods_used` | {int(hd['visibility_periods_used'])} |",
        f"| `duplicated_source` | {hd['duplicated_source']} |",
        f"| `astrometric_params_solved` | {int(hd['astrometric_params_solved'])} |",
        f"| `ipd_frac_multi_peak` | {hd['ipd_frac_multi_peak']:.6f} |",
        f"| `ipd_frac_odd_win` | {hd['ipd_frac_odd_win']:.6f} |",
        "",
        "## Against the 254-source reference sample",
        "",
        "| metric | HD 159176 | ref median | ref p16-p84 | HD midrank percentile | min tail [%] |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
        metric_line("ruwe"),
        metric_line("astrometric_excess_noise"),
        metric_line("astrometric_excess_noise_sig"),
        metric_line("visibility_periods_used"),
        metric_line("astrometric_gof_al"),
        metric_line("ipd_gof_harmonic_amplitude"),
        metric_line("ipd_frac_multi_peak"),
        metric_line("ipd_frac_odd_win"),
        "",
        "## Magnitude-aware caveat",
        "",
        f"The candidate catalogue contains {bright['n']} likely candidates with `G<13`.",
        "That subset is still much fainter than HD 159176 in most cases, so it does",
        "not remove the bright-source caveat. It only shows that the paper member",
        "sample does not provide a well-populated control set at `G~5.7`.",
        "",
        "## Recommended manuscript interpretation",
        "",
        "Keep the current HD 159176 membership statement conservative. The Gaia DR3",
        "declination proper-motion offset remains the main reason not to use HD 159176",
        "as a secure member, but the quality diagnostics show that the astrometric",
        "solution is not clean enough to make an overconfident claim from Gaia alone.",
        "Do not argue from RUWE alone.",
        "",
        "## Generated files",
        "",
        "- `ngc6383_members_gaia_quality.csv`",
        "- `ngc6383_members_gaia_quality.ecsv`",
        "- `hd159176_gaia_quality_summary.json`",
        "- `hd159176_gaia_quality_comparison.pdf`",
        "- `hd159176_ipd_vs_gmag.pdf`",
        "",
    ]
    (output_dir / "HD159176_GAIA_QUALITY_COMPARISON.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    context = _member_context(DEFAULT_MEMBERS)
    ids = sorted(set(int(x) for x in context["source_id"]) | {HD159176_SOURCE_ID})
    gaia = _query_gaia_source(ids)
    gaia.write(output_dir / "gaia_quality_raw_query.ecsv", overwrite=True)
    gaia.write(output_dir / "gaia_quality_raw_query.csv", overwrite=True)

    merged = join(gaia, context, keys="source_id", join_type="left")
    merged["is_ngc_candidate"] = np.isin(np.asarray(merged["source_id"], dtype=np.int64), np.asarray(context["source_id"], dtype=np.int64))
    for col in ("p_member", "fidelity_v2", "pms_probability"):
        if col in merged.colnames:
            merged[col] = np.where(np.asarray(merged["is_ngc_candidate"], dtype=bool), merged[col], np.nan)
    if "is_reference_sample" in merged.colnames:
        merged["is_reference_sample"] = np.asarray(merged["is_reference_sample"].filled(False), dtype=bool)
    else:
        merged["is_reference_sample"] = np.zeros(len(merged), dtype=bool)

    hd_row = merged[np.asarray(merged["source_id"], dtype=np.int64) == HD159176_SOURCE_ID]
    if len(hd_row) != 1:
        raise RuntimeError("Expected exactly one HD 159176 Gaia row")

    members = merged[np.asarray(merged["is_ngc_candidate"], dtype=bool)]
    members.write(output_dir / "ngc6383_members_gaia_quality.ecsv", overwrite=True)
    members.write(output_dir / "ngc6383_members_gaia_quality.csv", overwrite=True)
    hd_row.write(output_dir / "hd159176_gaiadr3_quality.csv", overwrite=True)
    hd_row.write(output_dir / "hd159176_gaiadr3_quality.ecsv", overwrite=True)

    summary = _summaries(merged, hd_row)
    (output_dir / "hd159176_gaia_quality_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_plot(merged, hd_row, output_dir)
    _write_report(summary, output_dir)
    print(f"Wrote Gaia-quality comparison to {output_dir}")


if __name__ == "__main__":
    main()
