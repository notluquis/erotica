"""Generate the paper-facing NGC 6383 member table for CDS preparation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.table import Table, join


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMENTS_ROOT = PROJECT_ROOT / "data" / "test" / "NGC6383" / "comments_paper"
DEFAULT_INPUT = COMMENTS_ROOT / "radius_robustness" / "generated" / "40" / "paperfaithful_with_clip_flags.ecsv"
DEFAULT_SAGITTA = COMMENTS_ROOT / "cluster_data.ecsv"
DEFAULT_OUTPUT_DIR = COMMENTS_ROOT / "cds"
DEFAULT_FINAL_OUTPUT_DIR = COMMENTS_ROOT / "cds_final"


DESCRIPTIONS = {
    "GaiaDR3": ("---", "Gaia DR3 source identifier"),
    "RAdeg": ("deg", "Right ascension, ICRS, Gaia DR3"),
    "DEdeg": ("deg", "Declination, ICRS, Gaia DR3"),
    "Plx": ("mas", "Zero-point corrected Gaia DR3 parallax"),
    "e_Plx": ("mas", "Gaia DR3 parallax uncertainty"),
    "pmRA": ("mas/yr", "Corrected proper motion in right ascension, mu_alpha*"),
    "e_pmRA": ("mas/yr", "Gaia DR3 proper-motion uncertainty in right ascension"),
    "pmDE": ("mas/yr", "Corrected proper motion in declination"),
    "e_pmDE": ("mas/yr", "Gaia DR3 proper-motion uncertainty in declination"),
    "Gmag": ("mag", "Gaia DR3 G-band magnitude"),
    "BPmag": ("mag", "Gaia DR3 GBP magnitude"),
    "RPmag": ("mag", "Gaia DR3 GRP magnitude"),
    "Jmag": ("mag", "2MASS J magnitude"),
    "Hmag": ("mag", "2MASS H magnitude"),
    "Ksmag": ("mag", "2MASS Ks magnitude"),
    "Fidelity": ("---", "Rybizki et al. astrometric fidelity v2"),
    "pHDBSCAN": ("---", "Native HDBSCAN membership probability for the selected branch"),
    "pFreq": ("---", "Fraction of HDBSCAN sweeps in which the source was recovered"),
    "pMember": ("---", "Composite membership pseudo-probability pHDBSCAN*pFreq"),
    "Ref": ("---", "Reference-sample flag: 1 if pMember >= 0.6 after clipping"),
    "PMSProb": ("---", "Sagitta PMS probability when available"),
    "AvSag": ("mag", "Sagitta visual extinction estimate when available"),
    "logAgeSag": ("log(yr)", "Sagitta logarithmic age estimate when available"),
}

CDS_FORMATS = {
    "GaiaDR3": ("I19", 19, 0),
    "RAdeg": ("F12.8", 12, 8),
    "DEdeg": ("F13.8", 13, 8),
    "Plx": ("F8.4", 8, 4),
    "e_Plx": ("F8.4", 8, 4),
    "pmRA": ("F8.4", 8, 4),
    "e_pmRA": ("F8.4", 8, 4),
    "pmDE": ("F8.4", 8, 4),
    "e_pmDE": ("F8.4", 8, 4),
    "Gmag": ("F8.4", 8, 4),
    "BPmag": ("F8.4", 8, 4),
    "RPmag": ("F8.4", 8, 4),
    "Jmag": ("F8.4", 8, 4),
    "Hmag": ("F8.4", 8, 4),
    "Ksmag": ("F8.4", 8, 4),
    "Fidelity": ("F7.4", 7, 4),
    "pHDBSCAN": ("F7.4", 7, 4),
    "pFreq": ("F7.4", 7, 4),
    "pMember": ("F7.4", 7, 4),
    "Ref": ("I1", 1, 0),
    "PMSProb": ("F7.4", 7, 4),
    "AvSag": ("F7.4", 7, 4),
    "logAgeSag": ("F8.4", 8, 4),
}


def _float_column(table: Table, name: str) -> np.ndarray:
    return np.asarray(table[name], dtype=float)


def _optional_float(table: Table, name: str, length: int) -> np.ndarray:
    if name not in table.colnames:
        return np.full(length, np.nan)
    return _float_column(table, name)


def _build_member_table(source: Table, sagitta: Table) -> Table:
    sagitta_cols = [col for col in ("source_id", "pms_sagitta", "av_sagitta", "age") if col in sagitta.colnames]
    merged = join(source, sagitta[sagitta_cols], keys="source_id", join_type="left")
    selected = np.asarray(merged["paper_reference_postclip_p05"], dtype=bool)
    data = merged[selected].copy(copy_data=True)
    data.sort("source_id")

    out = Table()
    out["GaiaDR3"] = np.asarray(data["source_id"], dtype=np.int64)
    out["RAdeg"] = _float_column(data, "ra")
    out["DEdeg"] = _float_column(data, "dec")
    out["Plx"] = _float_column(data, "parallax")
    out["e_Plx"] = _float_column(data, "parallax_error")
    out["pmRA"] = _float_column(data, "pmra")
    out["e_pmRA"] = _float_column(data, "pmra_error")
    out["pmDE"] = _float_column(data, "pmdec")
    out["e_pmDE"] = _float_column(data, "pmdec_error")
    out["Gmag"] = _float_column(data, "Gmag")
    out["BPmag"] = _float_column(data, "G_BPmag")
    out["RPmag"] = _float_column(data, "G_RPmag")
    out["Jmag"] = _float_column(data, "j_m")
    out["Hmag"] = _float_column(data, "h_m")
    out["Ksmag"] = _float_column(data, "ks_m")
    out["Fidelity"] = _float_column(data, "fidelity_v2")
    out["pHDBSCAN"] = _float_column(data, "probability_hdbscan")
    out["pFreq"] = _float_column(data, "probability_times")
    out["pMember"] = _float_column(data, "probability")
    out["Ref"] = np.asarray(data["paper_reference_p06"], dtype=bool).astype(int)
    out["PMSProb"] = _optional_float(data, "pms_sagitta", len(data))
    out["AvSag"] = _optional_float(data, "av_sagitta", len(data))
    out["logAgeSag"] = _optional_float(data, "age", len(data))

    out.meta["description"] = (
        "NGC 6383 likely members from the paper-faithful 40 arcmin workflow. "
        "Rows are the post-parallax-clipping candidates with pMember > 0.5; "
        "Ref=1 marks the pMember >= 0.6 reference sample."
    )
    return out


def _write_readme(table: Table, output_dir: Path) -> None:
    lines = [
        "NGC 6383 likely member table for CDS preparation",
        "================================================",
        "",
        "Source workflow:",
        "- Gaia DR3 cone search with the paper-faithful 40 arcmin radius.",
        "- PUMPS preprocessing, HDBSCAN in proper-motion space only.",
        "- Composite pseudo-probability pMember = pHDBSCAN * pFreq.",
        "- 2-sigma parallax clipping around the parallax histogram mode.",
        "",
        "Files:",
        "- ngc6383_members_cds.tsv: tab-separated ASCII staging table.",
        "- ngc6383_members_cds.ecsv: ECSV copy preserving metadata.",
        "",
        f"Rows: {len(table)} likely candidates with pMember > 0.5.",
        f"Reference sample: {int(np.count_nonzero(table['Ref']))} rows with pMember >= 0.6.",
        "",
        "Column descriptions:",
    ]
    for name in table.colnames:
        unit, desc = DESCRIPTIONS[name]
        lines.append(f"- {name:10s} [{unit}]: {desc}.")
    lines.extend(
        [
            "",
            "This is a staging product for the A&A revision. Before final CDS upload,",
            "run CDS/VizieR consistency checks and adapt the ReadMe to the final",
            "journal volume/page metadata after acceptance.",
            "",
        ]
    )
    (output_dir / "ReadMe").write_text("\n".join(lines), encoding="utf-8")


def _format_cds_value(value: object, width: int, precision: int) -> str:
    if isinstance(value, (np.integer, int)):
        return f"{int(value):>{width}d}"
    try:
        val = float(value)
    except (TypeError, ValueError):
        text = str(value)
        return text[:width].rjust(width)
    if not np.isfinite(val):
        return "...".rjust(width)
    return f"{val:>{width}.{precision}f}"


def _cds_column_positions(names: list[str]) -> list[tuple[int, int, str, str, str, str]]:
    positions = []
    start = 1
    for name in names:
        fmt, width, _precision = CDS_FORMATS[name]
        end = start + width - 1
        unit, desc = DESCRIPTIONS[name]
        positions.append((start, end, fmt, unit, name, desc))
        start = end + 2
    return positions


def _write_fixed_width_dat(table: Table, output_dir: Path) -> None:
    lines = []
    for row in table:
        fields = []
        for name in table.colnames:
            _fmt, width, precision = CDS_FORMATS[name]
            fields.append(_format_cds_value(row[name], width, precision))
        lines.append(" ".join(fields))
    (output_dir / "ngc6383_members.dat").write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_cds_readme(table: Table, output_dir: Path) -> None:
    byte_rows = _cds_column_positions(table.colnames)
    lines = [
        "J/A+A/xxxx/xxxx     NGC 6383 likely members from Gaia DR3 and 2MASS     Pulgar-Escobar et al.",
        "================================================================================",
        "Characterizing NGC 6383: a study of fundamental properties, pre-main sequence",
        "stars, mass segregation, and age using Gaia DR3 and 2MASS.",
        "    Pulgar-Escobar L.M., Henriquez-Salgado N.A., Mennickent R.E., Cerulo P.",
        "    <Astron. Astrophys., in revision, aa52082-24>",
        "================================================================================",
        "ADC_Keywords: Open clusters and associations; Proper motions; Parallaxes;",
        "              Photometry, infrared; Stars, pre-main sequence",
        "Keywords: open clusters and associations: individual: NGC 6383 -",
        "          stars: distances - techniques: photometric - parallaxes -",
        "          proper motions",
        "",
        "Abstract:",
        "  This table lists likely NGC 6383 candidates selected from Gaia DR3 and",
        "  2MASS with the paper-faithful 40 arcmin workflow. HDBSCAN was run in",
        "  the proper-motion plane only. The membership quantity pMember is a",
        "  composite pseudo-probability, pHDBSCAN*pFreq, not a calibrated Bayesian",
        "  posterior membership probability. Rows are post-parallax-clipping",
        "  candidates with pMember>0.5; Ref=1 marks the pMember>=0.6 reference",
        "  sample used for the main cluster parameters.",
        "",
        "Description:",
        "  The catalogue was produced from a Gaia DR3 cone search centred on NGC 6383,",
        "  cross-matched to 2MASS where available. Gaia parallaxes were corrected for",
        "  the DR3 zero point, and proper motions for bright-source frame rotation",
        "  where applicable. The preprocessing retained sources with astrometric",
        "  fidelity_v2>0.5. Missing photometric or Sagitta values are encoded as",
        "  '...' in the fixed-width table.",
        "",
        "File Summary:",
        "--------------------------------------------------------------------------------",
        " FileName              Lrecl  Records  Explanations",
        "--------------------------------------------------------------------------------",
        f" ngc6383_members.dat  {byte_rows[-1][1]:5d}  {len(table):7d}  Likely member candidates",
        "--------------------------------------------------------------------------------",
        "",
        "Byte-by-byte Description of file: ngc6383_members.dat",
        "--------------------------------------------------------------------------------",
        " Bytes Format Units   Label     Explanations",
        "--------------------------------------------------------------------------------",
    ]
    for start, end, fmt, unit, name, desc in byte_rows:
        byte_range = f"{start:3d}-{end:<3d}" if start != end else f"{start:3d}"
        lines.append(f" {byte_range:8s} {fmt:6s} {unit:7s} {name:9s} {desc}")
    lines.extend(
        [
            "--------------------------------------------------------------------------------",
            "Notes:",
            "  Ref=1 identifies the 254-source reference sample with pMember>=0.6 after",
            "  parallax-mode sigma clipping. Ref=0 rows are retained likely candidates",
            "  with 0.5<pMember<0.6.",
            "  PMSProb, AvSag, and logAgeSag are Sagitta outputs when available and are",
            "  not used as HDBSCAN clustering coordinates.",
            "",
            "Acknowledgements:",
            "  This table uses Gaia DR3 and 2MASS data products.",
            "================================================================================",
            "",
        ]
    )
    (output_dir / "ReadMe").write_text("\n".join(lines), encoding="ascii")


def _write_validation_manifest(table: Table, output_dir: Path) -> None:
    manifest = {
        "status": "prepared",
        "table": "ngc6383_members.dat",
        "rows": len(table),
        "reference_rows": int(np.count_nonzero(table["Ref"])),
        "selection": "post-parallax-clipping candidates with pMember > 0.5",
        "reference_selection": "Ref=1 if pMember >= 0.6",
        "missing_value": "...",
        "fixed_width_lrecl": _cds_column_positions(table.colnames)[-1][1],
        "columns": table.colnames,
    }
    (output_dir / "cds_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="ascii",
    )


def _write_cds_final(table: Table, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_fixed_width_dat(table, output_dir)
    table.write(output_dir / "ngc6383_members.ecsv", format="ascii.ecsv", overwrite=True)
    _write_cds_readme(table, output_dir)
    _write_validation_manifest(table, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sagitta", type=Path, default=DEFAULT_SAGITTA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--final-output-dir", type=Path, default=DEFAULT_FINAL_OUTPUT_DIR)
    parser.add_argument("--final", action="store_true", help="Also write fixed-width CDS-ready files.")
    args = parser.parse_args()

    source = Table.read(args.input, format="ascii.ecsv")
    sagitta = Table.read(args.sagitta, format="ascii.ecsv")
    table = _build_member_table(source, sagitta)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.write(args.output_dir / "ngc6383_members_cds.ecsv", format="ascii.ecsv", overwrite=True)
    table.write(args.output_dir / "ngc6383_members_cds.tsv", format="ascii.tab", overwrite=True)
    _write_readme(table, args.output_dir)
    if args.final:
        _write_cds_final(table, args.final_output_dir)

    print(f"Wrote {args.output_dir / 'ngc6383_members_cds.tsv'}")
    print(f"Wrote {args.output_dir / 'ngc6383_members_cds.ecsv'}")
    print(f"Wrote {args.output_dir / 'ReadMe'}")
    if args.final:
        print(f"Wrote {args.final_output_dir / 'ngc6383_members.dat'}")
        print(f"Wrote {args.final_output_dir / 'ngc6383_members.ecsv'}")
        print(f"Wrote {args.final_output_dir / 'ReadMe'}")
    print(f"rows={len(table)} reference_rows={int(np.count_nonzero(table['Ref']))}")


if __name__ == "__main__":
    main()
