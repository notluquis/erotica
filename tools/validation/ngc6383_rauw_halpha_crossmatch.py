"""Cross-match NGC 6383 members with the Rauw et al. (2010) H-alpha catalogue."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

home_astropy_cache = Path.home() / ".astropy" / "cache"
if not home_astropy_cache.exists() or not os.access(home_astropy_cache, os.W_OK):
    os.environ["HOME"] = tempfile.gettempdir()
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "cosmic-cache"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cosmic-mpl"))

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, join
from astroquery.vizier import Vizier
from scipy.interpolate import UnivariateSpline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMMENTS_ROOT = PROJECT_ROOT / "data" / "test" / "NGC6383" / "comments_paper"
DEFAULT_MEMBERS = COMMENTS_ROOT / "radius_robustness" / "generated" / "40" / "paperfaithful_with_clip_flags.ecsv"
DEFAULT_SAGITTA = COMMENTS_ROOT / "cluster_data.ecsv"
DEFAULT_OUTPUT_DIR = COMMENTS_ROOT / "rauw_halpha"
RAUW_CATALOG = "J/A+A/511/A25/table2"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_rauw_table() -> Table:
    Vizier.ROW_LIMIT = -1
    return Vizier.get_catalogs(RAUW_CATALOG)[0]


def _with_sagitta_flags(member_table: Table, sagitta_table: Table) -> Table:
    cols = [col for col in ("source_id", "pms_sagitta", "av_sagitta", "age") if col in sagitta_table.colnames]
    return join(member_table, sagitta_table[cols], keys="source_id", join_type="left")


def _classify_halpha(rauw: Table) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify Rauw H-alpha excess using a reproducible empirical ridge.

    Rauw et al. (2010) define candidates and emitters by offsets of
    0.12--0.24 mag and >0.24 mag in Rc-Halpha above the main-sequence
    relation. The VizieR table provides the photometry but not a published
    source-by-source emitter flag, so this is not a recovery of the original
    emitter list. We estimate the non-emitter locus from the catalogue itself
    with a smoothed running median and apply the published offset thresholds
    only as a diagnostic.
    """

    vi = np.asarray(rauw["V-Ic"], dtype=float)
    ha = np.asarray(rauw["Rc-Ha"], dtype=float)
    good = (
        np.isfinite(vi)
        & np.isfinite(ha)
        & (np.asarray(rauw["e_Vmag"], dtype=float) < 0.15)
        & (np.asarray(rauw["e_Rc-Ha"], dtype=float) < 0.21)
    )

    bins = np.linspace(np.nanpercentile(vi[good], 1), np.nanpercentile(vi[good], 99), 30)
    centers: list[float] = []
    medians: list[float] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = good & (vi >= lo) & (vi < hi)
        if np.count_nonzero(mask) >= 20:
            centers.append(float((lo + hi) / 2))
            medians.append(float(np.nanmedian(ha[mask])))
    ridge = UnivariateSpline(np.asarray(centers), np.asarray(medians), s=0.001)
    delta = ha - ridge(vi)

    halpha_class = np.full(len(rauw), "none", dtype=object)
    halpha_class[good & (delta >= 0.12) & (delta < 0.24)] = "candidate"
    halpha_class[good & (delta >= 0.24)] = "emitter"
    return halpha_class, delta, good


def _summarize(classes: np.ndarray, selection: np.ndarray) -> dict[str, int]:
    return {
        "n": int(np.count_nonzero(selection)),
        "candidate": int(np.count_nonzero(selection & (classes == "candidate"))),
        "emitter": int(np.count_nonzero(selection & (classes == "emitter"))),
        "none": int(np.count_nonzero(selection & (classes == "none"))),
        "candidate_or_emitter": int(np.count_nonzero(selection & (classes != "none"))),
    }


def _q_yso(table: Table) -> np.ndarray:
    j = np.asarray(table["j_m"], dtype=float)
    h = np.asarray(table["h_m"], dtype=float)
    ks = np.asarray(table["ks_m"], dtype=float)
    q = (j - h) - 1.55 * (h - ks)
    return np.isfinite(q) & (q < -0.05)


def _make_output_table(
    members: Table,
    rauw: Table,
    idx: np.ndarray,
    sep_arcsec: np.ndarray,
    halpha_class: np.ndarray,
    halpha_delta: np.ndarray,
) -> Table:
    matched_rauw = rauw[idx]
    out = Table()
    out["GaiaDR3"] = np.asarray(members["source_id"], dtype=np.int64)
    out["RAdeg"] = np.asarray(members["ra"], dtype=float)
    out["DEdeg"] = np.asarray(members["dec"], dtype=float)
    out["pMember"] = np.asarray(members["probability"], dtype=float)
    out["Ref"] = np.asarray(members["paper_reference_p06"], dtype=bool).astype(int)
    out["PMSProb"] = np.asarray(members["pms_sagitta"], dtype=float)
    out["YSO_Q"] = _q_yso(members).astype(int)
    out["Sep"] = sep_arcsec
    out["RauwRA"] = np.asarray(matched_rauw["RAJ2000"], dtype=float)
    out["RauwDE"] = np.asarray(matched_rauw["DEJ2000"], dtype=float)
    out["Vmag"] = np.asarray(matched_rauw["Vmag"], dtype=float)
    out["e_Vmag"] = np.asarray(matched_rauw["e_Vmag"], dtype=float)
    out["V-Ic"] = np.asarray(matched_rauw["V-Ic"], dtype=float)
    out["Rc-Ha"] = np.asarray(matched_rauw["Rc-Ha"], dtype=float)
    out["e_Rc-Ha"] = np.asarray(matched_rauw["e_Rc-Ha"], dtype=float)
    out["HaDelta"] = halpha_delta[idx]
    out["HaClass"] = halpha_class[idx]
    return out


def _write_plot(rauw: Table, matches: Table, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    ax.scatter(rauw["V-Ic"], rauw["Rc-Ha"], s=4, c="0.75", alpha=0.35, label="Rauw et al. catalogue")
    pms = np.asarray(matches["PMSProb"], dtype=float) >= 0.6
    excess = np.asarray(matches["HaClass"], dtype=str) != "none"
    ax.scatter(matches["V-Ic"], matches["Rc-Ha"], s=22, c="tab:blue", label="Matched candidates")
    ax.scatter(matches["V-Ic"][pms], matches["Rc-Ha"][pms], s=34, c="tab:orange", label="Sagitta PMS matches")
    ax.scatter(
        matches["V-Ic"][excess],
        matches["Rc-Ha"][excess],
        s=52,
        facecolors="none",
        edgecolors="tab:red",
        linewidths=1.1,
        label="H-alpha excess",
    )
    ax.set_xlabel(r"$V-I_C$ [mag]")
    ax.set_ylabel(r"$R_C-\mathrm{H}\alpha$ [mag]")
    ax.legend(fontsize=9)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", type=Path, default=DEFAULT_MEMBERS)
    parser.add_argument("--sagitta", type=Path, default=DEFAULT_SAGITTA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--match-radius-arcsec", type=float, default=1.0)
    args = parser.parse_args()

    raw_members = Table.read(args.members, format="ascii.ecsv")
    sagitta = Table.read(args.sagitta, format="ascii.ecsv")
    members_all = _with_sagitta_flags(raw_members, sagitta)
    candidate_mask = np.asarray(members_all["paper_reference_postclip_p05"], dtype=bool)
    members = members_all[candidate_mask]

    rauw = _load_rauw_table()
    halpha_class, halpha_delta, good_halpha = _classify_halpha(rauw)

    member_coords = SkyCoord(np.asarray(members["ra"], dtype=float) * u.deg, np.asarray(members["dec"], dtype=float) * u.deg)
    rauw_coords = SkyCoord(rauw["RAJ2000"], rauw["DEJ2000"])
    idx, sep2d, _ = member_coords.match_to_catalog_sky(rauw_coords)
    matched_mask = sep2d < args.match_radius_arcsec * u.arcsec
    matched = members[matched_mask]
    matched_idx = idx[matched_mask]
    matched_sep = sep2d[matched_mask].arcsec

    output_table = _make_output_table(matched, rauw, matched_idx, matched_sep, halpha_class, halpha_delta)

    pms = np.asarray(output_table["PMSProb"], dtype=float) >= 0.6
    ref = np.asarray(output_table["Ref"], dtype=int) == 1
    yso = np.asarray(output_table["YSO_Q"], dtype=int) == 1
    classes = np.asarray(output_table["HaClass"], dtype=str)
    summary = {
        "rauw_catalog": RAUW_CATALOG,
        "match_radius_arcsec": args.match_radius_arcsec,
        "n_rauw_rows": len(rauw),
        "n_rauw_good_halpha_photometry": int(np.count_nonzero(good_halpha)),
        "empirical_catalog_classification": _summarize(halpha_class.astype(str), np.ones(len(rauw), dtype=bool)),
        "n_candidates_input": len(members),
        "n_matches": len(output_table),
        "median_separation_arcsec": float(np.nanmedian(matched_sep)) if len(matched_sep) else None,
        "max_separation_arcsec": float(np.nanmax(matched_sep)) if len(matched_sep) else None,
        "matched_all": _summarize(classes, np.ones(len(output_table), dtype=bool)),
        "matched_reference": _summarize(classes, ref),
        "matched_sagitta_pms": _summarize(classes, pms),
        "matched_reference_sagitta_pms": _summarize(classes, ref & pms),
        "matched_q_yso": _summarize(classes, yso),
        "matched_reference_q_yso": _summarize(classes, ref & yso),
        "matched_sagitta_pms_q_yso": _summarize(classes, pms & yso),
        "classification_note": (
            "The VizieR table gives photometry but no published source-by-source H-alpha "
            "emitter flag. HaClass is not a recovery of the original Rauw et al. emitter "
            "list; it is a reproducible diagnostic classification using the Rauw et al. "
            "thresholds of 0.12--0.24 mag for candidates and >0.24 mag for emitters "
            "above an empirical non-emitter Rc-Halpha locus derived from their table."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_table.write(args.output_dir / "rauw_halpha_crossmatch.csv", format="ascii.csv", overwrite=True)
    output_table.write(args.output_dir / "rauw_halpha_crossmatch.ecsv", format="ascii.ecsv", overwrite=True)
    (args.output_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2) + "\n", encoding="utf-8")
    _write_plot(rauw, output_table, args.output_dir / "rauw_halpha_crossmatch.pdf")

    print(json.dumps(_jsonable(summary), indent=2))


if __name__ == "__main__":
    main()
