"""Sagitta integration helpers for COSMIC cluster analysis."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import QTable, join


def _compute_table_hash(table: QTable) -> str:
    """Return an MD5 hex digest of the key photometric columns, sorted by source_id.

    Used to detect whether the input data has changed since the last Sagitta run
    so we can skip the expensive CLI call when the data is identical.
    """
    sidx = np.argsort(np.asarray(table["source_id"], dtype=np.int64))
    h = hashlib.md5()
    for col in ("source_id", "parallax", "g", "bp", "rp", "j", "h", "k"):
        arr = np.asarray(table[col], dtype=np.float64)[sidx]
        h.update(arr.tobytes())
    return h.hexdigest()


def _ensure_sagitta() -> None:
    import importlib.util

    if importlib.util.find_spec("sagitta") is None:
        import subprocess as _sp

        _sp.run(
            ["pip", "install", "git+https://github.com/hutchresearch/Sagitta.git"],
            check=True,
        )


def _run_sagitta_pipeline(
    input_fits: Path,
    output_fits: Path,
    av_uncertainty: int,
    pms_uncertainty: int,
    age_uncertainty: int,
    av_scatter_range: float,
) -> None:
    """Execute the Sagitta inference pipeline in-process."""
    import argparse

    from sagitta.data_tools import DataTools  # type: ignore[import-untyped]
    from sagitta.sagitta import SagittaPipeline  # type: ignore[import-untyped]

    args = argparse.Namespace(
        tableIn=str(input_fits),
        tableOut=str(output_fits),
        version="edr3",
        device="cpu",
        batch_size=5000,
        test=False,
        download_only=False,
        single_object=False,
        no_av_prediction=False,
        no_pms_prediction=False,
        no_age_prediction=False,
        av_uncertainty=av_uncertainty,
        pms_uncertainty=pms_uncertainty,
        age_uncertainty=age_uncertainty,
        av_scatter_range=av_scatter_range,
        source_id="source_id",
        ra="ra",
        dec="dec",
        l="l",
        b="b",
        parallax="parallax",
        g="g",
        bp="bp",
        rp="rp",
        j="j",
        h="h",
        k="k",
        av=None,
        eg="eg",
        ebp="ebp",
        erp="erp",
        eparallax="eparallax",
        ej="ej",
        eh="eh",
        ek="ek",
        av_out="av_sagitta",
        pms_out="pms_sagitta",
        age_out="age",
    )

    pipeline = SagittaPipeline(args=args)
    pipeline.get_naming_changes()
    pipeline.load_input_table()
    pipeline.data_frame = DataTools.fix_and_mark_nans(
        data_frame=pipeline.data_frame,
        nan_column_extension=pipeline.nan_column_suffix,
    )
    pipeline.predict_av()
    if pipeline.should_generate_av_uncertainties():
        pipeline.generate_av_uncertainties()
    pipeline.predict_pms()
    if pipeline.should_generate_pms_uncertainties():
        pipeline.generate_pms_uncertainties()
    pipeline.predict_age()
    if pipeline.should_generate_age_uncertainties():
        pipeline.generate_age_uncertainties()
    pipeline.save_output_table()


def pms_characterization(
    table: QTable,
    *,
    cluster: int,
    base_dir: Path,
    input_dir: str | Path | None,
    output_prefix: str | None,
    overwrite_inputs: bool,
    run_cli: bool,
    return_data: bool,
    av_uncertainty: int = 50,
    pms_uncertainty: int = 50,
    age_uncertainty: int = 50,
    av_scatter_range: float = 0.1,
) -> QTable | None:
    _ensure_sagitta()

    mask_cluster = table["cluster"] == cluster
    if mask_cluster.sum() == 0:
        raise ValueError(f"Cluster {cluster} has zero rows in `table`.")

    cluster_table = table[mask_cluster].copy()

    have_l = "l" in cluster_table.colnames
    have_b = "b" in cluster_table.colnames
    if not (have_l and have_b):
        if "ra" not in cluster_table.colnames or "dec" not in cluster_table.colnames:
            raise ValueError(
                "Cannot compute galactic coordinates: need 'ra'/'dec' or pre-existing 'l'/'b'."
            )
        skycoord = SkyCoord(
            ra=cluster_table["ra"], dec=cluster_table["dec"], unit="deg", frame="icrs"
        )
        cluster_table["l"] = skycoord.galactic.l.to(u.deg)
        cluster_table["b"] = skycoord.galactic.b.to(u.deg)
    else:
        try:
            cluster_table["l"] = cluster_table["l"].to(u.deg)
            cluster_table["b"] = cluster_table["b"].to(u.deg)
        except Exception:
            pass

    work = QTable()
    for col in ("source_id", "parallax", "l", "b"):
        work[col] = cluster_table[col]

    for col in ("Gmag", "G_BPmag", "G_RPmag", "j_m", "h_m", "ks_m"):
        if col in cluster_table.colnames:
            work[col] = cluster_table[col]
        else:
            work[col] = np.full(len(cluster_table), np.nan)
            warnings.warn(f"[pms_characterization] Missing column '{col}', filled with NaN.")

    for col in (
        "parallax_error",
        "e_Gmag",
        "e_G_BPmag",
        "e_G_RPmag",
        "j_msigcom",
        "h_msigcom",
        "ks_msigcom",
    ):
        if col in cluster_table.colnames:
            work[col] = cluster_table[col]
        else:
            work[col] = np.full(len(cluster_table), np.nan)
            warnings.warn(f"[pms_characterization] Missing column '{col}', filled with NaN.")

    sagitta_input = work[
        "source_id",
        "parallax",
        "l",
        "b",
        "Gmag",
        "G_BPmag",
        "G_RPmag",
        "j_m",
        "h_m",
        "ks_m",
        "parallax_error",
        "e_Gmag",
        "e_G_BPmag",
        "e_G_RPmag",
        "j_msigcom",
        "h_msigcom",
        "ks_msigcom",
    ].copy()

    sagitta_input.rename_columns(
        sagitta_input.colnames,
        [
            "source_id",
            "parallax",
            "l",
            "b",
            "g",
            "bp",
            "rp",
            "j",
            "h",
            "k",
            "eparallax",
            "eg",
            "ebp",
            "erp",
            "ej",
            "eh",
            "ek",
        ],
    )

    base_dir = Path(base_dir)
    out_dir = Path(input_dir) / "sagitta" if input_dir is not None else base_dir / "sagitta"
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = output_prefix or f"cluster_{cluster}_sagitta"
    input_fits = out_dir / f"{prefix}.fits"
    output_fits = out_dir / f"{prefix}-sagitta.fits"

    sagitta_input.write(input_fits, overwrite=overwrite_inputs, format="fits")

    # ------------------------------------------------------------------
    # Cache check — skip the expensive CLI when the data is unchanged.
    # ------------------------------------------------------------------
    hash_file = out_dir / f"{prefix}.hash"
    current_hash = _compute_table_hash(sagitta_input)

    data_unchanged = (
        output_fits.exists()
        and hash_file.exists()
        and hash_file.read_text().strip() == current_hash
    )

    if run_cli:
        _run_sagitta_pipeline(input_fits, output_fits, av_uncertainty, pms_uncertainty,
                              age_uncertainty, av_scatter_range)
        hash_file.write_text(current_hash)
    elif data_unchanged:
        print(
            f"[pms_characterization] Data unchanged — loading cached results from "
            f"{output_fits.name}"
        )
    elif output_fits.exists():
        warnings.warn(
            f"[pms_characterization] Input data has changed since the last Sagitta run "
            f"but run_cli=False. Loading stale results from {output_fits.name}. "
            "Re-run with run_cli=True to update.",
            UserWarning,
            stacklevel=2,
        )
    else:
        raise FileNotFoundError(
            f"Expected Sagitta output not found: {output_fits}.\n"
            "This looks like a first run (or the data changed). "
            "Use run_cli=True to generate it."
        )

    pms_table = QTable.read(output_fits)
    for col in ("av_sagitta", "pms_sagitta", "age"):
        if col not in pms_table.colnames:
            raise KeyError(f"Column '{col}' not found in {output_fits}.")
        pms_table[col] = np.squeeze(pms_table[col])

    joined = join(
        cluster_table,
        pms_table["source_id", "av_sagitta", "pms_sagitta", "age"],
        keys="source_id",
        join_type="left",
    )

    for col in ("av_sagitta", "pms_sagitta", "age"):
        if col not in table.colnames:
            table[col] = np.full(len(table), np.nan)

    lookup = {
        int(sid): (av, pms_val, age_val)
        for sid, av, pms_val, age_val in zip(
            joined["source_id"],
            joined["av_sagitta"],
            joined["pms_sagitta"],
            joined["age"],
        )
    }

    for index, belongs in enumerate(mask_cluster):
        if not belongs:
            continue
        sid = int(table["source_id"][index])
        av_val, pms_val, age_val = lookup.get(sid, (np.nan, np.nan, np.nan))
        table["av_sagitta"][index] = av_val
        table["pms_sagitta"][index] = pms_val
        table["age"][index] = age_val

    if return_data:
        return joined
    return None


__all__ = ["pms_characterization"]
