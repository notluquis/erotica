"""Sagitta integration helpers for COSMIC cluster analysis."""

from __future__ import annotations

import subprocess
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import QTable, join


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
) -> QTable | None:
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

    sagitta = work[
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

    sagitta.rename_columns(
        sagitta.colnames,
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

    sagitta.write(input_fits, overwrite=overwrite_inputs, format="fits")

    if run_cli:
        cmd = [
            "sagitta",
            str(input_fits),
            "--tableOut",
            str(output_fits),
            "--av_out",
            "av_sagitta",
            "--pms_out",
            "pms_sagitta",
            "--age_out",
            "age",
            "--av_uncertainty",
            "50",
            "--pms_uncertainty",
            "50",
            "--age_uncertainty",
            "50",
            "--av_scatter_range",
            "0.1",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            message = (
                "Sagitta CLI failed (code {code}).\n" "STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            ).format(code=result.returncode, stdout=result.stdout, stderr=result.stderr)
            raise RuntimeError(message)
    if not output_fits.exists():
        message = (
            f"Expected output FITS not found: {output_fits}.\n"
            "If you did not run the CLI here, ensure the Sagitta output was generated externally."
        )
        raise FileNotFoundError(message)

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

    if "pms_sagitta" in joined.colnames:
        joined["pms_sagitta"] = np.nan_to_num(joined["pms_sagitta"], nan=0.0)

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
