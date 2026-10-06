#!/usr/bin/env python3
"""Gaia DR3 ``astrophysical_parameters`` (GSP-Phot, ESP-HS) for the NGC 6383 members (C1 sample, 254)
and for the lambda Ori Class III stars of Cao+22 (the oracle of the grid layer, 357 with a DR3 match).

WHY: the HR-space fit of NGC 6383 needs a T_eff per star that does not come from the BP-RP colour
through a grid's colour table; GSP-Phot is the only per-star T_eff covering the members. lambda Ori
has spectroscopic T_eff (Cao+22, APOGEE) for young stars, so the same query there measures
GSP-Phot's bias on young stars before it is used on NGC 6383.

Env: miniforge base (astroquery). Query by upload + JOIN, async (no 2000-row truncation). Output
ECSV (source_id survives; no CSV). Run once; the job ids go to ``gspphot_query.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.table import Table
from astroquery.gaia import Gaia

HERE = Path(__file__).resolve().parent
NGC = Path(
    "/Users/notluquis/erotica/data/test/NGC6383/comments_paper/radius_robustness/generated/40/"
    "paperfaithful_reference_p06.ecsv"
)
LORI = Path.home() / ".cache/erotica-grids/oracles/lambda_ori/lambda_ori_2mass_dr3.ecsv"
COLS = (
    "a.source_id, a.teff_gspphot, a.teff_gspphot_lower, a.teff_gspphot_upper, a.logg_gspphot, "
    "a.mh_gspphot, a.azero_gspphot, a.ag_gspphot, a.ebpminrp_gspphot, a.distance_gspphot, "
    "a.libname_gspphot, a.teff_esphs, a.ag_esphs, a.spectraltype_esphs, a.teff_gspspec, "
    "a.lum_flame, a.classprob_dsc_combmod_star, g.has_xp_continuous, g.phot_bp_rp_excess_factor"
)


def run(name: str, ids: np.ndarray) -> tuple[Table, str]:
    up = Table({"sid": np.asarray(ids, dtype=np.int64)})
    up_path = HERE / f"_upload_{name}.xml"
    up.write(up_path, format="votable", overwrite=True)
    job = Gaia.launch_job_async(
        f"SELECT u.sid, {COLS} FROM tap_upload.u AS u "
        "LEFT JOIN gaiadr3.astrophysical_parameters AS a ON a.source_id = u.sid "
        "LEFT JOIN gaiadr3.gaia_source AS g ON g.source_id = u.sid",
        upload_resource=str(up_path),
        upload_table_name="u",
    )
    r = job.get_results()
    up_path.unlink()
    return r, job.jobid


def main() -> None:
    meta = {}
    ngc = Table.read(NGC)
    lori = Table.read(LORI)
    for name, ids in (("ngc6383", ngc["source_id"]), ("lambda_ori", lori["Source"])):
        ids = np.asarray(ids, dtype=np.int64)
        r, jid = run(name, ids)
        assert len(r) == len(ids), (name, len(r), len(ids))  # one row per uploaded id
        r.write(HERE / f"gspphot_{name}.ecsv", overwrite=True)
        n_t = int(np.isfinite(np.asarray(r["teff_gspphot"].filled(np.nan), float)).sum())
        meta[name] = {
            "n_uploaded": int(len(ids)),
            "n_rows": int(len(r)),
            "n_teff_gspphot": n_t,
            "job_id": jid,
        }
        print(name, meta[name], flush=True)
    (HERE / "gspphot_query.json").write_text(json.dumps(meta, indent=1) + "\n")


if __name__ == "__main__":
    main()
