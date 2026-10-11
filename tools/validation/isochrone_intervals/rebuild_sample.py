#!/usr/bin/env python3
"""Rebuild the 254-member CMD sample of P01 (``paperfaithful_reference_p06.ecsv``) from Gaia DR3.

Why: the stored artefact lives under ``data/test/NGC6383/comments_paper/radius_robustness/generated/``
(not tracked) and did not survive the move to the new machine (2026-10-11); the raw 40' cone is also
absent, so the paper's own pipeline (``validation/ngc6383_radius_robustness.py`` in the paper repo)
cannot rerun. The membership is the published one: ``cds_final/ngc6383_members.ecsv`` rows with
``Ref == 1`` (the pMember >= 0.6 reference sample, 254 rows). Photometry and its errors come from
``gaiadr3.gaia_source`` by ``source_id`` (ESA TAP sync with an IN list of the 254 ids), with the
errors computed exactly as ``erotica.preprocess._helpers.add_photometric_errors`` does:
e = 2.5 log10(1 + sigma_F / F). Check: Gmag/BPmag/RPmag must equal the CDS table's to its 4 decimals.

Env: ``~/miniforge3/envs/erotica``. Output: ``members254_gaia.ecsv`` next to this.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
CDS = Path.home() / "paper-ngc6383-aa52082-24/cds_final/ngc6383_members.ecsv"
TAP = "https://gea.esac.esa.int/tap-server/tap/sync"


def main() -> None:
    import io
    import urllib.parse
    import urllib.request

    cds = Table.read(CDS)
    ref = cds[cds["Ref"] == 1]
    assert len(ref) == 254, len(ref)
    raw = HERE / "gaia_raw.ecsv"
    if raw.exists():  # the query's own output, kept so the rebuild does not depend on the archive
        r = Table.read(raw)
    else:
        # sync GET/POST with an IN list (254 ids, no truncation at TOP 2000): the async job with an
        # uploaded table was reset by the server twice on 2026-10-11
        ids = ",".join(str(int(x)) for x in ref["GaiaDR3"])
        q = (
            "SELECT source_id, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag, phot_g_mean_flux, "
            "phot_g_mean_flux_error, phot_bp_mean_flux, phot_bp_mean_flux_error, phot_rp_mean_flux, "
            f"phot_rp_mean_flux_error FROM gaiadr3.gaia_source WHERE source_id IN ({ids})"
        )
        data = urllib.parse.urlencode(
            {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "votable", "QUERY": q}
        ).encode()
        body = urllib.request.urlopen(TAP, data=data, timeout=240).read()  # noqa: S310
        r = Table.read(io.BytesIO(body), format="votable")
        r.write(raw, overwrite=True)
    r.rename_columns(r.colnames, [c.lower() for c in r.colnames])  # the archive says SOURCE_ID
    assert len(r) == 254, len(r)
    idx = {int(s): i for i, s in enumerate(r["source_id"])}
    o = np.array([idx[int(s)] for s in ref["GaiaDR3"]])
    r = r[o]

    def emag(f, ef):
        return 2.5 * np.log10(1 + np.clip(np.asarray(ef, float) / np.asarray(f, float), 0, None))

    out = Table()
    out["source_id"] = np.asarray(r["source_id"], dtype=np.int64)
    out["Gmag"] = np.asarray(r["phot_g_mean_mag"], float)
    out["G_BPmag"] = np.asarray(r["phot_bp_mean_mag"], float)
    out["G_RPmag"] = np.asarray(r["phot_rp_mean_mag"], float)
    out["e_Gmag"] = emag(r["phot_g_mean_flux"], r["phot_g_mean_flux_error"])
    out["e_G_BPmag"] = emag(r["phot_bp_mean_flux"], r["phot_bp_mean_flux_error"])
    out["e_G_RPmag"] = emag(r["phot_rp_mean_flux"], r["phot_rp_mean_flux_error"])
    out["e_BP_RP"] = np.hypot(out["e_G_BPmag"], out["e_G_RPmag"])
    out["probability_hdbscan"] = np.asarray(ref["pHDBSCAN"], float)
    out["pMember"] = np.asarray(ref["pMember"], float)
    chk = {
        b: float(np.nanmax(np.abs(out[c] - np.asarray(ref[k], float))))
        for b, c, k in (("G", "Gmag", "Gmag"), ("BP", "G_BPmag", "BPmag"), ("RP", "G_RPmag", "RPmag"))
    }
    out.meta.update({"query": "ESA TAP sync, source_id IN list, 2026-10-11", "max_abs_diff_vs_cds": chk, "source": str(CDS)})
    print(json.dumps(chk), file=sys.stderr)
    assert all(v < 6e-5 for v in chk.values()), chk
    out.write(HERE / "members254_gaia.ecsv", overwrite=True)


if __name__ == "__main__":
    main()
