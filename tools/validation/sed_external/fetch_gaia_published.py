"""Per-member Gaia DR3 photometric errors + published T_eff / A_V for the 254 NGC 6383 members.

Downloads into ``dl/`` (a fresh directory, untrusted data) and writes ``members_gaia_ap.ecsv``.

Sources (all queried by source_id, never by cone, so nothing is truncated by a row limit):
- CDS copy of Gaia DR3, I/355/gaiadr3 + I/355/paramp (GSP-Phot, ESP-HS), by Source in IN-lists
  (the ESA TAP reset the connection on 2026-10-11).
- AIP TAP ``gaiaedr3_contrib.starhorse`` (Anders+2022, StarHorse on EDR3) by ``source_id``.

Classification for the control (written before reading any value):
- GSP-Phot (Andrae+2023, Aeneas): fits XP with isochrone-based priors -> NOT isochrone-independent.
- StarHorse: Bayesian with PARSEC isochrones as the model -> NOT isochrone-independent.
- ESP-HS: hot stars, XP + RVS model atmospheres; isochrone-free for T_eff [I].

Env: ~/miniforge3/envs/erotica (astroquery 0.4.11, pyvo 1.9.1).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pyvo
from astropy.table import Table, join

HERE = Path(__file__).resolve().parent
DL = HERE / "dl"
MEMBERS = Path("/Users/notluquis/phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
OUT = HERE / "members_gaia_ap.ecsv"


def main():
    DL.mkdir(exist_ok=True)
    m = Table.read(MEMBERS)
    ids = Table({"source_id": np.asarray(m["GaiaDR3"], dtype=np.int64)})
    up = DL / "ids.xml"
    ids.write(up, format="votable", overwrite=True)

    # The ESA archive reset the connection twice on 2026-10-11 ("in preparation for Gaia DR4 ... may
    # be unstable"); the same DR3 columns are read from the CDS copy (I/355) by Source, in IN-lists.
    viz = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")
    sid = [int(x) for x in ids["source_id"]]
    cols_g = ('"Source","Gmag","e_Gmag","BPmag","e_BPmag","RPmag","e_RPmag","E(BP/RP)","NBPblend",'
              '"NRPblend","RUWE","XPcont","nueff","pscol","Solved","Plx","e_Plx"')
    cols_a = ('"Source","Teff","b_Teff","B_Teff","logg","[Fe/H]","A0","b_A0","B_A0","AG","E(BP-RP)",'
              '"Lib","Teff-HS","e_Teff-HS","A0-HS","e_A0-HS","Flags-HS","SpType-ELS","PTTau","Pstar"')
    parts = {"g": [], "a": []}
    for k in range(0, len(sid), 60):
        chunk = ",".join(str(x) for x in sid[k : k + 60])
        for key, tab, cols in (("g", "I/355/gaiadr3", cols_g), ("a", "I/355/paramp", cols_a)):
            t = viz.run_sync(f'SELECT {cols} FROM "{tab}" WHERE "Source" IN ({chunk})',
                             maxrec=1000).to_table()
            parts[key].append(t)
    from astropy.table import vstack

    gg, aa = vstack(parts["g"]), vstack(parts["a"])
    print(f"VizieR I/355: gaia_source {len(gg)} rows, paramp {len(aa)} rows (expected {len(sid)})")
    for c in aa.colnames:
        if c != "Source":
            aa.rename_column(c, "ap_" + c)
    g = join(gg, aa, keys="Source", join_type="left")
    g.rename_column("Source", "source_id")
    g.write(DL / "gaia_ap.ecsv", overwrite=True)

    # StarHorse (Anders+2022) at AIP; uploads are not needed for 254 ids: IN-list in chunks.
    sh_rows = []
    svc = pyvo.dal.TAPService("https://gaia.aip.de/tap")
    for k in range(0, len(sid), 100):
        chunk = ",".join(str(x) for x in sid[k : k + 100])
        for attempt in range(3):
            try:
                t = svc.run_sync(
                    "SELECT source_id, teff16, teff50, teff84, logg50, met50, av16, av50, av84, "
                    "dist50, mass50, fidelity, sh_photoflag, sh_outflag "
                    f"FROM gaiaedr3_contrib.starhorse WHERE source_id IN ({chunk})",
                    maxrec=1000,
                ).to_table()
                sh_rows.append(t)
                break
            except Exception as e:  # noqa: BLE001 - report and retry, never silence
                print(f"  StarHorse chunk {k}: {type(e).__name__}: {e}", file=sys.stderr)
                time.sleep(5)
    if sh_rows:
        from astropy.table import vstack

        sh = vstack(sh_rows)
        print(f"StarHorse: {len(sh)} rows")
        sh.write(DL / "starhorse.ecsv", overwrite=True)
        for c in sh.colnames:
            if c != "source_id":
                sh.rename_column(c, "sh_" + c)
        g = join(g, sh, keys="source_id", join_type="left")
    g.write(OUT, overwrite=True)
    print(f"wrote {OUT} ({len(g)} rows)")


if __name__ == "__main__":
    main()
