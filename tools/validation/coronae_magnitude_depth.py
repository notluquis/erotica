#!/usr/bin/env python3
"""How faint do open-cluster *coronae* get, and does the Gaia DR3 selection function reach them?

WHY THIS EXISTS
---------------
``docs/design-notes/king_model_validity.md`` and the completeness dossier report a census-wide null:
correcting an open-cluster radial profile for the DR3 selection function moves nothing (ceiling
-0.65 sigma on M11). The mechanism is that **members are too bright** -- median ``G_p98`` = 19.38
across Hunt & Reffert 2024, with only 7.1% of clusters above ``G_p98`` = 20, where ``S`` starts to
bite.

That mechanism is a property of the HR24 *sample*. Post-Gaia an open cluster is not a bounded King
sphere -- Meingast et al. (2021) resolve coronae far outside the tidal radius, Bouma et al. (2021)
trace a 500 pc halo around NGC 2516 -- and if coronae are mass-segregated they are exactly the
population nearest the DR3 edge. So the null could be an artefact of measuring only the bright core
of a population whose faint outskirts were never catalogued. **This script tests that directly.**

WHAT IT FINDS
-------------
Not one of the 7925 coronal members in Meingast et al. (2021) is fainter than ``G = 20``; their
``G_p98`` = 18.67 is *brighter* than the census. Taken at face value the objection makes the null
stronger.

But the mapped coronae span only **136-414 pc**. They reach ``M_G = 11.87 +/- 0.58`` at the 98th
percentile, and at that luminosity apparent ``G`` crosses 20 at **423 pc**. The census median
distance is ~2 kpc, 5.2 mag fainter, so the same population around a typical cluster would sit at
``G ~ 23`` -- undetectable. **Coronae look bright because they are only mappable when nearby**, which
is the boundary condition rather than a hedge: if a deeper survey ever reaches coronal members around
distant clusters, the correction becomes mandatory there.

USAGE
-----
    python tools/validation/coronae_magnitude_depth.py

Writes ``meingast2021_coronae_gmag.npz`` (per-star, gitignored) and
``meingast_coronae_depth.json`` (the per-cluster summary, committed).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

VIZIER_CATALOG = "J/A+A/645/A84"  # Meingast, Alves & Rottensteiner 2021, A&A 645, A84
CHUNK = 800  # source_ids per ADQL IN list
HR24_G_P98 = 19.38  # census median, for context
CENSUS_MEDIAN_DISTANCE_PC = 2000.0


def fetch():
    """Meingast coronal members joined to Gaia DR2 ``phot_g_mean_mag``."""
    from astroquery.gaia import Gaia
    from astroquery.vizier import Vizier

    table = Vizier(columns=["**"], row_limit=-1).get_catalogs(VIZIER_CATALOG)[0]
    ids = np.asarray(table["GaiaDR2"]).astype(np.int64)
    gmag = np.full(ids.size, np.nan)

    for i in range(0, ids.size, CHUNK):
        chunk = ids[i : i + CHUNK]
        query = (
            "SELECT source_id, phot_g_mean_mag FROM gaiadr2.gaia_source "
            f"WHERE source_id IN ({','.join(map(str, chunk))})"
        )
        result = Gaia.launch_job_async(query).get_results()
        lookup = dict(
            zip(
                np.asarray(result["source_id"]).astype(np.int64),
                np.asarray(result["phot_g_mean_mag"], dtype=float),
            )
        )
        gmag[i : i + chunk.size] = [lookup.get(int(s), np.nan) for s in chunk]
        print(f"  {i + chunk.size}/{ids.size}", flush=True)

    return (
        ids,
        gmag,
        np.asarray(table["Cluster"]).astype(str),
        np.asarray(table["Dist(xd)"], dtype=float),
    )


def main():
    warnings.filterwarnings("ignore")
    ids, gmag, cluster, distance = fetch()

    out = Path(__file__).with_name("meingast2021_coronae_gmag.npz")
    np.savez(out, source_id=ids, gmag=gmag, cluster=cluster, distance_pc=distance)

    rows = []
    print(f"\n{'cluster':12s} {'N':>5s} {'dist pc':>8s} {'G_med':>7s} {'G_p98':>7s} {'M_G,p98':>8s}")
    for name in sorted(set(cluster.tolist())):
        m = (cluster == name) & np.isfinite(gmag)
        dist = float(np.nanmedian(distance[m]))
        p98 = float(np.percentile(gmag[m], 98))
        abs_p98 = p98 - 5.0 * np.log10(dist / 10.0)
        rows.append(
            dict(cluster=name, n=int(m.sum()), dist_pc=dist, G_med=float(np.median(gmag[m])),
                 G_p98=p98, M_G_p98=abs_p98)
        )
        print(f"{name:12s} {m.sum():5d} {dist:8.1f} {np.median(gmag[m]):7.2f} {p98:7.2f} {abs_p98:8.2f}")

    ok = np.isfinite(gmag)
    abs_mean = float(np.mean([r["M_G_p98"] for r in rows]))
    dist_med = float(np.median([r["dist_pc"] for r in rows]))
    crossover = 10.0 ** ((20.0 - abs_mean) / 5.0 + 1.0)

    summary = dict(
        n_stars=int(ok.sum()),
        n_clusters=len(rows),
        G_p98_all=float(np.percentile(gmag[ok], 98)),
        G_max_all=float(np.nanmax(gmag[ok])),
        fraction_fainter_than_20=float((gmag[ok] > 20).mean()),
        M_G_p98_mean=abs_mean,
        M_G_p98_std=float(np.std([r["M_G_p98"] for r in rows])),
        distance_pc_median=dist_med,
        crossover_distance_pc=crossover,
        hr24_census_G_p98=HR24_G_P98,
        magnitudes_fainter_at_census_distance=float(
            5.0 * np.log10(CENSUS_MEDIAN_DISTANCE_PC / dist_med)
        ),
        per_cluster=rows,
    )

    print(f"\nall coronae: G_p98 {summary['G_p98_all']:.2f}  G_max {summary['G_max_all']:.2f}  "
          f"fraction G>20 = {summary['fraction_fainter_than_20']:.1%}")
    print(f"  (HR24 census G_p98 = {HR24_G_P98} -- the coronae are BRIGHTER)")
    print(f"M_G at p98 = {abs_mean:.2f} +/- {summary['M_G_p98_std']:.2f}")
    print(f"-> apparent G crosses 20 at d = {crossover:.0f} pc")
    print(f"-> census median ~{CENSUS_MEDIAN_DISTANCE_PC:.0f} pc is "
          f"{summary['magnitudes_fainter_at_census_distance']:.1f} mag fainter")

    js = Path(__file__).with_name("meingast_coronae_depth.json")
    js.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}\nwrote {js}")


if __name__ == "__main__":
    main()
