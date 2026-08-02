#!/usr/bin/env python3
r"""Fetch the Hunt & Reffert (2024) cluster table once, for the census sweep.

WHY A CACHE AT ALL
------------------
``tools/validation/CLAUDE.md`` says live queries stay live, so a stale catalogue cannot silently
persist. This is the documented exception: the full VizieR record ``J/A+A/686/A42`` includes a
member table of millions of rows and a plain ``get_catalogs`` on the record **times out past 600 s**.
Requesting the ``/clusters`` sub-table alone returns 7167 rows in seconds.

The cache is therefore regenerable by running this script, and the ``.ecsv`` is gitignored so the
script -- not the file -- is the source of truth.

WHAT THE TABLE CONTAINS, AND THE TRAP
-------------------------------------
7167 rows, of which only **5647 are ``Type='o'``** (open clusters). The rest are 1309 ``'m'``
moving groups, 132 ``'g'`` globulars, 62 ``'d'`` dubious and 17 ``'r'`` removed. **Every figure must
keep these separate** -- the moving groups in particular are what the solar-neighbourhood coronae
get reclassified into, so mixing them in would double-count exactly the population under discussion.

USAGE
-----
    python tools/validation/fetch_hr24.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

CATALOGUE = "J/A+A/686/A42/clusters"


def main():
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.vizier import Vizier

    table = Vizier(columns=["**"], row_limit=-1).get_catalogs(CATALOGUE)[0]
    dest = Path(__file__).with_name("hr24_clusters.ecsv")
    table.write(dest, overwrite=True)

    types, counts = np.unique(np.asarray(table["Type"], dtype=str), return_counts=True)
    print(f"wrote {dest}  ({len(table)} rows, {len(table.colnames)} columns)")
    print("  " + "  ".join(f"{t}={c}" for t, c in zip(types, counts)))


if __name__ == "__main__":
    main()
