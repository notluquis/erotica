#!/usr/bin/env python3
r"""Fetch the Hunt & Reffert (2024) *member* table once, for the A5 census sweep.

WHY A SECOND CACHE
------------------
``fetch_hr24.py`` caches the 7167-row cluster summary table. That table has no per-star
information, so it cannot support a profile fit: the A5 sweep needs each cluster's member
**radii**, which must be computed from member sky positions against the cluster centre.

The member table ``J/A+A/686/A42/members`` holds **1 291 929 rows** (counted via VizieR TAP,
not estimated). ``Vizier.get_catalogs`` on the parent record times out past 600 s -- the reason
``fetch_hr24.py`` exists at all -- so this script goes through the TAP endpoint with an explicit
column list and paginates on ``recno``.

Only four columns are pulled. ``Prob`` is kept so a membership-probability cut can be varied
without a refetch; ``inrt``/``inrj`` are deliberately NOT pulled because they were measured to be
redundant here: HR24's published member list is already truncated at ``rtot`` (verified on
NGC_6383, max member separation 33.738' against a catalogue ``rtot`` of 33.739').

The cache is regenerable by running this script, and the ``.parquet`` is gitignored, so the
script -- not the file -- is the source of truth. Same contract as ``fetch_hr24.py``.

USAGE
-----
    python tools/validation/fetch_hr24_members.py
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

TAP = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
TABLE = '"J/A+A/686/A42/members"'
COLUMNS = "recno, Name, RA_ICRS, DE_ICRS, Prob"
CHUNK = 100_000

DEST = Path(__file__).with_name("hr24_members.parquet")


def main():
    warnings.filterwarnings("ignore")
    import pandas as pd
    import pyvo

    tap = pyvo.dal.TAPService(TAP)
    total = int(tap.search(f"SELECT COUNT(*) AS n FROM {TABLE}").to_table()["n"][0])
    print(f"{TABLE} has {total} rows; paginating on recno in chunks of {CHUNK}")

    # VizieR's TAP front end returns "TAP service too busy!" under load and simply drops the
    # request. A single unretried chunk failure loses the whole 1.3M-row pull, so every chunk is
    # retried with exponential backoff, and finished chunks are cached to disk so a hard failure
    # resumes instead of restarting.
    cache = DEST.with_suffix(".chunks")
    cache.mkdir(exist_ok=True)

    frames, lo, t0 = [], 0, time.time()
    while lo < total:
        hi = lo + CHUNK
        part = cache / f"{lo:09d}.parquet"
        if part.exists():
            frames.append(pd.read_parquet(part))
            print(f"  recno ({lo:>8d}, {hi:>8d}]  -> cached", flush=True)
            lo = hi
            continue
        q = f"SELECT {COLUMNS} FROM {TABLE} WHERE recno > {lo} AND recno <= {hi}"
        for attempt in range(8):
            try:
                df_part = tap.search(q, maxrec=CHUNK + 10).to_table().to_pandas()
                break
            except Exception as exc:  # noqa: BLE001 -- transient TAP overload, not a bug
                wait = min(120, 5 * 2**attempt)
                print(f"    retry {attempt + 1}/8 in {wait}s ({type(exc).__name__})", flush=True)
                time.sleep(wait)
        else:
            raise RuntimeError(f"chunk ({lo}, {hi}] failed after 8 attempts")
        df_part.to_parquet(part, index=False)
        frames.append(df_part)
        print(
            f"  recno ({lo:>8d}, {hi:>8d}]  -> {len(df_part):>7d} rows  "
            f"[{sum(len(f) for f in frames):>8d} total, {time.time() - t0:6.1f}s]",
            flush=True,
        )
        lo = hi

    df = pd.concat(frames, ignore_index=True)
    df["Name"] = df["Name"].astype(str)
    df.to_parquet(DEST, index=False)
    print(f"\nwrote {DEST}  ({len(df)} rows, {df['Name'].nunique()} distinct clusters)")
    if len(df) != total:
        print(f"  WARNING: fetched {len(df)} but the table declares {total}")


if __name__ == "__main__":
    main()
