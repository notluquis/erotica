#!/usr/bin/env python3
"""Stream Hunt+2026 members.csv (Table B.2) into a compact membership-truth subset.

The source file is the per-star injection-and-retrieval table from
Hunt et al. 2026, "The selection function of the Gaia DR3 open cluster census"
(``2026A&A...706A.341H``), CDS ``J/A+A/706/A341``: 50,873,539 rows carrying, for each
star, BOTH the observed Gaia astrometry (with errors) and the TRUE simulated values,
plus a flag for whether the star is a real simulated member or a Gaia false positive.

That combination is what makes it a membership-calibration ground truth: it covers the
faint / low-mass regime where no real independent catalogue reaches (see
``docs/design-notes/membership_ground_truth.md`` §1).

Why streaming: the file is ~33 GB uncompressed (643 B x 50.9M rows). Do not unpack it.
Keep it gzipped and let pandas decompress on the fly, chunk by chunk, keeping only the
columns below. The result is a few hundred MB of Parquet.

Usage
-----
    python tools/data/extract_hunt2026_truth.py data/external/members.csv.gz

Then in analysis::

    import pandas as pd
    df = pd.read_parquet("data/external/hunt2026_truth.parquet")

Notes
-----
Column names follow the CDS ReadMe labels. If CDS ships the file with the *native*
Gaia-style names instead (the ReadMe lists both, e.g. ``pmRA`` / ``pmra``), the loader
maps them automatically -- run with ``--inspect`` first to see what the header actually
contains.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# CDS label -> native name, for the columns worth keeping.
WANTED = {
    "Det": "detection_id",          # detections this star belongs to (membership)
    "Cluster": "cluster_id",
    "SimuStar": "simulated_star",   # True member vs Gaia false positive
    "SourceId": "source_id",
    "pmRA": "pmra",                 # --- observed, with errors ---
    "e_pmRA": "pmra_error",
    "pmDE": "pmdec",
    "e_pmDE": "pmdec_error",
    "plx": "parallax",
    "e_plx": "parallax_error",
    "pmRAtrue": "pmra_true",        # --- ground truth ---
    "pmDEtrue": "pmdec_true",
    "plxtrue": "parallax_true",
    "Gmag": "phot_g_mean_mag",      # --- photometry + physical truth ---
    "Gmagtrue": "phot_g_mean_mag_true",
    "Mass": "mass",                 # lets us probe the M < 0.3 Msun regime
    "Ext": "extinction",
}

CHUNK = 1_000_000


def _resolve(header: list[str]) -> dict[str, str]:
    """Map whichever naming convention the file uses onto the CDS labels."""
    present = {}
    for cds, native in WANTED.items():
        if cds in header:
            present[cds] = cds
        elif native in header:
            present[native] = cds
    return present


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path, help="members.csv or members.csv.gz")
    ap.add_argument("-o", "--out", type=Path, default=None)
    ap.add_argument("--inspect", action="store_true", help="print the header and exit")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"not found: {args.source}", file=sys.stderr)
        return 1

    header = list(pd.read_csv(args.source, nrows=0).columns)
    if args.inspect:
        print(f"{len(header)} columns:\n  " + "\n  ".join(header))
        return 0

    usecols = _resolve(header)
    missing = set(WANTED) - set(usecols.values())
    if missing:
        print(f"WARNING: expected columns absent from header: {sorted(missing)}", file=sys.stderr)
    if not usecols:
        print("No expected columns found. Run with --inspect.", file=sys.stderr)
        return 1

    out = args.out or args.source.parent / "hunt2026_truth.parquet"
    frames, n_rows = [], 0
    for chunk in pd.read_csv(args.source, usecols=list(usecols), chunksize=CHUNK):
        chunk = chunk.rename(columns=usecols)
        frames.append(chunk)
        n_rows += len(chunk)
        print(f"\r  {n_rows:,} rows", end="", flush=True)

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(out, compression="zstd", index=False)
    print(f"\nwrote {out}  ({len(df):,} rows, {out.stat().st_size / 1e6:.0f} MB)")

    if "SimuStar" in df.columns:
        n_true = int(df["SimuStar"].astype(str).str.lower().isin({"true", "1", "t"}).sum())
        print(f"  simulated (true) members: {n_true:,} | Gaia false positives: {len(df) - n_true:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
