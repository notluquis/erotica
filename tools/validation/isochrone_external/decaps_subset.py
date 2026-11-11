#!/usr/bin/env python3
"""DECaPS 3D dust map (Zucker+2025, arXiv 2503.02657; Dataverse 10.7910/DVN/J9JCKO,
``decaps_mean.h5``, 8.6 GB): the pixels within 0.75 deg of NGC 6383, read over HTTP (``httpfile``).

What it is (read from the file attrs and the paper on 2026-10-05): nside 8192 NEST pixels (0.43'),
angular resolution 1' (paper abstract), ``mean`` = mean line-of-sight (cumulative) E(B-V) at 120
distance moduli 4.0 ... 18.875 (step 0.125; attrs ``DM_bin_edges``). A_V = 3.32 E(B-V) (paper: "E(B-V)
= A_V/R_V, with a mean of R_V = 3.32"). ``pixel_info/healpix_index`` is sorted, so the region is one
contiguous slice found by binary search. Quality columns (converged, infilled, DM_reliable_min/max,
n_stars) are read with it.

Output: ``decaps_subset.npz`` and ``decaps_subset.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import healpy as hp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from httpfile import HTTPFile  # noqa: E402

URL = "https://dataverse.harvard.edu/api/access/datafile/11838924"
NSIDE = 8192
L0, B0, RAD = 355.66, 0.04, 0.75


def bisect(ds, x, lo, hi):
    """First index i in [lo, hi) with ds[i] >= x (ds sorted)."""
    while lo < hi:
        m = (lo + hi) // 2
        if int(ds[m]) < x:
            lo = m + 1
        else:
            hi = m
    return lo


def main():
    t0 = time.time()
    f = HTTPFile(URL, block=1 << 16, max_blocks=8192)
    h = h5py.File(f, "r")
    pi = h["pixel_info"]
    idx = pi["healpix_index"]
    want = hp.query_disc(
        NSIDE, hp.ang2vec(L0, B0, lonlat=True), np.radians(RAD), nest=True, inclusive=True
    )
    n = idx.shape[0]
    i0 = bisect(idx, int(want.min()), 0, n)
    i1 = bisect(idx, int(want.max()) + 1, i0, n)
    print(
        f"slice {i0}:{i1} ({i1 - i0} pixels), {f.n_requests} requests, {time.time() - t0:.0f}s",
        flush=True,
    )
    out = {k: pi[k][i0:i1] for k in pi}
    out["mean"] = h["mean"][i0:i1, 0, :].astype(np.float32)
    dm = np.asarray(pi.attrs["DM_bin_edges"], float)
    np.savez_compressed(HERE / "decaps_subset.npz", dm=dm, **out)
    m = out["mean"]
    mono = float(np.mean(np.all(np.diff(m, axis=1) >= -1e-3, axis=1)))
    info = {
        "url": URL,
        "nside": NSIDE,
        "nest": True,
        "slice": [int(i0), int(i1)],
        "n_pixels": int(i1 - i0),
        "n_wanted_in_disc": int(want.size),
        "n_wanted_present": int(np.isin(want, out["healpix_index"]).sum()),
        "attrs_mean": {k: str(v) for k, v in h["mean"].attrs.items()},
        "attrs_pixel_info": {k: str(v)[:200] for k, v in pi.attrs.items()},
        "fraction_monotone_in_dm": mono,
        "http_requests": f.n_requests,
        "bytes": f.bytes_read,
        "seconds": round(time.time() - t0, 1),
    }
    (HERE / "decaps_subset.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
