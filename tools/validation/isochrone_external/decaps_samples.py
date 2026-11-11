#!/usr/bin/env python3
"""The 5 posterior samples of the DECaPS 3D map (``decaps_mean_and_samples.h5``, 33.8 GB, same
Dataverse record) for the slice ``decaps_subset.py`` read, at the four distance moduli that bracket
d_c +- 50 pc (10.0 ... 10.375). Gives a per-pixel sd of A_V at d_c (5 samples: a noisy sd, said so).
The slice is re-found by binary search and checked equal to the mean file's.
Output: ``decaps_samples.npz``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from decaps_subset import bisect  # noqa: E402
from httpfile import HTTPFile  # noqa: E402

URL = "https://dataverse.harvard.edu/api/access/datafile/11840498"


def main():
    t0 = time.time()
    z = np.load(HERE / "decaps_subset.npz")
    hpi = z["healpix_index"]
    f = HTTPFile(URL, block=1 << 16, max_blocks=8192)
    h = h5py.File(f, "r")
    idx = h["pixel_info"]["healpix_index"]
    n = idx.shape[0]
    i0 = bisect(idx, int(hpi[0]), 0, n)
    i1 = i0 + hpi.size
    assert np.array_equal(idx[i0:i1], hpi), "slice differs from the mean file"
    dm = np.asarray(h["pixel_info"].attrs["DM_bin_edges"], float)
    bins = np.where((dm >= 10.0 - 1e-9) & (dm <= 10.375 + 1e-9))[0]
    s = np.stack([h["samples"][i0:i1, :, int(k)].astype(np.float32) for k in bins], axis=-1)
    np.savez_compressed(HERE / "decaps_samples.npz", dm=dm[bins], samples=s, healpix_index=hpi)
    print(
        "samples",
        s.shape,
        "bins",
        dm[bins].tolist(),
        f.n_requests,
        f.bytes_read,
        round(time.time() - t0, 1),
    )


if __name__ == "__main__":
    main()
