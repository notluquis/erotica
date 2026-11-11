#!/usr/bin/env python3
"""Does Bayestar19 (Green+2019, ApJ 887, 93; Dataverse 10.7910/DVN/2EJ9TX) cover NGC 6383?

Documented footprint: Pan-STARRS 1, dec > -30 deg (dustmaps ``bayestar.py`` docstring). The members
are at dec -33.2 ... -31.9. This reads the map's own ``pixel_info`` table (nside, NEST healpix index
of every line of sight) over HTTP and asks, for each member and for a positive control in the plane
north of -30 deg (glon = 10, b = 0, dec ~ -20), whether a pixel of the map contains it.
Output: ``bayestar_coverage.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from edenhofer_subset import SAMPLE  # noqa: E402
from httpfile import HTTPFile  # noqa: E402

URL = "https://dataverse.harvard.edu/api/access/datafile/3424724"  # bayestar2019.h5


def covered(nside, hpix, glon, b):
    have = {}
    for ns in np.unique(nside):
        have[int(ns)] = set(hpix[nside == ns].tolist())
    out = []
    for li, bi in zip(glon, b, strict=True):
        out.append(
            any(int(hp.ang2pix(ns, li, bi, nest=True, lonlat=True)) in s for ns, s in have.items())
        )
    return np.array(out)


def main():
    t0 = time.time()
    f = HTTPFile(URL, block=1 << 22, max_blocks=64)
    pi = h5py.File(f, "r")["pixel_info"][:]
    nside, hpix = pi["nside"].astype(int), pi["healpix_index"].astype(int)
    t = Table.read(SAMPLE)
    cm = covered(nside, hpix, np.asarray(t["l"]), np.asarray(t["b"]))
    ctrl_l, ctrl_b = np.array([10.0, 0.0, 355.66]), np.array([0.0, 0.0, 5.0])
    cc = covered(nside, hpix, ctrl_l, ctrl_b)
    dec_ctrl = SkyCoord(ctrl_l, ctrl_b, unit="deg", frame="galactic").icrs.dec.deg
    # southern edge of the map in dec, from the pixel centres
    lon, lat = np.zeros(len(hpix)), np.zeros(len(hpix))
    for ns in np.unique(nside):
        m = nside == ns
        lon[m], lat[m] = hp.pix2ang(int(ns), hpix[m], nest=True, lonlat=True)
    dec_min = float(SkyCoord(lon, lat, unit="deg", frame="galactic").icrs.dec.deg.min())
    out = {
        "url": URL,
        "n_pixels": int(len(hpix)),
        "nsides": sorted(map(int, np.unique(nside))),
        "members_covered": int(cm.sum()),
        "members": int(len(cm)),
        "member_dec_range": [float(t["dec"].min()), float(t["dec"].max())],
        "controls": [
            {"l": float(a), "b": float(b), "dec": float(d), "covered": bool(c)}
            for a, b, d, c in zip(ctrl_l, ctrl_b, dec_ctrl, cc, strict=True)
        ],
        "map_min_dec_of_pixel_centres": dec_min,
        "http_requests": f.n_requests,
        "bytes": f.bytes_read,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, indent=1))
    (HERE / "bayestar_coverage.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
