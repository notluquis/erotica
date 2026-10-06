#!/usr/bin/env python3
"""R(V) toward the NGC 6383 members from the 3D extinction-curve map of Zhang & Green 2025
(Science 387, 1209; Zenodo 10.5281/zenodo.11394477, ``Rv_map_new.h5``, "preliminary maps",
uploaded 2024-05-17), read over HTTP (``httpfile``; the datasets are contiguous float64).

``map_combined``: HEALPix nside 256 (order read from the file), 25 distance bins; ``R55_map_int``
is the integrated R(55) (A(5500)/E(4400-5500), ~R(V)) out to each bin, ``AV_map_int`` the
integrated A_V. Each member takes its own pixel and the bin containing d_c = 1108 pc.
What it decides: the extinction coefficients of the layer (CCM89 with R_V 3.1). For the
members' R(55), the CCM89 k_G and k_col are recomputed at that R_V and compared with 3.1.

Output: ``rv_map.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import healpy as hp
import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent.parent))
from edenhofer_subset import SAMPLE  # noqa: E402
from httpfile import HTTPFile  # noqa: E402

URL = "https://zenodo.org/records/11394477/files/Rv_map_new.h5"
D_C = 10 ** (10.223 / 5 + 1)
RINGS = [(0, 5), (5, 10), (10, 20), (20, 41.5)]


def main():
    from erotica.analysis._isochrone import _ccm89

    t = Table.read(SAMPLE)
    au = np.load(
        "/Users/notluquis/erotica-wt-screen/tools/validation/isochrone_model_screen/audit_perstar.npz"
    )
    r = au["radius_arcmin"]
    f = HTTPFile(URL, block=1 << 16, max_blocks=256)
    h = h5py.File(f, "r")
    g = h["map_combined"]
    nside = int(g["nside"][()])
    order = g["hp_order"][()]
    order = order.decode() if isinstance(order, bytes) else str(order)
    nest = order.lower().startswith("nest")
    db = 1000.0 * np.asarray(h["distance_bins"][:], float)  # the file stores kpc
    pix = hp.ang2pix(
        nside, np.asarray(t["l"], float), np.asarray(t["b"], float), nest=nest, lonlat=True
    )
    lo, hi = int(pix.min()), int(pix.max())
    k = int(np.searchsorted(db, D_C) - 1)  # bin [db[k], db[k+1]) containing d_c
    R = g["R55_map_int"][k, lo : hi + 1][pix - lo]
    A = g["AV_map_int"][k, lo : hi + 1][pix - lo]
    Rn = g["nside_map_int"][k, lo : hi + 1][pix - lo]
    lam = (6390.7, 5182.6, 7825.1)  # the layer's EDR3 effective wavelengths (MIST v1.2 legacy)

    def coefs(rv):
        kG, kB, kR = (_ccm89(x, rv) for x in lam)
        return kG, kB - kR

    k31 = coefs(3.1)
    ok = np.isfinite(R)
    Rm = float(np.median(R[ok]))
    kR = coefs(Rm)
    out = {
        "url": URL,
        "nside": nside,
        "hp_order": order,
        "distance_bins_pc": db.tolist(),
        "bin_used": [float(db[k]), float(db[k + 1])],
        "n_members": int(len(R)),
        "n_finite": int(ok.sum()),
        "R55_median": Rm,
        "R55_p16_p84": np.percentile(R[ok], [16, 84]).tolist(),
        "R55_rings": [float(np.median(R[ok & (r >= a) & (r < c)])) for a, c in RINGS],
        "AV_int_median": float(np.nanmedian(A)),
        "effective_nside_median": float(np.nanmedian(Rn)),
        "ccm89_kG_kcol_at_3.1": list(k31),
        "ccm89_kG_kcol_at_R55_median": list(kR),
        "kcol_ratio": kR[1] / k31[1],
        "kG_ratio": kR[0] / k31[0],
        "http_requests": f.n_requests,
    }
    (HERE / "rv_map.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
