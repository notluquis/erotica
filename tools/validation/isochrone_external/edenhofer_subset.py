#!/usr/bin/env python3
"""Edenhofer+2024 (A&A 685, A82; Zenodo 10.5281/zenodo.8187943) 3D dust map: the HEALPix columns
around NGC 6383, read from the remote ``mean_and_std_healpix.fits`` by HTTP range (``rangefits``).

What it models: dust density in E of Zhang, Green & Rix 2023 per pc, mean and std of the posterior
samples, nside 256 (13.7 arcmin pixels), NEST, 516 log-spaced layers 69-1250 pc. A_V = 2.8 E
(Edenhofer+2024 §3: "multiplied the unitless ZGR23 extinction by a factor of 2.8", A(540 nm)).

Pixels: every HEALPix neighbour used by ``get_interp_weights`` for the 254 members, plus every
pixel centre within 1.5 deg of the cluster centre (for the map figure and the radial profile).

Check: one full row (layer 400) is read whole and compared with the subset (must be identical).

Output: ``edenhofer_subset.npz`` (pix, mean[516, npix], std[516, npix], radii, bounds, data0,
data0_std) and ``edenhofer_subset.json`` (provenance).
Run with the miniforge base python (healpy, astropy); ~10 min at the measured 1 MB/s.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from rangefits import _get, hdus, read_columns  # noqa: E402

URL = "https://zenodo.org/records/8187943/files/mean_and_std_healpix.fits"
SAMPLE = Path(
    "/Users/notluquis/erotica/data/test/NGC6383/comments_paper/radius_robustness/generated/40/"
    "paperfaithful_reference_p06.ecsv"
)
NSIDE = 256
L0, B0 = 355.66, 0.04  # cluster centre (galactic), members' median l, b


def pixel_set(t):
    idx, _ = hp.get_interp_weights(
        NSIDE, np.asarray(t["l"]), np.asarray(t["b"]), nest=True, lonlat=True
    )
    disc = hp.query_disc(
        NSIDE, hp.ang2vec(L0, B0, lonlat=True), np.radians(1.5), nest=True, inclusive=True
    )
    return np.unique(np.r_[idx.ravel(), disc])


def main():
    t0 = time.time()
    t = Table.read(SAMPLE)
    pix = pixel_set(t)
    H = hdus(URL)
    byname = {h["h"].get("EXTNAME", ""): h for h in H}
    mean_h, std_h = byname["MEAN"], byname["STD."]
    print(f"{pix.size} pixels; MEAN dims {mean_h['dims']}", flush=True)

    def table_col(h):
        nb = int(h["h"]["NAXIS1"]) * int(h["h"]["NAXIS2"])
        return np.frombuffer(
            _get(URL, h["data_offset"], h["data_offset"] + nb - 1), dtype=">f4"
        ).astype(float)

    radii = table_col(byname["RADIAL PIXEL CENTERS"])
    bounds = table_col(byname["RADIAL PIXEL BOUNDARIES"])

    def vec_cols(h):  # one range over [min pix, max pix] (716 px), not one request per pixel
        a = h["data_offset"] + 4 * int(pix.min())
        buf = np.frombuffer(_get(URL, a, a + 4 * int(pix.max() - pix.min() + 1) - 1), dtype=">f4")
        return buf[pix - pix.min()].astype(float)

    d0 = vec_cols(byname["MEAN OF INTEGRATED INNER 68.8 PC"])
    d0s = vec_cols(byname["STD. OF INTEGRATED INNER 68.8 PC"])
    p_, mean, runs = read_columns(URL, mean_h, pix)
    print(f"mean read, {len(runs)} runs/row, {time.time() - t0:.0f}s", flush=True)
    _, std, _ = read_columns(URL, std_h, pix)
    print(f"std read, {time.time() - t0:.0f}s", flush=True)

    # control: a full row against the subset
    row = 400
    a = mean_h["data_offset"] + 4 * row * mean_h["dims"][0]
    full = np.frombuffer(_get(URL, a, a + 4 * mean_h["dims"][0] - 1), dtype=">f4").astype(float)
    same = bool(np.array_equal(full[pix], mean[row]))
    print("full-row check identical:", same, flush=True)
    np.savez_compressed(
        HERE / "edenhofer_subset.npz",
        pix=pix,
        mean=mean,
        std=std,
        radii=radii,
        bounds=bounds,
        data0=d0,
        data0_std=d0s,
    )
    (HERE / "edenhofer_subset.json").write_text(
        json.dumps(
            {
                "url": URL,
                "nside": NSIDE,
                "nest": True,
                "n_pix": int(pix.size),
                "n_layers": int(radii.size),
                "radii_pc": [float(radii[0]), float(radii[-1])],
                "runs_per_row": len(runs),
                "full_row_check": {"row": row, "identical": same},
                "seconds": round(time.time() - t0, 1),
            },
            indent=1,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
