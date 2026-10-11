"""Zhang, Green & Rix 2023 (XP data-driven parameters, Zenodo 10.5281/zenodo.7811871 v1.2) for the
254 NGC 6383 members, read by HTTP range requests (no 1.47 GB download).

The catalogue files are contiguous HDF5 datasets ordered by ``gdr3_source_id`` (measured: file 00
spans 4295806720…1572297096530849536, file 09 6037153507566018304…6917528993281571840). We
binary-search the file and the rows that bracket our id range, read that contiguous slice, and
match exactly on source_id. Monotonicity of the slice is checked, not assumed.

Units (README): stellar_params_est = (T_eff [kK], [Fe/H], log g, E [ZGR23 units], parallax [mas]).
A_V is not a catalogue column: ``extinction_curve.txt`` x E gives A(lambda); we report A(550 nm).

Classification for the control (fixed before reading values): data-driven model trained on LAMOST
labels with 2MASS/WISE photometry, no isochrones -> isochrone-independent, but PMS are outside the
training set (LAMOST does not reach dec -32; young stars are rare in it) [I].
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import requests
from astropy.table import Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from httpfile import HTTPFile  # noqa: E402

URL = "https://zenodo.org/api/records/7811871/files/{}/content"
MEMBERS = Path("/Users/notluquis/phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
OUT = HERE / "members_zhang23.ecsv"


def open_h5(name):
    f = HTTPFile(URL.format(name), block=1 << 16)
    return f, h5py.File(f, "r")


def bsearch(d, value):
    lo, hi = 0, d.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if d[mid] < value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def main():
    ids = np.sort(np.asarray(Table.read(MEMBERS)["GaiaDR3"], dtype=np.int64))
    lo_id, hi_id = int(ids[0]), int(ids[-1])
    target = None
    for i in range(10):
        f, h = open_h5(f"stellar_params_catalog_{i:02d}.h5")
        d = h["gdr3_source_id"]
        first, last = int(d[0]), int(d[-1])
        if first <= hi_id and last >= lo_id:
            target = (i, f, h)
            print(f"file {i:02d}: {first}…{last} contains {lo_id}…{hi_id}")
            break
    if target is None:
        raise SystemExit("no file spans the member id range")
    i, f, h = target
    d = h["gdr3_source_id"]
    a, b = bsearch(d, lo_id), bsearch(d, hi_id + 1)
    print(f"slice rows {a}…{b} ({b - a} rows), {f.n_requests} requests so far")
    sl = slice(a, b)
    sid = d[sl]
    if not np.all(np.diff(sid) > 0):
        raise SystemExit("slice is not strictly increasing: ordering assumption broken")
    est, err = h["stellar_params_est"][sl], h["stellar_params_err"][sl]
    qf, chi2 = h["quality_flags"][sl], h["chi2_opt"][sl]
    tc, fc, lc = h["teff_confidence"][sl], h["feh_confidence"][sl], h["logg_confidence"][sl]
    k = np.searchsorted(sid, ids)
    k = np.clip(k, 0, sid.size - 1)
    hit = sid[k] == ids
    print(f"members in Zhang+23: {hit.sum()} of {ids.size}; {f.n_requests} requests, {f.bytes_read / 1e6:.1f} MB")

    curve = requests.get(URL.format("extinction_curve.txt"), timeout=60).text
    lam, ext = np.loadtxt(curve.splitlines(), unpack=True, comments="#", skiprows=1, delimiter=None)
    r550 = float(np.interp(550.0, lam, ext))
    print(f"extinction_curve at 550 nm = {r550:.4f} mag per unit E (A_V proxy = {r550:.3f} E)")

    kk = k[hit]
    t = Table()
    t["source_id"] = ids[hit]
    names = ["teff_kK", "feh", "logg", "E", "plx"]
    for j, n in enumerate(names):
        t[f"z23_{n}"] = est[kk, j]
        t[f"z23_e_{n}"] = err[kk, j]
    t["z23_teff"] = 1000 * t["z23_teff_kK"]
    t["z23_e_teff"] = 1000 * t["z23_e_teff_kK"]
    t["z23_A550"] = r550 * t["z23_E"]
    t["z23_e_A550"] = r550 * t["z23_e_E"]
    t["z23_quality_flags"] = qf[kk]
    t["z23_chi2_opt"] = chi2[kk]
    t["z23_teff_conf"], t["z23_feh_conf"], t["z23_logg_conf"] = tc[kk], fc[kk], lc[kk]
    t.meta.update(
        source="Zenodo 10.5281/zenodo.7811871 v1.2", file=f"stellar_params_catalog_{i:02d}.h5",
        rows=f"{a}-{b}", A550_per_E=r550,
    )
    t.write(OUT, overwrite=True)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
