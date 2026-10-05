#!/usr/bin/env python3
"""A_V per NGC 6383 member from 3D dust maps, its radial structure, and whether it explains the
radial gradient of the colour residual (hub finding ``isochrone-external-constraints.md`` §0.4).

Maps (each value is FOREGROUND dust smoothed at the map's scale, not the A_V of a star):

* Edenhofer+2024 (``edenhofer_subset.npz``, read by ``edenhofer_subset.py``): density in E(ZGR23)/pc,
  integrated exactly as ``dustmaps.edenhofer2023`` does with ``integrated=True`` (density x shell
  width, plus the inner 68.8 pc layer, cumulative sum, log, HEALPix x radius linear interpolation,
  exp), A_V = 2.8 E (Edenhofer+2024 §3). Checked against dustmaps on a toy sphere (``--selftest``).
* Vergely+2022 (J/A+A/664/A174) 10 pc and 25 pc cubes, Lallement+2022 (J/A+A/661/A147) 25 pc
  cube: A0(550 nm)/pc on a Cartesian grid, Sun at the cube centre; line-of-sight integral with
  trilinear interpolation and 1 pc steps.

Distance: d_c = 1110 pc for every member (dm 10.223, the layer's parallax prior), plus d_c +- 50 pc.
Extinction coefficients: the grid layer's own (CCM89, R_V 3.1, MIST v1.2 EDR3 effective wavelengths).

Output: ``av_maps.json`` (summary) and ``av_maps_perstar.ecsv`` (A_V per member and map).
Run: ``~/miniforge3/bin/python av_maps.py [--selftest]`` (base env: healpy, dustmaps, scipy).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import healpy as hp
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.ndimage import map_coordinates
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
SAMPLE = Path(
    "/Users/notluquis/erotica/data/test/NGC6383/comments_paper/radius_robustness/generated/40/"
    "paperfaithful_reference_p06.ecsv"
)
AUDIT = Path(
    "/Users/notluquis/erotica-wt-screen/tools/validation/isochrone_model_screen/audit_perstar.npz"
)
DUST = Path.home() / ".cache/erotica-dust"
D_C = 10 ** (10.223 / 5 + 1)
E_TO_AV = 2.8
RINGS = [(0, 5), (5, 10), (10, 20), (20, 41)]
RESID_RINGS = [0.046, 0.024, 0.045, -0.051]  # isochrone-model-screen.md §3.1


# ---------------------------------------------------------------- Edenhofer
def eden_cumlog(mean, bounds, data0):
    """log of the cumulative integrated extinction per layer, as dustmaps (integrated=True)."""
    d = mean * np.diff(bounds)[:, None]
    d[0] += data0
    return np.log(np.cumsum(d, axis=0))


def eden_query(cumlog, pix, radii, glon, b, dist, nside=256):
    """dustmaps' ``_interp_hpxr2lbd`` on the log cumulative, then exp; subset of pixels ``pix``."""
    return np.exp(_lin_query(cumlog, pix, radii, glon, b, dist, nside))


def eden_sigma(std, bounds, data0_std, pix, radii, glon, b, dist, nside=256):
    """Two bounds on the std of the integral (the file has no samples): independent shells and
    fully correlated shells, each interpolated like the mean (linear, not in log)."""
    s = std * np.diff(bounds)[:, None]
    s0 = data0_std
    ind = np.sqrt(np.cumsum(s**2, axis=0) + s0**2)
    cor = np.cumsum(s, axis=0) + s0
    return [_lin_query(arr, pix, radii, glon, b, dist, nside) for arr in (ind, cor)]


def _lin_query(arr, pix, radii, glon, b, dist, nside):
    """HEALPix bilinear (get_interp_weights) x linear in radius, as dustmaps; KeyError if a
    neighbour pixel was not read."""
    col = {int(p): i for i, p in enumerate(pix)}
    idx, w = hp.get_interp_weights(nside, glon, b, nest=True, lonlat=True)
    ir = np.searchsorted(radii, dist)
    il = (ir - 1).clip(0, radii.size - 1)
    ir = ir.clip(0, radii.size - 1)
    wl = np.abs(np.stack((radii[ir] - dist, radii[il] - dist)))
    wl /= wl.sum(axis=0)
    cols = np.vectorize(col.__getitem__)(idx)
    v = np.zeros(glon.size)
    for k, rr in enumerate((il, ir)):
        v += wl[k] * (arr[rr[None, :], cols] * w).sum(axis=0)
    return v


def selftest(mutate: bool = False) -> float:
    """Our subset integration against dustmaps' own code on a toy nside-8 sphere."""
    from dustmaps.edenhofer2023 import _interp_hpxr2lbd

    rng = np.random.default_rng(0)
    ns, nr = 8, 40
    bounds = np.geomspace(69.0, 1250.0, nr + 1)
    radii = np.sqrt(bounds[1:] * bounds[:-1])
    mean = rng.gamma(2.0, 1e-4, size=(nr, hp.nside2npix(ns)))
    d0 = rng.gamma(2.0, 1e-3, size=hp.nside2npix(ns))
    # dustmaps' path (integrated=True), on the full sphere
    full = mean * np.diff(bounds)[:, None]
    full[0] += d0
    full = np.log(np.cumsum(full, axis=0))
    glon, b = np.array([355.66, 355.9, 355.2]), np.array([0.04, -0.3, 0.5])
    dist = np.array([1110.0, 800.0, 1200.0])
    ref = np.exp(_interp_hpxr2lbd(full, radii, ns, True, glon, b, dist))
    idx, _ = hp.get_interp_weights(ns, glon, b, nest=True, lonlat=True)
    pix = np.unique(idx)
    cl = eden_cumlog(mean[:, pix], bounds, (0 * d0 if mutate else d0)[pix])
    ours = eden_query(cl, pix, radii, glon, b, dist, ns)
    return float(np.max(np.abs(ours / ref - 1)))


# ---------------------------------------------------------------- Cartesian cubes
def cube_los(path, glon, b, dist, step=None):
    """Integral of A0/pc from the Sun to ``dist`` along (glon, b), trilinear, 1 pc steps.
    Grid: index i along X is at (i - (SUN_POS - 0.5)) * STEP pc. SUN_POS = 300.5 / 80.5 on 601 / 161
    pixels is the centre of 0-based pixel 300 / 80 measured from the cube corner, i.e. the
    symmetric centre of a -1500...+1500 (-400...+400) pc axis. Measured 2026-10-05: reading it as
    a 1-based pixel (0-based 299.5) moves A_V at the cluster from 0.98 to 0.87 (a 5 pc shift in z
    through a thin dust layer); the first version of this script had that half-pixel error."""
    with fits.open(path, memmap=True) as h:
        hd = h[0].header
        cube = h[0].data  # (z, y, x)
        st = float(hd["STEP"])
        sun = np.array([hd["SUN_POSX"], hd["SUN_POSY"], hd["SUN_POSZ"]], float) - 0.5
        lr, br = np.radians(glon), np.radians(b)
        out = np.empty(glon.size)
        for i in range(glon.size):
            s = np.arange(0.5, dist[i], 1.0)
            x = s * np.cos(br[i]) * np.cos(lr[i])
            y = s * np.cos(br[i]) * np.sin(lr[i])
            z = s * np.sin(br[i])
            ix, iy, iz = x / st + sun[0], y / st + sun[1], z / st + sun[2]
            v = map_coordinates(cube, [iz, iy, ix], order=1, mode="nearest")
            out[i] = v.sum() * 1.0 - (s[-1] + 0.5 - dist[i]) * v[-1]
        return out, {
            "step_pc": st,
            "resol_pc": hd.get("RESOL"),
            "unit": hd.get("UNIT"),
            "sun_index_0based": sun.tolist(),
        }


# ---------------------------------------------------------------- main
def ext_coefs():
    from erotica.analysis._isochrone import IsochroneFitter
    from tools.validation.isochrone_colour.common import EDR3, PRI_PLX, mist  # noqa: F401

    f = IsochroneFitter(
        grid=mist(feh=[0.0], loga_range=(6.3, 6.5)), magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX
    )
    kG, kBP, kRP = f._compute_ext_coefs()
    return kG, kBP - kRP


def rings(r, v):
    return [
        float(np.nanmedian(v[(r >= a) & (r < c)])) if ((r >= a) & (r < c)).any() else None
        for a, c in RINGS
    ]


def rank_partial(y, x, z):
    """Spearman of y on x controlling for z: correlation of rank residuals."""
    from scipy.stats import rankdata

    ry, rx, rz = (rankdata(v) for v in (y, x, z))
    res = [v - np.polyval(np.polyfit(rz, v, 1), rz) for v in (ry, rx)]
    return float(np.corrcoef(*res)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    st = {"max_rel_diff": selftest(), "max_rel_diff_mutant_no_inner_layer": selftest(True)}
    print("selftest", st, flush=True)
    assert st["max_rel_diff"] < 1e-10 and st["max_rel_diff_mutant_no_inner_layer"] > 1e-3
    if a.selftest:
        return

    t = Table.read(SAMPLE)
    au = dict(np.load(AUDIT))
    assert np.array_equal(np.asarray(t["source_id"]), au["source_id"])
    glon, b = np.asarray(t["l"], float), np.asarray(t["b"], float)
    r_arc = au["radius_arcmin"]
    kG, kc = ext_coefs()
    per = Table(
        {
            "source_id": t["source_id"],
            "l": glon,
            "b": b,
            "radius_arcmin": r_arc,
            "G": au["G"],
            "resid_col": au["resid_col"],
        }
    )
    summ = {"d_c_pc": D_C, "kG": kG, "k_col": kc, "selftest": st, "maps": {}}

    e = np.load(HERE / "edenhofer_subset.npz")
    cl = eden_cumlog(e["mean"], e["bounds"], e["data0"])
    for tag, dd in (("", D_C), ("_m50", D_C - 50), ("_p50", D_C + 50)):
        per["AV_eden" + tag] = E_TO_AV * eden_query(
            cl, e["pix"], e["radii"], glon, b, np.full(glon.size, dd)
        )
    sind, scor = eden_sigma(
        e["std"],
        e["bounds"],
        e["data0_std"],
        e["pix"],
        e["radii"],
        glon,
        b,
        np.full(glon.size, D_C),
    )
    per["sAV_eden_ind"], per["sAV_eden_cor"] = E_TO_AV * sind, E_TO_AV * scor
    maps = {"eden": "AV_eden"}
    dp = HERE / "decaps_subset.npz"
    if dp.exists():
        # DECaPS 3D: nearest nside-8192 pixel, cumulative E(B-V) linear in DM, A_V = 3.32 E(B-V)
        z = np.load(dp)
        hpi = z["healpix_index"].astype(np.int64)
        p = hp.ang2pix(8192, glon, b, nest=True, lonlat=True)
        j = np.searchsorted(hpi, p)
        assert np.array_equal(hpi[j], p), "a member pixel missing from the DECaPS slice"
        dmc = 5 * np.log10(D_C / 10)
        for tag, dd in (("", D_C), ("_m50", D_C - 50), ("_p50", D_C + 50)):
            mu = 5 * np.log10(dd / 10)
            per["AV_decaps" + tag] = 3.32 * np.array(
                [np.interp(mu, z["dm"], z["mean"][k]) for k in j]
            )
        per["decaps_converged"] = z["converged"][j]
        per["decaps_infilled"] = z["infilled"][j]
        per["decaps_dm_rel_min"], per["decaps_dm_rel_max"] = (
            z["DM_reliable_min"][j],
            z["DM_reliable_max"][j],
        )
        per["decaps_n_stars"] = z["n_stars"][j]
        maps["decaps"] = "AV_decaps"
        sp_ = HERE / "decaps_samples.npz"
        if sp_.exists():  # sd over the 5 posterior samples (a noisy sd: 5 draws)
            zs = np.load(sp_)
            assert np.array_equal(zs["healpix_index"], z["healpix_index"])
            smp = zs["samples"][j]  # (n, 5, 4 DM)
            v = np.array(
                [
                    [np.interp(dmc, zs["dm"], smp[i, q]) for q in range(smp.shape[1])]
                    for i in range(len(j))
                ]
            )
            per["sAV_decaps"] = 3.32 * v.std(axis=1, ddof=1)
            per["AV_decaps_samplemean"] = 3.32 * v.mean(axis=1)
        summ["maps"]["decaps"] = {
            "file": str(dp),
            "dm_c": dmc,
            "A_V_per_EBV": 3.32,
            "n_converged": int((z["converged"][j] > 0).sum()),
            "n_infilled": int((z["infilled"][j] > 0).sum()),
            "n_dm_c_inside_reliable": int(
                ((z["DM_reliable_min"][j] <= dmc) & (z["DM_reliable_max"][j] >= dmc)).sum()
            ),
            "n_stars_median": float(np.median(z["n_stars"][j])),
        }
    cubes = {
        "verg10": DUST / "vergely2022/explore_cube_density_values_010pc_v2.fits",
        "verg25": DUST / "vergely2022/explore_cube_density_values_025pc_v2.fits",
        "lall22": DUST / "lallement2022/cube_ext.fits",
    }
    for k, p in cubes.items():
        if not p.exists():
            summ["maps"][k] = {"missing": str(p)}
            continue
        for tag, dd in (("", D_C), ("_m50", D_C - 50), ("_p50", D_C + 50)):
            v, info = cube_los(p, glon, b, np.full(glon.size, dd))
            per[f"AV_{k}{tag}"] = v
        summ["maps"][k] = {"file": str(p), **info}
        maps[k] = f"AV_{k}"

    G = au["G"]
    bright = G < 12
    ok = ~((G >= 12) & (G < 14))
    rc = au["resid_col"]
    summ["resid_rings_all_ok"] = rings(r_arc[ok], rc[ok])
    summ["resid_rings_bright_G_lt_12"] = rings(r_arc[bright], rc[bright])
    summ["n_bright"] = int(bright.sum())
    for k, c in maps.items():
        v = np.asarray(per[c], float)
        pc = kc * (v - np.median(v))
        s = summ["maps"].setdefault(k, {})
        s.update(
            {
                "AV_median": float(np.median(v)),
                "AV_p16_p84": np.percentile(v, [16, 84]).tolist(),
                "AV_min_max": [float(v.min()), float(v.max())],
                "AV_median_dc_m50_p50": [
                    float(np.median(per[c + "_m50"])),
                    float(np.median(per[c + "_p50"])),
                ],
                "AV_rings": rings(r_arc, v),
                "spearman_AV_radius": [float(x) for x in spearmanr(v, r_arc)],
                "pred_colour_rings": rings(r_arc, pc),
                "pred_outer_minus_inner": None,
                "spearman_resid_vs_predcol_ok": [float(x) for x in spearmanr(rc[ok], pc[ok])],
                "rank_partial_resid_radius_given_map_ok": rank_partial(rc[ok], r_arc[ok], pc[ok]),
                "spearman_resid_radius_ok": float(spearmanr(rc[ok], r_arc[ok])[0]),
            }
        )
        pr = s["pred_colour_rings"]
        s["pred_outer_minus_inner"] = pr[3] - pr[0]
        obs = RESID_RINGS[3] - RESID_RINGS[0]
        s["fraction_of_observed_gradient"] = s["pred_outer_minus_inner"] / obs
    summ["eden_sigma_AV_median_ind_cor"] = [
        float(np.median(per["sAV_eden_ind"])),
        float(np.median(per["sAV_eden_cor"])),
    ]
    # line-of-sight profile at the centre (Edenhofer)
    ds = np.arange(600, 1250, 25.0)
    summ["eden_los_centre"] = {
        "d_pc": ds.tolist(),
        "AV": (
            E_TO_AV
            * eden_query(
                cl, e["pix"], e["radii"], np.full(ds.size, 355.66), np.full(ds.size, 0.04), ds
            )
        ).tolist(),
    }
    per.write(HERE / "av_maps_perstar.ecsv", overwrite=True)
    (HERE / "av_maps.json").write_text(json.dumps(summ, indent=1, default=float) + "\n")
    print(json.dumps(summ, indent=1, default=float))


if __name__ == "__main__":
    main()
