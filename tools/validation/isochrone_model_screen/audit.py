#!/usr/bin/env python3
r"""Numerical oracles on the per-star isochrone likelihood of ``erotica`` (C1 model), NGC 6383.

WHY: the hub finding ``isochrone-model-screen.md`` §1 writes the likelihood by hand and lists
suspected defects. Each check here is a closed-form or conservation oracle, not a reading:

* ``norm``: the cluster density ``dens/F`` must integrate to 1 over the observable half-plane
  G < G_lim (all colours); the field term integrates to 1 over the members' bounding box. Reports
  how much cluster mass falls outside the box (the two components are normalised on different
  domains).
* ``imf``: the shape of ``_chabrier2014_xi`` against the constants it is labelled with.
* ``parallax``: the cluster parallax from the members (zero-point corrected) and its error with
  the 10.3 µas floor of Maíz Apellániz+2021 (methodology §J.2) -> sigma of dm.
* ``cstar``: Riello+2021 corrected BP/RP flux excess per member (formula as verified in the hub
  finding ``e6-bp-excess-blends-ngc6383.md``), and per-star colour residuals at the C1 median
  against G, C*, radius.

Run: ``OMP_NUM_THREADS=1 nice -n 19 python audit.py`` (~1 min); writes ``audit.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent.parent))
from astropy.table import QTable, Table  # noqa: E402
from isochrone_nuts_convergence import MIST, PRIORS, SAMPLE  # noqa: E402

from erotica.analysis._isochrone import IsochroneFitter, _chabrier2014_xi  # noqa: E402

C1 = HERE.parent / "isochrone_unbinned" / "ngc_0.json"


def fitter(**over):
    pri = {k: v for k, v in PRIORS.items() if k not in ("M_met", "M_loga")}
    f = IsochroneFitter(
        isochs_path=MIST, **{**pri, "loga_range": (5.5, 7.0), "Av_range": (0.0, 3.0), **over}
    )
    f.setup(QTable(Table.read(SAMPLE)), prob_threshold=0.0)
    return f


def cstar(tab):
    C = (np.asarray(tab["phot_bp_mean_flux"]) + np.asarray(tab["phot_rp_mean_flux"])) / np.asarray(
        tab["phot_g_mean_flux"]
    )
    x = np.asarray(tab["G_BPmag"]) - np.asarray(tab["G_RPmag"])
    f = np.where(
        x < 0.5,
        1.154360 + 0.033772 * x + 0.032277 * x**2,
        np.where(
            x < 4.0,
            1.162004 + 0.011464 * x + 0.049255 * x**2 - 0.005879 * x**3,
            1.057572 + 0.140537 * x,
        ),
    )
    G = np.asarray(tab["Gmag"])
    return C - f, 0.0059898 + 8.817481e-12 * G**7.618399


def main():
    out = {}
    f = fitter()
    tab = Table.read(SAMPLE)
    c1 = json.loads(C1.read_text())
    med = [c1[p]["q05_16_50_84_95"][2] for p in ("met", "loga", "dm", "Av", "sigma_int", "f_bg")]
    met, loga, dm, Av, s, fb = med

    # --- norm: integrate the cluster density on a fine grid
    Gg = np.arange(5.0, f._mag_lim + 1e-9, 0.02)
    Cg = np.arange(-1.0, 5.0, 0.02)
    GG, CC = np.meshgrid(Gg, Cg, indexing="ij")
    xg, xc = GG.ravel(), CC.ravel()
    # the density at a grid point with that point's error model (stars' own errors replaced)
    em, ec = (
        f._e_mag_fn(np.clip(xg, f._obs_mag.min(), f._obs_mag.max())),
        f._e_col_fn(np.clip(xg, f._obs_mag.min(), f._obs_mag.max())),
    )
    save = (f._obs_mag, f._obs_col, f._e_obs_mag, f._e_obs_col)
    dens = np.empty(xg.size)
    for a in range(0, xg.size, 3000):
        sl = slice(a, a + 3000)
        f._obs_mag, f._obs_col, f._e_obs_mag, f._e_obs_col = xg[sl], xc[sl], em[sl], ec[sl]
        ll = f._star_loglike(met, loga, dm, Av, s, 0.0, np)  # f_bg = 0 -> log(dens/F)
        dens[sl] = np.exp(ll)
    f._obs_mag, f._obs_col, f._e_obs_mag, f._e_obs_col = save
    cell = 0.02 * 0.02
    nan = ~np.isfinite(dens)
    out_nan = {
        "n_nan_grid_points": int(nan.sum()),
        "n_grid": int(dens.size),
        "nan_G_range": [float(xg[nan].min()), float(xg[nan].max())] if nan.any() else None,
        "nan_col_range": [float(xc[nan].min()), float(xc[nan].max())] if nan.any() else None,
    }
    dens = np.where(nan, 0.0, dens)
    tot = dens.sum() * cell
    inbox = (
        (xg >= f._obs_mag.min())
        & (xg <= f._obs_mag.max())
        & (xc >= f._obs_col.min())
        & (xc <= f._obs_col.max())
    )
    out["norm"] = {
        "integral_cluster_dens_over_F_halfplane": float(tot),
        "note": "grid density uses the error model at each grid G, not per-star errors; the "
        "cut G<G_lim is applied to the TRUE G in F but to the observed G here, so ~1 is the oracle",
        "cluster_mass_in_box": float(dens[inbox].sum() * cell),
        "box_area": f._box_area,
        "mixture_mass_in_box_at_c1": float((1 - fb) * dens[inbox].sum() * cell + fb),
        **out_nan,
    }

    # --- imf shape
    m = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    out["imf_xi_code"] = dict(
        zip(
            map(str, m),
            map(float, _chabrier2014_xi(m) / _chabrier2014_xi(np.array([1.0]))[0]),
            strict=True,
        )
    )

    # --- parallax
    plx = np.asarray(tab["parallax"], float)
    raw = np.asarray(tab["parallax_observed"], float)
    zp = np.asarray(tab["zpvals"], float)
    e = np.asarray(tab["parallax_error"], float)
    ok = np.isfinite(plx) & np.isfinite(e)
    out["parallax"] = {
        "max_abs(parallax-(observed-zp))": float(np.nanmax(np.abs(plx - (raw - zp))))
    }
    for k in (1.0, 1.3):
        w = 1 / (k * e[ok]) ** 2
        mu = float(np.sum(w * plx[ok]) / np.sum(w))
        st = float(1 / np.sqrt(np.sum(w)))
        chi2 = float(np.sum(w * (plx[ok] - mu) ** 2) / (ok.sum() - 1))
        tot_e = float(np.hypot(st * np.sqrt(max(chi2, 1.0)), 0.0103))
        dmv = 5 * np.log10(100.0 / mu)
        out["parallax"][f"k={k}"] = {
            "mean_mas": mu,
            "stat_mas": st,
            "reduced_chi2": chi2,
            "total_with_10.3uas_floor": tot_e,
            "dm": float(dmv),
            "sigma_dm": float(5 / np.log(10) * tot_e / mu),
            "N": int(ok.sum()),
        }
    out["parallax"]["median_mas"] = float(np.median(plx[ok]))

    # --- C* and per-star colour residuals at the C1 median (single-star isochrone)
    cs, sig = cstar(tab)
    X = f._interp_isochrone(met, loga, np)
    Gm, Cm = X[1] + dm + f._kG * Av, X[2] + f._k_col1 * Av
    xg, xc = f._obs_mag, f._obs_col
    # signed colour offset at the closest point of the polyline, colour weighted x3
    A = np.c_[Gm[:-1], Cm[:-1]]
    B = np.c_[Gm[1:], Cm[1:]]
    P = np.c_[xg, xc]
    wv = np.array([1.0, 3.0])
    d = (B - A) * wv
    L2 = np.maximum((d**2).sum(1), 1e-12)
    t = np.clip((((P[:, None, :] - A[None]) * wv) * d[None]).sum(-1) / L2, 0, 1)
    Q = A[None] + t[..., None] * (B - A)[None]
    dist = (((P[:, None, :] - Q) * wv) ** 2).sum(-1)
    k = dist.argmin(1)
    resid_col = xc - Q[np.arange(len(xc)), k, 1]
    resid_G = xg - Q[np.arange(len(xc)), k, 0]
    ra, dec = np.asarray(tab["ra"]), np.asarray(tab["dec"])
    r = (
        np.hypot((ra - np.median(ra)) * np.cos(np.deg2rad(np.median(dec))), dec - np.median(dec))
        * 60
    )
    flag = np.abs(cs) > 3 * sig
    out["cstar"] = {
        "N": int(len(cs)),
        "n_flag_3sigma": int(flag.sum()),
        "frac_flag": float(flag.mean()),
        "median_cstar": float(np.median(cs)),
        "flag_by_G": {
            f"{lo}-{hi}": [
                int(((xg >= lo) & (xg < hi) & flag).sum()),
                int(((xg >= lo) & (xg < hi)).sum()),
            ]
            for lo, hi in ((8, 14), (14, 17), (17, 19), (19, 21))
        },
    }
    from scipy.stats import spearmanr

    def rho(a, b, m=None):
        m = np.ones(len(a), bool) if m is None else m
        res = spearmanr(a[m], b[m])
        return [float(res.statistic), float(res.pvalue), int(m.sum())]

    faint = xg >= 16
    out["residuals"] = {
        "median_resid_col_flag_vs_not": [
            float(np.median(resid_col[flag])),
            float(np.median(resid_col[~flag])),
        ],
        "median_resid_col_flag_vs_not_faint": [
            float(np.median(resid_col[flag & faint])),
            float(np.median(resid_col[~flag & faint])),
        ],
        "spearman_resid_col_vs_cstar_all": rho(resid_col, cs),
        "spearman_resid_col_vs_cstar_over_sigma_faint": rho(resid_col, cs / sig, faint),
        "spearman_resid_col_vs_radius": rho(resid_col, r),
        "spearman_resid_col_vs_G": rho(resid_col, xg),
        "spearman_cstar_vs_radius": rho(cs / sig, r),
        "ols_resid_col_on_cstar_faint": [
            float(v) for v in np.polyfit(cs[faint], resid_col[faint], 1)
        ],
        "mad_resid_col_by_G": {
            f"{lo}-{hi}": float(
                1.4826
                * np.median(
                    np.abs(
                        resid_col[(xg >= lo) & (xg < hi)]
                        - np.median(resid_col[(xg >= lo) & (xg < hi)])
                    )
                )
            )
            for lo, hi in ((8, 12), (12, 16), (16, 18), (18, 21))
        },
    }
    np.savez(
        HERE / "audit_perstar.npz",
        source_id=np.asarray(tab["source_id"]),
        G=xg,
        col=xc,
        cstar=cs,
        sig_cstar=sig,
        resid_col=resid_col,
        resid_G=resid_G,
        radius_arcmin=r,
    )
    (HERE / "audit.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
