"""Piezas comunes del tercer experimento (R38; prerregistro hub §4).

- vergely_av(l, b, d): A0(550 nm) integrada desde el Sol hasta d (pc) en el cubo de 25 pc de Vergely+2022 (VizieR J/A+A/664/A174,
  explore_cube_density_values_025pc_v2.fits: 601x601x81, paso 10 pc, unidad A0/pc, Sol en (300,5; 300,5; 40,5) de la cabecera; se
  toma la convencion de pixel 1-based de FITS -> indice 0-based = pos - 1; la ambiguedad es de medio paso, 5 pc). NaN si el rayo sale
  del cubo antes de d. A0(550 nm) ~ A_V [I: la diferencia entre A0 y A_V depende del espectro; no se corrige].
- ms_curve(logt, plane): la MS de MIST (phase 0, bajo el turn-off: desde el punto de M_G maxima hasta 0,5 mag bajo el turn-off) en
  (BP-RP, M_G) o (G-Ks, M_Ks), ordenada por color.
- delta_av(...): el DeltaA_V que lleva una estrella por su vector de enrojecimiento (Wang & Chen 2019, core.EXT) hasta esa MS.
"""
from pathlib import Path

import numpy as np

import core

C = Path(__file__).parent / "cache"
_CUBE = None


def _cube():
    global _CUBE
    if _CUBE is None:
        from astropy.io import fits
        h = fits.open(C / "vergely" / "explore_cube_density_values_025pc_v2.fits", memmap=False)
        hd = h[0].header
        _CUBE = (np.asarray(h[0].data, np.float32), float(hd["STEP"]),
                 np.array([hd["SUN_POSX"], hd["SUN_POSY"], hd["SUN_POSZ"]]) - 1.0)
    return _CUBE


def vergely_av(l, b, d, ds=5.0):
    from scipy.ndimage import map_coordinates
    cube, step, sun = _cube()  # data[z, y, x]
    l, b, d = (np.atleast_1d(np.asarray(v, float)) for v in (l, b, d))
    out = np.full(len(l), np.nan)
    for i in range(len(l)):
        if not np.isfinite(d[i]):
            continue
        s = np.arange(ds / 2, d[i], ds)
        lr, br = np.radians(l[i]), np.radians(b[i])
        x = sun[0] + s * np.cos(br) * np.cos(lr) / step
        y = sun[1] + s * np.cos(br) * np.sin(lr) / step
        z = sun[2] + s * np.sin(br) / step
        inside = (x >= 0) & (x <= cube.shape[2] - 1) & (y >= 0) & (y <= cube.shape[1] - 1) & (z >= 0) & (z <= cube.shape[0] - 1)
        if not inside.all():
            continue
        rho = map_coordinates(cube, np.vstack([z, y, x]), order=1)
        out[i] = rho.sum() * ds
    return out


_MI = None


def ms_curve(logt, plane):
    global _MI
    if _MI is None:
        _MI = core._read_mist()
    m = _MI
    ages = np.unique(m["log10_isochrone_age_yr"])
    t0 = ages[np.argmin(np.abs(ages - logt))]
    s = (m["log10_isochrone_age_yr"] == t0) & (m["phase"] == 0)
    G, BP, RP, K = m["Gaia_G_EDR3"][s], m["Gaia_BP_EDR3"][s], m["Gaia_RP_EDR3"][s], m["2MASS_Ks"][s]
    to = G.min()
    keep = G > to + 0.5
    if plane == "BPRP":
        c, mag = (BP - RP)[keep], G[keep]
    else:
        c, mag = (G - K)[keep], K[keep]
    o = np.argsort(c)
    return c[o], mag[o], t0


def delta_av(cobs, mobs, mod, av0, plane, logt, kG=None, grid=np.arange(-4.0, 6.0, 0.01)):
    """DeltaA_V (sobre av0) que pone la estrella sobre la MS. Raiz de f = M0 - M_MS(c0) mas cercana a 0; NaN si no hay."""
    E = dict(core.EXT)
    if kG is not None:
        E["G"] = kG
    if plane == "BPRP":
        ec, am = E["BP"] - E["RP"], E["G"]
    else:
        ec, am = E["G"] - E["Ks"], E["Ks"]
    cc, mm, _ = ms_curve(logt, plane)
    out = np.full(len(cobs), np.nan)
    A = av0 + grid
    for i in range(len(cobs)):
        if not (np.isfinite(cobs[i]) and np.isfinite(mobs[i])):
            continue
        c0 = cobs[i] - ec * A
        m0 = mobs[i] - mod - am * A
        ok = (c0 >= cc[0]) & (c0 <= cc[-1])
        fv = np.where(ok, m0 - np.interp(c0, cc, mm), np.nan)
        sgn = np.sign(fv)
        idx = np.where(np.isfinite(fv[:-1]) & np.isfinite(fv[1:]) & (sgn[:-1] != sgn[1:]))[0]
        if len(idx):
            j = idx[np.argmin(np.abs(grid[idx]))]
            out[i] = grid[j]
    return out
