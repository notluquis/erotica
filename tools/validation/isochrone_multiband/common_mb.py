"""Shared pieces of the multiband experiments (``colour_synth_mb.py``, ``real_fit.py``).

Every choice carries the process it models (methodology §A.1); the ones inherited from the colour
experiment are stated in ``tools/validation/isochrone_colour/common.py`` and not repeated:

* **Grid**: MIST v1.2, [Fe/H] nodes -1.0...+0.5, bands Gaia EDR3 G, BP, RP **and 2MASS J, H, Ks**
  from the same UBVRIplus files (same rows, so no join is needed).
* **Near-infrared observations**: ``ngc6383_multiband_members.ecsv`` (``build_members.py``):
  ``*_fit`` in the 2MASS system; VIRAC2-sourced magnitudes carry the zero-point parameter
  N(0, 0.02) (Gonzalez-Fernandez+2018 precision). The field-plateau offsets measured for VIRAC2 -
  2MASS->VISTA (J +0.022, H -0.004, Ks -0.015; hub finding ``virac2-gaia-2mass-calibration.md``)
  are **not applied**: they lie within 1.1 sd of that prior and the fitted zero points are compared
  with them afterwards, as an external check.
* **Priors**: ``common.PRI_PLX`` (dm = parallax N(10.223, 0.026), A_V in [0, 3], log t in
  [5.7, 7.0]).
* **Search**: ``search_mb``, the colour experiment's ``common.search`` (lattice over
  ``feh_lattice`` x every ``age_stride``-th age at the wide width, (dm, A_V) optimised; then the
  ``polish`` best polished on every parameter) on the full multiband parameter vector, with the dm
  and zero-point priors added. Never seeded at a truth (§A.1.3).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "isochrone_colour"))
sys.path.insert(0, str(HERE.parent.parent.parent))
from common import EDR3, FEH_NODES, MIST12, PRI_PLX  # noqa: E402

from erotica.analysis._isochrone_multiband import NIRBand  # noqa: E402

MEMBERS = HERE / "ngc6383_multiband_members.ecsv"
TMASS = ("2MASS_J", "2MASS_H", "2MASS_Ks")
NIR = (
    NIRBand("2MASS_J", "J_fit", "eJ_fit", 12350.0, "J", "src_J", ("VIRAC2->2MASS",)),
    NIRBand("2MASS_H", "H_fit", "eH_fit", 16620.0, "H", "src_H", ("VIRAC2->2MASS",)),
    NIRBand("2MASS_Ks", "Ks_fit", "eKs_fit", 21590.0, "K", "src_Ks", ("VIRAC2->2MASS",)),
)


def mist_mb(feh=FEH_NODES, loga_range=(5.6, 7.1)):
    from erotica.analysis import grids as G

    return G.MISTGrid(MIST12, bands=EDR3 + TMASS, loga_range=loga_range).select(feh=feh)


def members():
    from astropy.table import QTable, Table

    return QTable(Table.read(MEMBERS))


def nir_pattern(f) -> dict:
    """The real members' missing-band and zero-point-group pattern, for ``draw_stars_nd``."""
    return {"G": f._obs_mag.copy(), "has": list(f._nir_has), "group": list(f._nir_group)}


def search_mb(
    f,
    feh_lattice=(-1.0, -0.5, 0.0, 0.5),
    age_stride: int = 2,
    polish: int = 4,
    verbose: bool = True,
    fixed: dict | None = None,
) -> dict:
    """MAP of the multiband fitter (see the module docstring). ``fixed`` = {name: value} holds
    parameters at a value (a profile point): they are given equal bounds."""
    from scipy.optimize import minimize

    t0 = time.time()
    names = f.param_names
    zlo, zhi = f._met_bounds()
    one_z = len(f._node_logz) == 1
    lo = np.array([zlo + 1e-9, f.loga_range[0], f.dm_range[0], f.Av_range[0], 1e-4, 1e-4])
    hi = np.array([zhi - 1e-9, f.loga_range[1], f.dm_range[1], f.Av_range[1], 0.5, 0.5])
    if one_z:
        lo[0] = hi[0] = zlo
    x_lo, x_hi, x0_extra = list(lo), list(hi), []
    if f.fit_sigma_av:
        x_lo.append(0.0)
        x_hi.append(1.5)
        x0_extra.append(0.05)
    for _ in f.zp_bands:
        x_lo.append(-0.15)
        x_hi.append(0.15)
        x0_extra.append(0.0)
    x_lo, x_hi = np.array(x_lo), np.array(x_hi)
    for k, v in (fixed or {}).items():
        i = names.index(k)
        x_lo[i] = x_hi[i] = float(v)
    fl = f._compiled_loglike_nd(None)

    def fn(y):
        v, g = fl(y)
        z = (y[2] - f.dm_mu) / f.dm_sigma
        g = g.copy()
        g[2] -= z / f.dm_sigma
        lp = -0.5 * z * z
        n0 = 7 if f.fit_sigma_av else 6
        for j in range(len(f.zp_bands)):
            lp -= 0.5 * (y[n0 + j] / f.zp_sigma) ** 2
            g[n0 + j] -= y[n0 + j] / f.zp_sigma**2
        return v + lp, g

    ages = [a for a in f._node_loga if x_lo[1] <= a <= x_hi[1]][::age_stride] or [
        0.5 * (x_lo[1] + x_hi[1])
    ]
    fehs = [zlo] if one_z else [x for x in feh_lattice if x_lo[0] <= x <= x_hi[0]] or [x_lo[0]]
    base_extra = np.array(x0_extra, float)
    for k, v in (fixed or {}).items():
        i = names.index(k)
        if i >= 6:
            base_extra[i - 6] = float(v)
    cands = []
    for z in fehs:
        for a in ages:
            y_fix = np.r_[z, a, 0.0, 0.0, 0.05, 0.02, base_extra]

            def nll2(y, y_fix=y_fix):
                yy = y_fix.copy()
                yy[2:4] = y
                v, g = fn(yy)
                return -v, -g[2:4]

            r = minimize(
                nll2,
                [f.dm_mu, 1.0],
                jac=True,
                method="L-BFGS-B",
                bounds=list(zip(x_lo[2:4], x_hi[2:4], strict=True)),
            )
            cands.append((float(r.fun), float(z), float(a), *map(float, r.x)))
            if verbose:
                print(
                    f"  lattice {len(cands)} z={z} a={a:.2f} t={time.time() - t0:.0f}s", flush=True
                )
    cands.sort(key=lambda c: c[0])

    def nll(y):
        v, g = fn(y)
        return -v, -g

    pol = []
    for c in cands[:polish]:
        y0 = np.clip(np.r_[c[1:], 0.03, 0.02, base_extra], x_lo, x_hi)
        r = minimize(
            nll, y0, jac=True, method="L-BFGS-B", bounds=list(zip(x_lo, x_hi, strict=True))
        )
        pol.append((-float(r.fun), np.asarray(r.x, float)))
        if verbose:
            print(f"  polish ll={-r.fun:.2f} t={time.time() - t0:.0f}s", flush=True)
    pol.sort(key=lambda p: -p[0])
    lp, b = pol[0]
    m = dict(zip(names, b.tolist(), strict=True))
    at = [
        k
        for i, k in enumerate(names)
        if x_hi[i] > x_lo[i]
        and min(b[i] - x_lo[i], x_hi[i] - b[i]) < 1e-3 * (x_hi[i] - x_lo[i])
        and k not in ("sigma_int", "f_bg")
    ]
    return {
        "mode": m,
        "logpost": lp,
        "loglike": fl(b)[0],
        "runner_up": [(p[0], p[1].tolist()) for p in pol[1:]],
        "at_bound": at,
        "n": int(f._N_obs),
        "n_lattice": len(cands),
        "seconds": round(time.time() - t0, 1),
    }


def fitter(grid, nir=(), **kw):
    from erotica.analysis._isochrone_multiband import MultibandIsochroneFitter

    return MultibandIsochroneFitter(
        grid=grid, magnitude=EDR3[0], color=EDR3[1:], nir_bands=nir, **{**PRI_PLX, **kw}
    )
