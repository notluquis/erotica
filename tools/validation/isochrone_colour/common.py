"""Shared pieces of the colour-mechanism experiments (``colour_synth.py``, ``hr_fit.py``).

Every choice carries the process it models (methodology §A.1):

* **Grid**: MIST v1.2 (the grid of C1 and of the screen), [Fe/H] nodes -1.0...+0.5, the grid layer's
  own coordinate ([Fe/H] label, uniform prior -- not U(Z), see ``_isochrone.py``).
* **dm prior = the parallax** (the screen's ``plx`` arm, the one change with a consistent effect,
  ``isochrone-model-screen.md`` §4): N(10.223, 0.026) from the members' mean parallax with the
  10.3 uas floor. Its zero point is not modelled (+-0.02 on the bright stars, screen §1.7).
* **A_V in [0, 3]**: widened as in the screen (the C1 bound 0.5 bites).
* **log t in [5.7, 7.0]**: below the 6.0 bound on which three screen configurations sat.
* **Binaries**: Offner+2022 fraction as in C1 (the fitter's default).
"""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
NGC = Path("/Users/notluquis/erotica/data/test/NGC6383")
SAMPLE = NGC / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
MIST12 = NGC / "MIST/UBVRIplus"
CACHE = Path.home() / ".cache/erotica-grids"
EDR3 = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
FEH_NODES = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
PRI_PLX = dict(loga_range=(5.7, 7.0), Av_range=(0.0, 3.0), dm_mu=10.223, dm_sigma=0.026,
               dm_range=(9.5, 10.7))


def mist(bands=EDR3, feh=FEH_NODES, loga_range=(5.6, 7.1)):
    from erotica.analysis import grids as G

    return G.MISTGrid(MIST12, bands=bands, loga_range=loga_range).select(feh=feh)


def fit(f, rng_seed: int = 42) -> dict:
    """``find_start`` (lattice over every node + L-BFGS-B polish), as the real fits of the grid layer
    do it: no seeding at the truth (§A.1.3)."""
    t0 = time.time()
    s = f.find_start(2, np.random.default_rng(rng_seed))
    m = s["mode"]
    bounds = {"feh": f._met_bounds(), "loga": f.loga_range, "dm": f.dm_range, "Av": f.Av_range}
    at = [k for k, (lo, hi) in bounds.items()
          if k in m and hi > lo and min(m[k] - lo, hi - m[k]) < 1e-3 * (hi - lo)]
    return {"mode": m, "loglike": s["loglike"], "laplace_sd": s["local_sd"],
            "runner_up": s["runner_up"], "at_bound": at, "n": int(f._N_obs),
            "seconds": round(time.time() - t0, 1)}


def shifted(grid, dcol, name: str):
    """A copy of ``grid`` whose BP magnitude is moved by ``dcol(logte)`` at every point, so that
    BP-RP changes by exactly that amount at fixed T_eff while G and RP keep the grid's values.
    This is the "same stars, another colour-T_eff relation" world; G is not touched because the
    tested mechanism is the colour table, not the bolometric correction."""
    from erotica.analysis.grids.base import GridNode

    g = copy.copy(grid)
    g._nodes = {}
    for k, n in grid._nodes.items():
        mags = dict(n.mags)
        mags[EDR3[1]] = n.mags[EDR3[1]] + dcol(n.extra["logte"])
        g._nodes[k] = GridNode(n.eep, n.mass, mags, n.extra)
    g.name = grid.name + f" [{name}]"
    return g


def search(f, feh_lattice=(-1.0, -0.5, 0.0, 0.5), age_stride: int = 2, polish: int = 4,
           verbose: bool = True) -> dict:
    """``find_start``'s two stages with a coarser lattice: (dm, A_V) optimised at every
    (feh, log t) of ``feh_lattice`` x every ``age_stride``-th age node inside the prior, at the wide
    width (0.05 mag) where the likelihood is smooth; then the ``polish`` best are polished on all six
    parameters (L-BFGS-B, bounded). Same procedure for synthetic and real fits, never seeded at a
    truth (§A.1.3). Why coarser: ``find_start`` takes ~1000 s on 254 stars with 7 x 27 nodes
    (``ngc6383_grids.json``), and this experiment needs tens of fits on a shared 8 GB machine; the
    cost of coarsening is measured on the control arm (it must recover the truth)."""
    from scipy.optimize import minimize

    t0 = time.time()
    zlo, zhi = f._met_bounds()
    one_z = len(f._node_logz) == 1
    lo = np.array([zlo + 1e-9, f.loga_range[0], f.dm_range[0], f.Av_range[0], 1e-4, 1e-4])
    hi = np.array([zhi - 1e-9, f.loga_range[1], f.dm_range[1], f.Av_range[1], 0.5, 0.5])
    if one_z:
        lo[0] = hi[0] = zlo
    fl = f._compiled_loglike(None)

    def fn(y):
        # log L + the Gaussian dm prior (``_compiled_loglike`` is the likelihood alone; the fitter
        # adds priors only in its PyMC model). Without this term the "parallax" prior is not
        # applied at all: measured on the real sample, dm went to 10.355, 5 sigma off it.
        v, g = fl(y)
        z = (y[2] - f.dm_mu) / f.dm_sigma
        g = g.copy()
        g[2] -= z / f.dm_sigma
        return v - 0.5 * z * z, g

    ages = [a for a in f._node_loga if lo[1] <= a <= hi[1]][::age_stride]
    fehs = [zlo] if one_z else [x for x in feh_lattice if zlo <= x <= zhi]
    cands = []
    for z in fehs:
        for a in ages:
            def nll2(y, z=float(z), a=float(a)):
                v, g = fn([z, a, y[0], y[1], 0.05, 0.02])
                return -v, -g[2:4]

            r = minimize(nll2, [f.dm_mu, 1.0], jac=True, method="L-BFGS-B",
                         bounds=list(zip(lo[2:4], hi[2:4], strict=True)))
            cands.append((float(r.fun), float(z), float(a), *map(float, r.x)))
            if verbose:
                print(f"  lattice {len(cands)} z={z} a={a:.2f} nfev={r.nfev} t={time.time()-t0:.0f}s", flush=True)
    cands.sort(key=lambda c: c[0])

    def nll6(y):
        v, g = fn(y)
        return -v, -g

    pol = []
    for c in cands[:polish]:
        y0 = np.clip(np.array([*c[1:], 0.03, 0.02]), lo, hi)
        r = minimize(nll6, y0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)))
        pol.append((-float(r.fun), np.asarray(r.x, float)))
        if verbose:
            print(f"  polish nfev={r.nfev} ll={-r.fun:.2f} t={time.time()-t0:.0f}s", flush=True)
    pol.sort(key=lambda p: -p[0])
    ll, b = pol[0]
    names = ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")
    m = dict(zip(names, b.tolist(), strict=True))
    bounds = {"feh": (zlo, zhi), "loga": f.loga_range, "dm": f.dm_range, "Av": f.Av_range}
    at = [k for k, (l_, h_) in bounds.items()
          if hi[0] > lo[0] or k != "feh"
          if h_ > l_ and min(m[k] - l_, h_ - m[k]) < 1e-3 * (h_ - l_)]
    return {"mode": m, "logpost": ll, "loglike": fl(b)[0], "runner_up": [(p[0], p[1].tolist()) for p in pol[1:]],
            "at_bound": at, "n": int(f._N_obs), "n_lattice": len(cands),
            "seconds": round(time.time() - t0, 1)}
