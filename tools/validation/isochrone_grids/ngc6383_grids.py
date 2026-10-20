#!/usr/bin/env python3
"""NGC 6383 with every grid backend: the same 254 stars, the same likelihood, maximum-likelihood fits.

WHY THIS EXISTS
---------------
The grid layer exists so that the age of NGC 6383 is reported with the grid systematic beside it
(methodology §J.6), and so that the run registry (``tools/validation/isochrone_runs``) can show the
same cluster with several grids side by side. This produces the point estimates: ``find_start``
(lattice over the nodes, then L-BFGS-B on all six parameters) per grid, with Laplace widths at the
mode. **Not posteriors**: no NUTS was run here (memory budget, see the hub finding), and the C1
posterior for MIST v1.2 is the one in ``isochrone_unbinned/ngc_0.json``.

Arms (each a CONTROL of the others, same window inside an arm):

* ``main``: C1's sample and priors (log t 6.0-7.0, A_V 0.5-2.0, dm ~ N(10.3, 0.2) in 9.5-10.7), no
  window. MIST v1.2 (grid path, [Fe/H] free over -1.0...+0.5), MIST v2.5 ([Fe/H] -1.0...+0.25: the
  +0.5 file has no model below ~0.5 Msun at log t 6-7), PARSEC v1.2S (local CMD 3.7 file: [M/H]
  -0.18...+0.70 only -- it cannot reach C1's [Fe/H] ~ -0.39, which is declared, not hidden).
* ``pms``: [Fe/H] fixed at 0 in all grids, window from ``safe_window`` over MIST v1.2, BHAC15 and
  SPOTS f = 0 / 0.34 and the priors (Gaia G). SPOTS's Gaia is DR2 via empirical MS colours and the
  stars are DR3/EDR3 photometry: a passband mismatch that the comparison inherits.

WHAT WOULD FALSIFY ITS USE
--------------------------
A grid whose mode sits on a prior bound in log t or [Fe/H]: then its age is the prior's, and the
difference between grids is not a model systematic. Each fit records which parameters are within
1e-3 of a bound.

Output: ``ngc6383_grids.json``.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent))
CACHE = Path.home() / ".cache/erotica-grids"
NGC = Path.home() / "erotica/data/test/NGC6383"


def main() -> None:
    from astropy.table import QTable, Table
    from isochrone_nuts_convergence import PRIORS, SAMPLE

    import erotica
    from erotica.analysis import grids as G
    from erotica.analysis._isochrone import IsochroneFitter, _ccm89

    pri = {k: v for k, v in PRIORS.items() if k not in ("M_met", "M_loga")}
    data = QTable(Table.read(SAMPLE))
    lr = (5.9, 7.1)
    edr3 = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
    # grids are built lazily, one at a time, and dropped after their fit (8 GB machine shared
    # with other fits: holding v1.2 + v2.5 + PARSEC at once pushed this process past 0.7 GB)
    feh_main = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    arms = {
        "main": {
            "MIST v1.2": (
                lambda: G.MISTGrid(NGC / "MIST/UBVRIplus", bands=edr3, loga_range=lr).select(
                    feh=feh_main
                ),
                edr3,
            ),
            "MIST v2.5": (
                lambda: G.MISTGrid(
                    CACHE / "mist_v2.5/UBVRIplus_afe0_vvcrit0.4", bands=edr3, loga_range=lr
                ).select(feh=feh_main[:-1]),
                edr3,
            ),
            "PARSEC v1.2S": (
                lambda: G.PARSECGrid(NGC / "PARSEC/gaiaedr3", loga_range=lr, max_mass=20.0),
                ("Gmag", "G_BPmag", "G_RPmag"),
            ),
        },
        "pms": {
            "MIST v1.2 [Fe/H]=0": (
                lambda: G.MISTGrid(NGC / "MIST/UBVRIplus", bands=edr3, loga_range=lr).select(
                    feh=0.0
                ),
                edr3,
            ),
            "BHAC15": (
                lambda: G.BHAC15Grid(CACHE / "bhac15", loga_range=(5.6, 7.5)),
                ("G", "G_BP", "G_RP"),
            ),
            "SPOTS f=0": (
                lambda: G.SPOTSGrid(CACHE / "spots", 0.0, loga_range=lr),
                ("G_mag", "BP_mag", "RP_mag"),
            ),
            "SPOTS f=0.34": (
                lambda: G.SPOTSGrid(CACHE / "spots", 0.34, loga_range=lr),
                ("G_mag", "BP_mag", "RP_mag"),
            ),
        },
    }
    obs_faint = float(np.max(np.asarray(data["Gmag"], float)))
    out = {
        "erotica_file": erotica.__file__,
        "priors": {k: list(v) if isinstance(v, tuple) else v for k, v in pri.items()},
        "arms": {},
    }
    try:
        win = G.safe_window(
            [g() for g, _ in arms["pms"].values()],
            [b[0] for _, b in arms["pms"].values()],
            loga_range=pri["loga_range"],
            dm_range=pri["dm_range"],
            Av_range=pri["Av_range"],
            k_mag=_ccm89(6390.7),
            data_faint=obs_faint,
        )
        out["pms_window_G"] = list(win)
    except ValueError as exc:
        # An empty window is a result: no star of this sample is inside the mass range of the
        # PMS-only grids for every (dm, A_V) the priors allow. Recorded, and the arm is not run.
        out["pms_window_G"] = None
        out["pms_window_empty"] = str(exc)
        arms.pop("pms")
    print("pms window G", out["pms_window_G"], out.get("pms_window_empty"), flush=True)
    for arm, grids in arms.items():
        out["arms"][arm] = {}
        for name, (make, b) in grids.items():
            t0 = time.time()
            g = make()
            f = IsochroneFitter(grid=g, magnitude=b[0], color=(b[1], b[2]), **pri)
            f.setup(data, prob_threshold=0.0, mag_window=win if arm == "pms" else None)
            s = f.find_start(2, np.random.default_rng(42))
            m = s["mode"]
            bounds = {
                "feh": f._met_bounds(),
                "loga": pri["loga_range"],
                "dm": pri["dm_range"],
                "Av": pri["Av_range"],
            }
            at_bound = [
                k
                for k, (lo, hi) in bounds.items()
                if k in m and hi > lo and min(m[k] - lo, hi - m[k]) < 1e-3 * (hi - lo)
            ]
            out["arms"][arm][name] = {
                "mode": m,
                "loglike": s["loglike"],
                "laplace_sd": s["local_sd"],
                "runner_up": s["runner_up"],
                "at_prior_bound": at_bound,
                "n_stars": int(f._N_obs),
                "age_Myr": 10 ** m["loga"] / 1e6,
                "grid": g.describe(),
                "bands": list(b),
                "sigma_floor": f.SIGMA_FLOOR,
                "seconds": round(time.time() - t0, 1),
            }
            print(
                arm,
                name,
                json.dumps(
                    {
                        k: out["arms"][arm][name][k]
                        for k in ("mode", "age_Myr", "at_prior_bound", "n_stars", "seconds")
                    }
                ),
                flush=True,
            )
            (HERE / "ngc6383_grids.json").write_text(
                json.dumps(out, indent=1, default=float) + "\n"
            )
            del f, g
            gc.collect()


if __name__ == "__main__":
    main()
