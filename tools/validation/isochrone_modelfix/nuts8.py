#!/usr/bin/env python3
"""``isochrone_intervals/nuts_chain.py`` on an arm of ``isochrone-model-fix.md`` instead of X1's model:
``ARM=K WINDOW=W python nuts8.py run|finalize <nuts_chain arguments>``. Same kernel, chunking,
resume and summaries; only the model (``aniso.build_model8``) and the summarised parameter list
(+ the arm's extra variables) change. ``--data`` is the table the model is built on (a synthetic
replicate or the 254 members), cut to the arm's window by ``setup``."""
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "isochrone_intervals"))
import aniso as A  # noqa: E402
import nuts_chain as N  # noqa: E402

ARM, WIN = os.environ.get("ARM", "K"), os.environ.get("WINDOW", "W")
EXTRA = {"k": "k", "beta": "beta", "sC": "sigma_c", "tau": "tau"}
# ZFIX=<[Fe/H] on MIST's A09 scale>: the arm with [Fe/H] fixed (the spectroscopic value), hub finding
# isochrone-intervals-nuts.md §2. The age it gives is conditional on that Z: it carries no [Fe/H]
# error. The starts file must then not carry a "feh" entry (kw_cells.py zstarts).
ZFIX = os.environ.get("ZFIX")


def build_model8_zfix(f, arm, feh):
    """``aniso.build_model8`` with the metallicity a constant instead of U(grid bounds)."""
    import pymc as pm
    import pytensor.tensor as pt

    zlo, zhi = f._met_bounds()
    if not zlo <= feh <= zhi:
        raise SystemExit(f"ZFIX {feh} outside the grid's [Fe/H] range [{zlo}, {zhi}]")
    free = A.ARM_FREE[arm]
    with pm.Model() as model:
        met = pt.constant(float(feh))
        loga = pm.Uniform("loga", lower=f.loga_range[0], upper=f.loga_range[1])
        dm = pm.TruncatedNormal("dm", mu=f.dm_mu, sigma=f.dm_sigma, lower=f.dm_range[0],
                                upper=f.dm_range[1])
        Av = pm.Uniform("Av", lower=f.Av_range[0], upper=f.Av_range[1])
        sG = pm.HalfNormal("sigma_int", sigma=0.05)
        f_bg = pm.Beta("f_bg", alpha=1.0, beta=19.0)
        v = {"sG": sG}
        if "k" in free:
            v["k"] = pm.Uniform("k", lower=1.0, upper=6.0)
        if "beta" in free:
            v["beta"] = pm.Uniform("beta", lower=-2.0, upper=3.0)
        if "sC" in free or "tau" in free:
            raise SystemExit("ZFIX is wired for the isotropic arms only")
        full = A._fill(arm, v, 0.0 * sG)
        w = f._star_weights

        def _logp(value, loga, dm, Av, sG, sC, tau, f_bg, k, beta):
            return w * f.star_loglike10(met, loga, dm, Av, sG, sC, tau, f_bg, k, beta, pt)

        pm.CustomDist("y", loga, dm, Av, full["sG"], full["sC"], full["tau"], f_bg, full["k"],
                      full["beta"], logp=_logp, observed=np.zeros(f._N_obs))
    return model


def _model(data_path):
    f = A.make_fitter(N._data(data_path), WIN)
    if ZFIX:
        return f, build_model8_zfix(f, ARM, float(ZFIX))
    return f, A.build_model8(f, ARM)


N._model = _model
N.C.PARAMS = tuple(N.C.PARAMS) + tuple(EXTRA[p] for p in A.ARM_FREE[ARM] if p in EXTRA)
if ZFIX:
    N.C.PARAMS = tuple(p for p in N.C.PARAMS if p != "feh")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "finalize"])
    for a, kw in [("--data", dict(required=True)), ("--starts", {}), ("--chain", dict(type=int, default=0)),
                  ("--outdir", {}), ("--outdirs", {}), ("--warmup", dict(type=int, default=1000)),
                  ("--draws", dict(type=int, default=1000)), ("--chunk", dict(type=int, default=25)),
                  ("--seed", dict(type=int, default=7)), ("--target-accept", dict(type=float, default=0.9)),
                  ("--json", {}), ("--fig", {}), ("--truth", dict(default="")), ("--title", dict(default="")),
                  ("--save-draws", dict(default=""))]:
        ap.add_argument(a, **kw)
    a = ap.parse_args()
    N.run(a) if a.stage == "run" else N.finalize(a)
