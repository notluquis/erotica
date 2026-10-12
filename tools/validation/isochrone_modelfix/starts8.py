#!/usr/bin/env python3
"""Chain starts for NUTS on an arm (``isochrone-model-fix.md`` hand-off), the same construction as
``iv_common.chain_starts`` but on the arm's own log posterior (``compiled8`` + dm prior): the
Laplace covariance at the arm's mode, the four starts moved along its (feh, dm, Av) regression line
on log t by START_DLOGA (+-0.06, +-0.17), nuisance at the mode x U(0.7, 1.4), local_sd = Laplace
marginal sd. ``python starts8.py <arm> <win> [fits/<arm>_<win>.json]`` -> ``starts_<arm>_<win>.json``
with PyMC variable names (feh, loga, dm, Av, sigma_int, f_bg, k, beta, sigma_c, tau)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402

PYMC = {"feh": "feh", "loga": "loga", "dm": "dm", "Av": "Av", "sG": "sigma_int", "f_bg": "f_bg",
        "k": "k", "beta": "beta", "sC": "sigma_c", "tau": "tau"}


def laplace_starts(f, arm, x, seed=11):
    """Laplace covariance of the arm's log posterior (log L + dm prior) at ``x`` and the four chain
    starts along its ridge line; ``f`` is any fitter (the real data or a synthetic replicate)."""
    names = A.free_names(arm)
    x = np.asarray(x, float)
    fl = f.compiled8(arm)

    def grad(y):
        v, g = fl(y)
        g = np.array(g, float).copy()
        g[2] -= (y[2] - f.dm_mu) / f.dm_sigma**2
        return g

    n = len(x)
    h = np.array([1e-3] * 4 + [1e-4] * (n - 4))
    H = np.empty((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = h[i]
        H[i] = (grad(x + e) - grad(x - e)) / (2 * h[i])
    S = np.linalg.inv(-0.5 * (H + H.T))
    v = S[:4, 1] / S[1, 1]
    lo = np.array([f._met_bounds()[0], f.loga_range[0], f.dm_range[0], f.Av_range[0]])
    hi = np.array([f._met_bounds()[1], f.loga_range[1], f.dm_range[1], f.Av_range[1]])
    margin = 1e-3 * (hi - lo)
    rng = np.random.default_rng(seed)
    starts = []
    for d in A.C.START_DLOGA:
        s = np.clip(x[:4] + d * v, lo + margin, hi - margin)
        st = {PYMC[k]: float(val) for k, val in zip(names[:4], s, strict=True)}
        for k, val in zip(names[4:], x[4:], strict=True):
            lo_, hi_ = A.WBOUNDS[k]
            st[PYMC[k]] = float(np.clip(val * rng.uniform(0.7, 1.4), lo_ + 1e-3, hi_ - 1e-3))
        starts.append(st)
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.diag(S))  # NaN where -H is not positive there (reported, not hidden)
    return {"arm": arm, "mode": {PYMC[k]: float(val) for k, val in zip(names, x, strict=True)},
            "local_sd": {PYMC[k]: float(val) for k, val in zip(names[:4], sd[:4], strict=True)
                         if np.isfinite(val)},
            "laplace_sd_all": {PYMC[k]: float(val) for k, val in zip(names, sd, strict=True)},
            "laplace_cov": S.tolist(), "ridge_vector": v.tolist(), "chain_starts": starts}


def main(arm, win, fitpath=None):
    real = A.C.real_table()
    f = A.make_fitter(real, win)
    fit = json.loads(Path(fitpath or HERE / "fits" / f"{arm}_{win}.json").read_text())
    out = laplace_starts(f, arm, fit["x"])
    out = {"arm": arm, "window": win, **{k: v for k, v in out.items() if k != "arm"}}
    A.C.dump(HERE / f"starts_{arm}_{win}.json", out)
    print(json.dumps({k: out[k] for k in ("mode", "laplace_sd_all")}))


if __name__ == "__main__":
    main(*sys.argv[1:])
