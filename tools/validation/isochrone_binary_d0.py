#!/usr/bin/env python3
r"""D0: where does the isochrone likelihood's age bias on binary synthetic clusters come from?

WHAT QUESTION THIS SETTLES
--------------------------
C3 of the hub finding ``isochrone-nuts-convergence-2026-09.md`` failed on binary synthetic clusters:
the four binary fits put log t at +0.013 to +0.024 dex above the truth (6.55), outside the 90 %
interval, while the six single-star fits recovered it (§10.15). The generator
(``isochrone_nuts_convergence.synthetic_cluster``) and the likelihood share the binary physics
(Offner fraction, D&K mass ratios, companion floor), so the bias must come from how the
likelihood *computes* the binary population. Pre-registered in §10.17 before this was run.

METHOD
------
Asymptotic, no sampling. Draw ~5·10³ stars from the generator at the batch-A binary truth and
evaluate the model's mean per-star log-likelihood ℓ(θ) on them, as probe stars with their own
errors, in a fitter set up on seed 1's data (so the completeness cut, error model and box are the
C3 fit's). If the model's density were the generator's, argmax ℓ would be the truth. The profile
of ℓ in log t (Z, dm, A_V re-optimised at each log t; σ_int = 0, f_bg fixed at 1e-3) gives the
**asymptotic log t** the model converges to on infinite data from this generator. One variant per
hypothesis, each changing one thing:

* ``V0``   production;
* ``V1``   binary sheet on every EEP point (``BINARY_STRIDE = 1``)              -- H1, primary;
* ``V2``   mass-ratio nodes doubled (midpoints added)                           -- H2, shape in q;
* ``V3``   D&K γ(m) as a step function, as the generator draws it              -- H3;
* ``CTRL`` single-star generator against the single-star model: must give 6.55.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
CTRL away from 6.55 by more than 0.003 dex means the method itself is biased (then nothing
below is attributed). A variant is the cause if it removes at least half of V0's offset. If none
does, H1-H3 are refuted and the cause is elsewhere.

USAGE
-----
    PYTHONPATH=~/erotica python tools/validation/isochrone_binary_d0.py V0
writes ``isochrone_binary_d0/<variant>.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from astropy.table import QTable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from isochrone_nuts_convergence import synthetic_cluster  # noqa: E402
from isochrone_unbinned_recovery import MIST, PRIORS  # noqa: E402

from erotica.analysis import _isochrone as iso_mod  # noqa: E402
from erotica.analysis._isochrone import IsochroneFitter  # noqa: E402

OUT = HERE / "isochrone_binary_d0"
TRUTH = dict(Z=0.014286, loga=6.55, dm=10.45, Av=1.24)
# 5000 probes: argmax noise ~ 0.008 dex * sqrt(254 / 5000) ~ 0.002 dex; 20 000 took ~7 min
# per profile point on the shared machine (measured), too slow for five variants
N_PROBE = 5_000
CHUNK = 2_000
LOGA_GRID = np.round(np.arange(6.530, 6.6001, 0.005), 3)
# a small fixed field fraction: with f_bg = 0, one probe star whose density underflows off the
# node (mmag errors, a fast phase) makes log L = -inf and every off-node profile point NaN
# (measured on the first run). Same value in every variant and in the control; the C3 posteriors
# have f_bg ~ 0.002.
F_BG = 1e-3


def _probe_table(binaries: bool) -> QTable:
    # a different seed from every C3 fit (seeds 1-10), drawn in one go
    return synthetic_cluster(
        TRUTH["Z"],
        TRUTH["loga"],
        TRUTH["dm"],
        TRUTH["Av"],
        777,
        binaries=binaries,
        n_target=N_PROBE,
    )


def main(variant: str) -> dict:
    t0 = time.time()
    binaries = variant != "CTRL"
    if variant == "V2":
        q = iso_mod._Q_NODES
        iso_mod._Q_NODES = np.unique(np.r_[q, 0.5 * (q[1:] + q[:-1])])
    if variant == "V3":
        # the generator's step function, written with xp.where so it also works on the
        # symbolic graph (the package's non-smooth branch is NumPy-only: first attempt failed)
        def step(m, xp=np, smooth=False):
            w = xp.where
            return w(m <= 0.1, 4.2, w(m <= 0.6, 0.4, w(m <= 1.4, 0.3, w(m <= 6.5, -0.5, 0.0))))

        iso_mod._dk_gamma = step
    kw = {} if binaries else {"alpha": 0.0, "beta": 0.0}
    f = IsochroneFitter(isochs_path=MIST, **{**PRIORS, **kw})
    if variant == "V1":
        f.BINARY_STRIDE = 1
    seed_data = synthetic_cluster(
        TRUTH["Z"], TRUTH["loga"], TRUTH["dm"], TRUTH["Av"], 1, binaries=binaries
    )
    f.setup(seed_data, prob_threshold=0.0)

    probe = _probe_table(binaries)
    g = np.asarray(probe["Gmag"], float)
    c = np.asarray(probe["G_BPmag"], float) - np.asarray(probe["G_RPmag"], float)
    eg = np.asarray(probe["e_Gmag"], float)
    ec = np.hypot(np.asarray(probe["e_G_BPmag"], float), np.asarray(probe["e_G_RPmag"], float))
    keep = g <= f._mag_lim  # the model conditions on detection brighter than its cut
    g, c, eg, ec = g[keep], c[keep], eg[keep], ec[keep]
    n = g.size

    # one compiled log-likelihood per chunk of probes (memory), summed
    fns = []
    for s in range(0, n, CHUNK):
        sl = slice(s, s + CHUNK)
        f._obs_mag, f._obs_col = g[sl], c[sl]
        f._e_obs_mag, f._e_obs_col = eg[sl], ec[sl]
        f._star_weights = np.ones(g[sl].size)
        fns.append(f._compiled_loglike("JAX"))

    def ell(x6: np.ndarray) -> tuple[float, np.ndarray]:
        v, gr = 0.0, np.zeros(6)
        for fn in fns:
            a, b = fn(x6)
            v += a
            gr += b
        return v / n, gr / n

    from scipy.optimize import minimize

    zlo, zhi = 10 ** float(f._node_logz[0]) * (1 + 1e-9), 10 ** float(f._node_logz[-1]) * (1 - 1e-9)
    bounds = [(zlo, zhi), f.dm_range, f.Av_range]
    y = np.array([TRUTH["Z"], TRUTH["dm"], TRUTH["Av"]])
    prof = []
    for la in LOGA_GRID:

        def nll(yy: np.ndarray, la: float = float(la)) -> tuple:
            v, gr = ell(np.array([yy[0], la, yy[1], yy[2], 0.0, F_BG]))
            return -v, -gr[[0, 2, 3]]

        best = minimize(nll, y, jac=True, method="L-BFGS-B", bounds=bounds)  # warm start
        y = best.x
        prof.append(
            {
                "loga": float(la),
                "ell": -float(best.fun),
                "met": float(y[0]),
                "dm": float(y[1]),
                "Av": float(y[2]),
            }
        )
        print(
            f"{variant} loga {la:.3f} ell {-best.fun:.6f} met {y[0]:.5f} dm {y[1]:.4f} "
            f"Av {y[2]:.4f}",
            flush=True,
        )

    ells = np.array([p["ell"] for p in prof])
    k = int(np.argmax(ells))
    # parabola through the best grid point and its neighbours (interior only)
    if 0 < k < len(ells) - 1:
        x3, y3 = LOGA_GRID[k - 1 : k + 2], ells[k - 1 : k + 2]
        a2, a1, _ = np.polyfit(x3, y3, 2)
        peak = float(-a1 / (2 * a2))
    else:
        peak = float(LOGA_GRID[k])
    res = {
        "variant": variant,
        "n_probe": int(n),
        "truth": TRUTH,
        "profile": prof,
        "argmax_grid": float(LOGA_GRID[k]),
        "argmax_parabola": peak,
        "offset_from_truth": peak - TRUTH["loga"],
        "ell_at_truth_minus_max": float(ells[list(LOGA_GRID).index(6.55)] - ells.max()),
        "seconds": round(time.time() - t0),
    }
    OUT.mkdir(exist_ok=True)
    (OUT / f"{variant}.json").write_text(json.dumps(res, indent=1) + "\n")
    return res


if __name__ == "__main__":
    r = main(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != "profile"}, indent=1))
