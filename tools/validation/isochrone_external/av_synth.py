#!/usr/bin/env python3
"""Synthetic with known truth and STRUCTURED A_V (hub finding ``isochrone-external-constraints.md``
§0.5, pre-registered there before any age was read).

Generator: the grid layer's forward model (``IsochroneFitter._draw_stars``: Chabrier IMF, Offner
binaries, the real error model and completeness cut), MIST v1.2, N = 254. Truth: [Fe/H] 0, log t
6.40, A_V level 1.0, sigma_int 0.03, dm ~ N(10.223, 0.026) per replicate.

Structure: each synthetic star takes the position, and so the map's Delta A_V, of the real member
with the same RANK IN G (in the data the radius depends on G: BP-faint stars sit at a median 21' vs
9.6'); a random permutation would put the structure on the wrong part of the CMD (§A.1.3). The
completeness cut acts before the structure is added (|k_G Delta A| <~ 0.1 mag; declared).

Arms (``--arm``):
  ctl       truth Delta A = 0                 fit plain
  map-ign   truth = map Delta A               fit plain
  map-cor   truth = map Delta A               fit corrected with the map (covariate)
  grad-ign  truth = map + radial step         fit plain
  grad-map  truth = map + radial step         fit corrected with the map only
  pri       truth = map                       corrected + Gaussian prior on the A_V level, sd 0.10 (+) map
  pri2      as pri with the 0.10 doubled (the sweep x2 of the pre-registration)
Radial step: Delta A_V = (ring median of the colour residual - its star-weighted mean) / k_col,
with the screen's ring medians (+0.046, +0.024, +0.045, -0.051; ``isochrone-model-screen.md`` §3.1);
residual = observed - model, so positive is redder, i.e. more A_V. Mapping a CMD colour residual
to A_V along k_col alone ignores the G component of the reddening vector [I].

Fit: ``common.search`` (coarse lattice + polish, never seeded at the truth; dm prior = parallax) +
optional A_V prior (``with_av_prior``) + Laplace width from a finite-difference Hessian of the log
posterior at the mode (``laplace``). Main map: see ``main_map`` (pre-registered DECaPS, replaced
post hoc by Edenhofer+2024 after DECaPS failed the predictive check).

Output: ``av_synth_<arm>.json`` (rewritten after every replicate).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "isochrone_colour"))
from common import EDR3, PRI_PLX, SAMPLE, mist, search  # noqa: E402

TRUTH = {"feh": 0.0, "loga": 6.40, "Av": 1.0, "sigma_int": 0.03, "dm_mu": 10.223, "dm_sd": 0.026}
N_STARS = 254
RINGS = [(0, 5), (5, 10), (10, 20), (20, 41.5)]
RESID_RINGS = [0.046, 0.024, 0.045, -0.051]
ARMS = {
    "ctl": (None, "plain", None),
    "map-ign": ("map", "plain", None),
    "map-cor": ("map", "cor", None),
    "grad-ign": ("map+grad", "plain", None),
    "grad-map": ("map+grad", "cor", None),
    "pri": ("map", "cor", 0.10),
    "pri2": ("map", "cor", 0.20),
}


def main_map(per):
    """(column, sigma column, name) of the main map. Pre-registered order: DECaPS 3D if read, else
    Edenhofer+2024. POST HOC (2026-10-05, after reading the map values, before any fit): DECaPS
    fails the generator's predictive check -- k_col x its Delta A_V has a MAD of 0.48 mag across the
    members against 0.092 mag for the whole observed colour residual, and does not correlate with it
    (r = 0.08) -- so its 1' structure is not the members' reddening, and the fallback (Edenhofer)
    is used. Override with ``EXT_MAIN_MAP=decaps``."""
    import os

    if os.environ.get("EXT_MAIN_MAP") == "decaps":
        return "AV_decaps", "sAV_decaps", "decaps"
    return "AV_eden", "sAV_eden_ind", "eden"


def radial_step(r, kc):
    ring = np.full(r.size, -1)
    for k, (a, b) in enumerate(RINGS):
        ring[(r >= a) & (r < b)] = k
    assert (ring >= 0).all(), "a member outside the rings"
    col = np.array(RESID_RINGS)[ring]
    return (col - col.mean()) / kc


def with_av_prior(f, mu, sd):
    """Adds log N(A_V | mu, sd) to the likelihood that ``search`` optimises (and to its gradient)."""
    orig = f._compiled_loglike

    def cl(mode=None):
        fl = orig(mode)

        def g(y):
            v, gr = fl(y)
            z = (y[3] - mu) / sd
            gr = np.array(gr, float).copy()
            gr[3] -= z / sd
            return v - 0.5 * z * z, gr

        return g

    f._compiled_loglike = cl


def laplace(f, x):
    """sd of each of (feh, loga, dm, Av, sigma_int, f_bg) from -H^-1 of the log posterior that
    ``search`` maximises (likelihood + dm prior + the A_V prior if attached), H by central
    differences of the analytic gradient. A mode on a bound gives a one-sided curvature: flagged by
    ``search``'s ``at_bound``, not corrected."""
    fl = f._compiled_loglike(None)

    def grad(y):
        v, g = fl(y)
        g = np.array(g, float).copy()
        g[2] -= (y[2] - f.dm_mu) / f.dm_sigma**2
        return g

    x = np.asarray(x, float)
    h = np.array([1e-3, 1e-3, 1e-3, 1e-3, 1e-4, 1e-4])
    H = np.empty((6, 6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = h[i]
        H[i] = (grad(x + e) - grad(x - e)) / (2 * h[i])
    H = 0.5 * (H + H.T)
    try:
        C = np.linalg.inv(-H)
        sd = np.sqrt(np.where(np.diag(C) > 0, np.diag(C), np.nan))
    except np.linalg.LinAlgError:
        sd = np.full(6, np.nan)
    return dict(zip(("feh", "loga", "dm", "Av", "sigma_int", "f_bg"), sd.tolist(), strict=True))


def fit_table(G, C, eG, eC, dA, sA, kG, kc, correct: bool):
    from astropy.table import QTable

    if correct:
        G, C = G - kG * dA, C - kc * dA
        eG, eC = np.hypot(eG, kG * sA), np.hypot(eC, kc * sA)
    return QTable(
        {
            "Gmag": G,
            "G_BPmag": C,
            "G_RPmag": np.zeros_like(G),
            "e_Gmag": eG,
            "e_G_BPmag": eC,
            "e_G_RPmag": np.zeros_like(G),
            "e_BP_RP": eC,
            "probability_hdbscan": np.ones_like(G),
        }
    )


def run_fit(data, prior_sd=None, prior_mu=None):
    from erotica.analysis._isochrone import IsochroneFitter

    f = IsochroneFitter(grid=mist(), magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
    f.setup(data, prob_threshold=0.0)
    if prior_sd is not None:
        with_av_prior(f, prior_mu, prior_sd)
    res = search(f, verbose=False)
    m = res["mode"]
    res["laplace_sd"] = laplace(f, [m[k] for k in ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")])
    return f, res


def main() -> None:
    from astropy.table import QTable, Table

    import erotica
    from erotica.analysis._isochrone import IsochroneFitter

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--reps", type=int, default=8)
    a = ap.parse_args()
    truth_kind, fit_kind, pri = ARMS[a.arm]

    per = Table.read(HERE / "av_maps_perstar.ecsv")
    real = QTable(Table.read(SAMPLE))
    assert np.array_equal(np.asarray(per["source_id"]), np.asarray(real["source_id"]))
    col, scol, mname = main_map(per)
    A = np.asarray(per[col], float)
    dA_map = A - np.median(A)
    sA = np.asarray(per[scol], float) if scol else np.zeros_like(A)
    r = np.asarray(per["radius_arcmin"], float)
    Greal = np.asarray(real["Gmag"], float)

    gen = IsochroneFitter(grid=mist(), magnitude=EDR3[0], color=EDR3[1:], **PRI_PLX)
    gen.setup(real, prob_threshold=0.0)
    kG, kc = gen._kG, gen._k_col1
    step = radial_step(r, kc)
    sd_level = float(np.median(sA))
    out = {
        "erotica_file": erotica.__file__,
        "arm": a.arm,
        "truth": TRUTH,
        "main_map": mname,
        "map_column": col,
        "sigma_column": scol,
        "kG": kG,
        "k_col": kc,
        "dA_map_p16_p84": np.percentile(dA_map, [16, 84]).tolist(),
        "radial_step_AV_by_ring": [float(step[(r >= lo) & (r < hi)].mean()) for lo, hi in RINGS],
        "prior": None
        if pri is None
        else {"mu": TRUTH["Av"], "sd": float(np.hypot(sd_level, pri)), "systematic": pri},
        "reps": [],
    }
    path = HERE / f"av_synth_{a.arm}.json"
    order = np.argsort(Greal)
    for k in range(a.reps):
        rng = np.random.default_rng(2000 + k)  # same seeds in every arm: paired replicates
        dm = float(rng.normal(TRUTH["dm_mu"], TRUTH["dm_sd"]))
        G, C = gen._draw_stars(
            TRUTH["feh"], TRUTH["loga"], dm, TRUTH["Av"], TRUTH["sigma_int"], N_STARS, rng
        )
        m = order[np.argsort(np.argsort(G))]  # real member with the same G rank
        dA_true = np.zeros(N_STARS)
        if truth_kind in ("map", "map+grad"):
            dA_true = dA_true + dA_map[m]
        if truth_kind == "map+grad":
            dA_true = dA_true + step[m]
        G, C = G + kG * dA_true, C + kc * dA_true
        eG, eC = gen._e_mag_fn(G), gen._e_col_fn(G)
        data = fit_table(G, C, eG, eC, dA_map[m], sA[m], kG, kc, fit_kind == "cor")
        _, res = run_fit(data, None if pri is None else out["prior"]["sd"], TRUTH["Av"])
        mo = res["mode"]
        res.update(
            {
                "truth_dm": dm,
                "d_feh": mo["feh"] - TRUTH["feh"],
                "d_loga": mo["loga"] - TRUTH["loga"],
                "d_Av": mo["Av"] - TRUTH["Av"],
                "gencheck_spearman_r_G": float(
                    __import__("scipy.stats").stats.spearmanr(r[m], G)[0]
                ),
            }
        )
        for p in ("feh", "loga"):
            s = res["laplace_sd"][p]
            res[f"cover_{p}"] = bool(np.isfinite(s) and abs(res[f"d_{p}"]) <= s)
        out["reps"].append(res)
        print(
            a.arm,
            k,
            json.dumps(
                {q: res[q] for q in ("d_feh", "d_loga", "d_Av", "at_bound", "seconds")},
                default=float,
            ),
            {p: round(res["laplace_sd"][p], 4) for p in ("feh", "loga")},
            flush=True,
        )
        path.write_text(json.dumps(out, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
