#!/usr/bin/env python3
r"""Wilson (1975) vs King vs EFF on the NGC 6383 member list: a SMOKE CHECK, one cluster.

WHY THIS EXISTS
---------------
Thread G7 added the Wilson model to ``compare_radial_profiles``. McLaughlin & van der Marel (2005)
report Wilson fitting the majority of their clusters as well as or better than King; NGC 6383's flat
component is the reason to ask. This script says what the three families do on the one cluster this
package is built around -- and, **with the same pipeline, on two synthetic controls**, because a
Delta AIC from a profile family that cannot be told apart from its rival is not evidence.

WHAT IS COMPUTED
----------------
Maximum-likelihood fits of the *same* unbinned point-process likelihood (Cash 1979) for the three
families, each with four free parameters (amplitude, background, scale, shape), so
``Delta AIC = -2 Delta lnL`` and ``Delta BIC`` differ only through ``k ln N`` (equal ``k`` -> equal
penalty): the two ranks coincide here by construction, not by finding. Maxima use several starts
(the Wilson surface has the W0 <-> r_0 ridge).

* sample: ``paperfaithful_reference_p06.ecsv`` (p > 0.6 members, N = 628 inside 70'), read from the
  hub-side data directory;
* controls: 8 synthetic samples each from a **King** truth and from a **Wilson** truth with the
  same N and field. If the Delta lnL distribution on Wilson-truth data does not separate from that
  on King-truth data, the NGC 6383 number cannot discriminate and is reported as such.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
"Wilson is preferred" is dead if the Delta lnL (Wilson - King) of the observed sample lies inside
the range spanned by the King-truth controls. "No preference" is dead if it lies outside it.

CAVEATS stated in the output: (1) ONE cluster; (2) members are a membership-selected sample, so
``b`` is not field contamination (``KingPriors``); (3) the evidence-based comparison
(``compare_radial_profiles``) is separate and prior-dependent; ML AIC/BIC is not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.optimize import minimize

from erotica.analysis import wilson as w
from erotica.analysis.structure import (
    eff_expected_count,
    eff_surface_density,
    king_expected_count,
)

B = Path("/Users/notluquis/erotica/data/test/NGC6383")
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
FIELD = 70.0
MEMBERS = B / "comments_paper/radius_robustness/generated/70/paperfaithful_reference_p06.ecsv"
NPAR = 4


def _king_sigma(r, k, b, R_c, R_t):
    core = 1.0 / np.sqrt(1.0 + (r / R_c) ** 2)
    edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
    return np.where(r <= R_t, k * (core - edge) ** 2 + b, b)


def _nll_factory(family, r):
    logr = np.log(2.0 * np.pi * r)

    def nll(p):
        k, b = np.exp(p[0]), np.exp(p[1])
        if family == "king":
            R_c = np.exp(p[2])
            R_t = R_c + np.exp(p[3])
            sig = _king_sigma(r, k, b, R_c, R_t)
            lam = king_expected_count(k, b, R_c, R_t, FIELD)
        elif family == "eff":
            a, gam = np.exp(p[2]), 0.1 + np.exp(p[3])
            sig = eff_surface_density(r, k=k, b=b, a=a, gamma=gam)
            lam = eff_expected_count(k, b, a, gam, FIELD)
        else:
            r_0, W0 = np.exp(p[2]), p[3]
            if not (w.W0_MIN <= W0 <= w.W0_MAX):
                return 1e30
            sig = w.wilson_surface_density(r, k=k, r_0=r_0, W0=W0, b=b)
            lam = w.wilson_expected_count(k, b, r_0, W0, FIELD)
        if np.any(~(sig > 0)):
            return 1e30
        out = -(np.sum(logr + np.log(sig)) - lam)
        return float(out) if np.isfinite(out) else 1e30

    return nll


def _starts(family, n):
    k0 = np.log(max(n, 10) / (np.pi * 10.0**2))
    b0 = np.log(max(n, 10) / (np.pi * FIELD**2) * 0.1)
    if family == "king":
        return [
            [k0, b0, np.log(rc), np.log(rt)]
            for rc in (2.0, 5.0, 10.0)
            for rt in (10.0, 40.0, 100.0)
        ]
    if family == "eff":
        return [[k0, b0, np.log(a), np.log(g)] for a in (2.0, 6.0, 15.0) for g in (1.5, 3.0)]
    return [[k0, b0, np.log(r0), W] for r0 in (1.0, 3.0, 8.0) for W in (2.5, 5.0, 8.0)]


def max_loglik(family, r):
    nll = _nll_factory(family, r)
    best = None
    for s in _starts(family, r.size):
        res = minimize(
            nll,
            s,
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-7, "maxiter": 6000, "maxfev": 6000},
        )
        if best is None or res.fun < best.fun:
            best = res
    return -best.fun, best.x


def fit_all(r):
    out = {}
    for fam in ("king", "eff", "wilson"):
        ll, x = max_loglik(fam, r)
        out[fam] = {"lnL": ll, "x": [float(v) for v in x]}
    n = r.size
    for fam in out:
        out[fam]["AIC"] = 2 * NPAR - 2 * out[fam]["lnL"]
        out[fam]["BIC"] = NPAR * np.log(n) - 2 * out[fam]["lnL"]
    return out


def draw_wilson(seed, n_target, W0, r_0, b_frac, field):
    r_t, s, S, C = w.projected_profile(W0, 2)
    scale = r_0 * r_t
    s_f = min(1.0, field / scale)
    ctot = np.interp(s_f, s, C)
    bg_area = np.pi * field**2
    # solve k from the target count: n_target = 2 pi k scale^2 ctot + b pi field^2, b = b_frac * k
    k = n_target / (2 * np.pi * scale**2 * ctot + b_frac * bg_area)
    rng = np.random.default_rng(seed)
    n_cl = rng.poisson(2 * np.pi * k * scale**2 * ctot)
    r_cl = np.interp(rng.random(n_cl) * ctot, C, s) * scale
    r_bg = field * np.sqrt(rng.random(rng.poisson(b_frac * k * bg_area)))
    return np.concatenate([r_cl, r_bg])


def draw_king(seed, n_target, R_c, R_t, b_frac, field):
    grid = np.linspace(0.0, field, 20001)
    shape = _king_sigma(grid, 1.0, b_frac, R_c, R_t)
    cdf = np.concatenate(
        [
            [0.0],
            np.cumsum(
                0.5
                * (2 * np.pi * grid[1:] * shape[1:] + 2 * np.pi * grid[:-1] * shape[:-1])
                * np.diff(grid)
            ),
        ]
    )
    rng = np.random.default_rng(seed)
    n = rng.poisson(n_target)
    return np.interp(rng.uniform(0.0, cdf[-1], n), cdf, grid)


def radii_arcmin(table):
    sky = SkyCoord(np.asarray(table["ra"], float) * u.deg, np.asarray(table["dec"], float) * u.deg)
    return CENTER.separation(sky).to(u.arcmin).value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--controls", type=int, default=8)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()

    r = radii_arcmin(Table.read(MEMBERS))
    r = r[np.isfinite(r) & (r > 0) & (r <= FIELD)]
    obs = fit_all(r)
    result = {"n_stars": int(r.size), "field_radius_arcmin": FIELD, "observed": obs, "controls": {}}

    def dl(fits):
        return {
            "dlnL_wilson_minus_king": fits["wilson"]["lnL"] - fits["king"]["lnL"],
            "dlnL_wilson_minus_eff": fits["wilson"]["lnL"] - fits["eff"]["lnL"],
        }

    result["observed_delta"] = dl(obs)
    for name, gen in (
        ("king_truth", lambda s: draw_king(s, r.size, 4.0, 25.0, 0.002, FIELD)),
        ("wilson_truth", lambda s: draw_wilson(s, r.size, 4.0, 3.0, 0.002, FIELD)),
    ):
        rows = []
        for seed in range(args.controls):
            x = gen(100 + seed)
            x = x[x > 0]
            rows.append(dl(fit_all(x)) | {"n": int(x.size)})
        result["controls"][name] = rows
    args.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k != "controls"}, indent=1))
    for name, rows in result["controls"].items():
        a = np.array([row["dlnL_wilson_minus_king"] for row in rows])
        print(f"{name}: dlnL(W-K) min {a.min():.2f} median {np.median(a):.2f} max {a.max():.2f}")


if __name__ == "__main__":
    main()
