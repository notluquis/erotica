#!/usr/bin/env python3
r"""Which likelihood does a King radial-density profile actually want?

WHY THIS EXISTS
---------------
The plan for EROTICA said: replace ``pm.Normal("obs_density", ...)`` in
``analysis/structure.py`` with a per-annulus **Poisson likelihood on counts**,
because "the Gaussian approximation fails in the sparse outer bins that set
R_t". Measuring the real NGC 6383 profile shows the premise is wrong: the paper
fits with ``method="equip"`` -- *equal-count* annuli -- so every bin holds 24-25
stars. There are no sparse outer bins.

That raises a sharper question. Under equal-count binning the number of stars per
annulus is **fixed by construction**; what varies between realizations is the
annulus *area*. A Poisson likelihood ``N_i ~ Poisson(rho(r_i) * A_i)`` asserts
that ``N_i`` is the random quantity with ``Var(N_i) = E[N_i]``. If the binner
already conditioned on ``N_i``, that assertion is false and the likelihood is
mis-specified -- the same "data used twice" defect as a data-dependent prior,
wearing a different hat.

THE ORACLE
----------
The Poisson dispersion index, ``Var(N_i) / E(N_i)``, which is **1 for a Poisson
variable by definition**. No golden numbers, no fitting: generate many
realizations of one known King point process, bin each, and measure the index
per bin. A binning scheme for which Poisson-on-counts is coherent must return
~1. Equal-count binning is predicted to return ~0.

USAGE
-----
    python tools/validation/king_binning_likelihood.py
    python tools/validation/king_binning_likelihood.py --n-realizations 500
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Truth. Chosen to resemble the published NGC 6383 fit (arcmin).
TRUE = {"k": 6.0, "b": 0.05, "R_c": 4.0, "R_t": 30.0}
R_FIELD = 70.0  # the Appendix D 70-arcmin extraction
N_BINS = 25


def king_surface_density(r, *, k, b, R_c, R_t):
    """King (1962) surface density, with an additive background."""
    r = np.asarray(r, dtype=float)
    core = 1.0 / np.sqrt(1.0 + (r / R_c) ** 2)
    edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
    return np.where(r <= R_t, k * (core - edge) ** 2 + b, b)


def sample_king_field(rng, *, r_field=R_FIELD, **params):
    """Draw one realization of the King inhomogeneous Poisson point process.

    The expected count is ``\\int 2 pi r Sigma(r) dr``; the total is Poisson about
    it and the radii follow ``p(r) ∝ 2 pi r Sigma(r)``. Sampling the *total* as
    Poisson rather than fixing it is what makes this a point process rather than
    a fixed-N sample -- and is exactly the property equal-count binning destroys.
    """
    grid = np.linspace(0.0, r_field, 20_001)
    intensity = 2.0 * np.pi * grid * king_surface_density(grid, **params)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (intensity[1:] + intensity[:-1]) * np.diff(grid))])
    expected = cdf[-1]
    n = rng.poisson(expected)
    return np.interp(rng.uniform(0.0, cdf[-1], n), cdf, grid), expected


def bin_equal_count(radii, n_bins):
    """The package default (``method="equip"``): edges at quantiles of the data."""
    s = np.sort(radii)
    edges = np.interp(np.linspace(0, len(s) - 1, n_bins + 1), np.arange(len(s)), s)
    return edges


def bin_fixed_width(radii, n_bins, r_field=R_FIELD):
    """Edges fixed in advance, independent of the realization."""
    return np.linspace(0.0, r_field, n_bins + 1)


def counts_and_areas(radii, edges):
    counts = np.histogram(radii, bins=edges)[0]
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    return counts.astype(float), areas


def run(n_real=400, seed=20260727, n_bins=N_BINS):
    rng = np.random.default_rng(seed)
    schemes = {"equal-count (current)": bin_equal_count, "fixed-width": bin_fixed_width}
    counts = {name: [] for name in schemes}
    areas = {name: [] for name in schemes}

    for _ in range(n_real):
        radii, _ = sample_king_field(rng, **TRUE)
        for name, binner in schemes.items():
            c, a = counts_and_areas(radii, binner(radii, n_bins))
            counts[name].append(c)
            areas[name].append(a)

    out = {"true": TRUE, "r_field": R_FIELD, "n_bins": n_bins,
           "n_realizations": n_real, "seed": seed, "schemes": {}}
    for name in schemes:
        c = np.asarray(counts[name])   # (n_real, n_bins)
        a = np.asarray(areas[name])
        mean_c = c.mean(axis=0)
        # Poisson dispersion index: Var(N_i)/E(N_i), equal to 1 for a Poisson variable.
        with np.errstate(divide="ignore", invalid="ignore"):
            dispersion = np.where(mean_c > 0, c.var(axis=0) / mean_c, np.nan)
            area_cv = a.std(axis=0) / np.where(a.mean(axis=0) > 0, a.mean(axis=0), np.nan)
        out["schemes"][name] = {
            "mean_dispersion_index": float(np.nanmean(dispersion)),
            "dispersion_by_bin": [float(x) for x in dispersion],
            "mean_counts_by_bin": [float(x) for x in mean_c],
            "mean_area_cv": float(np.nanmean(area_cv)),
            "min_mean_count": float(np.nanmin(mean_c)),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-realizations", type=int, default=400)
    ap.add_argument("--n-bins", type=int, default=N_BINS)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()
    res = run(n_real=args.n_realizations, n_bins=args.n_bins)

    print(f"King point process: k={TRUE['k']} b={TRUE['b']} R_c={TRUE['R_c']}' "
          f"R_t={TRUE['R_t']}' over a {R_FIELD}' field, {res['n_realizations']} realizations\n")
    print("Poisson dispersion index Var(N_i)/E(N_i) -- equals 1 for a Poisson count.\n")
    print(f"{'binning':24s} {'mean disp.':>11s} {'min E[N_i]':>11s} {'area CV':>9s}  verdict")
    for name, s in res["schemes"].items():
        d = s["mean_dispersion_index"]
        verdict = "Poisson OK" if 0.8 < d < 1.25 else "NOT Poisson"
        print(f"{name:24s} {d:11.3f} {s['min_mean_count']:11.1f} {s['mean_area_cv']:8.1%}  {verdict}")

    eq = res["schemes"]["equal-count (current)"]
    print(f"\nequal-count dispersion by bin: "
          f"{[round(x, 3) for x in eq['dispersion_by_bin'][:6]]} ... "
          f"{[round(x, 3) for x in eq['dispersion_by_bin'][-3:]]}")
    print(f"equal-count E[N_i] by bin:     "
          f"{[round(x, 1) for x in eq['mean_counts_by_bin'][:6]]} ... "
          f"{[round(x, 1) for x in eq['mean_counts_by_bin'][-3:]]}")

    args.out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
