#!/usr/bin/env python3
r"""Calibrate the finite-sample bias of the EFF slope, as a function of N and of the true slope.

WHY THIS EXISTS — IT BLOCKS THE CENSUS SWEEP
--------------------------------------------
The planned A5 sweep fits EFF to every cluster in Hunt & Reffert (2024) and asks whether the slope
`gamma` piles up near 2, the value at which EFF and an untruncated King coincide. That question is
only answerable if the estimator is unbiased, or if its bias is known.

**It is not unbiased.** On NGC 6383's geometry the smooth control recovers `gamma = 2.395` against an
injected 2.32 -- `+0.075 +/- 0.012`, 6 sigma, with no substructure, no selection effect and no
contamination. A first pass over three sample sizes gave `+0.061 / +0.031 / +0.011` at
`N = 628 / 2512 / 10048`, i.e. roughly `N^-0.6`.

The census spans four decades in N. **A bias that shrinks with N would masquerade as physics**: poor
clusters would appear systematically steeper than rich ones, and any statement about clustering near
`gamma = 2` would be contaminated by a selection-independent artefact. This script measures the
surface `bias(N, gamma_true)` so the sweep can correct for it -- or, if the bias turns out to depend
on the truth in a way that cannot be inverted, so the sweep can be scoped honestly.

WHAT IS ALREADY RULED OUT
-------------------------
**Not a sampler artefact.** The inverse-CDF EFF radius sampler was checked against the analytic EFF
CDF first: maximum deviation `7e-4` over 400,000 draws across `r = 0.5-70` arcmin. The verification
order was generator -> estimator -> interpretation, and the generator passed, which is what turned a
suspected bug into a measurable property of the estimator.

**Not the priors — the control was run and it passed.** `EFFPriors` are fixed scale-free constants
independent of the data, and the `--flat-prior` switch re-runs with a 2.5x wider `gamma` prior. The
fitted laws are unchanged:

    gamma_true   default             flat prior
    2.00         A=11.500 p=0.715    A=10.840 p=0.708
    2.32         A= 8.905 p=0.777    A= 8.352 p=0.770
    3.00         A=13.673 p=0.775    A=13.947 p=0.777

So the bias is driven by the likelihood, not by prior pull, and **it cannot be removed by widening the
prior.** (This also matches the arithmetic: with a Normal(3.0, 2.0) prior and a likelihood width of
0.086 at N=628, the prior's weight is 0.0018 and its pull is ~0.001 -- two orders below the measured
bias.)

BIAS(N) IS THE WRONG PARAMETERISATION — MEASURED, NOT SUSPECTED
---------------------------------------------------------------
The first pass fixed the geometry at NGC 6383's: ``a = 1.65'`` in a 70' field, a footprint-to-scale
ratio of **42**. The census does not live there: Hunt & Reffert's ``r_tot/r_c`` runs from below 2 to
about 10, with a 25th percentile of 1.89. Holding ``N = 60`` (the census median is 61) and
``gamma_true = 2.00`` and varying **only** the footprint:

    r_tot/a      bias
       2      +1.627 +/- 0.035
       4      +1.931 +/- 0.056
       8      +1.753 +/- 0.112
      16      +1.214 +/- 0.160
      42      +0.593 +/- 0.111   <- where the calibration had been done

**At the ratios the census actually has, a true gamma = 2 is measured as 3.6-3.9** -- beyond the whole
range the literature reports. The mechanism is a lever arm: ``gamma`` is a log-log slope and needs
dynamic range in ``log r``. At ``r_tot/a = 2`` there is 0.3 dex of it; at 42 there is 1.6 dex. And the
census sits at the low end *because* ``r_tot`` is defined by contrast against the field rather than by
the cluster's own scale, so the footprint shrinks exactly where the cluster is faint -- which is where
``N`` is small too, confounding the two axes.

Use ``--field-ratios`` to map the ``(N, r_tot/a)`` plane. The deliverable is a **recoverability
boundary**, not a correction surface: the bias is non-monotonic below ``r_tot/a ~ 4``, where prior pull
and likelihood push compete, so a fitted law there would be meaningless.

THE POWER LAW IS A PLACEHOLDER — SEE eff_gamma_estimator.py FOR THE MODEL THAT HAS CONTENT
------------------------------------------------------------------------------------------
`bias = A * N^-p` is fitted here for convenience, and the `p ~ 0.72-0.78` it returns **has no
theoretical content**. Cox & Snell (1968) -- matrix form in Kosmidis (2014, `arXiv:1311.6311`) --
expand the maximum-likelihood bias as `b1/n + b2/n^2 + O(n^-3)`, so the leading term is `O(1/n)` and
`p` should be 1. It is not, and that is not a regularity failure of the point process: by the
conditioning property of a Poisson process, given `N` points in a window they are iid draws from the
window-normalised intensity, so for shape parameters Cox-Snell applies with `n = N`.

Model comparison on this grid at fixed geometry (reduced chi-square, `gamma_true` = 2.00/2.32/3.00):

    H1  bias = A N^-p              0.93 / 0.23 / 0.95     empirical, no content
    H2  bias = b1/N                8.72 / 2.06 / 4.90     Cox-Snell alone -- REJECTED
    H3  bias = A/sqrt(N) + b1/N    1.12 / 0.43 / 1.48     accepted

So `p ~ 0.75` is just the effective slope of a `1/sqrt(N)` term plus a `1/N` term over this range.
**Quote H3's coefficients, never `p`.**

USAGE
-----
    python tools/validation/eff_gamma_bias.py --realizations 16
    python tools/validation/eff_gamma_bias.py --realizations 16 --flat-prior

Writes ``eff_gamma_bias.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import EFFPriors, eff_unbinned

FIELD_RADIUS = 70.0  # arcmin, the NGC 6383 footprint
A_TRUE = 1.65  # arcmin
# The census is much poorer than intuition suggests: Hunt & Reffert 2024's open clusters have a
# median N near 60, with ~40% below 50. A grid that starts at 150 would force the power law to be
# extrapolated an order of magnitude below its calibration floor -- the exact failure recorded in
# methodology PART K.1.5, in the direction where it is most dangerous. Hence the low-N cells.
N_GRID = (20, 35, 60, 100, 150, 300, 628, 1250, 2500, 5000)
GAMMA_GRID = (2.0, 2.32, 3.0)


def eff_radii(rng, n, *, a=A_TRUE, gamma, field_radius=FIELD_RADIUS):
    """Inverse-CDF draw from ``Sigma(r) ~ (1 + (r/a)^2)^(-gamma/2)``, truncated at the field edge.

    Validated against the analytic CDF ``[(1+(r/a)^2)^(1-gamma/2) - 1] / (1-gamma/2)`` to 7e-4 on
    400,000 draws -- see the module docstring. The grid is fine enough that interpolation error is
    far below the bias being measured.
    """
    grid = np.linspace(0.0, field_radius, 200_001)
    pdf = 2.0 * np.pi * grid * (1.0 + (grid / a) ** 2) ** (-gamma / 2.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


def run_cell(n, gamma_true, realizations, seed, priors, cfg, field_radius=FIELD_RADIUS):
    medians, sds = [], []
    for i in range(realizations):
        rng = np.random.default_rng(seed + 1000 * int(gamma_true * 100) + i)
        r = eff_radii(rng, n, gamma=gamma_true, field_radius=field_radius)
        fit = eff_unbinned(r, field_radius=field_radius, priors=priors,
                           sampling=cfg, progressbar=False)
        medians.append(float(fit["gamma_median"]))
        sds.append(float(fit["gamma_std"]))
    m = np.array(medians)
    return dict(
        n=int(n),
        gamma_true=float(gamma_true),
        field_radius=float(field_radius),
        realizations=int(realizations),
        gamma_mean=float(m.mean()),
        bias=float(m.mean() - gamma_true),
        # SEM of the mean bias -- the number that decides whether a cell is significant
        bias_sem=float(m.std(ddof=1) / np.sqrt(m.size)),
        reported_sigma_mean=float(np.mean(sds)),
        realization_scatter=float(m.std(ddof=1)),
    )


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--flat-prior", action="store_true",
                    help="widen the gamma prior, to separate prior pull from likelihood bias")
    ap.add_argument("--n-values", type=int, nargs="+", default=None,
                    help="override N_GRID, e.g. to fill in the low-N cells without redoing the rest")
    ap.add_argument("--field-ratios", type=float, nargs="+", default=None,
                    help="map bias against r_tot/a at fixed N instead of against N; this is the axis "
                         "that actually controls recoverability")
    args = ap.parse_args()

    priors = EFFPriors(gamma_sigma=5.0) if args.flat_prior else EFFPriors()
    cfg = SamplingConfig(draws=1500, tune=1000, chains=2, random_seed=5, progressbar=False)

    rows = []
    for gamma_true in GAMMA_GRID:
        for n in (args.n_values or N_GRID):
            for ratio in (args.field_ratios or [FIELD_RADIUS / A_TRUE]):
                field = ratio * A_TRUE
                row = run_cell(n, gamma_true, args.realizations, args.seed, priors, cfg,
                               field_radius=field)
                row["field_over_a"] = float(ratio)
                rows.append(row)
                print(f"  gamma_true={gamma_true:.2f}  N={n:5d}  r_tot/a={ratio:5.1f}  "
                      f"bias = {row['bias']:+.4f} +/- {row['bias_sem']:.4f}", flush=True)

    # Power law through the origin in log-space, per gamma_true: bias = A * N^-p.
    # Fitted only on cells where the bias is resolved, otherwise the fit chases noise.
    fits = {}
    for gamma_true in GAMMA_GRID:
        cells = [r for r in rows if r["gamma_true"] == gamma_true
                 and r["bias"] > 0 and r["bias"] > 2 * r["bias_sem"]]
        if len(cells) < 3:
            fits[str(gamma_true)] = dict(resolved_cells=len(cells), note="too few resolved cells")
            continue
        x = np.log([c["n"] for c in cells])
        y = np.log([c["bias"] for c in cells])
        w = 1.0 / (np.array([c["bias_sem"] / c["bias"] for c in cells]) ** 2)
        p = np.polyfit(x, y, 1, w=np.sqrt(w))
        fits[str(gamma_true)] = dict(
            resolved_cells=len(cells), exponent=float(-p[0]), amplitude=float(np.exp(p[1]))
        )

    print(f"\n{'gamma_true':>10s} {'cells':>6s} {'bias = A N^-p':>28s}")
    for g, f in fits.items():
        if "exponent" in f:
            print(f"{g:>10s} {f['resolved_cells']:6d}   "
                  f"A = {f['amplitude']:8.3f}, p = {f['exponent']:.3f}")
        else:
            print(f"{g:>10s} {f['resolved_cells']:6d}   {f['note']}")

    suffix = "_flatprior" if args.flat_prior else ""
    if args.n_values:
        suffix += "_lowN"
    if args.field_ratios:
        suffix += "_ratios"
    out = Path(__file__).with_name(f"eff_gamma_bias{suffix}.json")
    out.write_text(json.dumps(dict(
        field_radius=FIELD_RADIUS, a_true=A_TRUE, n_grid=list(args.n_values or N_GRID),
        gamma_grid=list(GAMMA_GRID), flat_prior=args.flat_prior,
        cells=rows, power_law_fits=fits,
    ), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
