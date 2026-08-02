#!/usr/bin/env python3
"""Is the EFF slope bias a property of the estimator, or of the summary we report from it?

WHY THIS EXISTS
---------------
``eff_gamma_bias.py`` measures that the recovered ``gamma`` is biased high. This script asks the
prior question: **biased how?** The answer decides whether the fix is a correction, a different
summary statistic, or neither.

THE THEORY THIS TESTS, AND WHY THE OBVIOUS FIT IS NOT ENOUGH
------------------------------------------------------------
An ad-hoc ``bias = A * N^-p`` was fitted first and gave ``p ~ 0.72-0.78``. That is a curve chosen
because it looked right, which is not a justification. Classical theory says something specific:

**Cox & Snell (1968)**, with the matrix form in Kosmidis (2014, WIREs Comp Stat 6, 185,
`arXiv:1311.6311`), expands the bias of a maximum-likelihood estimator as

.. math:: B(\theta) = b_1(\theta)/n + b_2(\theta)/n^2 + O(n^{-3})

so the **leading term is** :math:`O(1/n)`, i.e. ``p = 1``. Their regularity conditions are explicit:
identifiability, finite parameter count, a parameter space not depending on the sample space, and
enough log-likelihood derivatives.

The point-process framing is **not** an obstacle to applying this. By the conditioning property of a
Poisson process, given ``N`` points in a window they are iid draws from the window-normalised
intensity, so for *shape* parameters the unbinned inhomogeneous-Poisson likelihood **is** the iid
likelihood for ``N`` draws, and Cox-Snell applies with ``n = N``.

Model comparison on the measured grid at fixed geometry (``r_tot/a = 42.4``, ``N`` = 20 to 5000),
reduced chi-square, for ``gamma_true`` = 2.00 / 2.32 / 3.00:

    H1  bias = A N^-p              0.93 / 0.23 / 0.95     empirical, no theoretical content
    H2  bias = b1/N                8.72 / 2.06 / 4.90     Cox-Snell leading term alone -- REJECTED
    H3  bias = A/sqrt(N) + b1/N    1.12 / 0.43 / 1.48     accepted

**H2 is rejected**, so the bias is not the classical MLE bias. H3 fits as well as the empirical power
law while having two terms that each mean something, and it explains the ``p ~ 0.75``: that exponent
is the effective slope of a ``1/sqrt(N)`` term plus a ``1/N`` term over this range. It has no content
of its own.

WHAT PRODUCES A 1/sqrt(N) TERM, AND WHAT THIS SCRIPT FOUND
-----------------------------------------------------------
The natural candidate is the **summary statistic**. The posterior for ``gamma`` is right-skewed, and
for a skewed distribution the median sits a fixed fraction of the *width* away from the mode -- and
the width scales as :math:`1/\sqrt{N}`. If that were the whole story, reporting the posterior **mode**
would remove the ``1/sqrt(N)`` term and leave the classical ``1/N``.

**It is not the whole story.** Measured at ``gamma_true = 2.32``, 16 realizations per cell:

    N        median     mean      mode(KDE)   mode/median
    60      +0.4287   +0.4953    +0.3304        0.77
    150     +0.1918   +0.2134    +0.1641        0.86
    628     +0.0461   +0.0512    +0.0371        0.80
    2500    +0.0196   +0.0216    +0.0157        0.80

The ordering ``mode < median < mean`` is exactly right-skew, so the mechanism is real. But the mode
retains **~81%** of the median's bias. **The estimator itself is biased, not merely the summary taken
from it, and a different point estimate is not the fix.**

.. important::
   **The most informative number here is the constancy, not the size.** ``mode/median`` sits at
   0.77-0.86 across a factor of 40 in ``N``. Had the skewness contributed a ``1/sqrt(N)`` term while
   the remainder scaled as ``1/N``, that ratio would drift with ``N``. It does not, so the posterior
   shape is **self-similar in N** -- it shrinks without changing form.

   That weakens the tidy reading of H3 as "skewness plus Cox-Snell": the two terms fit, but they do
   not map onto two separable mechanisms. A single mechanism whose amplitude happens to scale between
   ``N^-1/2`` and ``N^-1`` fits the evidence better, and weak identification is the candidate --
   ``corr(a, gamma) = +0.871``, and the two parameters enter only as ``gamma/a^2`` for ``R << a``.
   **Quote H3 as a fitted description, not as a decomposition into named effects.**

That leaves identifiability as the remaining explanation, which is consistent with everything else
measured: ``a`` and ``gamma`` are correlated at **+0.871** in the posterior, and for ``R << a`` they
enter the profile only through ``gamma/a^2``. Dufour (1997, Econometrica 65, 1365) is the relevant
formal result -- any valid confidence set for a locally-almost-unidentified parameter must be
unbounded with positive probability, so an almost-surely-bounded interval has zero coverage. A bounded
credible interval on ``gamma`` in that regime is bounded because the prior bounds it.

USAGE
-----
    python tools/validation/eff_gamma_estimator.py --realizations 16

Writes ``eff_gamma_estimator.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import EFFPriors, eff_unbinned

FIELD_RADIUS = 70.0
A_TRUE = 1.65
N_GRID = (60, 150, 628, 2500)


def posterior_mode(draws, grid_points=512):
    """KDE mode. Gaussian KDE with Scott's rule -- adequate here because the posterior is
    unimodal and smooth; a histogram mode would be bandwidth-free but noisier at these sample sizes."""
    from scipy.stats import gaussian_kde

    grid = np.linspace(draws.min(), draws.max(), grid_points)
    return float(grid[np.argmax(gaussian_kde(draws)(grid))])


def run_cell(n, gamma_true, realizations, seed, cfg):
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from eff_gamma_bias import eff_radii

    summaries = {"median": [], "mean": [], "mode": []}
    for i in range(realizations):
        rng = np.random.default_rng(seed + 1000 * int(gamma_true * 100) + i)
        r = eff_radii(rng, n, gamma=gamma_true, field_radius=FIELD_RADIUS)
        fit = eff_unbinned(r, field_radius=FIELD_RADIUS, priors=EFFPriors(),
                           sampling=cfg, progressbar=False)
        draws = np.asarray(fit["eff_trace"].posterior["gamma"].values).ravel()
        summaries["median"].append(float(np.median(draws)))
        summaries["mean"].append(float(draws.mean()))
        summaries["mode"].append(posterior_mode(draws))

    row = dict(n=int(n), gamma_true=float(gamma_true), realizations=int(realizations))
    for key, vals in summaries.items():
        v = np.asarray(vals)
        row[f"{key}_bias"] = float(v.mean() - gamma_true)
        row[f"{key}_sem"] = float(v.std(ddof=1) / np.sqrt(v.size))
    row["mode_over_median"] = (
        row["mode_bias"] / row["median_bias"] if row["median_bias"] else float("nan")
    )
    return row


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=16)
    ap.add_argument("--gamma", type=float, default=2.32)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    cfg = SamplingConfig(draws=2000, tune=1000, chains=2, random_seed=5, progressbar=False)
    rows = []
    print(f"gamma_true = {args.gamma}, r_tot/a = {FIELD_RADIUS / A_TRUE:.1f}, "
          f"{args.realizations} realizations per cell\n")
    print(f"{'N':>6s} {'median':>18s} {'mean':>18s} {'mode':>18s} {'mode/median':>12s}")
    for n in N_GRID:
        row = run_cell(n, args.gamma, args.realizations, args.seed, cfg)
        rows.append(row)
        fmt = lambda k: f"{row[k + '_bias']:+8.4f}+/-{row[k + '_sem']:.4f}"  # noqa: E731
        print(f"{n:6d} {fmt('median'):>18s} {fmt('mean'):>18s} {fmt('mode'):>18s} "
              f"{row['mode_over_median']:12.2f}", flush=True)

    ratios = [r["mode_over_median"] for r in rows if np.isfinite(r["mode_over_median"])]
    print(f"\nmode retains {np.mean(ratios):.0%} of the median's bias on average.")
    print("If the bias were a summary artefact the mode would remove it; it does not, so the")
    print("estimator is biased and a different point estimate is not the fix.")

    out = Path(__file__).with_name("eff_gamma_estimator.json")
    out.write_text(json.dumps(dict(
        field_radius=FIELD_RADIUS, a_true=A_TRUE, gamma_true=args.gamma,
        footprint_over_scale=FIELD_RADIUS / A_TRUE, cells=rows,
        mean_mode_over_median=float(np.mean(ratios)),
    ), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
