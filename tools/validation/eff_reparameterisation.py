#!/usr/bin/env python3
r"""Is there a combination of the EFF parameters that the data DO identify?

WHY THIS EXISTS
---------------
``eff_gamma_bias.py`` establishes that the slope ``gamma`` is not recoverable at census-typical
geometry, and ``eff_gamma_estimator.py`` rules out the two easy explanations: it is not the classical
Cox-Snell maximum-likelihood bias (pure ``b1/N`` is rejected at reduced chi-square 2-8.7), and it is
not an artefact of reporting a posterior median (the mode retains ~80% of the bias). What is left is
**weak identification**, and the posterior says so directly: ``corr(a, gamma) = +0.871``.

When two parameters are not separately identified, the textbook response is not a bias correction --
it is to **reparameterise onto the combination that is identified** (Cox & Reid 1987 on parameter
orthogonalisation). This script tests whether such a combination exists here, and what it costs.

THE ALGEBRA THAT PREDICTS WHICH COMBINATION
-------------------------------------------
Expanding the EFF surface density for :math:`R \ll a`:

.. math:: \Sigma(r) = \Sigma_0\left(1 + (r/a)^2\right)^{-\gamma/2}
                    = \Sigma_0\left[1 - \frac{\gamma}{2}\frac{r^2}{a^2} + O(r^4/a^4)\right]

so to leading order ``gamma`` and ``a`` enter **only through** :math:`\kappa \equiv \gamma/a^2`. A
footprint that does not reach past ``a`` therefore constrains ``kappa`` and nothing else about the
pair -- which is exactly the regime the census sits in, with ``r_tot/r_c`` below 2 for 27.3% of Hunt &
Reffert's clusters.

``kappa`` is the central curvature of the profile, i.e. how sharply the density turns over at the
centre, in units of inverse area.

WHAT IT FINDS, AND THE TENSION THAT MATTERS
--------------------------------------------
First pass, ``gamma_true = 2.00``, ``a = 1.65'``, ``N = 150``, 10 realizations:

    r_tot/a     gamma bias              kappa fractional bias    corr(a, gamma)
       2      +1.734 +/- 0.116  (15 sigma)   +38% +/- 30%  (1.3 sigma)   0.745
       8      +1.445 +/- 0.251 (5.8 sigma)   -22% +/- 22%  (1.0 sigma)   0.891

**``kappa`` is consistent with unbiased exactly where ``gamma`` is not.** The ``kappa`` errors are
wide, so a firm claim needs the realizations this script runs; but the contrast is already an order of
magnitude in significance.

.. important::
   **The identified combination is not the quantity the science asks about.** The A5 question is
   whether ``gamma`` piles up near 2, the untruncated-King limit -- a statement about the *asymptotic
   slope*. ``kappa`` is the *central curvature*. So reparameterising fixes the statistics and changes
   the question.

   That tension is the sharpest form of the result: **the data identify something, but not the thing
   the question needs.** Reporting ``kappa`` honestly is better than reporting ``gamma``
   dishonestly, and saying which one the footprint supports is more useful than either.

USAGE
-----
    python tools/validation/eff_reparameterisation.py --realizations 24

Writes ``eff_reparameterisation.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import EFFPriors, eff_unbinned

A_TRUE = 1.65  # arcmin
RATIOS = (2.0, 4.0, 8.0, 16.0, 42.0)


def run_cell(ratio, gamma_true, n, realizations, seed, cfg):
    sys.path.insert(0, str(Path(__file__).parent))
    from eff_gamma_bias import eff_radii

    field = ratio * A_TRUE
    kappa_true = gamma_true / A_TRUE**2
    gammas, kappas, corrs, kappa_widths, gamma_widths = [], [], [], [], []
    for i in range(realizations):
        rng = np.random.default_rng(seed + i)
        r = eff_radii(rng, n, gamma=gamma_true, a=A_TRUE, field_radius=field)
        fit = eff_unbinned(r, field_radius=field, priors=EFFPriors(),
                           sampling=cfg, progressbar=False)
        post = fit["eff_trace"].posterior
        g = np.asarray(post["gamma"].values).ravel()
        a = np.asarray(post["a"].values).ravel()
        k = g / a**2
        gammas.append(float(np.median(g)))
        kappas.append(float(np.median(k)))
        corrs.append(float(np.corrcoef(a, g)[0, 1]))
        # relative posterior width -- the identifiability signal, independent of bias
        gamma_widths.append(float(np.std(g) / np.median(g)))
        kappa_widths.append(float(np.std(k) / np.median(k)))

    g_arr, k_arr = np.asarray(gammas), np.asarray(kappas)
    return dict(
        ratio=float(ratio), gamma_true=float(gamma_true), n=int(n),
        realizations=int(realizations), kappa_true=float(kappa_true),
        gamma_bias=float(g_arr.mean() - gamma_true),
        gamma_bias_sem=float(g_arr.std(ddof=1) / np.sqrt(g_arr.size)),
        kappa_frac_bias=float(k_arr.mean() / kappa_true - 1.0),
        kappa_frac_bias_sem=float(k_arr.std(ddof=1) / np.sqrt(k_arr.size) / kappa_true),
        corr_a_gamma=float(np.mean(corrs)),
        gamma_relative_width=float(np.mean(gamma_widths)),
        kappa_relative_width=float(np.mean(kappa_widths)),
    )


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=24)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7000)
    args = ap.parse_args()

    cfg = SamplingConfig(draws=1500, tune=1000, chains=2, random_seed=5, progressbar=False)
    print(f"gamma_true = {args.gamma}, a = {A_TRUE}', N = {args.n}, "
          f"kappa_true = {args.gamma / A_TRUE**2:.4f}\n")
    print(f"{'r_tot/a':>8s} {'gamma bias':>20s} {'kappa frac bias':>20s} "
          f"{'corr':>7s} {'width g':>8s} {'width k':>8s}")
    rows = []
    for ratio in RATIOS:
        row = run_cell(ratio, args.gamma, args.n, args.realizations, args.seed, cfg)
        rows.append(row)
        g_sig = abs(row["gamma_bias"]) / max(row["gamma_bias_sem"], 1e-9)
        k_sig = abs(row["kappa_frac_bias"]) / max(row["kappa_frac_bias_sem"], 1e-9)
        print(f"{ratio:8.1f} {row['gamma_bias']:+9.3f}+/-{row['gamma_bias_sem']:.3f}"
              f" ({g_sig:4.1f}s) {100 * row['kappa_frac_bias']:+9.1f}%+/-"
              f"{100 * row['kappa_frac_bias_sem']:.1f} ({k_sig:4.1f}s)"
              f" {row['corr_a_gamma']:7.3f} {row['gamma_relative_width']:8.3f}"
              f" {row['kappa_relative_width']:8.3f}", flush=True)

    print("\nIf kappa is unbiased where gamma is not, the fix is reparameterisation, not correction.")
    print("But kappa is the CENTRAL CURVATURE and the science question is about the ASYMPTOTIC")
    print("SLOPE -- so the data identify something, just not the thing the question needs.")

    out = Path(__file__).with_name("eff_reparameterisation.json")
    out.write_text(json.dumps(dict(a_true=A_TRUE, ratios=list(RATIOS), cells=rows), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
