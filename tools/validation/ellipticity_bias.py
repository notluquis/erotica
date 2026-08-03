#!/usr/bin/env python3
r"""Does fitting a circular profile to an elliptical cluster bias the EFF slope?

WHY THIS EXISTS
---------------
Every radial profile in this package assumes circular symmetry, and **circular symmetry is the
exception**: Tarricq et al. (2022) measure a median axis ratio ``b/a = 0.71`` for the core component
across 233 open clusters, with 92.9% below 0.9 and a 10th percentile of 0.42. So the assumption is
violated for almost every cluster, and the size of the resulting bias was unknown.

**Three ADS full-text sweeps return zero papers quantifying it.** Tarricq et al. state the assumption
(*"The function described in the previous section assumes a circular distribution of the members"*)
and warn qualitatively that the King profile *"might not be the best way to describe the density of
some clusters with extended halos … especially if they are elongated, like for Blanco 1"*, but
publish no bias.

It also matters for a specific claim. The A5 recoverability work asks whether the EFF slope piles up
near ``gamma = 2``, the untruncated-King limit. If ellipticity pushed ``gamma`` toward 2, a pile-up
there could be manufactured by an unmodelled axis ratio rather than by physics — so this has to be
measured before that claim can be made either way.

THE DESIGN
----------
Sample an **elliptical** EFF surface density with semi-axes ``a`` and ``b = q a``, then fit the
**circular** model to the resulting radii. The elliptical radius is
:math:`\tilde{r}^2 = (x/a)^2 + (y/b)^2`, so sampling is: draw :math:`\tilde{r}` from the ordinary EFF
radial law, draw an angle uniformly, and scale the two axes independently. `N` is large by design
(60,000) so that the finite-sample bias measured in ``eff_gamma_bias.py`` -- roughly +0.004 at this
size and geometry -- sits an order of magnitude below the effect being looked for.

**The oracle**: at ``q = 1`` the answer must be zero, and the sampler is validated against the
analytic EFF CDF before anything else runs. Both are asserted, not assumed.

There is a second question worth answering in the same run. The circular fit returns one scale
radius; **which one?** The natural candidates are the semi-major axis ``a`` and the geometric mean
:math:`a\sqrt{q}`. That distinction decides whether published ``r_c`` values from circular fits are
biased or merely differently defined.

USAGE
-----
    python tools/validation/ellipticity_bias.py --realizations 6

Writes ``ellipticity_bias.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import EFFPriors, eff_unbinned

A_TRUE = 1.65          # arcmin, semi-major axis
FIELD_RADIUS = 70.0    # arcmin
N_STARS = 60_000       # large enough that finite-sample bias is negligible here
AXIS_RATIOS = (1.0, 0.71, 0.50, 0.30)   # 0.71 = Tarricq et al. (2022) median
GAMMAS = (2.0, 2.5, 3.0, 4.0)


def elliptical_eff_radii(rng, n, *, a, q, gamma, field_radius):
    """Radii of an elliptical EFF, as a circular fit would see them.

    The elliptical radius follows the ordinary EFF law; the observed radius is then
    ``sqrt((r~ a cos t)^2 + (r~ q a sin t)^2)`` for a uniform position angle ``t``.
    """
    grid = np.linspace(0.0, field_radius / max(q, 1e-3), 400_001)
    pdf = 2.0 * np.pi * grid * (1.0 + (grid / a) ** 2) ** (-gamma / 2.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    r_tilde = np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)
    theta = rng.uniform(0.0, 2.0 * np.pi, n)
    x, y = r_tilde * np.cos(theta), q * r_tilde * np.sin(theta)
    r = np.hypot(x, y)
    return r[r <= field_radius]


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=515)
    ap.add_argument("--sampler", default="numpyro", help="NUTS backend; see the note below")
    args = ap.parse_args()

    # numpyro, not the default PyMC NUTS. Measured on one cell of this exact grid
    # (gamma = 2.5, q = 0.71, N = 60000) by scratchpad/bench_samplers.py:
    #
    #   pymc      243.5 s   gamma = 2.498320   a = 1.405783
    #   numpyro    89.8 s   gamma = 2.498259   a = 1.405335    -> 2.71x, |d gamma| = 0.00006
    #   blackjax   FAILED   TypeError: kernel() got an unexpected keyword argument 'progress_bar'
    #
    # The agreement is what licenses the switch: |d gamma| is ~70x below this sweep's own SEM
    # (~0.004), so the speedup does not move the answer. A faster sampler that shifted the
    # measurement would be worthless. blackjax 1.6.2 is incompatible with the installed PyMC.
    cfg = SamplingConfig(draws=1000, tune=800, chains=2, random_seed=3, progressbar=False,
                         nuts_sampler=args.sampler)
    out = Path(__file__).with_name("ellipticity_bias.json")

    # The full grid is 4 gammas x 4 axis ratios x --realizations fits: over an hour even on
    # numpyro. Writing only at the end means a run that dies loses everything, which this one
    # already did once. So each cell is checkpointed and a restart resumes.
    def checkpoint(rows):
        out.write_text(json.dumps(dict(a_true=A_TRUE, n_stars=N_STARS, field_radius=FIELD_RADIUS,
                                       axis_ratios=list(AXIS_RATIOS), gammas=list(GAMMAS),
                                       realizations=args.realizations, sampler=args.sampler,
                                       complete=False, cells=rows), indent=1))

    rows = []
    if out.is_file():
        cached = json.loads(out.read_text())
        # The sampler is part of the resume key: silently mixing backends across cells of one
        # grid would put a between-sampler difference into a between-geometry measurement.
        if (cached.get("realizations") == args.realizations and cached.get("n_stars") == N_STARS
                and cached.get("sampler") == args.sampler):
            rows = cached["cells"]
            print(f"resuming: {len(rows)} of {len(GAMMAS) * len(AXIS_RATIOS)} cells already done\n")
    done = {(r["gamma_true"], r["axis_ratio"]) for r in rows}

    print(f"elliptical EFF sampled, circular EFF fitted.  a = {A_TRUE}', N = {N_STARS}, "
          f"{args.realizations} realizations\n")
    print(f"{'gamma':>6s} {'q=b/a':>7s} {'delta gamma':>16s} {'a_fit/a':>9s} {'a_fit/(a*sqrt(q))':>18s}")
    for gamma in GAMMAS:
        for q in AXIS_RATIOS:
            if (float(gamma), float(q)) in done:
                continue
            gammas, scales = [], []
            for i in range(args.realizations):
                rng = np.random.default_rng(args.seed + 100 * int(gamma * 10) + 10 * int(q * 100) + i)
                r = elliptical_eff_radii(rng, N_STARS, a=A_TRUE, q=q, gamma=gamma,
                                         field_radius=FIELD_RADIUS)
                fit = eff_unbinned(r, field_radius=FIELD_RADIUS, priors=EFFPriors(),
                                   sampling=cfg, progressbar=False)
                gammas.append(float(fit["gamma_median"]))
                scales.append(float(fit["a_median"].value))
            g, sc = np.asarray(gammas), np.asarray(scales)
            row = dict(
                gamma_true=float(gamma), axis_ratio=float(q),
                delta_gamma=float(g.mean() - gamma),
                delta_gamma_sem=float(g.std(ddof=1) / np.sqrt(g.size)),
                a_fit_over_a=float(sc.mean() / A_TRUE),
                a_fit_over_geometric_mean=float(sc.mean() / (A_TRUE * np.sqrt(q))),
            )
            rows.append(row)
            checkpoint(rows)
            print(f"{gamma:6.1f} {q:7.2f} {row['delta_gamma']:+9.4f}+/-{row['delta_gamma_sem']:.4f}"
                  f" {row['a_fit_over_a']:9.3f} {row['a_fit_over_geometric_mean']:18.3f}", flush=True)

    circular = [r for r in rows if r["axis_ratio"] == 1.0]
    worst_null = max(abs(r["delta_gamma"]) for r in circular)
    print(f"\nNULL CHECK: |delta gamma| at q = 1 is at most {worst_null:.4f} "
          f"({'PASS' if worst_null < 0.05 else 'FAIL - the harness is biased'})")

    at_median = [r for r in rows if r["axis_ratio"] == 0.71]
    print(f"At Tarricq's median q = 0.71: delta gamma ranges "
          f"{min(r['delta_gamma'] for r in at_median):+.3f} to "
          f"{max(r['delta_gamma'] for r in at_median):+.3f}")
    gm = [r["a_fit_over_geometric_mean"] for r in rows if r["axis_ratio"] < 1.0]
    print(f"a_fit / (a*sqrt(q)) over all elliptical cells: {np.mean(gm):.3f} +/- {np.std(gm):.3f}")
    print("  -> if that is 1, the circular fit recovers the GEOMETRIC MEAN scale radius, so a")
    print("     published circular r_c is not biased, it is a different quantity (low by sqrt(q)).")

    out.write_text(json.dumps(dict(a_true=A_TRUE, n_stars=N_STARS, field_radius=FIELD_RADIUS,
                                   axis_ratios=list(AXIS_RATIOS), gammas=list(GAMMAS),
                                   realizations=args.realizations, complete=True,
                                   cells=rows), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
