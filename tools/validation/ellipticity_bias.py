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
radial law, draw an angle uniformly, and scale the two axes independently.

**The oracle is the circular cell**: at ``q = 1`` the answer must be zero. It is not decoration --
it is what caught the design error described below. Running it revealed a **+0.0116 +/- 0.0020**
bias where zero was required, traced to the model's free background absorbing wing density (the
truth is ``b = 0`` exactly, and the fit inferred 0.0072). The background is therefore pinned near
zero by default; ``--background`` reruns the grid with it free, which is a different question.
The sampler is separately validated against the analytic EFF CDF before anything runs.

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

import arviz as az
import numpy as np

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import EFFPriors, eff_unbinned

A_TRUE = 1.65          # arcmin, semi-major axis
FIELD_RADIUS = 70.0    # arcmin
# N is set by MEASUREMENT, not by "large is safe". Measured on the q = 0.71, gamma = 2.5 cell
# (scratchpad/bench_grid_cost.py), one fit each, truth gamma = 2.5:
#
#   N_KEEP   draws/tune   secs   gamma fitted   bias      SEM over 4 realizations
#   55000     1000/800    89.8   --             --        --
#   15000     1000/800    ~40    --             --        0.0082
#   15000      500/500    20.9   2.5012         +0.0012   0.0082   <- adopted
#    8000      500/500    15.4   2.5138         +0.0138   0.0098   <- REJECTED
#
# 8000 is rejected on accuracy, not speed: its finite-sample bias of +0.0138 is the same order as
# the ellipticity effect being measured (|d gamma| ~ 0.001), so it would manufacture the signal it
# is meant to bound. 15000 costs 5 s more per fit and is unbiased to within its own SEM.
# N_STARS is set by the WORST cell, not a typical one. The field cut keeps a fraction that depends
# on both q and gamma -- measured across the whole grid, it ranges from 1.000 (q=1) down to 0.843
# at gamma=2.0, q=0.30, because a shallower profile puts more stars outside the field. Sizing from
# the gamma=2.5 cell alone gave 17000 and the run died at gamma=2.0, q=0.30 with 14348 < 15000.
N_STARS = 19_000       # 15000 / 0.843 = 17793, rounded up with margin
N_KEEP = 15_000        # fixed post-cut length; see elliptical_eff_radii on why it must be fixed

# Convergence gate, applied to EVERY fit. A cell that fails is recorded and excluded from the
# summary rather than silently averaged in. This is not boilerplate: the 500/500 sampler config
# was chosen from a single fit at gamma=2.5, q=0.71 and produced rhat > 1.01 and ESS < 100 in
# other cells, whose results were still written to the checkpoint. Thresholds are Vehtari et al.
# (2021), 2021BayAn..16..667V.
RHAT_MAX = 1.01
ESS_MIN = 400
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
    r = r[r <= field_radius]

    # Truncate to a FIXED length, because a JAX-backed sampler recompiles for every new input
    # shape. The pruning above returns a different count for every (q, realization) -- measured
    # 60000 at q=1 down to 57154 at q=0.3, and different between realizations at the same q --
    # so every one of the 96 fits was paying a fresh JIT compilation. That is why the first
    # numpyro run was SLOWER than PyTensor NUTS despite being 2.71x faster on a single fit:
    # the benchmark used one shape and compiled once.
    #
    # Truncation is a uniform thinning, not a radial cut: the radii are drawn independently, so
    # their order carries no radial information and keeping the first N_KEEP is the same as
    # sampling N_KEEP at random. It cannot bias gamma; it only lowers N. At 15000 the residual
    # finite-sample bias measured on the circular null is -0.0038 +/- 0.0034, consistent with zero.
    if r.size < N_KEEP:
        raise RuntimeError(f"only {r.size} radii survived the field cut, need {N_KEEP}")
    return r[:N_KEEP]


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=6)
    ap.add_argument("--seed", type=int, default=515)
    ap.add_argument("--sampler", default="numpyro", help="NUTS backend; see the note below")
    # The fit runs WITHOUT a free background by default, and that is a correctness fix, not a
    # simplification. MEASURED on the circular null (q = 1, where the answer must be gamma):
    #
    #   gamma  background   bias in gamma        inferred b   (truth: b = 0 exactly)
    #    2.0   free         +0.0116 +/- 0.0020      0.00716    <- 5.8 sigma, spurious
    #    2.0   pinned ~0    -0.0038 +/- 0.0034      0.00000    <- 1.1 sigma, consistent with zero
    #    2.5   free         +0.0098 +/- 0.0065      0.00247    <- 1.5 sigma
    #    2.5   pinned ~0    -0.0010 +/- 0.0064      0.00000    <- 0.2 sigma
    #
    # Both gammas: a free background biases gamma UP; a pinned one is consistent with zero. The
    # spurious b is larger at gamma = 2.0 (0.0072) than at 2.5 (0.0025), which is the tell -- a
    # shallower profile leaves more wing density for the background to absorb.
    #
    # _eff_model fits a flat background b, which structure.py itself documents as "deliberately
    # near-degenerate with a flat background". elliptical_eff_radii generates NO background, so the
    # true b is 0 -- on the boundary of a half-Cauchy prior. The model puts density into b anyway
    # and raises gamma to compensate, and the size of that effect depends on how much wing density
    # there is, i.e. on gamma AND on q. So it does NOT cancel cleanly against the circular control:
    # it contaminates the very quantity the experiment exists to measure.
    #
    # --background reruns the same grid with the background free, which is the applied question
    # ("how much does the fit as actually performed get wrong?") rather than the clean one
    # ("does ellipticity bias gamma?"). Both are worth having; they are not the same experiment.
    ap.add_argument("--background", action="store_true",
                    help="let the flat background float (applied case); default pins it near zero")
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
    # 1000/800, not 500/500. Measured at three grid cells (scratchpad/bench_converge.py), gate
    # rhat < 1.01, ESS > 400, zero divergences:
    #
    #   gamma  q     draws/tune   secs   max rhat   min ESS   verdict
    #    2.0   0.30    500/500    20.5    1.0100      385     FAIL
    #    2.0   1.00    500/500    19.3    1.0100      210     FAIL
    #    4.0   0.30    500/500    15.9    1.0100      433     FAIL
    #    2.0   0.30   1000/800    31.1    1.0000      532     OK
    #    2.0   1.00   1000/800    30.6    1.0000      465     OK
    #    4.0   0.30   1000/800    37.7    1.0000      726     OK
    #
    # 500/500 was chosen from an earlier benchmark that measured wall time, gamma and posterior SD
    # -- and none of those reveal a failed fit. A non-converged chain still returns a plausible
    # median and a plausible SD. Convergence has to be measured directly, which is why the gate
    # above now runs on every fit.
    cfg = SamplingConfig(draws=1000, tune=800, chains=2, random_seed=3, progressbar=False,
                         nuts_sampler=args.sampler)
    priors = EFFPriors() if args.background else EFFPriors(b_scale=1e-6)
    out = Path(__file__).with_name(
        "ellipticity_bias.json" if not args.background else "ellipticity_bias_withbg.json")

    # The full grid is 4 gammas x 4 axis ratios x --realizations fits: over an hour even on
    # numpyro. Writing only at the end means a run that dies loses everything, which this one
    # already did once. So each cell is checkpointed and a restart resumes.
    def checkpoint(rows):
        out.write_text(json.dumps(dict(a_true=A_TRUE, n_stars=N_STARS, field_radius=FIELD_RADIUS,
                                       axis_ratios=list(AXIS_RATIOS), gammas=list(GAMMAS),
                                       realizations=args.realizations, sampler=args.sampler,
                                       n_keep=N_KEEP, draws=cfg.draws, tune=cfg.tune, background=bool(args.background),
                                       complete=False, cells=rows), indent=1))

    rows = []
    if out.is_file():
        cached = json.loads(out.read_text())
        # The sampler is part of the resume key: silently mixing backends across cells of one
        # grid would put a between-sampler difference into a between-geometry measurement.
        if (cached.get("realizations") == args.realizations and cached.get("n_stars") == N_STARS
                and cached.get("sampler") == args.sampler and cached.get("n_keep") == N_KEEP
                and cached.get("draws") == cfg.draws and cached.get("tune") == cfg.tune
                and cached.get("background") == bool(args.background)):
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
            worst_rhat, worst_ess, divergences = 0.0, np.inf, 0
            for i in range(args.realizations):
                rng = np.random.default_rng(args.seed + 100 * int(gamma * 10) + 10 * int(q * 100) + i)
                r = elliptical_eff_radii(rng, N_STARS, a=A_TRUE, q=q, gamma=gamma,
                                         field_radius=FIELD_RADIUS)
                fit = eff_unbinned(r, field_radius=FIELD_RADIUS, priors=priors,
                                   sampling=cfg, progressbar=False)
                gammas.append(float(fit["gamma_median"]))
                scales.append(float(fit["a_median"].value))

                summary = az.summary(fit["eff_trace"], var_names=["gamma", "a"])
                worst_rhat = max(worst_rhat, float(summary["r_hat"].max()))
                worst_ess = min(worst_ess, float(summary["ess_bulk"].min()))
                divergences += int(np.asarray(fit["eff_trace"].sample_stats["diverging"]).sum())

            g, sc = np.asarray(gammas), np.asarray(scales)
            converged = worst_rhat < RHAT_MAX and worst_ess > ESS_MIN and divergences == 0
            row = dict(
                gamma_true=float(gamma), axis_ratio=float(q),
                delta_gamma=float(g.mean() - gamma),
                delta_gamma_sem=float(g.std(ddof=1) / np.sqrt(g.size)),
                a_fit_over_a=float(sc.mean() / A_TRUE),
                a_fit_over_geometric_mean=float(sc.mean() / (A_TRUE * np.sqrt(q))),
                max_rhat=float(worst_rhat), min_ess=float(worst_ess),
                divergences=int(divergences), converged=bool(converged),
            )
            if not converged:
                print(f"  !! gamma={gamma} q={q} DID NOT CONVERGE: rhat={worst_rhat:.4f} "
                      f"ess={worst_ess:.0f} div={divergences} -- excluded from the summary",
                      flush=True)
            rows.append(row)
            checkpoint(rows)
            print(f"{gamma:6.1f} {q:7.2f} {row['delta_gamma']:+9.4f}+/-{row['delta_gamma_sem']:.4f}"
                  f" {row['a_fit_over_a']:9.3f} {row['a_fit_over_geometric_mean']:18.3f}", flush=True)

    # THE ELLIPTICITY EFFECT IS THE DIFFERENCE AGAINST THE CIRCULAR CELL, NOT delta_gamma ITSELF.
    #
    # A circular cell (q = 1) must recover gamma by construction, but it does NOT return exactly
    # zero: the EFF maximum-likelihood estimator carries a finite-sample bias that scales roughly
    # as 1/N and depends on gamma. Measured here at N = 15000: +0.0137 +/- 0.0028 at gamma = 2.0,
    # i.e. 4.9 sigma from zero, against +0.0012 at gamma = 2.5.
    #
    # That bias is a property of the estimator, not of the geometry, so it contaminates the
    # elliptical cells identically and cancels to first order in the difference. Quoting the raw
    # delta_gamma would attribute an estimator bias to ellipticity -- reporting an effect where
    # there is none. The per-gamma circular cell is the baseline, which is why it is in the grid.
    #
    # The absolute bias is itself worth having: it is the same quantity task #9 sets out to
    # calibrate, measured here as a by-product.
    null_by_gamma = {r["gamma_true"]: r["delta_gamma"] for r in rows if r["axis_ratio"] == 1.0}
    for r in rows:
        base = null_by_gamma.get(r["gamma_true"])
        r["delta_gamma_vs_circular"] = None if base is None else float(r["delta_gamma"] - base)

    circular = [r for r in rows if r["axis_ratio"] == 1.0]
    worst_null = max(abs(r["delta_gamma"]) for r in circular)
    worst_sem = max(r["delta_gamma_sem"] for r in circular)
    print(f"\nESTIMATOR BIAS at q = 1 (this is NOT the ellipticity effect): "
          f"largest |delta gamma| = {worst_null:.4f}, largest SEM = {worst_sem:.4f}")
    print(f"  significant vs zero: {'YES -- subtract it, do not ignore it' if worst_null > 2 * worst_sem else 'no'}")

    at_median = [r for r in rows if r["axis_ratio"] == 0.71
                 and r.get("delta_gamma_vs_circular") is not None]
    if at_median:
        print(f"ELLIPTICITY EFFECT at Tarricq's median q = 0.71, relative to the circular cell: "
              f"{min(r['delta_gamma_vs_circular'] for r in at_median):+.4f} to "
              f"{max(r['delta_gamma_vs_circular'] for r in at_median):+.4f}")
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
