#!/usr/bin/env python3
"""Does the point-process likelihood report honest error bars on a *substructured* cluster?

WHY THIS EXISTS
---------------
``eff_unbinned`` models the sky as an inhomogeneous Poisson point process. That is exact for a
smooth profile, and the smooth control below confirms the implementation is well calibrated under
its own assumptions (93% coverage at 2 sigma). Real young clusters are not smooth: they inherit
substructure from their natal cloud, and clumping is **over-dispersion the Poisson likelihood does
not model**. The reported credible interval on ``gamma`` should therefore be too narrow, and this
script measures by how much.

The result is quoted in ``docs/design-notes/king_model_validity.md`` as the reason the published
``gamma = 2.32 +/- 0.21`` interval cannot be taken at face value. **It previously had no script
behind it** -- the numbers were produced in an ad-hoc session and only the table survived. This file
closes that gap, and makes the generator pluggable so the experiment can be re-run against whatever
the current standard for synthetic cluster structure turns out to be.

THE CONTROLLED COMPARISON
-------------------------
Every configuration keeps the *azimuthally averaged radial profile unchanged in expectation*, so any
shift in the recovered ``gamma`` is attributable to substructure alone and not to a different
underlying profile:

* ``smooth``   -- the null. Radii drawn straight from EFF; this is the model the fit assumes.
* ``clumps``   -- a fraction of stars displaced into Gaussian clumps whose **centres are drawn from
  the same EFF profile**. Ad-hoc, and reproduces the number currently in the design note.
* ``fractal``  -- substructure from the box-fractal construction (:func:`erotica.analysis.synthetic.
  fractal_cluster`), radially remapped onto the same EFF profile so the marginal radial distribution
  is EFF by construction while the angular structure is fractal.

The last is the honest one: it puts real, scale-free substructure in without changing the profile
being fitted.

USAGE
-----
    python tools/validation/substructure_coverage.py --realizations 40
    python tools/validation/substructure_coverage.py --config fractal --fractal-dimension 2.0

Writes ``substructure_coverage.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from erotica.analysis.structure import EFFPriors, eff_unbinned
from erotica.analysis.synthetic import fractal_cluster

# NGC 6383 geometry, so the answer applies to the published fit rather than to a toy.
N_STARS = 628
FIELD_RADIUS = 70.0  # arcmin
GAMMA_TRUE = 2.32
A_TRUE = 1.65  # arcmin


def eff_radii(rng, n, *, a=A_TRUE, gamma=GAMMA_TRUE, field_radius=FIELD_RADIUS):
    """Draw ``n`` radii from the EFF surface density by inverse-CDF.

    ``Sigma(r) ~ (1 + (r/a)^2)^(-gamma/2)``, so ``p(r) ~ 2 pi r Sigma(r)`` and the CDF is analytic;
    it is tabulated rather than inverted in closed form to keep the truncation at ``field_radius``
    exact.
    """
    grid = np.linspace(0.0, field_radius, 200_001)
    pdf = 2.0 * np.pi * grid * (1.0 + (grid / a) ** 2) ** (-gamma / 2.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


def sample_smooth(rng, n=N_STARS, **kw):
    """The null: exactly the generative model the likelihood assumes."""
    r = eff_radii(rng, n, **kw)
    theta = rng.uniform(0.0, 2 * np.pi, n)
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def sample_clumps(rng, n=N_STARS, *, clumped_fraction=0.5, n_clumps=15, clump_sigma=1.0, **kw):
    """Ad-hoc Gaussian clumps, centres drawn from the same EFF profile.

    Reproduces the configuration behind the table currently in the design note. Kept so the
    historical number stays checkable, not because it is a good model of substructure.
    """
    n_clumped = int(round(clumped_fraction * n))
    field = sample_smooth(rng, n - n_clumped, **kw)
    centres = sample_smooth(rng, n_clumps, **kw)
    which = rng.integers(0, n_clumps, n_clumped)
    clumped = centres[which] + rng.normal(0.0, clump_sigma, (n_clumped, 2))
    return np.vstack([field, clumped])


def sample_fractal(rng, n=N_STARS, *, fractal_dimension=1.6, field_radius=FIELD_RADIUS, **kw):
    """Fractal angular substructure, radially remapped so the marginal profile stays EFF.

    The box-fractal gives scale-free substructure but its own radial profile. Sorting its projected
    radii and substituting the order statistics of an EFF draw keeps the **angular** structure
    (which star is near which) while forcing the **radial** marginal to be exactly EFF -- so the
    comparison against ``smooth`` isolates substructure and does not confound it with a change of
    profile.
    """
    xy = fractal_cluster(n, fractal_dimension=fractal_dimension, rng=rng)[:, :2]
    r = np.linalg.norm(xy, axis=1)
    unit = np.divide(xy, r[:, None], out=np.zeros_like(xy), where=r[:, None] > 0)
    target = np.sort(eff_radii(rng, n, field_radius=field_radius, **kw))
    remapped = np.empty(n)
    remapped[np.argsort(r)] = target
    return unit * remapped[:, None]


# The remap is not free: replacing each star's radius by the order statistic at its rank pulls
# clump members apart radially, so some substructure is lost. Measured (mean of 4 realizations,
# N = 628), it survives with room to spare:
#
#     D     raw fractal Q     after EFF rank-remap Q
#     1.6       0.419                 0.591
#     2.0       0.516                 0.729
#
# against a smooth EFF control at Q = 1.276 and the ad-hoc clumps at Q = 1.019 -- i.e. the remapped
# fractal is far MORE substructured than the configuration the published table used. For reference
# NGC 6383 itself sits at Q = 0.833, which the remapped fractal brackets between D = 2.0 and 2.5.
# Checking this mattered: had the remap flattened the structure, the fractal row would have come
# back null and read as "substructure does not matter".


CONFIGS = {
    "smooth": (sample_smooth, {}),
    "clumps15": (sample_clumps, dict(n_clumps=15, clump_sigma=1.0)),
    "clumps8": (sample_clumps, dict(n_clumps=8, clump_sigma=2.0)),
    "fractal": (sample_fractal, dict(fractal_dimension=1.6)),
}


def q_parameter(xy):
    """Cartwright & Whitworth (2004) Q, for context only -- it is not the oracle here."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = len(xy)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    edges = minimum_spanning_tree(d).toarray()
    edges = edges[edges > 0]
    r_cluster = np.linalg.norm(xy - xy.mean(axis=0), axis=1).max()
    return (edges.mean() / (np.sqrt(n * np.pi * r_cluster**2) / n)) / (
        d[np.triu_indices(n, 1)].mean() / r_cluster
    )


def run(config, realizations, seed, **overrides):
    sampler, kw = CONFIGS[config]
    kw = {**kw, **overrides}
    rows = []
    for i in range(realizations):
        rng = np.random.default_rng(seed + i)
        xy = sampler(rng, **kw)
        radii = np.linalg.norm(xy, axis=1)
        radii = radii[radii <= FIELD_RADIUS]
        fit = eff_unbinned(
            radii, field_radius=FIELD_RADIUS, priors=EFFPriors(), progressbar=False
        )
        post = np.asarray(fit["eff_trace"].posterior["gamma"].values).ravel()
        rows.append(
            dict(
                gamma_mean=float(post.mean()),
                gamma_sd=float(post.std()),
                q=float(q_parameter(xy)) if len(xy) <= 1200 else None,
                covered_1sigma=bool(abs(post.mean() - GAMMA_TRUE) <= post.std()),
                covered_2sigma=bool(abs(post.mean() - GAMMA_TRUE) <= 2 * post.std()),
            )
        )
        print(f"  [{config}] {i + 1}/{realizations}  gamma={rows[-1]['gamma_mean']:.3f}"
              f" +/- {rows[-1]['gamma_sd']:.3f}", flush=True)

    means = np.array([r["gamma_mean"] for r in rows])
    sds = np.array([r["gamma_sd"] for r in rows])
    qs = [r["q"] for r in rows if r["q"] is not None]
    return dict(
        config=config,
        settings=kw,
        realizations=realizations,
        q_median=float(np.median(qs)) if qs else None,
        gamma_recovered=float(means.mean()),
        reported_sigma=float(sds.mean()),
        realization_scatter=float(means.std(ddof=1)),
        # THE number: how much wider the truth is than the likelihood admits.
        understatement_factor=float(means.std(ddof=1) / sds.mean()),
        coverage_1sigma=float(np.mean([r["covered_1sigma"] for r in rows])),
        coverage_2sigma=float(np.mean([r["covered_2sigma"] for r in rows])),
        per_realization=rows,
    )


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="all", choices=[*CONFIGS, "all"])
    ap.add_argument("--realizations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260728)
    ap.add_argument("--fractal-dimension", type=float, default=None)
    args = ap.parse_args()

    configs = list(CONFIGS) if args.config == "all" else [args.config]
    overrides = {}
    if args.fractal_dimension is not None:
        overrides["fractal_dimension"] = args.fractal_dimension

    results = []
    for c in configs:
        kw = overrides if c == "fractal" else {}
        results.append(run(c, args.realizations, args.seed, **kw))

    print(f"\n{'config':10s} {'Q':>6s} {'gamma':>7s} {'reported':>9s} {'scatter':>8s} "
          f"{'factor':>7s} {'1sig':>6s} {'2sig':>6s}")
    for r in results:
        q = f"{r['q_median']:.3f}" if r["q_median"] is not None else "  --  "
        print(f"{r['config']:10s} {q:>6s} {r['gamma_recovered']:7.3f} {r['reported_sigma']:9.3f} "
              f"{r['realization_scatter']:8.3f} {r['understatement_factor']:7.2f} "
              f"{r['coverage_1sigma']:5.0%} {r['coverage_2sigma']:5.0%}")
    print("\nnominal coverage is 68% / 95%; 'factor' is how much the likelihood understates the")
    print("true realization-to-realization scatter in gamma")

    out = Path(__file__).with_name("substructure_coverage.json")
    out.write_text(json.dumps(dict(truth=dict(gamma=GAMMA_TRUE, a=A_TRUE, n=N_STARS,
                                              field_radius=FIELD_RADIUS),
                                   results=results), indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
