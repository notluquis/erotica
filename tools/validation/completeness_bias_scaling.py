#!/usr/bin/env python3
r"""Is the completeness-induced bias on ``R_c`` linear in the size of the suppression?

WHY THIS EXISTS
---------------
The completeness dossier and ``docs/design-notes/king_model_validity.md`` both argue that NGC 6383's
measured **1.156%** core suppression is negligible, and both do it by *extrapolation*: the repo's
synthetic benchmark uses a toy completeness with a **50.44%** core suppression and shows it inflates
``R_c`` by ~50%, so a 1.2% gradient "moves nothing".

That step is a ratio, and it is only valid if the bias scales linearly with the suppression. Nobody
measured the scaling -- it was assumed. **This script measures it.**

WHAT IT SHOULD FIND, AND WHY
----------------------------
The analytic result derived for this programme,

    delta_theta = epsilon * I^{-1} v,

says the first-order bias is *linear* in the perturbation size ``epsilon``, with the Fisher
information ``I`` and the score-covariance ``v`` fixed by the unperturbed model. So linearity is a
prediction, not a hope, and this is a test of the analytic backbone as much as of the extrapolation.
A measured departure from linearity at small ``epsilon`` would mean the extrapolation from 50% down
to 1.2% is unsafe and the dossier's central claim needs re-deriving.

THE DESIGN
----------
One-parameter family of radial completeness curves, same shape as the repo's test toy, with the core
suppression dialled from 0 to ~65%:

    S(r) = floor + (1 - floor) * (1 - exp(-r / scale))

Core suppression is ``1 - S_bar(core) / S_bar(field)``. For each level, many realizations of a King
point process are thinned by ``S``, fitted **without** the correction, and the fractional ``R_c``
bias recorded against the injected truth. The realization spread gives an error bar, so a
"consistent with zero" result is distinguishable from "too noisy to say".

USAGE
-----
    python tools/validation/completeness_bias_scaling.py --realizations 12

Writes ``completeness_bias_scaling.json``.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import king_unbinned

# NGC 6383 geometry and the repo's benchmark truth, so the answer transfers directly.
TRUE_KING = dict(k=1.0, R_c=1.38, R_t=54.0, b=0.0)
FIELD = 70.0
N_DRAW = 4000  # before thinning; the fit sees fewer
CORE_RADIUS = 1.38  # = R_c, the aperture the suppression is quoted over
# CORRECTED 2026-08-04. This was `0.01156  # order-12 patch mode`, and that number could not be
# reproduced: no patch-mode npz existed on disk, and `king_model_validity.md` asserted "1.156%"
# and "patch mode: Still to do" in the SAME section. A hardcoded constant whose provenance is a
# claim that the measurement had not been made.
#
# Measured for real by `a1_patch_selection_function.py` (sidecar
# `a1_patch_selection_function.json`): **2.371%** mean suppression inside R_c at `mode='patch'`,
# `min_points=20`. Note also that patch mode does NOT reach order 12 in this field at the default
# min_points -- 343 pixels at order 11, 1188 at order 10 -- so "order-12 patch mode" described a
# configuration that does not occur here. The min_points=5 variant that does reach order 12 gives
# 4.562%, but that excess is an artefact (core M10 sd doubles while its mean barely moves, and the
# subdivision is density-driven so core and field end up at different resolutions).
#
# Consequence for this script: every prediction derived from this constant was 2x low.
NGC6383_MEASURED_SUPPRESSION = (
    0.02371  # a1_patch_selection_function.json, mode='patch', min_points=20
)


def king_sigma(r, *, k, R_c, R_t, b):
    r = np.asarray(r, float)
    core = 1.0 / np.sqrt(1.0 + (r / R_c) ** 2)
    edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
    return np.where(r <= R_t, k * (core - edge) ** 2 + b, b)


def sample_king(rng, n, **params):
    grid = np.linspace(0.0, FIELD, 100_001)
    pdf = 2.0 * np.pi * grid * king_sigma(grid, **params)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


def completeness(floor, scale=6.0):
    return lambda r: floor + (1.0 - floor) * (1.0 - np.exp(-np.asarray(r, float) / scale))


def core_suppression(fn):
    """1 - mean S inside one core radius / mean S over the field, area-weighted."""
    inner = np.linspace(1e-6, CORE_RADIUS, 2000)
    outer = np.linspace(1e-6, FIELD, 20000)
    s_in = np.trapezoid(2 * np.pi * inner * fn(inner), inner) / (np.pi * CORE_RADIUS**2)
    s_out = np.trapezoid(2 * np.pi * outer * fn(outer), outer) / (np.pi * FIELD**2)
    return 1.0 - s_in / s_out


def run_level(floor, realizations, seed):
    fn = completeness(floor)
    supp = core_suppression(fn)
    cfg = SamplingConfig(draws=1200, tune=1000, chains=2, random_seed=7, progressbar=False)
    biases = []
    for i in range(realizations):
        rng = np.random.default_rng(seed + i)
        radii = sample_king(rng, N_DRAW, **TRUE_KING)
        observed = radii[rng.uniform(size=radii.size) < fn(radii)]
        fit = king_unbinned(observed, field_radius=FIELD, sampling=cfg)
        rc = float(fit["R_c_median"].value)
        biases.append(rc / TRUE_KING["R_c"] - 1.0)
    b = np.array(biases)
    return dict(
        floor=floor,
        core_suppression=float(supp),
        n_observed=int(observed.size),
        bias_mean=float(b.mean()),
        bias_sem=float(b.std(ddof=1) / np.sqrt(b.size)),
        realizations=realizations,
    )


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260728)
    args = ap.parse_args()

    # floor = 1.0 is no suppression at all (the null); 0.35 is the repo's test toy.
    floors = [1.0, 0.95, 0.90, 0.80, 0.65, 0.50, 0.35]
    rows = []
    for f in floors:
        r = run_level(f, args.realizations, args.seed)
        rows.append(r)
        print(
            f"  floor={f:.2f}  suppression={r['core_suppression']:7.2%}  "
            f"R_c bias = {r['bias_mean']:+7.2%} +/- {r['bias_sem']:.2%}",
            flush=True,
        )

    supp = np.array([r["core_suppression"] for r in rows])
    bias = np.array([r["bias_mean"] for r in rows])
    sem = np.array([r["bias_sem"] for r in rows])

    # Weighted fit through the origin: zero perturbation must give zero bias, so no intercept.
    # Fitted TWICE. The analytic result is a FIRST-ORDER expansion, so linearity is predicted only
    # for small epsilon; a global fit that includes 55% suppression is dominated by the regime where
    # first order has already broken down, and extrapolating that slope down to ~1% is wrong in the
    # unsafe direction. The number to quote comes from the low-suppression fit.
    def fit(mask):
        w = 1.0 / np.maximum(sem[mask], 1e-9) ** 2
        s = float((w * supp[mask] * bias[mask]).sum() / (w * supp[mask] ** 2).sum())
        chi2 = float((w * (bias[mask] - s * supp[mask]) ** 2).sum())
        return s, chi2, int(mask.sum()) - 1

    LINEAR_MAX = 0.20  # suppression below which first order is expected to hold
    low = supp <= LINEAR_MAX
    slope_all, chi2_all, dof_all = fit(np.ones_like(supp, dtype=bool))
    slope_low, chi2_low, dof_low = fit(low)

    print(
        f"\nglobal fit (all {len(rows)} levels): bias = {slope_all:.3f} x suppression, "
        f"chi2/dof = {chi2_all / dof_all:.1f}  -> NONLINEAR over the full range"
    )
    print(
        f"low-suppression fit (<= {LINEAR_MAX:.0%}, {low.sum()} levels): "
        f"bias = {slope_low:.3f} x suppression, chi2/dof = {chi2_low / max(dof_low, 1):.2f}"
    )
    print("  local slope by level:")
    # strict=True: all three come from the same rung list, so a length mismatch means a rung was
    # dropped upstream -- an error worth raising, not one to truncate away silently.
    for _r, s_i, b_i in zip(rows, supp, bias, strict=True):
        if s_i > 1e-6:
            print(f"    suppression {s_i:6.2%} -> bias/suppression = {b_i / s_i:.3f}")
    print(f"\n  => first order holds below ~{LINEAR_MAX:.0%} and breaks above it, which is exactly")
    print("     what delta_theta = eps I^-1 v predicts. Extrapolation is safe BELOW the knee only.")

    extrapolated = slope_low * NGC6383_MEASURED_SUPPRESSION
    print(
        f"\nNGC 6383 measured suppression = {NGC6383_MEASURED_SUPPRESSION:.3%}  "
        f"(inside the linear regime)"
    )
    print(
        f"  -> predicted R_c bias = {extrapolated:+.3%}   "
        f"[global-fit slope would have said {slope_all * NGC6383_MEASURED_SUPPRESSION:+.3%}]"
    )

    out = Path(__file__).with_name("completeness_bias_scaling.json")
    out.write_text(
        json.dumps(
            dict(
                truth=TRUE_KING,
                field_radius=FIELD,
                n_draw=N_DRAW,
                levels=rows,
                linear_regime_max_suppression=LINEAR_MAX,
                slope_low=slope_low,
                chi2_per_dof_low=chi2_low / max(dof_low, 1),
                slope_all=slope_all,
                chi2_per_dof_all=chi2_all / dof_all,
                linear_below_knee=bool(chi2_low / max(dof_low, 1) < 3),
                ngc6383_suppression=NGC6383_MEASURED_SUPPRESSION,
                ngc6383_predicted_Rc_bias=extrapolated,
            ),
            indent=1,
        )
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
