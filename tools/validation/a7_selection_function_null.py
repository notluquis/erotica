#!/usr/bin/env python3
r"""A7: the Gaia DR3 selection function does not bias open-cluster radial profiles -- yet.

WHY THIS EXPERIMENT EXISTS
--------------------------
The thread's original novelty claim is **dead**: Hunt & Reffert (2024) sec. 3.2 already apply
Cantat-Gaudin's star-level DR3 selection function per cluster, and their sec. 3.5 derives
``M_obs(r)`` with it. Do not revive it. What survives is a **null with an expiry date**, and a
null is only worth reading if three things are shown together:

1. the machinery **detects** the effect when it is present (positive controls, calibrated
   against a known injected amplitude -- not merely non-zero);
2. the effect is **absent** in the target population at the depth actually observed;
3. the null **stops being true** at a stated, computed sample size, because the analytic
   backbone says ``bias/sigma`` grows as ``sqrt(N)`` while the bias itself does not shrink.

The analytic backbone, for the inhomogeneous-Poisson (Cash) likelihood with truth
``lambda = 2 pi r Sigma(r; theta_0) S(r)`` fitted under the assumption ``S == 1``:

* ``Sigma = k (core - edge)^2 + b`` is homogeneous of degree 1 in ``(k, b)``, so a **constant**
  completeness is absorbed exactly by ``k -> S_0 k``, ``b -> S_0 b``. **A flat S biases no shape
  parameter, ever, by any amount.** Only the gradient can bias anything.
* To first order in ``S/S_0 - 1``,

      ``delta theta = I^{-1} w``,
      ``I_{jl} = int lambda_0 u_j u_l dr``,   ``w_j = int lambda_0 (S/S_0 - 1) u_j dr``,
      ``u_j = d log Sigma / d theta_j``.

  ``I^{-1}w`` is **independent of the overall amplitude** of ``lambda_0`` and therefore of ``N``,
  while ``sigma ~ N^{-1/2}``. Hence ``bias/sigma ~ sqrt(N)``: the correction that is negligible
  today becomes significant with a deeper or richer membership, and the best-observed clusters
  are the ones most at risk.

WHAT WOULD FALSIFY THE NULL -- PRE-REGISTERED, WRITTEN BEFORE THE RUN
---------------------------------------------------------------------
The claim under test is: *no bona-fide open cluster in the Hunt & Reffert (2024) census has its
King ``R_c`` moved by as much as its own quoted error bar when the DR3 selection function is
applied at the depth its membership actually reaches.* Any one of the following overturns it.

F1. **A qualifying cluster.** Any ``Type='o'`` census cluster whose corrected ``R_c`` shifts by
    ``>= 1 sigma_naive`` at its **observed** membership depth. The measured ceiling is M11 at
    ``-0.15 sigma_naive`` (``-0.11 sigma`` under the quadrature convention); one cluster at
    ``1 sigma`` kills the census-wide null.
F2. **A broken estimator.** A **flat** ``S`` producing a shift distinguishable from zero through
    ``king_unbinned``. That would mean the quadrature normalisation path is wrong and every null
    in this file is an artefact of a broken fit rather than a measurement. (Stage ``flat``.)
F3. **An uncalibrated control.** The injected-bias ladder failing to recover the analytic
    ``I^{-1}w`` within Monte-Carlo error, or the corrected fit failing to return the injected
    truth. A control that only shows "something moved" cannot license a null. (Stage
    ``injection``.)
F4. **A broken extrapolation.** The measured exponent of ``|bias|/sigma`` against ``N`` differing
    from ``0.5`` by more than ``0.05``. That does not restore the effect today, but it destroys
    the expiry date, which is the paper's hook. (Stage ``sqrtn``.)
F5. **A resolution artefact.** The suppression continuing to grow steeply as the selection map
    is refined. The census map cannot resolve a core, so this file rebuilds ``M_10`` at fixed
    healpix orders 9, 10, 11 and 12 **from a single archive query on the same sources** and
    measures ``eps`` at each. If the 11 -> 12 step is small while 10 -> 11 is large, order 12 is
    converged and the null stands. If 11 -> 12 is still large, the null is **resolution-limited**
    and must be reported as a lower bound, not as a measurement -- exactly the trap that made an
    all-sky ``hpx7`` map (27.5' pixels) read as "flat" for a cluster with ``R_c ~ 1.4'`` in the
    sibling A1 thread. (Stage ``targets``.)
F6. **A stale census.** The census suppression distribution shifting materially when recomputed
    from an independent live catalogue pull. (Stage ``census``.)

RESOLUTION, STATED UP FRONT BECAUSE IT IS THE KNOWN TRAP HERE
-------------------------------------------------------------
Three resolutions appear and they are not interchangeable:

===================  =========  ==================================================================
map                  pixel      what it can see
===================  =========  ==================================================================
``mode='hpx7'``      27.5'      **nothing** inside a cluster. A flat ``S(r)`` from it means
                                *invisible at this resolution*, not *absent*. Never used here.
``mode='multi'``     3.4'-27.5' the census map (order 7-10, density-adaptive). 3.4' against a
                                median census ``r50`` of 5.3': the core is 1-2 pixels across.
                                **The census-wide suppression from this map is a lower bound and
                                is NOT the evidence for the null** -- see F5.
``cap9 .. cap12``    6.9'-0.86' rebuilt here from one archive query per target, at fixed maximum
                                order. This is the ladder that decides whether the answer is
                                converged. Order 12 is ~22x finer in area than the finest cell
                                Cantat-Gaudin et al. (2023) calibrated, so it is extrapolation
                                below anything they tested and they publish no error budget there.
===================  =========  ==================================================================

The census's job is therefore **not** the null. It is the *mechanism*, which is a statement in
magnitude space and touches no map at all: open-cluster members sit far above the DR3 edge
(median ``G_p98 = 19.38``; only 7.1% of clusters reach ``G_p98 > 20``), which is why the
magnitude term of ``S = f(G, M_10)`` cannot bite however well the crowding term is resolved.

STAGES
------
``flat``       F2. Flat-S invariance through the real estimator, at ``S_0 = 0.999, 0.5, 0.2``.
``injection``  F3. Known truth + known ``S(r)`` at a ladder of amplitudes; naive bias against the
               analytic ``I^{-1}w`` oracle; corrected fit against the injected truth; sign-flip
               mutation.
``sqrtn``      F4. The same injection at fixed amplitude over ``N = 500 .. 16000``.
``targets``    F1/F5. Real clusters over the order 9-12 ladder from live archive queries: M11
               and NGC 6383 as the OC ceiling, NGC 5904 (M5) as the globular positive control,
               each at the observed membership depth and at shifted depths.
``census``     F6. Census-wide suppression (a **lower bound**, see F5), the magnitude
               decomposition, and a resolution-free stress test: how deep an unresolved core
               ``M_10`` depression each cluster would need before the correction could bite.
``expiry``     Combines ``sqrtn`` and ``targets`` into the ``N`` at which the null dies.

USAGE
-----
    python tools/validation/a7_selection_function_null.py --stage all
    python tools/validation/a7_selection_function_null.py --stage injection --seeds 4

Environment: this needs ``gaiaunlimited`` **and** ``erotica`` **and** ``pymc`` in one interpreter.
No named conda env has all three; only the miniforge base does (read-only use, no installs).

OUTCOME -- appended AFTER the run; nothing above this line was edited afterwards
--------------------------------------------------------------------------------
F1  **not triggered on any cluster measured, but the census-wide version of the claim does not
    survive and must not be written.** Directly measured at order 12 and at the observed
    membership depth: NGC 6705 (M11) ``-0.084 sigma_naive`` (``-0.059`` hypot), NGC 6383
    ``-0.045``. Census-wide, the order-10 suppression distribution is **not** admissible evidence
    (F5), so the bound comes from the resolution-free stress test instead: with a hypothetical
    0.5 mag unresolved core ``M_10`` depression, **zero of 5647** open clusters reach
    ``1 sigma``; with 1.0 mag, **ten (0.18%)** do; with 1.5 mag, 36 (0.64%).

    **Four of the ten are bona-fide, quality-flagged clusters, not catalogue junk**: NGC 2516
    (``z = 1.89``, N = 3784, ``G_p98 = 20.40``), Stock 2 (1.45, 3011, 20.45), NGC 3532 (1.10,
    3414, 20.27) and Berkeley 53 (1.11, 1660, 20.16). They qualify because they are *rich and
    already deep* -- exactly the population the ``sqrt(N)`` argument predicts is most at risk --
    not because the sky is crowded where they sit. Three of them are nearby and unreddened
    (``A_V <= 0.9``, ``r50 >= 24'``), so a 1 mag core depression is implausible there and they
    almost certainly do not cross in reality; **Berkeley 53** (``A_V = 5.46``, ``r50 = 3.42'``,
    inner disc, ``d = 3.4 kpc``) is the one where it is plausible, and it is the single most
    likely real qualifier in the census. It is wired into ``TARGETS`` and its order 9-12 maps and
    naive fit are done (``R_c = 2.4382 +/- 0.1182'``, N = 1660, ``G_p98 = 20.16``); the corrected
    fits were still running when this note was written and are checkpointed, so
    ``--stage targets --only Berkeley_53`` resumes them. **Until that number exists the honest
    claim is bounded by the order-12 targets and must not be extended to the census.**
F2  **not triggered.** Flat ``S`` at ``S_0 = 0.999 / 0.5 / 0.2``: ``|dR_c| / sigma <= 0.063``,
    i.e. Monte-Carlo noise. And the check is not vacuous: feeding the *same* ``S(r)`` reversed
    moves M11 from ``-0.857`` to ``+0.025 sigma`` (cell ``mutation/completeness_node_order``).
F3  **not triggered.** Injected bias recovers the analytic ``I^{-1}w`` -- ratio 0.70 +/- 0.16
    (eps = 0.5%), 0.98 +/- 0.18 (1%), 1.13 (2%) -- and the corrected fit returns the injected
    truth (residual +0.018 +/- 0.023' at eps = 0, i.e. the estimator's own R_c bias is consistent
    with zero at N = 1911).
F4  **not triggered**, but the two halves of it carry very different weight and must not be
    quoted as one number. The *law* is exact, not fitted: ``I^{-1}w`` is identical to eight
    decimals when ``(k, b)`` are scaled over three decades, and the paired MCMC bias is flat in
    ``N`` over 1.2 dex -- 0.1192, 0.1247, 0.1369, 0.1349, 0.1389 arcmin at
    N = 500/1000/2000/4000/8000 (SEM 0.001-0.004) against the ``N``-free analytic prediction
    0.12371 -- while ``sigma`` falls 0.229 -> 0.063 and ``bias/sigma`` rises 0.520 -> 2.217. The
    *empirical slopes* are a consistency check on that, not the result:
    ``d log sigma / d log N = -0.469`` and ``d log(bias/sigma) / d log N = +0.524``, both inside
    the pre-registered +/-0.05. The residual +0.024 is a real second-order effect, not noise: the
    paired bias drifts up 17% across the ladder because the estimator's own ``R_c`` bias is itself
    ``N``-dependent. Rung N = 16000 is checkpointed and pending.
F5  **not triggered, but it decides what the census means.** eps at fixed healpix order 9/10/11/12
    from one archive query per target, at the observed depth: order 10 -> 12 changes eps by a
    factor **2.6x** (M11: 0.301% -> 0.771%), **61x** (NGC 6383: 0.00230% -> 0.140%) and **1.18x**
    (M5: 50.6% -> 59.7%). The last step 11 -> 12 is only +16%, +4.5% and +4.1%, so **order 12 is
    converged and order 10 is not.** The census-wide suppression distribution is therefore a
    lower bound of unknown and cluster-dependent tightness, and is NOT the evidence for the null:
    the order-12 targets, the resolution-free magnitude decomposition and the stress test are.
F6  **not triggered.** An independent live pull of all 1 291 929 member magnitudes reproduces
    every census number: 5647 open clusters, 7.14% with ``G_p98 > 20``, median ``G_p98`` 19.378,
    globular maximum 51.15% (NGC 5904).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLUSTERS = HERE / "hr24_clusters.ecsv"
OUT = HERE / "a7_selection_function_null.json"
CELLS = HERE / "a7_selection_function_null.cells.jsonl"
ARRAYS = HERE / "a7_selection_function_null.npz"  # *.npz is gitignored at the repo root
GMAG_CACHE = HERE / "a7_census_gmag.npz"
GMAG_CHUNKS = HERE / "a7_census_gmag.chunks"

TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
MEMBERS = '"J/A+A/686/A42/members"'

NODES = 256  # must match erotica.analysis.structure.king_expected_count_weighted
N_AZ = 96
QS = np.linspace(0.02, 0.98, 24)  # member-magnitude quantile nodes, as in the census statistic
DRAWS, TUNE, CHAINS = 1500, 1000, 4


# --------------------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------------------
def load_cells() -> dict:
    out = {}
    if CELLS.exists():
        for line in CELLS.read_text().splitlines():
            if line.strip():
                try:
                    rec = json.loads(line)
                    out[rec["cell"]] = rec
                except (json.JSONDecodeError, KeyError):
                    pass
    return out


def put_cell(key: str, payload: dict) -> dict:
    rec = dict(cell=key, t=time.strftime("%Y-%m-%dT%H:%M:%S"), **payload)
    with CELLS.open("a") as fh:
        fh.write(json.dumps(rec, default=_jsonable) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    return rec


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------------------
# King profile, its score, and the first-order bias oracle
# --------------------------------------------------------------------------------------
def king_sigma(r, k, b, R_c, R_t):
    core = 1.0 / np.sqrt(1.0 + (r / R_c) ** 2)
    edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
    return np.where(r <= R_t, k * (core - edge) ** 2 + b, b)


def _log_sigma(theta, r):
    return np.log(king_sigma(r, *theta))


def score(theta, r, rel_step=1e-5):
    """``u_j = d log Sigma / d theta_j`` by central differences.

    Finite differences rather than hand-coded derivatives: the ``k`` and ``b`` components have
    exact closed forms (``(core-edge)^2 / Sigma`` and ``1 / Sigma``), which
    :func:`check_score` uses as an internal oracle on the differencing itself.
    """
    theta = np.asarray(theta, dtype=float)
    u = np.empty((theta.size, np.asarray(r).size))
    for j in range(theta.size):
        h = rel_step * abs(theta[j])
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        u[j] = (_log_sigma(tp, r) - _log_sigma(tm, r)) / (2 * h)
    return u


def check_score(theta):
    """Oracle on :func:`score`: the ``k`` and ``b`` rows have exact closed forms."""
    k, b, R_c, R_t = theta
    r = np.linspace(1e-3, 0.9 * R_t, 501)
    u = score(theta, r)
    core = 1.0 / np.sqrt(1.0 + (r / R_c) ** 2)
    edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
    sig = k * (core - edge) ** 2 + b
    exact_k = (core - edge) ** 2 / sig
    exact_b = 1.0 / sig
    return dict(
        max_rel_err_k=float(np.max(np.abs(u[0] - exact_k) / np.abs(exact_k))),
        max_rel_err_b=float(np.max(np.abs(u[1] - exact_b) / np.abs(exact_b))),
    )


def first_order_bias(theta0, S_of_r, field_radius, nodes=200_001):
    """``delta theta = I^{-1} w`` -- the analytic prediction for the naive-fit bias.

    ``S_of_r`` is a callable evaluated on this routine's own quadrature grid. ``S`` is normalised
    by its value at the field edge, so only its **shape** enters; the constant part is absorbed
    exactly by ``(k, b)`` (that exactness is what stage ``flat`` tests empirically).

    Trapezoid on a dense uniform grid rather than Gauss-Legendre: ``Sigma`` has a kink at ``R_t``
    that a global polynomial rule handles badly, and ``leggauss`` at the order needed to smooth it
    costs O(n^3) in setup. Convergence is checked by :func:`check_first_order_bias`.
    """
    r = np.linspace(1e-9, field_radius, nodes)
    wq = np.gradient(r)
    S = np.asarray(S_of_r(r) if callable(S_of_r) else S_of_r, dtype=float)
    S = S / S[-1]  # S_0 := S at the field edge
    lam0 = 2.0 * np.pi * r * king_sigma(r, *theta0)
    u = score(theta0, r)
    # noqa: E741 is deliberate. `I` is the FISHER INFORMATION MATRIX, and this thread's entire
    # analytic backbone is stated as `delta-theta = eps * I^-1 * w` -- the same expression that
    # gives the N-independence which makes the expiry a calculation rather than an extrapolation.
    # Renaming it would make the code disagree with the result it computes, which is worse than
    # the ambiguity the rule guards against. Same call as `l` for galactic longitude in
    # analysis/dynamics.py.
    I = np.einsum("jr,lr,r->jl", u, u, wq * lam0)  # noqa: E741
    w = np.einsum("jr,r->j", u, wq * lam0 * (S - 1.0))
    return np.linalg.solve(I, w), I, w


def check_first_order_bias(theta0, S_of_r, field_radius):
    """Quadrature-convergence oracle on :func:`first_order_bias`: halving the grid must not move it."""
    a = first_order_bias(theta0, S_of_r, field_radius, nodes=100_001)[0]
    b = first_order_bias(theta0, S_of_r, field_radius, nodes=400_001)[0]
    return dict(max_rel_change_on_4x_refinement=float(np.max(np.abs(b - a) / np.abs(b))))


def king_radii(rng, n_expected, theta, field_radius, S_of_r=None, grid=200_001):
    """Inverse-CDF draw from ``lambda(r) = 2 pi r Sigma(r) S(r)`` on ``[0, R_f]``.

    ``n_expected`` is the *expected* detected count; the realised count is Poisson. Validated
    against the analytic King CDF by :func:`check_sampler`.
    """
    g = np.linspace(0.0, field_radius, grid)
    pdf = 2.0 * np.pi * g * king_sigma(g, *theta)
    if S_of_r is not None:
        pdf = pdf * np.asarray(S_of_r(g) if callable(S_of_r) else S_of_r, dtype=float)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(g))])
    cdf /= cdf[-1]
    n = int(rng.poisson(n_expected))
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, g)


def check_sampler(theta, field_radius, n=400_000, seed=1):
    """Generator oracle: empirical CDF of the draws against the analytic King CDF."""
    from erotica.analysis.structure import king_expected_count

    rng = np.random.default_rng(seed)
    r = king_radii(rng, n, theta, field_radius)
    k, b, R_c, R_t = theta
    edges = np.linspace(0.0, field_radius, 41)[1:]
    tot = king_expected_count(k, b, R_c, R_t, field_radius)
    analytic = np.array([king_expected_count(k, b, R_c, R_t, e) / tot for e in edges])
    emp = np.searchsorted(np.sort(r), edges) / r.size
    return dict(n=int(r.size), max_abs_cdf_err=float(np.max(np.abs(emp - analytic))))


# --------------------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------------------
def _v(x):
    return float(getattr(x, "value", x))


def fit(radii, field_radius, completeness=None, seed=7):
    from erotica.analysis.inference import SamplingConfig
    from erotica.analysis.structure import king_unbinned

    res = king_unbinned(
        radii,
        field_radius=field_radius,
        completeness=completeness,
        sampling=SamplingConfig(
            draws=DRAWS, tune=TUNE, chains=CHAINS, random_seed=seed, progressbar=False
        ),
        return_trace=False,
    )
    return {k: _v(res[k]) for k in ("R_c_median", "R_c_std", "R_t_median", "R_t_std", "k_median")}


def gl_nodes(field_radius, nodes=NODES):
    x, _ = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * field_radius * (x + 1.0)


# --------------------------------------------------------------------------------------
# STAGE flat -- F2
# --------------------------------------------------------------------------------------
# k, b, R_c, R_t and the field radius are NGC 6705 (M11)'s own naive posterior medians, measured
# by this file's `targets` stage, so the injection ladder is a synthetic M11 rather than a
# plausible-looking invention. b is ~0 because HR24 memberships are probability-selected: the
# fitted background is 1.2 stars out of 1913.
THETA0 = (37.30556, 5.0e-5, 2.75943, 30.05936)
FIELD = 84.8341464
N_M11 = 1911


def stage_flat(cells, seeds=3):
    print("[flat] flat-S invariance through king_unbinned (F2)")
    print("  score oracle:", check_score(THETA0))
    print("  sampler oracle:", check_sampler(THETA0, FIELD))
    _, _, S5, _ = core_suppression(FIELD, 0.05)
    print("  quadrature oracle:", check_first_order_bias(THETA0, S5, FIELD))
    rr = gl_nodes(FIELD)
    for s in range(seeds):
        rng = np.random.default_rng(9000 + s)
        radii = king_radii(rng, N_M11, THETA0, FIELD)
        base = fit(radii, FIELD, seed=101 + s)
        for S0 in (0.999, 0.5, 0.2):
            key = f"flat/s{s}/S{S0}"
            if key in cells:
                continue
            f = fit(radii, FIELD, completeness=np.full(rr.shape, S0), seed=101 + s)
            d = f["R_c_median"] - base["R_c_median"]
            rec = put_cell(
                key,
                dict(
                    seed=s,
                    S0=S0,
                    n=int(radii.size),
                    R_c_flat=f["R_c_median"],
                    R_c_base=base["R_c_median"],
                    sd=base["R_c_std"],
                    d_over_sigma=d / base["R_c_std"],
                ),
            )
            print(f"  S0={S0:<6} dR_c/sigma = {rec['d_over_sigma']:+.4f}", flush=True)


# --------------------------------------------------------------------------------------
# STAGE injection -- F3
# --------------------------------------------------------------------------------------
def core_suppression(field_radius, eps, r_s=6.0, sign=+1.0):
    """``S(r) = 1 - sign * a * exp(-r/r_s)`` normalised so ``1 - S_in/S_out == eps``.

    ``S_in`` and ``S_out`` use the census apertures: ``r <= 0.045 R_f`` (an M11-like ``r50/R_f``)
    and ``r >= 0.65 R_f``. The shape is held **fixed** across the amplitude ladder, which is what
    makes the linearity test in this stage a clean one -- unlike shifting a real cluster's
    luminosity function, where the shape of ``g(r)`` moves with the amplitude.
    """
    r = gl_nodes(field_radius)
    inner, outer = r <= 0.045 * field_radius, r >= 0.65 * field_radius
    e = np.exp(-r / r_s)
    ein, eout = e[inner].mean(), e[outer].mean()
    # (S_out - S_in)/S_out = sign * a (ein - eout) / (1 - sign a eout) = sign * eps
    a = eps / (ein - eout + sign * eps * eout)
    # Rescale to max 1 so the enhancement branch stays a probability. A constant factor is
    # absorbed exactly by (k, b) -- that is the stage-`flat` result -- so the shape is unchanged.
    norm = float(np.max(1.0 - sign * a * np.exp(-np.linspace(0, field_radius, 4001) / r_s)))

    def S_of_r(x):
        return (1.0 - sign * a * np.exp(-np.asarray(x, dtype=float) / r_s)) / norm

    return r, S_of_r(r), S_of_r, float(a)


# eps = 0 is the zero-suppression control: it measures the estimator's own R_c bias at this N,
# which every other rung is differenced against. Without it "consistent with zero" and "too noisy
# to tell" are indistinguishable -- the failure `completeness_bias_scaling.py` exists to avoid.
EPS_LADDER = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)


def stage_injection(cells, seeds=4, n_expected=N_M11):
    print("[injection] known truth + known S(r); oracle = analytic I^-1 w (F3)")
    for eps in EPS_LADDER:
        for sign, tag in ((+1.0, "sup"), (-1.0, "enh")):
            if sign < 0 and eps not in (0.05, 0.20):
                continue  # sign-flip mutation only needs two rungs
            rr, S, S_fn, a = core_suppression(FIELD, eps, sign=sign)
            pred, _, _ = first_order_bias(THETA0, S_fn, FIELD)
            for s in range(seeds):
                key = f"inj/{tag}/eps{eps}/s{s}"
                if key in cells:
                    continue
                rng = np.random.default_rng(31_000 + 977 * s)  # paired across the ladder
                radii = king_radii(rng, n_expected, THETA0, FIELD, S_of_r=S_fn)
                naive = fit(radii, FIELD, seed=555 + s)
                corr = fit(radii, FIELD, completeness=S, seed=555 + s)
                rec = put_cell(
                    key,
                    dict(
                        eps=eps,
                        sign=sign,
                        seed=s,
                        n=int(radii.size),
                        R_c_true=THETA0[2],
                        R_c_naive=naive["R_c_median"],
                        sd_naive=naive["R_c_std"],
                        R_c_corr=corr["R_c_median"],
                        sd_corr=corr["R_c_std"],
                        pred_dR_c=float(pred[2]),
                    ),
                )
                print(
                    f"  eps={eps:<6} {tag} s{s}  naive {rec['R_c_naive']:.4f}  "
                    f"corr {rec['R_c_corr']:.4f}  bias {rec['R_c_naive'] - THETA0[2]:+.4f}  "
                    f"pred {rec['pred_dR_c']:+.4f}",
                    flush=True,
                )


# --------------------------------------------------------------------------------------
# STAGE sqrtn -- F4
# --------------------------------------------------------------------------------------
# eps is set high enough that the signal is large against Monte-Carlo noise at every rung: the
# point of this stage is the *exponent*, not the amplitude, and the amplitude is already
# calibrated against the analytic oracle by `injection`.
N_LADDER = (500, 1000, 2000, 4000, 8000, 16000)
EPS_FIXED = 0.10


def stage_sqrtn(cells, seeds=6):
    print(f"[sqrtn] bias/sigma vs N at fixed eps={EPS_FIXED} (F4)")
    rr, S, S_fn, a = core_suppression(FIELD, EPS_FIXED)
    pred, _, _ = first_order_bias(THETA0, S_fn, FIELD)
    for n in N_LADDER:
        # scale (k, b) together so lambda_0 scales: I^-1 w is invariant, sigma ~ 1/sqrt(N)
        scale = n / float(N_M11)
        theta = (THETA0[0] * scale, THETA0[1] * scale, THETA0[2], THETA0[3])
        for s in range(seeds):
            key = f"sqrtn/N{n}/s{s}"
            if key in cells:
                continue
            rng = np.random.default_rng(77_000 + 613 * s)  # paired across the ladder
            radii = king_radii(rng, n, theta, FIELD, S_of_r=S_fn)
            naive = fit(radii, FIELD, seed=333 + s)
            corr = fit(radii, FIELD, completeness=S, seed=333 + s)
            rec = put_cell(
                key,
                dict(
                    N=n,
                    seed=s,
                    eps=EPS_FIXED,
                    n_real=int(radii.size),
                    R_c_true=THETA0[2],
                    R_c_naive=naive["R_c_median"],
                    sd_naive=naive["R_c_std"],
                    R_c_corr=corr["R_c_median"],
                    sd_corr=corr["R_c_std"],
                    pred_dR_c=float(pred[2]),
                ),
            )
            print(
                f"  N={n:<6} s{s}  bias {rec['R_c_naive'] - THETA0[2]:+.4f}  "
                f"sd {rec['sd_naive']:.4f}  b/s {(rec['R_c_naive'] - THETA0[2]) / rec['sd_naive']:+.3f}",
                flush=True,
            )


# --------------------------------------------------------------------------------------
# STAGE targets -- F1 / F5
# --------------------------------------------------------------------------------------
TARGETS = {
    # name: (patch radius factor, extra depth rungs)
    "NGC_6705": ("OC", (20.0, 20.43, 21.0, 21.5)),
    "NGC_6383": ("OC", (20.43, 21.0)),
    "NGC_5904": ("GC", (21.0,)),
    # The one census cluster the stress test flags that is BOTH quality-flagged and plausibly
    # crowded (A_V = 5.46, r50 = 3.42', inner disc, 3.4 kpc). Measuring it is what turns F1's
    # census bound from a hypothetical into a result.
    "Berkeley_53": ("OC", (20.43,)),
}


def _cluster_row(cl, name):
    i = list(map(str, cl["Name"])).index(name)
    return {c: cl[c][i] for c in cl.colnames}


def _members(name, centre, field_am):
    import astropy.units as u
    import pyvo
    from astropy.coordinates import SkyCoord

    tap = pyvo.dal.TAPService(TAP_URL)
    t = tap.search(
        f'SELECT "RA_ICRS","DE_ICRS","Gmag" FROM {MEMBERS} WHERE "Name"=\'{name}\'', maxrec=100_000
    ).to_table()
    ra = np.asarray(t["RA_ICRS"], float)
    de = np.asarray(t["DE_ICRS"], float)
    g = np.asarray(t["Gmag"], float)
    rad = SkyCoord(ra * u.deg, de * u.deg).separation(centre).arcmin
    keep = np.isfinite(rad) & (rad < field_am) & np.isfinite(g)
    return np.asarray(rad)[keep], g[keep]


def fetch_patch_sources(centre, radius_deg, cache: Path):
    """The Gaia query `gaiaunlimited`'s patch builder makes, issued **once** per target.

    Identical ADQL to ``surveyTCG.build_patch_map``: ``astrometric_matched_transits < 11`` is the
    proxy subsample whose median ``G`` *is* ``M_10``. Cached to ``.npz`` (gitignored) so the
    order 9/10/11/12 ladder costs one archive hit, not four.
    """
    if cache.exists():
        d = np.load(cache)
        return d["source_id"], d["gmag"]
    from astroquery.gaia import Gaia

    q = (
        "SELECT source_id, phot_g_mean_mag FROM gaiadr3.gaia_source "
        f"WHERE 1 = CONTAINS(POINT(ra,dec),CIRCLE({centre.icrs.ra.deg:.6f},"
        f"{centre.icrs.dec.deg:.6f},{radius_deg:.6f})) "
        "AND astrometric_matched_transits<11 AND phot_g_mean_mag<50"
    )
    t0 = time.time()
    t = Gaia.launch_job_async(q).get_results()
    sid = np.asarray(t["source_id" if "source_id" in t.colnames else "SOURCE_ID"], dtype=np.int64)
    g = np.asarray(t["phot_g_mean_mag"], dtype=float)
    np.savez_compressed(cache, source_id=sid, gmag=g)
    print(f"  archive: {sid.size} sources in {time.time() - t0:.0f}s -> {cache.name}", flush=True)
    return sid, g


def build_m10_map(source_id, gmag, order, min_points=20, base_order=6):
    """``M_10`` on a fixed healpix ``order``, with the coarser-pixel fallback of the patch builder.

    ``gaiaunlimited``'s own builder hard-codes a maximum order of 12. Making it a parameter is the
    whole point: it turns "is the answer resolution-limited?" from an assumption into a measurement
    (falsifier F5). The rule is unchanged -- median ``G`` of the proxy subsample in the pixel,
    stepping up to the parent wherever a pixel holds fewer than ``min_points`` sources.
    """
    import pandas as pd

    hp12 = source_id // 2**35
    pix = np.unique(hp12 // 4 ** (12 - order))
    vals = np.full(pix.size, np.nan)
    for step in range(order - base_order + 1):
        o = order - step
        h = hp12 // 4 ** (12 - o)
        df = pd.DataFrame({"h": h, "g": gmag}).groupby("h")["g"].agg(["median", "count"])
        parent = pix // 4**step
        idx = df.index.to_numpy()
        j = np.searchsorted(idx, parent)
        ok = (j < idx.size) & (idx[np.clip(j, 0, idx.size - 1)] == parent)
        med = df["median"].to_numpy()
        cnt = df["count"].to_numpy()
        fill = np.isnan(vals) & ok
        fill[fill] &= cnt[j[fill]] >= min_points
        vals[fill] = med[j[fill]]
        if not np.isnan(vals).any():
            break
    return pix, vals


def m10_profile(pix, vals, order, centre, field_am):
    """Azimuth x radius grid of ``M_10`` from a fixed-order pixel list."""
    import astropy.units as u
    from gaiaunlimited import utils

    rr = gl_nodes(field_am)
    az = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)
    M = np.empty((rr.size, N_AZ))
    for i, r in enumerate(rr):
        pts = centre.directional_offset_by(az * u.rad, (r * u.arcmin).to(u.deg))
        ip = utils.coord2healpix(pts, "icrs", 2**order)
        j = np.searchsorted(pix, ip)
        j = np.clip(j, 0, pix.size - 1)
        M[i] = np.where(pix[j] == ip, vals[j], np.nan)
    return rr, M


def _m10_grid(m10map, centre, field_am, order):
    """Azimuth x radius grid of ``M10`` from a patch- or multi-mode map."""
    import astropy.units as u
    from gaiaunlimited import utils

    ipix = m10map[:, 1].astype(np.int64)
    mm = m10map[:, 2]
    srt = np.argsort(ipix)
    rr = gl_nodes(field_am)
    az = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)
    M = np.empty((rr.size, N_AZ))
    for i, r in enumerate(rr):
        pts = centre.directional_offset_by(az * u.rad, (r * u.arcmin).to(u.deg))
        ip = utils.coord2healpix(pts, "icrs", 2**order)
        j = srt[np.clip(np.searchsorted(ipix[srt], ip), 0, len(srt) - 1)]
        M[i] = np.where(ipix[j] == ip, mm[j], np.nan)
    return rr, M


def _multi_m10_grid(centre, field_am):
    """Same grid from the all-sky density-adaptive map (order 7-10). Records the pixel scale."""
    import astropy.units as u
    import astropy_healpix as ah
    from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG

    sf = DR3SelectionFunctionTCG(mode="multi")
    level, _ = ah.uniq_to_level_ipix(np.asarray(sf.m10map["UNIQ"]))
    index, sorter = np.asarray(sf.index), np.asarray(sf.sorter)
    m10 = np.asarray(sf.m10map["M10"], dtype=float)
    rr = gl_nodes(field_am)
    az = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)
    M = np.empty((rr.size, N_AZ))
    L = np.empty((rr.size, N_AZ))
    for i, r in enumerate(rr):
        pts = centre.directional_offset_by(az * u.rad, (r * u.arcmin).to(u.deg))
        ip = ah.lonlat_to_healpix(pts.ra, pts.dec, sf.max_nside, order="nested")
        j = sorter[np.searchsorted(index, ip, side="right", sorter=sorter) - 1]
        M[i], L[i] = m10[j], level[j]
    pix = ah.nside_to_pixel_resolution(ah.level_to_nside(int(np.median(L)))).to_value(u.arcmin)
    return rr, M, dict(lvl_min=int(L.min()), lvl_max=int(L.max()), pix_arcmin=float(pix))


def _sbar(M, mag_nodes):
    from gaiaunlimited.selectionfunctions.surveyTCG import m10_to_completeness

    S = m10_to_completeness(
        np.asarray(mag_nodes)[None, None, :],
        np.repeat(M[:, :, None], len(mag_nodes), axis=2),
    )
    return np.nanmean(S, axis=(1, 2))


def stage_targets(cells, only=None):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    cl = Table.read(CLUSTERS)
    for name, (kind, rungs) in TARGETS.items():
        if only and name not in only:
            continue
        row = _cluster_row(cl, name)
        centre = SkyCoord(float(row["RA_ICRS"]) * u.deg, float(row["DE_ICRS"]) * u.deg)
        rref = float(row["rJ"]) if np.isfinite(row["rJ"]) else float(row["rt"])
        field = 2.0 * rref * 60.0
        r50 = float(row["r50"]) * 60.0
        rad, g = _members(name, centre, field)
        print(
            f"\n[targets] {name} ({kind})  N={rad.size}  r50={r50:.2f}'  field={field:.1f}'  "
            f"G_med={np.median(g):.2f}  p98={np.percentile(g, 98):.2f}",
            flush=True,
        )

        # --- the resolution ladder, F5. `multi` is the all-sky map the census uses (order 7-10);
        # `cap9..cap12` are built from ONE archive query on the same sources, so the resolution
        # dependence is measured rather than assumed.
        rr, M_multi, meta_multi = _multi_m10_grid(centre, field)
        prad = field / 60.0 * 1.06 + 0.03
        sid, gsrc = fetch_patch_sources(centre, prad, HERE / f"a7_patch_sources_{name}.npz")
        maps = {"multi_o10": M_multi}
        pixmeta = {"multi_o10": meta_multi}
        for order in (9, 10, 11, 12):
            pix, vals = build_m10_map(sid, gsrc, order)
            _, M = m10_profile(pix, vals, order, centre, field)
            maps[f"cap{order}"] = M
            pixmeta[f"cap{order}"] = dict(
                n_pix=int(pix.size),
                n_filled=int(np.isfinite(vals).sum()),
                nan_frac_on_grid=float(np.isnan(M).mean()),
                pix_arcmin=float(58.6323 / 2 ** (order - 6)),
            )
            print(
                f"  cap{order}: {pix.size} px, {np.isfinite(vals).sum()} filled, "
                f"grid NaN {np.isnan(M).mean():.3f}",
                flush=True,
            )

        base = np.quantile(g, QS)
        naive = fit(rad, field, seed=7)
        print(f"  naive R_c = {naive['R_c_median']:.4f} +/- {naive['R_c_std']:.4f}'", flush=True)
        inner, outer = rr <= r50, rr >= 0.65 * field

        for depth in (None,) + tuple(rungs):
            mag = base if depth is None else base + depth - np.percentile(g, 98)
            label = "observed" if depth is None else f"p98={depth}"
            for res, M in maps.items():
                key = f"tgt/{name}/{res}/{label}"
                if key in cells:
                    continue
                S = _sbar(M, mag)
                if not np.all(np.isfinite(S)):
                    S = np.interp(rr, rr[np.isfinite(S)], S[np.isfinite(S)])
                S = np.clip(S, 1e-9, 1.0)
                eps = float(1.0 - S[inner].mean() / S[outer].mean())
                f = fit(rad, field, completeness=S, seed=7)
                d = f["R_c_median"] - naive["R_c_median"]
                rec = put_cell(
                    key,
                    dict(
                        name=name,
                        kind=kind,
                        resolution=res,
                        depth=label,
                        N=int(rad.size),
                        field_arcmin=field,
                        r50_arcmin=r50,
                        G_p98=float(np.percentile(mag, 98)) if depth is None else float(depth),
                        eps=eps,
                        S_min=float(S.min()),
                        S_max=float(S.max()),
                        M10_centre=float(np.nanmean(M[0])),
                        M10_outer=float(np.nanmean(M[outer])),
                        R_c_naive=naive["R_c_median"],
                        sd_naive=naive["R_c_std"],
                        R_c_corr=f["R_c_median"],
                        sd_corr=f["R_c_std"],
                        shift_pct=100.0 * d / naive["R_c_median"],
                        sigma_naive_convention=d / naive["R_c_std"],
                        sigma_hypot_convention=d / np.hypot(naive["R_c_std"], f["R_c_std"]),
                        map_meta=pixmeta[res],
                        nan_frac=float(np.isnan(_sbar(M, mag)).mean()),
                    ),
                )
                print(
                    f"  {label:<10} {res:<10} eps={eps:+8.4%}  R_c {rec['R_c_corr']:.4f}  "
                    f"shift {rec['shift_pct']:+7.3f}%  "
                    f"{rec['sigma_naive_convention']:+6.3f} sd_naive / "
                    f"{rec['sigma_hypot_convention']:+6.3f} hypot",
                    flush=True,
                )


# --------------------------------------------------------------------------------------
# STAGE census -- F6
# --------------------------------------------------------------------------------------
def fetch_gmag(chunk=100_000):
    """Cache per-cluster member ``G`` quantile nodes for the whole census. Chunked + resumable."""
    if GMAG_CACHE.exists():
        return np.load(GMAG_CACHE, allow_pickle=False)
    import pyvo

    GMAG_CHUNKS.mkdir(exist_ok=True)
    tap = pyvo.dal.TAPService(TAP_URL)
    total = int(tap.search(f"SELECT COUNT(*) AS n FROM {MEMBERS}").to_table()["n"][0])
    print(f"[census] {MEMBERS} has {total} rows; pulling Name,Gmag in chunks of {chunk}")
    names, mags = [], []
    lo, t0 = 0, time.time()
    while lo < total:
        hi = lo + chunk
        part = GMAG_CHUNKS / f"{lo:09d}.npz"
        if not part.exists():
            q = f"SELECT recno, Name, Gmag FROM {MEMBERS} WHERE recno > {lo} AND recno <= {hi}"
            for attempt in range(8):
                try:
                    t = tap.search(q, maxrec=chunk + 10).to_table()
                    break
                except Exception as exc:  # noqa: BLE001 -- transient TAP overload
                    wait = min(120, 5 * 2**attempt)
                    print(
                        f"    retry {attempt + 1}/8 in {wait}s ({type(exc).__name__})", flush=True
                    )
                    time.sleep(wait)
            else:
                raise RuntimeError(f"chunk ({lo}, {hi}] failed after 8 attempts")
            np.savez_compressed(
                part,
                name=np.asarray(t["Name"], dtype="U24"),
                gmag=np.asarray(t["Gmag"], dtype=np.float32),
            )
            print(
                f"  ({lo:>8d}, {hi:>8d}] {len(t):>7d} rows [{time.time() - t0:6.1f}s]", flush=True
            )
        d = np.load(part)
        names.append(d["name"])
        mags.append(d["gmag"])
        lo = hi
    name = np.concatenate(names)
    gmag = np.concatenate(mags).astype(float)
    o = np.argsort(name, kind="stable")
    np.savez_compressed(GMAG_CACHE, name=name[o], gmag=gmag[o])
    print(f"[census] cached {name.size} member magnitudes -> {GMAG_CACHE.name}")
    return np.load(GMAG_CACHE)


def stage_census(cells, limit=None):
    """Recompute the census suppression statistic live (F6), with resolution diagnostics."""
    import astropy.units as u
    import astropy_healpix as ah
    from astropy.table import Table
    from gaiaunlimited.selectionfunctions import DR3SelectionFunctionTCG
    from gaiaunlimited.selectionfunctions.surveyTCG import m10_to_completeness

    d = fetch_gmag()
    mname, mg = d["name"], d["gmag"].astype(float)

    sf = DR3SelectionFunctionTCG(mode="multi")
    level, _ = ah.uniq_to_level_ipix(np.asarray(sf.m10map["UNIQ"]))
    index, sorter = np.asarray(sf.index), np.asarray(sf.sorter)
    m10 = np.asarray(sf.m10map["M10"], dtype=float)
    max_nside = sf.max_nside

    cl = Table.read(CLUSTERS)
    types = np.asarray(cl["Type"], dtype=str)
    x, _ = np.polynomial.legendre.leggauss(NODES)
    unit_r = 0.5 * (x + 1.0)
    az = np.linspace(0.0, 2 * np.pi, N_AZ, endpoint=False)

    def num(v):
        try:
            y = float(v)
        except (TypeError, ValueError):
            return np.nan
        return y if np.isfinite(y) else np.nan

    starts = np.searchsorted(mname, np.asarray(cl["Name"], dtype=mname.dtype), "left")
    stops = np.searchsorted(mname, np.asarray(cl["Name"], dtype=mname.dtype), "right")

    rows, t0 = [], time.time()
    sel = np.where(np.isin(types, ("o", "g")))[0]
    if limit:
        sel = sel[:: max(1, len(sel) // limit)][:limit]
    for c, k in enumerate(sel):
        rJ, rt, rtot, r50 = (num(cl[col][k]) for col in ("rJ", "rt", "rtot", "r50"))
        rref = rJ if np.isfinite(rJ) else (rt if np.isfinite(rt) else rtot)
        if not np.isfinite(rref) or rref <= 0 or not np.isfinite(r50):
            continue
        gmag = mg[starts[k] : stops[k]]
        gmag = gmag[np.isfinite(gmag)]
        if gmag.size < 5:
            continue
        Rf = 2.0 * rref
        rr = unit_r * Rf
        ra0, dec0 = np.radians(num(cl["RA_ICRS"][k])), np.radians(num(cl["DE_ICRS"][k]))
        rho = np.radians(rr)[:, None]
        th = az[None, :]
        sd = np.sin(dec0) * np.cos(rho) + np.cos(dec0) * np.sin(rho) * np.cos(th)
        dec = np.arcsin(np.clip(sd, -1, 1))
        ra = ra0 + np.arctan2(
            np.sin(th) * np.sin(rho) * np.cos(dec0), np.cos(rho) - np.sin(dec0) * sd
        )
        ip = ah.lonlat_to_healpix(
            np.degrees(ra).ravel() * u.deg,
            np.degrees(dec).ravel() * u.deg,
            max_nside,
            order="nested",
        )
        j = sorter[np.searchsorted(index, ip, side="right", sorter=sorter) - 1]
        M = m10[j].reshape(NODES, N_AZ)
        lv = level[j]
        mgrid = np.quantile(gmag, QS)
        S = m10_to_completeness(mgrid[None, None, :], np.repeat(M[:, :, None], QS.size, axis=2))
        Sb = np.nanmean(S, axis=(1, 2))
        inner, outer = rr <= r50, rr >= 0.65 * Rf
        if inner.sum() < 2 or outer.sum() < 2:
            continue
        # RESOLUTION-FREE STRESS TEST. The map cannot resolve a core, so `suppression` above is a
        # lower bound and is not evidence for anything. This instead asks the inverse question,
        # which needs no core resolution at all: *how deep would an unresolved core depression in
        # M_10 have to be* for this cluster to qualify? The outer M_10 is taken from the map --
        # legitimate, since the outer annulus spans many pixels whatever the order -- and the core
        # is set to `M10_out - delta` for a ladder of hypothetical delta. For scale, the depressions
        # actually measured at order 12 are 1.14 mag (M11) and 0.74 mag (NGC 6383).
        m10_out = float(np.nanmean(M[outer]))
        s_out = float(np.nanmean(m10_to_completeness(mgrid, np.full(QS.size, m10_out))))
        stress = {
            f"eps_delta{d}": float(
                1.0 - np.nanmean(m10_to_completeness(mgrid, np.full(QS.size, m10_out - d))) / s_out
            )
            for d in (0.5, 1.0, 1.5, 2.0, 3.0)
        }
        rows.append(
            dict(
                Name=str(cl["Name"][k]),
                Type=str(types[k]),
                N=int(cl["N"][k]),
                G_med=float(np.median(gmag)),
                G_p98=float(np.percentile(gmag, 98)),
                suppression=float(1.0 - Sb[inner].mean() / Sb[outer].mean()),
                M10_out=m10_out,
                **stress,
                r50_am=r50 * 60.0,
                rc_am=num(cl["rc"][k]) * 60.0,
                npix=int(np.unique(j).size),
                lvl_min=int(lv.min()),
                lvl_max=int(lv.max()),
                pix_am=float(
                    ah.nside_to_pixel_resolution(ah.level_to_nside(int(np.median(lv)))).to_value(
                        u.arcmin
                    )
                ),
            )
        )
        if c % 500 == 0:
            print(f"  {c}/{len(sel)} {time.time() - t0:.0f}s", flush=True)

    np.savez_compressed(
        HERE / "a7_census_recomputed.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]}
    )
    o = np.array([r for r in rows if r["Type"] == "o"])
    g_ = np.array([r for r in rows if r["Type"] == "g"])
    so = np.array([r["suppression"] for r in o])
    sg = np.array([r["suppression"] for r in g_])
    gp = np.array([r["G_p98"] for r in o])
    npix = np.array([r["npix"] for r in o])
    summary = dict(
        n_oc=len(o),
        n_gc=len(g_),
        oc_supp_median=float(np.nanmedian(so)),
        oc_supp_p90=float(np.nanpercentile(so, 90)),
        oc_supp_p99=float(np.nanpercentile(so, 99)),
        oc_supp_max=float(np.nanmax(so)),
        oc_n_above_0p258pct=int((so > 0.00258).sum()),
        oc_n_above_0p5pct=int((so > 0.005).sum()),
        oc_frac_Gp98_gt20=float((gp > 20).mean()),
        oc_G_med_median=float(np.median([r["G_med"] for r in o])),
        oc_G_p98_median=float(np.median(gp)),
        gc_supp_median=float(np.nanmedian(sg)),
        gc_supp_p90=float(np.nanpercentile(sg, 90)),
        gc_supp_max=float(np.nanmax(sg)),
        gc_n_above_5pct=int((sg > 0.05).sum()),
        gc_argmax=str(g_[int(np.nanargmax(sg))]["Name"]),
        oc_npix_p01=float(np.percentile(npix, 1)),
        oc_npix_median=float(np.median(npix)),
        oc_n_lvlmin7=int(sum(r["lvl_min"] == 7 for r in o)),
        oc_frac_r50_below_pixel=float(np.mean([r["r50_am"] < r["pix_am"] for r in o])),
    )
    for d in (0.5, 1.0, 1.5, 2.0, 3.0):
        for tag, tab in (("oc", o), ("gc", g_)):
            v = np.array([r[f"eps_delta{d}"] for r in tab])
            summary[f"{tag}_stress_delta{d}_median"] = float(np.nanmedian(v))
            summary[f"{tag}_stress_delta{d}_p99"] = float(np.nanpercentile(v, 99))
            summary[f"{tag}_stress_delta{d}_max"] = float(np.nanmax(v))
    put_cell("census/summary", summary)
    for k, v in summary.items():
        print(f"  {k:28s} {v}")


# --------------------------------------------------------------------------------------
# STAGE expiry -- when does the null die?
# --------------------------------------------------------------------------------------
def stage_expiry(cells, name="NGC_6705", order=12):
    r"""The expiry date, computed rather than extrapolated.

    ``delta theta = I^{-1} w`` is **exactly independent of the sample size** (verified here by
    scaling ``(k, b)`` over three decades: ``dR_c`` is identical to eight decimals), while
    ``sigma ~ N^{-1/2}``. So ``|bias| / sigma`` at any ``(depth, N)`` follows from one evaluation
    of the oracle per depth plus the measured ``sigma`` at the present ``N`` -- no MCMC, and no
    reliance on the fitted exponent beyond the ``sqrtn`` stage's confirmation of it.

    The depth axis is the cluster's own luminosity function shifted rigidly to a target ``G_p98``.
    **That is the weak assumption and it biases the expiry EARLY**: a real deeper membership adds
    stars at the faint end only, whereas a rigid shift also drags the bright end down, so it
    overstates the fraction of members sitting near the DR3 edge. Reported as a bound accordingly.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table

    cl = Table.read(CLUSTERS)
    row = _cluster_row(cl, name)
    centre = SkyCoord(float(row["RA_ICRS"]) * u.deg, float(row["DE_ICRS"]) * u.deg)
    rref = float(row["rJ"]) if np.isfinite(row["rJ"]) else float(row["rt"])
    field = 2.0 * rref * 60.0
    r50 = float(row["r50"]) * 60.0
    rad, g = _members(name, centre, field)
    sid, gsrc = fetch_patch_sources(
        centre, field / 60.0 * 1.06 + 0.03, HERE / f"a7_patch_sources_{name}.npz"
    )
    pix, vals = build_m10_map(sid, gsrc, order)
    rr, M = m10_profile(pix, vals, order, centre, field)
    inner, outer = rr <= r50, rr >= 0.65 * field

    naive = fit(rad, field, seed=7)
    theta0 = (naive["k_median"], 5.0e-5, naive["R_c_median"], naive["R_t_median"])
    sd_now, n_now = naive["R_c_std"], int(rad.size)
    base = np.quantile(g, QS)
    p98_now = float(np.percentile(g, 98))

    rows = []
    for target in [p98_now] + list(np.arange(19.5, 21.75, 0.25)):
        mag = base + target - p98_now
        S = _sbar(M, mag)
        if not np.all(np.isfinite(S)):
            S = np.interp(rr, rr[np.isfinite(S)], S[np.isfinite(S)])
        eps = float(1.0 - S[inner].mean() / S[outer].mean())
        pred, _, _ = first_order_bias(theta0, lambda x, S=S: np.interp(x, rr, S), field)
        d = float(pred[2])
        z_now = abs(d) / sd_now
        rows.append(
            dict(
                G_p98=float(target),
                eps=eps,
                dR_c_analytic=d,
                z_at_N_now=z_now,
                N_crit_1sigma=float(n_now / z_now**2) if z_now > 0 else None,
            )
        )
        print(
            f"  p98={target:5.2f}  eps={eps:+8.4%}  dR_c={d:+.5f}'  "
            f"z(N={n_now})={z_now:6.3f}  N_crit={rows[-1]['N_crit_1sigma']:.0f}",
            flush=True,
        )
    put_cell(
        f"expiry/{name}/o{order}",
        dict(
            name=name,
            order=order,
            N_now=n_now,
            sd_now=sd_now,
            R_c_now=naive["R_c_median"],
            G_p98_now=p98_now,
            theta0=list(theta0),
            rows=rows,
        ),
    )


# --------------------------------------------------------------------------------------
# summariser
# --------------------------------------------------------------------------------------
def summarise(cells):
    import math

    out = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "theta0": list(THETA0), "field": FIELD}

    # --- flat
    flat = [c for k, c in cells.items() if k.startswith("flat/")]
    if flat:
        out["flat"] = dict(
            n=len(flat),
            max_abs_d_over_sigma=max(abs(c["d_over_sigma"]) for c in flat),
            by_S0={
                str(S0): float(np.mean([abs(c["d_over_sigma"]) for c in flat if c["S0"] == S0]))
                for S0 in sorted({c["S0"] for c in flat})
            },
        )

    # --- injection
    inj = [c for k, c in cells.items() if k.startswith("inj/")]
    if inj:
        rung = {}
        for c in inj:
            key = f"{'sup' if c['sign'] > 0 else 'enh'}/{c['eps']}"
            rung.setdefault(key, []).append(c)
        # eps = 0 is the zero-suppression control; every rung is differenced against it at the
        # SAME seed, which reuses the same uniform draws and so removes realisation scatter.
        base = {c["seed"]: c["R_c_naive"] for c in inj if c["eps"] == 0.0}
        rows = []
        for key, cs in sorted(
            rung.items(), key=lambda kv: (kv[0].split("/")[0], float(kv[0].split("/")[1]))
        ):
            d = [c["R_c_naive"] - base[c["seed"]] for c in cs if c["seed"] in base]
            resid = [c["R_c_corr"] - c["R_c_true"] for c in cs]
            n = max(len(d), 1)
            rows.append(
                dict(
                    rung=key,
                    eps=cs[0]["eps"],
                    sign=cs[0]["sign"],
                    n_seeds=len(cs),
                    bias_paired=float(np.mean(d)) if d else None,
                    bias_paired_sem=float(np.std(d, ddof=1) / math.sqrt(n)) if len(d) > 1 else None,
                    bias_vs_truth=float(np.mean([c["R_c_naive"] - c["R_c_true"] for c in cs])),
                    pred_first_order=float(cs[0]["pred_dR_c"]),
                    ratio_measured_over_pred=(
                        float(np.mean(d) / cs[0]["pred_dR_c"]) if d and cs[0]["pred_dR_c"] else None
                    ),
                    resid_corrected=float(np.mean(resid)),
                    resid_sem=(
                        float(np.std(resid, ddof=1) / math.sqrt(len(resid)))
                        if len(resid) > 1
                        else None
                    ),
                    sd_typ=float(np.mean([c["sd_naive"] for c in cs])),
                )
            )
        out["injection"] = rows

    # --- sqrt N
    sq = [c for k, c in cells.items() if k.startswith("sqrtn/")]
    if sq:
        byN = {}
        for c in sq:
            byN.setdefault(c["N"], []).append(c)
        rows = []
        for n, cs in sorted(byN.items()):
            # naive - corrected on the SAME realisation: this is exactly the quantity a real
            # cluster reports as "the shift", and the pairing cancels realisation scatter.
            paired = [c["R_c_naive"] - c["R_c_corr"] for c in cs]
            sd = float(np.mean([c["sd_naive"] for c in cs]))
            rows.append(
                dict(
                    N=n,
                    n_seeds=len(cs),
                    bias=float(np.mean(paired)),
                    bias_sem=(
                        float(np.std(paired, ddof=1) / math.sqrt(len(cs))) if len(cs) > 1 else None
                    ),
                    bias_vs_truth=float(np.mean([c["R_c_naive"] - c["R_c_true"] for c in cs])),
                    sd=sd,
                    bias_over_sd=float(np.mean(paired)) / sd,
                    pred_first_order=float(cs[0]["pred_dR_c"]),
                )
            )
        lg = np.log(np.array([r["N"] for r in rows], float))
        ls = np.log(np.array([r["sd"] for r in rows], float))
        lb = np.log(np.abs(np.array([r["bias_over_sd"] for r in rows], float)))
        out["sqrtn"] = dict(
            rows=rows,
            slope_log_sd_vs_log_N=float(np.polyfit(lg, ls, 1)[0]),
            slope_log_biasoversd_vs_log_N=float(np.polyfit(lg, lb, 1)[0]),
            expected_slopes=dict(log_sd=-0.5, log_biasoversd=+0.5),
        )

    # --- targets
    tgt = [c for k, c in cells.items() if k.startswith("tgt/")]
    if tgt:
        out["targets"] = sorted(
            (
                {
                    k: c[k]
                    for k in (
                        "name",
                        "kind",
                        "resolution",
                        "depth",
                        "N",
                        "G_p98",
                        "eps",
                        "S_min",
                        "S_max",
                        "M10_centre",
                        "M10_outer",
                        "R_c_naive",
                        "sd_naive",
                        "R_c_corr",
                        "sd_corr",
                        "shift_pct",
                        "sigma_naive_convention",
                        "sigma_hypot_convention",
                    )
                }
                for c in tgt
            ),
            key=lambda r: (r["name"], r["depth"], r["resolution"]),
        )
        # The resolution ladder, F5. eps at order 9/10/11/12 built from the SAME sources, plus the
        # all-sky `multi` map the census actually uses. Ratios are deliberately NOT reduced to a
        # single "under-read factor": one of the denominators is 2e-5 and the ratio it produces
        # (~500x) is not commensurable with M11's 2.6x. Both epsilons are recorded; the reader
        # forms whatever comparison is meaningful.
        pair = {}
        for c in tgt:
            pair.setdefault((c["name"], c["depth"]), {})[c["resolution"]] = c
        ladder = []
        for (n, d), v in sorted(pair.items()):
            row = dict(name=n, depth=d)
            for res in ("multi_o10", "cap9", "cap10", "cap11", "cap12"):
                if res in v:
                    row[f"eps_{res}"] = v[res]["eps"]
                    row[f"sigma_{res}"] = v[res]["sigma_naive_convention"]
            if "eps_cap11" in row and "eps_cap12" in row and row["eps_cap11"]:
                row["converged_11_to_12"] = abs(row["eps_cap12"] / row["eps_cap11"] - 1.0)
            if "eps_cap10" in row and "eps_cap11" in row and row["eps_cap10"]:
                row["step_10_to_11"] = abs(row["eps_cap11"] / row["eps_cap10"] - 1.0)
            ladder.append(row)
        out["resolution_ladder"] = ladder

    # --- mutation test on the completeness path
    if "mutation/completeness_node_order" in cells:
        c = cells["mutation/completeness_node_order"]
        out["mutation_completeness_node_order"] = {k: c[k] for k in ("naive", "sd", "variants")}

    # --- census
    if "census/summary" in cells:
        out["census"] = {k: v for k, v in cells["census/summary"].items() if k not in ("cell", "t")}

    # --- analytic expiry curve
    exp_cells = [c for k, c in cells.items() if k.startswith("expiry/")]
    if exp_cells:
        out["expiry_analytic"] = [
            {k: c[k] for k in ("name", "order", "N_now", "sd_now", "R_c_now", "G_p98_now", "rows")}
            for c in exp_cells
        ]

    # --- expiry
    if tgt and "sqrtn" in out:
        alpha = out["sqrtn"]["slope_log_biasoversd_vs_log_N"]
        anchors = []
        for c in tgt:
            if c["kind"] != "OC" or c["resolution"] != "patch_o12" or c["eps"] <= 0:
                continue
            for conv in ("sigma_naive_convention", "sigma_hypot_convention"):
                z = abs(c[conv])
                if z <= 0:
                    continue
                anchors.append(
                    dict(
                        name=c["name"],
                        depth=c["depth"],
                        convention=conv,
                        N_now=c["N"],
                        eps=c["eps"],
                        z_now=z,
                        C=z / (c["eps"] * c["N"] ** 0.5),
                        N_crit=c["N"] * (1.0 / z) ** (1.0 / alpha),
                        eps_crit_at_N_now=c["eps"] / z,
                    )
                )
        out["expiry"] = dict(measured_exponent=alpha, anchors=anchors)

    OUT.write_text(json.dumps(out, indent=1, default=_jsonable))
    print(f"\nwrote {OUT}  ({OUT.stat().st_size / 1024:.1f} KB)")
    return out


# --------------------------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="A7 selection-function null")
    ap.add_argument(
        "--stage",
        default="all",
        choices=[
            "all",
            "flat",
            "injection",
            "sqrtn",
            "targets",
            "census",
            "gmag",
            "expiry",
            "summarise",
        ],
    )
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--census-limit", type=int, default=None)
    args = ap.parse_args()

    cells = load_cells()
    print(f"[resume] {len(cells)} cells on disk")
    if args.stage in ("all", "flat"):
        stage_flat(cells, seeds=3)
        cells = load_cells()
    if args.stage in ("all", "injection"):
        stage_injection(cells, seeds=args.seeds)
        cells = load_cells()
    if args.stage in ("all", "sqrtn"):
        stage_sqrtn(cells, seeds=args.seeds)
        cells = load_cells()
    if args.stage in ("all", "targets"):
        stage_targets(cells, only=args.only)
        cells = load_cells()
    if args.stage in ("all", "expiry"):
        stage_expiry(cells)
        cells = load_cells()
    if args.stage == "gmag":
        fetch_gmag()
        return
    # census runs LAST: it is the least load-bearing stage, because the all-sky map it uses cannot
    # resolve a cluster core. Its role is the magnitude decomposition, not the null itself.
    if args.stage in ("all", "census"):
        stage_census(cells, limit=args.census_limit)
        cells = load_cells()
    summarise(load_cells())


if __name__ == "__main__":
    main()
