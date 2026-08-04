#!/usr/bin/env python3
r"""The decisive null for A5: inject a FLAT gamma population at census geometry and refit.

WHAT QUESTION THIS SETTLES
--------------------------
The census sweep finds ``gamma`` strongly anti-correlated with ``r_tot/a_fit`` (-0.764 on the
young subsample) and, under a ``r_tot/a_fit >= 16`` selection, a distribution that piles up near
2. Both could be physics. Both could also be pure estimator geometry, because ``a`` and ``gamma``
are correlated at a **median posterior +0.881** (measured, this sweep), so any axis formed by
dividing by ``a_fit`` is correlated with ``gamma`` **by construction**.

The discriminating measurement: inject a population with **no pile-up anywhere** -- ``gamma_true``
uniform on [1.5, 5.0] -- drawn **independently** of the scale radius, at each cluster's real
``N`` and real ``r_tot``, then refit with the identical pipeline and apply the identical gate.

* If the recovered cloud reproduces the observed one -- the negative correlation on the fitted
  axis, a null correlation on the catalogue axis, a gated pile-up near 2 -- then the observed
  structure is **fully explained by the estimator** and no population structure is needed.
* If it does not, the observed structure is a residual worth interpreting.

WHY ``a_true`` IS DRAWN, NOT TAKEN FROM THE CATALOGUE
-----------------------------------------------------
An earlier design set ``a_true = rc`` from Hunt & Reffert. That is wrong: HR24's ``rc`` is an
empirical, contrast-defined radius, not an EFF scale radius, so equating them injects a geometry
the census does not have and measures a mapping invented by the script. ``a_true`` is instead
drawn log-uniformly over ``r_tot/a_true`` in [2, 40] -- the range the pinned-background
recoverability grid covers -- **independently of** ``gamma_true``. Independence is the entire
point: any correlation in the *output* is then unambiguously produced by the estimator.

USAGE
-----
    python tools/validation/a5_injection_null.py --n-clusters 180 --workers 4
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
DEFAULT_OUT = HERE / "a5_injection_null.json"

DRAWS, TUNE, CHAINS = 1500, 1000, 4
GAMMA_LO, GAMMA_HI = 1.5, 5.0
RATIO_LO, RATIO_HI = 2.0, 40.0

_STATE: dict = {}


def eff_radii(rng, n, *, a, gamma, field_radius):
    """Inverse-CDF draw from ``Sigma(r) ~ (1 + (r/a)^2)^(-gamma/2)``, truncated at the field edge.

    Identical construction to ``eff_gamma_bias.eff_radii``, which was validated against the
    analytic EFF CDF to 7e-4 over 400,000 draws. Duplicated rather than imported so this script
    is self-contained and its generator cannot drift from the one that was checked.
    """
    grid = np.linspace(0.0, field_radius, 200_001)
    pdf = 2.0 * np.pi * grid * (1.0 + (grid / a) ** 2) ** (-gamma / 2.0)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


def _init_worker():
    warnings.filterwarnings("ignore")
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    os.environ["XLA_FLAGS"] = (
        "--xla_force_host_platform_device_count=1 "
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1"
    )
    from erotica.analysis.inference import SamplingConfig
    from erotica.analysis.structure import EFFPriors, eff_unbinned

    _STATE["cfg"] = SamplingConfig(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=5,
        progressbar=False,
        nuts_sampler="numpyro",
    )
    _STATE["priors"] = EFFPriors(b_scale=1e-6)  # pinned, exactly as the census sweep
    _STATE["eff"] = eff_unbinned


def fit_one(task):
    import arviz as az

    meta = dict(task)
    t0 = time.time()
    try:
        rng = np.random.default_rng(meta["seed"])
        r = eff_radii(
            rng,
            meta["n"],
            a=meta["a_true"],
            gamma=meta["gamma_true"],
            field_radius=meta["rtot_arcmin"],
        )
        fit = _STATE["eff"](
            r,
            field_radius=meta["rtot_arcmin"],
            priors=_STATE["priors"],
            sampling=_STATE["cfg"],
            progressbar=False,
        )
        idata = fit["eff_trace"]
        post = idata.posterior
        g = np.asarray(post["gamma"].values).ravel()
        a = np.asarray(post["a"].values).ravel()
        nm = list(post.data_vars)
        # method= is keyword-only in effect: az.ess's second POSITIONAL argument is var_names,
        # so az.ess(idata, "bulk") raises "tuple.index(x): x not in tuple". Caught by every one
        # of 192 cells erroring identically -- a uniform failure is a code bug, not data.
        rhat = az.rhat(idata)
        eb, et = az.ess(idata, method="bulk"), az.ess(idata, method="tail")
        meta.update(
            gamma_median=float(np.median(g)),
            gamma_q16=float(np.percentile(g, 16)),
            gamma_q84=float(np.percentile(g, 84)),
            a_median=float(np.median(a)),
            a_sd=float(a.std(ddof=1)),
            rtot_over_a_fit=float(meta["rtot_arcmin"] / np.median(a)),
            corr_a_gamma=float(np.corrcoef(a, g)[0, 1]),
            bias=float(np.median(g) - meta["gamma_true"]),
            eff_rhat_max=float(max(float(np.nanmax(rhat[n].values)) for n in nm)),
            eff_ess_bulk_min=float(min(float(np.nanmin(eb[n].values)) for n in nm)),
            eff_ess_tail_min=float(min(float(np.nanmin(et[n].values)) for n in nm)),
            eff_divergences=int(idata.sample_stats["diverging"].values.sum()),
            status="ok",
        )
    except Exception as exc:  # noqa: BLE001
        meta["status"] = f"error: {type(exc).__name__}: {exc}"
    meta["wall_s"] = time.time() - t0
    return meta


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-clusters", type=int, default=180)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    from astropy.table import Table

    cl = Table.read(CLUSTERS)
    cl = cl[np.asarray(cl["Type"], dtype=str) == "o"]
    N = np.asarray(cl["N"], dtype=float)
    rtot = np.asarray(cl["rtot"], dtype=float) * 60.0
    rc = np.asarray(cl["rc"], dtype=float) * 60.0
    ratio_cat = rtot / rc

    # Stratify over the two axes that matter: sample size and the catalogue's own (fit-independent)
    # footprint-to-core ratio. A simple random draw would under-sample the rich tail, which is
    # where the estimator is supposed to work.
    rng = np.random.default_rng(args.seed)
    nq = np.quantile(N, np.linspace(0, 1, 5))
    rq = np.quantile(ratio_cat[np.isfinite(ratio_cat)], np.linspace(0, 1, 5))
    per_cell = max(1, args.n_clusters // 16)
    idx = []
    for i in range(4):
        for j in range(4):
            sel = np.where(
                (N >= nq[i]) & (N <= nq[i + 1]) & (ratio_cat >= rq[j]) & (ratio_cat <= rq[j + 1])
            )[0]
            if sel.size:
                idx.extend(rng.choice(sel, size=min(per_cell, sel.size), replace=False))
    idx = np.unique(np.array(idx))
    print(f"[design] {idx.size} clusters, stratified 4x4 over (N, r_tot/r_c)")

    tasks = []
    for k, i in enumerate(idx):
        gamma_true = float(rng.uniform(GAMMA_LO, GAMMA_HI))
        # log-uniform in r_tot/a_true, drawn INDEPENDENTLY of gamma_true
        ratio_true = float(np.exp(rng.uniform(np.log(RATIO_LO), np.log(RATIO_HI))))
        tasks.append(
            dict(
                name=str(cl["Name"][i]),
                n=int(N[i]),
                rtot_arcmin=float(rtot[i]),
                rtot_over_rc=float(ratio_cat[i]),
                gamma_true=gamma_true,
                ratio_true=ratio_true,
                a_true=float(rtot[i] / ratio_true),
                seed=int(args.seed + 7919 * k),
            )
        )

    sidecar = args.out.with_suffix(".cells.jsonl")
    done = set()
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["name"])
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"[resume] {len(done)} already done")
    todo = [t for t in tasks if t["name"] not in done]
    print(
        f"[run] {len(todo)} injections, {args.workers} workers, "
        f"gamma_true ~ U({GAMMA_LO}, {GAMMA_HI}) INDEPENDENT of "
        f"r_tot/a_true ~ logU({RATIO_LO}, {RATIO_HI})"
    )

    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    t0 = time.time()
    # Single writer: the parent. Workers return dicts and never open the sidecar.
    with sidecar.open("a") as fh, ctx.Pool(args.workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.imap_unordered(fit_one, todo, chunksize=1), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            if i % 20 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] {(time.time() - t0) / i:.2f}s/fit", flush=True)

    rows = [json.loads(x) for x in sidecar.read_text().splitlines() if x.strip()]
    args.out.write_text(
        json.dumps(
            dict(
                gamma_prior=[GAMMA_LO, GAMMA_HI],
                ratio_prior=[RATIO_LO, RATIO_HI],
                background="pinned (EFFPriors(b_scale=1e-6))",
                n=len(rows),
                cells=rows,
            ),
            indent=1,
        )
    )
    print(f"wrote {args.out} ({len(rows)} cells)")


if __name__ == "__main__":
    main()
