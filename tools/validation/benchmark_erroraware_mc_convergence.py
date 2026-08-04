"""Is ``--ea-nmc 100`` enough draws, or is the error-aware gain Monte-Carlo noise?

WHY THIS EXISTS
---------------
``benchmark_erotica_vs_asteca.py``'s ``*_erroraware`` arms run 100 Monte-Carlo error draws
where ASteCA's ``fastmp`` runs 1000. That shortfall was declared as a limitation with an
argument attached -- ``f_i`` is already an average over sweep steps, so the MC error on its
mean should not be the binding uncertainty. **An argument is not a measurement.** This
script measures it.

The measured effect it has to be compared against is the held-out paired gain
``erotica_5d_erroraware - erotica_5d_coarsef``: **+0.1349 +- 0.0206 average precision**.
If re-running the same cells with a different MC seed moves AP by an amount small compared
with that, 100 draws is sufficient for the claim being made. If it moves AP by a comparable
amount, the gain is partly MC noise and the arms must be re-run with more draws.

WHAT WOULD FALSIFY "100 DRAWS IS ENOUGH"
----------------------------------------
The spread of per-cell AP across MC seeds at ``n_mc = 100`` being of the same order as the
effect (i.e. s.d. across seeds > ~0.03 AP), or the ``n_mc = 400`` AP sitting outside the
scatter of the three ``n_mc = 100`` runs. Either would mean the draw budget, not the
physics, is setting the answer.

SCOPE, STATED
-------------
Run on the contamination = 0.95 cells only, and only a few of them. That is deliberate, not
a truncation of convenience: the by-contamination breakdown shows the error-aware gain is
concentrated at c = 0.95 (AP 0.3460 -> 0.5168), so it is the regime where MC noise has the
most room to fabricate the effect. A budget check at c = 0.5, where the arms are identical,
would prove nothing.

USAGE
-----
    python tools/validation/benchmark_erroraware_mc_convergence.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

import benchmark_erotica_vs_asteca as B  # noqa: E402

FEATURES = ("ra", "dec", "pmra", "pmdec", "plx")


def ap_for(real, f_i, score) -> tuple[float, float]:
    y = real.truth.astype(int)
    p = score * f_i
    return roc_auc_score(y, p), average_precision_score(y, p)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=str(Path(__file__).with_name("erroraware_mc_convergence.json"))
    )
    ap.add_argument("--mcs-lo", type=int, default=10)
    ap.add_argument("--mcs-hi", type=int, default=100)
    ap.add_argument("--seeds", default="0,1,2", help="MC random_state values at n_mc=100")
    ap.add_argument("--big-nmc", type=int, default=400)
    args = ap.parse_args(argv)
    warnings.simplefilter("ignore")
    mcs_range = range(args.mcs_lo, args.mcs_hi)
    mc_seeds = [int(v) for v in args.seeds.split(",")]

    # Held-out cells (k in {3,4,5}) at contamination 0.95, the regime the gain lives in.
    cells = [
        {"n_members": N, "contamination": 0.95, "fractal_dimension": D, "seed": s}
        for (N, D, s) in [
            (30, 1.6, 1000 * 0 + 100 * 95 + 0 + 3),
            (30, 3.0, 1000 * 0 + 100 * 95 + 10 + 4),
            (61, 1.6, 1000 * 1 + 100 * 95 + 0 + 3),
            (61, 3.0, 1000 * 1 + 100 * 95 + 10 + 4),
        ]
    ]

    from erotica import Clustering

    t0 = time.perf_counter()
    rows = []
    for spec in cells:
        real = B.generate(
            n_members=spec["n_members"],
            contamination=spec["contamination"],
            fractal_dimension=spec["fractal_dimension"],
            seed=spec["seed"],
        )
        table = B._zscored_columns(real, FEATURES)
        fc = tuple(f"{q}_z" for q in FEATURES)
        # The score factor is fixed across every MC setting -- only f_i changes, which is
        # what makes the comparison a measurement of the draw budget and nothing else.
        clu = Clustering(table[list(fc)])
        try:
            clu.search_pseudoprobability(
                columns=fc,
                min_cluster_size_samples=mcs_range,
                probability_threshold=0.5,
                selection="max_members",
                probability_method="hdbscan",
            )
        except Exception as exc:
            rows.append({**spec, "error": f"{type(exc).__name__}: {exc}"})
            print(f"seed={spec['seed']} SEARCH FAILED: {exc}", file=sys.stderr)
            continue
        score = np.asarray(clu.data["probability_hdbscan"], dtype=float)
        row = {**spec, "n_sources": int(real.truth.size), "runs": []}
        for n_mc, rs in [(100, s) for s in mc_seeds] + [(args.big_nmc, mc_seeds[0])]:
            B.EA_SETTINGS.update(n_mc=n_mc, mcs_step=10, random_state=rs)
            B._F_MC_CACHE.clear()
            t1 = time.perf_counter()
            f_mc, f_sd = B._error_aware_fi(table, fc, mcs_range=mcs_range, mode="mc")
            roc, apv = ap_for(real, f_mc, score)
            row["runs"].append(
                {
                    "n_mc": n_mc,
                    "random_state": rs,
                    "roc": roc,
                    "ap": apv,
                    # MC standard error on each star's f_i, averaged over stars
                    "mean_f_sem": float(np.mean(f_sd) / np.sqrt(n_mc)),
                    "mean_f": float(np.mean(f_mc)),
                    "seconds": time.perf_counter() - t1,
                }
            )
            print(
                f"seed={spec['seed']} n_src={row['n_sources']:<5} n_mc={n_mc:<4} rs={rs} "
                f"ROC={roc:.4f} AP={apv:.4f} mean_f_sem={row['runs'][-1]['mean_f_sem']:.5f} "
                f"({row['runs'][-1]['seconds']:.0f}s)",
                file=sys.stderr,
                flush=True,
            )
        rows.append(row)

    # ---- verdict ---------------------------------------------------------
    spreads, offsets = [], []
    for r in rows:
        if "runs" not in r:
            continue
        base = [x["ap"] for x in r["runs"] if x["n_mc"] == 100]
        big = [x["ap"] for x in r["runs"] if x["n_mc"] == args.big_nmc]
        if len(base) > 1:
            spreads.append(float(np.std(base, ddof=1)))
        if big and base:
            offsets.append(float(big[0] - np.mean(base)))
    verdict = {
        "measured_effect_ap": 0.1349,
        "measured_effect_sem": 0.0206,
        "ap_sd_across_mc_seeds_at_nmc100": {
            "per_cell": spreads,
            "max": max(spreads) if spreads else None,
            "mean": float(np.mean(spreads)) if spreads else None,
        },
        f"ap_shift_nmc{args.big_nmc}_minus_mean_nmc100": {
            "per_cell": offsets,
            "max_abs": float(np.max(np.abs(offsets))) if offsets else None,
        },
    }
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": time.perf_counter() - t0,
        "config": vars(args),
        "cells": rows,
        "verdict": verdict,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(verdict, indent=2), file=sys.stderr)
    print(f"wrote {args.out} ({payload['wall_clock_s']:.0f} s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
