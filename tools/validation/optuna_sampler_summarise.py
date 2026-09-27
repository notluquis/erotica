"""Summarise ``optuna_sampler_runs.jsonl`` into ``optuna_sampler_benchmark.json``.

Implements the statistics fixed in the hub finding ``agent-findings/optuna-samplers-2026-09.md``
§2 before any run: median normalised regret with a 95 % bootstrap interval (10 000 resamples),
hit rate, trials to ``r <= 0.05`` (censored at B), IQR, distinct argmax across seeds, sampler CPU
per trial; Mann-Whitney (two-sided) of every arm against TPE-uni, Holm over all comparisons.

Also the construction control 3: RandomSampler's hit rate against ``1 - (1 - k/N)^B``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

HERE = Path(__file__).parent
RNG = np.random.default_rng(0)
BUDGETS = (50, 100, 600)


def regret(row, b):
    fb = row["best_curve"][b - 1]
    den = row["f_star"] - row["f_med"]
    return max(0.0, (row["f_star"] - fb) / den) if den > 0 else 0.0


def boot_ci(x, n=10_000):
    x = np.asarray(x)
    meds = np.median(RNG.choice(x, size=(n, x.size), replace=True), axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def trials_to(row, b, thr=0.05):
    den = row["f_star"] - row["f_med"]
    for i, fb in enumerate(row["best_curve"][:b]):
        if den <= 0 or (row["f_star"] - fb) / den <= thr:
            return i + 1
    return None  # censored


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj = [0.0] * len(pvals)
    running = 0.0
    m = len(pvals)
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def main():
    rows = [
        json.loads(line)
        for line in (HERE / "optuna_sampler_runs.jsonl").read_text().splitlines()
        if line
    ]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["case"], r["arm"])].append(r)

    cells = []
    for (case, arm), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["seed"])
        for b in BUDGETS:
            if rs[0]["budget"] < b:
                continue
            reg = [regret(r, b) for r in rs]
            hits = sum(1 for r in rs if r["best_curve"][b - 1] >= r["f_star"])
            ttt = [trials_to(r, b) for r in rs]
            reached = [t for t in ttt if t is not None]
            argmaxes = {json.dumps(r["argmax_at"][str(b)], sort_keys=True) for r in rs}
            lo, hi = boot_ci(reg)
            cell = {
                "case": case,
                "arm": arm,
                "B": b,
                "n_seeds": len(rs),
                "median_regret": float(np.median(reg)),
                "ci95": [lo, hi],
                "iqr": [float(np.percentile(reg, 25)), float(np.percentile(reg, 75))],
                "hit_rate": hits / len(rs),
                "hits": hits,
                "trials_to_r05_median": float(np.median(reached))
                if len(reached) * 2 > len(rs)
                else None,
                "reached_r05": len(reached),
                "distinct_argmax": len(argmaxes),
                "cpu_per_trial_median_s": float(np.median([r["cpu_per_trial"] for r in rs])),
                "load_median": float(np.median([r["load"] for r in rs])),
                "_regrets": reg,
            }
            if arm == "RS":
                k, n = rs[0]["k_opt"], rs[0]["n_domain"]
                expected = 1 - (1 - k / n) ** b
                wl, wh = wilson(hits, len(rs))
                cell["control_rs_expected_hit"] = expected
                cell["control_rs_wilson95"] = [wl, wh]
                cell["control_rs_pass"] = wl <= expected <= wh
            cells.append(cell)

    # Mann-Whitney of each arm vs TPE-uni in the same case and budget; Holm over all of them.
    base = {(c["case"], c["B"]): c for c in cells if c["arm"] == "TPE-uni"}
    comps = []
    for c in cells:
        ref = base.get((c["case"], c["B"]))
        if c["arm"] == "TPE-uni" or ref is None:
            continue
        a, r = c["_regrets"], ref["_regrets"]
        if a == r:
            p = 1.0  # identical samples (the 1-D multivariate and static-group controls)
        else:
            p = float(mannwhitneyu(a, r, alternative="two-sided").pvalue)
        comps.append((c, p, float(np.median(a) - np.median(r))))
    adj = holm([p for _, p, _ in comps])
    for (c, p, d), pa in zip(comps, adj, strict=True):
        c["vs_tpe_uni"] = {
            "delta_median_regret": d,
            "p": p,
            "p_holm": pa,
            "identical_to_tpe_uni": c["_regrets"] == base[(c["case"], c["B"])]["_regrets"],
        }
    for c in cells:
        c.pop("_regrets")
    out = {"n_cells": len(cells), "n_comparisons": len(comps), "cells": cells}
    (HERE / "optuna_sampler_benchmark.json").write_text(json.dumps(out, indent=1))

    # Printable table.
    for c in cells:
        v = c.get("vs_tpe_uni", {})
        print(
            f"{c['case']:3} B={c['B']:<3} {c['arm']:10} r={c['median_regret']:.3f} "
            f"[{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}] hit={c['hits']:>2}/{c['n_seeds']} "
            f"t05={c['trials_to_r05_median']} argmax={c['distinct_argmax']:>2} "
            f"cpu={c['cpu_per_trial_median_s'] * 1000:.1f}ms "
            + (f"d={v['delta_median_regret']:+.3f} pH={v['p_holm']:.3g}" if v else "")
            + (
                f" RS-ctrl exp={c['control_rs_expected_hit']:.3f} pass={c['control_rs_pass']}"
                if "control_rs_pass" in c
                else ""
            )
        )


if __name__ == "__main__":
    main()
