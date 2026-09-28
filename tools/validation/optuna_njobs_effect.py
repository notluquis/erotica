"""Hub finding agent-findings/optuna-samplers-2026-09.md §13.2.

Does n_jobs=-1 change the REPORTED optimum of an erotica optuna search, or only the path?

Same construction as run_optuna_search: `_build_sampler` from erotica origin/dev (so TPE gets
constant_liar = n_jobs != 1), `optuna.create_study(direction="maximize")`, `study.optimize(obj,
n_trials=B, n_jobs=...)`, and the reported optimum is `study.best_trial`.

The objective is a lookup in the exhaustive tables (R1-R3, and R4p = the finished part of the R4
table, min_cluster_size 10..135 x min_samples 10..200, step 5, real data). A real fit takes
1.4-4.8 s and varies with the parameters; here each trial sleeps 10-30 ms, a deterministic
function of its parameters, so that threads interleave with UNEQUAL durations as they would with
real fits, at 1/100 of the time. The sleep releases the GIL, as HDBSCAN's C code largely does.

Per case and seed: n_jobs=1 once (it is deterministic), n_jobs=-1 (8 threads here) REPS times.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import optuna

HERE = Path(__file__).parent
from erotica.core._search import _build_sampler  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.ERROR)
REPS = 5
SEEDS = range(10)


def load(name):
    if name in ("R4p", "R4"):
        # R4p = the part of the R4 table that existed when this was first run (mcs 10..135).
        # Same numbers: the table is deterministic and the checkpoint was this table's prefix.
        raw = json.loads((HERE / "optuna_sampler_tables_R4.json").read_text())["values"]
        hi = 135 if name == "R4p" else 200
        vals = {
            tuple(map(int, k.split(","))): (v if v is not None else -math.inf)
            for k, v in raw.items()
            if int(k.split(",")[0]) <= hi
        }
        space = {
            "min_cluster_size": {"low": 10, "high": hi, "type": "int", "step": 5},
            "min_samples": {"low": 10, "high": 200, "type": "int", "step": 5},
        }
        assert len(vals) == len(range(10, hi + 1, 5)) * 39
        return vals, space, ("min_cluster_size", "min_samples")
    tab = json.loads((HERE / f"optuna_sampler_tables_{name}.json").read_text())
    p = tab["params"][0]
    vals = {(int(k),): (v if v is not None else -math.inf) for k, v in tab["values"].items()}
    space = {"min_cluster_size": {"low": p["low"], "high": p["high"], "type": "int"}}
    return vals, space, ("min_cluster_size",)


def run(vals, space, names, seed, n_jobs, budget, sampler_kwargs):
    sampler = _build_sampler("TPESampler", space, {"seed": seed, **sampler_kwargs}, n_jobs=n_jobs)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def obj(trial):
        key = tuple(
            trial.suggest_int(n, space[n]["low"], space[n]["high"], step=space[n].get("step", 1))
            for n in names
        )
        h = int(hashlib.md5(repr(key).encode()).hexdigest()[:6], 16) / 0xFFFFFF
        time.sleep(0.010 + 0.020 * h)
        return vals[key]

    study.optimize(obj, n_trials=budget, n_jobs=n_jobs)
    bt = study.best_trial
    return {
        "best_value": bt.value,
        "best_params": [bt.params[n] for n in names],
        "n_distinct": len({tuple(t.params[n] for n in names) for t in study.trials}),
    }


def main():
    out = HERE / "optuna_njobs_effect.jsonl"
    nb = {"multivariate": True}  # what the PREPROCESS_PERSISTANCE notebook passes
    cases = [("R1", 50, {}), ("R2", 50, {}), ("R3", 50, {}), ("R4p", 50, nb), ("R4p", 150, nb)]
    cases += [("R4", 50, nb), ("R4", 150, nb), ("R4", 600, nb)]
    if len(sys.argv) > 1:
        cases = [c for c in cases if f"{c[0]}:{c[1]}" in sys.argv[1:]]
    for name, budget, kw in cases:
        vals, space, names = load(name)
        fstar = max(v for v in vals.values() if math.isfinite(v))
        for seed in SEEDS:
            rows = [("1", run(vals, space, names, seed, 1, budget, kw))]
            rows += [("-1", run(vals, space, names, seed, -1, budget, kw)) for _ in range(REPS)]
            with out.open("a") as fh:
                for nj, r in rows:
                    fh.write(
                        json.dumps(
                            {
                                "case": name,
                                "B": budget,
                                "seed": seed,
                                "n_jobs": nj,
                                "f_star": fstar,
                                **r,
                            }
                        )
                        + "\n"
                    )
        print(name, budget, "done", flush=True)


def summarise():
    import collections

    rows = [
        json.loads(line)
        for line in (HERE / "optuna_njobs_effect.jsonl").read_text().splitlines()
        if line
    ]
    cells = []
    for case, b in sorted({(r["case"], r["B"]) for r in rows}):
        rs = [r for r in rows if r["case"] == case and r["B"] == b]
        one = {r["seed"]: r for r in rs if r["n_jobs"] == "1"}
        par = [r for r in rs if r["n_jobs"] == "-1"]
        per_seed = [
            len({tuple(r["best_params"]) for r in par if r["seed"] == s}) for s in sorted(one)
        ]
        cells.append(
            {
                "case": case,
                "B": b,
                "hits_n_jobs_1": sum(r["best_value"] >= r["f_star"] for r in one.values()),
                "n_n_jobs_1": len(one),
                "hits_n_jobs_minus1": sum(r["best_value"] >= r["f_star"] for r in par),
                "n_n_jobs_minus1": len(par),
                "distinct_reported_optima_per_seed_over_reps": per_seed,
                "reps_reporting_other_params_than_serial": sum(
                    tuple(r["best_params"]) != tuple(one[r["seed"]]["best_params"]) for r in par
                ),
                "reported_params_n_jobs_minus1": collections.Counter(
                    json.dumps(r["best_params"]) for r in par
                ).most_common(5),
                "reported_params_n_jobs_1": collections.Counter(
                    json.dumps(r["best_params"]) for r in one.values()
                ).most_common(5),
            }
        )
    (HERE / "optuna_njobs_effect.json").write_text(
        json.dumps({"reps": REPS, "cells": cells}, indent=1, allow_nan=False)
    )
    for c in cells:
        print(
            c["case"],
            c["B"],
            c["hits_n_jobs_1"],
            "/",
            c["n_n_jobs_1"],
            c["hits_n_jobs_minus1"],
            "/",
            c["n_n_jobs_minus1"],
            c["distinct_reported_optima_per_seed_over_reps"],
            c["reps_reporting_other_params_than_serial"],
            c["reported_params_n_jobs_minus1"][:3],
        )


if __name__ == "__main__":
    if sys.argv[1:] == ["--summarise"]:
        summarise()
    else:
        main()
