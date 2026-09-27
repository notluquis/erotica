"""Which optuna sampler should ``erotica/core/_search.py::_build_sampler`` default to?

Hub finding ``agent-findings/optuna-samplers-2026-09.md`` (the metric, cases and decision rule
were committed there, hub ``f92197f``, BEFORE this script ran). Summary of that protocol:

* Metric: normalised value regret at budget B, ``r = (f* - f_best(B)) / (f* - f_med)``, with
  ``f_med`` the median of ``f`` over the whole domain. Secondary: hit rate (``f_best == f*``),
  trials to ``r <= 0.05``, IQR of ``r`` across seeds, distinct argmax across seeds, sampler CPU per
  trial (``process_time``; the objective here is a table lookup, so the CPU is the sampler's).
* 20 seeds (0..19), ``n_jobs=1``. B = 50 and 100; R4 also 600.
* Synthetic cases S1-S4 with analytic optima; real cases R1-R4 from the exhaustive tables built by
  ``optuna_sampler_tables.py``.

What would falsify "multivariate TPE is better for erotica": on R4 at B=50 the difference
TPE-multi - TPE-uni in ``r`` is not significant after Holm, or favours TPE-uni.

Construction controls (run and reported; if one fails the comparison is not interpreted):

1. in 1-D, TPE-multi and TPE-uni suggest identical sequences;
2. ``group=True`` on a static space suggests the same as ``group=False``;
3. RandomSampler's hit rate matches ``1 - (1 - k/N)^B`` (k optimal points in a domain of N)
   within its 95 % binomial interval.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import optuna

HERE = Path(__file__).parent
SEEDS = range(20)

# ----------------------------------------------------------------------------------------------
# Domains and objectives
# ----------------------------------------------------------------------------------------------


def _ints(low, high, step=1):
    return list(range(low, high + 1, step))


def s1(x):
    f = max(
        math.exp(-((x - 43) ** 2) / (2 * 4**2)), 0.8 * math.exp(-((x - 200) ** 2) / (2 * 50**2))
    )
    return math.floor(f / 0.02 + 1e-9) * 0.02


def s2(x, y, cx=60, cy=40):
    u = ((x - cx) + (y - cy)) / math.sqrt(2)
    v = ((x - cx) - (y - cy)) / math.sqrt(2)
    ridge = math.exp(-(u**2) / (2 * 40**2) - v**2 / (2 * 5**2))
    bump = 0.6 * math.exp(-((x - 160) ** 2 + (y - 160) ** 2) / (2 * 20**2))
    return max(ridge, bump)


def s3(x, y, c):
    return s2(x, y) if c == "leaf" else 0.9 * s2(x, y, 120, 100)


def s4_value(params):
    if params["branch"] == "A":
        return 0.8 * math.exp(-((params["a"] - 100) ** 2) / (2 * 30**2))
    return s2(params["b1"], params["b2"])


class Case:
    """A domain + objective. ``suggest`` asks the trial; ``value`` looks the point up."""

    def __init__(self, name, specs, value_fn, domain_values, conditional=False):
        self.name = name
        self.specs = specs  # list of (name, kind, arg)
        self.value_fn = value_fn
        self.domain_values = np.asarray(domain_values, dtype=float)
        self.conditional = conditional
        self.f_star = float(self.domain_values.max())
        self.f_med = float(np.median(self.domain_values))
        self.k_opt = int(np.count_nonzero(self.domain_values == self.f_star))
        self.n_domain = int(self.domain_values.size)
        self.dim = len(specs)
        self.has_categorical = any(kind == "cat" for _, kind, _ in specs)

    def suggest(self, trial):
        if self.conditional:
            branch = trial.suggest_categorical("branch", ["A", "B"])
            if branch == "A":
                return {"branch": "A", "a": trial.suggest_int("a", 10, 200)}
            return {
                "branch": "B",
                "b1": trial.suggest_int("b1", 10, 200),
                "b2": trial.suggest_int("b2", 10, 200),
            }
        out = {}
        for name, kind, arg in self.specs:
            if kind == "int":
                low, high, step = arg
                out[name] = trial.suggest_int(name, low, high, step=step)
            else:
                out[name] = trial.suggest_categorical(name, arg)
        return out

    def value(self, params):
        return float(self.value_fn(params))


def synthetic_cases():
    d1 = _ints(10, 299)
    d2 = _ints(10, 200)
    cases = {
        "S1": Case("S1", [("x", "int", (10, 299, 1))], lambda p: s1(p["x"]), [s1(x) for x in d1]),
        "S2": Case(
            "S2",
            [("x", "int", (10, 200, 1)), ("y", "int", (10, 200, 1))],
            lambda p: s2(p["x"], p["y"]),
            [s2(x, y) for x in d2 for y in d2],
        ),
        "S3": Case(
            "S3",
            [("x", "int", (10, 200, 1)), ("y", "int", (10, 200, 1)), ("c", "cat", ["eom", "leaf"])],
            lambda p: s3(p["x"], p["y"], p["c"]),
            [s3(x, y, c) for x in d2 for y in d2 for c in ("eom", "leaf")],
        ),
    }
    s4_domain = [s4_value({"branch": "A", "a": a}) for a in d2] + [s2(x, y) for x in d2 for y in d2]
    cases["S4"] = Case(
        "S4",
        [("branch", "cat", ["A", "B"]), ("_", "int", None)],
        s4_value,
        s4_domain,
        conditional=True,
    )
    return cases


def real_cases():
    cases = {}
    for name in ("R1", "R2", "R3", "R4"):
        path = HERE / f"optuna_sampler_tables_{name}.json"
        if not path.exists():
            continue
        tab = json.loads(path.read_text())
        specs = [(p["name"], "int", (p["low"], p["high"], p["step"])) for p in tab["params"]]
        raw = {k: float(v) for k, v in tab["values"].items()}
        finite = [v for v in raw.values() if math.isfinite(v)]
        floor = min(finite)
        # -inf (relative_validity undefined: 0 or 1 cluster) is mapped to the worst finite value so
        # the median and the regret stay finite. Declared in the finding (§5), not in the preregistration.
        lut = {k: (v if math.isfinite(v) else floor) for k, v in raw.items()}
        names = [p["name"] for p in tab["params"]]
        expected = math.prod(len(range(p["low"], p["high"] + 1, p["step"])) for p in tab["params"])
        if len(lut) != expected:
            raise SystemExit(f"{name}: table has {len(lut)} points, domain has {expected}")

        def fn(params, lut=lut, names=names):
            return lut[",".join(str(params[n]) for n in names)]

        cases[name] = Case(name, specs, fn, list(lut.values()))
        cases[name].n_neg_inf = sum(1 for v in raw.values() if not math.isfinite(v))
    return cases


# ----------------------------------------------------------------------------------------------
# Samplers
# ----------------------------------------------------------------------------------------------


def make_sampler(arm, seed):
    s = optuna.samplers
    if arm == "RS":
        return s.RandomSampler(seed=seed)
    if arm == "TPE-uni":
        return s.TPESampler(seed=seed, multivariate=False, constant_liar=False)
    if arm == "TPE-multi":
        return s.TPESampler(seed=seed, multivariate=True, constant_liar=False)
    if arm == "TPE-group":
        return s.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=False)
    if arm == "GP":
        return s.GPSampler(
            seed=seed, deterministic_objective=True, independent_sampler=s.RandomSampler(seed=seed)
        )
    if arm == "CMA":
        return s.CmaEsSampler(seed=seed, independent_sampler=s.RandomSampler(seed=seed))
    raise ValueError(arm)


def arms_for(case):
    arms = ["RS", "TPE-uni", "TPE-multi", "TPE-group", "GP"]
    if case.dim >= 2 and not case.has_categorical and not case.conditional:
        arms.append("CMA")
    return arms


def run_one(case, arm, seed, budget):
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    import warnings

    warnings.filterwarnings("ignore")
    study = optuna.create_study(direction="maximize", sampler=make_sampler(arm, seed))
    values, params_seq = [], []

    def objective(trial):
        p = case.suggest(trial)
        v = case.value(p)
        values.append(v)
        params_seq.append(p)
        return v

    t0 = time.process_time()
    study.optimize(objective, n_trials=budget, n_jobs=1)
    cpu = time.process_time() - t0
    best = np.maximum.accumulate(values)
    return {
        "values": values,
        "best_curve": best.tolist(),
        "params": params_seq,
        "cpu_per_trial": cpu / budget,
    }


# ----------------------------------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------------------------------


def control_sequences(case, budget, arm_a, arm_b, seeds=range(5)):
    same = []
    for seed in seeds:
        a = run_one(case, arm_a, seed, budget)["params"]
        b = run_one(case, arm_b, seed, budget)["params"]
        same.append(a == b)
    return same


def rs_hit_expectation(case, budget):
    return 1.0 - (1.0 - case.k_opt / case.n_domain) ** budget


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--arms", nargs="*")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    all_cases = {**synthetic_cases(), **real_cases()}
    out_path = Path(args.out) if args.out else HERE / "optuna_sampler_runs.jsonl"
    for name in args.cases:
        case = all_cases[name]
        for arm in [a for a in (args.arms or arms_for(case)) if a in arms_for(case)]:
            for seed in range(args.seeds):
                while os.getloadavg()[0] > 40.0:
                    time.sleep(60)
                load = os.getloadavg()[0]
                res = run_one(case, arm, seed, args.budget)
                row = {
                    "case": name,
                    "arm": arm,
                    "seed": seed,
                    "budget": args.budget,
                    "f_star": case.f_star,
                    "f_med": case.f_med,
                    "k_opt": case.k_opt,
                    "n_domain": case.n_domain,
                    "best_curve": res["best_curve"],
                    "argmax_at": {
                        str(b): res["params"][int(np.argmax(res["values"][:b]))]
                        for b in (50, 100, 600)
                        if b <= args.budget
                    },
                    "cpu_per_trial": res["cpu_per_trial"],
                    "load": load,
                    "optuna": optuna.__version__,
                }
                with out_path.open("a") as fh:
                    fh.write(json.dumps(row) + "\n")
            print(name, arm, "done", flush=True)


if __name__ == "__main__":
    main()
