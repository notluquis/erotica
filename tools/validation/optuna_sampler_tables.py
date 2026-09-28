"""Exhaustive objective tables for the optuna sampler benchmark (hub finding
``agent-findings/optuna-samplers-2026-09.md``).

Why this exists. optuna 5.0 made ``TPESampler(multivariate=True)`` the default, and the choice of
sampler in ``erotica/core/_search.py::_build_sampler`` has to be decided against a KNOWN optimum.
On real data the only honest optimum is the exhaustive one: every point of the domain evaluated
with erotica's own objective. These tables are that oracle; ``optuna_sampler_benchmark.py`` then
runs every sampler against them, so the number of seeds costs nothing.

Cases (all on the 40 arcmin NGC 6383 catalogue, 15 276 preprocessed sources, ``pmra``/``pmdec``):

* ``R1``: the P01 membership recipe (``validation/ngc6383_radius_robustness.py`` in the paper
  repository), 290-step sweep. Objective per step = ``desired_len`` when the step survives the
  sweep's filters, 0 otherwise. **Control:** the argmax must be ``min_cluster_size=43`` with
  ``desired_len=701``, the published run. If not, the table is not the published objective.
* ``R2``: the same sweep with the current defaults, objective ``selected_persistence``.
* ``R3``: ``Clustering.search`` default space (``min_cluster_size`` 5..100), objective
  ``relative_validity`` via ``_search._score_relative_validity``.
* ``R4``: the 2-D space of ``PREPROCESS_PERSISTANCE.ipynb`` (``min_cluster_size`` and
  ``min_samples`` 10..200) with its hdbscan kwargs, on a ``step=5`` lattice (39 x 39). The full
  191 x 191 lattice measured ~2.5 s per fit, ~12 h on the two cores this run may use; the benchmark
  suggests on the SAME lattice, so the oracle and the samplers see one domain.

What would falsify the benchmark built on these tables: R1 not reproducing 43/701.

Resolution: integer lattices exactly as stated; nothing is interpolated.
``core_dist_n_jobs=1`` everywhere (it does not change labels, only threads).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CATALOGUE = (
    Path(
        os.environ.get(
            "EROTICA_P01_RADIUS_DIR",
            "/Users/notluquis/erotica/data/test/NGC6383/comments_paper/radius_robustness/generated/40",
        )
    )
    / "paperfaithful_with_clip_flags.ecsv"
)
OUT = Path(__file__).with_name("optuna_sampler_tables.json")
MAX_LOAD = 40.0


def _wait_for_load():
    while os.getloadavg()[0] > MAX_LOAD:
        time.sleep(60)


def _data():
    from astropy.table import QTable

    return QTable.read(CATALOGUE)


def _sweep(table, **kwargs):
    from erotica.core import Clustering

    clu = Clustering(table.copy(), None)
    clu.search_pseudoprobability(
        columns=["pmra", "pmdec"], min_cluster_size_samples=range(10, 300), **kwargs
    )
    by_mcs = {int(r["min_cluster_size"]): r for r in clu.pseudoprobability_results_}
    sel = clu.pseudoprobability_selected_
    return by_mcs, sel


def table_r1(table):
    by_mcs, sel = _sweep(
        table,
        probability_threshold=0.5,
        min_cluster_members=200,
        max_cluster_members=1000,
        select_cluster=False,
        selection="max_members",
        approx_min_span_tree=True,
        match_reference_implementation=False,
        hdbscan_kwargs={
            "cluster_selection_method": "leaf",
            "allow_single_cluster": True,
            "core_dist_n_jobs": 1,
        },
    )
    values = {
        str(m): float(by_mcs[m]["desired_len"]) if m in by_mcs else 0.0 for m in range(10, 300)
    }
    return {
        "params": [{"name": "min_cluster_size", "low": 10, "high": 299, "step": 1}],
        "values": values,
        "selected_by_erotica": {
            "min_cluster_size": int(sel["min_cluster_size"]),
            "desired_len": int(sel["desired_len"]),
        },
        "n_eligible": len(by_mcs),
    }


def table_r2(table):
    by_mcs, sel = _sweep(table, hdbscan_kwargs={"core_dist_n_jobs": 1})
    values = {
        str(m): float(by_mcs[m]["selected_persistence"]) if m in by_mcs else 0.0
        for m in range(10, 300)
    }
    return {
        "params": [{"name": "min_cluster_size", "low": 10, "high": 299, "step": 1}],
        "values": values,
        "selected_by_erotica": {
            "min_cluster_size": int(sel["min_cluster_size"]),
            "selected_persistence": float(sel["selected_persistence"]),
        },
        "n_eligible": len(by_mcs),
    }


def _rv(X, mcs, ms, kwargs):
    from erotica.core._estimator import HDBSCANEstimator
    from erotica.core._search import _score_relative_validity

    est = HDBSCANEstimator(
        min_cluster_size=mcs, min_samples=ms, persistence_threshold=0.0, **kwargs
    ).fit(X)
    return _score_relative_validity(est, X, SimpleNamespace(number=-1))


def table_r3(table):
    X = np.column_stack([np.asarray(table["pmra"], float), np.asarray(table["pmdec"], float)])
    values = {}
    for m in range(5, 101):
        _wait_for_load()
        values[str(m)] = _rv(X, m, None, {"core_dist_n_jobs": 1})
    return {
        "params": [{"name": "min_cluster_size", "low": 5, "high": 100, "step": 1}],
        "values": values,
    }


R4_KW = {
    "cluster_selection_method": "eom",
    "allow_single_cluster": False,
    "max_cluster_size": 1000,
    "core_dist_n_jobs": 1,
}


def table_r4(table, partial: Path):
    X = np.column_stack([np.asarray(table["pmra"], float), np.asarray(table["pmdec"], float)])
    done = json.loads(partial.read_text()) if partial.exists() else {}
    grid = range(10, 201, 5)
    for mcs in grid:
        for ms in grid:
            key = f"{mcs},{ms}"
            if key in done:
                continue
            _wait_for_load()
            done[key] = _rv(X, mcs, ms, R4_KW)
        partial.write_text(json.dumps(done))
    return {
        "params": [
            {"name": "min_cluster_size", "low": 10, "high": 200, "step": 5},
            {"name": "min_samples", "low": 10, "high": 200, "step": 5},
        ],
        "values": done,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+", choices=["R1", "R2", "R3", "R4"])
    args = ap.parse_args()
    table = _data()
    for case in args.cases:
        t0 = time.perf_counter()
        if case == "R4":
            res = table_r4(table, OUT.with_name("optuna_sampler_tables_R4.partial.json"))
        else:
            res = {"R1": table_r1, "R2": table_r2, "R3": table_r3}[case](table)
        # -inf (relative_validity undefined: 0 or 1 cluster) is written as null, never as the bare
        # -Infinity token json.dumps emits, which is not JSON (tools/check_json_strict.py).
        res["values"] = {k: (v if math.isfinite(v) else None) for k, v in res["values"].items()}
        res["null_means"] = "objective undefined for that point (-inf in _score_relative_validity)"
        res["seconds"] = round(time.perf_counter() - t0, 1)
        res["load_at_end"] = os.getloadavg()[0]
        res["n_sources"] = len(table)
        path = OUT.with_name(f"optuna_sampler_tables_{case}.json")
        path.write_text(json.dumps(res, indent=1, allow_nan=False))
        print(case, "done in", res["seconds"], "s ->", path.name, flush=True)


if __name__ == "__main__":
    main()
