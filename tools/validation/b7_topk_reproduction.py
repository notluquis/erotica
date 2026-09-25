"""B7 (~/phd open-threads.md fila B7, agent-findings/b7-branch-selection.md): reproduce el
8/12 citado en la fila via el operating point TOP-K COMPARTIDO -- no la mascara a umbral 0.5
de `search_pseudoprobability`, que sale vacia para estas semillas -- y mide el candidato
`recovery_frequency="target"` sobre las mismas 12 semillas del brazo B' de B6 y sobre el
control (brazo A, sin vecino).

K se toma del propio `b6_neighbour_arms.json` (`recall * n_members / purity`, redondeado), que
el harness ya demostro constante entre los 16 metodos dentro de una celda -- no hace falta
correr ASteCA (1000 corridas) para reproducir la cifra citada en B7.

Uso::

    python3 b7_topk_reproduction.py [ARM ...]   # default: Bp A

Entorno: erotica-bench (hdbscan==0.8.44, el que fija pyproject.toml). El env `miniforge3` base
trae hdbscan 0.8.42, mas viejo que el pin -- no se usa aca.
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
from benchmark_erotica_vs_asteca import (  # noqa: E402
    PMDEC_C,
    PMRA_C,
    _zscored_columns,
    generate,
    recover_params,
)

from erotica import Clustering as PublicClustering  # noqa: E402
from erotica.core.clustering import NoCandidateClusters  # noqa: E402

SIGMA_PM_INT = 0.20
ARMS = {
    "A": dict(neighbour_fraction=0.0, delta_pm=1.25, delta_plx=0.8),
    "Bp": dict(neighbour_fraction=0.175, delta_pm=12.0, delta_plx=0.8),
}
N_MEMBERS = 61
CONTAMINATION = 0.8
FRACTAL_DIM = 1.6
MCS_RANGE = range(10, 100)
SEEDS = list(range(8000, 8012))

JSON_PATH = _HERE / "b6_neighbour_arms.json"
OUT_PATH = _HERE / "b7_topk_results.json"


def neighbour_centroid(delta_pm: float, delta_plx: float) -> tuple[float, float]:
    ang = math.radians(45.0)
    return (
        PMRA_C + delta_pm * SIGMA_PM_INT * math.cos(ang),
        PMDEC_C + delta_pm * SIGMA_PM_INT * math.sin(ang),
    )


def load_json_cells() -> dict:
    d = json.loads(JSON_PATH.read_text())
    return {(c["arm"], c["seed"]): c for c in d["cells"]}


def population_frac(mask: np.ndarray, truth: np.ndarray, neighbour: np.ndarray) -> dict:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "primary": 0.0, "neighbour": 0.0, "field": 0.0}
    return {
        "n": n,
        "primary": float(truth[mask].mean()),
        "neighbour": float(neighbour[mask].mean()),
        "field": float((~truth[mask] & ~neighbour[mask]).mean()),
    }


def run_one(arm: str, seed: int, json_cells: dict, *, test_target_aware: bool) -> dict:
    params = ARMS[arm]
    real = generate(
        n_members=N_MEMBERS,
        contamination=CONTAMINATION,
        fractal_dimension=FRACTAL_DIM,
        seed=seed,
        **params,
    )
    quantities = ("ra", "dec", "pmra", "pmdec", "plx")
    table = _zscored_columns(real, quantities)
    feature_cols = tuple(f"{q}_z" for q in quantities)

    cell = json_cells.get((arm, seed))
    m = cell["methods"]["erotica_5d"] if cell else None
    k_json = None
    if m and m["purity"]:
        k_json = int(round(m["recall"] * real.n_members / m["purity"]))

    out: dict = {"arm": arm, "seed": seed, "k_json": k_json}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clu = PublicClustering(table[list(feature_cols)])
            clu.search_pseudoprobability(
                columns=feature_cols,
                min_cluster_size_samples=MCS_RANGE,
                probability_threshold=0.5,
                selection="max_members",
                probability_method="hdbscan",
            )
    except NoCandidateClusters as exc:
        out["error"] = f"NoCandidateClusters: {exc}"
        return out
    sel = clu.pseudoprobability_selected_
    labels = np.asarray(clu.data["cluster_hdbscan"], dtype=int)
    score = np.asarray(clu.data["probability"], dtype=float)  # default (any-cluster) p-tilde
    sel_label = int(sel["selected_cluster"])

    out["selected_label_population"] = population_frac(
        labels == sel_label, real.truth, real.neighbour
    )

    if k_json is not None:
        order = np.argsort(-score, kind="stable")
        mk = np.zeros(score.size, bool)
        mk[order[: min(k_json, score.size)]] = True
        rec = recover_params(real, mk)
        out["topk_default"] = {
            "K": k_json,
            "pmra_c": rec["pmra_c"],
            "pmdec_c": rec["pmdec_c"],
            "purity": float(real.truth[mk].mean()),
            "recall": float(real.truth[mk].sum() / real.n_members),
            "population": population_frac(mk, real.truth, real.neighbour),
        }
        if m:
            out["topk_default"]["json_pmra_c"] = m["pmra_c"]
            out["topk_default"]["json_pmdec_c"] = m["pmdec_c"]
            out["topk_default"]["match"] = bool(
                abs(rec["pmra_c"] - m["pmra_c"]) < 1e-4
                and abs(rec["pmdec_c"] - m["pmdec_c"]) < 1e-4
            )
        pmra_n_c, pmdec_n_c = neighbour_centroid(params["delta_pm"], params["delta_plx"])
        dp = math.hypot(rec["pmra_c"] - PMRA_C, rec["pmdec_c"] - PMDEC_C)
        dn = math.hypot(rec["pmra_c"] - pmra_n_c, rec["pmdec_c"] - pmdec_n_c)
        out["topk_default"]["engages_neighbour"] = (
            bool(dn < dp) if params["neighbour_fraction"] else None
        )

    if test_target_aware and k_json is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clu_t = PublicClustering(table[list(feature_cols)])
                clu_t.search_pseudoprobability(
                    columns=feature_cols,
                    min_cluster_size_samples=MCS_RANGE,
                    probability_threshold=0.5,
                    selection="max_members",
                    probability_method="hdbscan",
                    recovery_frequency="target",
                )
        except NoCandidateClusters as exc:
            out["topk_target_aware_error"] = f"NoCandidateClusters: {exc}"
            return out
        score_t = np.asarray(clu_t.data["probability"], dtype=float)
        order_t = np.argsort(-score_t, kind="stable")
        mk_t = np.zeros(score_t.size, bool)
        mk_t[order_t[: min(k_json, score_t.size)]] = True
        rec_t = recover_params(real, mk_t)
        out["topk_target_aware"] = {
            "K": k_json,
            "pmra_c": rec_t["pmra_c"],
            "pmdec_c": rec_t["pmdec_c"],
            "purity": float(real.truth[mk_t].mean()),
            "recall": float(real.truth[mk_t].sum() / real.n_members),
            "population": population_frac(mk_t, real.truth, real.neighbour),
        }
        pmra_n_c, pmdec_n_c = neighbour_centroid(params["delta_pm"], params["delta_plx"])
        dp = math.hypot(rec_t["pmra_c"] - PMRA_C, rec_t["pmdec_c"] - PMDEC_C)
        dn = math.hypot(rec_t["pmra_c"] - pmra_n_c, rec_t["pmdec_c"] - pmdec_n_c)
        out["topk_target_aware"]["engages_neighbour"] = (
            bool(dn < dp) if params["neighbour_fraction"] else None
        )

    return out


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    json_cells = load_json_cells()
    arms_to_run = argv or ["Bp", "A"]
    all_out = []
    for arm in arms_to_run:
        for seed in SEEDS:
            print(f"=== arm {arm} seed {seed} ===", file=sys.stderr)
            r = run_one(arm, seed, json_cells, test_target_aware=True)
            all_out.append(r)
            print(json.dumps(r, indent=2), flush=True)
    OUT_PATH.write_text(json.dumps(all_out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
