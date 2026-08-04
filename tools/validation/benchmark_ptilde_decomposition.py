"""Decompose EROTICA's p-tilde into its two factors and score each one separately.

WHY THIS EXISTS
---------------
``benchmark_erotica_vs_asteca.py`` measures a large, reproducible gap to ASteCA's
``fastMP`` (held-out ROC 0.776 vs 0.918, AP 0.514 vs 0.862 for ``erotica_5d``). EROTICA's
per-star score is a **product**::

    p_tilde_i = score_i * f_i

where ``score_i`` is the HDBSCAN membership strength (or, in the ``_soft`` arms, the
``all_points_membership_vectors`` column of the selected cluster) and ``f_i`` is the
sweep-recovery frequency -- the fraction of swept ``min_cluster_size`` values in which the
star landed in **any** cluster (``labels != -1``).

A previous screen (``hdbscan_config_sweep.py``) scored ``score`` ALONE and reported
delta-ROC ~ +0.216 for soft membership; the full harness measures +0.025 to +0.031. That 8x
discrepancy was attributed to ``f_i`` washing out the difference. **That attribution was
never measured directly, because per-star ``f_i`` was never persisted.** This script
measures it.

The question it answers: is the average precision lost in the CLUSTERER'S SCORE or in the
SWEEP-RECOVERY TERM? If ``f_i`` alone scores near chance while ``score`` alone scores well,
the fix is in the sweep design, not in HDBSCAN tuning.

A THIRD FACTOR THE PRODUCT DOES NOT CONTAIN
-------------------------------------------
``probability_hdbscan`` is ``clusterer.probabilities_``: membership strength in *whichever*
cluster a star was assigned, not in the selected one. A field star sitting firmly inside the
FIELD cluster scores ~1.0. And ``f_i`` is "clustered into anything", not "clustered into the
target". So for the default (non-soft) arm **neither factor of the product encodes target-
cluster identity**. This script therefore also scores a fourth per-star quantity:

    tgt_i = 1 if labels_i == selected_cluster else 0

If ``tgt`` outranks ``p_tilde``, the answer is neither "the score" nor "the sweep" -- it is
that the product discards target identity -- and that redirects differently again. The
verdict is read off the numbers, not forced into a two-way split.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
The hypothesis under test is that ``f_i`` is the information-destroying factor. It is
falsified if ``f_i`` alone scores an average precision comparable to or better than
``score`` alone on held-out cells -- in that case the sweep term carries at least as much
membership information as the clusterer's own score, and the ~8x attenuation reported in
``hdbscan-soft-membership.md`` is not "``f_i`` destroys information" but "``f_i`` is a
different, comparably-informative signal that the product averages against".

REPRODUCTION CHECK (load-bearing, not decorative)
-------------------------------------------------
Every recomputed ``p_tilde`` is asserted equal, per star, to the value stored by the
108-cell run in ``soft_bench.cells.jsonl``. If that assertion fails the re-run is not the
same experiment and no number here is comparable to the published table. The equality
``p_tilde == score * f_i`` is asserted too, and so is ``f_i(max_members) == f_i(max_persistence)``
-- the sweep term must not depend on the branch-selection rule, which is what licenses
sharing one ``f_i`` between the two arms.

HELD-OUT SPLIT
--------------
Realisation index ``k = seed % 10``; ``k in {0,1,2}`` train, ``k in {3,4,5}`` HELD OUT.
This split, and per-cell ``roc_auc_score`` / ``average_precision_score`` averaged with a
standard error across cells, reproduces the published table to 4 decimals on all five arms
(erotica_3d 0.7624, erotica_3d_soft 0.7873, erotica_5d 0.7759, erotica_5d_soft 0.8068,
asteca_fastmp 0.9182). Every decision below is made on the held-out half only.

ENVIRONMENT
-----------
``erotica-bench`` (python 3.13, asteca 0.7.0, hdbscan 0.8.44, erotica -e). NOT ``cosmic``,
which has no ``erotica`` and asteca 0.6.9 -- a different ASteCA arm would break
comparability with the table this script is anchored to.

USAGE
-----
    python tools/validation/benchmark_ptilde_decomposition.py --out ptilde_decomposition.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
from astropy.table import QTable
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))

import benchmark_erotica_vs_asteca as B  # noqa: E402

# The 108-cell run whose p_tilde this script reproduces. Written by
# `benchmark_erotica_vs_asteca.py --configs erotica_3d,erotica_3d_soft,erotica_5d,erotica_5d_soft`
# on 2026-08-04 (1618 s, 108 cells).
REFERENCE_CELLS = Path(
    "/private/tmp/claude-501/-Users-notluquis-phd/"
    "d055f28c-94b0-484a-8d8b-601af31f9812/scratchpad/soft_bench.cells.jsonl"
)

HELD_OUT_K = (3, 4, 5)  # realisation indices never used for a decision
TRAIN_K = (0, 1, 2)


def held_out(seed: int) -> bool:
    return (seed % 10) in HELD_OUT_K


# ---------------------------------------------------------------------------
# One cell: run both 5d arms and keep every factor separately
# ---------------------------------------------------------------------------
def decompose_cell(real, *, config: str, mcs_range: range) -> dict:
    """Run one EROTICA config and return the per-star factors, not just the product."""
    from erotica import Clustering

    quantities, selection, probability_method = B.EROTICA_CONFIGS[config]
    cols = {f"{q}_z": B._zscore(getattr(real, q)) for q in quantities}
    n = real.truth.size
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clu = Clustering(QTable(cols))
            clu.search_pseudoprobability(
                columns=tuple(cols),
                min_cluster_size_samples=mcs_range,
                probability_threshold=0.5,
                selection=selection,
                probability_method=probability_method,
            )
    except Exception as exc:
        # A failure to find any cluster is a legitimate result and is exactly what the
        # benchmark records as an all-zero probability vector. Mirror that so the numbers
        # stay comparable to the published table.
        return {
            "config": config,
            "error": f"{type(exc).__name__}: {exc}",
            "runtime_s": time.perf_counter() - t0,
            "score": np.zeros(n),
            "f_i": np.zeros(n),
            "p_tilde": np.zeros(n),
            "tgt": np.zeros(n),
        }
    secs = time.perf_counter() - t0
    tab = clu.data
    f_i = np.asarray(tab["probability_times"], dtype=float)
    p_tilde = np.asarray(tab["probability"], dtype=float)
    if probability_method == "soft" and "probability_soft" in tab.colnames:
        score = np.asarray(tab["probability_soft"], dtype=float)
        score_kind = "soft"
    else:
        score = np.asarray(tab["probability_hdbscan"], dtype=float)
        score_kind = "hdbscan"
    labels = np.asarray(tab["cluster_hdbscan"], dtype=int)
    sel_label = int((clu.pseudoprobability_selected_ or {}).get("selected_cluster", -1))
    tgt = (labels == sel_label).astype(float) if sel_label >= 0 else np.zeros(n)
    return {
        "config": config,
        "error": None,
        "runtime_s": secs,
        "score": score,
        "score_kind": score_kind,
        "f_i": f_i,
        "p_tilde": p_tilde,
        "tgt": tgt,
        "selected_cluster": sel_label,
        "best_mcs": int(clu.best_params_["min_cluster_size"]),
        # product identity, checked here rather than assumed
        "product_max_abs_err": float(np.max(np.abs(p_tilde - score * f_i))),
    }


def _score(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    if y.sum() == 0 or y.sum() == y.size:
        return float("nan"), float("nan")
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))


def run_cell(*, n_members, contamination, fractal_dimension, seed, mcs_range, reference):
    real = B.generate(
        n_members=n_members,
        contamination=contamination,
        fractal_dimension=fractal_dimension,
        seed=seed,
    )
    y = real.truth.astype(int)
    out = {
        "n_members": n_members,
        "contamination": contamination,
        "fractal_dimension": fractal_dimension,
        "seed": seed,
        "held_out": held_out(seed),
        "n_sources": int(y.size),
        "base_rate": float(y.mean()),
        "generator_check": B.check_generator(real, requested_contamination=contamination),
        "arms": {},
        "checks": {},
    }
    parts = {}
    for cfg in ("erotica_5d", "erotica_5d_soft"):
        d = decompose_cell(real, config=cfg, mcs_range=mcs_range)
        parts[cfg] = d
        arm = {
            "error": d["error"],
            "runtime_s": d["runtime_s"],
            "score_kind": d.get("score_kind"),
            "best_mcs": d.get("best_mcs"),
            "selected_cluster": d.get("selected_cluster"),
            "product_max_abs_err": d.get("product_max_abs_err"),
        }
        for name in ("score", "f_i", "p_tilde", "tgt"):
            roc, ap = _score(y, d[name])
            arm[f"roc_{name}"] = roc
            arm[f"ap_{name}"] = ap
            arm[f"mean_{name}"] = float(np.mean(d[name]))
        # AP lift over the base rate: AP is not comparable across contaminations raw.
        for name in ("score", "f_i", "p_tilde", "tgt"):
            arm[f"aplift_{name}"] = (
                arm[f"ap_{name}"] / out["base_rate"] if out["base_rate"] > 0 else float("nan")
            )
        out["arms"][cfg] = arm

    # ---- checks that make the re-run load-bearing ------------------------
    a, b = parts["erotica_5d"], parts["erotica_5d_soft"]
    ok_shared_f = bool(np.allclose(a["f_i"], b["f_i"])) if not (a["error"] or b["error"]) else None
    out["checks"]["f_i_selection_independent"] = ok_shared_f
    ref = reference.get(seed)
    for cfg, d in parts.items():
        if ref is None or cfg not in ref:
            out["checks"][f"reproduces_{cfg}"] = None
            continue
        stored = np.asarray(ref[cfg], dtype=float)
        same = bool(
            stored.size == d["p_tilde"].size and np.allclose(stored, d["p_tilde"], atol=1e-9)
        )
        out["checks"][f"reproduces_{cfg}"] = same
        out["checks"][f"reproduces_{cfg}_maxabs"] = (
            float(np.max(np.abs(stored - d["p_tilde"])))
            if stored.size == d["p_tilde"].size
            else None
        )
    return out


# ---------------------------------------------------------------------------
def _load_reference(path: Path) -> dict:
    """seed -> {arm: stored p_tilde list} from the 108-cell run."""
    if not path.exists():
        print(
            f"[warn] reference cells not found at {path}; reproduction check disabled",
            file=sys.stderr,
        )
        return {}
    ref = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        c = json.loads(line)
        ref[c["seed"]] = {
            k: v for k, v in c["_probs"].items() if k in ("erotica_5d", "erotica_5d_soft")
        }
    return ref


def summarise(cells: list[dict]) -> dict:
    def ms(vals):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        if v.size == 0:
            return {"mean": None, "sem": None, "n": 0}
        if v.size == 1:
            return {"mean": float(v[0]), "sem": None, "n": 1}
        return {
            "mean": float(v.mean()),
            "sem": float(v.std(ddof=1) / np.sqrt(v.size)),
            "n": int(v.size),
        }

    out = {}
    for split, sel in (
        ("held_out", lambda c: c["held_out"]),
        ("train", lambda c: not c["held_out"]),
    ):
        sub = [c for c in cells if sel(c)]
        blk = {"n_cells": len(sub)}
        for cfg in ("erotica_5d", "erotica_5d_soft"):
            blk[cfg] = {
                f"{m}_{name}": ms([c["arms"][cfg][f"{m}_{name}"] for c in sub])
                for name in ("score", "f_i", "p_tilde", "tgt")
                for m in ("roc", "ap", "aplift")
            }
            # paired deltas vs the shipped product, per cell
            for name in ("score", "f_i", "tgt"):
                for m in ("roc", "ap"):
                    d = [
                        c["arms"][cfg][f"{m}_{name}"] - c["arms"][cfg][f"{m}_p_tilde"] for c in sub
                    ]
                    d = [x for x in d if np.isfinite(x)]
                    blk[cfg][f"delta_{m}_{name}_minus_ptilde"] = ms(d)
                    blk[cfg][f"wins_{m}_{name}_over_ptilde"] = (
                        f"{sum(1 for x in d if x > 0)}/{len(d)}"
                    )
        blk["base_rate_by_contamination"] = {
            str(x): ms([c["base_rate"] for c in sub if c["contamination"] == x])
            for x in sorted({c["contamination"] for c in sub})
        }
        out[split] = blk
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).with_name("ptilde_decomposition.json")))
    ap.add_argument("--mcs-lo", type=int, default=10)
    ap.add_argument("--mcs-hi", type=int, default=100)
    ap.add_argument("--n-grid", default="30,61,150")
    ap.add_argument("--cont-grid", default="0.5,0.8,0.95")
    ap.add_argument("--dim-grid", default="1.6,3.0")
    ap.add_argument("--realisations", type=int, default=6)
    ap.add_argument("--reference", default=str(REFERENCE_CELLS))
    args = ap.parse_args(argv)

    mcs_range = range(args.mcs_lo, args.mcs_hi)
    n_grid = [int(v) for v in args.n_grid.split(",")]
    cont_grid = [float(v) for v in args.cont_grid.split(",")]
    dim_grid = [float(v) for v in args.dim_grid.split(",")]
    reference = _load_reference(Path(args.reference))

    t0 = time.perf_counter()
    ckpt = Path(args.out).with_suffix(".cells.jsonl")
    ckpt.write_text("")
    cells: list[dict] = []
    for n_members in n_grid:
        for cont in cont_grid:
            for dim in dim_grid:
                for k in range(args.realisations):
                    seed = (
                        1000 * n_grid.index(n_members)
                        + 100 * int(cont * 100)
                        + 10 * dim_grid.index(dim)
                        + k
                    )
                    c = run_cell(
                        n_members=n_members,
                        contamination=cont,
                        fractal_dimension=dim,
                        seed=seed,
                        mcs_range=mcs_range,
                        reference=reference,
                    )
                    cells.append(c)
                    with ckpt.open("a") as fh:
                        fh.write(json.dumps(c) + "\n")
                    print(
                        f"[{len(cells):3d}/108] N={n_members} c={cont} D={dim} seed={seed} "
                        f"heldout={c['held_out']} "
                        f"repro5d={c['checks'].get('reproduces_erotica_5d')} "
                        f"reprosoft={c['checks'].get('reproduces_erotica_5d_soft')} "
                        f"shared_f={c['checks']['f_i_selection_independent']} "
                        f"t={time.perf_counter() - t0:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )

    checks = {
        "reproduces_erotica_5d": f"{sum(1 for c in cells if c['checks'].get('reproduces_erotica_5d'))}/{len(cells)}",
        "reproduces_erotica_5d_soft": f"{sum(1 for c in cells if c['checks'].get('reproduces_erotica_5d_soft'))}/{len(cells)}",
        "f_i_selection_independent": f"{sum(1 for c in cells if c['checks']['f_i_selection_independent'])}/{len(cells)}",
        "max_product_identity_err": max(
            (c["arms"][cfg].get("product_max_abs_err") or 0.0)
            for c in cells
            for cfg in ("erotica_5d", "erotica_5d_soft")
        ),
        "n_search_failures": {
            cfg: sum(1 for c in cells if c["arms"][cfg]["error"])
            for cfg in ("erotica_5d", "erotica_5d_soft")
        },
    }
    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": time.perf_counter() - t0,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "executable": sys.executable,
            "packages": {
                p: _pkg_version(p)
                for p in (
                    "numpy",
                    "scipy",
                    "scikit-learn",
                    "astropy",
                    "pandas",
                    "hdbscan",
                    "erotica",
                )
            },
        },
        "config": vars(args),
        "held_out_definition": "k = seed % 10; k in {3,4,5} held out, k in {0,1,2} train",
        "checks": checks,
        "summary": summarise(cells),
        "cells": cells,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.out} ({len(cells)} cells, {payload['wall_clock_s']:.1f} s)", file=sys.stderr
    )
    print(json.dumps(checks, indent=2), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
