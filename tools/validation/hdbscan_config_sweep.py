#!/usr/bin/env python3
"""Screen every HDBSCAN knob and every pseudo-probability variant against injected truth.

WHY THIS EXISTS
---------------
Two open questions, and they interact, so they are answered in one design rather than two.

1. **Seven constructor knobs are never passed by this package**: ``cluster_selection_epsilon``,
   ``cluster_selection_persistence``, ``max_cluster_size``, ``alpha``, ``leaf_size``,
   ``approx_min_span_tree``, ``cluster_selection_epsilon_max``. An unmeasured knob is a recorded
   liability, not a default.

2. **The pseudo-probability leaves discrimination on the table.** ``p_tilde = probabilities_ * f_i``
   scores AUC 0.776 against ASteCA's 0.917 on the benchmark, and neither selector fix can move it
   (both act after ``data["probability"]`` is written). ``all_points_membership_vectors`` measured
   0.7643 -> 0.9542 on an ad-hoc grid, but with a min of 0.2378 -- BELOW CHANCE -- from reading an
   anti-correlated cluster. See ``~/phd/agent-findings/hdbscan-soft-membership.md``.

They interact because the soft variants must read *a column*, i.e. they inherit whatever the
cluster-choice machinery decides, which is exactly what the config knobs perturb.

DESIGN, and why it is not a full factorial
------------------------------------------
A full factorial over 7 knobs is 576 configurations; crossed with the cell grid and repeated over
seeds it is hours, and most of it would measure nothing because most knobs will turn out inert.
Instead:

  Stage 1 (this script, ``--stage screen``): one-factor-at-a-time from a documented baseline.
      Each factor level is scored on the SAME cells with the SAME seeds as the baseline, so the
      comparison is paired and the seed variance cancels. A factor is promoted only if its effect
      exceeds the paired seed-to-seed spread.
  Stage 2 (``--stage factorial --factors a,b,c``): full factorial on the promoted factors only.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
* If the soft variants' advantage does not survive on held-out seeds, it is overfitting to this
  generator and must not be adopted.
* If a variant wins on AUC while losing on ECE, it must NOT be adopted: calibration is the
  package's lead claim, and a better-ranking, worse-calibrated score makes the paper worse.
* If ``soft[:, tree_selected]`` does not fix the below-chance tail of ``soft[:, argmax_persistence]``
  then the failure is not the column choice and the whole soft route needs rethinking.

Every number this prints is measured here; nothing is quoted from elsewhere.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# The baseline is what the package ships today, so every effect is measured as a delta FROM the
# current behaviour rather than from an arbitrary reference.
BASELINE = dict(
    cluster_selection_method="eom",
    allow_single_cluster=False,
    min_samples=None,
    cluster_selection_epsilon=0.0,
    cluster_selection_persistence=0.0,
    max_cluster_size=0,
    alpha=1.0,
    approx_min_span_tree=True,
    leaf_size=40,
)

# One-factor-at-a-time levels. `None` in min_samples means "couple to min_cluster_size", which is
# the current behaviour and is itself under test -- Hunt & Reffert fix m_Pts and sweep m_clSize.
FACTORS: dict[str, list] = {
    "cluster_selection_method": ["leaf"],
    "allow_single_cluster": [True],
    "min_samples": [5, 10, 25],
    # Epsilon is in DISTANCE units, and the smoke run returned exactly 0.0000 for 0.02-0.1 --
    # which is the signature of a level below every core distance, not of an inert knob. The
    # cluster spread is 0.35 and the field spans +/-8, so the levels are widened to straddle
    # that scale. An exact zero at a level that CAN bite is a finding; at a level that cannot,
    # it is a wasted cell.
    "cluster_selection_epsilon": [0.1, 0.5, 1.5],
    "cluster_selection_persistence": [0.005, 0.02, 0.05, 0.1],
    "max_cluster_size": ["quarter", "half"],
    # alpha also returned exactly 0.0000 at 0.7/1.3. Widened for the same reason.
    "alpha": [0.5, 2.0],
    "approx_min_span_tree": [False],
    "leaf_size": [10, 100],
}

VARIANTS = [
    "probabilities",  # what ships today
    "soft_max",  # max over all cluster columns
    "soft_tree",  # column of the label the tree selector returns  <- precondition (1)
    "soft_persist",  # column of argmax(cluster_persistence_)
    "prob_x_notoutlier",  # probabilities_ * (1 - outlier_score); design note warns this
    # double-penalises, since soft already carries GLOSH. Measured, not assumed.
]


def make_cell(seed: int, n_cluster: int, contamination: float, spread: float = 0.35):
    """Cluster + uniform field in 2D proper motion. Truth is known by construction."""
    rng = np.random.default_rng(seed)
    n_field = int(round(n_cluster * contamination / (1.0 - contamination)))
    X = np.vstack(
        [
            rng.normal([0.0, 0.0], spread, (n_cluster, 2)),
            rng.uniform(-8.0, 8.0, (n_field, 2)),
        ]
    )
    truth = np.r_[np.ones(n_cluster, bool), np.zeros(n_field, bool)]
    order = rng.permutation(truth.size)  # row order must carry no membership information
    return X[order], truth[order]


def _tree_selected_label(model, labels: np.ndarray) -> int:
    """Reproduce Clustering._cluster_label_from_tree without importing the package.

    Kept local on purpose: this script must be able to score a config the package cannot yet
    produce, so it may not depend on package behaviour it is trying to measure.
    """
    try:
        tree = model.condensed_tree_.to_pandas()
    except Exception:
        return -1
    if tree is None or len(tree) == 0:
        return -1
    n = labels.size
    parent = int(tree.at[tree["lambda_val"].idxmax(), "parent"])
    parents = tree["parent"].to_numpy(dtype=int)
    children = tree["child"].to_numpy(dtype=int)
    frontier, seen, leaves = [parent], set(), []
    while frontier:
        node = frontier.pop()
        if node in seen:
            continue
        seen.add(node)
        for child in children[parents == node]:
            (leaves if child < n else frontier).append(int(child))
    if not leaves:
        return -1
    lab = labels[np.asarray(leaves, dtype=int)]
    lab = lab[lab != -1]
    if lab.size == 0:
        return -1
    vals, counts = np.unique(lab, return_counts=True)
    return int(vals[np.argmax(counts)])


def scorecard(truth: np.ndarray, score: np.ndarray, bins: int = 10) -> dict[str, float]:
    """Every metric this comparison needs, on one score vector.

    ROC-AUC alone is the wrong headline here and reporting it alone would mislead:

    * **ROC-AUC is insensitive to class imbalance.** At contamination 0.95 only 5% of rows are
      members, and ROC-AUC can look excellent while the top of the ranking is mostly field.
      ``average_precision`` (area under precision-recall) is the imbalance-aware twin, and its
      chance level is the positive RATE, not 0.5 -- so it is reported alongside that baseline.
    * **AUC measures ranking, never calibration.** This package's LEAD claim is calibrated
      per-star membership, so a variant that ranks better and calibrates worse is a regression
      dressed as an improvement.

    The Murphy decomposition is the load-bearing one, because it splits a single proper scoring
    rule into the two things being traded against each other:

        Brier = reliability - resolution + uncertainty

    ``reliability`` is miscalibration (lower better), ``resolution`` is how far the conditional
    rates move away from the base rate (higher better), ``uncertainty`` is irreducible and depends
    only on the cell. Two variants with the same Brier can differ entirely in which term they spend
    it on, and only the decomposition shows it.

    Also reported:

    * ``ece`` / ``mce`` on **equal-mass** bins, not equal-width. Equal-width bins are dominated by
      whichever bin happens to hold the mass, which for a peaked score is a binning artefact rather
      than a property of the score.
    * ``cal_slope`` / ``cal_intercept`` from a logistic fit on the logit of the score (Cox
      calibration). Perfect is slope 1, intercept 0; slope < 1 means over-confident, the usual
      failure of a raw density-based score. Unlike ECE this needs no bins at all.
    * ``spiegelhalter_z`` -- a formal test with a null: |z| > 1.96 rejects calibration at 5%.
    * ``sharpness`` -- sd of the predictions. A perfectly calibrated constant predictor at the base
      rate has zero reliability and zero resolution and is useless; sharpness is what rules it out.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(truth, dtype=float)
    s = np.clip(np.asarray(score, dtype=float), 1e-12, 1 - 1e-12)
    out: dict[str, float] = {}
    base = float(y.mean())
    out["base_rate"] = base

    if len(np.unique(y)) < 2:
        return {k: np.nan for k in ("roc_auc", "avg_precision", "brier")} | out

    out["roc_auc"] = float(roc_auc_score(y, s))
    out["avg_precision"] = float(average_precision_score(y, s))
    # Lift over the chance line for PR, which is the base rate rather than 0.5.
    out["ap_lift"] = out["avg_precision"] / base if base > 0 else np.nan

    out["brier"] = float(np.mean((s - y) ** 2))
    out["log_loss"] = float(-np.mean(y * np.log(s) + (1 - y) * np.log(1 - s)))
    out["sharpness"] = float(np.std(s))

    # --- Murphy decomposition on equal-mass bins ---
    order = np.argsort(s)
    edges = np.array_split(order, bins)
    rel = res = 0.0
    for idx in edges:
        if idx.size == 0:
            continue
        w = idx.size / y.size
        s_bar, y_bar = float(s[idx].mean()), float(y[idx].mean())
        rel += w * (s_bar - y_bar) ** 2
        res += w * (y_bar - base) ** 2
    out["reliability"] = rel
    out["resolution"] = res
    out["uncertainty"] = base * (1 - base)

    # --- calibration error on the same equal-mass bins ---
    gaps = []
    ece = 0.0
    for idx in edges:
        if idx.size == 0:
            continue
        g = abs(float(y[idx].mean()) - float(s[idx].mean()))
        gaps.append(g)
        ece += (idx.size / y.size) * g
    out["ece"] = ece
    out["mce"] = max(gaps) if gaps else np.nan

    # --- Cox calibration slope/intercept: bin-free, and it names the failure mode ---
    try:
        from sklearn.linear_model import LogisticRegression

        logit = np.log(s / (1 - s)).reshape(-1, 1)
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=500).fit(logit, y)
        out["cal_slope"] = float(lr.coef_[0][0])
        out["cal_intercept"] = float(lr.intercept_[0])
    except Exception:
        out["cal_slope"] = out["cal_intercept"] = np.nan

    # --- Spiegelhalter z: a calibration TEST, not just a distance ---
    num = float(np.sum((y - s) * (1 - 2 * s)))
    den = float(np.sqrt(np.sum(((1 - 2 * s) ** 2) * s * (1 - s))))
    out["spiegelhalter_z"] = num / den if den > 0 else np.nan
    return out


def recalibrated(truth: np.ndarray, score: np.ndarray, seed: int) -> dict[str, float]:
    """Scorecard after out-of-sample isotonic recalibration.

    This is the question that actually decides adoption. Reliability is *fixable* by the
    recalibration the package already ships; resolution is NOT -- no monotone map creates
    discrimination that the score does not contain. So a variant with high resolution and poor
    reliability is a better starting point than one that is merely well-calibrated, and comparing
    raw scores alone would rank them backwards.

    Fitted on half the rows and scored on the other half, so this measures recalibrated
    performance rather than the recalibrator's ability to memorise.
    """
    from sklearn.isotonic import IsotonicRegression

    y = np.asarray(truth, dtype=float)
    s = np.asarray(score, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(y.size)
    a, b = idx[: y.size // 2], idx[y.size // 2 :]
    if len(np.unique(y[a])) < 2 or len(np.unique(y[b])) < 2:
        return {}
    try:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s[a], y[a])
        return {f"iso_{k}": v for k, v in scorecard(y[b], iso.predict(s[b])).items()}
    except Exception:
        return {}


def score_cell(args) -> list[dict]:
    """Fit once, score every pseudo-probability variant off that single fit."""
    import hdbscan
    from sklearn.metrics import roc_auc_score  # oracle ceiling only

    factor, level, seed, n_cluster, contamination = args
    X, truth = make_cell(seed, n_cluster, contamination)
    cfg = dict(BASELINE)
    if factor != "_baseline":
        cfg[factor] = level
    if cfg["max_cluster_size"] == "quarter":
        cfg["max_cluster_size"] = max(2, X.shape[0] // 4)
    elif cfg["max_cluster_size"] == "half":
        cfg["max_cluster_size"] = max(2, X.shape[0] // 2)

    mcs = max(5, n_cluster // 4)
    t0 = time.time()
    try:
        model = hdbscan.HDBSCAN(
            min_cluster_size=mcs, prediction_data=True, metric="euclidean", **cfg
        ).fit(X)
    except Exception as exc:
        return [
            dict(
                factor=factor,
                level=str(level),
                seed=seed,
                n_cluster=n_cluster,
                contamination=contamination,
                variant=v,
                roc_auc=np.nan,
                oracle=np.nan,
                n_clusters=0,
                secs=np.nan,
                error=type(exc).__name__,
            )
            for v in VARIANTS
        ]
    t_fit = time.time() - t0

    labels = np.asarray(model.labels_, dtype=int)
    n_clusters = int(len(set(labels.tolist())) - (1 if -1 in labels else 0))
    if n_clusters < 1:
        return [
            dict(
                factor=factor,
                level=str(level),
                seed=seed,
                n_cluster=n_cluster,
                contamination=contamination,
                variant=v,
                roc_auc=np.nan,
                oracle=np.nan,
                n_clusters=0,
                secs=t_fit,
                error="no_cluster",
            )
            for v in VARIANTS
        ]

    # Oracle ceiling: the best AUC any single flat cluster's indicator could give. A variant
    # cannot be judged without knowing what the labelling made available.
    oracle = 0.0
    for lab in np.unique(labels):
        if lab == -1:
            continue
        ind = (labels == lab).astype(float)
        if 0 < ind.sum() < ind.size:
            oracle = max(oracle, float(roc_auc_score(truth, ind)))

    prob = np.asarray(model.probabilities_, dtype=float)
    outl = np.asarray(getattr(model, "outlier_scores_", np.zeros_like(prob)), dtype=float)
    outl = np.nan_to_num(outl, nan=0.0, posinf=1.0, neginf=0.0)
    try:
        soft = hdbscan.all_points_membership_vectors(model)
        soft = np.atleast_2d(np.asarray(soft, dtype=float))
        if soft.shape[0] != truth.size:
            soft = None
    except Exception:
        soft = None

    tree_lab = _tree_selected_label(model, labels)
    pers = np.asarray(getattr(model, "cluster_persistence_", []), dtype=float)
    pers_lab = int(np.argmax(pers)) if pers.size else -1

    scores: dict[str, np.ndarray | None] = {
        "probabilities": prob,
        "prob_x_notoutlier": prob * (1.0 - np.clip(outl, 0.0, 1.0)),
        "soft_max": soft.max(axis=1) if soft is not None and soft.shape[1] else None,
        "soft_tree": (
            soft[:, tree_lab] if soft is not None and 0 <= tree_lab < soft.shape[1] else None
        ),
        "soft_persist": (
            soft[:, pers_lab] if soft is not None and 0 <= pers_lab < soft.shape[1] else None
        ),
    }

    out = []
    for v in VARIANTS:
        sc = scores.get(v)
        row = dict(
            factor=factor,
            level=str(level),
            seed=seed,
            n_cluster=n_cluster,
            contamination=contamination,
            variant=v,
            oracle=oracle,
            n_clusters=n_clusters,
            secs=t_fit,
            error="",
        )
        if sc is None or not np.isfinite(sc).all():
            row["roc_auc"] = np.nan
        else:
            row.update(scorecard(truth, sc))
            row.update(recalibrated(truth, sc, seed))
        out.append(row)
    return out


def build_jobs(seeds, sizes, contams, factors: dict[str, list]):
    jobs = [("_baseline", None, s, n, c) for s in seeds for n in sizes for c in contams]
    for f, levels in factors.items():
        for lv in levels:
            jobs += [(f, lv, s, n, c) for s in seeds for n in sizes for c in contams]
    return jobs


def summarise(rows: list[dict], seeds_held_out: set[int]) -> None:
    import statistics as st

    def sel(f, v, held):
        return [
            r
            for r in rows
            if r["factor"] == f
            and r["variant"] == v
            and np.isfinite(r.get("roc_auc", np.nan))
            and ((r["seed"] in seeds_held_out) == held)
        ]

    print("\n" + "=" * 78)
    print("VARIANT COMPARISON at the shipped baseline config")
    print("=" * 78)
    print(
        f"{'variant':>20} {'AUC train':>11} {'AUC HELD-OUT':>13} {'ECE':>8} {'min AUC':>9} {'<0.5':>6}"
    )
    for v in VARIANTS:
        tr, ho = sel("_baseline", v, False), sel("_baseline", v, True)
        if not tr:
            print(f"{v:>20} {'--':>11} {'--':>13}")
            continue
        a_tr = np.mean([r["roc_auc"] for r in tr])
        a_ho = np.mean([r["roc_auc"] for r in ho]) if ho else float("nan")
        ece = np.mean([r["ece"] for r in tr if np.isfinite(r["ece"])])
        allauc = [r["roc_auc"] for r in tr + ho]
        below = sum(1 for x in allauc if x < 0.5)
        print(
            f"{v:>20} {a_tr:>11.4f} {a_ho:>13.4f} {ece:>8.4f} "
            f"{min(allauc):>9.4f} {below:>3}/{len(allauc)}"
        )
    orc = [
        r["oracle"] for r in sel("_baseline", "probabilities", False) if np.isfinite(r["oracle"])
    ]
    if orc:
        print(f"{'ORACLE ceiling':>20} {np.mean(orc):>11.4f}")

    print("\n" + "=" * 78)
    print("FACTOR SCREEN — paired delta vs baseline, per variant (same cells, same seeds)")
    print("A factor is PROMOTED only if |mean delta| exceeds the paired seed-to-seed spread.")
    print("=" * 78)
    promoted = set()
    for v in ("probabilities", "soft_tree", "soft_max"):
        base = {
            (r["seed"], r["n_cluster"], r["contamination"]): r["roc_auc"]
            for r in sel("_baseline", v, False)
        }
        if not base:
            continue
        print(f"\n--- variant: {v}")
        print(f"{'factor':>32} {'level':>10} {'d AUC':>9} {'sd':>8} {'n':>4}  verdict")
        for f in FACTORS:
            for lv in FACTORS[f]:
                rs = [r for r in sel(f, v, False) if str(r["level"]) == str(lv)]
                d = [
                    r["roc_auc"] - base[(r["seed"], r["n_cluster"], r["contamination"])]
                    for r in rs
                    if (r["seed"], r["n_cluster"], r["contamination"]) in base
                ]
                if len(d) < 3:
                    continue
                m, sd = float(np.mean(d)), float(st.stdev(d))
                sig = abs(m) > sd  # paired effect must clear its own spread
                if sig:
                    promoted.add(f)
                print(
                    f"{f:>32} {str(lv):>10} {m:>+9.4f} {sd:>8.4f} {len(d):>4}  "
                    f"{'PROMOTE' if sig else '.'}"
                )
    print(f"\nPromoted factors (stage 2 candidates): {sorted(promoted) or 'none'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="screen", choices=["screen", "factorial"])
    ap.add_argument("--factors", default="", help="stage 2: comma-separated promoted factors")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--held-out", type=int, default=3, help="seeds reserved for out-of-sample")
    ap.add_argument("--out", default="tools/validation/hdbscan_config_sweep.json")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    seeds = list(range(args.seeds + args.held_out))
    held = set(range(args.seeds, args.seeds + args.held_out))
    sizes, contams = (30, 61, 150), (0.5, 0.8, 0.95)

    if args.stage == "factorial":
        names = [f for f in args.factors.split(",") if f in FACTORS]
        if not names:
            print("stage 2 needs --factors from the promoted list", file=sys.stderr)
            return 2
        grids = [[None] + FACTORS[n] for n in names]
        jobs = [("_baseline", None, s, n, c) for s in seeds for n in sizes for c in contams]
        for combo in itertools.product(*grids):
            if all(c is None for c in combo):
                continue
            tag = "+".join(f"{n}={v}" for n, v in zip(names, combo, strict=True) if v is not None)
            for s in seeds:
                for n in sizes:
                    for c in contams:
                        jobs.append((tag, combo, s, n, c))
        print(f"stage 2: {len(names)} factors -> {len(jobs)} fits", flush=True)
        print("NOT IMPLEMENTED: stage 2 needs a combo-aware score_cell; run screen first.")
        return 3

    jobs = build_jobs(seeds, sizes, contams, FACTORS)
    print(
        f"screen: {len(jobs)} fits "
        f"({len(seeds)} seeds, {args.held_out} held out | {len(sizes)}x{len(contams)} cells | "
        f"{1 + sum(len(v) for v in FACTORS.values())} configs) on {args.workers} workers",
        flush=True,
    )

    t0 = time.time()
    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(score_cell, jobs, chunksize=8), 1):
            rows.extend(res)
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    elapsed = time.time() - t0

    # Two files, deliberately. The raw per-fit rows are 11.5 MB for a 2700-fit screen -- a
    # checkpoint, not a sidecar -- and the same defect that made the benchmark sidecars
    # unreviewable. The REDUCED file is what a paper quotes and what git keeps; the raw file is
    # regenerable by re-running this script and is gitignored.
    reduced_path = Path(str(args.out).replace(".json", ".summary.json"))
    # Only the metrics a conclusion is ever quoted from. Keeping all ~35 produced a 1.08 MB
    # "summary", which is still a checkpoint wearing a sidecar's name.
    KEEP = (
        "roc_auc",
        "avg_precision",
        "brier",
        "reliability",
        "resolution",
        "ece",
        "cal_slope",
        "spiegelhalter_z",
        "iso_ece",
        "iso_resolution",
        "oracle",
    )
    agg: dict[str, dict] = {}
    for r in rows:
        key = f"{r['factor']}|{r['level']}|{r['variant']}"
        blk = "held_out" if r["seed"] in held else "train"
        d = agg.setdefault(key, {})
        for m in KEEP:
            v = r.get(m)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v):
                d.setdefault(f"{blk}:{m}", []).append(float(v))
    reduced = {
        k: {
            m: {
                "mean": float(np.mean(v)),
                "sd": float(np.std(v)),
                "n": len(v),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
            }
            for m, v in d.items()
        }
        for k, d in agg.items()
    }
    reduced_path.write_text(
        json.dumps(
            {
                "note": "Aggregated from the raw per-fit rows; quote THIS file, not the raw one.",
                "baseline": {k: str(v) for k, v in BASELINE.items()},
                "held_out_seeds": sorted(held),
                "n_fits": len(jobs),
                "elapsed_s": elapsed,
                "by_factor_level_variant": reduced,
            },
            indent=1,
        )
    )
    print(f"wrote {reduced_path} (reduced, quotable)")

    out = Path(args.out)
    out.write_text(
        json.dumps(
            {
                "baseline": {
                    k: (str(v) if not isinstance(v, (int, float, bool, type(None))) else v)
                    for k, v in BASELINE.items()
                },
                "factors": {k: [str(x) for x in v] for k, v in FACTORS.items()},
                "variants": VARIANTS,
                "seeds": seeds,
                "held_out_seeds": sorted(held),
                "sizes": list(sizes),
                "contaminations": list(contams),
                "elapsed_s": elapsed,
                "n_fits": len(jobs),
                "rows": rows,
            },
            indent=1,
        )
    )
    print(f"\nwrote {out} ({len(jobs)} fits, {elapsed:.0f}s)")
    summarise(rows, held)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
