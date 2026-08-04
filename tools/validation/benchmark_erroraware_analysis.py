"""Held-out scoring of the error-aware arms, against the pre-registered falsification.

WHY THIS EXISTS
---------------
``benchmark_erotica_vs_asteca_summarise.py`` reports calibration, recovery, runtime and
agreement, but it does **not** report the per-cell ROC-AUC / average precision split by the
held-out seeds -- which is the axis the ``*_erroraware`` arms were added to move, and the
axis the pre-registered falsification criterion is written on. This script does exactly
that, and nothing else.

THE CRITERION IT EVALUATES (pre-registered in ``benchmark_erotica_vs_asteca.py``)
--------------------------------------------------------------------------------
The hypothesis is that the EROTICA-to-ASteCA gap is per-star error propagation. It is
falsified if the error-aware arms do not improve **held-out average precision** over their
non-error-aware counterparts. The comparison that decides it is
``erotica_5d_erroraware - erotica_5d_coarsef``, not ``- erotica_5d``: the ``coarsef`` arm is
the matched control that holds the decimated sweep grid fixed and turns the error
resampling off, so the difference against it isolates error propagation from grid
coarseness. Both deltas are printed; the one against ``coarsef`` is the one the criterion
is read from.

HELD-OUT SPLIT
--------------
``k = seed % 10``; ``k in {3,4,5}`` held out. Verified to reproduce the published
108-cell table to 4 decimals -- see ``benchmark_ptilde_decomposition.py``.

METRIC CHOICE, WHICH IS NOT COSMETIC
------------------------------------
Base rates here are 0.50 / 0.20 / 0.05 by contamination. ROC-AUC is insensitive to that;
average precision is not, and it is the metric the gap is largest on (0.514 vs 0.862).
Both are printed, AP is the headline, and the AP-over-base-rate lift is printed too so the
three contamination levels can be compared at all.

USAGE
-----
    python tools/validation/benchmark_erroraware_analysis.py --json benchmark_erroraware.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HELD_OUT_K = (3, 4, 5)

ARMS = [
    "erotica_3d",
    "erotica_3d_soft",
    "erotica_5d",
    "erotica_5d_soft",
    "erotica_5d_coarsef",
    "erotica_5d_erroraware",
    "erotica_5d_soft_erroraware",
    "asteca_fastmp",
]

# (new arm, its matched control) -- the pairs the falsification criterion is read from.
PAIRS = [
    ("erotica_5d_erroraware", "erotica_5d_coarsef"),
    ("erotica_5d_erroraware", "erotica_5d"),
    ("erotica_5d_soft_erroraware", "erotica_5d_soft"),
    ("erotica_5d_coarsef", "erotica_5d"),
    ("erotica_5d_soft", "erotica_5d"),
]


def ms(vals) -> tuple[float, float, int]:
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    if v.size == 1:
        return float(v[0]), float("nan"), 1
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size)), int(v.size)


def fmt(m: float, s: float, nd: int = 4) -> str:
    if not np.isfinite(m):
        return "--"
    return f"{m:.{nd}f}" if not np.isfinite(s) else f"{m:.{nd}f} ± {s:.{nd}f}"


def per_cell(cells, arm):
    """[(seed, roc, ap, base_rate)] for one arm."""
    out = []
    for c in cells:
        if arm not in c["_probs"]:
            continue
        y = np.asarray(c["_truth"], dtype=int)
        p = np.asarray(c["_probs"][arm], dtype=float)
        if y.sum() == 0 or y.sum() == y.size:
            continue
        out.append((c["seed"], roc_auc_score(y, p), average_precision_score(y, p), float(y.mean())))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=str(Path(__file__).with_name("benchmark_erroraware.json")))
    args = ap.parse_args(argv)
    path = Path(args.json)
    jl = path.with_suffix(".cells.jsonl")
    cells = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()]
    payload = json.loads(path.read_text()) if path.exists() else {}

    held = [c for c in cells if (c["seed"] % 10) in HELD_OUT_K]
    train = [c for c in cells if (c["seed"] % 10) not in HELD_OUT_K]
    print(f"# cells: {len(cells)}  (held out {len(held)}, train {len(train)})")
    if payload:
        print(f"# wall clock: {payload.get('wall_clock_s', float('nan')):.0f} s")
        cfgd = payload.get("config", {})
        print(
            f"# ea_nmc={cfgd.get('ea_nmc')}  ea_mcs_step={cfgd.get('ea_mcs_step')}  "
            f"asteca_runs={cfgd.get('asteca_runs')}  mcs={cfgd.get('mcs_lo')}..{cfgd.get('mcs_hi')}"
        )

    arms = [a for a in ARMS if any(a in c["_probs"] for c in cells)]

    # ---- errors per arm, so a "result" that is really a crash is visible ----
    print("\n## arm failures (a cell whose search raised returns an all-zero p, and is kept)")
    for a in arms:
        n = sum(1 for c in cells if a in (c.get("errors") or {}))
        print(f"  {a:<30} {n:3d}/{len(cells)}")

    # ---- headline table --------------------------------------------------
    for label, sub in (("HELD-OUT", held), ("train", train)):
        print(f"\n## {label} — per-cell ROC-AUC and average precision, mean ± s.e. across cells")
        print(f"{'arm':<30} {'ROC-AUC':>20} {'AvgPrec':>20} {'AP lift over base':>20}")
        for a in arms:
            rows = [r for r in per_cell(sub, a)]
            roc = ms([r[1] for r in rows])
            apv = ms([r[2] for r in rows])
            lift = ms([r[2] / r[3] for r in rows if r[3] > 0])
            print(f"{a:<30} {fmt(*roc[:2]):>20} {fmt(*apv[:2]):>20} {fmt(*lift[:2], nd=3):>20}")

    # ---- paired deltas, held-out ----------------------------------------
    # A cell whose search raised contributes an all-zero p to BOTH arms of a pair, so its
    # paired delta is exactly 0.0: it cannot register as a win, and it drags the mean toward
    # zero. Those cells are real results and stay in the headline row, but the second row
    # restricts to cells where both arms actually produced a score, because "18/34 wins" and
    # "18/20 wins with 14 ties" are very different claims.
    print("\n## HELD-OUT paired deltas (same cell, same seed — seed variance cancels)")
    print(f"{'comparison':<48} {'dROC':>20} {'dAP':>20} {'AP wins':>12}")
    for new, base in PAIRS:
        if new not in arms or base not in arms:
            continue
        a = {s: (r, p) for s, r, p, _ in per_cell(held, new)}
        b = {s: (r, p) for s, r, p, _ in per_cell(held, base)}
        failed = {
            c["seed"]
            for c in held
            if new in (c.get("errors") or {}) or base in (c.get("errors") or {})
        }
        common = sorted(set(a) & set(b))
        live = [s for s in common if s not in failed]
        for tag, sel in ((" [all cells]", common), (" [both arms scored]", live)):
            if not sel:
                continue
            dr = [a[s][0] - b[s][0] for s in sel]
            dp = [a[s][1] - b[s][1] for s in sel]
            wins = sum(1 for x in dp if x > 0)
            ties = sum(1 for x in dp if x == 0.0)
            print(
                f"{new + ' − ' + base + tag:<48} {fmt(*ms(dr)[:2]):>20} "
                f"{fmt(*ms(dp)[:2]):>20} {f'{wins}/{len(sel)}' + (f' ({ties} tie)' if ties else ''):>12}"
            )

    # ---- AP by contamination (base rate varies 0.5 / 0.2 / 0.05) ---------
    print("\n## HELD-OUT average precision by contamination (base rate = AP chance level)")
    conts = sorted({c["contamination"] for c in held})
    print(f"{'arm':<30} " + " ".join(f"{'c=' + str(x):>20}" for x in conts))
    for a in arms:
        row = []
        for x in conts:
            sub = [c for c in held if c["contamination"] == x]
            row.append(f"{fmt(*ms([r[2] for r in per_cell(sub, a)])[:2]):>20}")
        print(f"{a:<30} " + " ".join(row))
    print(f"{'(base rate)':<30} " + " ".join(f"{1 - x:>20.4f}" for x in conts))

    # ---- negative control -------------------------------------------------
    ctl = payload.get("negative_control")
    if ctl:
        print(
            "\n## NEGATIVE CONTROL — field only, NO cluster injected. Anything selected is "
            "a false positive."
        )
        for c in ctl:
            print(f"  seed {c['seed']}  n_sources={c['n_sources']}  n_true_members=0")
            for m, r in c["methods"].items():
                print(
                    f"    {m:<30} n_selected={r['n_selected']:5d}  max_prob={r['max_prob']:.4f}"
                    f"  err={r['error']}"
                )
    else:
        print("\n## NEGATIVE CONTROL: absent from the payload — not run, or run interrupted.")

    # ---- calibration, since the harness computes it anyway ---------------
    print("\n## HELD-OUT calibration (Brier / ECE), mean ± s.e.")
    print(f"{'arm':<30} {'Brier':>20} {'ECE':>20}")
    for a in arms + ["_truth_positive_control"]:
        br = [r["brier"] for c in held for r in c["calibration"] if r["method"] == a]
        ec = [r["ece"] for c in held for r in c["calibration"] if r["method"] == a]
        if not br:
            continue
        print(f"{a:<30} {fmt(*ms(br)[:2]):>20} {fmt(*ms(ec)[:2]):>20}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
