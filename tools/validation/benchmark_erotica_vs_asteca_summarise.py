"""Aggregate the EROTICA-vs-ASteCA benchmark cells into the tables the report quotes.

Reads ``benchmark_erotica_vs_asteca.json`` (or the ``.cells.jsonl`` checkpoint if the run
was interrupted) and emits, to stdout, the four axes with across-seed uncertainties:

* (d) CALIBRATION  -- Brier / ECE / MCE, mean +- standard error over seeds
* (b) RECOVERY     -- signed error per parameter at the default threshold and at the
                      shared truth-free top-K operating point
* (c) RUNTIME      -- median seconds, and the scaling with n_sources
* (a) AGREEMENT    -- descriptive only (no truth); AUC vs truth IS truth-based and is
                      reported with it because it is the one threshold-free ranking score

Uncertainties are the standard error across independent seeds within a cell group, which
is the only uncertainty this design supports: a single realisation's ECE has no error bar.

Usage:  python tools/validation/benchmark_erotica_vs_asteca_summarise.py [--json PATH]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    jl = path.with_suffix(".cells.jsonl")
    cells = [json.loads(line) for line in jl.read_text().splitlines() if line.strip()]
    return {
        "cells": cells,
        "negative_control": None,
        "sweep_sensitivity": None,
        "out_of_sample_recalibration": {},
        "environment": {},
        "config": {},
        "_partial": True,
    }


def _ms(values) -> tuple[float, float]:
    """mean and standard error, NaN-safe."""
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return float("nan"), float("nan")
    if v.size == 1:
        return float(v[0]), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def _fmt(m: float, s: float, nd: int = 3) -> str:
    if not np.isfinite(m):
        return "--"
    if not np.isfinite(s):
        return f"{m:.{nd}f}"
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--json", default=str(Path(__file__).with_name("benchmark_erotica_vs_asteca.json"))
    )
    args = ap.parse_args(argv)
    d = _load(Path(args.json))
    cells = d["cells"]
    print(
        f"# cells: {len(cells)}"
        + ("  [PARTIAL: read from checkpoint]" if d.get("_partial") else "")
    )
    if d.get("environment"):
        print(
            "# env:",
            d["environment"].get("python"),
            {k: v for k, v in d["environment"].get("packages", {}).items()},
        )

    # ---- generator self-check -------------------------------------------
    bad = [
        c
        for c in cells
        if not c["generator_check"].get("contamination_ok", True)
        or not c["generator_check"].get("centroid_ok", True)
    ]
    print(
        f"\n## generator self-check: {len(cells) - len(bad)}/{len(cells)} cells pass "
        f"(contamination within 0.02, centroid within 5 sigma/sqrt(N))"
    )

    # K vs truth: how much of the n_members error column is ASteCA's Ripley count error
    kk = [(c["k_shared"], c["n_members"]) for c in cells if c.get("k_shared")]
    if kk:
        dk = [k - n for k, n in kk]
        print(
            f"## shared-K provenance: K = ASteCA Ripley N_cluster; "
            f"K - N_true = {_fmt(*_ms(dk), nd=2)} over {len(kk)} cells "
            f"(this is Ripley's counting error, charged equally to every method)"
        )
    if cells and cells[0].get("adaptive"):
        caps = defaultdict(list)
        for c in cells:
            for cfg, v in (c.get("adaptive_caps") or {}).items():
                caps[cfg].append(v / max(c["n_members"], 1))
        print(
            "## adaptive sweep cap / N_true: "
            + "  ".join(f"{k}={_fmt(*_ms(v), nd=2)}" for k, v in sorted(caps.items()))
        )

    # ---- (d) calibration, pooled over all cells -------------------------
    print("\n## (d) CALIBRATION  — mean ± s.e. across all cells")
    print(f"{'method':<28} {'Brier':>18} {'ECE':>18} {'MCE':>18}")
    cal = defaultdict(lambda: defaultdict(list))
    for c in cells:
        for r in c["calibration"]:
            for k in ("brier", "ece", "mce"):
                cal[r["method"]][k].append(r[k])
    for m in cal:
        print(
            f"{m:<28} " + " ".join(f"{_fmt(*_ms(cal[m][k])):>18}" for k in ("brier", "ece", "mce"))
        )

    # ---- calibration split by contamination -----------------------------
    print("\n## (d) ECE by contamination fraction")
    conts = sorted({c["contamination"] for c in cells})
    methods = sorted({r["method"] for c in cells for r in c["calibration"]})
    print(f"{'method':<28} " + " ".join(f"{'c=' + str(x):>18}" for x in conts))
    for m in methods:
        row = []
        for x in conts:
            vals = [
                r["ece"]
                for c in cells
                if c["contamination"] == x
                for r in c["calibration"]
                if r["method"] == m
            ]
            row.append(f"{_fmt(*_ms(vals)):>18}")
        print(f"{m:<28} " + " ".join(row))

    print("\n## (d) ECE by true cluster size N")
    ns = sorted({c["n_members"] for c in cells})
    counts = {x: sum(1 for c in cells if c["n_members"] == x) for x in ns}
    print(f"{'method':<28} " + " ".join(f"{'N=' + str(x) + f' (n={counts[x]})':>18}" for x in ns))
    for m in methods:
        row = []
        for x in ns:
            vals = [
                r["ece"]
                for c in cells
                if c["n_members"] == x
                for r in c["calibration"]
                if r["method"] == m
            ]
            row.append(f"{_fmt(*_ms(vals)):>18}")
        print(f"{m:<28} " + " ".join(row))

    # ---- AUC vs truth (threshold-free ranking quality) -------------------
    print(
        "\n## (a/d) AUC of each method's probabilities against TRUTH (threshold-free; 0.5 = chance)"
    )
    auc = defaultdict(list)
    for c in cells:
        for a in c["agreement"]:
            if a["method_b"] == "_truth_positive_control":
                auc[a["method_a"]].append(a["auc_a_scores_b"])
            elif a["method_a"] == "_truth_positive_control":
                auc[a["method_b"]].append(a["auc_b_scores_a"])
    for m, v in auc.items():
        print(f"{m:<28} {_fmt(*_ms(v)):>18}  (n={len(v)})")

    # ---- (b) recovery at the DEFAULT threshold --------------------------
    for label, key in (
        ("default threshold p>=0.5", "recovery"),
        ("shared truth-free top-K", "recovery_topk_shared"),
    ):
        print(f"\n## (b) PARAMETER RECOVERY — {label}: signed error, mean ± s.e.")
        params = ["pmra_c", "pmdec_c", "plx_c", "sigma_pm", "n_members"]
        truth = {"pmra_c": -1.20, "pmdec_c": -0.50, "plx_c": 1.00, "sigma_pm": 0.20}
        acc = defaultdict(lambda: defaultdict(list))
        nfail = defaultdict(int)
        for c in cells:
            if key == "recovery":
                for r in c["recovery"]:
                    acc[r["method"]][r["param"]].append(r["abs_error"])
                    if r["param"] == "pmra_c" and not np.isfinite(r["recovered"]):
                        nfail[r["method"]] += 1
            else:
                for m, rec in (c.get("recovery_topk_shared") or {}).items():
                    # 0/1 probabilities have no rank order, so the truth control cannot be
                    # rank-matched; it stays the positive control for calibration only.
                    if m == "_truth_positive_control":
                        continue
                    for p in params:
                        v = rec.get(p)
                        t = truth.get(p, float(c["n_members"]))
                        acc[m][p].append(
                            (v - t) if v is not None and np.isfinite(v) else float("nan")
                        )
                    for extra in ("purity", "recall"):
                        acc[m][extra].append(rec.get(extra, float("nan")))
        cols = params + (["purity", "recall"] if key != "recovery" else [])
        print(
            f"{'method':<28} "
            + " ".join(f"{p:>17}" for p in cols)
            + ("   n_no_members" if key == "recovery" else "")
        )
        for m in sorted(acc):
            row = " ".join(f"{_fmt(*_ms(acc[m][p])):>17}" for p in cols)
            tail = f"   {nfail[m]}/{len(cells)}" if key == "recovery" else ""
            print(f"{m:<28} {row}{tail}")

    # ---- (c) runtime ----------------------------------------------------
    print("\n## (c) RUNTIME — median seconds per realisation (all cells)")
    rt = defaultdict(list)
    for c in cells:
        for r in c["runtime"]:
            if r["runtime_s"] is not None and np.isfinite(r["runtime_s"]):
                rt[r["method"]].append(r["runtime_s"])
    for m in sorted(rt):
        v = np.asarray(rt[m])
        print(f"{m:<28} median={np.median(v):8.2f}  min={v.min():7.2f}  max={v.max():8.2f}")

    # ---- (a) agreement, descriptive -------------------------------------
    print("\n## (a) AGREEMENT — all method pairs (descriptive, no truth)")
    agr = defaultdict(lambda: defaultdict(list))
    for c in cells:
        for a in c["agreement"]:
            if "_truth_positive_control" in (a["method_a"], a["method_b"]):
                continue
            agr[f"{a['method_a']} vs {a['method_b']}"]["jaccard"].append(a["jaccard"])
            agr[f"{a['method_a']} vs {a['method_b']}"]["kappa"].append(a["cohen_kappa"])
    for m in sorted(agr):
        print(
            f"{m:<45} jaccard={_fmt(*_ms(agr[m]['jaccard'])):>16}  "
            f"kappa={_fmt(*_ms(agr[m]['kappa'])):>16}"
        )

    # ---- branch diagnostic ----------------------------------------------
    print("\n## DIAGNOSTIC — where EROTICA's default selection goes")
    diag = defaultdict(lambda: defaultdict(list))
    smatch = defaultdict(list)
    empt = defaultdict(lambda: defaultdict(int))
    changed = defaultdict(int)
    for c in cells:
        for dgn in c.get("branch_diagnostic") or []:
            if "error" in dgn:
                continue
            cfg = dgn["config"]
            for k in (
                "selected_purity",
                "selected_recall",
                "oracle_best_purity",
                "oracle_best_recall",
                "f_max",
            ):
                diag[cfg][k].append(dgn[k])
            # present only in runs produced after the issue-#7 fix
            for k in ("legacy_selected_purity", "legacy_selected_recall"):
                if k in dgn:
                    diag[cfg][k].append(dgn[k])
            for side in ("selected", "legacy_selected"):
                if dgn.get(f"{side}_is_empty"):
                    empt[cfg][side] += 1
            changed[cfg] += bool(dgn.get("selector_changed"))
            smatch[cfg].append(bool(dgn["size_match_found"]))
    for cfg in sorted(diag):
        n = len(smatch[cfg])
        print(
            f"{cfg:<22} selected purity={_fmt(*_ms(diag[cfg]['selected_purity'])):>15} "
            f"recall={_fmt(*_ms(diag[cfg]['selected_recall'])):>15} | "
            f"ORACLE branch purity={_fmt(*_ms(diag[cfg]['oracle_best_purity'])):>15} "
            f"recall={_fmt(*_ms(diag[cfg]['oracle_best_recall'])):>15} | "
            f"size-match hit {sum(smatch[cfg])}/{n} | "
            f"f_max={_fmt(*_ms(diag[cfg]['f_max'])):>15}"
        )
        if diag[cfg]["legacy_selected_purity"]:
            print(
                f"{'  ^ LEGACY selector':<22} "
                f"purity={_fmt(*_ms(diag[cfg]['legacy_selected_purity'])):>15} "
                f"recall={_fmt(*_ms(diag[cfg]['legacy_selected_recall'])):>15} | "
                f"selector changed the label in {changed[cfg]}/{n} cells | "
                f"empty selection: fixed {empt[cfg]['selected']}/{n}, "
                f"legacy {empt[cfg]['legacy_selected']}/{n}"
            )

    # ---- negative control ------------------------------------------------
    if d.get("negative_control"):
        print(
            "\n## NEGATIVE CONTROL — field only, no cluster injected "
            "(anything selected is a false positive)"
        )
        for ctl in d["negative_control"]:
            print(f"  seed {ctl['seed']}  n_sources={ctl['n_sources']}")
            for m, r in ctl["methods"].items():
                print(
                    f"    {m:<26} n_selected={r['n_selected']:5d}  "
                    f"max_prob={r['max_prob']:.3f}  err={r['error']}"
                )

    # ---- sweep sensitivity ----------------------------------------------
    if d.get("sweep_sensitivity"):
        print(
            "\n## SWEEP SENSITIVITY — p-tilde depends on the mcs sweep width "
            "(same realisation, N=61, c=0.8)"
        )
        for c in d["sweep_sensitivity"]:
            hi = c["mcs_range"][1]
            fm = {g["config"]: g.get("f_max") for g in c.get("branch_diagnostic") or []}
            n_half = {
                g["config"]: g.get("n_ptilde_over_half") for g in c.get("branch_diagnostic") or []
            }
            ece = {r["method"]: r["ece"] for r in c["calibration"]}
            print(
                f"  mcs 10..{hi:<4d} f_max(3d)={fm.get('erotica_3d')}  "
                f"n(p>0.5)(3d)={n_half.get('erotica_3d')}  "
                f"ECE(erotica_3d)={ece.get('erotica_3d', float('nan')):.3f}  "
                f"ECE(asteca)={ece.get('asteca_fastmp', float('nan')):.3f}"
            )

    # ---- out-of-sample recalibration -------------------------------------
    if d.get("out_of_sample_recalibration"):
        print("\n## OUT-OF-SAMPLE RECALIBRATION — isotonic/Platt fitted on disjoint seeds")
        print(
            f"{'method':<28} {'raw ECE':>10} {'iso ECE':>10} {'platt ECE':>10} "
            f"{'raw Brier':>10} {'iso Brier':>10}"
        )
        for m, r in d["out_of_sample_recalibration"].items():
            if "error" in r:
                print(f"{m:<28} {r['error']}")
                continue
            print(
                f"{m:<28} {r.get('raw_ece_test', float('nan')):10.4f} "
                f"{r.get('isotonic_ece_test', float('nan')):10.4f} "
                f"{r.get('platt_ece_test', float('nan')):10.4f} "
                f"{r.get('raw_brier_test', float('nan')):10.4f} "
                f"{r.get('isotonic_brier_test', float('nan')):10.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
