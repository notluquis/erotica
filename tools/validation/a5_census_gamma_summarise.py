#!/usr/bin/env python3
r"""Summarise the A5 census gamma sweep: distribution, recoverability gate, convergence.

Reads the sidecar written by ``a5_census_gamma_sweep.py`` (``*.cells.jsonl``) and the pinned
background recoverability calibration (``eff_gamma_bias_lowN_ratios_pinnedbg.json``), and reports
the numbers the A5 write-up needs. Computes nothing that the sweep did not measure.

THE GATE IS TWO-PART AND BOTH PARTS MATTER
------------------------------------------
1. **Convergence**: r-hat < 1.01, min ESS (bulk and tail) > 400, zero divergences -- read from
   ``az.rhat``/``az.ess`` at fit time, never from ``az.summary``.
2. **Recoverability**: ``r_tot/a_fit`` above the threshold at which the pinned-background
   lever-arm bias is consistent with zero. A cluster can converge beautifully and still not have
   a measured ``gamma``: the sampler is reporting the posterior it was given, and below the
   threshold that posterior is dominated by a bias the likelihood cannot see.

USAGE
-----
    python tools/validation/a5_census_gamma_summarise.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Convergence gate, as specified for A5.
RHAT_MAX, ESS_MIN, DIV_MAX = 1.01, 400.0, 0


def load(sidecar: Path):
    rows = []
    for line in sidecar.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def q(x, *ps):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, ps)) if x.size else tuple(np.nan for _ in ps)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sidecar", type=Path, default=HERE / "a5_census_gamma_sweep.cells.jsonl")
    ap.add_argument("--calib", type=Path, default=HERE / "eff_gamma_bias_lowN_ratios_pinnedbg.json")
    ap.add_argument(
        "--ratio-gate",
        type=float,
        default=16.0,
        help="minimum r_tot/a_fit for gamma to count as measured",
    )
    ap.add_argument("--out", type=Path, default=HERE / "a5_census_gamma_summary.json")
    args = ap.parse_args()

    rows = load(args.sidecar)
    print(f"cells in sidecar: {len(rows)}")
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        print(f"  WARNING: {len(names) - len(set(names))} duplicate names -- resume key broken")

    ok = [r for r in rows if r.get("status") == "ok"]
    print(f"  status ok:  {len(ok)}   skipped/error: {len(rows) - len(ok)}")
    for r in rows:
        s = r.get("status", "?")
        if s != "ok" and not s.startswith("skipped"):
            print(f"    {r['name']}: {s}")

    # ---- convergence gate, from the recorded az.rhat / az.ess values -------------------------
    def conv(r):
        return (
            r.get("eff_rhat_max", 9) < RHAT_MAX
            and min(r.get("eff_ess_bulk_min", 0), r.get("eff_ess_tail_min", 0)) > ESS_MIN
            and r.get("eff_divergences", 1) <= DIV_MAX
        )

    passed = [r for r in ok if conv(r)]
    print(f"\n[convergence] r-hat < {RHAT_MAX}, ESS > {ESS_MIN}, divergences <= {DIV_MAX}")
    print(
        f"  pass: {len(passed)}/{len(ok)} of fitted  ({100 * len(passed) / max(len(ok), 1):.2f}%)"
        f"   = {100 * len(passed) / max(len(rows), 1):.2f}% of attempted"
    )
    for label, key, worst in (
        ("max r-hat", "eff_rhat_max", max),
        ("min ESS bulk", "eff_ess_bulk_min", min),
        ("min ESS tail", "eff_ess_tail_min", min),
        ("max divergences", "eff_divergences", max),
    ):
        v = [r[key] for r in ok if key in r]
        if v:
            print(f"  {label:16s} worst {worst(v):.4g}   median {np.median(v):.4g}")
    nfail = [r for r in ok if not conv(r)]
    for r in nfail[:20]:
        print(
            f"    FAIL {r['name']}: rhat={r.get('eff_rhat_max'):.4f} "
            f"essb={r.get('eff_ess_bulk_min'):.0f} esst={r.get('eff_ess_tail_min'):.0f} "
            f"div={r.get('eff_divergences')}"
        )

    # ---- recoverability -----------------------------------------------------------------------
    ratio = np.array([r["rtot_over_a"] for r in passed], dtype=float)
    gam = np.array([r["gamma_median"] for r in passed], dtype=float)
    n = np.array([r["n_fitted"] for r in passed], dtype=float)
    age = np.array([r["logAge50"] for r in passed], dtype=float)
    ratio_rc = np.array([r["rtot_over_rc"] for r in passed], dtype=float)

    print(
        f"\n[recoverability] r_tot/a_fit  median {np.median(ratio):.2f}  "
        f"p16/p84 {q(ratio, 16, 84)[0]:.2f}/{q(ratio, 16, 84)[1]:.2f}"
    )
    print(
        f"  catalogue r_tot/r_c        median {np.nanmedian(ratio_rc):.2f}   "
        f"(NOT interchangeable with r_tot/a_fit)"
    )
    print(f"  Spearman-ish corr(log ratio, gamma) = {np.corrcoef(np.log(ratio), gam)[0, 1]:+.3f}")
    for thr in (2, 4, 8, 16, 32):
        sel = ratio >= thr
        print(
            f"  r_tot/a >= {thr:3d}: {sel.sum():5d} ({100 * sel.mean():5.1f}%)  "
            f"gamma median {np.median(gam[sel]) if sel.any() else np.nan:.3f}"
        )

    keep = ratio >= args.ratio_gate
    print(
        f"\n[BOTH GATES] converged AND r_tot/a_fit >= {args.ratio_gate}: "
        f"{int(keep.sum())} of {len(rows)} attempted "
        f"({100 * keep.sum() / max(len(rows), 1):.2f}%)"
    )

    # ---- the gamma distribution, gated and ungated --------------------------------------------
    out = {}
    for label, g in (("all converged", gam), (f"r_tot/a>={args.ratio_gate}", gam[keep])):
        if g.size == 0:
            continue
        p = np.percentile(g, [2.5, 16, 25, 50, 75, 84, 97.5])
        # A pile-up at 2 has to show up as an excess of the mass near 2 relative to a smooth
        # unimodal spread. Report the raw fractions and let the reader judge.
        print(f"\n[gamma] {label}  (n={g.size})")
        print("  percentiles 2.5/16/25/50/75/84/97.5: " + " ".join(f"{v:.3f}" for v in p))
        print(f"  mean {g.mean():.3f}  sd {g.std(ddof=1):.3f}")
        for lo, hi in ((0, 1.8), (1.8, 2.2), (2.2, 2.6), (2.6, 3.0), (3.0, 4.0), (4.0, 99)):
            f = float(np.mean((g >= lo) & (g < hi)))
            print(f"  {lo:4.1f} <= gamma < {hi:4.1f}: {f * 100:5.1f}%  ({int(f * g.size):5d})")
        out[label] = dict(
            n=int(g.size),
            median=float(np.median(g)),
            mean=float(g.mean()),
            sd=float(g.std(ddof=1)),
            pct={str(k): float(v) for k, v in zip([2.5, 16, 25, 50, 75, 84, 97.5], p, strict=True)},
            frac_1p8_2p2=float(np.mean((g >= 1.8) & (g < 2.2))),
            frac_below_2=float(np.mean(g < 2.0)),
        )

    # ---- the young subsample: the scale claim -------------------------------------------------
    young = age < 7.0
    print(
        f"\n[young] logAge50 < 7.0 (10 Myr): {int(young.sum())} converged; "
        f"{int((young & keep).sum())} also pass the ratio gate"
    )
    if (young & keep).any():
        gy = gam[young & keep]
        print(
            f"  gamma median {np.median(gy):.3f}  n={gy.size}  "
            f"16/84 {q(gy, 16, 84)[0]:.3f}/{q(gy, 16, 84)[1]:.3f}"
        )
        out["young_gated"] = dict(n=int(gy.size), median=float(np.median(gy)))

    # ---- N dependence, the artefact check -----------------------------------------------------
    print(
        "\n[N dependence] gated sample only -- a residual trend here would be the artefact "
        "the pinned background was meant to remove"
    )
    if keep.sum() > 20:
        gk, nk = gam[keep], n[keep]
        edges = np.percentile(nk, [0, 25, 50, 75, 100])
        for i in range(4):
            s = (nk >= edges[i]) & (nk <= edges[i + 1])
            if s.any():
                print(
                    f"  N in [{edges[i]:6.0f}, {edges[i + 1]:6.0f}]: n={int(s.sum()):5d}  "
                    f"gamma median {np.median(gk[s]):.3f}"
                )
        print(f"  corr(log N, gamma) = {np.corrcoef(np.log(nk), gk)[0, 1]:+.3f}")

    # ---- the calibration this is gated against ------------------------------------------------
    if args.calib.exists():
        cal = json.loads(args.calib.read_text())
        print(f"\n[calibration] {args.calib.name}  background_free={cal.get('background_free')}")
        for c in cal["cells"]:
            print(
                f"  gamma_true={c['gamma_true']:.2f} N={c['n']:5d} "
                f"r_tot/a={c['field_over_a']:5.1f}  bias {c['bias']:+.4f} "
                f"+/- {c['bias_sem']:.4f}"
            )

    args.out.write_text(
        json.dumps(
            dict(
                n_attempted=len(rows),
                n_fitted=len(ok),
                n_converged=len(passed),
                ratio_gate=args.ratio_gate,
                n_both_gates=int(keep.sum()),
                gamma=out,
            ),
            indent=1,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
