#!/usr/bin/env python3
r"""Target-decoy false-discovery estimate for HDBSCAN cluster membership.

WHY THIS EXISTS
---------------
``p̃ = probabilities_ × probability_times`` is an operational ranking statistic,
not a posterior (see the decision log). P01 says so explicitly. What it has never
had is an *empirical* false-discovery rate: given a cut at ``p̃ ≥ 0.6``, how many
of the selected stars would the same pipeline have selected from a field with no
cluster in it?

THE IDEA, BORROWED FROM PROTEOMICS
----------------------------------
Target-decoy FDR (Elias & Gygi 2007) estimates the false-discovery proportion by
running the identical search against a **decoy** database constructed to have the
same statistical character as the target but to contain no true positives. The
decoy hit rate at a given score threshold estimates the false-positive rate there.

The astronomical translation has to respect what the search actually uses.
``Clustering.search_pseudoprobability`` clusters in **proper-motion space only** --
sky position never enters -- so a decoy must destroy the PM overdensity while
preserving the field's PM distribution and its error structure. Permuting sky
positions would do nothing at all.

**Selecting the right branch.** The production run does NOT take the biggest cluster. Its
``summary.json`` records ``algorithm_selected_label: 1`` but ``label: 0``, with the note that "the
science summary uses the branch nearest to the 40 arcmin NGC 6383 reference proper motion"
(2.54, -1.71 mas/yr), under a size bound of ~1000. A first version of this script took the largest
cluster instead and found 14259 stars where the pipeline finds 798 -- an 18x error that measured a
field blob, not the cluster. **The decoy must use the identical rule**: the branch nearest that same
reference proper motion. A decoy "detection" is then exactly what a false positive means here -- a
field fluctuation that lands where the cluster would have been.

**Decoy construction.** Take stars beyond ``--field-radius`` arcmin from the
cluster centre, where the cluster contributes negligibly, bootstrap them up to
the full catalogue size, and jitter each draw by its own proper-motion
uncertainty. This preserves the field PM distribution, its magnitude-dependent
error structure and the sample size, while containing no compact PM overdensity.
The jitter also prevents the exact duplicates a plain bootstrap would create,
which HDBSCAN would see as zero-distance points and read as infinite density.

WHAT IS REPORTED
----------------
For each ``p̃`` threshold: the number of target stars selected, the mean number of
decoy stars selected across realizations, and their ratio -- the estimated false
discovery proportion. **An FDP above the threshold's nominal reading is the
result that matters.**

CAVEAT, STATED UP FRONT
-----------------------
This measures the false-discovery rate *of the clustering step against a
structureless field*. It does not capture contamination by other real comoving
structures along the line of sight, which a decoy built this way cannot contain
by construction. The number is a floor, not a total.

RESULT AT 40 REALIZATIONS — THE FDP IS NOT A RATE
--------------------------------------------------
Run 2026-08-02. **Reporting a single false-discovery proportion for this pipeline is misleading**,
and the 20-realization run that gave 6.1% at ``p >= 0.6`` was not converged — 40 realizations give a
*mean* of 10.5% and a *median* of 2.8%.

    threshold   FDP mean   FDP median   FDP p90   realizations finding zero decoys
    p >= 0.5     0.120       0.053       0.326     7/40
    p >= 0.6     0.105       0.028       0.308    11/40
    p >= 0.7     0.082       0.003       0.293    18/40
    p >= 0.8     0.072       0.000       0.317    27/40
    p >= 0.9     0.081       0.000       0.415    30/40

At ``p >= 0.6`` the decoy finds **nothing at all in 11 of 40 realizations**, and as many as **516
stars** in the worst — against a target count of 771.

**So this is not a steady contamination rate; it is occasional catastrophic failure.** Most of the
time the mirrored field yields no spurious cluster; roughly a quarter of the time HDBSCAN latches
onto a chance overdensity and returns a structure comparable in size to the real one. The mean is
dominated by that tail, which is why it moved 72% between run sizes while the median did not.

**How to quote it.** The median is the honest central estimate and the p90 is the honest risk. A
single mean should not be used as a contamination bound, and any argument resting on "the FDP is
X%" — including this repository's own comparison of the King background against it — must say which
statistic and acknowledge the tail.

USAGE
-----
    python tools/validation/decoy_fdp.py --realizations 30
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

CONE = Path(
    "/Users/notluquis/erotica/data/test/NGC6383/comments_paper/"
    "radius_robustness/generated/70/paperfaithful_with_clip_flags.ecsv"
)
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def load():
    t = Table.read(CONE)
    pmra = np.asarray(t["pmra"], dtype=float)
    pmdec = np.asarray(t["pmdec"], dtype=float)
    e_pmra = np.asarray(t["pmra_error"], dtype=float)
    e_pmdec = np.asarray(t["pmdec_error"], dtype=float)
    sky = SkyCoord(np.asarray(t["ra"], float) * u.deg, np.asarray(t["dec"], float) * u.deg)
    radius = CENTER.separation(sky).to(u.arcmin).value
    ok = np.isfinite(pmra) & np.isfinite(pmdec) & np.isfinite(e_pmra) & np.isfinite(e_pmdec)
    return pmra[ok], pmdec[ok], e_pmra[ok], e_pmdec[ok], radius[ok]


def pseudoprobability(X, mcs_values, reference_pm, max_members=1000, min_members=50):
    """Local reimplementation of the package sweep, matched to the production rule.

    ``probability_times`` = fraction of sweep iterations in which a source landed in
    any cluster; ``p̃`` = HDBSCAN membership probability x that fraction, as
    ``Clustering.search_pseudoprobability`` builds it.

    Branch selection follows the production run: among clusters within the size
    bound, take the one whose centroid is **nearest `reference_pm`**, not the
    largest. Taking the largest finds a field blob (14259 stars vs the pipeline's
    798) and makes the whole experiment meaningless.
    """
    import hdbscan

    n = X.shape[0]
    in_cluster = np.zeros(n, dtype=float)
    chosen_prob = np.zeros(n, dtype=float)
    best_dist = np.inf
    ref = np.asarray(reference_pm, dtype=float)
    for mcs in mcs_values:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(mcs), algorithm="best", cluster_selection_method="eom",
            allow_single_cluster=False, metric="euclidean",
            match_reference_implementation=True, gen_min_span_tree=False,
        ).fit(X)
        labels = clusterer.labels_
        in_cluster += labels >= 0
        if labels.max() < 0:
            continue
        for lab in range(labels.max() + 1):
            sel = labels == lab
            size = int(sel.sum())
            if size < min_members or size > max_members:
                continue
            dist = float(np.hypot(*(X[sel].mean(axis=0) - ref)))
            if dist < best_dist:
                best_dist = dist
                chosen_prob = np.where(sel, clusterer.probabilities_, 0.0)
    return chosen_prob * (in_cluster / len(mcs_values)), best_dist


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=30)
    ap.add_argument("--field-radius", type=float, default=45.0,
                    help="arcmin; stars beyond this build the decoy field")
    ap.add_argument("--mcs-step", type=int, default=10)
    ap.add_argument("--ref-pm", type=float, nargs=2, default=(2.54, -1.71),
                    help="reference proper motion the production run selects on (mas/yr)")
    ap.add_argument("--max-members", type=int, default=1000)
    ap.add_argument("--min-members", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20260727)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()

    pmra, pmdec, e_pmra, e_pmdec, radius = load()
    n = pmra.size
    field = radius > args.field_radius
    mcs_values = list(range(10, 300, args.mcs_step))
    print(f"catalogue N={n}, field stars (r>{args.field_radius:.0f}') = {field.sum()} "
          f"({field.mean():.1%})")
    print(f"mcs sweep: {len(mcs_values)} values, {mcs_values[0]}..{mcs_values[-1]} "
          f"step {args.mcs_step}\n", flush=True)

    t0 = time.perf_counter()
    sel = dict(reference_pm=tuple(args.ref_pm), max_members=args.max_members,
               min_members=args.min_members)
    target, target_dist = pseudoprobability(np.column_stack([pmra, pmdec]), mcs_values, **sel)
    n06 = int((target >= 0.6).sum())
    print(f"target sweep done in {time.perf_counter() - t0:.0f}s; max p~ = {target.max():.3f}; "
          f"n(p~>=0.6) = {n06}; branch centroid {target_dist:.3f} mas/yr from reference",
          flush=True)
    if not (300 < n06 < 2000):
        print(f"  *** WARNING: the production run reports ~798 members. {n06} is far from that, so "
              f"this reimplementation is not tracking the pipeline. Treat the FDP as invalid. ***",
              flush=True)

    rng = np.random.default_rng(args.seed)
    decoy_counts = {p: [] for p in THRESHOLDS}
    decoy_max = []
    for it in range(args.realizations):
        idx = rng.choice(np.flatnonzero(field), size=n, replace=True)
        dx = pmra[idx] + rng.normal(0.0, e_pmra[idx])
        dy = pmdec[idx] + rng.normal(0.0, e_pmdec[idx])
        p, _ = pseudoprobability(np.column_stack([dx, dy]), mcs_values, **sel)
        for thr in THRESHOLDS:
            decoy_counts[thr].append(int((p >= thr).sum()))
        decoy_max.append(float(p.max()))
        print(f"  decoy {it + 1:3d}/{args.realizations}  max p~={p.max():.3f}  "
              f"n(p>=0.6)={int((p >= 0.6).sum())}", flush=True)

    print(f"\n{'p~ threshold':>12s} {'target':>8s} {'d.median':>9s} {'d.mean':>8s} "
          f"{'d.p90':>8s} {'FDP med':>8s} {'FDP p90':>8s}")
    results = {}
    for thr in THRESHOLDS:
        n_t = int((target >= thr).sum())
        d = np.asarray(decoy_counts[thr], dtype=float)
        fdp = d.mean() / n_t if n_t else np.nan  # noqa: F841 (kept for the mean row)
        # The decoy distribution is heavily right-skewed (sd is typically ~2x the
        # mean), so the mean is the wrong summary. Store every realization and
        # quote the median plus a 90th percentile.
        results[str(thr)] = {
            "target": n_t, "counts": [int(x) for x in d],
            "decoy_mean": float(d.mean()), "decoy_sd": float(d.std()),
            "decoy_median": float(np.median(d)), "decoy_p90": float(np.percentile(d, 90)),
            "fdp_mean": float(fdp),
            "fdp_median": float(np.median(d) / n_t) if n_t else float("nan"),
            "fdp_p90": float(np.percentile(d, 90) / n_t) if n_t else float("nan"),
        }
        print(f"{thr:12.2f} {n_t:8d} {np.median(d):9.1f} {d.mean():8.1f} "
              f"{np.percentile(d, 90):8.1f} {np.median(d) / n_t:8.2%} "
              f"{np.percentile(d, 90) / n_t:8.2%}")

    print(f"\nhighest p~ reached by any decoy realization: {max(decoy_max):.3f}")
    payload = {
        "catalogue_n": int(n), "field_radius_arcmin": args.field_radius,
        "n_field": int(field.sum()), "realizations": args.realizations,
        "mcs_values": mcs_values, "seed": args.seed,
        "target_max_p": float(target.max()), "target_n_p06": n06,
        "reference_pm": list(args.ref_pm), "max_members": args.max_members,
        "decoy_max_p": decoy_max,
        "by_threshold": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
