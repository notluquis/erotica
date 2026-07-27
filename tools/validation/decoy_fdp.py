#!/usr/bin/env python3
"""Target-decoy false-discovery estimate for HDBSCAN cluster membership.

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


def pseudoprobability(X, mcs_values, seed=0):
    """Reimplementation of the package sweep, kept local so the decoy runs are cheap.

    ``probability_times`` = fraction of sweep iterations in which a source landed in
    any cluster; ``p̃`` = HDBSCAN membership probability x that fraction, exactly as
    ``Clustering.search_pseudoprobability`` builds it.
    """
    import hdbscan

    n = X.shape[0]
    in_cluster = np.zeros(n, dtype=float)
    best_prob = np.zeros(n, dtype=float)
    best_size = -1
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
        sizes = np.bincount(labels[labels >= 0])
        biggest = int(np.argmax(sizes))
        if sizes[biggest] > best_size:
            best_size = int(sizes[biggest])
            best_prob = np.where(labels == biggest, clusterer.probabilities_, 0.0)
    return best_prob * (in_cluster / len(mcs_values))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=30)
    ap.add_argument("--field-radius", type=float, default=45.0,
                    help="arcmin; stars beyond this build the decoy field")
    ap.add_argument("--mcs-step", type=int, default=10)
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
    target = pseudoprobability(np.column_stack([pmra, pmdec]), mcs_values)
    print(f"target sweep done in {time.perf_counter() - t0:.0f}s; "
          f"max p~ = {target.max():.3f}", flush=True)

    rng = np.random.default_rng(args.seed)
    decoy_counts = {p: [] for p in THRESHOLDS}
    decoy_max = []
    for it in range(args.realizations):
        idx = rng.choice(np.flatnonzero(field), size=n, replace=True)
        dx = pmra[idx] + rng.normal(0.0, e_pmra[idx])
        dy = pmdec[idx] + rng.normal(0.0, e_pmdec[idx])
        p = pseudoprobability(np.column_stack([dx, dy]), mcs_values)
        for thr in THRESHOLDS:
            decoy_counts[thr].append(int((p >= thr).sum()))
        decoy_max.append(float(p.max()))
        print(f"  decoy {it + 1:3d}/{args.realizations}  max p~={p.max():.3f}  "
              f"n(p>=0.6)={int((p >= 0.6).sum())}", flush=True)

    print(f"\n{'p~ threshold':>12s} {'target':>8s} {'decoy mean':>11s} {'decoy sd':>9s} "
          f"{'FDP':>8s}")
    results = {}
    for thr in THRESHOLDS:
        n_t = int((target >= thr).sum())
        d = np.asarray(decoy_counts[thr], dtype=float)
        fdp = d.mean() / n_t if n_t else np.nan
        results[str(thr)] = {"target": n_t, "decoy_mean": float(d.mean()),
                             "decoy_sd": float(d.std()), "fdp": float(fdp)}
        print(f"{thr:12.2f} {n_t:8d} {d.mean():11.1f} {d.std():9.1f} {fdp:8.2%}")

    print(f"\nhighest p~ reached by any decoy realization: {max(decoy_max):.3f}")
    payload = {
        "catalogue_n": int(n), "field_radius_arcmin": args.field_radius,
        "n_field": int(field.sum()), "realizations": args.realizations,
        "mcs_values": mcs_values, "seed": args.seed,
        "target_max_p": float(target.max()), "decoy_max_p": decoy_max,
        "by_threshold": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
