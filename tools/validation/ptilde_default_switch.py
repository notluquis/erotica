#!/usr/bin/env python3
r"""What flipping ``recovery_frequency`` to ``"target"`` does to the PUBLISHED NGC 6383 sample.

WHY THIS EXISTS
---------------
EROTICA's per-star membership score is a product, ``p_tilde_i = score_i * f_i``. ``f_i`` is
``probability_times``: over a sweep of ``min_cluster_size``, the fraction of steps in which the
star was clustered **into anything**. Neither factor refers to the cluster finally selected, and
``benchmark_ptilde_decomposition.py`` / ``benchmark_tgt_selector_robustness.py`` established that
this is structural, not an artefact of one selector.

``Clustering.search_pseudoprobability(recovery_frequency="target")`` counts recovery into the
**target** instead, matching each sweep step's cluster to the finally-selected member set by
maximum Jaccard overlap. Default is still ``"any"``.

``benchmark_target_aware_fi.py`` measured it on 108 synthetic cells, 54 held out: held-out average
precision **0.5170 -> 0.8120 +- 0.0357** (paired +0.2950 +- 0.0420, 7 SE), post-isotonic resolution
0.0099 -> 0.0663, zero selections on field-only frames by either arm. It clears every adoption bar
**on synthetic data**.

This script is the last gate. The synthetic generator injects ONE cluster into a field. NGC 6383
has **real comoving neighbours** -- that is ``b1_beyond_jacobi.py``'s finding: 179 of 271 members
beyond ``r_J`` (66.1%) sit inside the total radius of catalogued comoving clusters Antalova 2 and
Theia 1645, against a 33.3% catalogue null (binomial p = 4.6e-28). A target-aware ``f_i`` may
behave differently when the competing clusters are genuine rather than field noise. The delta on
the real, published cluster must be **reported, not discovered later**.

THE FINDING THAT REFRAMES THE QUESTION -- READ BEFORE THE NUMBERS
-----------------------------------------------------------------
``recovery_frequency="target"`` is computed inside the ``if select_cluster:`` branch of
``Clustering._annotate_pseudoprobability_results``. **The production NGC 6383 recipe passes
``select_cluster=False``** (``ngc6383_radius_robustness.run_radius`` line 212; ``b1_beyond_jacobi``
``run_pipeline`` likewise). So flipping the shipped default today is a **no-op for the published
pipeline**: ``probability`` would still be ``probabilities_ * f_any``. The question is therefore
not "should the default move" but "moving it is inert until the wiring changes -- here is what it
*would* do if wired". The counterfactual is measured below regardless; an inert flip is still a
flip, and the next person to wire ``select_cluster=True`` inherits the delta.

WHAT IS MEASURED, AND THE ONE DELIBERATE DEVIATION FROM "RUN IT TWICE"
----------------------------------------------------------------------
The task specifies running the production recipe twice. **Both arms are computed from a single
sweep instead**, deliberately:

* ``recovery_frequency`` affects only post-sweep annotation -- the sweep loop of
  ``search_pseudoprobability`` is byte-identical either way. Two runs would produce the same
  ``labels_matrix`` and cost twice the compute.
* ``benchmark_target_aware_fi.py``'s bar 2 demands a **matched grid**: decimation alone costs
  0.0385 +- 0.0114 average precision, the size of a real effect. Deriving both ``f`` vectors from
  one ``labels_matrix`` object makes that control hold **by construction**, and the code asserts
  the shared provenance rather than assuming it.

The sweep matrix is not stored on the instance (92 MB at 290 steps, deliberately dropped), so it
is captured with a subclass that overrides ``_annotate_pseudoprobability_results`` to record its
``sweep_labels`` argument and delegate. **No package source is modified.**

The branch label is selected ONCE, by production's rule (``ngc_like_label``: the label whose
``p_tilde_any >= 0.5`` members sit nearest the reference proper motion), and **both arms use that
same branch**. Selecting it per-arm would be circular -- ``f_target`` is defined against the branch
-- and it is also what the package does: ``selected_label`` comes from the condensed tree and never
sees ``p_tilde``. Whether ``p_tilde_target`` would have chosen a different branch is reported as a
sensitivity, not used.

Everything downstream of the score is b1's verbatim recipe, per arm: ``preclip = branch AND
p_tilde >= 0.5``; a 2-sigma ``sigma_clip`` on parallax with a ``histogram_mode`` centre re-derived
from that preclip set; ``members = branch AND p_tilde >= 0.6 AND clip``. A second variant holds the
**published** clip band ``[0.7839647672, 1.0432678929]`` fixed, so "the score moved membership" is
separable from "the clip band moved under it".

FIDELITY GATE -- A FORK IN THE REPORT, BOTH BRANCHES PRE-REGISTERED
--------------------------------------------------------------------
Two gates, in order:

* **G0 (fork validity).** This file cannot call ``b1.run_pipeline`` verbatim -- the capturing
  subclass forces a copy of its body -- so the copy is validated against b1's own recorded
  ``stage0_fidelity.step10`` oracle at ``--mcs-step 10``: ``branch_label`` 1, ``branch_n`` 791,
  ``preclip_p05`` 746, ``best_mcs`` 110, ``clip_low`` 0.796869742599954, ``clip_high``
  0.030922626468977 (sic: 1.030922626468977), ``n_members`` 567. Exact equality on the integers,
  1e-12 on the bounds. **If G0 fails the fork is not b1's recipe and nothing below is about the
  published pipeline.** Run G0 before spending the 290-step sweep.
* **G1 (production fidelity).** The any-arm at production's real grid ``range(10, 300)`` must
  reproduce the published sample: 628 members and clip bounds ``[0.7839647672, 1.0432678929]``.
    - **G1 passes** -> the target-arm delta is a statement about the published sample and its
      count is quoted against 627-628 plainly.
    - **G1 fails** -> the delta is measured on a *near-miss* pipeline. That caveat leads the
      headline, exactly as b1's own stage-0 rule requires, and no count is quoted against 628
      without it. The comparison between arms remains valid (same pipeline both sides); only the
      claim "this is what happens to the published 628" is withdrawn.

PRE-REGISTERED: WHAT WOULD MAKE MOVING THE DEFAULT UNACCEPTABLE
----------------------------------------------------------------
Fixed before any number was looked at. Numeric, so it cannot be argued after the fact.

  **U1 -- membership collapse.** ``n_target < 314`` (below half the published 628). The score and
      the threshold would have decoupled so far that ``p_tilde >= 0.6`` no longer names anything
      resembling the published sample, and the published threshold could not be carried over.
      *Soft band:* ``314 <= n_target < 565`` (50-90% of 628) is NOT disqualifying, but the report
      must state that the threshold and the score changed together and 0.6 needs recalibrating
      before the target arm is used with the published cut.

  **U2 -- ``R_c`` leaves the published band.** The band is **1.32-1.50'** and its provenance was
      checked before use (``docs/design-notes/king_model_validity.md`` L100-117): it spans
      *likelihood and tidal-prior* variation on the SAME published member list -- binned 1.384 +-
      0.039, unbinned scale-free 1.324 +- 0.209, unbinned Jacobi +-20% 1.497 +- 0.214. Comparing an
      unbinned target-arm number against the binned 1.384 would be a category error, so the
      **primary comparison is unbinned-any vs unbinned-target**, paired, same likelihood, same
      ``SamplingConfig``, same seed. U2 fires only if **both**:
        (a) ``|R_c_target - R_c_any| > sd(R_c_any)`` -- the shift exceeds the any-arm posterior
            width, i.e. it is resolvable at all; and
        (b) the target-arm posterior median falls outside 1.32-1.50'.
      Both are required because the unbinned posterior SD is ~0.21', **wider than the 0.18'-wide
      band itself**: a band exit inside that noise is not a detection, and reporting one would
      manufacture a false alarm against a published paper. If (b) holds but (a) does not, the
      report says "not resolvable at this precision" and does not raise a correction.

  **U3 -- the B1 conclusion breaks.** The attributed fraction of members beyond ``r_J`` falls
      below **0.20** on the target member list. 0.20 is not chosen here: it is the threshold in
      ``b1_beyond_jacobi.verdict`` at which the NOT-NGC-6383 downgrade fires. Below it, b1's
      headline no longer holds on the new sample and the switch would silently rewrite a published
      conclusion.

  **U4 -- no verdict.** Any sampler gate fails: ``r_hat >= 1.01``, ``ess_bulk <= 400``, or
      ``divergences > 0`` on either arm. Then no ``R_c`` verdict is issued at all -- U2 is reported
      as UNDETERMINED rather than as a null.

  **U5 -- the drop is mechanical, not target-aware.** ``n_steps_matched / n_steps < 0.5``. Steps
      where no cluster overlaps the branch contribute zero to ``f_target`` while still counting in
      its denominator, so a sparse match manufactures a member-count drop that has nothing to do
      with target-awareness. Under U5 the delta is a **grid artefact** and says nothing about the
      proposal; the recommendation is "do not move it, and fix the matching first".
      ``mean_matched_jaccard`` and ``mean_matched_size_ratio`` are reported for the opposite
      failure: ``allow_single_cluster=True`` with ``leaf`` can give a step whose one giant cluster
      swallows the branch, in which case ``f_target ~ f_any`` for a bad reason.

PRE-REGISTERED: WHAT WOULD MAKE IT A *FEATURE*
-----------------------------------------------
  **P1 -- it drops the second cluster.** The members ``f_target`` removes are preferentially those
      b1 attributed to Antalova 2 / Theia 1645. This is tested as a **rate difference conditioned
      on radius**, never as a raw count: ``f_target`` plausibly drops weakly-clustered outer stars
      for a purely mechanical reason, and the neighbours also live outer, so an unconditioned count
      reproduces b1's own confound in a new form. Reported as
      ``P(dropped | inside a neighbour circle, r > r_J)`` vs ``P(dropped | outside, r > r_J)``,
      stratified by b1's annuli with a Mantel-Haenszel common odds ratio, plus a Fisher exact test
      on the pooled 2x2. Only an inside-circle excess that **survives radius conditioning**
      supports P1.

NEGATIVE CONTROLS AND CHECKS THAT MUST BE ABLE TO FAIL
-------------------------------------------------------
* **Monotonicity.** ``f_target <= f_any`` pointwise, hence ``n_target <= n_any`` at fixed
  threshold and ``members_target subset-of members_any``. This holds **by construction**
  (``col == best`` implies ``col != -1``, and both divide by the same ``n_steps``), so as an
  assertion on real data it *cannot fail* -- which this directory's rule forbids. It is therefore
  mutation-tested in ``--selftest``: one ``f_target`` entry is nudged upward by 1e-12 and the check
  must fire. Same for the subset check.
* **Product identity.** ``score * f_any`` must equal the package's own ``data["probability"]`` to
  1e-12. If it does not, this file is not reproducing the shipped product and the whole comparison
  is void.
* **Shared provenance.** Both ``f`` vectors are asserted to derive from the same ``labels_matrix``
  object (``id()`` recorded), which is what makes the matched-grid control hold by construction.
* **Sweep-grid null.** ``f_any`` recomputed from the captured matrix must equal the package's
  ``probability_times`` exactly -- proof the capture is the real matrix and not a copy that drifted.

ENVIRONMENT
-----------
Two environments, because neither has everything:

    stages 0 / sweep / neighbours / report   /Users/notluquis/miniforge3/envs/erotica-bench/bin/python
                                             (editable ``erotica`` + hdbscan; NO pymc)
    stage king                               PYTHONPATH=/Users/notluquis/erotica \
                                             /Users/notluquis/miniforge3/envs/cosmic/bin/python
                                             (pymc 6.0.1 + arviz 1.2.0; ``erotica`` not installed)

USAGE
-----
    python tools/validation/ptilde_default_switch.py --selftest
    python tools/validation/ptilde_default_switch.py --stage gate0          # ~35 s, validates the fork
    python tools/validation/ptilde_default_switch.py --stage sweep          # ~6 min, the real grid
    python tools/validation/ptilde_default_switch.py --stage neighbours     # live VizieR
    PYTHONPATH=... cosmic/bin/python .../ptilde_default_switch.py --stage king
    python tools/validation/ptilde_default_switch.py --report

Per-star arrays go to ``ptilde_default_switch.npz`` (gitignored); the JSON sidecar carries
quantiles and counts only and is checked against the 500 KB budget on every write.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="erotica-mpl-"))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b1_beyond_jacobi as b1  # noqa: E402  -- constants + pure helpers, no side effects on import

OUT = Path(__file__).with_suffix(".json")
NPZ = Path(__file__).with_name("ptilde_default_switch.npz")

R_J = b1.R_J
CONE_RADIUS = b1.CONE_RADIUS
EDGES = b1.EDGES
CENTER_RA, CENTER_DEC = b1.CENTER_RA, b1.CENTER_DEC

# The published sample, and the clip band production derived for it.
PUBLISHED_N = 628
PUBLISHED_CLIP = (0.7839647672, 1.0432678929)
# docs/design-notes/king_model_validity.md L117 -- across likelihood AND tidal prior, same list.
RC_BAND = (1.32, 1.50)
RC_PUBLISHED = 1.384

# b1_beyond_jacobi.json stage0_fidelity.step10 -- the oracle for G0.
G0_ORACLE = {
    "branch_label": 1,
    "branch_n": 791,
    "preclip_p05": 746,
    "best_mcs": 110,
    "n_members": 567,
    "clip_low": 0.796869742599954,
    "clip_high": 1.030922626468977,
}

# Pre-registered thresholds (see docstring).
U1_HARD = 314  # < half of 628
U1_SOFT = 565  # < 90% of 628 -> threshold needs recalibrating
U3_ATTRIBUTION_FLOOR = 0.20
U5_MATCH_FLOOR = 0.50
GATE_RHAT, GATE_ESS, GATE_DIV = 1.01, 400.0, 0


# ------------------------------------------------------------------ sidecar plumbing
def jsonable(value: Any) -> Any:
    return b1.jsonable(value)


def load_sidecar() -> dict:
    return json.loads(OUT.read_text()) if OUT.exists() else {}


def save_sidecar(payload: dict) -> None:
    text = json.dumps(jsonable(payload), indent=1)
    kb = len(text.encode()) / 1024
    if kb > 500:
        raise RuntimeError(f"sidecar is {kb:.0f} KB, over the 500 KB budget")
    OUT.write_text(text)


def checkpoint(stage: str, key: str, value: Any) -> dict:
    """Write one result immediately. A killed run keeps everything already measured."""
    payload = load_sidecar()
    payload.setdefault(stage, {})[key] = value
    payload["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_sidecar(payload)
    return payload


def quantiles(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {"n": 0}
    qs = np.percentile(finite, [0, 25, 50, 75, 90, 95, 99, 100])
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "min": float(qs[0]),
        "q25": float(qs[1]),
        "median": float(qs[2]),
        "q75": float(qs[3]),
        "p90": float(qs[4]),
        "p95": float(qs[5]),
        "p99": float(qs[6]),
        "max": float(qs[7]),
        "frac_exactly_zero": float(np.mean(finite == 0.0)),
        "frac_gt_0.5": float(np.mean(finite > 0.5)),
        "frac_ge_0.6": float(np.mean(finite >= 0.6)),
    }


# ------------------------------------------------------------------ the fork
def _capturing_clustering_cls():
    """Subclass that keeps the sweep label matrix ``search_pseudoprobability`` throws away.

    The matrix is (n_sources x n_steps) int32 -- 51 MB for this catalogue at 290 steps -- and the
    package deliberately does not store it (``clustering.py`` L508-511). It is the only object from
    which BOTH ``f`` vectors can be derived, and deriving them from one object is what makes the
    matched-grid control hold by construction instead of by assumption. Overriding the annotation
    hook is the smallest seam that reaches it without touching package source.
    """
    from erotica.core import Clustering

    class _Capturing(Clustering):
        captured_sweep_labels = None

        def _annotate_pseudoprobability_results(self, **kwargs):
            self.captured_sweep_labels = kwargs.get("sweep_labels")
            return super()._annotate_pseudoprobability_results(**kwargs)

    return _Capturing, Clustering


def _clip_bounds(parallax: np.ndarray, preclip: np.ndarray) -> tuple[float, float]:
    """b1's verbatim parallax clip: 2-sigma, ``histogram_mode`` centre, units stripped.

    Units are stripped first: the ECSV round-trip gives ``parallax`` a mas unit the production
    in-memory table does not carry, and astropy's ``sigma_clip`` then compares a Quantity centre
    against a dimensionless std and raises. Numerically identical -- G0 checks the bounds against
    b1's recorded values to 1e-12.
    """
    from astropy.stats import sigma_clip

    from erotica.analysis import histogram_mode

    def mode(values, axis=None):
        if axis is not None:
            return np.apply_along_axis(histogram_mode, axis, values)
        return histogram_mode(values)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, lo, hi = sigma_clip(
            parallax[preclip], sigma=2, cenfunc=mode, stdfunc="std", return_bounds=True
        )
    return float(np.asarray(lo).reshape(-1)[0]), float(np.asarray(hi).reshape(-1)[0])


def arm(branch: np.ndarray, ptilde: np.ndarray, parallax: np.ndarray, label: str) -> dict:
    """b1's membership recipe applied to one score vector. Re-derived clip AND published clip."""
    preclip = branch & (ptilde >= 0.5)
    lo, hi = _clip_bounds(parallax, preclip)
    clip = (parallax >= lo) & (parallax <= hi)
    members = branch & (ptilde >= 0.6) & clip

    pub_clip = (parallax >= PUBLISHED_CLIP[0]) & (parallax <= PUBLISHED_CLIP[1])
    members_pubclip = branch & (ptilde >= 0.6) & pub_clip
    return {
        "label": label,
        "branch_n": int(branch.sum()),
        "preclip_p05": int(preclip.sum()),
        "clip_low": lo,
        "clip_high": hi,
        "n_members": int(members.sum()),
        "n_members_published_clip": int(members_pubclip.sum()),
        "n_p06_preclip": int((branch & (ptilde >= 0.6)).sum()),
        "_members": members,
        "_members_pubclip": members_pubclip,
        "_preclip": preclip,
    }


def run_both_arms(table, mcs_step: int) -> dict:
    """One sweep; both ``f`` vectors; both memberships. Returns a dict with per-star arrays."""
    Capturing, Clustering = _capturing_clustering_cls()

    work = table.copy(copy_data=True)
    clust = Capturing(work, None)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clust.search_pseudoprobability(
            columns=["pmra", "pmdec"],
            min_cluster_size_samples=range(10, 300, mcs_step),
            probability_threshold=0.5,
            min_cluster_members=200,
            max_cluster_members=1000,
            select_cluster=False,
            hdbscan_kwargs={
                "cluster_selection_method": "leaf",
                "allow_single_cluster": True,
                "match_reference_implementation": False,
                "core_dist_n_jobs": 1,
            },
        )
    seconds = time.perf_counter() - t0

    sweep = clust.captured_sweep_labels
    if sweep is None:
        raise RuntimeError("sweep_labels capture failed -- the annotation hook did not fire")
    data = clust.data
    labels = np.asarray(data["cluster_hdbscan"], int)
    score = np.asarray(data["probability_hdbscan"], float)
    f_any = np.asarray(data["probability_times"], float)
    shipped = np.asarray(data["probability"], float)

    # --- checks that must be able to fail ------------------------------------------------
    # 1. the capture is the REAL matrix: f_any recomputed from it must equal the package's.
    f_any_recomputed = np.mean(sweep != -1, axis=1)
    if not np.allclose(f_any_recomputed, f_any, rtol=0, atol=1e-12):
        raise RuntimeError("captured sweep matrix does not reproduce probability_times")
    # 2. this file reproduces the shipped product exactly.
    if not np.allclose(score * f_any, shipped, rtol=0, atol=1e-12):
        raise RuntimeError("score * f_any != data['probability'] -- not the shipped product")

    # Branch: production's rule, selected ONCE on p_tilde_any, shared by both arms.
    label, pm_dist = b1.ngc_like_label(data, labels, shipped)
    if label < 0:
        raise RuntimeError("no admissible branch")
    branch = labels == label

    f_tgt, info = Clustering._target_recovery_frequency(sweep, branch)
    ptilde_any = score * f_any
    ptilde_tgt = score * f_tgt

    # 3. monotonicity -- by construction, hence mutation-tested in --selftest, not here alone.
    mono = monotonicity_check(f_any, f_tgt)

    plx = np.asarray(data["parallax"], float)
    a_any = arm(branch, ptilde_any, plx, "any")
    a_tgt = arm(branch, ptilde_tgt, plx, "target")

    # 4. subset -- members_target must nest inside members_any at fixed threshold.
    subset = subset_check(a_any["_members"], a_tgt["_members"])

    # Sensitivity: would p_tilde_target have picked a different branch? Reported, not used.
    label_tgt, pm_dist_tgt = b1.ngc_like_label(data, labels, ptilde_tgt)

    return {
        "seconds": seconds,
        "n_grid": len(range(10, 300, mcs_step)),
        "mcs_step": mcs_step,
        "branch_label": int(label),
        "branch_pm_dist": float(pm_dist),
        "branch_n": int(branch.sum()),
        "best_mcs": int((clust.pseudoprobability_selected_ or {}).get("min_cluster_size", -1)),
        "target_recovery_info": info,
        "shared_labels_matrix_id": int(id(sweep)),
        "monotonicity": mono,
        "subset_target_in_any": subset,
        "branch_sensitivity_under_target_score": {
            "label": int(label_tgt),
            "pm_dist": float(pm_dist_tgt),
            "same_as_any_arm": bool(label_tgt == label),
        },
        "any": a_any,
        "target": a_tgt,
        "_branch": branch,
        "_score": score,
        "_f_any": f_any,
        "_f_target": f_tgt,
        "_ptilde_any": ptilde_any,
        "_ptilde_target": ptilde_tgt,
        "_labels": labels,
    }


def monotonicity_check(f_any: np.ndarray, f_tgt: np.ndarray) -> dict:
    """``f_target <= f_any`` pointwise. TRUE BY CONSTRUCTION -- see --selftest for the mutation."""
    viol = np.flatnonzero(f_tgt > f_any + 1e-12)
    return {
        "holds": bool(viol.size == 0),
        "n_violations": int(viol.size),
        "max_excess": float((f_tgt - f_any).max()),
        "note": "holds by construction (col==best implies col!=-1, same denominator); "
        "mutation-tested in --selftest because a check that cannot fail is worse than none",
    }


def subset_check(members_any: np.ndarray, members_tgt: np.ndarray) -> dict:
    extra = np.flatnonzero(members_tgt & ~members_any)
    return {
        "holds": bool(extra.size == 0),
        "n_in_target_not_in_any": int(extra.size),
        "note": "consequence of monotonicity at a FIXED threshold; the re-derived parallax clip "
        "can in principle break it, which is why it is measured rather than asserted",
    }


# ------------------------------------------------------------------ stage: G0
def stage_gate0(mcs_step: int = 10) -> dict:
    """Validate this file's fork of b1's recipe against b1's own recorded step-10 oracle."""
    t = b1.cone()
    res = run_both_arms(t, mcs_step)
    got = {
        "branch_label": res["branch_label"],
        "branch_n": res["branch_n"],
        "preclip_p05": res["any"]["preclip_p05"],
        "best_mcs": res["best_mcs"],
        "n_members": res["any"]["n_members"],
        "clip_low": res["any"]["clip_low"],
        "clip_high": res["any"]["clip_high"],
    }
    mismatches = []
    for k, want in G0_ORACLE.items():
        have = got[k]
        ok = abs(have - want) <= 1e-12 if isinstance(want, float) else have == want
        if not ok:
            mismatches.append({"field": k, "b1_oracle": want, "this_fork": have})
    out = {
        "oracle_source": "b1_beyond_jacobi.json :: stage0_fidelity.step10",
        "oracle": G0_ORACLE,
        "measured": got,
        "mismatches": mismatches,
        "gate_passed": bool(not mismatches),
        "seconds": round(res["seconds"], 1),
        # measured at step 10 too, so the step-1 numbers can be read against a decimated baseline
        "target_recovery_info": res["target_recovery_info"],
        "n_members_target": res["target"]["n_members"],
    }
    checkpoint("gate0_fork_fidelity", f"step{mcs_step}", out)
    return out


# ------------------------------------------------------------------ stage: sweep
def stage_sweep(mcs_step: int) -> dict:
    t = b1.cone()
    r = b1.radius_of(t)
    prod = np.asarray(t["paper_reference_p06"], bool)
    res = run_both_arms(t, mcs_step)

    m_any = res["any"]["_members"]
    m_tgt = res["target"]["_members"]
    enter = m_tgt & ~m_any
    leave = m_any & ~m_tgt
    inter = int((m_any & m_tgt).sum())
    union = int((m_any | m_tgt).sum())

    # G1: does the any-arm reproduce production?
    g1 = {
        "production_n": int(prod.sum()),
        "any_arm_n": int(m_any.sum()),
        "count_matches": bool(int(m_any.sum()) == PUBLISHED_N),
        "clip_low": res["any"]["clip_low"],
        "clip_high": res["any"]["clip_high"],
        "published_clip": list(PUBLISHED_CLIP),
        "clip_matches": bool(
            abs(res["any"]["clip_low"] - PUBLISHED_CLIP[0]) < 1e-9
            and abs(res["any"]["clip_high"] - PUBLISHED_CLIP[1]) < 1e-9
        ),
        "jaccard_vs_production": (
            int((m_any & prod).sum()) / int((m_any | prod).sum()) if (m_any | prod).any() else 0.0
        ),
    }
    g1["gate_passed"] = bool(g1["count_matches"] and g1["clip_matches"])

    def radial(mask):
        return {
            "n": int(mask.sum()),
            "n_beyond_rJ": int((r[mask] > R_J).sum()),
            "fraction_beyond_rJ": float((r[mask] > R_J).mean()) if mask.any() else None,
            "median_radius": float(np.median(r[mask])) if mask.any() else None,
            "counts_by_annulus": b1.annulus_counts(r, mask).tolist(),
        }

    ni = res["target_recovery_info"]
    match_rate = ni["n_steps_matched"] / max(ni["n_steps"], 1)

    n_tgt = int(m_tgt.sum())
    out = {
        "mcs_step": mcs_step,
        "n_grid": res["n_grid"],
        "seconds": round(res["seconds"], 1),
        "branch_label": res["branch_label"],
        "branch_n": res["branch_n"],
        "branch_pm_dist": res["branch_pm_dist"],
        "best_mcs": res["best_mcs"],
        "branch_sensitivity_under_target_score": res["branch_sensitivity_under_target_score"],
        "G1_production_fidelity": g1,
        # ---- 1. membership -------------------------------------------------------------
        "membership": {
            "n_any": int(m_any.sum()),
            "n_target": n_tgt,
            "delta": n_tgt - int(m_any.sum()),
            "retained_fraction": n_tgt / max(int(m_any.sum()), 1),
            "intersection": inter,
            "union": union,
            "jaccard_any_vs_target": inter / union if union else 0.0,
            "n_enter": int(enter.sum()),
            "n_leave": int(leave.sum()),
            "vs_published_628": {
                "n_target_minus_628": n_tgt - PUBLISHED_N,
                "n_target_over_628": n_tgt / PUBLISHED_N,
            },
            "any": radial(m_any),
            "target": radial(m_tgt),
            "leavers": radial(leave),
            "enterers": radial(enter),
            "published_clip_variant": {
                "n_any": res["any"]["n_members_published_clip"],
                "n_target": res["target"]["n_members_published_clip"],
            },
            "p06_before_clip": {
                "n_any": res["any"]["n_p06_preclip"],
                "n_target": res["target"]["n_p06_preclip"],
            },
        },
        # ---- 2. the score itself -------------------------------------------------------
        "score_distributions": {
            "all_sources": {
                "ptilde_any": quantiles(res["_ptilde_any"]),
                "ptilde_target": quantiles(res["_ptilde_target"]),
                "f_any": quantiles(res["_f_any"]),
                "f_target": quantiles(res["_f_target"]),
            },
            "branch_only": {
                "ptilde_any": quantiles(res["_ptilde_any"][res["_branch"]]),
                "ptilde_target": quantiles(res["_ptilde_target"][res["_branch"]]),
                "f_any": quantiles(res["_f_any"][res["_branch"]]),
                "f_target": quantiles(res["_f_target"][res["_branch"]]),
                "score_hdbscan": quantiles(res["_score"][res["_branch"]]),
            },
            "published_members_only": {
                "ptilde_any": quantiles(res["_ptilde_any"][prod]),
                "ptilde_target": quantiles(res["_ptilde_target"][prod]),
            },
        },
        "monotonicity": res["monotonicity"],
        "subset_target_in_any": res["subset_target_in_any"],
        "shared_labels_matrix_id": res["shared_labels_matrix_id"],
        # ---- U5 mechanism ---------------------------------------------------------------
        "target_recovery_info": ni,
        "U5_match_rate": match_rate,
        "U5_mechanical": bool(match_rate < U5_MATCH_FLOOR),
        "clip": {
            "any": [res["any"]["clip_low"], res["any"]["clip_high"]],
            "target": [res["target"]["clip_low"], res["target"]["clip_high"]],
        },
    }
    np.savez_compressed(
        NPZ,
        radius=r,
        branch=res["_branch"],
        labels=res["_labels"],
        score=res["_score"],
        f_any=res["_f_any"],
        f_target=res["_f_target"],
        ptilde_any=res["_ptilde_any"],
        ptilde_target=res["_ptilde_target"],
        members_any=m_any,
        members_target=m_tgt,
        members_production=prod,
        ra=np.asarray(t["ra"], float),
        dec=np.asarray(t["dec"], float),
        parallax=np.asarray(t["parallax"], float),
        mcs_step=np.array([mcs_step]),
    )
    checkpoint("sweep", f"step{mcs_step}", out)
    return out


# ------------------------------------------------------------------ stage: neighbours
def stage_neighbours() -> dict:
    """P1: are the leavers preferentially the stars b1 attributed to a comoving neighbour?

    Conditioned on radius throughout. ``f_target`` plausibly drops weakly-clustered outer stars
    for a mechanical reason and Antalova 2 / Theia 1645 also live outer, so an unconditioned count
    reproduces b1's confound in a new form.
    """
    from scipy import stats as st

    if not NPZ.exists():
        raise RuntimeError("run --stage sweep first")
    z = np.load(NPZ)
    r = z["radius"]
    ra, dec = z["ra"], z["dec"]
    m_any, m_tgt = z["members_any"], z["members_target"]
    leave = m_any & ~m_tgt

    neighbours, meta = b1.hunt_neighbours()
    cosd = np.cos(np.deg2rad(CENTER_DEC))
    inside = np.zeros(len(r), bool)
    per = []
    for nb in neighbours:
        d = np.hypot((ra - nb["ra"]) * cosd, dec - nb["dec"]) * 60.0
        radius = nb["rtot_arcmin"] if np.isfinite(nb["rtot_arcmin"]) else nb["rJ_arcmin"]
        circ = d < radius
        inside |= circ
        outer_any = m_any & (r > R_J) & (r <= CONE_RADIUS)
        per.append(
            {
                "name": nb["name"],
                "separation_arcmin": nb["separation_arcmin"],
                "matched_radius_arcmin": radius,
                "any_arm_outer_inside": int((outer_any & circ).sum()),
                "target_arm_outer_inside": int((m_tgt & (r > R_J) & circ).sum()),
                "leavers_inside": int((leave & (r > R_J) & circ).sum()),
            }
        )

    def attribution(mask):
        outer = mask & (r > R_J) & (r <= CONE_RADIUS)
        n_out = int(outer.sum())
        att = int((outer & inside).sum())
        return {
            "n_outer": n_out,
            "n_attributed": att,
            "fraction_attributed": att / n_out if n_out else None,
        }

    att_any, att_tgt = attribution(m_any), attribution(m_tgt)

    # ---- P1: rate difference conditioned on radius --------------------------------------
    outer = m_any & (r > R_J) & (r <= CONE_RADIUS)
    a = int((outer & inside & leave).sum())  # inside, dropped
    b = int((outer & inside & ~leave).sum())  # inside, kept
    c = int((outer & ~inside & leave).sum())  # outside, dropped
    d = int((outer & ~inside & ~leave).sum())  # outside, kept
    fisher = st.fisher_exact([[a, b], [c, d]]) if (a + b) and (c + d) else (float("nan"),) * 2

    # Mantel-Haenszel across b1's annuli beyond r_J -- the radius-conditioned version.
    strata, num, den = [], 0.0, 0.0
    for lo, hi in zip(EDGES[:-1], EDGES[1:], strict=True):
        if lo < R_J:
            continue
        s = outer & (r >= lo) & (r < hi)
        if not s.any():
            continue
        aa = int((s & inside & leave).sum())
        bb = int((s & inside & ~leave).sum())
        cc = int((s & ~inside & leave).sum())
        dd = int((s & ~inside & ~leave).sum())
        n = aa + bb + cc + dd
        if n == 0:
            continue
        num += aa * dd / n
        den += bb * cc / n
        strata.append(
            {
                "annulus": [float(lo), float(hi)],
                "inside_dropped": aa,
                "inside_kept": bb,
                "outside_dropped": cc,
                "outside_kept": dd,
                "drop_rate_inside": aa / (aa + bb) if (aa + bb) else None,
                "drop_rate_outside": cc / (cc + dd) if (cc + dd) else None,
            }
        )
    mh_or = num / den if den > 0 else None

    out = {
        **meta,
        "neighbours": per,
        "attribution_any_arm": att_any,
        "attribution_target_arm": att_tgt,
        "b1_published_attribution": 0.6605166051660517,
        "U3_attribution_floor": U3_ATTRIBUTION_FLOOR,
        "U3_breached": bool(
            att_tgt["fraction_attributed"] is not None
            and att_tgt["fraction_attributed"] < U3_ATTRIBUTION_FLOOR
        ),
        "P1_pooled_2x2": {
            "inside_dropped": a,
            "inside_kept": b,
            "outside_dropped": c,
            "outside_kept": d,
            "drop_rate_inside": a / (a + b) if (a + b) else None,
            "drop_rate_outside": c / (c + d) if (c + d) else None,
            "odds_ratio": float(fisher[0]),
            "fisher_p": float(fisher[1]),
        },
        "P1_radius_conditioned": {
            "strata": strata,
            "mantel_haenszel_odds_ratio": mh_or,
            "note": "OR > 1 means the leavers are preferentially inside a catalogued comoving "
            "neighbour AT MATCHED RADIUS -- the only form of P1 that is not b1's own confound",
        },
    }
    checkpoint("neighbours", "result", out)
    return out


# ------------------------------------------------------------------ stage: king
def stage_king(draws: int = 2000, seed: int = 20260804) -> dict:
    """Unbinned King refit on both member lists. Needs pymc -> run under the ``cosmic`` env."""
    import arviz as az
    import pandas as pd
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    from erotica.analysis.inference import SamplingConfig
    from erotica.analysis.structure import king_unbinned

    if not NPZ.exists():
        raise RuntimeError("run --stage sweep first")
    z = np.load(NPZ)
    centre = SkyCoord(CENTER_RA * u.deg, CENTER_DEC * u.deg)
    sky = SkyCoord(z["ra"] * u.deg, z["dec"] * u.deg)
    # Great-circle separation, matching king_unbinned_delta.py (the fit the band came from).
    # b1's tangent-plane radius is used only for annuli, never for the likelihood.
    sep = centre.separation(sky).to(u.arcmin).value

    results = {}
    for name in ("members_any", "members_target", "members_production"):
        mask = z[name]
        rad = sep[mask]
        keep = rad[np.isfinite(rad) & (rad > 0) & (rad <= CONE_RADIUS)]
        t0 = time.perf_counter()
        res = king_unbinned(
            keep,
            field_radius=CONE_RADIUS,
            sampling=SamplingConfig(
                draws=draws, tune=1000, chains=4, random_seed=seed, progressbar=False
            ),
        )
        smry = az.summary(res["king_trace"], var_names=["k", "b", "R_c", "R_t"])
        rec = {
            "n_stars": int(len(keep)),
            "dropped_outside_field": int(len(rad) - len(keep)),
            "seconds": round(time.perf_counter() - t0, 1),
            "r_hat_max": float(pd.to_numeric(smry["r_hat"], errors="coerce").max()),
            "ess_bulk_min": float(pd.to_numeric(smry["ess_bulk"], errors="coerce").min()),
            "divergences": int(res["king_trace"].sample_stats["diverging"].values.sum()),
        }
        post = res["king_trace"].posterior
        for p in ("k", "b", "R_c", "R_t"):
            v = np.asarray(post[p].values).ravel()
            rec[p] = {
                "median": float(np.median(v)),
                "sd": float(np.std(v)),
                "hdi_3": float(np.percentile(v, 3)),
                "hdi_97": float(np.percentile(v, 97)),
            }
        rec["gates_passed"] = bool(
            rec["r_hat_max"] < GATE_RHAT
            and rec["ess_bulk_min"] > GATE_ESS
            and rec["divergences"] == GATE_DIV
        )
        results[name] = rec
        checkpoint("king", name, rec)  # per-fit checkpoint: a killed run keeps what it measured
        print(
            f"  {name:20s} N={rec['n_stars']:5d}  R_c={rec['R_c']['median']:.3f}"
            f"+/-{rec['R_c']['sd']:.3f}  r_hat={rec['r_hat_max']:.3f} "
            f"ess={rec['ess_bulk_min']:.0f} div={rec['divergences']} ({rec['seconds']}s)",
            flush=True,
        )

    verdict = king_verdict(results)
    checkpoint("king", "verdict", verdict)
    return {"fits": results, "verdict": verdict}


def king_verdict(fits: dict) -> dict:
    """U2, applied as written. Both conditions required; U4 pre-empts."""
    a, t = fits.get("members_any"), fits.get("members_target")
    if not a or not t:
        return {"verdict": "PENDING", "reasons": ["both arms not yet fitted"]}
    if not (a["gates_passed"] and t["gates_passed"]):
        return {
            "verdict": "UNDETERMINED (U4)",
            "reasons": ["a sampler gate failed; no R_c verdict is issued"],
            "gates": {k: v["gates_passed"] for k, v in fits.items()},
        }
    rc_a, sd_a = a["R_c"]["median"], a["R_c"]["sd"]
    rc_t = t["R_c"]["median"]
    shift = rc_t - rc_a
    resolvable = abs(shift) > sd_a
    outside = not (RC_BAND[0] <= rc_t <= RC_BAND[1])
    if outside and resolvable:
        call = "CORRECTION-REQUIRED"
        why = (
            f"R_c moved {shift:+.3f}' (|shift| > any-arm posterior SD {sd_a:.3f}') and the "
            f"target-arm median {rc_t:.3f}' sits OUTSIDE the published band "
            f"{RC_BAND[0]}-{RC_BAND[1]}'. This is a correction to a published paper."
        )
    elif outside:
        call = "OUTSIDE-BAND-BUT-NOT-RESOLVABLE"
        why = (
            f"target-arm median {rc_t:.3f}' is outside {RC_BAND[0]}-{RC_BAND[1]}' but the shift "
            f"{shift:+.3f}' is inside the any-arm posterior SD {sd_a:.3f}'. Not a detection; "
            f"reporting it as a correction would manufacture a false alarm."
        )
    else:
        call = "NULL-PROTECTS-PUBLISHED"
        why = (
            f"target-arm R_c {rc_t:.3f}' stays inside {RC_BAND[0]}-{RC_BAND[1]}' "
            f"(shift {shift:+.3f}', any-arm SD {sd_a:.3f}'). The published value is unaffected."
        )
    return {
        "verdict": call,
        "reasons": [why],
        "R_c_any": [rc_a, sd_a],
        "R_c_target": [rc_t, t["R_c"]["sd"]],
        "R_c_production_list": (
            [fits["members_production"]["R_c"]["median"], fits["members_production"]["R_c"]["sd"]]
            if "members_production" in fits
            else None
        ),
        "shift": shift,
        "resolvable": resolvable,
        "band": list(RC_BAND),
        "published_binned_R_c": RC_PUBLISHED,
        "band_provenance": "king_model_validity.md L100-117: spans likelihood AND tidal prior on "
        "the SAME published member list. The primary comparison here is unbinned-any vs "
        "unbinned-target, paired, so the likelihood is not a confound.",
    }


# ------------------------------------------------------------------ overall recommendation
def recommendation(payload: dict) -> dict:
    sw = payload.get("sweep", {})
    key = "step1" if "step1" in sw else (sorted(sw)[0] if sw else None)
    s = sw.get(key, {}) if key else {}
    if not s:
        return {"recommendation": "PENDING", "reasons": ["--stage sweep has not run"]}
    mem = s["membership"]
    n_t = mem["n_target"]
    reasons, blockers = [], []

    if s.get("U5_mechanical"):
        blockers.append(
            f"U5: only {s['U5_match_rate']:.1%} of sweep steps matched the branch at all, so the "
            f"drop is a grid artefact, not target-awareness"
        )
    if n_t < U1_HARD:
        blockers.append(f"U1: {n_t} members < {U1_HARD} (half the published {PUBLISHED_N})")
    elif n_t < U1_SOFT:
        reasons.append(
            f"U1 soft band: {n_t} members is {n_t / PUBLISHED_N:.1%} of the published "
            f"{PUBLISHED_N}. Not disqualifying, but the threshold and the score changed together "
            f"-- 0.6 must be recalibrated before the target arm is used with the published cut"
        )

    nb = payload.get("neighbours", {}).get("result", {})
    if nb.get("U3_breached"):
        blockers.append(
            f"U3: attributed fraction of outer members fell to "
            f"{nb['attribution_target_arm']['fraction_attributed']:.1%} < "
            f"{U3_ATTRIBUTION_FLOOR:.0%}; b1's conclusion no longer holds on the new sample"
        )
    kv = payload.get("king", {}).get("verdict", {})
    if kv.get("verdict") == "CORRECTION-REQUIRED":
        blockers.append("U2: " + kv["reasons"][0])
    elif kv.get("verdict", "").startswith("UNDETERMINED"):
        reasons.append("U4: sampler gates failed, so R_c is UNDETERMINED rather than a null")
    elif kv.get("verdict"):
        reasons.append("R_c: " + kv["reasons"][0])

    if not s.get("G1_production_fidelity", {}).get("gate_passed"):
        reasons.append(
            "G1 FAILED: the any-arm does not reproduce the published 628 / published clip band, "
            "so counts here are NOT quotable against the published sample. The between-arm "
            "comparison is still valid -- same pipeline both sides"
        )

    if blockers:
        call = "DO NOT MOVE THE DEFAULT"
    else:
        call = "MOVE THE DEFAULT (with the wiring caveat)"
    return {
        "recommendation": call,
        "blockers": blockers,
        "notes": reasons,
        "wiring_caveat": "recovery_frequency='target' is computed inside the "
        "`if select_cluster:` branch of _annotate_pseudoprobability_results, and the production "
        "NGC 6383 recipe passes select_cluster=False. Flipping the default today changes NOTHING "
        "for the published pipeline. The delta above is the counterfactual.",
    }


# ------------------------------------------------------------------ selftest
def selftest() -> None:
    """Mutation tests. A check that cannot fail is worse than none -- so break each on purpose."""
    rng = np.random.default_rng(0)

    # --- 1. monotonicity check must FIRE when f_target is nudged above f_any.
    f_any = rng.random(1000)
    f_tgt = f_any * rng.random(1000)
    assert monotonicity_check(f_any, f_tgt)["holds"], "clean case should hold"
    bad = f_tgt.copy()
    bad[17] = f_any[17] + 1e-9
    m = monotonicity_check(f_any, bad)
    assert not m["holds"] and m["n_violations"] == 1, m
    print("  monotonicity check fires on a 1e-9 upward nudge (mutation test)")

    # --- 2. subset check must FIRE on one extra member.
    a = rng.random(1000) > 0.5
    t = a & (rng.random(1000) > 0.3)
    assert subset_check(a, t)["holds"]
    t2 = t.copy()
    t2[np.flatnonzero(~a)[0]] = True
    assert not subset_check(a, t2)["holds"], "subset check cannot fail -> useless"
    print("  subset check fires on one member outside the any-arm set (mutation test)")

    # --- 3. _target_recovery_frequency: parameter-free properties on a constructed matrix.
    from erotica.core import Clustering

    # 4 sources, 3 steps. Target = sources 0,1.
    sweep = np.array([[0, 0, 0], [0, 0, -1], [1, -1, 0], [-1, -1, -1]], dtype=np.int64)
    mask = np.array([True, True, False, False])
    f, info = Clustering._target_recovery_frequency(sweep, mask)
    f_any_c = np.mean(sweep != -1, axis=1)
    assert np.all(f <= f_any_c + 1e-12), (f, f_any_c)
    assert info["n_steps"] == 3
    # step 0: cluster 0 = {0,1}, inter=2, union=2 -> J=1; cluster 1 = {2}, inter 0. best=0.
    # step 1: cluster 0 = {0,1}, J=1. best=0.
    # step 2: cluster 0 = {0,3?} -> sources 0 and 2 have label 0 at step 2. inter={0} -> J=1/3.
    assert abs(f[0] - 1.0) < 1e-12, f
    assert abs(f[1] - 2.0 / 3.0) < 1e-12, f
    print(f"  _target_recovery_frequency on a hand-checked 4x3 matrix: f = {f.tolist()}")

    # A step with NO overlap must contribute zero, not an arbitrary argmax.
    sweep2 = np.array([[-1, 0], [-1, 0], [0, -1], [0, -1]], dtype=np.int64)
    mask2 = np.array([True, True, False, False])
    f2, info2 = Clustering._target_recovery_frequency(sweep2, mask2)
    assert info2["n_steps_matched"] == 1, info2
    assert abs(f2[2]) < 1e-12 and abs(f2[3]) < 1e-12, f2
    print("  a step whose clusters miss the target contributes zero (no arbitrary argmax)")

    # --- 4. the king verdict rule must reach each of its values.
    def _fit(rc, sd, gates=True):
        return {
            "R_c": {"median": rc, "sd": sd},
            "gates_passed": gates,
            "r_hat_max": 1.0,
            "ess_bulk_min": 2000,
            "divergences": 0,
        }

    cases = [
        (_fit(1.35, 0.20), _fit(1.36, 0.20), "NULL-PROTECTS-PUBLISHED"),
        (_fit(1.35, 0.20), _fit(1.25, 0.20), "OUTSIDE-BAND-BUT-NOT-RESOLVABLE"),
        (_fit(1.35, 0.05), _fit(1.10, 0.05), "CORRECTION-REQUIRED"),
        (_fit(1.35, 0.20), _fit(1.36, 0.20, gates=False), "UNDETERMINED (U4)"),
    ]
    for fa, ft, want in cases:
        got = king_verdict({"members_any": fa, "members_target": ft})["verdict"]
        assert got == want, f"{want} unreachable: got {got}"
    print("  king verdict rule reaches all four of its values")

    # --- 5. the recommendation rule must be able to say both things.
    base = {
        "sweep": {
            "step1": {
                "membership": {"n_target": 600, "n_any": 628},
                "U5_mechanical": False,
                "U5_match_rate": 1.0,
                "G1_production_fidelity": {"gate_passed": True},
            }
        }
    }
    assert recommendation(base)["recommendation"].startswith("MOVE"), recommendation(base)
    for mutate in (
        lambda p: p["sweep"]["step1"]["membership"].__setitem__("n_target", 10),
        lambda p: p["sweep"]["step1"].__setitem__("U5_mechanical", True),
    ):
        p = json.loads(json.dumps(base))
        mutate(p)
        assert recommendation(p)["recommendation"] == "DO NOT MOVE THE DEFAULT", p
    p = json.loads(json.dumps(base))
    p["neighbours"] = {
        "result": {"U3_breached": True, "attribution_target_arm": {"fraction_attributed": 0.05}}
    }
    assert recommendation(p)["recommendation"] == "DO NOT MOVE THE DEFAULT"
    print("  recommendation rule reaches MOVE and DO-NOT-MOVE, and each blocker fires alone")

    # --- 6. quantiles must not silently swallow an empty array.
    assert quantiles(np.array([]))["n"] == 0
    print("selftest OK")


# ------------------------------------------------------------------ report
def report() -> None:
    d = load_sidecar()
    g0 = d.get("gate0_fork_fidelity", {})
    for k, v in g0.items():
        print(f"G0 fork fidelity [{k}]: {'PASS' if v['gate_passed'] else 'FAIL'}", end="")
        print(f"  {v['mismatches']}" if v["mismatches"] else "")
    sw = d.get("sweep", {})
    for k, s in sw.items():
        g1 = s["G1_production_fidelity"]
        m = s["membership"]
        print(f"\n=== sweep [{k}], {s['n_grid']} steps, {s['seconds']}s ===")
        print(
            f"G1 production fidelity: {'PASS' if g1['gate_passed'] else 'FAIL'} -- any-arm "
            f"n={g1['any_arm_n']} vs published {g1['production_n']}, clip "
            f"[{g1['clip_low']:.10f}, {g1['clip_high']:.10f}], Jaccard "
            f"{g1['jaccard_vs_production']:.4f}"
        )
        print(
            f"membership: any {m['n_any']} -> target {m['n_target']} "
            f"({m['retained_fraction']:.1%} retained), Jaccard {m['jaccard_any_vs_target']:.4f}, "
            f"{m['n_leave']} leave / {m['n_enter']} enter"
        )
        print(
            f"  leavers: median radius {m['leavers']['median_radius']}', "
            f"{m['leavers']['n_beyond_rJ']}/{m['leavers']['n']} beyond r_J"
        )
        ni = s["target_recovery_info"]
        print(
            f"  U5 mechanism: {ni['n_steps_matched']}/{ni['n_steps']} steps matched "
            f"({s['U5_match_rate']:.1%}), mean Jaccard {ni['mean_matched_jaccard']}, "
            f"mean size ratio {ni['mean_matched_size_ratio']}"
        )
        for scope in ("branch_only", "published_members_only"):
            sd = s["score_distributions"][scope]
            for arm_name in ("ptilde_any", "ptilde_target"):
                q = sd[arm_name]
                print(
                    f"  {scope:24s} {arm_name:14s} median {q['median']:.4f} "
                    f"[q25 {q['q25']:.4f}, q75 {q['q75']:.4f}] "
                    f"zero {q['frac_exactly_zero']:.1%} >=0.6 {q['frac_ge_0.6']:.1%}"
                )
        print(
            f"  monotonicity {s['monotonicity']['holds']}, subset {s['subset_target_in_any']['holds']}"
        )
    nb = d.get("neighbours", {}).get("result")
    if nb:
        print(
            f"\nattribution beyond r_J: any {nb['attribution_any_arm']['n_attributed']}/"
            f"{nb['attribution_any_arm']['n_outer']} "
            f"({(nb['attribution_any_arm']['fraction_attributed'] or 0):.1%}) -> target "
            f"{nb['attribution_target_arm']['n_attributed']}/"
            f"{nb['attribution_target_arm']['n_outer']} "
            f"({(nb['attribution_target_arm']['fraction_attributed'] or 0):.1%})"
        )
        p = nb["P1_pooled_2x2"]
        print(
            f"P1 pooled: drop rate inside circles {(p['drop_rate_inside'] or 0):.1%} vs outside "
            f"{(p['drop_rate_outside'] or 0):.1%}, OR {p['odds_ratio']:.3f}, "
            f"Fisher p {p['fisher_p']:.3e}"
        )
        print(
            f"P1 radius-conditioned: Mantel-Haenszel OR = "
            f"{nb['P1_radius_conditioned']['mantel_haenszel_odds_ratio']}"
        )
    kf = d.get("king", {})
    for k, v in kf.items():
        if k == "verdict":
            continue
        print(
            f"king[{k:20s}] N={v['n_stars']:5d} R_c={v['R_c']['median']:.3f}+/-{v['R_c']['sd']:.3f}"
            f"  R_t={v['R_t']['median']:.1f}  r_hat={v['r_hat_max']:.3f} "
            f"ess={v['ess_bulk_min']:.0f} div={v['divergences']} gates={v['gates_passed']}"
        )
    if "verdict" in kf:
        print(f"R_c VERDICT: {kf['verdict']['verdict']} -- {kf['verdict']['reasons'][0]}")
    rec = recommendation(d)
    print(f"\nRECOMMENDATION: {rec['recommendation']}")
    for r in rec.get("blockers", []):
        print("  BLOCKER:", r)
    for r in rec.get("notes", []):
        print("  note:", r)
    print("  wiring:", rec["wiring_caveat"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", choices=["gate0", "sweep", "neighbours", "king"])
    ap.add_argument("--mcs-step", type=int, default=1)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.report:
        report()
        return
    if args.stage == "gate0":
        r = stage_gate0(args.mcs_step if args.mcs_step != 1 else 10)
        print(json.dumps(jsonable(r), indent=1))
        if not r["gate_passed"]:
            raise SystemExit("G0 FAILED -- this fork is not b1's recipe; stop here")
    elif args.stage == "sweep":
        r = stage_sweep(args.mcs_step)
        print(json.dumps(jsonable({k: v for k, v in r.items() if not k.startswith("_")}), indent=1))
    elif args.stage == "neighbours":
        r = stage_neighbours()
        print(json.dumps(jsonable(r), indent=1))
    elif args.stage == "king":
        r = stage_king(args.draws, args.seed)
        print(json.dumps(jsonable(r["verdict"]), indent=1))
    else:
        ap.print_help()
        return
    payload = load_sidecar()
    payload["recommendation"] = recommendation(payload)
    payload["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_sidecar(payload)
    print("\nRECOMMENDATION:", payload["recommendation"]["recommendation"])


if __name__ == "__main__":
    main()
