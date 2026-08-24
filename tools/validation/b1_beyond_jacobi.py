#!/usr/bin/env python3
r"""B1 -- is NGC 6383's 43%-beyond-Jacobi population real, or did the pipeline make it?

WHY THIS EXISTS
---------------
``docs/design-notes/king_model_validity.md`` states, and P01's discussion carries, that
**43% of the p > 0.6 members lie beyond the Jacobi radius** (273 of 628 inside a 70' cone;
``r_J`` = 12.14 pc = 37.9' from Hunt & Reffert 2024). Two readings are observationally
degenerate on the radial marginal alone:

  (a) a real unbound corona / tidal structure, or
  (b) membership contamination above ~20%, in which case the *pipeline manufactured the halo*
      -- Pang et al. (2022) Sect. 4.2: "An artificial 'halo'-like substructure will appear when
      the contamination rate rises above 20%."

This script is the instrument that separates them. It is the radius-resolved successor to
``decoy_fdp.py``, whose global false-discovery proportion cannot decide the question: a flat
FDP under a real halo and an FDP that climbs to 1 with radius give the *same* global number.

WHAT THE ESTIMATOR ACTUALLY CONSUMES -- WRITTEN DOWN BEFORE ANY NUMBER WAS LOOKED AT
-----------------------------------------------------------------------------------
The production membership (``ngc6383_radius_robustness.run_radius``) is

    branch(HDBSCAN sweep over pmra,pmdec)  AND  p~ >= 0.6  AND  parallax in sigma-clip band

so the axes are used as follows, and this dictates which questions are answerable:

  | axis                | consumed by membership? | usable as an independent oracle?          |
  |---------------------|-------------------------|-------------------------------------------|
  | ``pmra, pmdec``     | YES -- the only cluster axis | NO. **Proper-motion coherence of the outer
  |                     |                         | population is circular**: they were selected
  |                     |                         | for it. A "coherent" answer is guaranteed
  |                     |                         | by construction and is not evidence.       |
  | ``parallax``        | PARTLY -- a hard 2-sigma band | WEAKLY. Within-band *shape* is free, but
  |                     |                         | both real members and survivors of the band
  |                     |                         | are truncated the same way.                |
  | ``ra, dec``         | NO -- never enters      | YES, fully independent.                    |
  | photometry (G, BP-RP)| NO -- never enters     | YES, fully independent.                    |

Consequence, stated up front: *any* statistic that is a function of the radial marginal alone
inherits the degeneracy this script exists to break. The break has to come from an axis the
membership never saw (position, photometry) or from a null that pushes a known truth through
the identical pipeline. Both are used below.

THE PRE-REGISTERED FALSIFICATION CRITERIA
-----------------------------------------
Fixed before the radius-resolved FDP was computed. ``r_J`` = 37.9'; "beyond" means r > r_J
inside the 70' cone; the target sample is the production ``paper_reference_p06`` (628 stars).

Let ``T_out`` = production members beyond ``r_J``; ``D_out^(j)`` = members the *identical*
pipeline selects beyond ``r_J`` in decoy realization *j* (a catalogue with the cluster's
proper-motion overdensity destroyed but positions, magnitudes, parallaxes and the
proper-motion error structure preserved), after the same parallax clip re-derived per
realization.

    FDP_out(median) = median_j(D_out^(j)) / T_out
    FDP_out(p90)    = percentile_90_j(D_out^(j)) / T_out

**VERDICT RULE -- applied as written, no rounding toward the interesting answer.**

  MANUFACTURED (withdraw B1's headline; P01's discussion must be corrected):
      FDP_out(median) >= 0.20                                        [the Pang threshold]
      OR the injection control (below) recovers >= 0.20 of its members beyond ``r_J`` when the
      injected truth is truncated at ``r_J`` and therefore has *exactly zero* stars there.

  NOT-MANUFACTURED (hypothesis (b) is excluded; **this is not the same as "the halo is real"**):
      FDP_out(median) < 0.20  AND  FDP_out(p90) < 0.50
      AND the injection control recovers < 0.20 beyond ``r_J``.

  UNDETERMINED (report as undetermined; do not soften either way):
      anything else -- in particular median < 0.20 with p90 >= 0.50, or the two controls
      disagreeing. Straddling is a result, not a failure.

  NOT-NGC-6383 (added at stage 4; see "THE HYPOTHESIS THE RULE HAS NO TERM FOR"):
      >= 20% of the members beyond ``r_J`` fall inside the total radius of a *catalogued*
      comoving cluster. Overrides NOT-MANUFACTURED, because excluding field contamination
      says nothing about a second real structure.

Additionally pre-registered, because they can independently overturn the reading:

  * **The area baseline.** A 70' cone has 70.7% of its area beyond ``r_J``. A *uniform*
    contaminant therefore puts 70.7% of itself beyond ``r_J``. So "43% beyond ``r_J``" is
    **below** the pure-contamination expectation and above the zero of a truncated cluster; the
    raw fraction is not evidence for anything on its own, and this script reports the number so
    that fact cannot be quoted away.
  * **Radial shape.** If the member surface density beyond ``r_J`` is statistically flat, the
    outer population is degenerate with a uniform contaminant *and* with a corona wider than the
    footprint (the King+corona fit already put R2 > 70' at P = 1.000). A flat outer profile is
    therefore not evidence for (a); a *declining* one is.
  * **Azimuth.** A uniform contaminant is azimuthally distributed like the catalogue. A chi^2 of
    outer-member azimuth against the catalogue azimuth in the same annuli is a test on an axis
    the membership never saw. Rejecting uniformity falsifies "pure uniform field contamination"
    -- but it does NOT establish that the structure belongs to NGC 6383, since a second real
    comoving group on the same sightline would also reject.

THE HYPOTHESIS THE RULE HAS NO TERM FOR
---------------------------------------
B1 offers two readings, and the rule above decides between them. **The data support a third.**
The decoy null is a structureless field *by construction*, so it can contain no comoving
neighbour, and its false-discovery proportion is explicitly a floor rather than a total
(``decoy_fdp.py``'s own caveat says so). A proper-motion-only membership admits *any* comoving
structure on the sightline:

  (c) a distinct, catalogued, comoving cluster -- neither NGC 6383's halo nor field
      contamination.

A low FDP therefore licenses "not manufactured by field contamination" and nothing more. Stage 4
tests (c) on sky position, the axis the membership never used, against Hunt & Reffert (2024)
pulled live. The rule is not rewritten to accommodate the answer; the verdict is downgraded
explicitly, and the FDP result is reported unchanged alongside it.

NEGATIVE CONTROLS -- both required, both reported whatever they say
-------------------------------------------------------------------
1. **Field-only.** The decoy catalogue is a pure field. The halo statistic must not fire on it.
   The prediction is sharp and uncomfortable: *it will*. Any selection from a structureless
   field lands ~70.7% beyond ``r_J``. Reported explicitly, because it is the proof that the
   fraction-beyond-``r_J`` statistic cannot by itself discriminate.
2. **Synthetic bound cluster with no halo.** A cluster with truth truncated at ``r_J`` (EFF,
   gamma and a from the published fit), matched in N, proper-motion centroid and intrinsic
   proper-motion dispersion, injected into the *neutralised* field and pushed through the
   identical pipeline. If the pipeline returns 43% beyond ``r_J`` there, that settles it and
   nothing else in this file matters.

AMENDMENT, RECORDED RATHER THAN BACKDATED
-----------------------------------------
The rule above was fixed before any decoy realization ran. **One change was made after the
first realization**, and it is logged here rather than folded silently into the
pre-registration, because a rule edited after seeing an outcome is not a rule.

*What changed:* the decoy donor pool now excludes the 798 production branch stars. *Why:* a
count, not a result -- **301 of those 798 sit at r > 45'**, so 1.20% of the original donor pool
carried the real cluster's proper motion and ~535 of 44438 decoy sources inherited it. A null
containing the signal at half amplitude is degenerate with its own control. *What was known at
the time:* realization 1 of the uncorrected variant (41 selected, 28 beyond r_J). *Effect on
the verdict:* none by construction -- the uncorrected variant is retained in full as the
`fullpool` sensitivity, the corrected one is `nopool_branch`, they bias in opposite directions,
and the verdict is reported as decided only if **both land on the same side of 0.20**.

THE CHECK ``decoy_fdp.py`` DID NOT DO, AND WHY IT COMES FIRST
------------------------------------------------------------
``decoy_fdp.py`` reimplements the sweep with ``cluster_selection_method="eom"``,
``allow_single_cluster=False``, ``match_reference_implementation=True``, ``min_members=50`` and
an mcs grid of step 10. **Production uses ``leaf``, ``allow_single_cluster=True``,
``match_reference_implementation=False``, ``min_cluster_members=200`` and step 1.** Its
apparent agreement -- 771 -- is an accident: 771 is production's *p >= 0.5 pre-clip* count and
the reimplementation's *p >= 0.6* count. It also never applies the parallax clip that takes
production from 771 to 628. A false-discovery proportion measured on a different pipeline is
not the false-discovery proportion of the published sample.

So stage 0 runs the pipeline used here on the *real* catalogue and measures the Jaccard overlap
against the production ``paper_reference_p06`` column. **If the overlap is below 0.90 the FDP
from this script is reported as measuring a related-but-different pipeline, and the verdict
rule is not applied.** That is a falsifier for the instrument, not for the cluster.

The mcs grid is decimated (``--mcs-step``) because the production grid is 290 HDBSCAN fits on
44438 sources (~24 min per realization). ``probability_times`` is the *fraction of sweep steps*
a source was in any cluster, so decimating changes the estimator; stage 0 is exactly the
measurement that says whether it changed it enough to matter.

STAGES, AND WHY THEY CHECKPOINT
-------------------------------
Long uninterrupted runs in this programme have died mid-flight. Every stage appends to the
JSON sidecar as soon as it has a result, and every decoy/injection realization is written the
moment it finishes, with the verdict rule re-evaluated on whatever realizations exist. Twenty
realizations that produce a verdict are worth more than forty that die at the thirty-first.

    stage 0  fidelity     reimplementation vs production membership (gates everything)
    stage 1  geometry     radial + azimuthal profiles; no clustering, cannot fail
    stage 2  decoy        radius-resolved FDP  [negative control 1: field-only]
    stage 3  injection    truncated-cluster injection  [negative control 2]
    stage 4  character    only if the gate is passed

USAGE
-----
    python tools/validation/b1_beyond_jacobi.py --stage 0
    python tools/validation/b1_beyond_jacobi.py --stage 1
    python tools/validation/b1_beyond_jacobi.py --stage 2 --realizations 24
    python tools/validation/b1_beyond_jacobi.py --stage 3
    python tools/validation/b1_beyond_jacobi.py --stage 4
    python tools/validation/b1_beyond_jacobi.py --report

Environment: ``/Users/notluquis/miniforge3/envs/erotica-bench/bin/python`` (editable ``erotica``).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="erotica-mpl-"))

import numpy as np

from erotica.core.clustering import NoCandidateClusters

BASE = Path("/Users/notluquis/erotica")
CONE = (
    BASE
    / "data/test/NGC6383/comments_paper/radius_robustness/generated/70"
    / "paperfaithful_with_clip_flags.ecsv"
)
OUT = Path(__file__).with_suffix(".json")
NPZ = Path(__file__).with_name("b1_beyond_jacobi.npz")

# NGC 6383 centre used by every other script in this directory.
CENTER_RA, CENTER_DEC = 263.6826, -32.5838
# Hunt & Reffert 2024 (2024A&A...686A..42H): 12.14 pc at 1101 pc.
R_J = 37.9
CONE_RADIUS = 70.0
# Production branch-selection reference (summary.json: branch_selection_reference_pm).
REFERENCE_PM = (2.54, -1.71)
# Published EFF fit for the 70' sample (king_model_validity.md).
EFF_GAMMA, EFF_A = 2.322, 1.649

# Annulus edges. Fine enough beyond r_J to see a gradient, coarse enough to keep counts usable.
EDGES = np.array(
    [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        10.0,
        14.0,
        18.0,
        24.0,
        30.0,
        R_J,
        42.0,
        46.0,
        50.0,
        54.0,
        58.0,
        62.0,
        66.0,
        CONE_RADIUS,
    ]
)


# ----------------------------------------------------------------------------- helpers
def jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def load_sidecar() -> dict:
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {}


def save_sidecar(payload: dict) -> None:
    text = json.dumps(jsonable(payload), indent=1)
    OUT.write_text(text)
    kb = len(text.encode()) / 1024
    if kb > 500:
        raise RuntimeError(f"sidecar is {kb:.0f} KB, over the 500 KB budget")


def checkpoint(stage: str, key: str, value: Any) -> dict:
    """Write one result into the sidecar immediately. Survives a killed run."""
    payload = load_sidecar()
    payload.setdefault(stage, {})[key] = value
    payload["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_sidecar(payload)
    return payload


# ----------------------------------------------------------------------------- data
_CACHE: dict[str, Any] = {}


def cone():
    """Preprocessed 70' catalogue + production membership flags, cached in-process."""
    if "table" in _CACHE:
        return _CACHE["table"]
    from astropy.table import Table

    t = Table.read(CONE)
    ra = np.asarray(t["ra"], float)
    dec = np.asarray(t["dec"], float)
    cosd = np.cos(np.deg2rad(CENTER_DEC))
    # Small-angle tangent-plane offsets: the field is 70', where the error is < 1e-4.
    dx = (ra - CENTER_RA) * cosd
    dy = dec - CENTER_DEC
    t["b1_radius"] = np.hypot(dx, dy) * 60.0
    t["b1_theta"] = np.degrees(np.arctan2(dy, dx)) % 360.0
    _CACHE["table"] = t
    return t


def radius_of(t) -> np.ndarray:
    return np.asarray(t["b1_radius"], float)


# ----------------------------------------------------------------------------- pipeline
def ngc_like_label(table, labels, probability, reference_pm=REFERENCE_PM):
    """Production branch rule: the label whose p>=0.5 members sit nearest ``reference_pm``."""
    pmra = np.asarray(table["pmra"], float)
    pmdec = np.asarray(table["pmdec"], float)
    best, best_d = -1, np.inf
    for lab in sorted({int(v) for v in np.unique(labels) if int(v) != -1}):
        high = (labels == lab) & (probability >= 0.5)
        if not np.count_nonzero(high):
            continue
        d = float(
            np.hypot(
                np.nanmedian(pmra[high]) - reference_pm[0],
                np.nanmedian(pmdec[high]) - reference_pm[1],
            )
        )
        if d < best_d:
            best, best_d = lab, d
    return best, best_d


def run_pipeline(table, mcs_step: int, quiet: bool = True):
    """The production membership recipe, with the mcs grid decimated by ``mcs_step``.

    Returns ``(members, info)``; ``members`` is a boolean mask matching
    ``paper_reference_p06`` semantics: branch AND p~>=0.6 AND parallax inside the
    re-derived 2-sigma clip band. Returns an all-False mask when the sweep finds no
    admissible cluster -- a legitimate outcome on a decoy, not an error.
    """
    from astropy.stats import sigma_clip

    from erotica.analysis import histogram_mode
    from erotica.core import Clustering

    work = table.copy(copy_data=True)
    clust = Clustering(work, None)
    t0 = time.perf_counter()
    try:
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
    # El tipo lo dice ahora, en vez de un comentario que nadie puede ejecutar: si la búsqueda
    # se rompiera por otra cosa, este `except` la habría tratado como «no encontró nada».
    except NoCandidateClusters as exc:
        return np.zeros(len(table), bool), {
            "detected": False,
            "reason": str(exc),
            "seconds": time.perf_counter() - t0,
        }

    data = clust.data
    labels = np.asarray(data["cluster_hdbscan"], int)
    prob = np.asarray(data["probability"], float)
    label, pm_dist = ngc_like_label(data, labels, prob)
    if label < 0:
        return np.zeros(len(table), bool), {
            "detected": False,
            "reason": "no branch",
            "seconds": time.perf_counter() - t0,
        }

    branch = labels == label
    preclip = branch & (prob >= 0.5)

    def mode(values, axis=None):
        if axis is not None:
            m = np.apply_along_axis(histogram_mode, axis, values)
        else:
            m = histogram_mode(values)
        unit = getattr(values, "unit", None)
        return m * unit if unit is not None else m

    # Units are stripped first: the ECSV round-trip gives ``parallax`` a mas unit that the
    # production in-memory table does not carry, and astropy's sigma_clip then compares a
    # Quantity centre against a dimensionless std and raises. Numerically identical, and
    # ``stage0`` checks the bounds against production's own clip_low/clip_high.
    plx = np.asarray(data["parallax"], float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, lo, hi = sigma_clip(
            plx[preclip], sigma=2, cenfunc=mode, stdfunc="std", return_bounds=True
        )
    lo = float(np.asarray(lo).reshape(-1)[0])
    hi = float(np.asarray(hi).reshape(-1)[0])
    clip = (plx >= lo) & (plx <= hi)
    members = branch & (prob >= 0.6) & clip
    info = {
        "detected": True,
        "seconds": time.perf_counter() - t0,
        "branch_label": int(label),
        "branch_pm_dist": pm_dist,
        "branch_n": int(branch.sum()),
        "preclip_p05": int(preclip.sum()),
        "clip_low": lo,
        "clip_high": hi,
        "n_members": int(members.sum()),
        "best_mcs": int((clust.pseudoprobability_selected_ or {}).get("min_cluster_size", -1)),
    }
    return members, info


# ----------------------------------------------------------------------------- stage 0
def stage0(mcs_step: int) -> dict:
    """Does this reimplementation reproduce the production membership? Gates everything."""
    t = cone()
    prod = np.asarray(t["paper_reference_p06"], bool)
    members, info = run_pipeline(t, mcs_step)
    inter = int((members & prod).sum())
    union = int((members | prod).sum())
    res = {
        "mcs_step": mcs_step,
        "n_grid": len(range(10, 300, mcs_step)),
        "production_n": int(prod.sum()),
        "reimplementation_n": int(members.sum()),
        "intersection": inter,
        "union": union,
        "jaccard": inter / union if union else 0.0,
        "recall_of_production": inter / int(prod.sum()) if prod.sum() else 0.0,
        "production_beyond_rJ": int((radius_of(t)[prod] > R_J).sum()),
        "reimplementation_beyond_rJ": int((radius_of(t)[members] > R_J).sum()),
        # The FDP denominator must come from the SAME configuration that measures the decoy
        # numerator, or the ratio mixes two pipelines. Both are reported.
        "reimplementation_counts": annulus_counts(radius_of(t), members).tolist(),
        "is_strict_subset_of_production": bool(inter == int(members.sum())),
        **info,
    }
    res["gate_passed"] = bool(res["jaccard"] >= 0.90)
    checkpoint("stage0_fidelity", f"step{mcs_step}", res)
    return res


# ----------------------------------------------------------------------------- stage 1
def annulus_counts(r: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    sel = r if mask is None else r[mask]
    return np.histogram(sel, bins=EDGES)[0]


def stage1() -> dict:
    """Radial and azimuthal geometry of the production members. No clustering; cannot fail."""
    t = cone()
    r = radius_of(t)
    prod = np.asarray(t["paper_reference_p06"], bool)
    inside_cone = r <= CONE_RADIUS
    area = np.pi * (EDGES[1:] ** 2 - EDGES[:-1] ** 2)
    cat = annulus_counts(r, inside_cone)
    mem = annulus_counts(r, prod & inside_cone)
    out = EDGES[:-1] >= R_J

    # Is the outer member surface density flat, or declining? Weighted least squares on
    # log(density) against log(midpoint) beyond r_J, Poisson weights.
    mid = 0.5 * (EDGES[1:] + EDGES[:-1])
    dens = mem / area
    ok = out & (mem > 0)
    slope = slope_err = float("nan")
    if ok.sum() >= 3:
        x = np.log(mid[ok])
        y = np.log(dens[ok])
        w = mem[ok].astype(float)  # var(log N) ~ 1/N
        p, cov = np.polyfit(x, y, 1, w=np.sqrt(w), cov=True)
        slope, slope_err = float(p[0]), float(np.sqrt(cov[0, 0]))

    # Azimuth on an axis the membership never saw.
    naz = 12
    azb = np.linspace(0, 360, naz + 1)
    th = np.asarray(t["b1_theta"], float)
    beyond = inside_cone & (r > R_J)
    cat_az = np.histogram(th[beyond], bins=azb)[0]
    mem_az = np.histogram(th[beyond & prod], bins=azb)[0]
    exp_az = cat_az / cat_az.sum() * mem_az.sum()
    chi2 = float(np.sum((mem_az - exp_az) ** 2 / np.maximum(exp_az, 1e-9)))
    from scipy import stats

    p_az = float(stats.chi2.sf(chi2, naz - 1))

    res = {
        "r_J_arcmin": R_J,
        "cone_radius_arcmin": CONE_RADIUS,
        "area_fraction_beyond_rJ": float((CONE_RADIUS**2 - R_J**2) / CONE_RADIUS**2),
        "n_members": int(prod.sum()),
        "n_members_beyond_rJ": int((r[prod] > R_J).sum()),
        "fraction_beyond_rJ": float((r[prod] > R_J).mean()),
        "edges": EDGES,
        "annulus_area_arcmin2": area,
        "catalogue_counts": cat,
        "member_counts": mem,
        "member_density": dens,
        "catalogue_density": cat / area,
        "outer_logslope": slope,
        "outer_logslope_err": slope_err,
        "azimuth_edges": azb,
        "azimuth_catalogue_beyond_rJ": cat_az,
        "azimuth_members_beyond_rJ": mem_az,
        "azimuth_chi2": chi2,
        "azimuth_dof": naz - 1,
        "azimuth_p": p_az,
    }
    checkpoint("stage1_geometry", "result", res)
    return res


# ----------------------------------------------------------------------------- decoy
def build_decoy(
    t,
    rng,
    field_radius: float = 45.0,
    k_match: int = 50,
    dedup_jitter: float = 1e-3,
    exclude_branch: bool = True,
):
    """Field-PM decoy that *keeps every position*.

    ``decoy_fdp.py`` bootstraps field stars into a bare (pmra, pmdec) array, so its
    detections have no sky position and its FDP cannot be resolved in radius. Here the real
    catalogue is kept intact -- positions, magnitudes, parallaxes, proper-motion errors -- and
    only the proper motions are replaced, by a donor drawn from the ``r > field_radius`` field
    pool matched in G. That destroys the cluster's proper-motion overdensity (the only thing
    the clustering sees) while preserving the position <-> magnitude <-> astrometric-precision
    coupling that decides *which* stars a chance clump can recruit and *where they sit*.

    Jitter is ``dedup_jitter`` = 1e-3 mas/yr, ~200x below a typical proper-motion error: it
    exists only to stop bootstrap duplicates reading as zero-distance points. ``decoy_fdp.py``
    jitters by the full per-star error instead, which broadens the decoy proper-motion
    distribution and therefore *suppresses* chance clumps -- biased toward a low FDP, i.e.
    toward declaring the halo real. Not repeated here.

    ``exclude_branch`` -- **a decoy must contain no true positives, and the obvious pool does.**
    ``decoy_fdp.py`` draws donors from every star at ``r > 45'``, but **301 of the production
    branch's 798 stars live there**, so 1.20% of the pool carries the real cluster's proper
    motion and ~535 of 44438 decoy sources inherit it. The null then contains a real
    overdensity at roughly half the true amplitude, and the experiment is degenerate with its
    own control -- it can "discover" the cluster it was built to exclude.

    Neither setting is unbiased, so both are run and both are reported:

      ``exclude_branch=True``  (primary) drops the 798 branch stars from the donor pool. If the
          outer members are in fact field, this also removes ~1.2% of the field's own density at
          the cluster proper motion, biasing the FDP **down** -- toward declaring the halo real.
      ``exclude_branch=False`` (sensitivity) keeps them, biasing the FDP **up** -- toward
          declaring it manufactured.

    A verdict is only reported as decided if both variants land on the same side of 0.20.
    """
    r = radius_of(t)
    pmra = np.asarray(t["pmra"], float)
    pmdec = np.asarray(t["pmdec"], float)
    g = np.asarray(t["Gmag"], float)
    keep = np.ones(len(t), bool)
    if exclude_branch:
        keep &= ~np.asarray(t["paper_reference_branch"], bool)
    pool = np.flatnonzero((r > field_radius) & keep & np.isfinite(pmra) & np.isfinite(pmdec))
    gp = g[pool]
    finite = np.isfinite(gp)
    pool, gp = pool[finite], gp[finite]
    order = np.argsort(gp)
    pool, gp = pool[order], gp[order]

    n = len(t)
    gq = np.where(np.isfinite(g), g, np.nanmedian(gp))
    lo = np.clip(np.searchsorted(gp, gq) - k_match // 2, 0, len(pool) - k_match)
    donors = pool[lo + rng.integers(0, k_match, size=n)]

    new = t.copy(copy_data=True)
    new["pmra"] = pmra[donors] + rng.normal(0.0, dedup_jitter, n)
    new["pmdec"] = pmdec[donors] + rng.normal(0.0, dedup_jitter, n)
    return new, donors


def stage2(n_real: int, mcs_step: int, seed: int, exclude_branch: bool = True) -> dict:
    """Radius-resolved FDP. Also negative control 1 (field-only). Checkpoints per realization."""
    t = cone()
    r = radius_of(t)
    variant = "nopool_branch" if exclude_branch else "fullpool"
    payload = load_sidecar()
    done = payload.get("stage2_decoy", {}).get(variant, {}).get("realizations", [])
    rng = np.random.default_rng(seed)
    for _ in range(len(done)):  # replay the stream so a resume is reproducible
        build_decoy(t, rng, exclude_branch=exclude_branch)

    for j in range(len(done), n_real):
        decoy, _ = build_decoy(t, rng, exclude_branch=exclude_branch)
        members, info = run_pipeline(decoy, mcs_step)
        rr = r[members]
        rec = {
            "j": j,
            "detected": info.get("detected", False),
            "n": int(members.sum()),
            "n_beyond_rJ": int((rr > R_J).sum()),
            "counts": annulus_counts(r, members).tolist(),
            "seconds": round(info.get("seconds", 0.0), 1),
            "branch_pm_dist": info.get("branch_pm_dist"),
            "best_mcs": info.get("best_mcs"),
            "preclip_p05": info.get("preclip_p05"),
        }
        done.append(rec)
        s = summarise_stage2(done, mcs_step)
        checkpoint(
            "stage2_decoy",
            variant,
            {"realizations": done, "summary": s, "exclude_branch": exclude_branch},
        )
        print(
            f"  {variant} {j + 1:3d}/{n_real}  n={rec['n']:4d}  "
            f"beyond_rJ={rec['n_beyond_rJ']:4d}  ({rec['seconds']:.0f}s)  running FDP_out "
            f"median={s['fdp_out_median']:.3f} p90={s['fdp_out_p90']:.3f}",
            flush=True,
        )
    s = summarise_stage2(done, mcs_step)
    checkpoint(
        "stage2_decoy",
        variant,
        {"realizations": done, "summary": s, "exclude_branch": exclude_branch},
    )
    return s


def summarise_stage2(done: list[dict], mcs_step: int = 10) -> dict:
    t = cone()
    r = radius_of(t)
    prod = np.asarray(t["paper_reference_p06"], bool)
    prod_counts = annulus_counts(r, prod).astype(float)
    # Primary denominator: the same configuration that produced the decoy numerator
    # (stage 0). Falls back to production if stage 0 has not been run at this step.
    s0 = load_sidecar().get("stage0_fidelity", {}).get(f"step{mcs_step}", {})
    tgt = np.asarray(s0.get("reimplementation_counts", prod_counts), float)
    out = EDGES[:-1] >= R_J
    t_out = float(tgt[out].sum())
    t_all = float(tgt.sum())
    if not done:
        return {"n_realizations": 0, "fdp_out_median": float("nan"), "fdp_out_p90": float("nan")}
    C = np.array([d["counts"] for d in done], float)  # (J, nbin)
    d_out = C[:, out].sum(axis=1)
    d_all = C.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_bin_med = np.median(C, axis=0) / np.where(tgt > 0, tgt, np.nan)
        per_bin_p90 = np.percentile(C, 90, axis=0) / np.where(tgt > 0, tgt, np.nan)
    detected = np.array([d["detected"] for d in done], bool)
    frac_beyond = d_out[d_all > 0] / d_all[d_all > 0]
    return {
        "n_realizations": len(done),
        "n_detected": int(detected.sum()),
        "n_null_detections": int((d_all > 0).sum()),
        "target_counts": tgt.tolist(),
        "target_beyond_rJ": t_out,
        "target_total": t_all,
        "target_source": "stage0 reimplementation"
        if "reimplementation_counts" in s0
        else "production paper_reference_p06",
        "fdp_out_median_vs_production": float(np.median(d_out) / prod_counts[out].sum()),
        "fdp_out_p90_vs_production": float(np.percentile(d_out, 90) / prod_counts[out].sum()),
        "decoy_total_median": float(np.median(d_all)),
        "decoy_total_p90": float(np.percentile(d_all, 90)),
        "decoy_beyond_median": float(np.median(d_out)),
        "decoy_beyond_p90": float(np.percentile(d_out, 90)),
        "fdp_all_median": float(np.median(d_all) / t_all),
        "fdp_all_p90": float(np.percentile(d_all, 90) / t_all),
        "fdp_out_median": float(np.median(d_out) / t_out),
        "fdp_out_p90": float(np.percentile(d_out, 90) / t_out),
        "fdp_per_bin_median": [None if not np.isfinite(v) else float(v) for v in per_bin_med],
        "fdp_per_bin_p90": [None if not np.isfinite(v) else float(v) for v in per_bin_p90],
        # NEGATIVE CONTROL 1: where does a pure field put its "members"?
        "control_field_only_fraction_beyond_rJ_median": float(np.median(frac_beyond))
        if frac_beyond.size
        else None,
        "control_field_only_n_with_detection": int((d_all > 0).sum()),
        "area_fraction_beyond_rJ": float((CONE_RADIUS**2 - R_J**2) / CONE_RADIUS**2),
    }


# ----------------------------------------------------------------------------- stage 3
def sample_eff(n, gamma, a, rmax, rng):
    """Radii from a projected EFF profile Sigma ~ (1+(r/a)^2)^(-gamma/2), truncated at rmax.

    CDF(r) = [1-(1+(r/a)^2)^(1-gamma/2)] / [1-(1+(rmax/a)^2)^(1-gamma/2)], inverted exactly.
    Checked against the empirical CDF of its own draws in ``--selftest``.
    """
    q = 1.0 - gamma / 2.0
    norm = (1.0 + (rmax / a) ** 2) ** q - 1.0
    u = rng.random(n)
    return a * np.sqrt((1.0 + u * norm) ** (1.0 / q) - 1.0)


def stage3(n_inject: int, mcs_step: int, seed: int) -> dict:
    """Negative control 2: a bound cluster truncated at r_J, no halo, same pipeline."""
    t = cone()
    r = radius_of(t)
    prod = np.asarray(t["paper_reference_p06"], bool)
    inner = prod & (r <= R_J)
    pmra = np.asarray(t["pmra"], float)
    pmdec = np.asarray(t["pmdec"], float)
    e_pmra = np.asarray(t["pmra_error"], float)
    e_pmdec = np.asarray(t["pmdec_error"], float)
    plx = np.asarray(t["parallax"], float)

    # Match the real cluster: centroid, intrinsic PM dispersion (observed minus median error
    # in quadrature), parallax centroid.
    c_pmra, c_pmdec = float(np.nanmedian(pmra[inner])), float(np.nanmedian(pmdec[inner]))
    obs_sd = np.array([np.nanstd(pmra[inner]), np.nanstd(pmdec[inner])])
    med_err = np.array([np.nanmedian(e_pmra[inner]), np.nanmedian(e_pmdec[inner])])
    intr = np.sqrt(np.maximum(obs_sd**2 - med_err**2, 1e-4))
    c_plx = float(np.nanmedian(plx[inner]))
    plx_sd = float(np.nanstd(plx[inner]))

    rng = np.random.default_rng(seed)
    decoy, _ = build_decoy(t, rng)  # neutralise the real cluster first

    # RECIPIENTS ARE THE REAL MEMBERS' OWN ROWS, NOT RANDOM CATALOGUE ROWS.
    # The first version drew recipients uniformly from the catalogue and added each
    # recipient's own proper-motion error on top of the intrinsic dispersion. Random rows are
    # mostly faint, so their errors are far larger than the real (bright) members' -- the
    # injected cluster came out smeared, the pipeline never found it (**recall = 0.000**, the
    # selected branch 2.19 mas/yr from the reference proper motion), and the control returned
    # "73.3% beyond r_J" computed over a 15-star chance blob. That would have produced a
    # MANUFACTURED verdict from an experiment that never ran. Reusing the members' own rows
    # keeps their magnitudes, astrometric precision and photometry exactly, so the *only*
    # synthetic thing is the spatial truth -- which is the one thing this control must control.
    # ``recall`` is reported first for exactly this reason: a control that did not recover its
    # own injection reports nothing about the pipeline.
    pool_idx = np.flatnonzero(inner) if n_inject <= int(inner.sum()) else np.flatnonzero(prod)
    idx = rng.choice(pool_idx, size=min(n_inject, pool_idx.size), replace=False)
    n_inject = int(idx.size)

    rad = sample_eff(n_inject, EFF_GAMMA, EFF_A, R_J, rng)
    th = rng.random(n_inject) * 2 * np.pi
    cosd = np.cos(np.deg2rad(CENTER_DEC))
    decoy["ra"][idx] = CENTER_RA + (rad / 60.0) * np.cos(th) / cosd
    decoy["dec"][idx] = CENTER_DEC + (rad / 60.0) * np.sin(th)
    decoy["pmra"][idx] = (
        c_pmra + rng.normal(0, intr[0], n_inject) + rng.normal(0, e_pmra[idx], n_inject)
    )
    decoy["pmdec"][idx] = (
        c_pmdec + rng.normal(0, intr[1], n_inject) + rng.normal(0, e_pmdec[idx], n_inject)
    )
    decoy["parallax"][idx] = c_plx + rng.normal(0, plx_sd, n_inject)

    dra = (np.asarray(decoy["ra"], float) - CENTER_RA) * cosd
    ddec = np.asarray(decoy["dec"], float) - CENTER_DEC
    r_inj = np.hypot(dra, ddec) * 60.0
    truth = np.zeros(len(t), bool)
    truth[idx] = True

    members, info = run_pipeline(decoy, mcs_step)
    rr = r_inj[members]
    n_mem = int(members.sum())
    res = {
        "n_inject": n_inject,
        "recall_READ_THIS_FIRST": float((members & truth).sum() / n_inject),
        "truth_beyond_rJ": int((r_inj[truth] > R_J).sum()),  # must be 0 by construction
        "truth_max_radius": float(r_inj[truth].max()),
        "injected_pm": [c_pmra, c_pmdec],
        "injected_pm_intrinsic_sd": intr.tolist(),
        "injected_parallax": [c_plx, plx_sd],
        "eff_gamma": EFF_GAMMA,
        "eff_a": EFF_A,
        "truncation_arcmin": R_J,
        "n_members": n_mem,
        "n_members_beyond_rJ": int((rr > R_J).sum()),
        "fraction_beyond_rJ": float((rr > R_J).mean()) if n_mem else None,
        "recovered_true": int((members & truth).sum()),
        "recall": float((members & truth).sum() / n_inject),
        "contamination": float(1.0 - (members & truth).sum() / n_mem) if n_mem else None,
        "counts": annulus_counts(r_inj, members).tolist(),
        **info,
    }
    checkpoint("stage3_injection", f"n{n_inject}", res)
    return res


# ----------------------------------------------------------------------------- stage 4
HUNT_URL = (
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=J/A+A/686/A42/clusters"
    "&-out=Name,RA_ICRS,DE_ICRS,pmRA,pmDE,Plx,rJ,rt,r50,rtot,MassTot"
    "&-out.max=unlimited&-c=263.6826%20-32.5838&-c.rd={rd}"
)


def hunt_neighbours(rd_deg: float = 1.6, d_pm: float = 1.2, d_plx: float = 0.12):
    """Live VizieR pull of Hunt & Reffert 2024 clusters that share NGC 6383's astrometry.

    Live rather than cached, per this directory's rule -- a stale catalogue must not silently
    persist. ``d_pm`` / ``d_plx`` are the co-moving tolerances; both are reported so a reader
    can see how much of the result depends on them.
    """
    import urllib.request

    raw = urllib.request.urlopen(HUNT_URL.format(rd=rd_deg), timeout=180).read()
    lines = [
        ln for ln in raw.decode("utf-8", "replace").splitlines() if ln and not ln.startswith("#")
    ]
    hdr = lines[0].split("\t")
    recs = [ln.split("\t") for ln in lines[3:] if ln.strip()]
    recs = [[c.strip() for c in r] for r in recs if len(r) == len(hdr)]

    def col(rec, key):
        try:
            return float(rec[hdr.index(key)])
        except Exception:
            return float("nan")

    ref = [r for r in recs if r[hdr.index("Name")] == "NGC_6383"]
    if not ref:
        raise RuntimeError("NGC 6383 absent from the Hunt & Reffert pull -- query changed?")
    ref = ref[0]
    ref_pm = (col(ref, "pmRA"), col(ref, "pmDE"))
    ref_plx = col(ref, "Plx")
    cosd = np.cos(np.deg2rad(CENTER_DEC))
    out = []
    for rec in recs:
        name = rec[hdr.index("Name")]
        ra, dec = col(rec, "RA_ICRS"), col(rec, "DE_ICRS")
        if not np.isfinite(ra) or name == "NGC_6383":
            continue
        dpm = float(np.hypot(col(rec, "pmRA") - ref_pm[0], col(rec, "pmDE") - ref_pm[1]))
        dplx = abs(col(rec, "Plx") - ref_plx)
        if not (dpm <= d_pm and dplx <= d_plx):
            continue
        rtot = col(rec, "rtot") * 60.0
        out.append(
            {
                "name": name,
                "ra": ra,
                "dec": dec,
                "separation_arcmin": float(
                    np.hypot((ra - CENTER_RA) * cosd, dec - CENTER_DEC) * 60
                ),
                "delta_pm": dpm,
                "delta_parallax": dplx,
                "pm": [col(rec, "pmRA"), col(rec, "pmDE")],
                "parallax": col(rec, "Plx"),
                "rJ_arcmin": col(rec, "rJ") * 60.0,
                "rtot_arcmin": rtot,
            }
        )
    return out, {
        "reference_pm": list(ref_pm),
        "reference_parallax": ref_plx,
        "d_pm_tolerance": d_pm,
        "d_parallax_tolerance": d_plx,
        "search_radius_deg": rd_deg,
    }


def stage4() -> dict:
    """Attribution: how much of the 'halo' is a *catalogued neighbouring cluster*?

    Hypothesis (c), which the FDP cannot see. The decoy null is a structureless field by
    construction, so it contains no comoving neighbours and its false-discovery proportion is a
    floor, not a total (``decoy_fdp.py`` says exactly this). A proper-motion-only membership
    admits any comoving structure on the sightline, and sky position -- the axis the
    membership never used -- is what reveals it.
    """
    t = cone()
    r = radius_of(t)
    prod = np.asarray(t["paper_reference_p06"], bool)
    ra = np.asarray(t["ra"], float)
    dec = np.asarray(t["dec"], float)
    cosd = np.cos(np.deg2rad(CENTER_DEC))
    outer = prod & (r > R_J) & (r <= CONE_RADIUS)
    inner = prod & (r <= R_J)

    neighbours, meta = hunt_neighbours()
    assigned = np.zeros(len(t), bool)
    per = []
    for nb in neighbours:
        dist = np.hypot((ra - nb["ra"]) * cosd, dec - nb["dec"]) * 60.0
        radius = nb["rtot_arcmin"] if np.isfinite(nb["rtot_arcmin"]) else nb["rJ_arcmin"]
        inside = outer & (dist < radius)
        assigned |= inside
        per.append(
            {
                **nb,
                "matched_radius_arcmin": radius,
                "outer_members_inside": int(inside.sum()),
                "outer_members_inside_rJ": int((outer & (dist < nb["rJ_arcmin"])).sum()),
            }
        )

    # THE CONTROL THIS CLAIM NEEDS. "66% of the halo sits inside a neighbour" is worthless
    # without knowing how much of the region those circles cover -- a big enough circle
    # contains everything. Two parameter-free nulls: the *catalogue* stars beyond r_J inside
    # the same circles (which carries the real density and extinction structure), and the plain
    # geometric area fraction. The members must beat the catalogue rate or the attribution is
    # an artefact of circle size, and the claim must be withdrawn.
    cat_outer = (r > R_J) & (r <= CONE_RADIUS)
    cat_inside = np.zeros(len(t), bool)
    for nb in neighbours:
        dist = np.hypot((ra - nb["ra"]) * cosd, dec - nb["dec"]) * 60.0
        radius = nb["rtot_arcmin"] if np.isfinite(nb["rtot_arcmin"]) else nb["rJ_arcmin"]
        cat_inside |= dist < radius
    p0 = float((cat_outer & cat_inside).sum() / max(int(cat_outer.sum()), 1))

    rng = np.random.default_rng(7)
    n_mc = 400_000
    rad_mc = CONE_RADIUS * np.sqrt(rng.random(n_mc))
    th_mc = rng.random(n_mc) * 2 * np.pi
    px, py = rad_mc * np.cos(th_mc), rad_mc * np.sin(th_mc)
    cov = np.zeros(n_mc, bool)
    for nb in neighbours:
        cx = (nb["ra"] - CENTER_RA) * cosd * 60.0
        cy = (nb["dec"] - CENTER_DEC) * 60.0
        radius = nb["rtot_arcmin"] if np.isfinite(nb["rtot_arcmin"]) else nb["rJ_arcmin"]
        cov |= np.hypot(px - cx, py - cy) < radius
    area_cov = float(cov[rad_mc > R_J].mean())

    from scipy import stats as _st

    k_in, n_out = int(assigned.sum()), int(outer.sum())
    attribution_control = {
        "catalogue_outer_n": int(cat_outer.sum()),
        "catalogue_outer_inside_circles": int((cat_outer & cat_inside).sum()),
        "catalogue_inside_fraction_NULL": p0,
        "geometric_area_coverage_beyond_rJ": area_cov,
        "members_inside_fraction": float(k_in / max(n_out, 1)),
        "expected_inside_if_field_like": float(n_out * p0),
        "excess_over_field_expectation": float(k_in - n_out * p0),
        "binomial_p": float(_st.binom.sf(k_in - 1, n_out, p0)) if n_out else None,
    }

    residual = outer & ~assigned
    plx = np.asarray(t["parallax"], float)
    pmra = np.asarray(t["pmra"], float)
    pmdec = np.asarray(t["pmdec"], float)

    def astro(mask):
        return {
            "n": int(mask.sum()),
            "parallax_median": float(np.nanmedian(plx[mask])),
            "parallax_sd": float(np.nanstd(plx[mask])),
            "pm_median": [float(np.nanmedian(pmra[mask])), float(np.nanmedian(pmdec[mask]))],
            "pm_sd": [float(np.nanstd(pmra[mask])), float(np.nanstd(pmdec[mask]))],
        }

    # Does an asymmetry survive after the catalogued neighbours are removed?
    from scipy import stats

    th = np.asarray(t["b1_theta"], float)
    azb = np.linspace(0, 360, 13)
    beyond = (r > R_J) & (r <= CONE_RADIUS)
    cat_az = np.histogram(th[beyond], bins=azb)[0]
    res_az = np.histogram(th[residual], bins=azb)[0]
    exp_az = cat_az / cat_az.sum() * max(res_az.sum(), 1)
    chi2 = float(np.sum((res_az - exp_az) ** 2 / np.maximum(exp_az, 1e-9)))

    # PHOTOMETRY -- the one axis the membership never touched at all (the clustering is
    # proper-motion-only and the clip is parallax-only). An empirical pre-main-sequence ridge
    # is built from the *inner* members and every other sample is scored against it
    # out-of-sample. Colour cannot separate "NGC 6383 halo" from "neighbour member" -- same
    # complex, same distance, same age -- so it is not used for attribution; it answers the
    # different question of whether a sample is a young population at all.
    cmd = {}
    try:
        col = np.asarray(t["G_BPmag"], float) - np.asarray(t["G_RPmag"], float)
        gmag = np.asarray(t["Gmag"], float)
        okp = np.isfinite(col) & np.isfinite(gmag)
        gbins = np.arange(10.0, 19.5, 0.75)
        gmid = 0.5 * (gbins[1:] + gbins[:-1])
        ridge = np.full(len(gmid), np.nan)
        # strict=True: both slices come from one `gbins`, so a length mismatch would mean the
        # array changed under us -- an error, not something to truncate silently.
        for i, (a, b) in enumerate(zip(gbins[:-1], gbins[1:], strict=True)):
            s = inner & okp & (gmag >= a) & (gmag < b)
            if s.sum() >= 5:
                ridge[i] = np.median(col[s])

        def offsets(mask):
            idx = np.digitize(gmag[mask], gbins) - 1
            v = np.full(int(mask.sum()), np.nan)
            s = (idx >= 0) & (idx < len(gmid))
            v[s] = col[mask][s] - ridge[idx[s]]
            return v[np.isfinite(v)]

        band = (plx >= 0.7839647672) & (plx <= 1.0432678929)  # production clip band
        field_ctrl = cat_outer & band & ~prod & okp
        samples = {
            "inner_IN_SAMPLE": inner & okp,
            "attributed": assigned & okp,
            "residual": residual & okp,
            "field_control": field_ctrl,
        }
        offs = {k: offsets(v) for k, v in samples.items()}
        cmd = {
            "ridge_gmag": gmid.tolist(),
            "ridge_colour": ridge.tolist(),
            "note": "inner is IN-SAMPLE (it defines the ridge); the rest are out-of-sample",
        }
        for k, v in offs.items():
            cmd[k] = {
                "n": int(v.size),
                "median_offset": float(np.median(v)),
                "mad": float(_st.median_abs_deviation(v)),
                "frac_within_0.15": float(np.mean(np.abs(v) < 0.15)),
            }
        for a, b in (
            ("residual", "field_control"),
            ("residual", "inner_IN_SAMPLE"),
            ("attributed", "inner_IN_SAMPLE"),
            ("attributed", "field_control"),
        ):
            ks = _st.ks_2samp(offs[a], offs[b])
            cmd[f"ks_{a}_vs_{b}"] = {"D": float(ks.statistic), "p": float(ks.pvalue)}
    except Exception as exc:  # photometry columns absent -> say so
        cmd = {"error": repr(exc)}

    res = {
        **meta,
        "neighbours": per,
        "attribution_control": attribution_control,
        "cmd_ridge_test": cmd,
        "n_outer": int(outer.sum()),
        "n_outer_attributed_to_neighbour": int(assigned.sum()),
        "fraction_of_halo_attributed": float(assigned.sum() / max(outer.sum(), 1)),
        "n_outer_residual": int(residual.sum()),
        "residual_fraction_of_members": float(residual.sum() / int(prod.sum())),
        "astrometry_inner": astro(inner),
        "astrometry_outer": astro(outer),
        "astrometry_attributed": astro(assigned),
        "astrometry_residual": astro(residual),
        "residual_azimuth_chi2": chi2,
        "residual_azimuth_dof": 11,
        "residual_azimuth_p": float(stats.chi2.sf(chi2, 11)),
        "residual_counts": annulus_counts(r, residual).tolist(),
    }
    checkpoint("stage4_attribution", "result", res)
    return res


# ----------------------------------------------------------------------------- verdict
def verdict(payload: dict) -> dict:
    s2root = payload.get("stage2_decoy", {})
    variants = {
        k: v.get("summary", {}) for k, v in s2root.items() if isinstance(v, dict) and "summary" in v
    }
    if not variants and "summary" in s2root:  # pre-variant sidecar layout
        variants = {"fullpool": s2root["summary"]}
    s0 = payload.get("stage0_fidelity", {})
    inj = payload.get("stage3_injection", {})
    # The primary null is the true-positive-free one; the sensitivity must agree in sign.
    primary = variants.get("nopool_branch") or next(iter(variants.values()), {})
    med = primary.get("fdp_out_median")
    p90 = primary.get("fdp_out_p90")
    meds = {k: v.get("fdp_out_median") for k, v in variants.items()}
    # An injection with no recall never tested the pipeline -- its fraction is measured over a
    # chance blob. Requiring recall >= 0.20 is what stops a failed control voting.
    inj_fracs = [
        v.get("fraction_beyond_rJ")
        for v in inj.values()
        if v.get("fraction_beyond_rJ") is not None and (v.get("recall") or 0.0) >= 0.20
    ]
    inj_failed = [k for k, v in inj.items() if (v.get("recall") or 0.0) < 0.20]
    worst_inj = max(inj_fracs) if inj_fracs else None
    gate = any(v.get("gate_passed") for v in s0.values()) if s0 else None

    reasons = []
    if med is None or not np.isfinite(med):
        call = "PENDING"
        reasons.append("stage 2 has no realizations yet")
    elif med >= 0.20:
        call = "MANUFACTURED"
        reasons.append(f"FDP_out median {med:.3f} >= 0.20 (Pang threshold)")
    elif worst_inj is not None and worst_inj >= 0.20:
        call = "MANUFACTURED"
        reasons.append(
            f"injection control puts {worst_inj:.3f} beyond r_J from a truth "
            f"truncated at r_J (>= 0.20)"
        )
    elif p90 is not None and p90 >= 0.50:
        call = "UNDETERMINED"
        reasons.append(f"FDP_out median {med:.3f} < 0.20 but p90 {p90:.3f} >= 0.50")
    elif worst_inj is None:
        call = "UNDETERMINED"
        reasons.append("injection control (negative control 2) has not run")
    else:
        call = "NOT-MANUFACTURED"
        reasons.append(
            f"FDP_out median {med:.3f} < 0.20, p90 {p90:.3f} < 0.50, "
            f"injection control {worst_inj:.3f} < 0.20"
        )
    # Hypothesis (c). The verdict rule was written to decide between "real halo" and
    # "manufactured by field contamination". It has no term for a *third* real structure --
    # a comoving neighbour that a proper-motion-only membership admits -- and the decoy null
    # cannot contain one by construction. So a low FDP licenses "not manufactured by field
    # contamination"; it does NOT license "this is NGC 6383's halo". Stage 4 measures the
    # attribution and, if it is material, the verdict is downgraded here rather than the rule
    # being quietly rewritten.
    att = payload.get("stage4_attribution", {}).get("result", {})
    frac_att = att.get("fraction_of_halo_attributed")

    live = {k: v for k, v in meds.items() if v is not None and np.isfinite(v)}
    if call in ("NOT-MANUFACTURED", "MANUFACTURED") and len(live) > 1:
        sides = {v >= 0.20 for v in live.values()}
        if len(sides) > 1:
            call = "UNDETERMINED"
            reasons.append(
                "the two decoy nulls straddle the 0.20 threshold: "
                + ", ".join(f"{k}={v:.3f}" for k, v in sorted(live.items()))
            )
    if frac_att is not None and frac_att >= 0.20 and call != "MANUFACTURED":
        call = "NOT-NGC-6383"
        reasons.append(
            f"{frac_att:.1%} of the members beyond r_J fall inside the total radius of a "
            f"CATALOGUED comoving neighbour (Hunt & Reffert 2024). The 'halo' is neither a "
            f"field-contamination artefact nor established as NGC 6383's own structure -- it "
            f"is substantially a different cluster, admitted by a proper-motion-only "
            f"membership. The FDP cannot see this: its null is a structureless field."
        )
    if inj_failed:
        reasons.append(
            f"injection run(s) {sorted(inj_failed)} recovered <20% of their own "
            f"injected truth and are EXCLUDED from the verdict -- a control that "
            f"did not find its own cluster measures nothing"
        )
    if gate is False:
        reasons.append(
            "STAGE 0 GATE FAILED: the reimplementation does not reproduce the "
            "production membership, so the verdict rule is NOT applied"
        )
        call = "INSTRUMENT-INVALID"
    return {
        "verdict": call,
        "reasons": reasons,
        "fdp_out_median": med,
        "fdp_out_p90": p90,
        "fdp_out_median_by_variant": meds,
        "fraction_of_halo_attributed": frac_att,
        "injection_worst_fraction_beyond_rJ": worst_inj,
        "stage0_gate": gate,
        "scope_note": "The FDP rule decides hypothesis (b) -- manufactured by field "
        "contamination -- ONLY. 'NOT-MANUFACTURED' is not 'real halo'.",
    }


def finalise(mcs_step: int = 10) -> dict:
    """Migrate the pre-variant sidecar layout and re-summarise both nulls consistently.

    The ``fullpool`` realizations were written before the variant keying existed, and before
    stage 0 recorded ``reimplementation_counts`` -- so their FDP was divided by production's
    271 rather than by the 240 the *same configuration* selects. Dividing a numerator from one
    pipeline by a denominator from another understates the FDP by 13%. Re-summarising fixes it
    without re-running anything.
    """
    payload = load_sidecar()
    s2 = payload.get("stage2_decoy", {})
    if "realizations" in s2:  # pre-variant layout -> fullpool
        s2["fullpool"] = {"realizations": s2.pop("realizations"), "exclude_branch": False}
        s2.pop("summary", None)
    for _name, blob in s2.items():
        if isinstance(blob, dict) and "realizations" in blob:
            blob["summary"] = summarise_stage2(blob["realizations"], mcs_step)
    payload["stage2_decoy"] = s2
    payload["verdict"] = {"current": verdict(payload)}
    payload["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_sidecar(payload)
    return payload["verdict"]["current"]


def selftest() -> None:
    """Parameter-free checks on the generators. A test that cannot fail is worse than none."""
    rng = np.random.default_rng(0)
    # EFF sampler against its own analytic CDF.
    for gamma, a, rmax in ((2.322, 1.649, 37.9), (4.0, 5.0, 20.0), (3.0, 1.0, 100.0)):
        x = sample_eff(200000, gamma, a, rmax, rng)
        assert x.max() <= rmax + 1e-9, "EFF sampler broke its truncation"
        q = 1.0 - gamma / 2.0
        grid = np.linspace(0.01, rmax * 0.999, 40)
        cdf = ((1 + (grid / a) ** 2) ** q - 1) / ((1 + (rmax / a) ** 2) ** q - 1)
        emp = np.searchsorted(np.sort(x), grid) / x.size
        dmax = float(np.abs(cdf - emp).max())
        assert dmax < 0.005, f"EFF sampler vs analytic CDF: D={dmax:.4f} (gamma={gamma})"
        print(f"  EFF sampler gamma={gamma} a={a} rmax={rmax}: max|CDF-ECDF| = {dmax:.5f}")
    # Annulus areas against the closed form, and edges covering the cone exactly.
    assert np.isclose(np.sum(np.pi * (EDGES[1:] ** 2 - EDGES[:-1] ** 2)), np.pi * CONE_RADIUS**2)
    # The area fraction beyond r_J, the baseline the whole report leans on.
    af = (CONE_RADIUS**2 - R_J**2) / CONE_RADIUS**2
    assert abs(af - 0.7068) < 1e-3, af
    print(f"  area fraction beyond r_J = {af:.4f}  (a uniform contaminant's 'halo fraction')")
    # The verdict rule must be able to return each of its values.
    for med, p90, inj, want in (
        (0.30, 0.9, 0.05, "MANUFACTURED"),
        (0.05, 0.9, 0.05, "UNDETERMINED"),
        (0.05, 0.2, 0.50, "MANUFACTURED"),
        (0.05, 0.2, 0.05, "NOT-MANUFACTURED"),
    ):
        got = verdict(
            {
                "stage2_decoy": {"summary": {"fdp_out_median": med, "fdp_out_p90": p90}},
                "stage3_injection": {"x": {"fraction_beyond_rJ": inj, "recall": 0.8}},
            }
        )["verdict"]
        assert got == want, f"verdict rule: ({med},{p90},{inj}) -> {got}, expected {want}"
    print("  verdict rule reaches MANUFACTURED / UNDETERMINED / NOT-MANUFACTURED")
    straddle = verdict(
        {
            "stage2_decoy": {
                "nopool_branch": {"summary": {"fdp_out_median": 0.05, "fdp_out_p90": 0.2}},
                "fullpool": {"summary": {"fdp_out_median": 0.40, "fdp_out_p90": 0.9}},
            },
            "stage3_injection": {"x": {"fraction_beyond_rJ": 0.05, "recall": 0.8}},
        }
    )["verdict"]
    assert straddle == "UNDETERMINED", straddle
    print("  two decoy nulls straddling 0.20 -> UNDETERMINED")
    downgraded = verdict(
        {
            "stage2_decoy": {
                "nopool_branch": {"summary": {"fdp_out_median": 0.05, "fdp_out_p90": 0.2}}
            },
            "stage3_injection": {"x": {"fraction_beyond_rJ": 0.05, "recall": 0.8}},
            "stage4_attribution": {"result": {"fraction_of_halo_attributed": 0.66}},
        }
    )["verdict"]
    assert downgraded == "NOT-NGC-6383", downgraded
    kept = verdict(
        {
            "stage2_decoy": {
                "nopool_branch": {"summary": {"fdp_out_median": 0.05, "fdp_out_p90": 0.2}}
            },
            "stage3_injection": {"x": {"fraction_beyond_rJ": 0.05, "recall": 0.8}},
            "stage4_attribution": {"result": {"fraction_of_halo_attributed": 0.02}},
        }
    )["verdict"]
    assert kept == "NOT-MANUFACTURED", kept
    print("  neighbour attribution downgrades NOT-MANUFACTURED -> NOT-NGC-6383 at >=20%")
    failed = verdict(
        {
            "stage2_decoy": {
                "nopool_branch": {"summary": {"fdp_out_median": 0.05, "fdp_out_p90": 0.2}}
            },
            "stage3_injection": {"n355": {"fraction_beyond_rJ": 0.73, "recall": 0.0}},
        }
    )["verdict"]
    assert failed == "UNDETERMINED", failed
    passed = verdict(
        {
            "stage2_decoy": {
                "nopool_branch": {"summary": {"fdp_out_median": 0.05, "fdp_out_p90": 0.2}}
            },
            "stage3_injection": {"n355": {"fraction_beyond_rJ": 0.73, "recall": 0.8}},
        }
    )["verdict"]
    assert passed == "MANUFACTURED", passed
    print("  a zero-recall injection cannot vote; the same fraction with recall does")
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--stage", type=int, choices=[0, 1, 2, 3, 4])
    ap.add_argument("--mcs-step", type=int, default=10)
    ap.add_argument("--realizations", type=int, default=24)
    ap.add_argument("--n-inject", type=int, default=355)
    ap.add_argument("--seed", type=int, default=20260804)
    ap.add_argument(
        "--include-branch-in-pool",
        action="store_true",
        help="sensitivity variant: leave the production branch in the decoy donor "
        "pool, as decoy_fdp.py does (biases the FDP up)",
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument(
        "--finalise",
        action="store_true",
        help="migrate the sidecar layout and re-summarise both nulls; no compute",
    )
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if args.finalise:
        v = finalise(args.mcs_step)
        print(json.dumps(jsonable(v), indent=1))
        return
    if args.report:
        d = load_sidecar()
        g = d.get("stage1_geometry", {}).get("result", {})
        print(
            f"members {g.get('n_members')}, beyond r_J {g.get('n_members_beyond_rJ')} "
            f"({g.get('fraction_beyond_rJ', float('nan')):.1%}); a uniform contaminant would "
            f"put {g.get('area_fraction_beyond_rJ', float('nan')):.1%} there"
        )
        for k, v in d.get("stage2_decoy", {}).items():
            sm = v.get("summary", {})
            print(
                f"decoy[{k}] {sm.get('n_realizations')} realizations: FDP_out median "
                f"{sm.get('fdp_out_median', float('nan')):.3f} p90 "
                f"{sm.get('fdp_out_p90', float('nan')):.3f}; field-only puts "
                f"{sm.get('control_field_only_fraction_beyond_rJ_median') or float('nan'):.1%} "
                f"of its own selections beyond r_J"
            )
        for k, v in d.get("stage3_injection", {}).items():
            print(
                f"injection[{k}] recall {v.get('recall', float('nan')):.3f}, "
                f"{v.get('n_members')} recovered, beyond r_J {v.get('n_members_beyond_rJ')} "
                f"(truth beyond r_J = {v.get('truth_beyond_rJ')})"
            )
        a = d.get("stage4_attribution", {}).get("result", {})
        if a:
            c = a["attribution_control"]
            print(
                f"attribution: {a['n_outer_attributed_to_neighbour']}/{a['n_outer']} "
                f"({a['fraction_of_halo_attributed']:.1%}) inside a catalogued comoving "
                f"neighbour vs a {c['catalogue_inside_fraction_NULL']:.1%} catalogue null "
                f"(p={c['binomial_p']:.1e})"
            )
        v = verdict(d)
        print("\nVERDICT:", v["verdict"])
        for reason in v["reasons"]:
            print("  -", reason)
        return
    if args.stage == 0:
        r = stage0(args.mcs_step)
        print(json.dumps(jsonable({k: v for k, v in r.items() if k != "counts"}), indent=1))
    elif args.stage == 1:
        r = stage1()
        print(
            f"fraction beyond r_J = {r['fraction_beyond_rJ']:.4f} "
            f"(area fraction {r['area_fraction_beyond_rJ']:.4f})"
        )
        print(f"outer log-slope = {r['outer_logslope']:.3f} +/- {r['outer_logslope_err']:.3f}")
        print(
            f"azimuth chi2 = {r['azimuth_chi2']:.1f} / {r['azimuth_dof']} dof, p = "
            f"{r['azimuth_p']:.3e}"
        )
    elif args.stage == 2:
        r = stage2(
            args.realizations,
            args.mcs_step,
            args.seed,
            exclude_branch=not args.include_branch_in_pool,
        )
        print(json.dumps(jsonable(r), indent=1))
    elif args.stage == 3:
        r = stage3(args.n_inject, args.mcs_step, args.seed)
        print(json.dumps(jsonable(r), indent=1))
    elif args.stage == 4:
        r = stage4()
        print(
            f"outer members = {r['n_outer']}; inside a catalogued comoving neighbour = "
            f"{r['n_outer_attributed_to_neighbour']} ({r['fraction_of_halo_attributed']:.1%})"
        )
        for nb in r["neighbours"]:
            print(
                f"  {nb['name']:<14} sep={nb['separation_arcmin']:6.1f}'  "
                f"dpm={nb['delta_pm']:.3f}  dplx={nb['delta_parallax']:.3f}  "
                f"r_tot={nb['matched_radius_arcmin']:5.1f}'  -> {nb['outer_members_inside']:4d}"
            )
        c = r["attribution_control"]
        print(
            f"  NULL: outer catalogue stars inside the same circles = "
            f"{c['catalogue_inside_fraction_NULL']:.1%} (area coverage "
            f"{c['geometric_area_coverage_beyond_rJ']:.1%})"
        )
        print(
            f"  excess over that null = {c['excess_over_field_expectation']:.0f} stars, "
            f"binomial p = {c['binomial_p']:.2e}"
        )
        print(
            f"residual (unattributed) beyond r_J = {r['n_outer_residual']}, azimuth p = "
            f"{r['residual_azimuth_p']:.2e}"
        )
        c2 = r.get("cmd_ridge_test", {})
        if "error" not in c2:
            for k in ("inner_IN_SAMPLE", "attributed", "residual", "field_control"):
                v = c2[k]
                print(
                    f"  CMD {k:<16s} n={v['n']:5d}  median offset {v['median_offset']:+.3f}  "
                    f"MAD {v['mad']:.3f}  within 0.15: {v['frac_within_0.15']:.1%}"
                )
            for k, v in c2.items():
                if k.startswith("ks_"):
                    print(f"  {k:<44s} D={v['D']:.3f}  p={v['p']:.2e}")

    payload = load_sidecar()
    v = verdict(payload)
    checkpoint("verdict", "current", v)
    print("\nVERDICT:", v["verdict"])
    for reason in v["reasons"]:
        print("  -", reason)


if __name__ == "__main__":
    main()
