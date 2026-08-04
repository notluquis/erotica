"""Head-to-head membership benchmark: EROTICA vs ASteCA vs pyUPMASK, on synthetic truth.

WHY THIS EXISTS
---------------
``erotica/analysis/external/benchmark.py`` scores four axes but had **never been run**, so
every superiority claim in the paper was a claim about capability. This script is the run.
It generates synthetic cluster+field mixtures where membership truth is known by
construction, pushes the *same* rows through every pipeline, and scores them with the
shared harness (``erotica.analysis.external.benchmark``) so the comparison is
apples-to-apples.

pyUPMASK is included (``--pyupmask``). It genuinely has no PyPI release and its repository
is archived, but it is still *runnable* as a script on the current stack -- see
``benchmark_pyupmask_runner`` for the verification and the co-indexing contract.

Only two of the four axes admit a ranking: **calibration** and **parameter recovery** have
ground truth. Agreement has none and is reported as descriptive. Runtime is context.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
The headline is expected to be that EROTICA's raw pseudo-probability
``p̃ = p_HDBSCAN · f_i`` is *badly calibrated* while its member selection is competitive,
and that isotonic recalibration (which EROTICA ships, ASteCA does not) fixes it. That is
falsified if any of the following hold:

* ASteCA's ``fastMP`` resample frequency has a **worse** out-of-sample ECE than raw ``p̃``
  at the same contamination — then the "EROTICA needs its own recalibrator" framing is
  wrong and ASteCA is the one that needs it;
* out-of-sample recalibrated ECE is **not** below raw ECE — then the recalibration module
  does not earn its place and the fix is illusory;
* EROTICA's recovered ``sigma_pm`` is biased **high** relative to ASteCA's at every
  contamination level — that would mean the member lists are less pure, not more, and no
  calibration story rescues that.

A single realisation cannot decide any of these. Every cell is repeated over
``--realisations`` independent seeds and the tables carry the across-seed standard error.

DESIGN DECISIONS THAT ARE NOT FREE
----------------------------------
1. **Feature parity.** ``run_erotica``'s default is ``("pmra", "pmdec")`` — the NGC 6383
   paper configuration — but ASteCA's ``fastMP`` consumes ``(ra, dec, pmra, pmde, plx)``
   plus per-star errors, and pyUPMASK is configured here on the same five (``_x _y`` +
   ``pmRA pmDE Plx``). Scoring 2D against 5D and calling the difference a win would be
   dishonest. The headline EROTICA row is therefore ``erotica_5d`` on the same five
   columns; ``erotica_pm`` (2D) and ``erotica_3d`` (PM + parallax) are reported as
   clearly-labelled secondary rows.
2. **Standardisation.** ``HDBSCANEstimator`` feeds ``.to_pandas().values`` straight into
   ``hdbscan.HDBSCAN(metric="euclidean")`` with **no** internal scaling (verified in
   ``erotica/core/_estimator.py``). Parallax in mas and proper motion in mas/yr have
   different numeric ranges, so the unscaled distance is dominated by whichever has the
   larger spread. Every column handed to ``Clustering`` here is z-scored first.
3. **ASteCA is not handed the truth.** ``radec_c`` / ``pms_c`` / ``plx_c`` / ``N_cluster``
   come from ``Cluster.get_center(algo="knn_5d")`` and ``Cluster.get_nmembers("ripley")``,
   never from the injected values. Otherwise an EROTICA loss would be uninterpretable.
4. **No photometry is passed to ASteCA.** ``fastMP`` ignores it and ``Cluster._max_mag_cut``
   would drop rows, breaking the co-indexing the harness requires.
5. **Recalibration is out-of-sample.** Isotonic is fitted on a disjoint set of seeds and
   evaluated on held-out seeds. Fitting and scoring on the same rows returns ECE ~ 0 and
   is a fabricated number.

NEGATIVE CONTROLS AND NULLS (run, and reported)
-----------------------------------------------
* **Field-only control**: a realisation with no cluster injected at all. If the mcs sweep
  manufactures a cluster out of pure field, that is a more important result than any win.
* **Generator self-check**: the realised contamination fraction and the realised cluster
  centroid are asserted against the requested values before anything downstream is
  believed (generator -> estimator -> interpretation, per ``tools/validation/CLAUDE.md``).
* **Sweep-range sensitivity**: ``f_i`` is bounded above by the fraction of
  ``min_cluster_size`` values that recover *any* cluster, so the calibration of ``p̃``
  depends on the sweep range. One cell is re-run with a narrower sweep to size that
  dependence rather than leave it as an unquantified free parameter.

NO SAMPLED FIT IS RUN
---------------------
Parameter recovery here is a *downstream estimator applied identically to both member
lists* (means, deconvolved dispersion, count) — not a NUTS fit. No PyMC model is sampled,
so the r-hat/ESS/divergence convergence gate is inapplicable. Isochrone-based recovery of
age / extinction / distance modulus is deliberately out of scope: it needs PARSEC grids and
ASteCA's synthetic-cluster machinery, and it would measure isochrone fitting, not
membership.

ENVIRONMENT
-----------
Never run from mamba ``base``. Built for the isolated env ``erotica-bench``
(python 3.13, asteca 0.7.0, erotica installed with ``pip install -e . --no-deps``).

USAGE
-----
    python tools/validation/benchmark_erotica_vs_asteca.py --out benchmark_erotica_vs_asteca.json
    python tools/validation/benchmark_erotica_vs_asteca.py --quick      # smoke, 1 cell
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from dataclasses import dataclass
from importlib.metadata import version as _pkg_version
from pathlib import Path

import numpy as np
from astropy.table import QTable

from erotica.analysis.external import benchmark as bm
from erotica.analysis.synthetic import fractal_cluster

# ---------------------------------------------------------------------------
# Physical constants of the synthetic field
# ---------------------------------------------------------------------------
RA_C, DEC_C = 250.0, -32.0  # deg; NGC 6383-like, |dec| small enough that the
# local flat-sky approximation is harmless
FOV_RADIUS_DEG = 0.35  # deg; a typical OC cone
CLUSTER_RADIUS_DEG = 0.10  # deg; angular radius of the injected cluster
PMRA_C, PMDEC_C = -1.20, -0.50  # mas/yr, injected cluster proper motion
PLX_C = 1.00  # mas -> 1 kpc
SIGMA_PM_INT = 0.20  # mas/yr; ~1 km/s at 1 kpc (sigma_v / 4.74 d_kpc)
SIGMA_PLX_INT = 0.005  # mas; ~5 pc line-of-sight depth at 1 kpc
FIELD_PM_C = (-3.0, -3.0)  # mas/yr, field proper-motion centroid
FIELD_PM_SIGMA = 4.0  # mas/yr, field proper-motion spread
FIELD_L_KPC = 1.35  # exponentially-decreasing space density scale
# (Bailer-Jones et al. 2018 prior form)
G_MIN, G_MAX = 13.0, 19.0  # both populations share this G range on purpose:
# magnitude then carries no membership information

# Gaia DR3-like astrometric uncertainty vs G, log-linear between these anchors.
# Approximate, and stated as such: the benchmark is about membership assignment, not
# about reproducing the DR3 error model.
_G_ANCHORS = np.array([13.0, 15.0, 17.0, 19.0, 20.5])
_EPLX_ANCHORS = np.array([0.020, 0.025, 0.070, 0.220, 0.600])  # mas
_EPM_ANCHORS = np.array([0.020, 0.028, 0.075, 0.250, 0.700])  # mas/yr


def _astrometric_errors(gmag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(e_plx, e_pm) in mas / (mas/yr) from G, by log-linear interpolation."""
    g = np.clip(np.asarray(gmag, dtype=float), _G_ANCHORS[0], _G_ANCHORS[-1])
    e_plx = np.exp(np.interp(g, _G_ANCHORS, np.log(_EPLX_ANCHORS)))
    e_pm = np.exp(np.interp(g, _G_ANCHORS, np.log(_EPM_ANCHORS)))
    return e_plx, e_pm


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
@dataclass
class Realisation:
    ra: np.ndarray
    dec: np.ndarray
    pmra: np.ndarray
    pmdec: np.ndarray
    plx: np.ndarray
    e_pmra: np.ndarray
    e_pmdec: np.ndarray
    e_plx: np.ndarray
    gmag: np.ndarray
    truth: np.ndarray  # bool, True = injected cluster member
    n_members: int
    n_field: int
    contamination: float  # realised n_field / n_total

    def as_qtable(self, columns: dict[str, np.ndarray]) -> QTable:
        return QTable(columns)


def _sample_field_parallax(rng: np.random.Generator, n: int) -> np.ndarray:
    """plx = 1/d with d ~ d^2 exp(-d/L): rejection-free via the Gamma(3, L) form."""
    d_kpc = rng.gamma(shape=3.0, scale=FIELD_L_KPC, size=n)
    d_kpc = np.clip(d_kpc, 0.05, 30.0)
    return 1.0 / d_kpc


def generate(
    *,
    n_members: int,
    contamination: float,
    fractal_dimension: float,
    seed: int,
    inject_cluster: bool = True,
) -> Realisation:
    """Synthesise one cluster+field realisation with known membership truth.

    ``contamination`` is the target field fraction ``n_field / n_total``. With
    ``inject_cluster=False`` the cluster is omitted entirely (negative control) and
    ``n_members`` field stars are added in its place so the total is unchanged.
    """
    rng = np.random.default_rng(seed)
    if not 0.0 <= contamination < 1.0:
        raise ValueError("contamination must lie in [0, 1).")
    n_field = int(round(n_members * contamination / (1.0 - contamination)))
    if not inject_cluster:
        n_field += n_members
        n_members = 0

    cosd = np.cos(np.deg2rad(DEC_C))

    # --- cluster members -------------------------------------------------
    if n_members:
        # fractal_cluster returns 3D positions in a unit sphere; the first two axes are
        # the sky plane (its own radial_profile_of projects the same way).
        pos = fractal_cluster(
            n_members,
            fractal_dimension=fractal_dimension,
            radius=CLUSTER_RADIUS_DEG,
            rng=rng,
        )
        ra_m = RA_C + pos[:, 0] / cosd
        dec_m = DEC_C + pos[:, 1]
        pmra_m = PMRA_C + rng.normal(0.0, SIGMA_PM_INT, n_members)
        pmdec_m = PMDEC_C + rng.normal(0.0, SIGMA_PM_INT, n_members)
        plx_m = PLX_C + rng.normal(0.0, SIGMA_PLX_INT, n_members)
    else:
        ra_m = dec_m = pmra_m = pmdec_m = plx_m = np.empty(0)

    # --- field -----------------------------------------------------------
    # uniform on the disc: r = R sqrt(u)
    r = FOV_RADIUS_DEG * np.sqrt(rng.random(n_field))
    th = rng.uniform(0.0, 2.0 * np.pi, n_field)
    ra_f = RA_C + (r * np.cos(th)) / cosd
    dec_f = DEC_C + r * np.sin(th)
    pmra_f = rng.normal(FIELD_PM_C[0], FIELD_PM_SIGMA, n_field)
    pmdec_f = rng.normal(FIELD_PM_C[1], FIELD_PM_SIGMA, n_field)
    plx_f = _sample_field_parallax(rng, n_field)

    ra_t = np.concatenate([ra_m, ra_f])
    dec_t = np.concatenate([dec_m, dec_f])
    pmra_t = np.concatenate([pmra_m, pmra_f])
    pmdec_t = np.concatenate([pmdec_m, pmdec_f])
    plx_t = np.concatenate([plx_m, plx_f])
    truth = np.concatenate([np.ones(n_members, bool), np.zeros(n_field, bool)])

    # --- observational scatter -------------------------------------------
    n_tot = truth.size
    gmag = rng.uniform(G_MIN, G_MAX, n_tot)
    e_plx, e_pm = _astrometric_errors(gmag)
    pmra_o = pmra_t + rng.normal(0.0, e_pm)
    pmdec_o = pmdec_t + rng.normal(0.0, e_pm)
    plx_o = plx_t + rng.normal(0.0, e_plx)

    # shuffle so row order carries no membership information
    order = rng.permutation(n_tot)
    return Realisation(
        ra=ra_t[order],
        dec=dec_t[order],
        pmra=pmra_o[order],
        pmdec=pmdec_o[order],
        plx=plx_o[order],
        e_pmra=e_pm[order],
        e_pmdec=e_pm[order],
        e_plx=e_plx[order],
        gmag=gmag[order],
        truth=truth[order],
        n_members=int(n_members),
        n_field=int(n_field),
        contamination=float(n_field / max(n_tot, 1)),
    )


def check_generator(real: Realisation, *, requested_contamination: float) -> dict:
    """Negative-control style self-check of the generator, before anything downstream."""
    got = real.contamination
    ok_cont = abs(got - requested_contamination) < 0.02 or real.n_members == 0
    out = {
        "requested_contamination": requested_contamination,
        "realised_contamination": got,
        "contamination_ok": bool(ok_cont),
    }
    if real.n_members:
        m = real.truth
        out.update(
            {
                "member_pmra_mean": float(np.mean(real.pmra[m])),
                "member_pmdec_mean": float(np.mean(real.pmdec[m])),
                "member_plx_mean": float(np.mean(real.plx[m])),
                # 5 sigma_pm/sqrt(N) tolerance on the injected centroid
                "centroid_ok": bool(
                    abs(np.mean(real.pmra[m]) - PMRA_C) < 5 * SIGMA_PM_INT / np.sqrt(m.sum()) + 0.05
                ),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = np.std(x)
    return (x - np.mean(x)) / (s if s > 0 else 1.0)


# name -> (quantities fed to HDBSCAN, branch-selection rule).
#
# ``selection`` is a documented ``search_pseudoprobability`` argument with exactly two
# legal values and NEITHER uses ground truth, so both are fair game:
#   'max_members' (the DEFAULT) picks the sweep step whose densest condensed-tree branch
#       is LARGEST -- i.e. it is biased toward the biggest structure in the frame;
#   'max_lambda' picks the step with the highest lambda -- the densest structure.
# In a contaminated frame the largest structure is the field and the densest is the
# cluster, so the two rules can disagree completely. Reporting only the default would
# understate the package; reporting only the alternative would hide a real default-value
# failure mode. Both are measured.
EROTICA_CONFIGS = {
    "erotica_pm": (("pmra", "pmdec"), "max_members"),
    "erotica_3d": (("pmra", "pmdec", "plx"), "max_members"),
    "erotica_5d": (("ra", "dec", "pmra", "pmdec", "plx"), "max_members"),
    "erotica_pm_maxlambda": (("pmra", "pmdec"), "max_lambda"),
    "erotica_3d_maxlambda": (("pmra", "pmdec", "plx"), "max_lambda"),
    "erotica_5d_maxlambda": (("ra", "dec", "pmra", "pmdec", "plx"), "max_lambda"),
}


def run_erotica_config(
    real: Realisation,
    config: str,
    *,
    mcs_range: range,
) -> tuple[np.ndarray, np.ndarray, float, str | None]:
    """Return (p_tilde, member_mask, seconds, error). Columns are z-scored first."""
    quantities, selection = EROTICA_CONFIGS[config]
    cols = {f"{q}_z": _zscore(getattr(real, q)) for q in quantities}
    table = QTable(cols)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probs, mask, secs = bm.run_erotica(
                table,
                columns=tuple(cols),
                min_cluster_size_samples=mcs_range,
                probability_threshold=0.5,
                selection=selection,
            )
        return probs, mask, secs, None
    except Exception as exc:  # a failure to find any cluster is a legitimate result
        n = real.truth.size
        return np.zeros(n), np.zeros(n, bool), float("nan"), f"{type(exc).__name__}: {exc}"


def erotica_branch_diagnostic(
    real: Realisation,
    config: str,
    *,
    mcs_range: range,
) -> dict:
    """Separate WHERE EROTICA fails: the clustering, or the branch it then picks.

    Until 2026-08-03 the package selected the reported branch with
    ``Clustering._cluster_label_for_size``: it looked for a flat cluster whose size equals
    ``desired_len`` -- a count of rows in the *condensed tree* sharing the max-lambda
    parent -- and fell back to the LARGEST cluster when no size matched. In a contaminated
    frame the largest cluster is the field, so an exact-size miss silently swapped the
    answer for the contaminant (issue #7). The package now resolves the max-lambda tree
    node to the label its leaves carry (``_cluster_label_from_tree``).

    This re-runs the same search and reports, for the same labelling:

    * ``best_recall_*``  -- the HDBSCAN branch that actually contains the cluster. **This
      uses ground truth. It is an ORACLE CEILING, not a fair comparison point**; it exists
      only to separate "HDBSCAN never found the cluster" from "the selector discarded it".
    * ``selected_*``     -- what the package actually returned. **Read back from
      ``pseudoprobability_selected_['selected_cluster']``, never re-derived here.** A local
      re-derivation is how this diagnostic would keep reporting the pre-fix answer after the
      package was fixed, and report a no-op as a result.
    * ``legacy_selected_*`` -- what the pre-2026-08-03 selector would have returned on the
      SAME labelling, so both sides of the fix come out of one run.
    * ``f_max``          -- the largest recovery frequency reached, which bounds p-tilde.

    ``selected_purity`` is 0.0 when the selector returns nothing, and that 0.0 enters any
    mean taken over cells. ``selected_n`` and ``selected_is_empty`` are reported alongside
    so a purity change can be told apart from a change in how often nothing is selected.
    """
    quantities, selection = EROTICA_CONFIGS[config]
    cols = {f"{q}_z": _zscore(getattr(real, q)) for q in quantities}
    from erotica import Clustering
    from erotica.core.clustering import Clustering as _C

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clu = Clustering(QTable(cols))
            clu.search_pseudoprobability(
                columns=tuple(cols),
                min_cluster_size_samples=mcs_range,
                probability_threshold=0.5,
                selection=selection,
            )
        labels = np.asarray(clu.data["cluster_hdbscan"], dtype=int)
        f_i = np.asarray(clu.data["probability_times"], dtype=float)
        desired = int((clu.pseudoprobability_selected_ or {}).get("desired_len", -1))
        # What the package returned. Read back, not recomputed -- see the docstring.
        sel_label = int((clu.pseudoprobability_selected_ or {}).get("selected_cluster", -1))
        sel_mask = labels == sel_label if sel_label != -1 else np.zeros(labels.size, bool)
        # What the pre-fix selector would have returned on this same labelling.
        leg_label = _C._cluster_label_for_size(labels, desired)
        leg_mask = labels == leg_label if leg_label != -1 else np.zeros(labels.size, bool)
        best_label, best_recall = None, -1.0
        for u in np.unique(labels):
            if u == -1:
                continue
            rec = real.truth[labels == u].sum() / max(real.n_members, 1)
            if rec > best_recall:
                best_label, best_recall = int(u), float(rec)
        best_mask = labels == best_label if best_label is not None else np.zeros_like(sel_mask)
        return {
            "config": config,
            "best_mcs": int(clu.best_params_["min_cluster_size"]),
            "desired_len": desired,
            "n_labels": int(len(np.unique(labels)) - (1 if -1 in labels else 0)),
            "selected_label": int(sel_label),
            "selected_n": int(sel_mask.sum()),
            "selected_is_empty": bool(not sel_mask.any()),
            "selected_purity": float(real.truth[sel_mask].mean()) if sel_mask.any() else 0.0,
            "selected_recall": float(real.truth[sel_mask].sum() / max(real.n_members, 1)),
            "legacy_selected_label": int(leg_label),
            "legacy_selected_n": int(leg_mask.sum()),
            "legacy_selected_is_empty": bool(not leg_mask.any()),
            "legacy_selected_purity": float(real.truth[leg_mask].mean()) if leg_mask.any() else 0.0,
            "legacy_selected_recall": float(real.truth[leg_mask].sum() / max(real.n_members, 1)),
            "selector_changed": bool(int(leg_label) != int(sel_label)),
            "oracle_best_label": best_label,
            "oracle_best_n": int(best_mask.sum()),
            "oracle_best_purity": float(real.truth[best_mask].mean()) if best_mask.any() else 0.0,
            "oracle_best_recall": float(best_recall if best_recall > 0 else 0.0),
            "size_match_found": bool((np.unique(labels, return_counts=True)[1] == desired).any()),
            "f_max": float(f_i.max()),
            "n_ptilde_over_half": int((np.asarray(clu.data["probability"], float) > 0.5).sum()),
            "oracle_recovery": recover_params(real, best_mask),
        }
    except Exception as exc:
        return {"config": config, "error": f"{type(exc).__name__}: {exc}"}


def run_asteca(
    real: Realisation,
    *,
    n_runs: int = 1000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, str | None, dict]:
    """Run ASteCA 0.7.0 get_center -> get_nmembers(ripley) -> Membership.fastmp.

    Centres and N_cluster are ESTIMATED by ASteCA, never taken from the injected truth.
    """
    import asteca

    n = real.truth.size
    info: dict = {}
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cl = asteca.Cluster(
                ra=real.ra,
                dec=real.dec,
                pmra=real.pmra,
                pmde=real.pmdec,
                plx=real.plx,
                e_pmra=real.e_pmra,
                e_pmde=real.e_pmdec,
                e_plx=real.e_plx,
                verbose=0,
            )
            cl.get_center(algo="knn_5d")
            cl.get_nmembers(algo="ripley")
            info = {
                "asteca_radec_c": [float(v) for v in cl.radec_c],
                "asteca_pms_c": [float(v) for v in cl.pms_c],
                "asteca_plx_c": float(cl.plx_c),
                "asteca_N_cluster": int(cl.N_cluster),
            }
            memb = asteca.Membership(cl)
            memb.fastmp(N_runs=n_runs, seed=seed)
        probs = np.asarray(cl.probs, dtype=float)
        secs = time.perf_counter() - t0
        if probs.size != n:
            return (
                np.zeros(n),
                np.zeros(n, bool),
                secs,
                f"co-indexing broken: {probs.size} probs for {n} rows",
                info,
            )
        return probs, probs >= 0.5, secs, None, info
    except Exception as exc:
        return (
            np.zeros(n),
            np.zeros(n, bool),
            time.perf_counter() - t0,
            f"{type(exc).__name__}: {exc}",
            info,
        )


# ---------------------------------------------------------------------------
# Parameter recovery (identical estimator applied to both member lists)
# ---------------------------------------------------------------------------
def recover_params(real: Realisation, mask: np.ndarray) -> dict[str, float]:
    """Astrometric parameters estimated from a member list.

    The SAME estimator is applied to every method's selection, so any difference is a
    difference in the member list and nothing else. ``sigma_pm`` is deconvolved for
    measurement error, which is what makes it sensitive to contamination: an impure list
    inflates the dispersion.
    """
    mask = np.asarray(mask, dtype=bool)
    k = int(mask.sum())
    if k < 3:
        nan = float("nan")
        return {"pmra_c": nan, "pmdec_c": nan, "plx_c": nan, "sigma_pm": nan, "n_members": float(k)}
    var_pm = 0.5 * (np.var(real.pmra[mask], ddof=1) + np.var(real.pmdec[mask], ddof=1))
    mean_e2 = float(np.mean(real.e_pmra[mask] ** 2))
    return {
        "pmra_c": float(np.mean(real.pmra[mask])),
        "pmdec_c": float(np.mean(real.pmdec[mask])),
        "plx_c": float(np.mean(real.plx[mask])),
        "sigma_pm": float(np.sqrt(max(var_pm - mean_e2, 0.0))),
        "n_members": float(k),
    }


def truth_params(real: Realisation) -> dict[str, float]:
    return {
        "pmra_c": PMRA_C,
        "pmdec_c": PMDEC_C,
        "plx_c": PLX_C,
        "sigma_pm": SIGMA_PM_INT,
        "n_members": float(real.n_members),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def adaptive_mcs_cap(
    real: Realisation,
    quantities: tuple[str, ...],
    *,
    lo: int = 10,
    hard_cap: int = 300,
    step: int = 5,
    patience: int = 3,
) -> int:
    """Largest ``min_cluster_size`` at which HDBSCAN still resolves >= 2 structures.

    WHY THIS RULE EXISTS. ``p_tilde = p_HDBSCAN * f_i`` and ``f_i`` is the fraction of
    swept ``mcs`` values in which a source landed in *any* cluster. Every ``mcs`` past the
    point where the cluster has dissolved contributes only field membership to ``f_i``, so
    a sweep wider than the cluster both DEPRESSES p-tilde toward zero (measured: f_max
    0.66 -> 0.37 -> 0.17 as the sweep top goes 60 -> 100 -> 200) and INVERTS its ranking
    toward the field. A fixed sweep therefore cannot be scale-free across cluster sizes,
    and the fixed-sweep arm of this benchmark shows exactly that failure.

    This rule uses no ground truth: it probes ``mcs`` on a coarse grid and caps the sweep
    at the last value that yields >= 2 clusters (with ``allow_single_cluster=False``, the
    frame collapsing to one structure means the smaller one is gone). Measured against
    injected sizes it returns 25 / 45 / 150 for true 30 / 61 / 150.

    It is MY rule, added because the package ships no scale-free default, and it is
    reported as such -- not as a package feature.
    """
    import hdbscan

    X = np.column_stack([_zscore(getattr(real, q)) for q in quantities])
    cap, misses = lo, 0
    for mcs in range(lo, hard_cap + 1, step):
        if mcs >= X.shape[0]:
            break
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            labels = (
                hdbscan.HDBSCAN(
                    min_cluster_size=int(mcs),
                    cluster_selection_method="eom",
                    allow_single_cluster=False,
                    metric="euclidean",
                    gen_min_span_tree=False,
                    match_reference_implementation=True,
                )
                .fit(X)
                .labels_
            )
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            cap, misses = mcs, 0
        else:
            misses += 1
            if misses >= patience:
                break
    return int(cap)


def run_cell(
    *,
    n_members: int,
    contamination: float,
    fractal_dimension: float,
    seed: int,
    mcs_range: range,
    configs: list[str],
    asteca_runs: int,
    diagnostic_configs: tuple[str, ...] = ("erotica_3d", "erotica_5d"),
    adaptive: bool = False,
    pyupmask_cfg: dict | None = None,
) -> dict:
    real = generate(
        n_members=n_members,
        contamination=contamination,
        fractal_dimension=fractal_dimension,
        seed=seed,
    )
    gen_check = check_generator(real, requested_contamination=contamination)
    caps: dict[str, int] = {}
    if adaptive:
        for cfg in set(configs) | set(diagnostic_configs):
            quantities, _ = EROTICA_CONFIGS[cfg]
            caps[cfg] = adaptive_mcs_cap(real, quantities, lo=mcs_range.start)

    method_probs: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    runtimes: dict[str, float] = {}
    errors: dict[str, str] = {}
    recovered: dict[str, dict[str, float]] = {}

    def _range_for(cfg: str) -> range:
        return range(mcs_range.start, caps[cfg] + 1) if adaptive else mcs_range

    for cfg in configs:
        p, m, s, err = run_erotica_config(real, cfg, mcs_range=_range_for(cfg))
        method_probs[cfg] = p
        masks[cfg] = m
        runtimes[cfg] = s
        if err:
            errors[cfg] = err
        recovered[cfg] = recover_params(real, m)

    p, m, s, err, info = run_asteca(real, n_runs=asteca_runs, seed=seed)
    method_probs["asteca_fastmp"] = p
    masks["asteca_fastmp"] = m
    runtimes["asteca_fastmp"] = s
    if err:
        errors["asteca_fastmp"] = err
    recovered["asteca_fastmp"] = recover_params(real, m)

    if pyupmask_cfg is not None:
        from benchmark_pyupmask_runner import run_pyupmask

        try:
            pp, ps, pnotes = run_pyupmask(
                real,
                workdir=Path(pyupmask_cfg["workdir"]),
                ol_runs=pyupmask_cfg["ol_runs"],
                il_runs=pyupmask_cfg["il_runs"],
                n_membs=pyupmask_cfg["n_membs"],
                seed=seed,
                timeout_s=pyupmask_cfg["timeout_s"],
            )
            if "error" in pnotes:
                errors["pyupmask"] = f"{pnotes['error']}: {pnotes.get('stderr', '')[-300:]}"
            method_probs["pyupmask"] = pp
            masks["pyupmask"] = pp >= 0.5
            runtimes["pyupmask"] = ps
            recovered["pyupmask"] = recover_params(real, pp >= 0.5)
            info["pyupmask_notes"] = {
                k: v for k, v in pnotes.items() if k not in ("stderr", "stdout")
            }
        except Exception as exc:
            errors["pyupmask"] = f"{type(exc).__name__}: {exc}"

    # POSITIVE CONTROL for the harness itself: a "method" whose probabilities ARE the
    # truth must score Brier = ECE = 0 and AUC = 1. If it does not, the metric code is
    # broken and nothing else in this file means anything.
    method_probs["_truth_positive_control"] = real.truth.astype(float)
    masks["_truth_positive_control"] = real.truth.copy()
    recovered["_truth_positive_control"] = recover_params(real, real.truth)

    # Shared, TRUTH-FREE operating point: top-K by each method's own probability, with the
    # same K for every method, taken from ASteCA's Ripley N_cluster estimate. This makes
    # parameter recovery measurable even where a method's absolute probability scale
    # collapses (EROTICA's p-tilde is bounded by the sweep width, see the docstring), and
    # it charges every method the same number of members.
    #
    # ``_truth_positive_control`` is DELIBERATELY EXCLUDED here. Its probability vector is
    # 0/1, so ``argsort`` breaks the ties arbitrarily and the top-K cut slices through
    # them: the control scores purity 0.96 / recall 0.86 instead of the 1.0 it must score,
    # which would invalidate the block it is supposed to license. It remains the positive
    # control for the CALIBRATION axis (where it correctly scores Brier = ECE = MCE = 0)
    # and, via ``recovered['_truth_positive_control']`` in the default-threshold block,
    # for the ``recover_params`` estimator itself.
    #
    # ``k_shared`` is ASteCA's Ripley estimate, not the truth: it is reported next to
    # ``n_members`` so the reader can see how much of the ``n_members`` error column is
    # Ripley's counting error rather than any method's.
    k_shared = int(info.get("asteca_N_cluster", 0) or 0)
    recovered_topk: dict[str, dict[str, float]] = {}
    if k_shared >= 3:
        for name, pv in method_probs.items():
            if name == "_truth_positive_control":
                continue
            order = np.argsort(-np.asarray(pv, dtype=float), kind="stable")
            mk = np.zeros(pv.size, bool)
            mk[order[: min(k_shared, pv.size)]] = True
            recovered_topk[name] = recover_params(real, mk)
            recovered_topk[name]["purity"] = float(real.truth[mk].mean())
            recovered_topk[name]["recall"] = float(real.truth[mk].sum() / max(real.n_members, 1))

    report = bm.build_report(
        dataset=f"N{n_members}_c{contamination}_D{fractal_dimension}_s{seed}",
        method_probs=method_probs,
        truth_membership=real.truth.astype(float),
        member_masks=masks,
        truth_params=truth_params(real),
        recovered_params=recovered,
        runtimes=runtimes,
        threshold=0.5,
    )

    cell = {
        "n_members": n_members,
        "contamination": contamination,
        "fractal_dimension": fractal_dimension,
        "seed": seed,
        "n_sources": int(real.truth.size),
        "mcs_range": [mcs_range.start, mcs_range.stop],
        "adaptive": bool(adaptive),
        "adaptive_caps": caps,
        "generator_check": gen_check,
        "errors": errors,
        "asteca_info": info,
        "calibration": report.calibration_table().to_dict("records"),
        "recovery": report.recovery_table().to_dict("records"),
        "recovery_topk_shared": recovered_topk,
        "k_shared": k_shared,
        "runtime": report.runtime_table().to_dict("records"),
        "agreement": (
            report.agreement_table().to_dict("records") if len(report.agreements) else []
        ),
        "branch_diagnostic": [
            erotica_branch_diagnostic(real, cfg, mcs_range=_range_for(cfg))
            for cfg in diagnostic_configs
        ],
        # kept for the out-of-sample recalibration step
        "_probs": {k: v.tolist() for k, v in method_probs.items()},
        "_truth": real.truth.astype(int).tolist(),
    }
    return cell


def negative_control(
    *, seed: int, n_field_like: int, mcs_range: range, configs: list[str], asteca_runs: int
) -> dict:
    """Field-only realisation: no cluster injected. Anything found is a false positive."""
    real = generate(
        n_members=n_field_like,
        contamination=0.9,
        fractal_dimension=1.6,
        seed=seed,
        inject_cluster=False,
    )
    out: dict = {
        "seed": seed,
        "n_sources": int(real.truth.size),
        "n_true_members": 0,
        "methods": {},
    }
    for cfg in configs:
        p, m, s, err = run_erotica_config(real, cfg, mcs_range=mcs_range)
        out["methods"][cfg] = {
            "n_selected": int(m.sum()),
            "max_prob": float(np.max(p)) if p.size else 0.0,
            "runtime_s": s,
            "error": err,
        }
    p, m, s, err, info = run_asteca(real, n_runs=asteca_runs, seed=seed)
    out["methods"]["asteca_fastmp"] = {
        "n_selected": int(m.sum()),
        "max_prob": float(np.max(p)) if p.size else 0.0,
        "runtime_s": s,
        "error": err,
        "asteca_info": info,
    }
    return out


def out_of_sample_recalibration(cells: list[dict], *, method: str) -> dict:
    """Fit isotonic / Platt on one half of the seeds, score ECE on the held-out half.

    Fitting and scoring on the same rows drives ECE to ~0 and is a fabricated number;
    the split here is by SEED, so no row used for fitting is ever scored.
    """
    from erotica.calibration import brier_score, expected_calibration_error, fit_isotonic, fit_platt

    seeds = sorted({c["seed"] for c in cells})
    if len(seeds) < 2:
        return {"error": "need >= 2 seeds for an out-of-sample split"}
    train_seeds = set(seeds[::2])
    test_seeds = set(seeds[1::2])

    def pool(sel: set[int]) -> tuple[np.ndarray, np.ndarray]:
        ps, ys = [], []
        for c in cells:
            if c["seed"] in sel and method in c["_probs"] and method not in c["errors"]:
                ps.append(np.asarray(c["_probs"][method], dtype=float))
                ys.append(np.asarray(c["_truth"], dtype=int))
        if not ps:
            return np.empty(0), np.empty(0, dtype=int)
        return np.concatenate(ps), np.concatenate(ys)

    p_tr, y_tr = pool(train_seeds)
    p_te, y_te = pool(test_seeds)
    if p_tr.size == 0 or p_te.size == 0:
        return {"error": "empty train or test pool"}

    res = {
        "method": method,
        "n_train": int(p_tr.size),
        "n_test": int(p_te.size),
        "train_seeds": sorted(train_seeds),
        "test_seeds": sorted(test_seeds),
        # erotica.calibration.expected_calibration_error returns (ECE, MCE), not a scalar.
        "raw_ece_test": float(expected_calibration_error(p_te, y_te)[0]),
        "raw_mce_test": float(expected_calibration_error(p_te, y_te)[1]),
        "raw_brier_test": float(brier_score(p_te, y_te)),
    }
    for name, fitter in (("isotonic", fit_isotonic), ("platt", fit_platt)):
        try:
            rec = fitter(p_tr, y_tr)
            p_cal = np.clip(np.asarray(rec(p_te), dtype=float), 0.0, 1.0)
            ece_cal, mce_cal = expected_calibration_error(p_cal, y_te)
            res[f"{name}_ece_test"] = float(ece_cal)
            res[f"{name}_mce_test"] = float(mce_cal)
            res[f"{name}_brier_test"] = float(brier_score(p_cal, y_te))
        except Exception as exc:
            res[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=str(Path(__file__).with_name("benchmark_erotica_vs_asteca.json"))
    )
    ap.add_argument("--realisations", type=int, default=6)
    ap.add_argument("--quick", action="store_true", help="one small cell, one seed")
    ap.add_argument("--mcs-lo", type=int, default=10)
    ap.add_argument("--mcs-hi", type=int, default=100)
    ap.add_argument("--asteca-runs", type=int, default=1000)
    ap.add_argument("--configs", default=",".join(EROTICA_CONFIGS))
    ap.add_argument("--n-grid", default="30,61,150")
    ap.add_argument("--cont-grid", default="0.5,0.8,0.95")
    ap.add_argument("--dim-grid", default="1.6,3.0")
    ap.add_argument(
        "--pyupmask",
        action="store_true",
        help="also run pyUPMASK (cloned repo, driven as a script)",
    )
    ap.add_argument(
        "--pyupmask-repo",
        default=str(__import__("benchmark_pyupmask_runner", fromlist=["DEFAULT_REPO"]).DEFAULT_REPO)
        if False
        else "",
    )
    ap.add_argument("--pyupmask-ol", type=int, default=10)
    ap.add_argument("--pyupmask-il", type=int, default=15)
    ap.add_argument("--pyupmask-nmembs", type=int, default=25)
    ap.add_argument("--pyupmask-timeout", type=int, default=900)
    ap.add_argument(
        "--pyupmask-workdir",
        default="/private/tmp/claude-501/"
        "-Users-notluquis-phd/9a0f9b71-83f6-46ec-ba5e-2fc967bc6851/"
        "scratchpad/pyu_cell",
    )
    ap.add_argument(
        "--adaptive-sweep",
        action="store_true",
        help="cap the mcs sweep with the truth-free adaptive rule "
        "(see adaptive_mcs_cap); otherwise use --mcs-lo..--mcs-hi",
    )
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args(argv)

    configs = [c for c in args.configs.split(",") if c]
    mcs_range = range(args.mcs_lo, args.mcs_hi)
    pyupmask_cfg = None
    if args.pyupmask:
        sys.path.insert(0, str(Path(__file__).parent))
        pyupmask_cfg = {
            "workdir": args.pyupmask_workdir,
            "ol_runs": args.pyupmask_ol,
            "il_runs": args.pyupmask_il,
            "n_membs": args.pyupmask_nmembs,
            "timeout_s": args.pyupmask_timeout,
        }

    if args.quick:
        n_grid, cont_grid, dim_grid, n_real = [61], [0.8], [1.6], 1
    else:
        n_grid = [int(v) for v in args.n_grid.split(",")]
        cont_grid = [float(v) for v in args.cont_grid.split(",")]
        dim_grid = [float(v) for v in args.dim_grid.split(",")]
        n_real = args.realisations

    t_start = time.perf_counter()
    cells: list[dict] = []
    # Checkpoint every cell: a long grid that is interrupted must not lose the cells that
    # already ran (this has happened once in this workspace).
    ckpt = Path(args.out).with_suffix(".cells.jsonl")
    ckpt.write_text("")
    for n_members in n_grid:
        for cont in cont_grid:
            for dim in dim_grid:
                for k in range(n_real):
                    seed = (
                        1000 * n_grid.index(n_members)
                        + 100 * int(cont * 100)
                        + 10 * dim_grid.index(dim)
                        + k
                    )
                    print(
                        f"[cell {len(cells) + 1}] N={n_members} cont={cont} D={dim} "
                        f"seed={seed} t={time.perf_counter() - t_start:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
                    cell = run_cell(
                        n_members=n_members,
                        contamination=cont,
                        fractal_dimension=dim,
                        seed=seed,
                        mcs_range=mcs_range,
                        configs=configs,
                        asteca_runs=args.asteca_runs,
                        adaptive=args.adaptive_sweep,
                        pyupmask_cfg=pyupmask_cfg,
                    )
                    cells.append(cell)
                    with ckpt.open("a") as fh:
                        fh.write(json.dumps(cell) + "\n")

    control = None
    if not args.skip_control:
        print("[control] field-only", file=sys.stderr, flush=True)
        control = [
            negative_control(
                seed=90000 + i,
                n_field_like=61,
                mcs_range=mcs_range,
                configs=configs,
                asteca_runs=args.asteca_runs,
            )
            for i in range(3)
        ]

    sensitivity = None
    if not args.skip_sensitivity:
        print("[sensitivity] narrow mcs sweep", file=sys.stderr, flush=True)
        sensitivity = []
        for hi in (40, 60, 100, 200):
            c = run_cell(
                n_members=61,
                contamination=0.8,
                fractal_dimension=1.6,
                seed=777,
                mcs_range=range(10, hi),
                configs=configs,
                asteca_runs=args.asteca_runs,
            )
            c.pop("_probs"), c.pop("_truth")
            sensitivity.append(c)

    recal_methods = configs + ["asteca_fastmp"] + (["pyupmask"] if args.pyupmask else [])
    recal = {m: out_of_sample_recalibration(cells, method=m) for m in recal_methods}

    payload = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "wall_clock_s": time.perf_counter() - t_start,
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
                    "asteca",
                    "erotica",
                )
            },
        },
        "config": vars(args),
        "constants": {
            "RA_C": RA_C,
            "DEC_C": DEC_C,
            "FOV_RADIUS_DEG": FOV_RADIUS_DEG,
            "CLUSTER_RADIUS_DEG": CLUSTER_RADIUS_DEG,
            "PMRA_C": PMRA_C,
            "PMDEC_C": PMDEC_C,
            "PLX_C": PLX_C,
            "SIGMA_PM_INT": SIGMA_PM_INT,
            "SIGMA_PLX_INT": SIGMA_PLX_INT,
            "FIELD_PM_C": list(FIELD_PM_C),
            "FIELD_PM_SIGMA": FIELD_PM_SIGMA,
            "FIELD_L_KPC": FIELD_L_KPC,
            "G_MIN": G_MIN,
            "G_MAX": G_MAX,
        },
        "cells": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cells],
        "negative_control": control,
        "sweep_sensitivity": sensitivity,
        "out_of_sample_recalibration": recal,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2))
    print(
        f"wrote {args.out} ({len(cells)} cells, {payload['wall_clock_s']:.1f} s)", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
