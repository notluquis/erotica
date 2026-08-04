#!/usr/bin/env python3
r"""Does a Gaia DR3 selection-function correction change NGC 6383's radial profile?

WHY THIS EXISTS
---------------
``structure.king_unbinned(completeness=...)`` folds a mean radial detection
probability :math:`\bar S(r)` into the point-process normalisation. On synthetic
data ignoring a real gradient inflates ``R_c`` by ~50% and halves the central
density. The machinery is built and validated; what has never been done is
**running it on the real NGC 6383 member sample and reporting the delta**. A
literature search for selection-function-corrected open-cluster radial density
profiles returns zero papers, so either answer -- a correction or a null -- is new.

P01 publishes ``R_c = 1.384'`` for NGC 6383, robust across analysis choices at
1.32--1.50'. A corrected fit landing outside that band is a correction to a
published paper. A corrected fit landing inside it protects the published result.

PRE-REGISTERED FALSIFICATION -- WRITTEN BEFORE THE FIRST FIT WAS RUN
--------------------------------------------------------------------
Registered 2026-08-04, before any number in the sidecar existed. Each threshold
is a number, not an adjective, and each can fail.

**H1 -- "the survey selection function matters for this cluster's RDP."**
H1 is **falsified** when both hold:

  (a) ``core_suppression`` of :math:`\bar S(r)`, measured at a **spatial
      resolution <= 1.2 R_c = 1.66'** (healpix order >= 11, or an empirical curve
      whose innermost bin centre is inside that radius), is **< 1.0%**; and
  (b) ``|median(R_c_corrected) - median(R_c_uncorrected)| < 0.064'`` -- the
      *tighter* half-width of the published robustness band (1.384 - 1.32) --
      **and** ``< 0.25 x sd(R_c_uncorrected)``.

**H2 -- "the correction changes the published P01 number."**
H2 is **falsified** when ``median(R_c_corrected)`` stays inside [1.32, 1.50]'.
H2 is **confirmed**, and must be flagged as a correction to a published paper,
when it leaves that band.

**What would falsify the null in the other direction**: any of -- core
suppression >= 1%; ``|Delta R_c| >= 0.064'``; the corrected central-density
posterior median moving by >= 10%. If none of those trip, the correction does not
matter for this cluster and this script says so rather than manufacturing an effect.

**A flat result is only meaningful at resolving resolution.** ``mode='hpx7'``
pixels are 27.5' against ``R_c = 1.38'``; the whole 70' field spans 2.55 pixels,
so cluster-scale radial structure is below that map by construction and a flat
``S(r)`` there means "invisible", not "absent". ``mode='multi'`` reaches order 10
(3.44'), which resolves the field but not the core. ``mode='patch'`` can in
principle reach order 12 (0.86'), but **measured on this field it does not**:
``a1_patch_selection_function.py`` finds a mean effective order inside ``R_c`` of
10.2 (3.36') at the default ``min_points=20``, and only 1.73' at ``min_points=5``
-- so no ``S(r)`` available for NGC 6383 clears the 1.66' gate, and the survey
answer below is a bound at 1.73--3.4', not a resolving null. Every
:math:`\bar S(r)` therefore carries a **measured resolution in arcmin** into the
sidecar, and anything coarser than 1.66' is reported as **non-resolving** and
cannot falsify H1.

.. warning::
   The ``resolving`` field was first computed as ``max(orders) >= 11``, with
   non-healpix curves carrying a sentinel ``orders=[99]``. That made the gate
   **unfailable for precisely the curves whose resolution was in doubt**: the
   pipeline retention curve, whose innermost bin centre is 1.70', was written
   into the verdict as resolving. The gate is now a length, and 1.70' fails it.
   The same class of defect -- a resolution asserted rather than measured -- is
   what caused a 4.5x under-read recorded in ``tools/validation/CLAUDE.md``.

NEGATIVE CONTROLS, AND WHY THE OBVIOUS ONE IS RESTATED
-------------------------------------------------------
"A flat ``S(r) = 1`` must reproduce the uncorrected fit *exactly*" cannot be
satisfied as written: the corrected path builds a different PyMC graph (256-node
Gauss-Legendre quadrature) than the uncorrected one (closed form), so the two
NUTS runs differ in floating-point trajectory even at an identical seed. Split
into two tiers, both of which can fail sharply:

* **C1 deterministic.** ``king_expected_count_weighted(..., S=1)`` against the
  closed-form ``king_expected_count`` over a grid spanning the plausible
  posterior. Gate: ``max relative error < 1e-4``. This also measures the
  **numerical floor**: if the quadrature error were not far below the size of the
  effect being hunted (~1e-2), the whole exercise would be under the machinery's
  own noise and *that* would be the headline. The weighted integrand carries a
  ``switch`` at ``R_t`` -- a derivative discontinuity inside the interval, where
  Gauss-Legendre is no longer spectrally accurate -- so the docstring claim
  "integrates the King form to machine precision" is tested here, not assumed.
* **C2 posterior.** ``completeness=ones(256)`` vs ``completeness=None`` at the
  same seed. Gate: ``|mean(R_c_a) - mean(R_c_b)| < 2 sqrt(mcse_a^2 + mcse_b^2)``,
  i.e. a two-sigma Monte-Carlo test. Reported in MCSE units, never as "exactly".

**C3 mutation -- the check that makes a null believable.** A null from C1+C2 is
also what a dead code path produces. So a deliberately steep
``S(r) = 0.5 + 0.5 (1 - exp(-r/6))`` (aperture core suppression 42.5%) is applied
to the **real** radii. Predicted, from this repo's own measured bias-vs-suppression
curve (``completeness_bias_scaling.json``: 42.5% suppression <-> +37.9% ``R_c``
bias): correcting data that was *not* actually suppressed over-corrects in the
same direction, so ``R_c`` must **fall** by a ratio near ``1/1.379 = 0.725``.
Gate: the ratio must be **< 0.90** and within [0.55, 0.90]. If a 42.5% central
suppression does not move ``R_c``, the correction path is inert and no null from
it is believable.

**C4 positive control on synthetic data.** A King point process with known
``R_c = 1.38'`` is thinned by a known ``S(r)``; the uncorrected fit must recover
biased-high (gate: bias > +15%) and the corrected fit must recover the truth
(gate: ``|R_c - 1.38| < 2 sd``). Generator, estimator and interpretation are
checked in that order, per ``tools/validation/CLAUDE.md``.

SCOPE CAVEAT -- STATE IT BEFORE A REFEREE DOES
-----------------------------------------------
The sample is ``paperfaithful_reference_p06.ecsv``, a ``p >= 0.6`` membership cut.
That cut is a third selection layer which neither the survey ``S(r)`` nor the
pipeline clip captures, and ``king_unbinned``'s normalisation assumes completeness
within the disc. This is legitimate for a **both-ways delta on the same sample the
paper fitted**, which is the question here; it is not a claim that the sample is
complete.

USAGE
-----
    python tools/validation/a1_selection_corrected_rdp.py --stage controls
    python tools/validation/a1_selection_corrected_rdp.py --stage fits \
        --sbar tools/validation/a1_sbar_patch.npz
    python tools/validation/a1_selection_corrected_rdp.py --stage all --sbar ...

Every stage writes into ``a1_selection_corrected_rdp.json`` **as it completes**,
merging with whatever is already there, so a death costs one stage. Posterior
arrays go to ``a1_selection_corrected_rdp.npz`` (gitignored); the JSON carries
quantiles, r-hat, ESS, divergences and MCSE only, and stays well under 500 KB.

Environment: ``/Users/notluquis/miniforge3/envs/cosmic/bin/python`` run from the
repo root (that env has PyMC 6.0.1 + ArviZ 1.2.0; ``erotica`` resolves from the
working directory). ``erotica-bench`` has ``erotica`` installed but no PyMC.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from erotica.analysis.inference import SamplingConfig
from erotica.analysis.structure import (
    king_expected_count,
    king_expected_count_weighted,
    king_unbinned,
)

HERE = Path(__file__).resolve().parent
JSON_OUT = HERE / "a1_selection_corrected_rdp.json"
NPZ_OUT = HERE / "a1_selection_corrected_rdp.npz"

B = Path("/Users/notluquis/erotica/data/test/NGC6383")
MEMBERS = B / "comments_paper/radius_robustness/generated/70/paperfaithful_reference_p06.ecsv"
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
FIELD = 70.0  # arcmin
NODES = 256  # must match king_expected_count_weighted

# --- pre-registered constants -------------------------------------------------
PUBLISHED_RC = 1.384  # arcmin, P01
PUBLISHED_BAND = (1.32, 1.50)  # arcmin, P01 robustness range
DELTA_RC_GATE = PUBLISHED_RC - PUBLISHED_BAND[0]  # 0.064', the tighter half-width
DELTA_RC_SD_FRACTION = 0.25
SUPPRESSION_GATE = 0.01  # 1%
RESOLVING_ORDER = 11  # order 11 = 1.72', the coarsest healpix order that can see the core
RESOLVING_ARCMIN = 1.2 * PUBLISHED_RC  # 1.66'; the gate is applied in arcmin, not orders
CENTRAL_DENSITY_GATE = 0.10  # 10%
C1_GATE = 1e-4
C2_SIGMA = 2.0
C3_RATIO_BAND = (0.55, 0.90)
C3_PREDICTED_RATIO = 1.0 / 1.379  # from completeness_bias_scaling.json
C4_MIN_UNCORRECTED_BIAS = 0.15
CORE_APERTURE = 1.38  # arcmin, the aperture core suppression is quoted over

SEED = 20260804


# ---------------------------------------------------------------------------
# incremental persistence: every stage merges into the same JSON
# ---------------------------------------------------------------------------
def _load_json() -> dict:
    if JSON_OUT.exists():
        try:
            return json.loads(JSON_OUT.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def _save(key: str, payload) -> None:
    doc = _load_json()
    doc[key] = payload
    doc["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    JSON_OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(f"  [saved '{key}' -> {JSON_OUT.name}]", flush=True)


def _save_arrays(**arrays) -> None:
    existing = {}
    if NPZ_OUT.exists():
        with np.load(NPZ_OUT, allow_pickle=False) as z:
            existing = {k: z[k] for k in z.files}
    existing.update(arrays)
    np.savez_compressed(NPZ_OUT, **existing)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def quadrature_radii(field_radius=FIELD, nodes=NODES):
    x, _ = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * field_radius * (x + 1.0)


def member_radii():
    table = Table.read(MEMBERS)
    for a, d in (("ra", "dec"), ("RA_ICRS", "DE_ICRS"), ("ra_epoch2000", "dec_epoch2000")):
        if a in table.colnames and d in table.colnames:
            sky = SkyCoord(np.asarray(table[a], float) * u.deg, np.asarray(table[d], float) * u.deg)
            r = CENTER.separation(sky).to(u.arcmin).value
            break
    else:
        raise KeyError(f"no sky columns in {table.colnames[:12]}")
    gmag = np.asarray(table["Gmag"], dtype=float) if "Gmag" in table.colnames else None
    keep = np.isfinite(r) & (r > 0) & (r <= FIELD)
    return r[keep], (gmag[keep] if gmag is not None else None), int((~keep).sum())


def king_sigma(r, *, k, R_c, R_t, b):
    r = np.asarray(r, float)
    core = 1.0 / np.sqrt(1.0 + (r / R_c) ** 2)
    edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
    return np.where(r <= R_t, k * (core - edge) ** 2 + b, b)


def sample_king(rng, n, **params):
    grid = np.linspace(0.0, FIELD, 100_001)
    pdf = 2.0 * np.pi * grid * king_sigma(grid, **params)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0.0, 1.0, n), cdf, grid)


def steep_completeness(floor=0.5, scale=6.0):
    """The mutation curve. Same family as ``completeness_bias_scaling.py``."""
    return lambda r: floor + (1.0 - floor) * (1.0 - np.exp(-np.asarray(r, float) / scale))


def core_suppression(s_of_r, aperture=CORE_APERTURE, field=FIELD):
    r"""``1 - <S>_{r<aperture} / <S>_{r<field}``, both area-weighted.

    Identical definition to ``completeness_bias_scaling.core_suppression`` so the
    measured bias-vs-suppression curve there transfers without rescaling.
    """
    inner = np.linspace(1e-6, aperture, 2000)
    outer = np.linspace(1e-6, field, 20000)
    s_in = np.trapezoid(2 * np.pi * inner * s_of_r(inner), inner) / (np.pi * aperture**2)
    s_out = np.trapezoid(2 * np.pi * outer * s_of_r(outer), outer) / (np.pi * field**2)
    return float(1.0 - s_in / s_out)


def central_density(trace):
    r"""Posterior of the King central surface density ``Sigma(0) = k (1 - edge)^2``."""
    post = trace.posterior
    k = np.asarray(post["k"].values, float)
    rc = np.asarray(post["R_c"].values, float)
    rt = np.asarray(post["R_t"].values, float)
    edge = 1.0 / np.sqrt(1.0 + (rt / rc) ** 2)
    return (k * (1.0 - edge) ** 2).ravel()


def _hdi(arr, prob=0.94):
    """ArviZ renamed ``hdi_prob`` -> ``prob`` at 1.0; support both."""
    import arviz as az

    try:
        out = az.hdi(arr, prob=prob)
    except TypeError:
        out = az.hdi(arr, hdi_prob=prob)
    return [float(v) for v in np.asarray(out).ravel()[:2]]


def _summarise(trace, label, seconds):
    import arviz as az

    smry = az.summary(trace, var_names=["k", "b", "R_c", "R_t"])
    out = {
        "label": label,
        "seconds": round(seconds, 1),
        "r_hat_max": float(smry["r_hat"].max()),
        "ess_bulk_min": float(smry["ess_bulk"].min()),
        "ess_tail_min": float(smry["ess_tail"].min()),
        "divergences": int(trace.sample_stats["diverging"].values.sum()),
    }
    for p in ("k", "b", "R_c", "R_t"):
        arr = np.asarray(trace.posterior[p].values, float).ravel()
        out[p] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "sd": float(arr.std(ddof=1)),
            "q025": float(np.quantile(arr, 0.025)),
            "q16": float(np.quantile(arr, 0.16)),
            "q84": float(np.quantile(arr, 0.84)),
            "q975": float(np.quantile(arr, 0.975)),
            "hdi94": _hdi(arr, 0.94),
            "mcse_mean": float(smry.loc[p, "mcse_mean"]),
        }
    sig0 = central_density(trace)
    out["Sigma0"] = {
        "median": float(np.median(sig0)),
        "sd": float(sig0.std(ddof=1)),
        "q16": float(np.quantile(sig0, 0.16)),
        "q84": float(np.quantile(sig0, 0.84)),
    }
    out["gates_pass"] = bool(
        out["r_hat_max"] < 1.01 and out["ess_bulk_min"] > 400 and out["divergences"] == 0
    )
    return out


def _fit(radii, label, *, completeness=None, draws=2000, chains=4, seed=SEED):
    t0 = time.perf_counter()
    res = king_unbinned(
        radii,
        field_radius=FIELD,
        completeness=completeness,
        sampling=SamplingConfig(
            draws=draws, tune=1000, chains=chains, random_seed=seed, progressbar=False
        ),
    )
    trace = res["king_trace"]
    summary = _summarise(trace, label, time.perf_counter() - t0)
    summary["n_stars"] = int(res["n_stars"])
    summary["completeness_corrected"] = bool(res["completeness_corrected"])
    print(
        f"  {label:38s} R_c={summary['R_c']['median']:.4f}+/-{summary['R_c']['sd']:.4f}"
        f"  Sigma0={summary['Sigma0']['median']:.3f}"
        f"  rhat={summary['r_hat_max']:.4f} ess={summary['ess_bulk_min']:.0f}"
        f" div={summary['divergences']}",
        flush=True,
    )
    return summary, trace


def _delta(a, b, key="R_c"):
    """Posterior deltas b - a, plus the Monte-Carlo tolerance on the mean shift."""
    mcse = float(np.hypot(a[key]["mcse_mean"], b[key]["mcse_mean"]))
    return {
        "delta_median": b[key]["median"] - a[key]["median"],
        "delta_mean": b[key]["mean"] - a[key]["mean"],
        "ratio_median": b[key]["median"] / a[key]["median"],
        "mcse_combined": mcse,
        "delta_mean_in_mcse": (b[key]["mean"] - a[key]["mean"]) / mcse if mcse > 0 else np.inf,
        "delta_in_sd_of_a": (b[key]["median"] - a[key]["median"]) / a[key]["sd"],
    }


# ---------------------------------------------------------------------------
# C1: deterministic quadrature control + numerical floor
# ---------------------------------------------------------------------------
def stage_c1():
    print("\n=== C1: quadrature vs closed form (S=1), and the numerical floor ===", flush=True)
    ones = np.ones(NODES)
    rows = []
    for R_c in (0.8, 1.0, 1.3, 1.384, 1.6, 2.0, 3.0):
        for R_t in (4.0, 8.0, 15.0, 30.0, 54.0, 80.0, 150.0):
            if R_t <= R_c:
                continue
            for k, b in ((8.5, 0.024), (1.0, 0.0), (26.0, 5.1)):
                closed = float(king_expected_count(k, b, R_c, R_t, FIELD))
                weighted = float(king_expected_count_weighted(k, b, R_c, R_t, FIELD, ones))
                rows.append((R_c, R_t, k, b, abs(weighted / closed - 1.0)))
    err = np.array([r[-1] for r in rows])
    worst = rows[int(np.argmax(err))]
    r_nodes = quadrature_radii()
    payload = {
        "n_configs": len(rows),
        "max_rel_err": float(err.max()),
        "median_rel_err": float(np.median(err)),
        "worst_config": {"R_c": worst[0], "R_t": worst[1], "k": worst[2], "b": worst[3]},
        "gate_max_rel_err": C1_GATE,
        "pass": bool(err.max() < C1_GATE),
        "nodes_inside_published_Rc": int((r_nodes < PUBLISHED_RC).sum()),
        "nodes_inside_5arcmin": int((r_nodes < 5.0).sum()),
        "smallest_node_arcmin": float(r_nodes.min()),
        "effect_size_being_hunted": SUPPRESSION_GATE,
        "floor_far_below_effect": bool(err.max() < 0.1 * SUPPRESSION_GATE),
    }
    print(
        f"  max rel err {payload['max_rel_err']:.3e} (gate {C1_GATE:.0e})  "
        f"median {payload['median_rel_err']:.3e}  -> {'PASS' if payload['pass'] else 'FAIL'}"
    )
    print(
        f"  {payload['nodes_inside_published_Rc']} of {NODES} quadrature nodes lie inside "
        f"R_c={PUBLISHED_RC}'; smallest node {payload['smallest_node_arcmin']:.2e}'"
    )
    _save("c1_quadrature_control", payload)
    return payload


# ---------------------------------------------------------------------------
# S-bar loading and profile description
# ---------------------------------------------------------------------------
def order_arcmin(order):
    """Mean healpix pixel diameter at `order`, in arcmin."""
    return 2.0 * 60.0 * np.sqrt(41253.0 / (12 * 4**order) / np.pi)


def describe_sbar(r_nodes, s_nodes, *, source, orders=None, resolution_arcmin=None):
    r"""Summarise an :math:`\bar S(r)`, including whether it can see the core.

    ``resolving`` is decided from a **measured spatial resolution in arcmin**, not
    from a source label: a healpix map contributes the pixel diameter at its
    finest attained order, a binned empirical curve contributes the radius below
    which it is a flat extrapolation (its innermost bin centre). The gate is
    ``resolution <= 1.2 R_c = 1.66'``.

    .. note::
       This field was first written as ``max(orders) >= 11``, with non-healpix
       sources carrying a sentinel ``orders=[99]``. That made the gate
       **unfailable for exactly the curves whose resolution was in question** --
       the pipeline retention curve, whose innermost bin centre is 1.70', was
       reported as resolving. A check that cannot fail is worse than none.
    """
    interp = lambda rr: np.interp(np.asarray(rr, float), r_nodes, s_nodes)  # noqa: E731
    supp = core_suppression(interp)
    if resolution_arcmin is None and orders is not None:
        resolution_arcmin = float(order_arcmin(max(orders)))
    inner = s_nodes[r_nodes <= 5.0]
    outer = s_nodes[r_nodes >= 50.0]
    payload = {
        "source": str(source),
        "orders_attained": orders,
        "n_nodes": int(s_nodes.size),
        "min": float(s_nodes.min()),
        "max": float(s_nodes.max()),
        "range": float(s_nodes.max() - s_nodes.min()),
        "S_at_smallest_node": float(s_nodes[0]),
        "mean_inside_5arcmin": float(inner.mean()),
        "mean_beyond_50arcmin": float(outer.mean()),
        "ratio_inner_outer": float(inner.mean() / outer.mean()),
        "core_suppression": supp,
        "core_suppression_pct": 100.0 * supp,
        "suppression_gate_pct": 100.0 * SUPPRESSION_GATE,
        "flat_at_this_resolution": bool(abs(supp) < SUPPRESSION_GATE),
        "resolution_arcmin": resolution_arcmin,
        "resolution_gate_arcmin": RESOLVING_ARCMIN,
        "resolution_in_units_of_Rc": (
            None if resolution_arcmin is None else float(resolution_arcmin / PUBLISHED_RC)
        ),
        "resolving": bool(resolution_arcmin is not None and resolution_arcmin <= RESOLVING_ARCMIN),
        "profile_at": {
            f"{rr:g}": float(interp(rr))
            for rr in (0.5, 1.0, 1.384, 2.0, 5.0, 10.0, 20.0, 35.0, 50.0, 70.0)
        },
    }
    return payload


# Healpix orders the archive-free maps top out at. `patch` records its own.
KNOWN_MODE_ORDERS = {"hpx7": [7], "multi": [10]}


def load_sbar(path: Path):
    with np.load(path, allow_pickle=False) as z:
        r = np.asarray(z["r_arcmin"], float)
        s = np.asarray(z["completeness"], float)
        orders = [int(v) for v in np.atleast_1d(z["orders"])] if "orders" in z.files else None
        mode = str(z["mode"]) if "mode" in z.files else path.stem
        res = float(z["resolution_arcmin"]) if "resolution_arcmin" in z.files else None
    if orders is None:
        orders = KNOWN_MODE_ORDERS.get(mode)
    expected = quadrature_radii()
    if r.shape != expected.shape or not np.allclose(r, expected, rtol=1e-10, atol=1e-10):
        raise ValueError(
            f"{path.name}: r_arcmin is not the 256-node Gauss-Legendre grid for "
            f"field_radius={FIELD}. It cannot be passed as completeness= without "
            "re-interpolation."
        )
    return r, s, orders, mode, res


def stage_sbar(sbar_path: Path):
    print(f"\n=== S-bar profile from {sbar_path.name} ===", flush=True)
    r, s, orders, mode, res = load_sbar(sbar_path)
    payload = describe_sbar(
        r,
        s,
        source=f"{sbar_path.name} (mode={mode})",
        orders=orders,
        resolution_arcmin=res,
    )
    print(
        f"  min={payload['min']:.5f} max={payload['max']:.5f} "
        f"core suppression={payload['core_suppression_pct']:+.3f}% "
        f"(gate {payload['suppression_gate_pct']:.1f}%)"
    )
    print(
        f"  orders attained: {orders}  resolution {payload['resolution_arcmin']}'"
        f" = {payload['resolution_in_units_of_Rc']} R_c  "
        f"resolving(<= {RESOLVING_ARCMIN:.2f}'): {payload['resolving']}"
    )
    _save(f"sbar_{mode}", payload)
    _save_arrays(**{f"sbar_{mode}_r": r, f"sbar_{mode}_S": s})
    return r, s, payload, mode


# ---------------------------------------------------------------------------
# main both-ways fit + C2 + C3
# ---------------------------------------------------------------------------
def stage_fits(sbar_path: Path | None, draws=2000, chains=4):
    radii, gmag, dropped = member_radii()
    print(
        f"\n=== both-ways fit on the real member sample: N={radii.size} "
        f"(dropped outside {FIELD}': {dropped}) ===",
        flush=True,
    )
    if gmag is not None:
        print(f"  Gmag median {np.median(gmag):.2f}, p98 {np.percentile(gmag, 98):.2f}")

    uncorr, tr_u = _fit(radii, "uncorrected (completeness=None)", draws=draws, chains=chains)
    _save("fit_uncorrected", uncorr)
    _save_arrays(
        radii_arcmin=radii, R_c_uncorrected=np.asarray(tr_u.posterior["R_c"].values).ravel()
    )

    ones = np.ones(NODES)
    c2_fit, tr_1 = _fit(radii, "C2 control (S=1)", completeness=ones, draws=draws, chains=chains)
    c2 = {
        "fit": c2_fit,
        "delta_R_c": _delta(uncorr, c2_fit, "R_c"),
        "delta_k": _delta(uncorr, c2_fit, "k"),
        "gate_sigma": C2_SIGMA,
        "note": (
            "Exact reproduction is impossible: the corrected path builds a "
            "quadrature graph and the uncorrected one a closed form, so NUTS "
            "trajectories differ at identical seed. This is a two-sigma "
            "Monte-Carlo agreement test on the posterior mean."
        ),
    }
    c2["pass"] = bool(abs(c2["delta_R_c"]["delta_mean_in_mcse"]) < C2_SIGMA)
    print(
        f"  C2: dR_c(mean) = {c2['delta_R_c']['delta_mean']:+.5f}' = "
        f"{c2['delta_R_c']['delta_mean_in_mcse']:+.2f} MCSE -> "
        f"{'PASS' if c2['pass'] else 'FAIL'}"
    )
    _save("c2_flat_control", c2)

    # C3 mutation: a deliberately steep S applied to the real radii.
    steep = steep_completeness()
    r_nodes = quadrature_radii()
    s_steep = steep(r_nodes)
    supp = core_suppression(steep)
    c3_fit, _ = _fit(
        radii,
        f"C3 mutation (steep S, supp={supp:.1%})",
        completeness=s_steep,
        draws=draws,
        chains=chains,
    )
    ratio = c3_fit["R_c"]["median"] / uncorr["R_c"]["median"]
    c3 = {
        "fit": c3_fit,
        "injected_core_suppression": supp,
        "predicted_ratio": C3_PREDICTED_RATIO,
        "predicted_from": "completeness_bias_scaling.json: 42.5% suppression <-> +37.9% R_c bias",
        "observed_ratio": float(ratio),
        "band": list(C3_RATIO_BAND),
        "pass": bool(C3_RATIO_BAND[0] <= ratio <= C3_RATIO_BAND[1]),
    }
    print(
        f"  C3: R_c ratio corrected/uncorrected = {ratio:.3f} "
        f"(predicted {C3_PREDICTED_RATIO:.3f}, band {C3_RATIO_BAND}) -> "
        f"{'PASS' if c3['pass'] else 'FAIL'}"
    )
    _save("c3_mutation", c3)

    if sbar_path is None:
        print("  (no --sbar given: skipping the real corrected fit)")
        return
    r, s, sbar_desc, mode = stage_sbar(sbar_path)
    corr, tr_c = _fit(radii, f"corrected ({mode})", completeness=s, draws=draws, chains=chains)
    _save(f"fit_corrected_{mode}", corr)
    _save_arrays(**{f"R_c_corrected_{mode}": np.asarray(tr_c.posterior["R_c"].values).ravel()})

    d_rc = _delta(uncorr, corr, "R_c")
    sig0_u, sig0_c = uncorr["Sigma0"]["median"], corr["Sigma0"]["median"]
    verdict = {
        "sbar": sbar_desc,
        "delta_R_c": d_rc,
        "delta_R_t": _delta(uncorr, corr, "R_t"),
        "delta_k": _delta(uncorr, corr, "k"),
        "Sigma0_uncorrected": sig0_u,
        "Sigma0_corrected": sig0_c,
        "Sigma0_fractional_change": float(sig0_c / sig0_u - 1.0),
        "published_R_c": PUBLISHED_RC,
        "published_band": list(PUBLISHED_BAND),
        # --- the pre-registered decisions ---
        "H1_criterion_a_suppression_below_gate": bool(
            abs(sbar_desc["core_suppression"]) < SUPPRESSION_GATE
        ),
        "H1_criterion_b_delta_below_gate": bool(
            abs(d_rc["delta_median"]) < DELTA_RC_GATE
            and abs(d_rc["delta_in_sd_of_a"]) < DELTA_RC_SD_FRACTION
        ),
        "H1_measured_at_resolving_resolution": bool(sbar_desc["resolving"]),
        "central_density_moved": bool(abs(sig0_c / sig0_u - 1.0) >= CENTRAL_DENSITY_GATE),
        "H2_corrected_median_inside_published_band": bool(
            PUBLISHED_BAND[0] <= corr["R_c"]["median"] <= PUBLISHED_BAND[1]
        ),
        # H2 as pre-registered tests an ABSOLUTE position against the published
        # band, so it charges the completeness correction for a shift the
        # likelihood/sample change had already produced. The decomposition is
        # recorded rather than the threshold being moved after the fact: a gate
        # that tripped for an attributable reason is worth more than a moved gate.
        "H2_decomposition": {
            "published_binned_cone_R_c": PUBLISHED_RC,
            "unbinned_same_members_uncorrected_R_c": uncorr["R_c"]["median"],
            "unbinned_corrected_R_c": corr["R_c"]["median"],
            "shift_from_likelihood_and_sample": uncorr["R_c"]["median"] - PUBLISHED_RC,
            "shift_from_completeness_correction": d_rc["delta_median"],
            "fraction_of_total_shift_from_correction": float(
                abs(d_rc["delta_median"]) / max(abs(corr["R_c"]["median"] - PUBLISHED_RC), 1e-12)
            ),
            "note": (
                "the likelihood/sample change is measured in "
                "tools/validation/king_unbinned_delta.py and is not this thread's result"
            ),
        },
    }
    verdict["H1_falsified_correction_does_not_matter"] = bool(
        verdict["H1_criterion_a_suppression_below_gate"]
        and verdict["H1_criterion_b_delta_below_gate"]
        and not verdict["central_density_moved"]
    )
    verdict["P01_correction_required"] = not verdict["H2_corrected_median_inside_published_band"]
    print(
        f"\n  dR_c = {d_rc['delta_median']:+.5f}' ({d_rc['delta_in_sd_of_a']:+.3f} sd) "
        f"| Sigma0 change {verdict['Sigma0_fractional_change']:+.2%}"
    )
    print(
        f"  H1 (correction matters) falsified: {verdict['H1_falsified_correction_does_not_matter']}"
        f"  | P01 correction required: {verdict['P01_correction_required']}"
    )
    _save(f"verdict_{mode}", verdict)


# ---------------------------------------------------------------------------
# C4: positive control on synthetic data with known injected completeness
# ---------------------------------------------------------------------------
def stage_c4(realizations=6, n_draw=4000, draws=1500, chains=2):
    print("\n=== C4: recovery on synthetic data with known injected completeness ===", flush=True)
    truth = dict(k=1.0, R_c=1.38, R_t=54.0, b=0.0)
    fn = steep_completeness()
    supp = core_suppression(fn)
    r_nodes = quadrature_radii()
    s_nodes = fn(r_nodes)
    rows = []
    for i in range(realizations):
        rng = np.random.default_rng(SEED + 991 * i)
        radii = sample_king(rng, n_draw, **truth)
        observed = radii[rng.uniform(size=radii.size) < fn(radii)]
        u_fit, _ = _fit(observed, f"C4 r{i} uncorrected", draws=draws, chains=chains, seed=SEED + i)
        c_fit, _ = _fit(
            observed,
            f"C4 r{i} corrected",
            completeness=s_nodes,
            draws=draws,
            chains=chains,
            seed=SEED + i,
        )
        rows.append(
            {
                "realization": i,
                "n_observed": int(observed.size),
                "uncorrected_R_c": u_fit["R_c"]["median"],
                "uncorrected_R_c_sd": u_fit["R_c"]["sd"],
                "corrected_R_c": c_fit["R_c"]["median"],
                "corrected_R_c_sd": c_fit["R_c"]["sd"],
                "uncorrected_bias": u_fit["R_c"]["median"] / truth["R_c"] - 1.0,
                "corrected_bias": c_fit["R_c"]["median"] / truth["R_c"] - 1.0,
                "corrected_z": (c_fit["R_c"]["median"] - truth["R_c"]) / c_fit["R_c"]["sd"],
                "gates_pass": bool(u_fit["gates_pass"] and c_fit["gates_pass"]),
            }
        )
        _save(
            "c4_synthetic_recovery",
            {"partial": rows, "truth": truth, "injected_core_suppression": supp},
        )
    ub = np.array([r["uncorrected_bias"] for r in rows])
    cb = np.array([r["corrected_bias"] for r in rows])
    cz = np.array([r["corrected_z"] for r in rows])
    payload = {
        "truth": truth,
        "injected_core_suppression": supp,
        "realizations": rows,
        "uncorrected_bias_mean": float(ub.mean()),
        "uncorrected_bias_sem": float(ub.std(ddof=1) / np.sqrt(ub.size)),
        "corrected_bias_mean": float(cb.mean()),
        "corrected_bias_sem": float(cb.std(ddof=1) / np.sqrt(cb.size)),
        "corrected_mean_abs_z": float(np.abs(cz).mean()),
        "gate_uncorrected_bias_above": C4_MIN_UNCORRECTED_BIAS,
        "pass_uncorrected_is_biased": bool(ub.mean() > C4_MIN_UNCORRECTED_BIAS),
        "pass_corrected_recovers_truth": bool(np.abs(cz).mean() < 2.0),
    }
    payload["pass"] = bool(
        payload["pass_uncorrected_is_biased"] and payload["pass_corrected_recovers_truth"]
    )
    print(
        f"  injected suppression {supp:.1%}: uncorrected bias {ub.mean():+.1%} "
        f"+/- {payload['uncorrected_bias_sem']:.1%}, corrected bias {cb.mean():+.1%} "
        f"+/- {payload['corrected_bias_sem']:.1%}, |z|={payload['corrected_mean_abs_z']:.2f} "
        f"-> {'PASS' if payload['pass'] else 'FAIL'}"
    )
    _save("c4_synthetic_recovery", payload)
    return payload


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "c1", "sbar", "fits", "c4", "controls"],
    )
    ap.add_argument("--sbar", type=Path, default=None, help="npz with r_arcmin + completeness")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--realizations", type=int, default=6)
    args = ap.parse_args()

    if args.stage in ("all", "c1", "controls"):
        stage_c1()
    if args.stage == "sbar":
        if args.sbar is None:
            ap.error("--stage sbar needs --sbar")
        stage_sbar(args.sbar)
    if args.stage in ("all", "fits"):
        stage_fits(args.sbar, draws=args.draws, chains=args.chains)
    if args.stage in ("all", "c4", "controls"):
        stage_c4(realizations=args.realizations)

    print(f"\nJSON: {JSON_OUT} ({JSON_OUT.stat().st_size / 1024:.1f} KB)")
    if NPZ_OUT.exists():
        print(f"NPZ:  {NPZ_OUT} ({NPZ_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
