#!/usr/bin/env python3
r"""The pipeline's own selection function on NGC 6383, versus the survey's.

WHY THIS EXISTS
---------------
Gaia DR3's source selection function is magnitude-sensitive (S = 1 to G ~ 19,
0.85 at 20.7, 0.38 at 21, ~0 by 21.5), but NGC 6383's members are bright --
median G = 17.3, 98th percentile 20.4 -- so they sit where survey completeness is
essentially unity. If Gaia's completeness is ~1 for this sample, then **the
dominant magnitude-dependent selection acting on it is ours, not the survey's**,
and correcting the radial profile for the survey while ignoring the pipeline
corrects the smaller of the two.

That corollary was written when the pipeline clipped the **raw** parallax and
retained 99% of the bright ``Gmag`` quartile against 27% of the faintest. The
clip was redesigned (``erotica/analysis/_clipping.py``: normalised residual with
a fitted excess-variance term, ``method="normalised"`` now the default), so those
figures are historical. **This script re-measures the gradient on the clip that
currently ships**, and adds the piece A1 needs and no previous script produced:
the pipeline's retention as a function of **radius**, on the same 256-node
Gauss-Legendre grid the weighted King normalisation evaluates on, so it can be
fed to ``king_unbinned(completeness=...)`` exactly as a survey ``S(r)`` can.

WHY THE SHAPE IS THE ONLY THING THAT MATTERS
---------------------------------------------
The point-process normalisation is
:math:`\Lambda = \int 2\pi r\,\Sigma(r)\,\bar S(r)\,dr`. Scaling
:math:`\bar S` by a constant scales :math:`\Lambda` by that constant, which the
amplitude ``k`` absorbs exactly. So an overall normalisation of the pipeline's
retention is unidentifiable and irrelevant; only its **radial shape** can move
``R_c``. ``S_pipe(r)`` is therefore normalised to a maximum of 1 before being
written, and that is not a fudge -- it is the gauge the likelihood already fixes.

THE ORACLES, AND WHAT EACH CAN AND CANNOT SEE
-----------------------------------------------
1. **Real catalogue, no oracle.** Retention measured directly on the stored
   pre-clip flag column of the published extraction. This is what actually
   happened to the real stars, but it counts *rejections*, not *false*
   rejections: the sample contains contaminants and some rejections are correct.
2. **Synthetic, oracle true by construction.** Clusters in which **every star is
   a real member**, carrying the **real** per-star ``parallax_error`` from the
   published catalogue, so every rejection is a false rejection and the retention
   curve *is* the induced selection function. It cannot see contamination, and it
   cannot see the ADQL parallax window -- which a previous measurement in this
   directory established contributes ~78% of the historical faint-end loss. Both
   limitations are stated in the output rather than absorbed.

Reporting both is the point: agreement means the gradient is a property of the
clip, disagreement localises it to the real sample's structure.

FALSIFICATION -- PRE-REGISTERED
--------------------------------
Registered before the numbers existed.

* **"The pipeline's selection dominates the survey's for this cluster."** This is
  **falsified** if the pipeline's radial retention shape has a core-aperture
  suppression *smaller in magnitude* than the survey ``S(r)``'s, measured with
  the identical aperture-averaged definition. The claim is comparative, so it is
  reported as a ratio and it can come out either way.
* **"The redesigned clip removed the magnitude gradient."** **Falsified** if the
  shipped ``method="normalised"`` clip's bright-to-faint retention gradient
  exceeds **+0.10** on the real catalogue. (The historical raw clip measured
  +0.214 as published, +0.146 with package estimators.)
* **A gradient in magnitude does not imply one in radius.** They are only linked
  through ``rho(G, r)``, which is reported. If the magnitude gradient is large
  and the radial one is not, the pipeline's selection does not touch the RDP and
  this script says so.

NEGATIVE CONTROLS
-----------------
* **Reproduction gate.** The stored flag columns must still give the published
  counts before anything is measured; a drifted input file invalidates every row.
* **Permutation null.** ``Gmag`` is shuffled against the clip decision and the
  radial retention recomputed, many times. The measured radial gradient must lie
  outside that null band, otherwise it is what a magnitude-blind clip would
  produce by chance and there is nothing to correct.

USAGE
-----
    python tools/validation/a1_pipeline_selection.py

Writes ``a1_pipeline_selection.json`` (incrementally, per section) and
``a1_sbar_pipeline.npz``, which has the same layout as the survey ``S(r)`` files
and can be passed to ``a1_selection_corrected_rdp.py --sbar``.
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
from astropy.table import QTable
from scipy import stats

from erotica.analysis._clipping import sigma_clip_parallax

HERE = Path(__file__).resolve().parent
JSON_OUT = HERE / "a1_pipeline_selection.json"
SBAR_NPZ = HERE / "a1_sbar_pipeline.npz"
SBAR_PUBLISHED_NPZ = HERE / "a1_sbar_pipeline_published.npz"

B = Path("/Users/notluquis/erotica/data/test/NGC6383")
FLAGS_70 = B / "comments_paper/radius_robustness/generated/70/paperfaithful_with_clip_flags.ecsv"
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
FIELD = 70.0
NODES = 256
SIGMA = 2.0
CORE_APERTURE = 1.38
SEED = 20260804

# pre-registered
MAG_GRADIENT_GATE = 0.10
PUBLISHED_COUNTS_70 = {"preclip_p05": 771, "postclip_p05": 650, "p06": 628}


def _save(key, payload):
    doc = {}
    if JSON_OUT.exists():
        try:
            doc = json.loads(JSON_OUT.read_text())
        except json.JSONDecodeError:
            pass
    doc[key] = payload
    doc["_written"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    JSON_OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(f"  [saved '{key}']", flush=True)


def quadrature_radii(field_radius=FIELD, nodes=NODES):
    x, _ = np.polynomial.legendre.leggauss(nodes)
    return 0.5 * field_radius * (x + 1.0)


def core_suppression_from_curve(r_grid, s_grid, aperture=CORE_APERTURE, field=FIELD):
    """Identical definition to the survey side: area-weighted inner/outer ratio."""
    inner = np.linspace(1e-6, aperture, 2000)
    outer = np.linspace(1e-6, field, 20000)
    f = lambda rr: np.interp(rr, r_grid, s_grid)  # noqa: E731
    s_in = np.trapezoid(2 * np.pi * inner * f(inner), inner) / (np.pi * aperture**2)
    s_out = np.trapezoid(2 * np.pi * outer * f(outer), outer) / (np.pi * field**2)
    return float(1.0 - s_in / s_out)


def load():
    t = QTable.read(FLAGS_70)
    preclip = np.asarray(t["paper_reference_preclip_p05"], bool)
    postclip = np.asarray(t["paper_reference_postclip_p05"], bool)
    p06 = np.asarray(t["paper_reference_p06"], bool)
    sky = SkyCoord(np.asarray(t["ra"], float) * u.deg, np.asarray(t["dec"], float) * u.deg)
    r = CENTER.separation(sky).to(u.arcmin).value
    return {
        "table": t,
        "preclip": preclip,
        "postclip": postclip,
        "p06": p06,
        "gmag": np.asarray(t["Gmag"], float),
        "plx": np.asarray(t["parallax"], float),
        "eplx": np.asarray(t["parallax_error"], float),
        "radius": r,
    }


def _clip(plx, eplx, method):
    work = QTable(
        {
            "parallax": plx,
            "parallax_error": eplx,
            "cluster": np.zeros(plx.size, dtype=np.int64),
        }
    )
    _lo, _hi, _copy, keep, _noise = sigma_clip_parallax(
        work,
        cluster=0,
        sigma=SIGMA,
        use_biweight=True,
        in_place=False,
        mark_label=-1,
        print_results=False,
        return_mask=True,
        preselector_mask=None,
        method=method,
    )
    return np.asarray(keep, bool)


def _by_quartile(values, keep):
    edges = np.quantile(values, [0.25, 0.5, 0.75])
    q = np.digitize(values, edges)
    per = [float(keep[q == k].mean()) for k in range(4)]
    return {
        "overall": float(keep.mean()),
        "by_quartile": per,
        "gradient_first_minus_last": float(per[0] - per[3]),
        "quartile_medians": [float(np.median(values[q == k])) for k in range(4)],
    }


#: Fixed radial edges that reach inside ``R_c``. Equal-count binning cannot:
#: only 43 of the 771 pre-clip stars lie inside 1.384'.
CORE_EDGES = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, FIELD]


def _fixed_bin_retention(radius, keep, edges):
    """Retention in fixed radial bins, with the binomial SE on each point."""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=False):
        m = (radius >= lo) & (radius < hi)
        n = int(m.sum())
        p = float(keep[m].mean()) if n else float("nan")
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "n": n,
                "retention": p,
                "binomial_se": float(np.sqrt(p * (1 - p) / n)) if n else None,
            }
        )
    return rows


def _radial_retention(radius, keep, n_bins=8):
    """Retention in equal-count radial bins, plus a smooth curve on the GL grid."""
    edges = np.quantile(radius, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = 0.0, FIELD
    idx = np.clip(np.digitize(radius, edges[1:-1]), 0, n_bins - 1)
    centres, ret, n = [], [], []
    for k in range(n_bins):
        m = idx == k
        centres.append(float(np.median(radius[m])))
        ret.append(float(keep[m].mean()))
        n.append(int(m.sum()))
    centres, ret = np.asarray(centres), np.asarray(ret)
    nodes = quadrature_radii()
    curve = np.interp(nodes, centres, ret)
    return centres, ret, np.asarray(n), nodes, curve


# ---------------------------------------------------------------------------
def section_reproduction(d):
    got = {
        "preclip_p05": int(d["preclip"].sum()),
        "postclip_p05": int(d["postclip"].sum()),
        "p06": int(d["p06"].sum()),
    }
    ok = got == PUBLISHED_COUNTS_70
    print(f"  reproduction gate: {got} vs {PUBLISHED_COUNTS_70} -> {'PASS' if ok else 'FAIL'}")
    _save("reproduction_gate", {"measured": got, "expected": PUBLISHED_COUNTS_70, "pass": ok})
    if not ok:
        raise SystemExit(
            "flag columns drifted; every number below would be measured on a "
            "different sample than the published one"
        )


def section_magnitude(d):
    print("\n=== magnitude dependence: the clip that currently ships ===", flush=True)
    pre = d["preclip"]
    plx, eplx, gmag = d["plx"][pre], d["eplx"][pre], d["gmag"][pre]
    rows = {}
    rows["published (stored flags: histogram mode + std, raw parallax)"] = _by_quartile(
        gmag, d["postclip"][pre]
    )
    for label, method in (
        ("package raw clip (biweight, raw parallax)", "raw"),
        ("SHIPPED normalised clip (z with excess variance)", "normalised"),
    ):
        rows[label] = _by_quartile(gmag, _clip(plx, eplx, method))
    for label, r in rows.items():
        print(
            f"  {label:52s} overall {r['overall']:6.1%}  "
            f"Q1 {r['by_quartile'][0]:6.1%} Q4 {r['by_quartile'][3]:6.1%}  "
            f"gradient {r['gradient_first_minus_last']:+.3f}"
        )
    shipped = rows["SHIPPED normalised clip (z with excess variance)"]
    payload = {
        "n_preclip": int(pre.sum()),
        "median_e_plx_by_gmag_quartile": _by_quartile(gmag, np.ones(gmag.size, bool))[
            "quartile_medians"
        ],
        "rows": rows,
        "gate_gradient_below": MAG_GRADIENT_GATE,
        "shipped_gradient": shipped["gradient_first_minus_last"],
        "shipped_gradient_within_gate": bool(
            abs(shipped["gradient_first_minus_last"]) <= MAG_GRADIENT_GATE
        ),
        "historical_raw_gradient_published": 0.214,
    }
    # e_Plx per quartile, measured properly
    edges = np.quantile(gmag, [0.25, 0.5, 0.75])
    q = np.digitize(gmag, edges)
    payload["median_e_plx_by_gmag_quartile"] = [float(np.median(eplx[q == k])) for k in range(4)]
    payload["median_gmag_by_quartile"] = [float(np.median(gmag[q == k])) for k in range(4)]
    _save("magnitude_dependence", payload)
    return payload


def section_synthetic(d, n_real=200):
    """Oracle arm: every simulated star is a member, so any rejection is false."""
    print("\n=== synthetic oracle: every star a true member ===", flush=True)
    pre = d["preclip"]
    eplx, gmag = d["eplx"][pre], d["gmag"][pre]
    ok = np.isfinite(eplx) & np.isfinite(gmag) & (eplx > 0)
    eplx, gmag = eplx[ok], gmag[ok]
    rng = np.random.default_rng(SEED)
    true_plx = 0.90
    keeps = {"raw": [], "normalised": []}
    for _ in range(n_real):
        obs = true_plx + rng.normal(0, eplx)
        for method in ("raw", "normalised"):
            keeps[method].append(_clip(obs, eplx, method))
    payload = {"n_stars": int(eplx.size), "n_realizations": n_real, "true_parallax_mas": true_plx}
    for method, arr in keeps.items():
        a = np.asarray(arr)
        payload[method] = _by_quartile(gmag, a.mean(axis=0))
        print(
            f"  {method:11s} overall {payload[method]['overall']:6.1%}  "
            f"Q1 {payload[method]['by_quartile'][0]:6.1%} "
            f"Q4 {payload[method]['by_quartile'][3]:6.1%}  "
            f"gradient {payload[method]['gradient_first_minus_last']:+.3f}"
        )
    payload["caveats"] = [
        "no contaminants: cannot price the normalised clip's greater permissiveness",
        "no ADQL parallax window: a prior measurement attributed ~78% of the historical "
        "faint-end loss to the query window rather than the clip",
    ]
    _save("synthetic_oracle", payload)
    return payload


def section_radial(d, n_perm=2000):
    print("\n=== the piece A1 needs: retention versus RADIUS ===", flush=True)
    pre = d["preclip"] & (d["radius"] <= FIELD)
    radius, gmag = d["radius"][pre], d["gmag"][pre]
    plx, eplx = d["plx"][pre], d["eplx"][pre]
    rho, pval = stats.spearmanr(gmag, radius)
    print(f"  Spearman rho(Gmag, r) = {rho:+.4f}  p = {pval:.3g}   (N = {pre.sum()})")

    out = {
        "n": int(pre.sum()),
        "spearman_rho_gmag_radius": float(rho),
        "spearman_p": float(pval),
        "curves": {},
    }
    nodes = quadrature_radii()
    curves = {}
    for label, keep in (
        ("published_stored_flags", d["postclip"][pre]),
        ("shipped_normalised", _clip(plx, eplx, "normalised")),
        ("raw", _clip(plx, eplx, "raw")),
    ):
        centres, ret, n, nodes, curve = _radial_retention(radius, keep)
        supp = core_suppression_from_curve(nodes, curve / curve.max())
        # permutation null: shuffle the clip decision, keep the radii
        rng = np.random.default_rng(SEED)
        null = np.empty(n_perm)
        for i in range(n_perm):
            _, r_perm, _, _, c_perm = _radial_retention(radius, rng.permutation(keep))
            null[i] = core_suppression_from_curve(nodes, c_perm / max(c_perm.max(), 1e-12))
        p_two = float(np.mean(np.abs(null) >= abs(supp)))
        curves[label] = curve
        # binning robustness: the suppression must not be an artefact of n_bins
        by_bins = {}
        for nb in (5, 6, 8, 10, 12):
            _, _, _, nn, cc = _radial_retention(radius, keep, n_bins=nb)
            by_bins[str(nb)] = core_suppression_from_curve(nn, cc / cc.max())
        # A binning that actually reaches inside R_c = 1.38'. Equal-count octiles
        # put the innermost centre at 1.70' = 1.23 R_c, so nothing below that is
        # measured; these fixed edges do reach in, at the price of small N -- the
        # binomial standard error is carried so the inner points cannot be over-read.
        fixed = _fixed_bin_retention(radius, keep, CORE_EDGES)
        out["curves"][label] = {
            "bin_centres_arcmin": centres.tolist(),
            "retention": ret.tolist(),
            "n_per_bin": n.tolist(),
            "overall_retention": float(keep.mean()),
            "core_suppression": supp,
            "core_suppression_pct": 100.0 * supp,
            "core_suppression_by_n_bins": by_bins,
            "core_suppression_bin_spread": float(max(by_bins.values()) - min(by_bins.values())),
            "core_resolving_bins": fixed,
            "permutation_null_sd": float(null.std(ddof=1)),
            "permutation_null_p": p_two,
            "n_permutations": n_perm,
            "beyond_null": bool(p_two < 0.05),
        }
        print(
            f"  {label:24s} retention {ret.min():.3f}..{ret.max():.3f}  "
            f"core suppression {100 * supp:+.3f}%  perm-null sd {100 * null.std(ddof=1):.3f}%  "
            f"p={p_two:.4f}"
        )
    _save("radial_selection", out)

    # Both variants get an npz: the shipped clip is what a *new* analysis induces,
    # the published one is what the *published* sample carries. They answer
    # different questions and only the second speaks to P01.
    for label, mode, path in (
        ("shipped_normalised", "pipeline", SBAR_NPZ),
        ("published_stored_flags", "pipeline_published", SBAR_PUBLISHED_NPZ),
    ):
        c = curves[label]
        # `resolution_arcmin` is the innermost bin centre: below it the curve is a
        # flat extrapolation and can see nothing. It is NOT a sentinel -- the
        # downstream resolving gate is applied to this number and 1.70' fails it.
        np.savez_compressed(
            path,
            r_arcmin=nodes,
            completeness=np.clip(c / c.max(), 0.0, 1.0),
            mode=mode,
            resolution_arcmin=float(out["curves"][label]["bin_centres_arcmin"][0]),
            field_radius=FIELD,
            note="pipeline retention shape, max-normalised; overall scale is absorbed by k",
        )
        print(f"  wrote {path.name}")
    return out


def section_comparison(survey_json=None):
    """Survey versus pipeline, on the identical aperture-averaged definition."""
    print("\n=== which selection dominates? ===", flush=True)
    doc = json.loads(JSON_OUT.read_text())
    curves = doc["radial_selection"]["curves"]
    pipe = curves["shipped_normalised"]["core_suppression"]
    pipe_pub = curves["published_stored_flags"]["core_suppression"]
    survey = {}
    for _name, path in (("multi_order10", HERE / "a1_selection_corrected_rdp.json"),):
        if path.exists():
            sub = json.loads(path.read_text())
            for key, val in sub.items():
                # Only *survey* S(r) belongs on the other side of this comparison.
                # The pipeline curves in that file were written by this script, so
                # including them produced circular "pipeline vs itself" rows.
                if key.startswith("sbar_") and "pipeline" not in key and "combined" not in key:
                    survey[key] = {
                        "core_suppression": val["core_suppression"],
                        "orders_attained": val.get("orders_attained"),
                        "resolving": val.get("resolving"),
                    }
    payload = {
        "pipeline_shipped_core_suppression": pipe,
        "pipeline_published_core_suppression": pipe_pub,
        "pipeline_shipped_beyond_null": curves["shipped_normalised"]["beyond_null"],
        "pipeline_published_beyond_null": curves["published_stored_flags"]["beyond_null"],
        "survey": survey,
        "definition": "1 - <S>_{r<1.38'} / <S>_{r<70'}, both area-weighted",
        "sign_convention": "positive = core suppressed relative to the field; "
        "negative = core over-retained",
    }
    for key, val in survey.items():
        s = val["core_suppression"]
        for who, v in (("shipped", pipe), ("published", pipe_pub)):
            payload[f"ratio_pipeline_{who}_over_{key}"] = float(v / s) if s else None
            print(
                f"  pipeline[{who:9s}] {100 * v:+.4f}%  vs  {key} {100 * s:+.4f}%  "
                f"ratio {v / s:+.2f}x  (survey resolving={val['resolving']})"
            )
    _save("survey_vs_pipeline", payload)
    return payload


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--realizations", type=int, default=200)
    ap.add_argument("--permutations", type=int, default=2000)
    args = ap.parse_args()

    d = load()
    section_reproduction(d)
    section_magnitude(d)
    section_synthetic(d, n_real=args.realizations)
    section_radial(d, n_perm=args.permutations)
    section_comparison()
    print(f"\nJSON: {JSON_OUT} ({JSON_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
