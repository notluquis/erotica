#!/usr/bin/env python3
r"""What the three 2026-08-02 wiring fixes do to the published NGC 6383 numbers.

WHY THIS EXISTS
---------------
Three defects were fixed in one pass, and each of them changes a number that P01
reports. The fixes share a shape -- *the correct capability already existed and
the default path did not use it* -- but they do **not** share a mechanism, and
two of them interact: the clip changes which stars are members, and the members
are the input to everything else.

1. ``ClusterInferenceAnalyzer.distance_and_parallax_by_probability`` read
   ``parallax_error`` only in order to discard stars, and forwarded neither the
   per-star errors, nor ``r_lo_geo``/``r_hi_geo``, nor ``zero_point`` to the
   models. Affects the mean parallax, its uncertainty, and the distance.
2. ``ClusterStructureAnalyzer.fit_king_profile`` routed to ``RDP_bayesian``
   (Gaussian on equal-count binned densities, data-derived prior bounds) while
   ``king_unbinned`` existed. Affects ``R_c`` and ``R_t``.
3. ``sigma_clip_parallax`` clipped the **raw** parallax, which is a
   magnitude-dependent selection function. Affects the member count, and through
   it everything above.

WHY A LADDER AND NOT A BEFORE/AFTER
-----------------------------------
Flipping all three at once produces a delta nobody can attribute. This project
has already published a causal attribution twice and had to retract it twice
(``decisions.md``, the ADQL-window-versus-clip entry), for exactly this reason.
So each switch is thrown **one at a time, in a fixed order**, and every rung
reports the number it moved.

The ladder is not order-free either -- near-redundant switches never are -- and
where the order matters that is stated rather than hidden behind a single
"contribution" column.

WHAT WOULD FALSIFY THE CONCLUSIONS
----------------------------------
* **Stage A** claims the clip basis changes the member list. Falsified if the
  raw and normalised clips select the same stars, or if the magnitude-quartile
  retention gradient of the two agrees.
* **Stage B** claims the wiring changes the *width* of the parallax and distance
  posteriors far more than their centres. Falsified if ``mu_parallax`` or
  ``mu_r`` moves by more than its own published uncertainty.
* **Stage C** claims the likelihood change moves ``R_c`` modestly and destroys
  the apparent constraint on ``R_t``. Falsified if the unbinned ``R_t``
  posterior comes back with a comparable width to the binned one -- that would
  mean the binned interval was data-driven after all.

NEGATIVE CONTROLS RUN AND REPORTED
----------------------------------
* **A reproduction gate.** Stage A reproduces the published member counts
  (701 / 412 / 321 / 254 at 40') from the stored flag columns before anything is
  changed. Stage B0 reruns the *old* defaults, so the ladder starts from the
  published configuration rather than from an assumed one.
* **An estimator control.** The published clip used
  ``cenfunc=histogram_mode, stdfunc="std"``; the package function uses biweight
  location and scale. Stage A therefore runs the package's own **raw** clip as
  well, so the raw-to-normalised difference is measured with the estimator held
  fixed and is not confounded with the estimator change.
* **A null.** Stage B2 adds only the zero-point nuisance, which is exactly
  degenerate with the mean parallax. Its effect on ``mu_parallax`` itself must be
  consistent with zero; only the uncertainty may move. If the mean moves, the
  nuisance is acting as a correction, which is not what it is for.

USAGE
-----
    python tools/validation/wiring_fix_deltas.py
    python tools/validation/wiring_fix_deltas.py --stage A     # clip only, no sampling
    python tools/validation/wiring_fix_deltas.py --draws 500   # quick pass

Writes ``wiring_fix_deltas.json``.

.. note::
   ``chains=2, cores=1`` throughout. Four parallel chains have exhausted RAM on
   the development machine for models of this size.
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

BASE = Path("/Users/notluquis/erotica/data/test/NGC6383")
FLAGS_40 = BASE / "comments_paper/radius_robustness/generated/40/paperfaithful_with_clip_flags.ecsv"
FLAGS_70 = BASE / "comments_paper/radius_robustness/generated/70/paperfaithful_with_clip_flags.ecsv"

#: ICRS centre used by the published structural fits.
CENTRE = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
FIELD_40 = 40.0 * u.arcmin
FIELD_70 = 70.0 * u.arcmin
SIGMA = 2.0
PROBABILITY_CUT = 0.6

#: Published values, transcribed from the manuscript so the deltas below are
#: against what a reader of the paper actually has. Section references are to
#: ``submission_package/clean_source/aanda.tex``.
PUBLISHED = {
    "n_members_40": 254,  # abstract; reference sample, p >= 0.6
    "n_candidates_40": 321,  # abstract; candidates, p > 0.5
    "mean_parallax_mas": 0.908,  # Table 1 and Sect. astrometry
    "parallax_dispersion_mas": 0.046,
    "parallax_sem_mas": 0.004,
    "distance_kpc": 1.11,
    "distance_err_kpc": 0.06,
    "R_c_arcmin": 1.96,  # abstract; 40' production fit
    "R_c_err_arcmin": (0.19, 0.16),
    "R_t_arcmin": 54.0,  # abstract; measured on the 70' extraction
    "R_t_err_arcmin": (7.0, 11.0),
}


def sampling(draws, seed):
    from erotica.analysis.inference import SamplingConfig

    return SamplingConfig(
        draws=draws,
        tune=1000,
        chains=2,
        random_seed=seed,
        progressbar=False,
        extra_kwargs={"cores": 1},
    )


def radii_arcmin(table):
    sky = SkyCoord(np.asarray(table["ra"], float) * u.deg, np.asarray(table["dec"], float) * u.deg)
    return CENTRE.separation(sky).to(u.arcmin)


def retention_by_magnitude_quartile(gmag_all, keep):
    """Retention per ``Gmag`` quartile, and the bright-to-faint gradient.

    The quartile edges come from the stars entering the clip, not from the
    survivors, so the four bins are the same for every clip basis and the numbers
    are comparable across rows.
    """
    edges = np.quantile(gmag_all, [0.25, 0.5, 0.75])
    q = np.digitize(gmag_all, edges)
    per_q = [float(keep[q == k].mean()) for k in range(4)]
    return {
        "overall": float(keep.mean()),
        "by_gmag_quartile": per_q,
        "bright_to_faint_gradient": float(per_q[0] - per_q[3]),
    }


# ---------------------------------------------------------------------------
# Stage A -- the membership clip. No sampling.
# ---------------------------------------------------------------------------


def stage_a(path, field_label):
    from erotica.analysis._clipping import sigma_clip_parallax

    table = QTable.read(path)
    preclip = np.asarray(table["paper_reference_preclip_p05"], dtype=bool)
    published = np.asarray(table["paper_reference_p06"], dtype=bool)
    probability = np.asarray(table["probability"], dtype=float)
    gmag = np.asarray(table["Gmag"], dtype=float)

    work = QTable(
        {
            "parallax": np.asarray(table["parallax"], dtype=float),
            "parallax_error": np.asarray(table["parallax_error"], dtype=float),
            "cluster": np.where(preclip, 0, -1).astype(np.int64),
        }
    )
    common = dict(
        cluster=0,
        sigma=SIGMA,
        use_biweight=True,
        in_place=False,
        mark_label=-1,
        print_results=False,
        return_mask=True,
        preselector_mask=None,
    )

    postclip = np.asarray(table["paper_reference_postclip_p05"], dtype=bool)

    rows = {}
    for label, kwargs in (
        ("published (histogram mode + std, raw parallax)", None),
        ("package raw clip (biweight, raw parallax)", {"method": "raw"}),
        ("package normalised clip (biweight, z = (plx - plx0)/err)", {"method": "normalised"}),
    ):
        if kwargs is None:
            # The published clip survivors, recovered from the stored flags rather
            # than re-derived: `postclip_p05` IS `preclip & clip`, so restricting it
            # to `preclip` isolates the clip from the probability cut and makes the
            # retention numbers directly comparable with the rows below.
            clip_keep = postclip
            members = published
        else:
            _lo, _hi, _copy, clip_keep, _noise = sigma_clip_parallax(work, **common, **kwargs)
            clip_keep = np.asarray(clip_keep, dtype=bool)
            members = clip_keep & (probability >= PROBABILITY_CUT)
        rows[label] = {
            "n_after_clip_p05": int((clip_keep & preclip).sum()),
            "n_members_p06": int(members.sum()),
            **retention_by_magnitude_quartile(gmag[preclip], clip_keep[preclip]),
        }
        rows[label]["member_mask"] = members

    # Reproduction gate: the stored flag columns must still be what the paper says.
    reproduces = {
        "branch": int(np.asarray(table["paper_reference_branch"], bool).sum()),
        "preclip_p05": int(preclip.sum()),
        "postclip_p05": int(np.asarray(table["paper_reference_postclip_p05"], bool).sum()),
        "p06": int(published.sum()),
    }

    # maxiters sensitivity, on the raw basis so it is comparable with the published clip.
    iters = {}
    for n_iter in (1, 2, 3, 5, 10, None):
        _lo, _hi, _copy, keep, _noise = sigma_clip_parallax(
            work, **common, method="raw", maxiters=n_iter
        )
        iters[str(n_iter)] = int(np.asarray(keep, dtype=bool).sum())

    return {
        "field": field_label,
        "reproduces_published_counts": reproduces,
        "median_e_plx_by_gmag_quartile": [
            float(
                np.median(
                    np.asarray(table["parallax_error"], float)[preclip][
                        np.digitize(gmag[preclip], np.quantile(gmag[preclip], [0.25, 0.5, 0.75]))
                        == k
                    ]
                )
            )
            for k in range(4)
        ],
        "maxiters_sensitivity_raw_basis": iters,
        "rows": rows,
        "_table": table,
    }


# ---------------------------------------------------------------------------
# Stage A2 -- does P01's faint-quartile conclusion survive the shipped clip?
#
# P01 reports that the faintest magnitude quartile is less centrally concentrated
# and attributes it to Gaia incompleteness rather than mass segregation, on the
# grounds that the contrast is NOT strongest against the brightest quartile.
# `faint_quartile_clip_test.py` already re-ran that under a normalised clip -- but
# under the errors-only form, which is not what ships. This re-runs it under the
# selection the package now produces, because a conclusion validated on a
# selection nobody uses is not validated.
# ---------------------------------------------------------------------------

PUBLISHED_KS = [  # Sect. 5, faintest quartile vs each brighter one
    {"against": "8.80-15.44", "D": 0.16, "p": 0.42},
    {"against": "15.44-16.99", "D": 0.23, "p": 0.052},
    {"against": "16.99-17.93", "D": 0.28, "p": 0.010},
]


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values -- the correction P01 applies."""
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def quartile_ks(gmag, radius):
    """Faintest quartile against each brighter one, exactly as P01 does it."""
    from scipy.stats import ks_2samp

    edges = np.percentile(gmag, [25, 50, 75])
    masks = [
        gmag <= edges[0],
        (gmag > edges[0]) & (gmag <= edges[1]),
        (gmag > edges[1]) & (gmag <= edges[2]),
        gmag > edges[2],
    ]
    faint = radius[masks[3]]
    out = []
    for mask in masks[:3]:
        test = ks_2samp(faint, radius[mask])
        out.append(
            {
                "against": f"{gmag[mask].min():.2f}-{gmag[mask].max():.2f}",
                "n": int(mask.sum()),
                "D": float(test.statistic),
                "p": float(test.pvalue),
            }
        )
    adjusted = holm(np.array([c["p"] for c in out]))
    for comparison, value in zip(out, adjusted, strict=True):
        comparison["p_holm"] = float(value)
    return {
        "n_total": int(len(gmag)),
        "edges": [float(gmag.min()), *map(float, edges), float(gmag.max())],
        "comparisons": out,
        "n_significant_after_holm": int((adjusted < 0.05).sum()),
    }


def stage_a2(table, masks):
    gmag = np.asarray(table["Gmag"], dtype=float)
    radius = radii_arcmin(table).to_value(u.arcmin)
    out = {}
    for label, mask in masks.items():
        out[label] = quartile_ks(gmag[mask], radius[mask])
    return out


# ---------------------------------------------------------------------------
# Stage B -- the parallax / distance wiring, one switch at a time.
# ---------------------------------------------------------------------------

#: ``precut`` is applied by this script rather than by the analyzer, because B0
#: has to reproduce a configuration the analyzer can no longer express: the old
#: code used ``parallax_error`` to *select* and then withheld it from the model.
#: That combination is now refused (it is the defect), so the rung is built by
#: filtering the table here and handing the models a table they may not use.
B_RUNGS = [
    (
        "B0  old defaults (published configuration)",
        dict(
            precut=0.1,
            fractional_parallax_error_max=None,
            parallax_error_column=None,
            distance_lo_column=None,
            distance_hi_column=None,
            zero_point=False,
        ),
    ),
    (
        "B1  + per-star errors and Bailer-Jones bounds",
        dict(precut=0.1, fractional_parallax_error_max=None, zero_point=False),
    ),
    (
        "B2  + zero-point nuisance",
        dict(precut=0.1, fractional_parallax_error_max=None, zero_point=True),
    ),
    ("B3  - the fractional-error pre-cut  [NEW DEFAULT]", dict(precut=None)),
]


def stage_b(table, member_mask, *, draws, seed, label):
    from erotica.analysis.inference import ClusterInferenceAnalyzer

    selected = table[member_mask]
    out = []
    for rung, spec in B_RUNGS:
        kwargs = dict(spec)
        cut = kwargs.pop("precut")
        if cut is None:
            members = selected
        else:
            plx = np.asarray(selected["parallax"], float)
            err = np.asarray(selected["parallax_error"], float)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(plx > 0, err / np.where(plx > 0, plx, 1.0), np.inf)
            members = selected[(plx > 0) & (ratio <= cut)]
        analyzer = ClusterInferenceAnalyzer(members, sampling=sampling(draws, seed))
        t0 = time.perf_counter()
        res = analyzer.distance_and_parallax_by_probability(
            (PROBABILITY_CUT,), return_trace=True, **kwargs
        )
        n_fitted = int(len(members))
        # PART A floor, on the reported parameters only. The hierarchical distance
        # model also carries one latent r_true per star; those are nuisance
        # parameters and are diagnosed separately below rather than silently
        # folded into a single pass/fail.
        distance_trace, parallax_trace = res["traces"][0], res["traces"][1]
        diagnostics = {
            **{
                f"distance_{k}": v
                for k, v in _convergence(distance_trace, ["mu_r", "std_r"]).items()
            },
            **{
                f"parallax_{k}": v
                for k, v in _convergence(parallax_trace, ["mu_parallax", "sigma_parallax"]).items()
            },
            "latent_r_true_r_hat_max": _latent_rhat(distance_trace),
        }
        out.append(
            {
                **diagnostics,
                "sample": label,
                "rung": rung,
                "n_selected": int(len(selected)),
                "n_fitted": n_fitted,
                "mu_parallax_mas": res["mu_parallax_mean"][0],
                "mu_parallax_sd_mas": res["mu_parallax_std"][0],
                "sigma_parallax_mas": res["sigma_parallax_mean"][0],
                "mu_r_kpc": res["mu_r_mean"][0],
                "mu_r_sd_kpc": res["mu_r_std"][0],
                "std_r_kpc": res["std_r_mean"][0],
                "seconds": round(time.perf_counter() - t0, 1),
            }
        )
        row = out[-1]
        latent = row["latent_r_true_r_hat_max"]
        flag = (
            "" if max(row["distance_r_hat_max"], row["parallax_r_hat_max"]) < 1.01 else "  R-HAT!"
        )
        print(
            f"    {rung:52s} N={n_fitted:4d}  plx={row['mu_parallax_mas']:.4f}"
            f"+/-{row['mu_parallax_sd_mas']:.4f}  d={row['mu_r_kpc']:.4f}"
            f"+/-{row['mu_r_sd_kpc']:.4f} kpc  rhat<={max(row['distance_r_hat_max'], row['parallax_r_hat_max']):.4f}"
            f" (latent {'sin latentes' if latent is None else format(latent, '.3f')})"
            f"  ({row['seconds']}s){flag}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# Stage C -- which King fitter.
# ---------------------------------------------------------------------------


def _convergence(trace, names):
    """PART A floor. ``az.rhat``/``az.ess``, never ``az.summary``.

    arviz >= 1 formats the summary frame for display and rounds R-hat to two
    decimals, so a genuine 1.0051 reads back as exactly 1.01 and the floor cannot
    be applied to it.
    """
    import arviz as az

    return {
        "r_hat_max": float(max(float(az.rhat(trace, var_names=[n])[n]) for n in names)),
        "ess_bulk_min": float(min(float(az.ess(trace, var_names=[n])[n]) for n in names)),
        "divergences": int(trace.sample_stats["diverging"].values.sum()),
    }


def _latent_rhat(trace):
    """Worst R-hat over the per-star latent distances, if the model has them.

    ``distance_model``'s error-aware branch **used to** add one ``r_true`` per star. Those were
    nuisance parameters -- each constrained by a single observation, so a low ESS there was expected
    and was not evidence about ``mu_r``. Reported separately so a warning from PyMC could be
    attributed rather than guessed at.

    Desde 2026-08-24 esa rama marginaliza los latentes en forma cerrada, asi que ya no existen y
    esto devuelve ``None`` en toda traza que produzca este script.

    Devolvia ``float("nan")``, y `json.dumps(..., default=float)` lo emite como el token desnudo
    ``NaN``: Python lo relee, pero eso NO es JSON — jq, `JSON.parse` y cualquier validador de
    esquema rechazan el fichero entero. El sidecar commiteado es de donde salen las cifras que cita
    el paper, asi que dejarlo invalido rompe a todo lector que no sea Python. ``None`` sale como
    ``null``, que es valido y significa lo mismo: este modelo no tiene latentes.

    Se conserva la clave, y no la justificacion que la acompañaba: decia que se guardaba porque
    "una traza guardada de antes del cambio si los trae", y no hay ninguna ruta de codigo que le
    pase una traza guardada — solo se llama sobre el `distance_trace` de la misma corrida.
    """
    import arviz as az

    if "r_true" not in trace.posterior:
        return None
    return float(np.nanmax(az.rhat(trace, var_names=["r_true"])["r_true"].values))


def stage_c(table, member_mask, *, field, draws, seed, label):
    from erotica.analysis.structure import ClusterStructureAnalyzer

    members = table[member_mask]
    radius = radii_arcmin(members)
    inside = radius <= field
    members = members[inside]
    analyzer = ClusterStructureAnalyzer(members)
    profile = analyzer.radial_density_profile(CENTRE)

    out = []
    for fitter_label, kwargs in (
        (
            "binned Gaussian, data-derived priors (RDP_bayesian)",
            dict(method="binned", return_trace=True),
        ),
        (
            "unbinned point process (king_unbinned)  [NEW DEFAULT]",
            dict(method="unbinned", field_radius=field),
        ),
    ):
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            res = analyzer.fit_king_profile(profile, sampling=sampling(draws, seed), **kwargs)
        row = {
            "sample": label,
            "field_arcmin": float(field.to_value(u.arcmin)),
            "fitter": fitter_label,
            "n_stars": int(len(members)),
            "seconds": round(time.perf_counter() - t0, 1),
        }
        for name in ("R_c", "R_t", "b"):
            median = res[f"{name}_median"]
            sd = res[f"{name}_std"]
            row[name] = float(getattr(median, "value", median))
            row[f"{name}_sd"] = float(getattr(sd, "value", sd))
        row.update(_convergence(res["king_trace"], ["R_c", "R_t"]))
        out.append(row)
        print(
            f"    {fitter_label:54s} N={row['n_stars']:4d}  "
            f"R_c={row['R_c']:.3f}+/-{row['R_c_sd']:.3f}'  "
            f"R_t={row['R_t']:.1f}+/-{row['R_t_sd']:.1f}'  "
            f"div={row['divergences']} ({row['seconds']}s)",
            flush=True,
        )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["A", "B", "C", "all"], default="all")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("-o", "--out", type=Path, default=Path(__file__).with_suffix(".json"))
    args = ap.parse_args()

    results = {"published": PUBLISHED, "draws": args.draws, "seed": args.seed}

    print("STAGE A -- the membership clip (40', the adopted production sample)")
    a40 = stage_a(FLAGS_40, "40 arcmin")
    print(f"  reproduces published counts: {a40['reproduces_published_counts']}")
    print(
        f"  median e_Plx by Gmag quartile (mas): "
        f"{[f'{v:.3f}' for v in a40['median_e_plx_by_gmag_quartile']]}"
    )
    print(
        f"  {'clip basis':56s} {'post-clip':>9s} {'p>=0.6':>7s} {'Q1':>6s} {'Q4':>6s} {'grad':>7s}"
    )
    for label, row in a40["rows"].items():
        q = row["by_gmag_quartile"]
        print(
            f"  {label:56s} {row['n_after_clip_p05']:9d} {row['n_members_p06']:7d} "
            f"{q[0]:6.1%} {q[3]:6.1%} {row['bright_to_faint_gradient']:+7.3f}"
        )
    print(
        f"  maxiters sensitivity (raw basis, post-clip count): "
        f"{a40['maxiters_sensitivity_raw_basis']}"
    )

    table40 = a40.pop("_table")
    masks = {label: row.pop("member_mask") for label, row in a40["rows"].items()}
    results["stage_a"] = a40

    published_mask = masks["published (histogram mode + std, raw parallax)"]
    normalised_mask = masks["package normalised clip (biweight, z = (plx - plx0)/err)"]

    print("\nSTAGE A2 -- P01's faint-quartile KS test under each member list")
    ks = stage_a2(table40, masks)
    results["stage_a2"] = ks
    print(
        f"  {'member list':56s} {'N':>4s} {'D vs Q1':>8s} {'D vs Q2':>8s} {'D vs Q3':>8s} {'sig':>4s}"
    )
    print(
        f"  {'PAPER (Sect. 5)':56s} {254:4d} "
        f"{PUBLISHED_KS[0]['D']:8.3f} {PUBLISHED_KS[1]['D']:8.3f} "
        f"{PUBLISHED_KS[2]['D']:8.3f} {1:4d}"
    )
    for label, res in ks.items():
        d = [c["D"] for c in res["comparisons"]]
        print(
            f"  {label:56s} {res['n_total']:4d} {d[0]:8.3f} {d[1]:8.3f} {d[2]:8.3f} "
            f"{res['n_significant_after_holm']:4d}"
        )
    for label, res in ks.items():
        p = [c["p"] for c in res["comparisons"]]
        print(f"    p ({label[:34]:34s}): " + "  ".join(f"{v:.4f}" for v in p))

    if args.stage in {"B", "all"}:
        print("\nSTAGE B -- parallax and distance, one switch at a time (40', p >= 0.6)")
        print("  on the PUBLISHED member list, so the clip is held fixed:")
        rows = stage_b(
            table40, published_mask, draws=args.draws, seed=args.seed, label="published clip"
        )
        print("  on the NORMALISED-clip member list, new defaults only:")
        rows += stage_b(
            table40, normalised_mask, draws=args.draws, seed=args.seed, label="normalised clip"
        )[-1:]
        results["stage_b"] = rows

    if args.stage in {"C", "all"}:
        print("\nSTAGE C -- which King fitter")
        print("  40' production sample, published member list:")
        rows = stage_c(
            table40,
            published_mask,
            field=FIELD_40,
            draws=args.draws,
            seed=args.seed,
            label="40' published clip",
        )
        print("  40' production sample, normalised-clip member list:")
        rows += stage_c(
            table40,
            normalised_mask,
            field=FIELD_40,
            draws=args.draws,
            seed=args.seed,
            label="40' normalised clip",
        )
        print("  70' extraction (the sample R_t was measured on):")
        a70 = stage_a(FLAGS_70, "70 arcmin")
        table70 = a70.pop("_table")
        masks70 = {label: row.pop("member_mask") for label, row in a70["rows"].items()}
        results["stage_a_70"] = a70
        rows += stage_c(
            table70,
            masks70["published (histogram mode + std, raw parallax)"],
            field=FIELD_70,
            draws=args.draws,
            seed=args.seed,
            label="70' published clip",
        )
        rows += stage_c(
            table70,
            masks70["package normalised clip (biweight, z = (plx - plx0)/err)"],
            field=FIELD_70,
            draws=args.draws,
            seed=args.seed,
            label="70' normalised clip",
        )
        results["stage_c"] = rows

    args.out.write_text(json.dumps(results, indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
