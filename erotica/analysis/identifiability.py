r"""Is this parameter measured, or is the prior answering for it?

WHY THIS MODULE EXISTS
----------------------
This programme has repeatedly reported a number that the data did not determine, and each time it was
found late and by accident:

* ``R_t`` for NGC 6383 has a posterior SD of ~7000′ under a scale-free prior. What is normally quoted
  is a **prior**, not a measurement.
* The King background ``b`` absorbs 56% of a membership-selected sample while the measured
  false-discovery proportion is 6.1% — the parameter is real but it is not measuring contamination.
* The EFF slope ``gamma`` recovers a true 2.00 as **3.6** at the footprint-to-scale ratio where most
  of the Gaia census sits, and no bias correction repairs it, because the problem is identifiability
  rather than bias.

The tools that would have caught all three exist and are standard. This module packages them so the
question *"is this parameter measured?"* is asked **before** a number is quoted, not after a referee
asks.

THE FOUR ANGLES, AND WHY ONE IS NOT ENOUGH
------------------------------------------
Each answers a **different question**, and a parameter can pass one while failing another. The
questions, and the function that answers each:

======  ================================  ==============================  ==========================
angle   the question it answers           what a failure means            function
======  ================================  ==============================  ==========================
1       *Do the data determine this, or   the number quoted is a prior     :func:`prior_sensitivity`
        is the prior answering?*          restated, not a measurement
2       *Is it separately identified,     a prior on one parameter         :func:`posterior_geometry`
        or locked to another             silently determines the other
        parameter?*
3       *Is it precise — and is that      narrow because the prior is      :func:`posterior_geometry`
        precision from the data?*         narrow, not because the data     (``relative_width``)
                                          are informative
4       *Does the geometry admit          the object should be excluded,   :func:`munoz_criteria`
        measuring it at all?*             not corrected
======  ================================  ==============================  ==========================

:func:`identifiability_report` runs all four and returns a per-parameter ``verdict`` that is
deliberately conservative: **measured** only if angle 1 clears it *and* angle 2 does not find it
locked. Angles 3 and 4 are reported for judgement rather than folded into the verdict, because a wide
interval is not by itself a defect and a geometric criterion can be ill-posed (see
:func:`munoz_criteria`).

**Why all four.** In the case that motivated this module, angle 1 alone gives the right answer and
angle 3 alone gives the wrong one: reparameterising the EFF fit onto the identified combination
:math:`\kappa = \gamma/a^2` produces an *unbiased* parameter with a **213%** relative width, while the
biased ``gamma`` has 43%. Judging by precision would have preferred the wrong quantity. Conversely
angle 2 alone misses the case where a single parameter is prior-dominated without being correlated
with anything.

**1. Power-scaling prior sensitivity** — Kallioinen, Paananen, Bürkner & Vehtari (2024, *Statistics
and Computing* 34, 57, `arXiv:2107.14054`). Perturbs the prior and the likelihood by small powers
(:math:`\alpha = 0.99, 1.01`) and measures how far the posterior moves, as a
cumulative-Jensen-Shannon distance. Their own summary of the decisive pattern:

    *"The presence of prior sensitivity but relatively low (or no) likelihood sensitivity is an
    indication that the likelihood is weakly informative (or noninformative) in relation to the
    prior."*

Threshold 0.05. This is the **direct** test, and it is the one that settled the EFF case: at a
footprint-to-scale ratio of 2 the diagnosis is *strong prior / weak likelihood*
(prior 0.194, likelihood 0.027), while at ratio 42 it is clean (prior 0.011, likelihood 0.194).

**2. Posterior geometry** — the correlation matrix and its condition number. A high correlation says
the data constrain a *combination* far better than either parameter alone, so the marginal
uncertainties are not the whole story and a prior on one silently determines the other. Measured on
NGC 6383: EFF condition number **87.8** against King's 17.8, with ``corr(a, gamma) = +0.871``.

**3. Relative width** — the precision question, and the one that exposes a trap. A parameter can be
*unbiased* and useless: reparameterising the EFF fit onto :math:`\kappa = \gamma/a^2`, which the
algebra says is the identified combination for :math:`R \ll a`, gives an unbiased ``kappa`` with a
**212%** relative width where the biased ``gamma`` has 43%. Narrowness is not knowledge — where the
likelihood is weak, a narrow interval is narrow *because the prior is*.

**4. Geometric admissibility** — Muñoz, Padmanabhan & Geha (2012, ApJ 745, 127,
`2012ApJ...745..127M`), the published precedent for refusing to quote a parameter:

    *"to recover structural parameters within 10% or better of their true values: (1) the ratio of the
    field of view to the half-light radius must be greater than three, (2) the total number of stars,
    including background objects, should be larger than 1000, and (3) the central to background
    stellar density ratio must be higher than 20."*

Criterion (1) **is** the footprint-to-scale ratio this repository re-derived empirically. Cite Muñoz
rather than presenting it as new.

WHAT TO DO WHEN A PARAMETER FAILS
---------------------------------
Not a bias correction. Dufour (1997, *Econometrica* 65, 1365) proves that any valid confidence set for
a locally-almost-unidentified parameter must be **unbounded with positive probability**, so an
almost-surely-bounded (Wald-type) interval has **zero** coverage. The escape hatch is that
likelihood-ratio and profile procedures stay valid *and are allowed to be unbounded*. So:

* report the parameter as **not identified over this footprint** and give the profile interval,
  unbounded if that is what it is;
* or report the identified combination instead, stating plainly that it is a different quantity;
* or exclude the object, which is what a recoverability floor is for.

And note what does **not** help: the Cox-Snell/Firth/Kosmidis bias-reduction family all correct an
:math:`O(1/n)` term, and for the EFF slope that term is measured to be *rejected* — they fix
something that is not the problem. (Verified separately: an ADS full-text search for ``"Cox-Snell"``
restricted to astronomy returns 11 hits, none of which is a bias correction.)

.. note::
   ``pm.Potential`` contributes to the model log density but is **not an observed variable**, so PyMC
   produces no ``log_likelihood`` group for point-process models built that way, and every ArviZ
   diagnostic needing one fails. This is documented behaviour, not a bug -- ``compute_log_likelihood``
   defaults to "all observed variables", and potentials are a separate category summed into ``logp``.
   :func:`attach_log_likelihood` builds the group instead. The per-star split is exact: by the
   conditioning property of a Poisson process, given ``N`` points in a window they are iid draws from
   the window-normalised intensity, so attributing :math:`\Lambda/N` to each star reproduces the total.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "attach_log_likelihood",
    "prior_sensitivity",
    "posterior_geometry",
    "munoz_criteria",
    "identifiability_report",
]

PSENSE_THRESHOLD = 0.05  # Kallioinen et al. (2024)
CORRELATION_WARN = 0.90  # above this, two parameters are not separately identified
MUNOZ_FOV_OVER_RHALF = 3.0
MUNOZ_MIN_STARS = 1000
MUNOZ_MIN_CONTRAST = 20.0


def attach_log_likelihood(idata, radii, field_radius, *, model):
    r"""Add the ``log_likelihood`` group a ``Potential``-based point-process fit does not produce.

    Parameters
    ----------
    idata : DataTree
        Posterior from :func:`~erotica.analysis.structure.king_unbinned` or
        :func:`~erotica.analysis.structure.eff_unbinned` (the ``*_trace`` entry of the returned dict).
    radii : array-like
        The radii that were fitted, in arcmin.
    field_radius : float
        Same footprint used in the fit, in arcmin.
    model : {"king", "eff"}

    Returns
    -------
    DataTree
        The same object, with ``log_likelihood`` attached. Modified in place and returned for chaining.
    """
    import xarray as xr

    from .structure import eff_expected_count, king_expected_count

    r = np.asarray(radii, dtype=float)
    post = idata.posterior

    def draws(name):
        return np.asarray(post[name].values)[..., None]

    k, b = draws("k"), draws("b")
    if model == "king":
        R_c, R_t = draws("R_c"), draws("R_t")
        core = 1.0 / np.sqrt(1.0 + (r / R_c) ** 2)
        edge = 1.0 / np.sqrt(1.0 + (R_t / R_c) ** 2)
        sigma = np.where(r <= R_t, k * (core - edge) ** 2 + b, b)
        expected = king_expected_count(k[..., 0], b[..., 0], R_c[..., 0], R_t[..., 0], field_radius)
    elif model == "eff":
        a, gamma = draws("a"), draws("gamma")
        sigma = k * (1.0 + (r / a) ** 2) ** (-gamma / 2.0) + b
        expected = eff_expected_count(k[..., 0], b[..., 0], a[..., 0], gamma[..., 0], field_radius)
    else:
        raise ValueError(f"model must be 'king' or 'eff', got {model!r}")

    per_star = np.log(2.0 * np.pi * r * sigma) - expected[..., None] / r.size
    idata["log_likelihood"] = xr.DataTree(
        xr.Dataset(
            {"point_process": (("chain", "draw", "star"), per_star)},
            coords={"chain": post.chain, "draw": post.draw, "star": np.arange(r.size)},
        )
    )
    return idata


def prior_sensitivity(idata, var_names=None, threshold=PSENSE_THRESHOLD):
    """Power-scaling sensitivity (Kallioinen et al. 2024). Requires ``log_likelihood`` and ``log_prior``.

    Returns the ArviZ summary frame, whose ``diagnosis`` column carries the verdict string --
    ``"potential strong prior / weak likelihood"`` is the one that means *not measured*.
    """
    from pymc.stats import psense_summary

    return psense_summary(idata, var_names=var_names, threshold=threshold)


def posterior_geometry(idata, var_names):
    """Correlation matrix, ranked pairs and condition number of the posterior.

    The condition number is the scale-free summary of how close the fit is to rank-deficient; the
    ranked pairs say which parameters are trading against each other.
    """
    post = idata.posterior
    stacked = np.column_stack([np.asarray(post[n].values).ravel() for n in var_names])
    corr = np.corrcoef(stacked, rowvar=False)
    pairs = sorted(
        (
            dict(a=var_names[i], b=var_names[j], r=float(corr[i, j]))
            for i in range(len(var_names))
            for j in range(i + 1, len(var_names))
        ),
        key=lambda p: -abs(p["r"]),
    )
    return dict(
        parameters=list(var_names),
        correlation=corr.tolist(),
        pairs=pairs,
        condition_number=float(np.linalg.cond(corr)),
        not_separately_identified=[p for p in pairs if abs(p["r"]) > CORRELATION_WARN],
        relative_width={
            n: float(np.std(np.asarray(post[n].values)) / abs(np.median(np.asarray(post[n].values))))
            for n in var_names
        },
    )


def munoz_criteria(*, field_radius, half_number_radius, n_stars, central_to_background):
    """The three Muñoz, Padmanabhan & Geha (2012) recoverability criteria, scored.

    Returns each ratio, its threshold and whether it passes. **Criterion 1 is the
    footprint-to-scale ratio**, so this is the published precedent for a geometric floor -- not a
    novel criterion.

    .. warning::
       Criterion 1 is **ill-posed** when the half-number radius is set by where the survey stopped
       rather than by an edge of the object, which is the normal situation for a Gaia-era cluster with
       a corona. Read a failure there as "inapplicable", not as a defect of the cluster. Criteria 2
       and 3 are also **not independent** on an astrometrically pre-filtered sample: filtering removes
       stars, so ``N`` falls while the contrast rises, and scoring them separately double-counts.
    """
    checks = {
        "fov_over_half_radius": (field_radius / half_number_radius, MUNOZ_FOV_OVER_RHALF),
        "total_stars": (float(n_stars), MUNOZ_MIN_STARS),
        "central_to_background": (float(central_to_background), MUNOZ_MIN_CONTRAST),
    }
    return {
        name: dict(value=value, threshold=threshold, passes=bool(value > threshold))
        for name, (value, threshold) in checks.items()
    }


def identifiability_report(idata, var_names, *, radii=None, field_radius=None, model=None):
    """All four angles at once. Attaches ``log_likelihood`` first if it is missing and can be built.

    The ``verdict`` per parameter is deliberately conservative: a parameter is reported as measured
    only if power-scaling clears it **and** it is not locked to another parameter above
    ``CORRELATION_WARN``.
    """
    report = {"geometry": posterior_geometry(idata, var_names)}

    if "log_likelihood" not in [g.strip("/") for g in idata.groups]:
        if radii is None or field_radius is None or model is None:
            report["prior_sensitivity"] = "unavailable: pass radii, field_radius and model"
            return report
        attach_log_likelihood(idata, radii, field_radius, model=model)
    try:
        frame = prior_sensitivity(idata, var_names=var_names)
        report["prior_sensitivity"] = frame.to_dict()
        diagnoses = frame["diagnosis"].to_dict()
    except Exception as exc:  # log_prior missing is the usual cause
        report["prior_sensitivity"] = f"unavailable: {type(exc).__name__}: {exc}"
        diagnoses = {}

    locked = {p["a"] for p in report["geometry"]["not_separately_identified"]} | {
        p["b"] for p in report["geometry"]["not_separately_identified"]
    }
    report["verdict"] = {
        name: (
            "NOT MEASURED — prior-dominated"
            if "strong prior" in str(diagnoses.get(name, ""))
            else "NOT SEPARATELY IDENTIFIED — locked to another parameter"
            if name in locked
            else "measured"
            if diagnoses
            else "undetermined — no prior-sensitivity result"
        )
        for name in var_names
    }
    return report
