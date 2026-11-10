"""Empirical bias of ASPCAP [Fe/H] in young late-G/K (pre-main-sequence) stars.

ASPCAP (APOGEE DR17) returns [Fe/H] too low for young K stars. In the calibration sample (104 stars
in 12 clusters younger than ~10 Myr) the median of ``fe_h - anchor``, with the Pleiades + alpha Per
G-dwarf anchor at +0.021, is -0.17 dex in the 82 stars without infrared excess (Ks - [4.5] < 0.4)
and -0.45 dex in the 22 with it. This module models that deficit as

    delta = fe_h - anchor
          = a + b * (logg - 4) + c * (teff - 4800) / 500 + d * max(Ks - [4.5], 0),

where ``logg`` is **ASPCAP's** surface gravity and ``Ks - [4.5]`` the mid-infrared excess (Spitzer
IRAC [4.5]; WISE W2 where [4.5] is missing). The correction to apply is ``-delta``:
``fe_h_corrected = fe_h_aspcap - delta``.

Domain
------
T_eff 4000-5500 K, ASPCAP log g 3.0-4.8, Ks - [4.5] from -0.2 to 2 mag, ASPCAP DR17 (DR19 gives the
same values star by star). Outside it :func:`predict_bias` warns and flags the star.

What the model is not
---------------------
It is **empirical**. Its terms are not mechanisms: the log g term does not mean that surface
gravity biases the abundance. In the hub's young-K sample the infrared excess correlates with
ASPCAP's log g (rho = -0.78, N 122) and not with the physical log g from the Gaia luminosity
(+0.02), and, inside a
cluster at fixed T_eff, [Fe/H] follows ASPCAP's log g (+0.26 dex per dex in stars without excess)
and not the physical one. Whatever veils the spectrum lowers ASPCAP's log g and [Fe/H] together.

**So the regressor is the symptom, and the model must not be applied blindly.** It is valid only
with the log g that ASPCAP returned for the same spectrum; a log g from isochrones, asteroseismology
or another pipeline is a different variable and gives a wrong correction. That is why
``logg_source`` is a required argument of :func:`predict_bias`.

A second limit: the slope ``b`` is learned mostly from stars *with* a disc at low log g, where
ASPCAP's log g carries the veiling. No calibration star without excess (Ks - [4.5] < 0.4) has log g
below 3.507, so a star without excess at log g < 3.45 is an extrapolation and is flagged.

Calibration and uncertainty
---------------------------
Ordinary least squares on the 104 stars; the per-star scatter ``sigma_e`` (0.061) and the
between-cluster scatter ``tau`` (0.028) come from the residuals. The uncertainty of a prediction is

    sigma = sqrt(x^T C x + tau^2 + sigma_ref^2 + anchor_err^2),

the parameters, the between-cluster term, ``sigma_ref`` = 0.05 (the true metallicity difference
between young associations and the Pleiades / alpha Per reference) and the anchor error (0.005). It
does **not** include ``sigma_e``: averaging ``n`` stars of one cluster adds ``sigma_e / sqrt(n)``
separately. Leaving one cluster out, the per-star RMS is 0.076 dex against 0.137 for a constant.

Provenance: hub repository ``~/phd``, ``agent-findings/apogee-pms-bias-deep-2026-11-09.md`` sections
4.3 and 7 (2026-11-09), prototype ``agent-findings/scripts/apogee_pms_bias_correction_2026_11_09.py``
and its ``.json``; author decision D26-52 (2026-11-09) to move it into EROTICA.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

__all__ = [
    "ANCHOR_FE_H",
    "ANCHOR_FE_H_ERR",
    "CALIBRATED_MODEL",
    "SIGMA_REF",
    "BiasModel",
    "BiasPrediction",
    "PMSBiasDomainWarning",
    "fit_bias_model",
    "leave_one_cluster_out",
    "predict_bias",
    "select_calibration_sample",
]

#: [Fe/H] of the G-dwarf anchor (Pleiades + alpha Per, ASPCAP DR17) and its error.
ANCHOR_FE_H = 0.021
ANCHOR_FE_H_ERR = 0.005
#: True metallicity spread between young associations and the anchor clusters (hub finding
#: 2026-11-06, amendment 1).
SIGMA_REF = 0.05

#: Calibration domain. The excess range is the one of the calibration sample.
TEFF_RANGE = (4000.0, 5500.0)
LOGG_RANGE = (3.0, 4.8)
EXCESS_RANGE = (-0.2, 2.0)
#: "No excess" is Ks - [4.5] < 0.4, the split used in the hub findings
#: (b9-hd159176-fate-undecided-best-practice-2026-11-06.md sec. 4.2 and
#: scripts/apogee_pms_bias_noexcess_2026_11_09.py). Below log g 3.45 such a star is outside the
#: calibrated region (lowest calibration star without excess: log g 3.507).
NO_EXCESS_BELOW = 0.4
EXTRAPOLATION_LOGG = 3.45

_VALID_LOGG_SOURCES = ("aspcap_dr17", "aspcap_dr19")
_MIN_STARS_PER_CLUSTER = 3


class PMSBiasDomainWarning(UserWarning):
    """A star lies outside the calibration domain of the PMS [Fe/H] bias model."""


@dataclass(frozen=True)
class BiasModel:
    """Calibrated coefficients of the bias model.

    ``beta`` = (a, b, c, d) and ``cov`` its covariance, scaled by ``sigma_e**2 + tau**2``.
    """

    beta: np.ndarray
    cov: np.ndarray
    sigma_e: float
    tau: float
    n_stars: int
    n_clusters: int
    anchor: float = ANCHOR_FE_H


class BiasPrediction(NamedTuple):
    """Output of :func:`predict_bias`. ``delta`` is what ASPCAP under-reports; subtract it."""

    delta: np.ndarray
    sigma: np.ndarray
    extrapolated: np.ndarray
    out_of_domain: np.ndarray


# Embedded calibration: full-precision copy of the hub file
# ~/phd/agent-findings/scripts/apogee_pms_bias_correction_2026_11_09.json (2026-11-09, hub commit
# 23fc4bc), fitted on the 104 stars of tests/data/apogee_dr17_young_k_pms_bias.ecsv.
# tests/test_pms_bias.py refits them and fails if they drift.
CALIBRATED_MODEL = BiasModel(
    beta=np.array(
        [
            -0.18090877702474475,
            0.18251503390151047,
            0.04068476651176977,
            -0.12350123453595746,
        ]
    ),
    cov=np.array(
        [
            [
                0.0001706696555834176,
                -8.403624737518603e-05,
                8.213871069358338e-05,
                -0.00013085943783651473,
            ],
            [
                -8.403624737518603e-05,
                0.0008089247183066067,
                0.0002210640102320234,
                0.0005582051726782148,
            ],
            [
                8.213871069358338e-05,
                0.0002210640102320234,
                0.00017677022206905163,
                0.00016059789310413815,
            ],
            [
                -0.0001308594378365147,
                0.0005582051726782148,
                0.00016059789310413818,
                0.0006523193022818866,
            ],
        ]
    ),
    sigma_e=0.06050772077911833,
    tau=0.028302879141487953,
    n_stars=104,
    n_clusters=12,
)


def _design(teff, logg, excess):
    teff, logg, excess = np.broadcast_arrays(
        np.asarray(teff, float), np.asarray(logg, float), np.asarray(excess, float)
    )
    return np.stack(
        [np.ones_like(teff), logg - 4.0, (teff - 4800.0) / 500.0, np.clip(excess, 0.0, None)],
        axis=-1,
    )


def _columns(table):
    def col(name):
        return np.asarray(table[name], float)

    return (
        col("teff"),
        col("logg"),
        col("ks_minus_45"),
        col("fe_h"),
        np.asarray(table["Name"]).astype(str),
    )


def select_calibration_sample(table):
    """Boolean mask of the calibration selection.

    Finite ``ks_minus_45`` and ``fe_h``, ``logg >= 3.0``, and clusters (column ``Name``) keeping at
    least three such stars. On the vendored table this gives 104 stars in 12 clusters.
    """
    teff, logg, excess, fe_h, name = _columns(table)
    ok = np.isfinite(excess) & np.isfinite(fe_h) & (logg >= LOGG_RANGE[0])
    big = [c for c in np.unique(name[ok]) if (ok & (name == c)).sum() >= _MIN_STARS_PER_CLUSTER]
    return ok & np.isin(name, big)


def _fit(teff, logg, excess, y, name):
    X = _design(teff, logg, excess)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    clusters = np.unique(name)
    means = np.array([resid[name == c].mean() for c in clusters])
    counts = np.array([(name == c).sum() for c in clusters])
    sigma_e = np.sqrt(
        np.mean(
            [
                np.var(resid[name == c], ddof=1)
                for c, n in zip(clusters, counts, strict=True)
                if n > 1
            ]
        )
    )
    tau2 = max(0.0, np.var(means, ddof=1) - np.mean(sigma_e**2 / counts))
    cov = np.linalg.inv(X.T @ X) * (sigma_e**2 + tau2)
    return beta, cov, float(sigma_e), float(np.sqrt(tau2)), clusters.size


def fit_bias_model(table, *, anchor=ANCHOR_FE_H, select=True):
    """Fit the bias model to a table with columns Name, teff, logg, fe_h, ks_minus_45.

    Parameters
    ----------
    table : astropy Table, pandas DataFrame or mapping of columns
        ``logg`` and ``fe_h`` must be ASPCAP's, from the same spectrum.
    anchor : float
        [Fe/H] of the reference; the fitted quantity is ``fe_h - anchor``.
    select : bool
        Apply :func:`select_calibration_sample` first.
    """
    teff, logg, excess, fe_h, name = _columns(table)
    if select:
        keep = select_calibration_sample(table)
        teff, logg, excess, fe_h, name = (v[keep] for v in (teff, logg, excess, fe_h, name))
    beta, cov, sigma_e, tau, n_clusters = _fit(teff, logg, excess, fe_h - anchor, name)
    return BiasModel(beta, cov, sigma_e, tau, int(fe_h.size), int(n_clusters), float(anchor))


def leave_one_cluster_out(table, *, anchor=ANCHOR_FE_H, select=True):
    """Validate by leaving one cluster out, against a constant (the training median).

    Returns a dict with ``rms_star`` and ``mean_bias`` (per-star prediction errors),
    ``rms_cluster`` (RMS of the per-cluster mean error), ``baseline_rms_star`` (same split,
    predictor = median of the training ``fe_h - anchor``), ``n_stars`` and ``n_clusters``.
    """
    teff, logg, excess, fe_h, name = _columns(table)
    if select:
        keep = select_calibration_sample(table)
        teff, logg, excess, fe_h, name = (v[keep] for v in (teff, logg, excess, fe_h, name))
    y = fe_h - anchor
    errs, cluster_errs, base = [], [], []
    for c in np.unique(name):
        out = name == c
        beta, *_ = _fit(teff[~out], logg[~out], excess[~out], y[~out], name[~out])
        e = y[out] - _design(teff[out], logg[out], excess[out]) @ beta
        errs.append(e)
        cluster_errs.append(e.mean())
        base.append(y[out] - np.median(y[~out]))
    errs, base, cluster_errs = np.concatenate(errs), np.concatenate(base), np.array(cluster_errs)
    return {
        "rms_star": float(np.sqrt(np.mean(errs**2))),
        "mean_bias": float(errs.mean()),
        "rms_cluster": float(np.sqrt(np.mean(cluster_errs**2))),
        "baseline_rms_star": float(np.sqrt(np.mean(base**2))),
        "n_stars": int(y.size),
        "n_clusters": int(cluster_errs.size),
    }


def predict_bias(
    teff,
    logg,
    ks_minus_45,
    *,
    logg_source,
    model=None,
    sigma_ref=SIGMA_REF,
    anchor_err=ANCHOR_FE_H_ERR,
):
    """ASPCAP [Fe/H] deficit ``delta`` of a young K star, and its uncertainty.

    The corrected metallicity is ``fe_h_aspcap - delta``. Vectorized: the three inputs broadcast.

    **Do not use blindly.** The regressor is ASPCAP's log g, which is itself the symptom: the
    veiling that lowers [Fe/H] lowers ASPCAP's log g with it, so the model is empirical and its
    terms are not mechanisms. It holds only for the log g that ASPCAP DR17 (or DR19, which agrees
    star by star) returned for the same spectrum; ``logg_source`` must say so.

    Parameters
    ----------
    teff, logg : array_like
        ASPCAP T_eff (K) and log g (cgs). Domain 4000-5500 K and 3.0-4.8.
    ks_minus_45 : array_like
        Ks - [4.5] (IRAC; W2 if [4.5] is missing), mag. Negative values count as no excess.
    logg_source : {"aspcap_dr17", "aspcap_dr19"}
        Required. Anything else raises ``ValueError``.
    model : BiasModel, optional
        Defaults to :data:`CALIBRATED_MODEL`.
    sigma_ref, anchor_err : float
        Reference-metallicity spread and anchor error added in quadrature.

    Returns
    -------
    BiasPrediction
        ``delta``, ``sigma`` (parameters ⊕ tau ⊕ sigma_ref ⊕ anchor_err; no per-star scatter),
        ``extrapolated`` (no excess, Ks - [4.5] < 0.4, and log g < 3.45: no calibration star
        supports it) and ``out_of_domain`` (outside the ranges above, or a missing input; a
        :class:`PMSBiasDomainWarning` is emitted).
    """
    if logg_source not in _VALID_LOGG_SOURCES:
        raise ValueError(
            f"logg_source={logg_source!r}: the PMS [Fe/H] bias model is calibrated on ASPCAP's own "
            f"log g, which carries the bias; use one of {_VALID_LOGG_SOURCES} or do not apply it."
        )
    m = CALIBRATED_MODEL if model is None else model
    teff, logg, excess = np.broadcast_arrays(
        np.asarray(teff, float), np.asarray(logg, float), np.asarray(ks_minus_45, float)
    )
    X = _design(teff, logg, excess)
    delta = X @ m.beta
    var_par = np.einsum("...i,ij,...j->...", X, m.cov, X)
    sigma = np.sqrt(var_par + m.tau**2 + sigma_ref**2 + anchor_err**2)

    inside = (
        (teff >= TEFF_RANGE[0])
        & (teff <= TEFF_RANGE[1])
        & (logg >= LOGG_RANGE[0])
        & (logg <= LOGG_RANGE[1])
        & (excess >= EXCESS_RANGE[0])
        & (excess <= EXCESS_RANGE[1])
    )
    out_of_domain = ~inside
    if np.any(out_of_domain):
        warnings.warn(
            f"{int(np.sum(out_of_domain))} star(s) outside the calibration domain "
            f"(T_eff {TEFF_RANGE} K, ASPCAP log g {LOGG_RANGE}, Ks - [4.5] {EXCESS_RANGE}) "
            "or with a missing input; the PMS [Fe/H] bias there is an extrapolation.",
            PMSBiasDomainWarning,
            stacklevel=2,
        )
    extrapolated = (excess < NO_EXCESS_BELOW) & (logg < EXTRAPOLATION_LOGG)
    return BiasPrediction(delta, sigma, extrapolated, out_of_domain)
