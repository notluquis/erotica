r"""Sigma-clipping utilities for EROTICA cluster analysis.

The default clip basis changed on 2026-08-02 -- see :func:`sigma_clip_parallax`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from astropy import units as u
from astropy.stats import biweight_location, biweight_scale, mad_std, sigma_clip
from astropy.table import QTable

from .units import quantity_values

#: Astropy's own default is ``maxiters=5``, and it is applied **per call** --
#: the clip re-estimates centre and scale and re-clips up to five times, so the
#: bounds tighten monotonically with the iteration count. It was left implicit
#: until 2026-08-02, which meant the published NGC 6383 selection depended on an
#: upstream default nobody had written down. Measured on the 412 pre-clip
#: members of ``paperfaithful_with_clip_flags.ecsv`` (40', biweight/biweight,
#: ``sigma = 2``), the surviving count is:
#:
#: ========  ===  ===  ===  ===  ===  ===========
#: maxiters    1    2    3    5   10  (converged)
#: kept      396  385  377  369  358          358
#: ========  ===  ===  ===  ===  ===  ===========
#:
#: i.e. **38 stars, 9.2% of the sample**, separate ``maxiters=1`` from
#: convergence. It is pinned at 5 rather than 1 or ``None`` so that making it
#: explicit does not itself move a number: 5 is what astropy applied, so 5 is
#: what the published selection used.
DEFAULT_MAXITERS = 5


def _mas_values(values) -> np.ndarray:
    """Return a column of parallaxes or parallax errors as float mas.

    A column that **carries** a unit is converted, and therefore validated:
    handing this a length where an angle is meant raises
    :class:`~astropy.units.UnitConversionError` rather than silently returning
    numbers in the wrong scale. A unitless column is taken as mas, which is the
    same convention a bare :class:`numpy.ndarray` follows everywhere else in this
    package (see the units policy in ``erotica/analysis/CLAUDE.md``).
    """
    if getattr(values, "unit", None) is None:
        return np.asarray(getattr(values, "value", values), dtype=float)
    return np.asarray(quantity_values(values, u.mas), dtype=float)


def _finite_positive_errors(values, name: str) -> np.ndarray:
    errors = _mas_values(values)
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError(f"{name} must be finite and positive for every row.")
    return errors


def _excess_variance(residual: np.ndarray, variance: np.ndarray) -> float:
    r"""Maximum-likelihood **excess variance** under heteroscedastic noise.

    Solves for :math:`\sigma_{\mathrm{exc}}^2` in
    :math:`\varpi_i \sim \mathcal{N}(\varpi_0,\ \sigma_{\mathrm{exc}}^2 +
    \sigma_{\varpi,i}^2)` -- the *same* error model the package's parallax
    likelihood uses (:func:`erotica.analysis.inference.fit_parallax_model`), so
    the clip and the fit agree on what a parallax uncertainty means.

    .. important::
       **This is not a measurement of the cluster's line-of-sight depth, and must
       not be reported as one.** It is whatever the quoted per-star errors do not
       account for, and on NGC 6383 it comes back at 0.044 mas -- 4.8% of the
       0.926 mas mean parallax, i.e. an implied ~52 pc, against the paper's own
       ``R_t`` of 17.4 pc and against :class:`~erotica.analysis.inference.DistancePriors`,
       which states that 50 pc is "larger than any bound cluster". At least three
       things are folded into it: real depth; **underestimated Gaia uncertainties
       at the bright end** (measured on the pre-clip sample, the brightest
       magnitude quartile has an observed robust scatter of 0.0469 mas against a
       0.0295 mas median quoted error, a ratio of 1.59); and residual
       contamination. It is used here as a **total-scatter term that makes the cut
       calibrated**, which is what a clip needs, and it is deliberately named for
       what it is rather than for what one would like it to be -- the same
       discipline this project applies to the pseudo-probability.

    The score

    .. math:: \sum_i \frac{d_i^2 - (\sigma^2 + \sigma_i^2)}{(\sigma^2 + \sigma_i^2)^2}

    is strictly decreasing in :math:`\sigma^2`, so it is bracketed and bisected
    rather than handed to an optimiser: no dependency, no starting point, and a
    deterministic answer.

    .. note::
       A **moment** estimator -- observed robust variance minus the *median*
       squared error -- is not equivalent and fails here. On the NGC 6383
       pre-clip sample it returns exactly zero, because the global scatter is set
       by the faint stars whose parallaxes are truncated by the survey query
       window, while the maximum-likelihood value (0.044 mas) is set by the
       bright stars that actually constrain it.
    """
    def score(s2: float) -> float:
        total = s2 + variance
        return float(np.sum((residual**2 - total) / total**2))

    if score(0.0) <= 0.0:
        return 0.0
    hi = float(np.max(residual**2)) or 1.0
    for _ in range(60):
        if score(hi) <= 0.0:
            break
        hi *= 2.0
    lo = 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-14 * max(hi, 1.0):
            break
    return 0.5 * (lo + hi)


def _normalised_residual(parallax, errors, selection, *, sigma, maxiters, cenfunc):
    r"""Iterate the normalised residual and return it for every row.

    .. math:: z_i = \frac{\varpi_i - \varpi_0}
                        {\sqrt{\sigma_{\mathrm{exc}}^2 + \sigma_{\varpi,i}^2}}

    Three choices here are load-bearing, and each was made after measuring the
    alternative on the real NGC 6383 sample.

    **The denominator carries the excess variance**, not the quoted errors alone
    -- see :func:`_excess_variance` for what that term does and does not mean. With
    :math:`\sigma_{\varpi,i}` alone the residual is not a standard normal for a
    cluster of finite depth: the best-measured stars, whose errors are smaller
    than the depth, get inflated :math:`|z|` and are clipped preferentially.
    Measured on the 412 pre-clip members, that variant retains **50.5%** of the
    brightest magnitude quartile against 100% of the faintest -- a gradient of
    **-0.495**, i.e. it does not remove the magnitude dependence of the raw clip,
    it inverts and enlarges it. Including
    :math:`\sigma_{\mathrm{exc}} = 0.044` mas brings the gradient to **-0.068**.

    **The threshold is fixed at** ``sigma``, **not re-estimated from the sample.**
    Normalising already fixes the scale of :math:`z` at 1; estimating it again
    from the data and multiplying is double-counting. On the same sample the
    re-estimated scale is 0.71, so a nominal 2-sigma cut would act at 1.4 sigma
    and discard 56 extra stars.

    **The dispersion is fitted on the whole selection, not on the survivors.**
    Re-fitting a variance inside its own 2-sigma cut is the classic sigma-clip
    shrinkage, and here it does not merely bias the estimate, it destroys it:
    iterating :math:`\sigma_{\mathrm{exc}}` on the survivors drives it
    0.0443 -> 0.0241 -> **0.0000** mas over three passes, and with it the
    bright-quartile retention 93.2% -> 86.4% -> 82.5% and the gradient
    -0.068 -> -0.175. Fitted on the full selection it is stable at 0.043 mas and
    the iteration converges in a single pass. The centre *is* iterated on the
    survivors, because a centre is what an outlier drags and a 2-sigma cut is an
    adequate guard for it.

    Consequently ``maxiters`` changes nothing here in practice -- the selection is
    identical for 1, 2, 5 and 20 iterations on the NGC 6383 sample -- which is the
    intended behaviour, not an accident, and is pinned by
    ``tests/test_clipping.py::test_the_normalised_clip_does_not_shrink_its_own_dispersion``.
    """
    parallax = np.asarray(parallax, dtype=float)
    errors = np.asarray(errors, dtype=float)
    selection = np.asarray(selection, dtype=bool)
    keep = selection.copy()
    variance_all = errors**2
    centre = float(cenfunc(parallax[selection]))
    intrinsic2 = _excess_variance(parallax[selection] - centre, variance_all[selection])
    z_all = np.zeros_like(parallax)

    for _ in range(200 if maxiters is None else max(int(maxiters), 1)):
        weight = 1.0 / (intrinsic2 + variance_all[keep])
        centre = float(np.sum(weight * parallax[keep]) / np.sum(weight))
        # Refit on the FULL selection about the updated centre -- never on `keep`.
        intrinsic2 = _excess_variance(
            parallax[selection] - centre, variance_all[selection]
        )
        z_all = (parallax - centre) / np.sqrt(intrinsic2 + variance_all)
        proposed = selection & (np.abs(z_all) <= sigma)
        if np.array_equal(proposed, keep):
            break
        if np.count_nonzero(proposed) < 3:
            break  # too few survivors to re-centre; keep the current iterate
        keep = proposed

    return z_all, centre, float(np.sqrt(intrinsic2))


def sigma_clip_parallax(
    table: QTable,
    *,
    cluster: int,
    sigma: float,
    use_biweight: bool,
    in_place: bool,
    mark_label: int,
    print_results: bool,
    return_mask: bool,
    preselector_mask: Sequence[bool] | None,
    method: str = "normalised",
    parallax_error_column: str = "parallax_error",
    maxiters: int | None = DEFAULT_MAXITERS,
):
    r"""Apply robust sigma clipping to the parallaxes of one cluster.

    Parameters
    ----------
    method : {"normalised", "raw"}, default ``"normalised"``
        What the clip is applied *to*.

        ``"normalised"`` clips the **normalised residual**

        .. math:: z_i = \frac{\varpi_i - \varpi_0}
                             {\sqrt{\sigma_{\mathrm{exc}}^2 + \sigma_{\varpi,i}^2}} ,
                  \qquad |z_i| \le \texttt{sigma},

        with :math:`\sigma_{\varpi,i}` each star's own uncertainty from
        `parallax_error_column` and :math:`\sigma_{\mathrm{exc}}` an **excess
        variance** term -- everything the quoted errors do not account for, not a
        depth measurement (see :func:`_excess_variance`) -- estimated by maximum
        likelihood alongside :math:`\varpi_0`. A star is an outlier
        only if it is discrepant **relative to its own precision**, so a large
        uncertainty buys tolerance rather than a rejection -- the same principle,
        and the same total :math:`\sqrt{\sigma^2 + \sigma_{\varpi,i}^2}`, as
        :func:`erotica.analysis.inference.fit_parallax_model`.

        ``"raw"`` clips the parallax column itself, which is what this function
        did before 2026-08-02.
    parallax_error_column : str, default ``"parallax_error"``
        Column holding :math:`\sigma_{\varpi,i}` in mas. Read only when
        ``method="normalised"``.
    use_biweight : bool
        Selects the robust scale (biweight, or MAD when False) for the ``"raw"``
        basis. On the ``"normalised"`` basis it sets only the **starting** centre:
        the scale of :math:`z` is 1 by construction and is asserted rather than
        re-estimated, and the centre is refined by inverse-variance weighting.
    maxiters : int or None, default 5
        Iterations. On the ``"raw"`` basis this is passed to
        :func:`astropy.stats.sigma_clip`; on the ``"normalised"`` basis it is the
        number of times :math:`(\varpi_0, \sigma_{\mathrm{int}})` are re-estimated
        on the survivors. It is **not** a free parameter: see
        :data:`DEFAULT_MAXITERS` for the measured 9.2% swing between
        ``maxiters=1`` and convergence.

    Returns
    -------
    lower, upper : float
        The clip bounds. **Their meaning depends on `method`**: bounds on the
        parallax in mas for ``"raw"``; bounds on the dimensionless :math:`z` for
        ``"normalised"``, where no single parallax interval exists because the
        admissible interval is per-star.

    Notes
    -----
    The remaining return values are flag-dependent -- an API wart left unchanged
    and pinned by ``tests/test_clipping.py::test_return_arity_is_flag_dependent``.

    **Why the default moved, 2026-08-02.**

    Gaia's per-star parallax uncertainty rises steeply with magnitude: across the
    published NGC 6383 member list the median ``e_Plx`` per ``Gmag`` quartile is
    **0.027, 0.065, 0.123, 0.293 mas**, an 11x spread. A clip on the *raw*
    parallax at a fixed multiple of the *sample* scatter therefore removes faint
    stars preferentially even when every one of them is a genuine member: it is a
    magnitude-dependent **selection function**, not only an outlier test.

    Measured in ``tools/validation/parallax_clip_selection_function.py`` against
    an oracle that is true by construction -- synthetic clusters in which every
    star is a real member carrying the *real* per-star ``e_Plx``, so every
    rejection is a false rejection -- over 400 realizations at ``sigma = 2``:

    ==================================  =======  =========  ======  ======  ========
    clip basis                          overall  Q1 bright      Q2      Q3  Q4 faint
    ==================================  =======  =========  ======  ======  ========
    raw parallax (old default)            67.5%      99.1%   85.6%   59.1%     26.7%
    ``z = d/err`` (**not** the shipped)    95.7%      95.6%   95.7%   95.8%     95.8%
    ==================================  =======  =========  ======  ======  ========

    Bright-to-faint retention gradient there: **+0.724 raw, -0.002 normalised**.
    The 95.7% is simply what a 2-sigma cut should retain. **The second row is the
    errors-only residual, which is not what ships** -- in that simulation the two
    coincide, because it injects no intrinsic scatter, so the excess variance is
    zero by construction. See the warning below for what happens when it is not.

    None of the seven Gaia open-cluster catalogues surveyed clips the raw
    parallax at 2 sigma: Cantat-Gaudin et al. (2018) use 3 MAD, Cordoni et al.
    (2023) 3 sigma, and Hunt & Reffert (2023, 2024) apply no parallax clip at all.

    .. warning::
       **On the real catalogue the gain is much smaller than that table, and a
       naive normalised residual is worse than the raw clip.** The synthetic
       oracle draws each star with the same error the residual then divides by
       and applies no survey query window, so :math:`z` is exactly
       :math:`\mathcal{N}(0,1)` and the flat row is close to tautological.
       Measured on the 412 pre-clip NGC 6383 members, with the package's own
       estimators throughout (``tools/validation/wiring_fix_deltas.py``):

       ==================================================  ======  ======  ========
       clip basis                                              Q1      Q4  gradient
       ==================================================  ======  ======  ========
       published (histogram mode + std, raw parallax)       94.2%   72.8%    +0.214
       raw parallax, package estimators                     96.1%   81.6%    +0.146
       :math:`z = d/\sigma_{\varpi,i}`, scale re-fitted      50.5%  100.0%    -0.495
       :math:`z = d/\sigma_{\varpi,i}`, fixed 2-sigma        81.6%  100.0%    -0.184
       **shipped: total sigma, fixed 2-sigma**               93.2%  100.0%    **-0.068**
       ==================================================  ======  ======  ========

       Two effects the synthetic experiment cannot show drive this. The cluster
       has a real depth, :math:`\sigma_{\mathrm{int}} = 0.044` mas by maximum
       likelihood, which exceeds the 0.030 mas median error of the brightest
       quartile -- so dividing by :math:`\sigma_{\varpi,i}` alone inflates
       :math:`|z|` for exactly the best-measured stars. And the sample was
       queried with ``parallax BETWEEN 0.75 AND 1.1``, a window only ~1.2 median
       errors wide for the faintest quartile, which makes a *large*
       :math:`|z_i|` structurally unreachable there. Both are handled by the
       shipped form; neither is visible in a simulation without a query window.

    .. caution::
       **Every measurement above counts rejections, not false rejections.** The
       real sample contains contaminants, and no arm of this experiment sees the
       cost of the normalised clip's greater permissiveness: a field star offset
       from the centroid with a large ``e_Plx`` gets small :math:`|z|` and
       survives where the raw clip removes it. Retention and contamination move
       together and only one of them has been measured. The missing arm is a
       contaminated ROC -- inject field stars from the real cone at a distinct
       parallax and report retention of members *and* contaminants for both
       bases. Until that exists the defensible claim is that the raw clip has a
       measured magnitude dependence and this one has a smaller one, not that the
       normalised clip is uniformly better.
    """
    if method not in {"normalised", "raw"}:
        raise ValueError("method must be 'normalised' or 'raw'.")

    cluster_col = table["cluster"]
    parallax_col = _mas_values(table["parallax"])

    base_mask = np.asarray(cluster_col == cluster)
    if not np.any(base_mask):
        raise ValueError(f"Cluster {cluster} has zero rows in the provided table.")

    if preselector_mask is not None:
        preselector_mask = np.asarray(preselector_mask, dtype=bool)
        if preselector_mask.shape != base_mask.shape:
            raise ValueError("preselector_mask must have the same length as the table.")
        selection_mask = base_mask & preselector_mask
    else:
        selection_mask = base_mask

    parallax_values = parallax_col[selection_mask]
    if parallax_values.size == 0:
        raise ValueError("No rows selected for sigma clipping (mask is empty).")
    n_finite = np.count_nonzero(np.isfinite(parallax_values))
    if n_finite < 3:
        raise ValueError(f"Not enough finite parallax values for sigma clipping: {n_finite}")

    cenfunc = biweight_location
    stdfunc = biweight_scale if use_biweight else mad_std

    if method == "raw":
        statistic_all = parallax_col
        _, lower, upper = sigma_clip(
            statistic_all[selection_mask],
            sigma=sigma,
            cenfunc=cenfunc,
            stdfunc=stdfunc,
            maxiters=maxiters,
            return_bounds=True,
        )
    else:
        if parallax_error_column not in table.colnames:
            raise ValueError(
                f"method='normalised' needs the per-star uncertainty column "
                f"{parallax_error_column!r}, which is not in the table. Supply it, or "
                "pass method='raw' to clip the raw parallax as before 2026-08-02 -- "
                "which is a magnitude-dependent selection, not only an outlier test."
            )
        errors_all = _finite_positive_errors(table[parallax_error_column], parallax_error_column)
        statistic_all, centre, intrinsic = _normalised_residual(
            parallax_col,
            errors_all,
            selection_mask,
            sigma=sigma,
            maxiters=maxiters,
            cenfunc=cenfunc,
        )
        # The scale of z is 1 by construction, so it is asserted, not re-estimated.
        lower, upper = -float(sigma), float(sigma)
        del centre, intrinsic

    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise RuntimeError(f"sigma_clip returned non-finite bounds: lower={lower}, upper={upper}")

    within_bounds = (statistic_all >= lower) & (statistic_all <= upper)
    if preselector_mask is not None:
        final_keep = base_mask & preselector_mask & within_bounds
    else:
        final_keep = base_mask & within_bounds
    to_noise = base_mask & ~final_keep

    if in_place:
        if mark_label < 0 and not np.issubdtype(cluster_col.dtype, np.signedinteger):
            raise TypeError(
                f"'cluster' dtype must be signed int to assign {mark_label}, got {cluster_col.dtype}"
            )
        table["cluster"][to_noise] = mark_label
        if print_results:
            print(f"{int(base_mask.sum())} before sigma clipping")
            print(f"{int(final_keep.sum())} after sigma clipping")
        if return_mask:
            return lower, upper, final_keep, to_noise
        return lower, upper

    data_copy = table.copy(copy_data=True)
    data_copy["cluster"][to_noise] = mark_label
    if print_results:
        print(f"{int(base_mask.sum())} before sigma clipping")
        print(f"{int(final_keep.sum())} after sigma clipping")
    if return_mask:
        return lower, upper, data_copy, final_keep, to_noise
    return lower, upper, data_copy


__all__ = ["DEFAULT_MAXITERS", "sigma_clip_parallax"]
