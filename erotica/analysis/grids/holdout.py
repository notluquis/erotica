r"""Validation of a grid by **held-out nodes** -- the grid's own error, separate from sampling.

Convergence is not calibration (``methodology.md`` §A.1.2): an R-hat of 1.00 certifies that the
chains agree about a likelihood, not that the likelihood's isochrone is right. A coarse emulator in
this programme passed every convergence check and was biased by 22 sigma
(``agent-findings/differentiable-emulator-design.md``). So every backend is validated here, before
any fit, by the question the fitter actually asks it: *the isochrone between nodes*.

**Procedure.** For an interior node ``(feh, loga_j)`` the node is removed (:meth:`IsochroneGrid.
without`, which refuses a removal that removes nothing) and the isochrone at ``loga_j`` is
interpolated from the remaining neighbours ``loga_{j-1}, loga_{j+1}`` -- **twice the native step**
-- exactly as ``IsochroneFitter._interp_isochrone`` does (linear in [Fe/H] and log t at fixed EEP;
``test_holdout_interpolation_is_the_fitters`` holds the two to rounding). The prediction is compared
with the true node in three ways:

``mass-matched``
    :math:`|\Delta G|`, :math:`|\Delta c|` of the star of the same initial mass -- the physical
    question, independent of how the grid parametrises EEPs, which is what lets pseudo-EEP schemes be
    compared with native EEPs on the same isochrones;
``orthogonal``
    the CMD distance from each true point to the predicted polyline -- what the likelihood sees,
    since the mass of an observed star is not known;
``inferred age``
    the log t, on the reduced grid, that best matches the true node (IMF-weighted mean squared
    orthogonal distance, 1-D search at the node's [Fe/H]): :math:`\Delta\log t` = that minus the
    truth. This is the error in **age** that interpolation alone would put in a fit.

All three are IMF-weighted (Chabrier 2014 system IMF, the fitter's) over the true node's points
inside a mass window. Numbers are reported at 2x the native step, as measured; scaling them to the
native step as the square of the step (linear interpolation's error) is an assumption, stated
wherever it is used (``tools/validation/isochrone_likelihood_d1/budget.py`` did the same).
"""

from __future__ import annotations

import numpy as np

from .base import IsochroneGrid, common_eep_table


def _xi(m: np.ndarray) -> np.ndarray:
    # Chabrier (2014) system IMF, the constants of erotica.analysis._isochrone._chabrier2014_xi
    mc, sigma = 0.20, 0.55
    scale = float(np.exp(-0.5 * (np.log10(mc) / sigma) ** 2))
    ln = np.exp(-0.5 * ((np.log10(m) - np.log10(mc)) / sigma) ** 2) / m
    return np.where(m < 1.0, ln, scale * m**-2.35)


def _bracket(nodes: np.ndarray, x: float) -> tuple[int, int, float]:
    n = len(nodes)
    if n == 1:
        return 0, 0, 0.0
    k = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, n - 2))
    return k, k + 1, float(np.clip((x - nodes[k]) / (nodes[k + 1] - nodes[k]), 0, 1))


def interpolate(grid: IsochroneGrid, feh: float, loga: float, bands: tuple[str, str, str]):
    """Isochrone at ``(feh, loga)`` from ``grid``'s bracketing nodes: ``(mass, G, colour)`` on the
    integer EEP axis, bilinear at fixed EEP -- the fitter's formula, without the binary rows."""
    fn, an = grid.feh_nodes, grid.loga_nodes
    i0, i1, wz = _bracket(fn, feh)
    j0, j1, wa = _bracket(an, loga)
    fehs = np.array(sorted({fn[i0], fn[i1]}))
    ages = np.array(sorted({an[j0], an[j1]}))
    _, T = common_eep_table(grid, fehs, ages, bands)
    ii = {fn[i0]: 0, fn[i1]: len(fehs) - 1}
    jj = {an[j0]: 0, an[j1]: len(ages) - 1}
    X = (
        (1 - wz) * (1 - wa) * T[ii[fn[i0]], jj[an[j0]]]
        + wz * (1 - wa) * T[ii[fn[i1]], jj[an[j0]]]
        + (1 - wz) * wa * T[ii[fn[i0]], jj[an[j1]]]
        + wz * wa * T[ii[fn[i1]], jj[an[j1]]]
    )
    return X[0], X[1], X[2] - X[3]


def _densify(G: np.ndarray, c: np.ndarray, k: int = 8) -> np.ndarray:
    t = (np.arange(k) / k)[None, :]
    g = (G[:-1, None] + (G[1:] - G[:-1])[:, None] * t).ravel()
    cc = (c[:-1, None] + (c[1:] - c[:-1])[:, None] * t).ravel()
    return np.column_stack([np.r_[g, G[-1]], np.r_[cc, c[-1]]])


def _wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, cw, x[o]))


def holdout_node(
    grid: IsochroneGrid,
    feh: float,
    loga: float,
    bands: tuple[str, str, str],
    mass_window: tuple[float, float] = (0.1, 7.0),
    truth: IsochroneGrid | None = None,
    n_age: int = 161,
) -> dict:
    """Hold out every node at ``loga`` and measure the interpolated isochrone at ``(feh, loga)``.

    ``truth`` (default ``grid``) supplies the true node -- pass the native grid when ``grid`` is a
    re-parametrised copy, so that schemes are scored against the same isochrone.
    """
    from scipy.spatial import cKDTree

    truth = grid if truth is None else truth
    reduced = grid.without(loga=loga)
    if reduced.has(feh, loga):
        raise AssertionError("the held-out node is still in the reduced grid")
    tn = truth.node(feh, loga)
    o = np.argsort(tn.mass, kind="stable")
    m = tn.mass[o]
    G = tn.mags[bands[0]][o]
    c = tn.mags[bands[1]][o] - tn.mags[bands[2]][o]
    sel = (m >= mass_window[0]) & (m <= mass_window[1])
    if sel.sum() < 3:
        raise ValueError(f"fewer than 3 true points in the mass window at ({feh}, {loga})")
    w = _xi(m) * np.abs(np.gradient(m))
    m, G, c, w = m[sel], G[sel], c[sel], w[sel]

    pm, pG, pc = interpolate(reduced, feh, loga, bands)
    keep = np.r_[True, np.diff(pm) > 0]  # clamped ends repeat a point
    pm, pG, pc = pm[keep], pG[keep], pc[keep]
    inside = (m >= pm.min()) & (m <= pm.max())
    dG = np.abs(np.interp(m, pm, pG) - G)[inside]
    dc = np.abs(np.interp(m, pm, pc) - c)[inside]
    wi = w[inside]
    orth = cKDTree(_densify(pG, pc)).query(np.column_stack([G, c]))[0]

    an = reduced.loga_nodes
    j = int(np.searchsorted(an, loga))
    lo, hi = an[max(j - 1, 0)], an[min(j, len(an) - 1)]
    grid_a = np.linspace(lo, hi, n_age)
    cost = []
    for a in grid_a:
        _, qG, qc = interpolate(reduced, feh, a, bands)
        d = cKDTree(_densify(qG, qc)).query(np.column_stack([G, c]))[0]
        cost.append(float(np.sum(w * d**2) / w.sum()))
    a_best = float(grid_a[int(np.argmin(cost))])

    def q(x: np.ndarray, ww: np.ndarray) -> dict:
        return {f"p{int(100 * p)}": _wquantile(x, ww, p) for p in (0.5, 0.68, 0.95)}

    out = {
        "feh": float(feh),
        "loga": float(loga),
        "neighbours": [float(lo), float(hi)],
        "n_points": int(sel.sum()),
        "frac_mass_matched": float(inside.mean()),
        "dG": q(dG, wi),
        "dcol": q(dc, wi),
        "orth": q(orth, w),
        "dloga_inferred": a_best - float(loga),
        "age_search_step": float(grid_a[1] - grid_a[0]),
    }
    if not (out["orth"]["p95"] > 0 or out["dG"]["p95"] > 0):
        raise AssertionError("zero hold-out error: the node was not held out")
    return out


def holdout_feh_node(
    grid: IsochroneGrid,
    feh: float,
    loga: float,
    bands: tuple[str, str, str],
    mass_window: tuple[float, float] = (0.1, 7.0),
    n_feh: int = 161,
) -> dict:
    """The [Fe/H] analogue of :func:`holdout_node`: remove every node at ``feh``, interpolate from
    the neighbouring [Fe/H] nodes at the same log t, and infer [Fe/H] by the same 1-D search."""
    from scipy.spatial import cKDTree

    reduced = grid.without(feh=feh)
    if reduced.has(feh, loga):
        raise AssertionError("the held-out node is still in the reduced grid")
    tn = grid.node(feh, loga)
    o = np.argsort(tn.mass, kind="stable")
    m, G = tn.mass[o], tn.mags[bands[0]][o]
    c = tn.mags[bands[1]][o] - tn.mags[bands[2]][o]
    w = _xi(m) * np.abs(np.gradient(m))
    sel = (m >= mass_window[0]) & (m <= mass_window[1])
    m, G, c, w = m[sel], G[sel], c[sel], w[sel]
    pm, pG, pc = interpolate(reduced, feh, loga, bands)
    keep = np.r_[True, np.diff(pm) > 0]
    pm, pG, pc = pm[keep], pG[keep], pc[keep]
    inside = (m >= pm.min()) & (m <= pm.max())
    dG = np.abs(np.interp(m, pm, pG) - G)[inside]
    dc = np.abs(np.interp(m, pm, pc) - c)[inside]
    orth = cKDTree(_densify(pG, pc)).query(np.column_stack([G, c]))[0]
    fn = reduced.feh_nodes
    j = int(np.searchsorted(fn, feh))
    lo, hi = fn[max(j - 1, 0)], fn[min(j, len(fn) - 1)]
    grid_f = np.linspace(lo, hi, n_feh)
    cost = []
    for f in grid_f:
        _, qG, qc = interpolate(reduced, f, loga, bands)
        d = cKDTree(_densify(qG, qc)).query(np.column_stack([G, c]))[0]
        cost.append(float(np.sum(w * d**2) / w.sum()))
    q = lambda x, ww: {f"p{int(100 * p)}": _wquantile(x, ww, p) for p in (0.5, 0.68, 0.95)}  # noqa: E731
    return {
        "feh": float(feh),
        "loga": float(loga),
        "neighbours": [float(lo), float(hi)],
        "dG": q(dG, w[inside]),
        "dcol": q(dc, w[inside]),
        "orth": q(orth, w),
        "dfeh_inferred": float(grid_f[int(np.argmin(cost))]) - float(feh),
    }


def holdout_grid(
    grid: IsochroneGrid,
    bands: tuple[str, str, str],
    *,
    feh: float | None = None,
    loga_range: tuple[float, float] | None = None,
    mass_window: tuple[float, float] = (0.1, 7.0),
    truth: IsochroneGrid | None = None,
) -> dict:
    """:func:`holdout_node` at every interior age node (both neighbours present) of one [Fe/H]."""
    fehs = grid.feh_nodes
    feh = float(fehs[np.argmin(np.abs(fehs))]) if feh is None else float(feh)
    ages = grid.select(feh=feh).loga_nodes
    rows = []
    for k in range(1, len(ages) - 1):
        a = float(ages[k])
        if loga_range is not None and not (loga_range[0] <= a <= loga_range[1]):
            continue
        rows.append(holdout_node(grid, feh, a, bands, mass_window, truth))
    if not rows:
        raise ValueError("no interior node in range")

    def agg(key: str, p: str) -> float:
        return float(np.median([r[key][p] for r in rows]))

    da = np.array([r["dloga_inferred"] for r in rows])
    return {
        "grid": grid.name,
        "eep_kind": grid.eep_kind,
        "feh": feh,
        "bands": list(bands),
        "mass_window": list(mass_window),
        "native_step_median": float(np.median(np.diff(ages))),
        "n_nodes": len(rows),
        "median_over_nodes": {
            "dG_p68": agg("dG", "p68"),
            "dG_p95": agg("dG", "p95"),
            "dcol_p68": agg("dcol", "p68"),
            "dcol_p95": agg("dcol", "p95"),
            "orth_p68": agg("orth", "p68"),
            "orth_p95": agg("orth", "p95"),
        },
        "dloga_inferred": {
            "median_abs": float(np.median(np.abs(da))),
            "max_abs": float(np.max(np.abs(da))),
            "mean": float(np.mean(da)),
        },
        "nodes": rows,
    }


def recommended_sigma_floor(result: dict, reference: float = 0.01) -> dict:
    """Forward-model floor from a hold-out result: the IMF-weighted 68th percentile of the
    orthogonal error at 2x the native step, scaled to the native step as the square of the step
    (an assumption: exact for a quadratic error of linear interpolation), never below the MIST
    reference ``reference`` (the value ``IsochroneFitter.SIGMA_FLOOR`` was set to from MIST's own
    hold-out, 2026-09-23). Returns the number with its derivation."""
    p68 = result["median_over_nodes"]["orth_p68"]
    return {
        "orth_p68_at_2x_step": p68,
        "scaled_to_native_step": p68 / 4.0,
        "floor": max(reference, p68 / 4.0),
        "assumption": "error scales as step**2 (linear interpolation of a smooth function)",
    }
