r"""Pseudo-EEPs for isochrone grids that ship phase labels but no EEP (PARSEC), and the two
alternatives they were measured against.

**Method (scheme ``"arclength"``).** Dotter 2016 (``2016ApJS..222....8D``, §II.1-II.3) builds EEPs
on *tracks*: primary EEPs at physically defined points, and between each pair a fixed number of
secondary EEPs equally spaced in a metric distance

.. math::
   D_{i+1} = D_i + \sqrt{\textstyle\sum_j w_j (x_{j,i+1} - x_{j,i})^2},
   \qquad x = (\log T_{\rm eff}, \log L),

with weights chosen so that both terms "contribute in roughly equal amounts". Here the same
construction is applied to each *isochrone*, with the grid's phase labels as primaries: inside every
phase block the isochrone is resampled at ``n`` points equally spaced in :math:`D`, and block ``p``
always occupies the EEP numbers ``[offset_p, offset_p + n_p)``. The weights are global constants
(:math:`w_j = 1/\Delta_j^2`, :math:`\Delta_j` the span of :math:`x_j` over the whole grid), so
a given EEP number means the same fractional distance along the same phase at every node.

**What differs from Dotter's EEPs, said plainly.** His secondary EEPs live on tracks (one mass, many
ages); these live on isochrones (one age, many masses). At fixed pseudo-EEP the star is "the one at
the same fraction of the PMS branch", not "the same mass at the same phase of its own evolution". The
two coincide only where the branch moves rigidly between ages. Whether that is good enough is
measured, not assumed: :mod:`.holdout` compares it with MIST's native EEPs on MIST itself.

**Prior art, integrated as inspiration (no code copied).** kiauhoku (Claytor et al.,
``2020ascl.soft11027C``; MIT) implements Dotter's secondary-EEP resampling for *tracks*
(``kiauhoku/utils/eep.py::_eep_interpolate``, ``_HRD_distance``); the metric and the
equal-distance resampling here follow it. It is not a dependency: its primaries are found from
central abundances that isochrone files do not carry, and it pulls numba, pyarrow and emcee.
The two comparison schemes reproduce published code paths, also re-written rather than imported:

``"index"``
    ezpadova (Fouesneau; MIT) ``parsec.resample_evolution_label``: the continuous coordinate
    ``label + j / n_label`` -- equal steps in *row index* within each label.
``"massquantile"``
    ASteCA 0.7.0 (Perren et al., ``2015A&A...576A...6P``; MIT)
    ``modules/isochrones_priv.py::interp_isochrones``: phase-agnostic, the isochrone resampled in
    initial mass on five blocks at 25/50/75/90 % of the maximum mass with 0.5:1:1.5:2:5 points.

**PARSEC's labels need one fix first** (measured on the CMD 3.7 v1.2S Gaia EDR3 file, 2026-10-04):
in 1168 of 1800 isochrones the label is not monotone in mass -- it flickers 0/1 for a few masses at
the PMS/MS boundary, and the massive branch (>= 20 Msun, from the separate massive-star tracks) is
labelled 0 (PMS) after the MS. :func:`effective_phase` uses the running maximum of the label along
the isochrone, so every row belongs to the latest phase reached by a lower mass.
"""

from __future__ import annotations

import numpy as np

# ASteCA 0.7.0 isochrones_priv.interp_isochrones, mass-based branch
_ASTECA_WEIGHTS = np.array([0.5, 1, 1.5, 2, 5], dtype=float)
_ASTECA_PERCS = np.array([0.25, 0.5, 0.75, 0.9], dtype=float)


def effective_phase(label: np.ndarray) -> np.ndarray:
    """Running maximum of the phase label along the isochrone (rows ordered by initial mass)."""
    return np.maximum.accumulate(np.asarray(label, float))


def metric_distance(logte: np.ndarray, logl: np.ndarray, w: tuple[float, float]) -> np.ndarray:
    """Cumulative Dotter-2016 Eq. 1 distance along the rows, ``D[0] = 0``."""
    d = np.sqrt(w[0] * np.diff(logte) ** 2 + w[1] * np.diff(logl) ** 2)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample(
    cols: dict[str, np.ndarray],
    *,
    scheme: str,
    phase: np.ndarray | None = None,
    phase_blocks: dict[float, tuple[int, int]] | None = None,
    weights: tuple[float, float] = (1.0, 1.0),
    n_total: int = 400,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Resample one isochrone onto a pseudo-EEP axis.

    Parameters
    ----------
    cols : dict
        Arrays along the isochrone, ordered by initial mass; must include ``"mass"``, and for
        ``"arclength"`` also ``"logte"`` and ``"logl"``. Every array is interpolated linearly in
        the scheme's coordinate.
    scheme : {"arclength", "index", "massquantile", "raw"}
    phase : array, optional
        Effective phase of each row (``"arclength"``, ``"index"``).
    phase_blocks : dict
        ``{phase: (offset, n)}`` -- the EEP numbers each phase occupies, the same at every node.
    weights : (w_logte, w_logl)
        Metric weights of :func:`metric_distance`.
    n_total : int
        Points for ``"massquantile"`` (ASteCA default is 2000; its own comments call 2000
        "borderline" and 5000 "good").

    Returns
    -------
    eep, resampled : ndarray, dict
    """
    mass = np.asarray(cols["mass"], float)
    if scheme == "raw":  # the rows as given: the truth that resampled schemes are scored against
        return np.arange(mass.size, dtype=float), {k: np.asarray(v, float) for k, v in cols.items()}
    if scheme == "massquantile":
        b = (n_total * _ASTECA_WEIGHTS / _ASTECA_WEIGHTS.sum())[:4].astype(int)
        b5 = n_total - b.sum()
        mmin, mmax = mass.min(), mass.max()
        s1, s2, s3, s4 = mmax * _ASTECA_PERCS
        m = np.concatenate(
            [
                np.linspace(mmin, s1, b[0], endpoint=False),
                np.linspace(s1, s2, b[1], endpoint=False),
                np.linspace(s2, s3, b[2], endpoint=False),
                np.linspace(s3, s4, b[3], endpoint=False),
                np.linspace(s4, mmax, b5),
            ]
        )
        return np.arange(n_total, dtype=float), {k: np.interp(m, mass, v) for k, v in cols.items()}

    if phase is None or phase_blocks is None:
        raise ValueError(f"scheme {scheme!r} needs `phase` and `phase_blocks`")
    eeps, out = [], {k: [] for k in cols}
    for p, (offset, n) in sorted(phase_blocks.items(), key=lambda kv: kv[1][0]):
        rows = np.flatnonzero(phase == p)
        if rows.size == 0:
            continue
        if rows.size == 1:
            eeps.append(np.array([float(offset)]))
            for k, v in cols.items():
                out[k].append(np.asarray(v, float)[rows])
            continue
        if scheme == "arclength":
            coord = metric_distance(cols["logte"][rows], cols["logl"][rows], weights)
        elif scheme == "index":
            coord = np.arange(rows.size, dtype=float)
        else:
            raise ValueError(f"unknown scheme {scheme!r}")
        if coord[-1] <= 0:
            coord = np.arange(rows.size, dtype=float)
        target = np.linspace(0.0, coord[-1], n)
        eeps.append(offset + np.arange(n, dtype=float))
        for k, v in cols.items():
            out[k].append(np.interp(target, coord, np.asarray(v, float)[rows]))
    eep = np.concatenate(eeps)
    return eep, {k: np.concatenate(v) for k, v in out.items()}


#: MIST ``phase`` values (-1 PMS, 0 MS, 2 RGB, 3 CHeB, 4 EAGB, 5 TPAGB, 6 post-AGB, 9 WR) -> blocks
MIST_BLOCKS = {
    -1: (0, 200),
    0: (200, 200),
    2: (400, 40),
    3: (440, 60),
    4: (500, 40),
    5: (540, 40),
    6: (580, 40),
    9: (620, 40),
}


def regrid(grid, scheme: str, blocks: dict | None = None, n_total: int = 400):
    """A copy of ``grid`` with every node re-parametrised by ``scheme``.

    Needs ``logte``, ``logl`` and ``phase`` in each node's ``extra`` (MIST provides them). Used to
    measure a pseudo-EEP scheme against MIST's native EEPs on the same isochrones.
    """
    import copy

    from .base import GridNode

    blocks = MIST_BLOCKS if blocks is None else blocks
    nodes = list(grid._nodes.values())
    lt = np.concatenate([n.extra["logte"] for n in nodes])
    ll = np.concatenate([n.extra["logl"] for n in nodes])
    w = (1.0 / np.ptp(lt) ** 2, 1.0 / np.ptp(ll) ** 2)
    g = copy.copy(grid)
    g._nodes = {}
    for k, n in grid._nodes.items():
        o = np.argsort(n.mass, kind="stable")
        cols = {"mass": n.mass[o], "logte": n.extra["logte"][o], "logl": n.extra["logl"][o]}
        cols.update({b: v[o] for b, v in n.mags.items()})
        eep, r = resample(
            cols,
            scheme=scheme,
            phase=effective_phase(n.extra["phase"][o]),
            phase_blocks=blocks,
            weights=w,
            n_total=n_total,
        )
        g._nodes[k] = GridNode(
            eep, r["mass"], {b: r[b] for b in n.mags}, {"logte": r["logte"], "logl": r["logl"]}
        )
    g.eep_kind = "pseudo"
    g.name = f"{grid.name} [{scheme}]"
    return g
