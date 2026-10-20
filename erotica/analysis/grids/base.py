r"""The isochrone-grid contract: nodes in ([Fe/H], log t), points along an (equivalent) phase axis.

A grid is a set of **nodes** :math:`(\mathrm{[Fe/H]}_i, \log t_j)`. At every node it provides one
isochrone as arrays ordered along an **equivalent-evolutionary-phase coordinate** ``eep``:

.. math::
   (\mathrm{[Fe/H]}_i, \log t_j) \mapsto \big(\mathrm{eep}_k,\ m_{\mathrm{ini},k},\
   M_{b,k}\ \text{for every declared band } b\big)_{k}

The fitter (:class:`erotica.analysis.IsochroneFitter`) interpolates **linearly between nodes at
fixed ``eep``** -- in [Fe/H] and in log t -- so the coordinate must mean "the same physical state"
across nodes. Three kinds exist, declared in :attr:`IsochroneGrid.eep_kind`:

``"native"``
    the grid ships EEPs (MIST: Dotter 2016, ``2016ApJS..222....8D``);
``"pseudo"``
    built here from the grid's phase labels (PARSEC; :mod:`.pseudo_eep`);
``"mass"``
    the index of a fixed initial-mass list (BHAC15, SPOTS: PMS-only grids tabulated on the same
    masses at every age). Interpolating at fixed mass between ages is interpolating along the
    track of that mass, which is what a track *is*; it is only valid while no phase boundary
    (ZAMS, TAMS) is crossed between the two nodes -- true inside these grids' PMS/MS window.

**Why [Fe/H] and not Z.** The linear ``Zinit`` of a file header is not comparable across grids or
even across versions of one grid: the solar Z and the definition of the [Fe/H] label differ. Measured
on the MIST headers (2026-10-04, ``tools/validation/isochrone_grids/feh_conventions.py``): MIST v1.2
labels obey ``[Fe/H] = log10(Z / 0.0142857)`` exactly, while MIST v2.5 labels obey
``[Fe/H] = log10(Z/X) - log10(Z/X)_sun`` with ``Z_sun = 0.0163577``; the same label "+0.50" is
``log(Z/X)`` offset +0.549 in v1.2 and +0.500 in v2.5. So the fit coordinate is the grid's own
**[Fe/H] label**, and Z is kept only as provenance.

``feh_to_z`` / ``z_to_feh`` are the **node map**: piecewise linear in ``log10 Z`` between the
grid's own (Z, [Fe/H]) node pairs. It is exact at the nodes and, between them, it is the map that
makes a fit in [Fe/H] reproduce a fit in ``log10 Z`` with identical interpolation weights -- which
is what the regression test against the legacy Z fit needs. It is not a physical formula off the
nodes; the physical ones are in each backend's docstring, with their measured residual.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_KEY_DECIMALS = 6


def _key(feh: float, loga: float) -> tuple[float, float]:
    return (round(float(feh), _KEY_DECIMALS), round(float(loga), 4))


@dataclass
class GridNode:
    """One isochrone of a grid, ordered by ``eep``."""

    eep: np.ndarray
    mass: np.ndarray
    mags: dict[str, np.ndarray]
    extra: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class GridProvenance:
    """Where the numbers come from. Every field is a statement a reader can check."""

    family: str
    version: str
    references: tuple[str, ...]
    source: str
    license: str
    metallicity: str
    passbands: str
    sha256: tuple[str, ...] = ()
    notes: str = ""


class IsochroneGrid:
    """Base class: a dictionary of :class:`GridNode` keyed by ``([Fe/H], log t)``.

    Subclasses fill ``_nodes``, ``_z`` (Z of each [Fe/H] node, or ``None`` when the grid does
    not declare it), :attr:`bands`, :attr:`eep_kind`, :attr:`provenance` and, optionally,
    :attr:`sigma_floor`, the forward-model floor measured on this grid's own hold-out
    (:mod:`.holdout`). ``None`` means "not measured": the fitter then keeps its own default and
    says so.
    """

    eep_kind: str = "native"
    sigma_floor: float | None = None
    #: bands the fitter uses when it is not told otherwise: (magnitude, colour blue, colour red)
    default_bands: tuple[str, str, str] = ("", "", "")
    #: effective wavelengths (Angstrom) of ``default_bands`` for the CCM89 extinction law
    default_effl: tuple[float, float, float] = (6390.7, 5182.6, 7825.1)

    def __init__(self) -> None:
        self._nodes: dict[tuple[float, float], GridNode] = {}
        self._z: dict[float, float | None] = {}
        self.bands: tuple[str, ...] = ()
        self.provenance: GridProvenance | None = None
        self.name: str = type(self).__name__

    # -- node access -------------------------------------------------------------------------

    @property
    def feh_nodes(self) -> np.ndarray:
        return np.array(sorted({k[0] for k in self._nodes}), float)

    @property
    def loga_nodes(self) -> np.ndarray:
        return np.array(sorted({k[1] for k in self._nodes}), float)

    def has(self, feh: float, loga: float) -> bool:
        return _key(feh, loga) in self._nodes

    def node(self, feh: float, loga: float) -> GridNode:
        """The isochrone AT a node; ``KeyError`` if ``(feh, loga)`` is not one (no nearest)."""
        k = _key(feh, loga)
        if k not in self._nodes:
            raise KeyError(f"({feh}, {loga}) is not a node of {self.name}")
        return self._nodes[k]

    def nodes(self) -> list[tuple[float, float]]:
        return sorted(self._nodes)

    def _add(self, feh: float, loga: float, node: GridNode, z: float | None) -> None:
        o = np.argsort(node.eep, kind="stable")
        node = GridNode(
            node.eep[o],
            node.mass[o],
            {b: v[o] for b, v in node.mags.items()},
            {b: v[o] for b, v in node.extra.items()},
        )
        self._nodes[_key(feh, loga)] = node
        self._z[round(float(feh), _KEY_DECIMALS)] = z

    # -- views -------------------------------------------------------------------------------

    def _view(self, keep: Any) -> IsochroneGrid:
        g = copy.copy(self)
        g._nodes = {k: v for k, v in self._nodes.items() if keep(k)}
        g._z = {f: z for f, z in self._z.items() if any(k[0] == f for k in g._nodes)}
        if not g._nodes:
            raise ValueError("the selection leaves no node")
        return g

    def select(
        self, feh: float | list[float] | None = None, loga_range: tuple[float, float] | None = None
    ) -> IsochroneGrid:
        """A view restricted to some [Fe/H] nodes and/or a log-t range (inclusive)."""
        fs = None if feh is None else {round(float(f), _KEY_DECIMALS) for f in np.atleast_1d(feh)}
        if fs is not None and not fs <= set(self._z):
            raise KeyError(f"[Fe/H] {sorted(fs - set(self._z))} not among the nodes of {self.name}")

        def keep(k: tuple[float, float]) -> bool:
            ok = fs is None or k[0] in fs
            if loga_range is not None:
                ok = ok and loga_range[0] - 1e-9 <= k[1] <= loga_range[1] + 1e-9
            return ok

        return self._view(keep)

    def without(self, feh: float | None = None, loga: float | None = None) -> IsochroneGrid:
        """A view with every node at ``feh`` (if given) and/or at ``loga`` (if given) removed --
        the hold-out operation. Raises if nothing was removed (a hold-out that holds nothing out
        is the mutation that makes the hold-out error zero)."""

        def drop(k: tuple[float, float]) -> bool:
            a = feh is None or k[0] == round(float(feh), _KEY_DECIMALS)
            b = loga is None or k[1] == round(float(loga), 4)
            return a and b

        g = self._view(lambda k: not drop(k))
        if len(g._nodes) == len(self._nodes):
            raise ValueError(f"nothing removed: ({feh}, {loga}) matches no node of {self.name}")
        return g

    # -- metallicity -------------------------------------------------------------------------

    def _z_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        f = self.feh_nodes
        z = [self._z.get(round(float(x), _KEY_DECIMALS)) for x in f]
        if any(v is None for v in z):
            raise ValueError(f"{self.name} does not declare Z at every [Fe/H] node")
        return f, np.asarray(z, float)

    def feh_to_z(self, feh: Any) -> Any:
        """Node map [Fe/H] -> Z: piecewise linear in log10 Z between the nodes (see module)."""
        f, z = self._z_pairs()
        x = np.asarray(feh, float)
        if np.any(x < f[0] - 1e-9) or np.any(x > f[-1] + 1e-9):
            raise ValueError(f"[Fe/H] {feh} outside the nodes [{f[0]}, {f[-1]}] of {self.name}")
        return 10.0 ** np.interp(x, f, np.log10(z))

    def z_to_feh(self, z: Any) -> Any:
        """Inverse node map Z -> [Fe/H]."""
        f, zz = self._z_pairs()
        lz = np.log10(np.asarray(z, float))
        if np.any(lz < np.log10(zz[0]) - 1e-9) or np.any(lz > np.log10(zz[-1]) + 1e-9):
            raise ValueError(f"Z {z} outside the nodes [{zz[0]}, {zz[-1]}] of {self.name}")
        return np.interp(lz, np.log10(zz), f)

    def describe(self) -> dict:
        p = self.provenance
        return {
            "grid": self.name,
            "family": p.family if p else None,
            "version": p.version if p else None,
            "eep_kind": self.eep_kind,
            "feh_nodes": self.feh_nodes.tolist(),
            "loga_range": [float(self.loga_nodes[0]), float(self.loga_nodes[-1])],
            "n_loga": int(len(self.loga_nodes)),
            "bands": list(self.bands),
            "sigma_floor": self.sigma_floor,
            "license": p.license if p else None,
            "sha256": list(p.sha256) if p else [],
        }


def common_eep_table(
    grid: IsochroneGrid,
    fehs: np.ndarray,
    ages: np.ndarray,
    bands: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack nodes on one integer ``eep`` axis, as the fitter's node table does.

    Rows: ``mass`` then each band. EEPs a node lacks are clamped to its first/last point
    (``np.interp``), where the mass step is zero and the IMF weight vanishes -- the convention of
    ``IsochroneFitter._build_nodes``. Returns ``(axis, T[n_feh, n_age, 1 + n_bands, n_eep])``.
    """
    per = {}
    lo, hi = np.inf, -np.inf
    for f in fehs:
        for a in ages:
            n = grid.node(f, a)
            per[(f, a)] = n
            lo, hi = min(lo, n.eep.min()), max(hi, n.eep.max())
    axis = np.arange(int(np.floor(lo)), int(np.ceil(hi)) + 1, dtype=float)
    T = np.empty((len(fehs), len(ages), 1 + len(bands), len(axis)))
    for i, f in enumerate(fehs):
        for j, a in enumerate(ages):
            n = per[(f, a)]
            T[i, j, 0] = np.interp(axis, n.eep, n.mass)
            for r, b in enumerate(bands):
                T[i, j, 1 + r] = np.interp(axis, n.eep, n.mags[b])
    return axis, T


def safe_window(
    grids: list[IsochroneGrid],
    mag_band: str | list[str],
    *,
    loga_range: tuple[float, float],
    dm_range: tuple[float, float],
    Av_range: tuple[float, float],
    k_mag: float,
    data_faint: float | None = None,
) -> tuple[float, float]:
    r"""The observed-magnitude window inside which **every** grid in ``grids`` has a model star for
    **every** parameter value allowed by the priors.

    .. math::
       m_{\rm bright} = \max_{\rm nodes,\,grids} \min_{\rm EEP} M + \mathrm{dm}_{\max}
       + k\,A_{V,\max},\qquad
       m_{\rm faint} = \min\Big(\min_{\rm nodes,\,grids} \max_{\rm EEP} M + \mathrm{dm}_{\min}
       + k\,A_{V,\min},\ m_{\rm data}\Big)

    over the nodes whose log t brackets ``loga_range``. Built from grids and priors only -- never
    from the stars -- so the cut cannot follow the data. A control and its main fit must use the
    SAME window, which is why this takes the list of grids. ``mag_band`` is one band name, or one
    per grid when the grids name the same passband differently.
    """
    bands = [mag_band] * len(grids) if isinstance(mag_band, str) else list(mag_band)
    bright, faint = -np.inf, np.inf
    for g, b in zip(grids, bands, strict=True):
        an = g.loga_nodes
        a0 = an[max(int(np.searchsorted(an, loga_range[0], side="right")) - 1, 0)]
        a1 = an[min(int(np.searchsorted(an, loga_range[1], side="left")), len(an) - 1)]
        for f, a in g.nodes():
            if a0 - 1e-9 <= a <= a1 + 1e-9:
                M = g.node(f, a).mags[b]
                bright = max(bright, float(M.min()))
                faint = min(faint, float(M.max()))
    lo = bright + dm_range[1] + k_mag * Av_range[1]
    hi = faint + dm_range[0] + k_mag * Av_range[0]
    if data_faint is not None:
        hi = min(hi, float(data_faint))
    if not lo < hi:
        raise ValueError(f"empty window: bright {lo:.3f} >= faint {hi:.3f}")
    return float(lo), float(hi)
