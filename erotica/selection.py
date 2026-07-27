"""Selection-function / completeness helpers for EROTICA member samples.

This module adds the two pieces of "how complete is my sample" machinery that
the paper backlog (P02/P03/P04/P05/P08) needs, without dragging heavy optional
dependencies into ``import erotica``:

1. **Per-star completeness** of a member table, via the empirical Gaia DR3
   source selection function shipped in :mod:`gaiaunlimited`
   (``DR3SelectionFunctionTCG``). This answers, for each surviving member,
   "what fraction of stars with this sky position and this G magnitude made it
   into Gaia DR3?" -- the natural weight for Horvitz-Thompson-style number /
   luminosity / mass corrections.

2. **Cluster-census detectability**, via
   :mod:`hr_selection_function` (``HR24SelectionFunction``; Hunt et al. 2026,
   A&A 706, A341). This answers, for a *whole cluster*,
   ``P(detected | N_stars, median parallax error, local Gaia data density,
   significance threshold)`` -- the natural weight for a cluster-census
   completeness correction.

Both packages are **optional** and are *not* installed by default. They are
imported lazily, exactly like :mod:`erotica.analysis.external.asteca`, so that
importing this module stays cheap and only raises a helpful ``ImportError``
(with install instructions) when a selection-function path is actually used.

Attribution note
----------------
The default per-star selection function ``DR3SelectionFunctionTCG`` is the
Cantat-Gaudin et al. (2023) empirical Gaia DR3 *source* selection function --
the right default for member-sample completeness. :mod:`gaiaunlimited` also
ships the Castro-Ginard et al. (2023) *subsample* selection functions
(``SubsampleSelectionFunction``); pass one via ``selection_function=`` to weight
by subsample completeness (e.g. an astrometric / RVS quality cut) instead.

Data model
----------
A EROTICA member table is an :class:`astropy.table.QTable` with (at least) the
columns ``ra`` and ``dec`` (deg), ``Gmag`` (Gaia G), ``parallax_error`` (mas),
and a membership ``probability`` column -- the same names used across
:mod:`erotica.analysis.structure` and :mod:`erotica.io`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable

__all__ = [
    "attach_completeness_weights",
    "cluster_census_detectability",
    "census_detectability_from_members",
    "default_gaia_selection_function",
]


# ---------------------------------------------------------------------------
# Lazy optional-dependency guards (mirrors erotica.analysis.external.asteca)
# ---------------------------------------------------------------------------
def _require_gaiaunlimited():
    """Return the ``gaiaunlimited.selectionfunctions`` module or raise a helpful error."""
    try:
        from gaiaunlimited import selectionfunctions
    except ImportError as exc:  # pragma: no cover - exercised via mocked sys.modules
        raise ImportError(
            "gaiaunlimited is required for per-star Gaia DR3 completeness weights, "
            "but it is not installed.\n"
            "Install it with:\n"
            "    pip install gaiaunlimited\n"
            "and point it at a data cache directory (it downloads the DR3 selection "
            "function map on first use), e.g.:\n"
            "    export GAIAUNLIMITED_DATADIR=~/.gaiaunlimited\n"
            "See https://gaiaunlimited.readthedocs.io for details."
        ) from exc
    return selectionfunctions


def _require_hr_selection_function():
    """Return the ``hr_selection_function`` module or raise a helpful error."""
    try:
        import hr_selection_function
    except ImportError as exc:  # pragma: no cover - exercised via mocked sys.modules
        raise ImportError(
            "hr_selection_function is required for cluster-census detectability, "
            "but it is not installed.\n"
            "Install it with:\n"
            "    pip install hr-selection-function\n"
            "It downloads ~500 MB of model data on first use -- choose the location "
            "with hr_selection_function.set_data_directory(path) -- requires Python "
            ">=3.10, and is not Windows-compatible (it depends on healpy).\n"
            "See https://github.com/emilyhunt/hr_selection_function for details."
        ) from exc
    return hr_selection_function


def default_gaia_selection_function():
    """Instantiate the default Gaia DR3 source selection function.

    This is a thin, explicit constructor so callers can build the (expensive,
    data-loading) selection function *once* and reuse it across many tables::

        sf = default_gaia_selection_function()
        members = attach_completeness_weights(members, selection_function=sf)
    """
    return _require_gaiaunlimited().DR3SelectionFunctionTCG()


# ---------------------------------------------------------------------------
# Small unit helper (kept local to avoid importing the heavy erotica.analysis
# package just for erotica.analysis.units.quantity_values).
# ---------------------------------------------------------------------------
def _values_in(value: Any, unit: u.UnitBase | None = None) -> np.ndarray:
    """Return a float ndarray from a Quantity/Column/array, converting to ``unit``.

    Mirrors :func:`erotica.analysis.units.quantity_values` so behaviour matches the
    rest of the codebase, without triggering the heavy ``erotica.analysis`` import.
    """
    if hasattr(value, "to") and unit is not None:
        arr = np.asarray(value.to(unit).value, dtype=float)
    elif hasattr(value, "value"):
        arr = np.asarray(value.value, dtype=float)
    else:
        arr = np.asarray(value, dtype=float)
    if np.ma.isMaskedArray(arr):
        arr = arr.filled(np.nan)
    return arr


# ---------------------------------------------------------------------------
# 1. Per-star completeness weights
# ---------------------------------------------------------------------------
def attach_completeness_weights(
    members: QTable,
    *,
    ra_column: str = "ra",
    dec_column: str = "dec",
    gmag_column: str = "Gmag",
    completeness_column: str = "completeness",
    weight_column: str = "completeness_weight",
    selection_function: Any | None = None,
    min_completeness: float = 1e-3,
    frame: str = "icrs",
    copy: bool = True,
) -> QTable:
    """Attach per-star Gaia DR3 completeness and inverse-completeness weights.

    For every member the Gaia DR3 selection probability ``P`` (the fraction of
    stars of that sky position and G magnitude that enter the catalogue) is
    written to ``completeness_column`` and its Horvitz-Thompson weight
    ``1 / max(P, min_completeness)`` to ``weight_column``. Multiply the weight by
    the membership ``probability`` yourself if you want an effective weight.

    Parameters
    ----------
    members : QTable
        Member table with sky-coordinate and G-magnitude columns.
    ra_column, dec_column, gmag_column : str
        Column names for right ascension (deg), declination (deg) and Gaia G mag.
    completeness_column, weight_column : str
        Output column names for the completeness ``P`` and its inverse weight.
    selection_function : object, optional
        A pre-built selection function exposing ``query(coords, gmag) -> array``.
        Defaults to :func:`default_gaia_selection_function` (``DR3SelectionFunctionTCG``),
        constructed lazily. Inject a ``gaiaunlimited`` ``SubsampleSelectionFunction``
        (Castro-Ginard et al. 2023) here to weight by subsample completeness.
    min_completeness : float
        Floor applied before inverting, to keep weights finite. Default ``1e-3``.
    frame : str
        Coordinate frame for the built :class:`~astropy.coordinates.SkyCoord`.
    copy : bool
        If ``True`` (default) operate on a copy; otherwise mutate ``members``.

    Returns
    -------
    QTable
        The table with the two new columns. Rows with non-finite ``ra``/``dec``/
        ``Gmag`` get ``NaN`` completeness and weight (they are never passed to the
        selection function).
    """
    for col in (ra_column, dec_column, gmag_column):
        if col not in members.colnames:
            raise ValueError(f"Missing required column {col!r} in member table.")

    out = members.copy() if copy else members

    ra = _values_in(members[ra_column], u.deg)
    dec = _values_in(members[dec_column], u.deg)
    gmag = _values_in(members[gmag_column], u.mag)

    sf = selection_function if selection_function is not None else default_gaia_selection_function()

    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(gmag)
    completeness = np.full(len(members), np.nan, dtype=float)
    if np.any(finite):
        coords = SkyCoord(ra=ra[finite] * u.deg, dec=dec[finite] * u.deg, frame=frame)
        completeness[finite] = np.asarray(sf.query(coords, gmag[finite]), dtype=float)

    out[completeness_column] = completeness
    out[weight_column] = 1.0 / np.clip(completeness, min_completeness, None)
    return out


# ---------------------------------------------------------------------------
# 2. Cluster-census detectability
# ---------------------------------------------------------------------------
def cluster_census_detectability(
    n_stars: Any,
    median_parallax_error: Any,
    significance_threshold: Any,
    *,
    data_density: Any | None = None,
    coordinates: SkyCoord | None = None,
    mode: str = "median",
    selection_function: Any | None = None,
) -> np.ndarray:
    """Query the Hunt et al. (2026) open-cluster census detectability.

    Returns ``P(detected)`` for one or more clusters given their size, median
    member parallax error, local Gaia data density (or 3D coordinates from which
    the density is computed internally) and the significance threshold used to
    claim a detection.

    Parameters
    ----------
    n_stars : int or array-like
        Number of member stars per cluster, shape ``(n_clusters,)``.
    median_parallax_error : float or array-like
        Median member parallax error per cluster, in **mas**, shape ``(n_clusters,)``.
    significance_threshold : float or array-like
        Detection significance threshold per cluster, shape ``(n_clusters,)``.
    data_density : float or array-like, optional
        Local Gaia data density per cluster. Provide **either** this **or**
        ``coordinates`` (exactly one).
    coordinates : SkyCoord, optional
        3D coordinates (position + proper motion) from which
        ``hr_selection_function`` computes the Gaia data density itself. Provide
        **either** this **or** ``data_density``.
    mode : str
        MCMC-parameter estimate to use: ``"median"`` (default), ``"mean"`` or
        ``"random_sample"`` (for epistemic-uncertainty sampling).
    selection_function : object, optional
        A pre-built ``HR24SelectionFunction`` (or compatible callable). Defaults to
        constructing one lazily.

    Returns
    -------
    numpy.ndarray
        1D array of detection probabilities, one per cluster.
    """
    if (data_density is None) == (coordinates is None):
        raise ValueError("Provide exactly one of `data_density` or `coordinates`.")

    model = (
        selection_function
        if selection_function is not None
        else _require_hr_selection_function().HR24SelectionFunction()
    )

    n_stars = np.atleast_1d(n_stars)
    median_parallax_error = np.atleast_1d(median_parallax_error)
    significance_threshold = np.atleast_1d(significance_threshold)
    density_or_coordinates = coordinates if data_density is None else np.atleast_1d(data_density)

    # Keyword args for the numeric parameters: if the underlying positional order
    # ever differs from what we expect, this fails loudly rather than silently
    # feeding the model transposed values.
    prob = model(
        density_or_coordinates,
        n_stars=n_stars,
        median_parallax_error=median_parallax_error,
        threshold=significance_threshold,
        mode=mode,
    )
    return np.asarray(prob, dtype=float)


def census_detectability_from_members(
    members: QTable,
    significance_threshold: Any,
    *,
    data_density: Any | None = None,
    coordinates: SkyCoord | None = None,
    probability_column: str = "probability",
    probability_threshold: float | None = None,
    parallax_error_column: str = "parallax_error",
    mode: str = "median",
    selection_function: Any | None = None,
) -> np.ndarray:
    """Convenience bridge: derive ``n_stars`` and median parallax error from a table.

    Selects members above ``probability_threshold`` (if given), then calls
    :func:`cluster_census_detectability` with ``n_stars = len(selected)`` and
    ``median_parallax_error = nanmedian(parallax_error)`` (mas). Provide exactly
    one of ``data_density`` / ``coordinates``.
    """
    table = members
    if probability_threshold is not None and probability_column in members.colnames:
        table = members[members[probability_column] >= probability_threshold]

    n_stars = len(table)
    median_parallax_error = float(np.nanmedian(_values_in(table[parallax_error_column], u.mas)))
    return cluster_census_detectability(
        n_stars,
        median_parallax_error,
        significance_threshold,
        data_density=data_density,
        coordinates=coordinates,
        mode=mode,
        selection_function=selection_function,
    )

