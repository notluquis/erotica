"""Photometric helpers and legacy isochrone utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.table import QTable
from scipy.spatial import KDTree

from .units import quantity_values

MIST_COLUMN_SETS = {
    "DR3": [
        "Zini",
        "MH",
        "logAge",
        "Mini",
        "int_IMF",
        "Mass",
        "logL",
        "logTe",
        "logg",
        "label",
        "McoreTP",
        "C_O",
        "period0",
        "period1",
        "period2",
        "period3",
        "period4",
        "pmode",
        "Mloss",
        "tau1m",
        "X",
        "Y",
        "Xc",
        "Xn",
        "Xo",
        "Cexcess",
        "Z",
        "mbolmag",
        "Gmag",
        "G_BPmag",
        "G_RPmag",
    ],
    "2MASS": [
        "Zini",
        "MH",
        "logAge",
        "Mini",
        "int_IMF",
        "Mass",
        "logL",
        "logTe",
        "logg",
        "label",
        "McoreTP",
        "C_O",
        "period0",
        "period1",
        "period2",
        "period3",
        "period4",
        "pmode",
        "Mloss",
        "tau1m",
        "X",
        "Y",
        "Xc",
        "Xn",
        "Xo",
        "Cexcess",
        "Z",
        "mbolmag",
        "Jmag",
        "Hmag",
        "Ksmag",
    ],
}


def read_isochrones_with_metadata(file_path, *, columns_type="DR3", split_blocks: bool = True):
    """Read a legacy MIST/Padova text file and retain header metadata.

    The paper notebooks expected ``(metadata, list[pandas.DataFrame])`` where
    each block between comment separators is an isochrone. ``split_blocks=False``
    returns a single concatenated DataFrame plus metadata for package-style use.
    """
    columns_type = columns_type.upper()
    if columns_type not in MIST_COLUMN_SETS:
        raise ValueError(f"columns_type must be one of {sorted(MIST_COLUMN_SETS)}.")
    columns = MIST_COLUMN_SETS[columns_type]

    with open(file_path, encoding="utf-8", errors="ignore") as handle:
        lines = handle.readlines()

    try:
        data_start_index = next(i for i, line in enumerate(lines) if line.startswith("# Zini"))
    except StopIteration as exc:
        raise ValueError(f"Could not find '# Zini' data header in {file_path!r}.") from exc

    metadata = lines[:data_start_index]
    separator_indices = [
        index for index, line in enumerate(lines[data_start_index:]) if line.startswith("#")
    ]
    separator_indices.append(len(lines) - data_start_index)

    frames = []
    for idx in range(len(separator_indices) - 1):
        start = data_start_index + separator_indices[idx] + 1
        end = data_start_index + separator_indices[idx + 1]
        rows = [line.strip().split() for line in lines[start:end] if line.strip()]
        if not rows:
            continue
        frame = pd.DataFrame(rows, columns=columns[: len(rows[0])])
        frames.append(frame)

    if not frames:
        raise ValueError(f"No isochrone rows found in {file_path!r}.")
    if split_blocks:
        return metadata, frames
    return pd.concat(frames, ignore_index=True), metadata


def set_column_types(isochrones, columns):
    """Convert selected DataFrame columns to numeric values in-place."""
    for df in isochrones:
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")
    return isochrones


def add_photometric_errors(table: QTable) -> list[str]:
    """Add Gaia-like photometric error columns to ``table``."""
    g = quantity_values(table["Gmag"], u.mag)

    def interp(vals, low, high, low_err, high_err):
        return np.interp(vals, [low, high], [low_err, high_err])

    e_g = (
        np.where(
            g < 13,
            0.3,
            np.where(
                g < 17, interp(g, 13, 17, 0.3, 1), np.where(g <= 20, interp(g, 17, 20, 1, 6), 6)
            ),
        )
        / 1000
    )
    e_bp = (
        np.where(
            g < 13,
            0.9,
            np.where(
                g < 17,
                interp(g, 13, 17, 0.9, 12),
                np.where(g <= 20, interp(g, 17, 20, 12, 108), 108),
            ),
        )
        / 1000
    )
    e_rp = (
        np.where(
            g < 13,
            0.6,
            np.where(
                g < 17, interp(g, 13, 17, 0.6, 6), np.where(g <= 20, interp(g, 17, 20, 6, 52), 52)
            ),
        )
        / 1000
    )
    table["e_Gmag"] = e_g * u.mag
    table["e_G_BPmag"] = e_bp * u.mag
    table["e_G_RPmag"] = e_rp * u.mag
    table["e_BP_RP"] = np.sqrt(e_bp**2 + e_rp**2) * u.mag
    return ["e_Gmag", "e_G_BPmag", "e_G_RPmag", "e_BP_RP"]


def assign_mass_nearest_isochrone_point_kdtree(
    first,
    second,
    *legacy_args,
    magnitude_column: str = "Gmag",
    color_column: str = "BP_RP",
    designation_column: str = "designation",
):
    r"""Assign masses from the nearest isochrone point in colour-magnitude space.

    New API: ``(stars, (iso_mag, iso_color, iso_mass), ...)``.
    Legacy API: ``(isochrones_df, stars, logAge_range, Zini, color_col, mag_col, dm, A_V)``.

    .. danger::
       **The Euclidean distance in a CMD is not a metric with physical meaning, and this function's
       per-star output depends on an arbitrary choice that nobody states.** Colour and magnitude have
       different units, different uncertainties and different dynamic ranges, so "nearest point"
       depends on their relative scaling — which is to say, on the aspect ratio you happen to plot at.

       Measured on a schematic isochrone with 300 stars, rescaling the **colour axis alone**:

       ==============  =============  ==============================
       colour scale    median mass    stars whose assigned mass moves
       ==============  =============  ==============================
       1.0 (baseline)  1.628          —
       2.0             1.645          **96.7%**
       5.0             1.632          98.7%
       0.5             1.628          91.3%
       ==============  =============  ==============================

       **Population statistics survive; individual masses do not.** The median moves by 1%, so a mass
       function or a mean mass computed from these is defensible. A per-star mass is not, and neither
       is anything that keys on individual assignments — mass segregation by rank, or a
       mass-matched sample.

       Three further limitations, none of which this function models:

       * **No error weighting.** A star with a 0.3 mag colour error is matched as confidently as one
         with 0.01.
       * **Bias where the isochrone folds.** Near the turn-off and along the pre-main-sequence the
         isochrone doubles back in colour, so a single CMD position corresponds to two very different
         masses and the nearest point picks one without saying so.
       * **Binaries sit above the sequence** and are assigned the mass of a more luminous single star.

    .. seealso::
       **This is not what the published NGC 6383 masses use.** P01 takes per-star masses and binary
       probabilities from **ASteCA v0.6.9**, which matches each observed star to its photometrically
       nearest *synthetic* star across 200 realizations drawn from the fitted posterior, and reports
       the median over realizations — so it carries an uncertainty and a binary probability rather
       than a point assignment. For anything published, prefer that. This function is retained for
       the legacy notebook path and for quick exploration.
    """
    if legacy_args:
        if len(legacy_args) != 6:
            raise TypeError(
                "Legacy call requires logAge_range, Zini, color_column, "
                "magnitude_column, dm, and A_V."
            )
        isochrones = first
        stars = second
        log_age_range, zini, color_column, magnitude_column, dm, av = legacy_args
        filtered = isochrones[
            (isochrones["Zini"] == zini) & (isochrones["logAge"].between(*log_age_range))
        ]
        iso_mag = filtered[magnitude_column] + dm + av
        iso_color = filtered[color_column]
        iso_mass = filtered["Mass"].values
    else:
        stars = first
        iso_mag, iso_color, iso_mass = second

    tree = KDTree(np.vstack([quantity_values(iso_color), quantity_values(iso_mag)]).T)
    star_features = np.vstack(
        [quantity_values(stars[color_column]), quantity_values(stars[magnitude_column])]
    ).T
    _, indices = tree.query(star_features)
    return QTable(
        [stars[designation_column], quantity_values(iso_mass)[indices] * u.Msun],
        names=(designation_column, "mass"),
    )


def assign_masses(isochrones, mag_column, color_column, source_id, *, k: int = 5) -> QTable:
    """Assign mean masses from the ``k`` nearest sampled isochrone points.

    Parameters mirror the old ``PUMPS_aux.assign_masses`` helper, but the
    implementation is side-effect free and returns a QTable with explicit mass
    units.
    """
    points = []
    masses = []
    # Lo que se descarta se CUENTA. Estas dos guardas -- isocrona con menos de cuatro columnas, y
    # punto no finito -- se saltaban en silencio, y esta es la funcion que produce la masa del
    # cumulo: la masa fija el radio de Jacobi, y r_J es el eje sobre el que gira todo el argumento
    # de R_t. Si nueve de diez isocronas llegan sin su columna de masa, la masa sale de UNA sola y
    # el resultado es un numero perfectamente formado. El unico caso que hoy avisa es el de cero
    # puntos, que es el que menos falta hace: ese ya revienta solo.
    descartadas = 0
    no_finitos = 0
    for iso in isochrones:
        if len(iso) < 4:
            descartadas += 1
            continue
        for mag, color, mass in zip(iso[0], iso[1], iso[3], strict=False):
            if np.isfinite(mag) and np.isfinite(color) and np.isfinite(mass):
                points.append([color, mag])
                masses.append(mass)
            else:
                no_finitos += 1
    if descartadas or no_finitos:
        warnings.warn(
            f"assign_masses ignoro {descartadas} isocrona(s) con menos de cuatro columnas y "
            f"{no_finitos} punto(s) no finito(s); la masa sale de {len(points)} punto(s) de "
            f"{len(list(isochrones))} isocrona(s). La masa fija r_J, asi que un descarte silencioso "
            f"mueve el radio de Jacobi sin mover ninguna senal.",
            UserWarning,
            stacklevel=2,
        )
    if not points:
        raise ValueError("No finite isochrone points with masses were supplied.")

    tree = KDTree(np.asarray(points, dtype=float))
    star_points = np.vstack([quantity_values(color_column), quantity_values(mag_column)]).T
    k = max(1, min(int(k), len(points)))
    _, indices = tree.query(star_points, k=k)
    indices = np.atleast_2d(indices)
    if indices.shape[0] != len(star_points):
        indices = indices.T
    mass_values = np.asarray(masses, dtype=float)
    mean_masses = np.nanmean(mass_values[indices], axis=1)
    std_masses = np.nanstd(mass_values[indices], axis=1)

    table = QTable()
    table["source_id"] = source_id
    table["mass"] = mean_masses * u.Msun
    table["mass_std"] = std_masses * u.Msun
    return table


def _classify_isochrone_form(isochrones) -> tuple[str, list]:
    """Decide which of the two isochrone layouts was supplied, or refuse.

    Returns ``(form, items)``. The materialised ``items`` list is part of the contract, not a
    convenience: this function consumes the input with ``list(...)`` in order to inspect it, so a
    caller that stores the ORIGINAL object keeps an exhausted iterator.

    The two layouts this module accepts are not distinguishable once you are
    inside a KDTree query, which is where the mistake used to surface:

    ``"single"``
        exactly three numeric 1-D arrays, read as ``(mag, color, mass)``.
    ``"samples"``
        a sequence of *sampled* isochrones, each itself a sequence of at least
        four arrays, read positionally as ``iso[0]`` = magnitude, ``iso[1]`` =
        colour, ``iso[3]`` = mass (``iso[2]`` is unused).

    The discriminator is whether the top-level entries are numeric arrays
    (``"single"``) or sequences of arrays (``"samples"``). It is deliberately
    strict about the ``"single"`` case being *exactly* three arrays: four plain
    arrays are the sampled form's inner layout leaked one level up, and reading
    them as three isochrones would silently take ``iso[3]`` -- a mass -- as a
    magnitude and return numbers that look like masses.

    Returns
    -------
    str
        ``"single"`` or ``"samples"``.

    Raises
    ------
    TypeError
        If `isochrones` is not a sequence at all.
    ValueError
        If it is empty, if the three-array form has mismatched lengths, or if
        the layout matches neither form.
    """
    try:
        items = list(isochrones)
    except TypeError as exc:
        raise TypeError(f"isochrones must be a sequence, got {type(isochrones).__name__}.") from exc
    if not items:
        raise ValueError("isochrones is empty; nothing to assign masses from.")

    def _numeric_1d(entry) -> bool:
        arr = np.asarray(quantity_values(entry) if hasattr(entry, "unit") else entry)
        return arr.ndim == 1 and arr.dtype.kind in "fiu"

    all_plain = all(_numeric_1d(entry) for entry in items)
    if all_plain:
        if len(items) != 3:
            raise ValueError(
                f"isochrones is {len(items)} plain numeric arrays, which is neither "
                "accepted form. The single-isochrone form is exactly three arrays "
                "(mag, color, mass); the sampled form is a sequence of isochrones, "
                "each itself a sequence of at least four arrays with iso[0]=mag, "
                "iso[1]=color, iso[3]=mass."
            )
        lengths = {len(np.asarray(entry)) for entry in items}
        if len(lengths) != 1:
            raise ValueError(
                f"the three isochrone arrays (mag, color, mass) have lengths {sorted(lengths)}; "
                "they must describe the same points."
            )
        return "single", items

    usable = [entry for entry in items if hasattr(entry, "__len__") and len(entry) >= 4]
    if not usable:
        raise ValueError(
            "isochrones matches neither accepted form: no entry is a sequence of at "
            "least four arrays (iso[0]=mag, iso[1]=color, iso[3]=mass), and it is not "
            "the single-isochrone form of exactly three numeric arrays "
            "(mag, color, mass)."
        )
    return "samples", items


class PhotometricMassEstimator:
    """Mass-assignment facade for sampled isochrones and CMD source tables."""

    def __init__(self, isochrones, *, k: int = 5) -> None:
        r"""Bind the isochrone set and the neighbour count.

        Parameters
        ----------
        isochrones : sequence of array-like
            One of two layouts, **classified and validated here** by
            :func:`_classify_isochrone_form` so that supplying the wrong one
            raises at construction with a message naming both forms, rather
            than somewhere inside a KDTree query:

            *Sampled* (``self.isochrone_form == "samples"``) -- a sequence of
            sampled isochrones, each indexable with at least four entries read
            positionally as ``iso[0]`` = magnitude, ``iso[1]`` = colour,
            ``iso[3]`` = mass in :math:`M_\odot` (``iso[2]`` is not used).
            Entries shorter than four are skipped by
            :func:`assign_masses`; non-finite points are dropped. Use
            :meth:`assign_from_samples`.

            *Single* (``self.isochrone_form == "single"``) -- exactly three
            numeric arrays ``(iso_mag, iso_color, iso_mass)`` of equal length.
            Use :meth:`assign_nearest`.

            Calling the method that does not match the stored form raises
            :class:`ValueError` naming the form that was given.
        k : int, default 5
            Number of nearest isochrone points averaged per star by
            :meth:`assign_from_samples`; overridable per call. Clamped to
            ``[1, n_points]``, so an over-large `k` silently becomes "average
            the whole isochrone" rather than raising. It sets what ``mass_std``
            in the output measures: the scatter of the `k` neighbours in the
            CMD, which is a **local sampling spread of the isochrone, not a
            propagated photometric uncertainty**. With ``k=1`` it is 0 by
            construction -- a real zero, because one *sample* of the isochrone
            set genuinely has no spread; contrast :meth:`assign_nearest`, whose
            ``mass_std`` is ``NaN``.

        Raises
        ------
        TypeError
            If `isochrones` is not a sequence.
        ValueError
            If `isochrones` is empty, or matches neither accepted layout.

        Warnings
        --------
        Both methods rank isochrone points by **Euclidean distance in the
        colour-magnitude plane**, which is not a physically meaningful metric:
        colour and magnitude carry different units and dynamic ranges, so
        "nearest" depends on their relative scaling. Measured on a schematic
        isochrone, rescaling the colour axis alone moves the assigned mass of
        90-99% of stars while shifting the median by about 1%. Population
        statistics survive this; **individual masses do not**. See the danger
        note on :func:`assign_mass_nearest_isochrone_point_kdtree` for the
        measurements and the further limitations.

        Notes
        -----
        The two methods take the **same arguments** and return the **same
        schema** -- ``source_id``, ``mass`` and ``mass_std``, masses as Astropy
        Quantities in :math:`M_\odot`, inputs never mutated. They differ only
        in which isochrone layout they consume and in what ``mass_std`` means.

        Until 2026-08-04 they did not: ``assign_from_samples`` took three
        positional arrays and hardcoded the output id column to ``source_id``,
        while ``assign_nearest`` took a ``QTable`` plus column-*name* strings,
        named its id column after `designation_column`, ignored `k`, and
        returned no ``mass_std`` at all -- so ``color_column`` meant an array in
        one and a string in the other, and the two outputs could not be
        concatenated. ``assign_nearest`` had no callers and no tests, so it was
        moved onto ``assign_from_samples``' contract rather than the reverse.
        """
        # Take the MATERIALISED list back from the classifier. It already does `list(...)` to
        # inspect the input, so storing the original object left any generator or iterator
        # exhausted on the instance: `assign_masses` then iterated nothing, `points` stayed
        # empty, and the call died with "No finite isochrone points with masses were supplied"
        # -- an error that blames the data for a defect in the constructor.
        self.isochrone_form, self.isochrones = _classify_isochrone_form(isochrones)
        self.k = k

    def _require_form(self, form: str, method: str) -> None:
        if self.isochrone_form != form:
            raise ValueError(
                f"{method}() needs the {form!r} isochrone form, but this estimator was "
                f"built with the {self.isochrone_form!r} form. "
                + (
                    "Pass a sequence of sampled isochrones (iso[0]=mag, iso[1]=color, "
                    "iso[3]=mass), or call assign_nearest()."
                    if form == "samples"
                    else "Pass a single (iso_mag, iso_color, iso_mass) triple, or call "
                    "assign_from_samples()."
                )
            )

    def assign_from_samples(
        self, mag_column, color_column, source_id, *, k: int | None = None
    ) -> QTable:
        r"""Assign masses averaged over the `k` nearest points of *sampled* isochrones.

        Parameters
        ----------
        mag_column, color_column : array-like
            Per-star magnitude and colour. Arrays, not column names.
        source_id : array-like
            Per-star identifier, copied unchanged into the output.
        k : int, optional
            Overrides ``self.k`` for this call.

        Returns
        -------
        QTable
            Columns ``source_id``, ``mass`` and ``mass_std``, masses in
            :math:`M_\odot`. ``mass_std`` is the standard deviation of the `k`
            neighbouring isochrone points -- a spread over the isochrone
            *samples*, not a propagated photometric error. It is exactly 0 when
            ``k = 1``, which is a true zero rather than a missing value.

        Raises
        ------
        ValueError
            If this estimator holds the single-isochrone form.
        """
        self._require_form("samples", "assign_from_samples")
        return assign_masses(
            self.isochrones,
            mag_column,
            color_column,
            source_id,
            k=self.k if k is None else k,
        )

    def assign_nearest(self, mag_column, color_column, source_id) -> QTable:
        r"""Assign the mass of the single nearest point of one isochrone.

        Same arguments and same output schema as :meth:`assign_from_samples`,
        so the two are interchangeable downstream.

        Parameters
        ----------
        mag_column, color_column : array-like
            Per-star magnitude and colour. Arrays, not column names.
        source_id : array-like
            Per-star identifier, copied unchanged into the output.

        Returns
        -------
        QTable
            Columns ``source_id``, ``mass`` and ``mass_std``, masses in
            :math:`M_\odot`. **``mass_std`` is all-NaN, deliberately.** A single
            nearest point has no spread to report, and 0.0 would assert the
            opposite of the truth -- that the mass is known exactly. NaN
            propagates through :func:`numpy.nanmean`-style aggregation as
            "unmeasured", which is what it is. Do not read it as a photometric
            uncertainty in either method: see the ``.. danger::`` note on
            :func:`assign_mass_nearest_isochrone_point_kdtree`.

        Raises
        ------
        ValueError
            If this estimator holds the sampled-isochrone form.

        Notes
        -----
        ``self.k`` is not consulted -- "the nearest point" is the definition of
        this method, so there is no neighbour count to set.
        """
        self._require_form("single", "assign_nearest")
        stars = QTable(
            {
                "source_id": np.asarray(source_id),
                "_mag": quantity_values(mag_column),
                "_color": quantity_values(color_column),
            }
        )
        table = assign_mass_nearest_isochrone_point_kdtree(
            stars,
            self.isochrones,
            magnitude_column="_mag",
            color_column="_color",
            designation_column="source_id",
        )
        table["mass_std"] = np.full(len(table), np.nan) * u.Msun
        return table


__all__ = [
    "add_photometric_errors",
    "assign_masses",
    "assign_mass_nearest_isochrone_point_kdtree",
    "PhotometricMassEstimator",
    "read_isochrones_with_metadata",
    "set_column_types",
]
