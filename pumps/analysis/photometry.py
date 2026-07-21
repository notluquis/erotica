"""Photometric helpers and legacy isochrone utilities."""

from __future__ import annotations

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

    e_g = np.where(g < 13, 0.3, np.where(g < 17, interp(g, 13, 17, 0.3, 1), np.where(g <= 20, interp(g, 17, 20, 1, 6), 6))) / 1000
    e_bp = np.where(g < 13, 0.9, np.where(g < 17, interp(g, 13, 17, 0.9, 12), np.where(g <= 20, interp(g, 17, 20, 12, 108), 108))) / 1000
    e_rp = np.where(g < 13, 0.6, np.where(g < 17, interp(g, 13, 17, 0.6, 6), np.where(g <= 20, interp(g, 17, 20, 6, 52), 52))) / 1000
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
    """Assign masses from nearest isochrone point in CMD space.

    New API: ``(stars, (iso_mag, iso_color, iso_mass), ...)``.
    Legacy API: ``(isochrones_df, stars, logAge_range, Zini, color_col,
    mag_col, dm, A_V)``.
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
    for iso in isochrones:
        if len(iso) < 4:
            continue
        for mag, color, mass in zip(iso[0], iso[1], iso[3]):
            if np.isfinite(mag) and np.isfinite(color) and np.isfinite(mass):
                points.append([color, mag])
                masses.append(mass)
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


class PhotometricMassEstimator:
    """Mass-assignment facade for sampled isochrones and CMD source tables."""

    def __init__(self, isochrones, *, k: int = 5) -> None:
        self.isochrones = isochrones
        self.k = k

    def assign_from_samples(self, mag_column, color_column, source_id, *, k: int | None = None) -> QTable:
        """Assign masses from sampled legacy isochrone arrays."""
        return assign_masses(
            self.isochrones,
            mag_column,
            color_column,
            source_id,
            k=self.k if k is None else k,
        )

    def assign_nearest(
        self,
        stars: QTable,
        *,
        magnitude_column: str = "Gmag",
        color_column: str = "BP_RP",
        designation_column: str = "designation",
    ) -> QTable:
        """Assign masses from a single ``(mag, color, mass)`` isochrone tuple."""
        return assign_mass_nearest_isochrone_point_kdtree(
            stars,
            self.isochrones,
            magnitude_column=magnitude_column,
            color_column=color_column,
            designation_column=designation_column,
        )


__all__ = [
    "add_photometric_errors",
    "assign_masses",
    "assign_mass_nearest_isochrone_point_kdtree",
    "PhotometricMassEstimator",
    "read_isochrones_with_metadata",
    "set_column_types",
]
