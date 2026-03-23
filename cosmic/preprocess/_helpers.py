"""Helper functions for COSMIC data preprocessing."""

from __future__ import annotations

import warnings
from typing import Iterable, Sequence

import astropy.units as u
import numpy as np
from astropy.table import QTable

from ._constants import (
    COLUMN_MAPPING,
    DEFAULT_DROP_COLUMNS,
    PM_CORRECTION_ROWS,
    PSEUDOCOLOUR_RANGE,
    ZERO_POINT_REQUIRED_COLUMNS,
)


def fill_missing_values(table: QTable) -> int:
    """Replace masked values with NaN, casting integer columns to float when needed."""
    updated = 0
    for column_name in table.colnames:
        column = table[column_name]
        if hasattr(column, "mask"):
            if np.issubdtype(column.dtype, np.integer):
                table[column_name] = column.astype(float)
                column = table[column_name]
            table[column_name] = column.filled(np.nan)
            updated += 1
    return updated


def rename_columns(table: QTable, mapping: dict[str, str] | None = None) -> list[str]:
    """Rename columns according to *mapping* (defaults to :data:`COLUMN_MAPPING`)."""
    mapping = mapping or COLUMN_MAPPING
    renamed: list[str] = []
    for old_name, new_name in mapping.items():
        if old_name in table.colnames:
            table.rename_column(old_name, new_name)
            renamed.append(new_name)
    return renamed


def apply_zero_point_correction(table: QTable, zpt_module, *, logger=None) -> dict[str, object]:
    """Apply Gaia zero-point correction to the parallax column in-place."""
    missing = [col for col in ZERO_POINT_REQUIRED_COLUMNS if col not in table.colnames]
    if missing:
        message = (
            "Missing columns for zero-point correction: "
            + ", ".join(missing)
            + ". Skipping correction."
        )
        if logger:
            logger.warning(message)
        else:
            warnings.warn(message, UserWarning)
        return {"applied": False, "missing_columns": missing}

    pseudocolour = table["pseudocolour"]
    lower, upper = PSEUDOCOLOUR_RANGE
    out_of_range = (pseudocolour < lower) | (pseudocolour > upper)
    out_count = int(np.sum(out_of_range))
    if out_count:
        message = (
            f"{out_count} source(s) have pseudocolour values outside the expected range "
            f"{lower} - {upper}. Maximum corrections are applied at the range boundaries."
        )
        if logger:
            logger.warning(message)
        else:
            warnings.warn(message, UserWarning)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        table["parallax_observed"] = table["parallax"]
        corrections = (
            zpt_module.get_zpt(
                table["Gmag"],
                table["nu_eff_used_in_astrometry"],
                table["pseudocolour"],
                table["ecl_lat"],
                table["astrometric_params_solved"],
            )
            * u.mas
        )

    table["zpvals"] = np.ma.masked_invalid(corrections).filled(0)
    table["parallax"] = table["parallax_observed"] - table["zpvals"]
    return {"applied": True, "out_of_range": out_count}


def correct_proper_motion(table: QTable) -> None:
    """Transform proper motions to the ICRF frame using the Gaia DR3 table."""
    required = {"Gmag", "ra", "dec", "pmra", "pmdec"}
    missing = required - set(table.colnames)
    if missing:
        raise ValueError(
            f"Columns required for proper motion correction are missing: {sorted(missing)}"
        )

    pm_table = np.array(PM_CORRECTION_ROWS).T
    gmin, gmax, omega_x, omega_y, omega_z = pm_table

    G = table["Gmag"]
    ra = table["ra"]
    dec = table["dec"]

    pmra = table["pmra"]
    pmdec = table["pmdec"]

    pmra_corr = np.zeros(len(G))
    pmdec_corr = np.zeros(len(G))

    mask = G < 13 * u.mag
    indices = np.where(mask)[0]

    for idx in indices:
        g_val = G[idx].to_value(u.mag)
        candidates = np.where((gmin <= g_val) & (gmax > g_val))[0]
        if candidates.size == 0:
            continue
        j = candidates[0]
        ox, oy, oz = omega_x[j], omega_y[j], omega_z[j]
        ra_val = _to_radians(ra[idx])
        dec_val = _to_radians(dec[idx])
        pmra_corr[idx] = (
            -np.sin(dec_val) * np.cos(ra_val) * ox
            - np.sin(dec_val) * np.sin(ra_val) * oy
            + np.cos(dec_val) * oz
        )
        pmdec_corr[idx] = np.sin(ra_val) * ox - np.cos(ra_val) * oy

    table["pmra_obs"] = pmra
    table["pmdec_obs"] = pmdec
    table["pmra"] = pmra - pmra_corr / 1000.0 * (u.mas / u.yr)
    table["pmdec"] = pmdec - pmdec_corr / 1000.0 * (u.mas / u.yr)


def add_photometric_errors(table: QTable) -> list[str]:
    """Compute magnitude errors from flux measurements."""
    required = [
        "phot_g_mean_flux",
        "phot_g_mean_flux_error",
        "phot_bp_mean_flux",
        "phot_bp_mean_flux_error",
        "phot_rp_mean_flux",
        "phot_rp_mean_flux_error",
    ]
    missing = [col for col in required if col not in table.colnames]
    if missing:
        raise ValueError(f"Columns required for photometric errors are missing: {missing}")

    def _calculate_mag_error(flux, flux_error):
        flux_qty = flux.quantity if hasattr(flux, "quantity") else flux
        flux_err_qty = flux_error.quantity if hasattr(flux_error, "quantity") else flux_error
        flux_vals = flux_qty.to(u.electron / u.s).value
        flux_err_vals = flux_err_qty.to(u.electron / u.s).value
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = flux_err_vals / flux_vals
            ratio = np.clip(ratio, 0, None)
            return 2.5 * np.log10(1 + ratio) * u.mag

    table["e_Gmag"] = _calculate_mag_error(
        table["phot_g_mean_flux"], table["phot_g_mean_flux_error"]
    )
    table["e_G_BPmag"] = _calculate_mag_error(
        table["phot_bp_mean_flux"], table["phot_bp_mean_flux_error"]
    )
    table["e_G_RPmag"] = _calculate_mag_error(
        table["phot_rp_mean_flux"], table["phot_rp_mean_flux_error"]
    )
    table["e_BP_RP"] = np.sqrt(table["e_G_BPmag"] ** 2 + table["e_G_RPmag"] ** 2)
    return ["e_Gmag", "e_G_BPmag", "e_G_RPmag", "e_BP_RP"]


def drop_invalid_sources(
    table: QTable, columns_to_check: Sequence[str] | None = None
) -> tuple[QTable, int]:
    """Drop rows containing non-finite values in the selected columns."""
    columns = list(columns_to_check) if columns_to_check is not None else list(DEFAULT_DROP_COLUMNS)
    missing = [col for col in columns if col not in table.colnames]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    mask = np.ones(len(table), dtype=bool)
    for column in columns:
        values = _as_float_array(table[column])
        mask &= np.isfinite(values)

    invalid_count = int(len(table) - np.sum(mask))
    filtered = table[mask]
    return filtered, invalid_count


def split_by_fidelity(
    table: QTable,
    fidelity_column: str = "fidelity_v2",
    fidelity_threshold: float = 0.5,
) -> tuple[QTable, QTable, dict[str, int]]:
    """Split the table into high- and low-fidelity subsets and report statistics."""
    if fidelity_column not in table.colnames:
        raise ValueError(f"Column '{fidelity_column}' not found in the data.")

    high_mask = table[fidelity_column] > fidelity_threshold
    low_mask = ~high_mask

    good = table[high_mask]
    bad = table[low_mask]

    total = max(len(table), 1)
    stats = {
        "high_fidelity": int(len(good)),
        "low_fidelity": int(len(bad)),
        "good_fraction": float(len(good) / total),
        "tmass_good": _count_string_non_nan(good, "tmass_designation"),
        "tmass_bad": _count_string_non_nan(bad, "tmass_designation"),
        "wise_good": _count_numeric_non_nan(good, "allwise_oid"),
        "wise_bad": _count_numeric_non_nan(bad, "allwise_oid"),
    }

    return good, bad, stats


def _count_string_non_nan(table: QTable, column: str) -> int:
    if column not in table.colnames:
        return 0
    data = table[column].astype(str)
    mask = (data != "nan") & (data != "")
    return int(np.sum(mask))


def _count_numeric_non_nan(table: QTable, column: str) -> int:
    if column not in table.colnames:
        return 0
    values = _as_float_array(table[column])
    return int(np.sum(np.isfinite(values)))


def _as_float_array(column) -> np.ndarray:
    if hasattr(column, "quantity"):
        data = column.quantity.to_value()
    elif hasattr(column, "to_value"):
        data = column.to_value()
    else:
        data = column
    try:
        array = np.asarray(data, dtype=float)
    except Exception:
        array = np.asarray(data)
        try:
            array = array.astype(float)
        except Exception:
            return np.full(np.size(array), np.nan)
    if np.ma.isMaskedArray(array):
        array = array.filled(np.nan)
    return array


def _to_radians(value) -> float:
    if hasattr(value, "to"):
        return float(value.to(u.rad).value)
    return float(np.radians(value))


__all__ = [
    "fill_missing_values",
    "rename_columns",
    "apply_zero_point_correction",
    "correct_proper_motion",
    "add_photometric_errors",
    "drop_invalid_sources",
    "split_by_fidelity",
]
