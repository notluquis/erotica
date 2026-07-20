"""Helper utilities supporting the COSMIC data loader."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from astropy.table import QTable

from ._constants import (
    ALIASES,
    FLUX_ERROR_COLUMNS,
    GAIA_DISTANCE_COLUMNS,
    GAIA_PHOTOMETRY_COLUMNS,
    PHOTOMETRIC_SYSTEMS,
    PROPER_MOTION_COLUMNS,
    TMASS_PHOTOMETRY_COLUMNS,
    UNIT_CORRECTIONS,
    WISE_PHOTOMETRY_COLUMNS,
    ZP_COLUMNS,
)

AliasMap = Mapping[str, Iterable[str]]

# Integer identifier columns (e.g. Gaia ``source_id``, WISE ``allwise_oid``) hold
# 64-bit values that exceed 2**53 and therefore cannot be represented exactly as
# float64.  They must never be promoted to float when unmasking, otherwise the
# identifiers are silently corrupted and every downstream crossmatch breaks.
_IDENTIFIER_COLUMNS = frozenset({"source_id", "allwise_oid"})


def _is_identifier_column(name: str) -> bool:
    """Return ``True`` when *name* denotes an integer identifier column."""
    lowered = name.lower()
    return name in _IDENTIFIER_COLUMNS or lowered.endswith(("_id", "_oid"))


def resolve_alias(
    available_cols: Iterable[str], canonical_name: str, aliases: AliasMap | None = None
) -> str | None:
    """Return the matching column name in *available_cols* for the canonical name."""
    alias_map = aliases or ALIASES
    candidates = alias_map.get(canonical_name, {canonical_name})
    for candidate in candidates:
        if candidate in available_cols:
            return candidate
    return None


def map_requested_columns(
    requested: Iterable[str],
    available: Sequence[str],
    *,
    normalize_names: bool = True,
    aliases: AliasMap | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Map canonical column names to actual names present in *available*.

    Returns a tuple ``(present_map, missing)`` where ``present_map`` maps canonical
    names to the concrete column name in the table, and ``missing`` lists the
    canonical names that could not be located.
    """
    available_set = set(available)
    present: dict[str, str] = {}
    missing: list[str] = []
    for canonical in requested:
        if normalize_names:
            source = resolve_alias(available_set, canonical, aliases=aliases)
        else:
            source = canonical if canonical in available_set else None
        if source is None:
            missing.append(canonical)
        else:
            present[canonical] = source
    return present, missing


def collect_requested_columns(
    systems: Sequence[str] | None,
    include_distances: Sequence[str] | None,
    include_zp_cols: bool,
    include_flux_errors: bool,
    fidelity: str | None,
    probability: str | None,
    available: Sequence[str],
) -> set[str]:
    """Construct the canonical column names that should be extracted."""
    available_set = set(available)
    requested: set[str] = set()

    if systems:
        for system in systems:
            if system not in PHOTOMETRIC_SYSTEMS:
                raise ValueError(f"Unknown photometric system: {system}")
            requested.update(PHOTOMETRIC_SYSTEMS[system])

    if include_distances:
        for distance_key in include_distances:
            if distance_key not in GAIA_DISTANCE_COLUMNS:
                raise ValueError(f"Unknown distance type: {distance_key}")
            requested.update(GAIA_DISTANCE_COLUMNS[distance_key])

    if include_zp_cols:
        requested.update(ZP_COLUMNS)

    if include_flux_errors:
        requested.update(FLUX_ERROR_COLUMNS)

    if fidelity and fidelity in available_set:
        requested.add(fidelity)

    if probability and probability in available_set:
        requested.add(probability)

    return requested


def handle_masked_columns(table: QTable) -> None:
    """Replace masked values by NaN, promoting integer columns to float when needed.

    Integer *identifier* columns (e.g. ``source_id``) are exempt from the float
    promotion: their 64-bit values exceed 2**53 and would be silently corrupted
    by a float64 cast.  Such columns keep integer dtype when nothing is masked,
    or become ``object`` (integers preserved exactly, missing entries set to NaN)
    when some values are masked.
    """
    for column_name in table.colnames:
        column = table[column_name]
        if not hasattr(column, "mask"):
            continue
        if _is_identifier_column(column_name) and np.issubdtype(column.dtype, np.integer):
            if np.any(column.mask):
                table[column_name] = column.astype(object).filled(np.nan)
            else:
                table[column_name] = column.filled()
            continue
        if np.issubdtype(column.dtype, np.integer):
            table[column_name] = column.astype(float)
            column = table[column_name]
        table[column_name] = column.filled(np.nan)


def apply_unit_corrections(table: QTable, *, logger=None) -> None:
    """Apply known unit corrections in-place."""
    for column_name in table.colnames:
        column = table[column_name]
        if hasattr(column, "unit"):
            unit_key = str(column.unit)
            if unit_key in UNIT_CORRECTIONS:
                corrected_unit = UNIT_CORRECTIONS[unit_key]
                table[column_name] = column.value * corrected_unit
                if logger is not None:
                    logger.info(
                        "Updated units for column '%s' to %s.",
                        column_name,
                        corrected_unit,
                    )


def compute_valid_source_counts(table: QTable) -> dict[str, int]:
    """Return counts of valid entries per photometric category."""
    counts: dict[str, int] = {}

    # ``source_id`` is a 64-bit integer identifier and is no longer promoted to
    # float64, so count present values via the dtype-agnostic helper (which also
    # handles the ``object``/NaN representation used when some IDs are masked)
    # instead of ``np.isnan`` (which raises on object/integer dtypes).
    counts["Gaia IDs"] = _count_single_column(table, "source_id")

    counts["Gaia Photometry"] = _count_any_finite(table, GAIA_PHOTOMETRY_COLUMNS)
    counts["Gaia Parallaxes"] = _count_single_column(table, "parallax")
    counts["Gaia Proper Motions"] = _count_any_finite(table, PROPER_MOTION_COLUMNS)

    if "tmass_designation" in table.colnames:
        tmass_column = table["tmass_designation"].astype(str)
        counts["TMASS IDs"] = int(np.sum((tmass_column != "nan") & (tmass_column != "")))
    else:
        counts["TMASS IDs"] = 0

    counts["TMASS Photometry"] = _count_any_finite(table, TMASS_PHOTOMETRY_COLUMNS)
    counts["WISE IDs"] = _count_single_column(table, "allwise_oid")
    counts["WISE Photometry"] = _count_any_finite(table, WISE_PHOTOMETRY_COLUMNS)

    return counts


def build_available_systems(
    column_names: Sequence[str],
) -> dict[str, dict[str, object]]:
    """Summarise which photometric systems are fully available in *column_names*."""
    available = {}
    column_set = set(column_names)
    for system, required_columns in PHOTOMETRIC_SYSTEMS.items():
        present = [col for col in required_columns if col in column_set]
        available[system] = {
            "available": len(present) == len(required_columns),
            "columns": present,
        }
    return available


def _count_single_column(table: QTable, column_name: str) -> int:
    if column_name not in table.colnames:
        return 0
    column = table[column_name]
    array = _as_numeric_array(column)
    if array is None:
        # fall back to counting non-empty strings
        try:
            string_array = np.asarray(column, dtype=str)
        except Exception:
            return 0
        return int(np.sum(string_array != ""))
    return int(np.sum(np.isfinite(array)))


def _count_any_finite(table: QTable, column_names: Sequence[str]) -> int:
    present_columns = [name for name in column_names if name in table.colnames]
    if not present_columns:
        return 0
    finite_masks = []
    for name in present_columns:
        array = _as_numeric_array(table[name])
        if array is None:
            continue
        finite_masks.append(np.isfinite(array))
    if not finite_masks:
        return 0
    combined = np.any(finite_masks, axis=0)
    return int(np.sum(combined))


def _as_numeric_array(column) -> np.ndarray | None:
    if hasattr(column, "unit"):
        return np.asarray(column.value, dtype=float)
    try:
        return np.asarray(column, dtype=float)
    except Exception:
        return None


__all__ = [
    "resolve_alias",
    "map_requested_columns",
    "collect_requested_columns",
    "handle_masked_columns",
    "apply_unit_corrections",
    "compute_valid_source_counts",
    "build_available_systems",
]
