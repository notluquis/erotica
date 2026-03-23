"""High-level utilities for preprocessing COSMIC catalogues."""

from __future__ import annotations

import logging
import warnings
from typing import Sequence

from astropy.table import QTable
from zero_point import zpt

from ._constants import COLUMN_MAPPING, DEFAULT_DROP_COLUMNS
from ._helpers import (
    add_photometric_errors,
    apply_zero_point_correction,
    correct_proper_motion,
    drop_invalid_sources,
    fill_missing_values,
    rename_columns,
    split_by_fidelity,
)

warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite' in 1.6 and will be removed in 1.8.",
    category=FutureWarning,
    module="sklearn.utils.deprecation",
)


class DataPreprocessor:
    """Wrap preprocessing steps commonly applied to COSMIC-ready tables."""

    def __init__(
        self,
        data: QTable,
        *,
        zero_point_module=zpt,
        logger: logging.Logger | None = None,
    ):
        self.data = data
        self.total_count = len(data)
        self._logger = logger
        self.zero_point_module = zero_point_module

        if hasattr(self.zero_point_module, "load_tables"):
            try:
                self.zero_point_module.load_tables()
            except Exception:
                # Defer loading errors to the first correction attempt.
                pass

        if self._logger is None:
            self._logger = logging.getLogger("DataPreprocessor")
            if not self._logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter("%(message)s"))
                self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    # ------------------------------------------------------------------
    # Basic sanitation steps
    # ------------------------------------------------------------------
    def fill_missing_values(self) -> int:
        """Fill masked values with NaNs and return the number of columns touched."""
        count = fill_missing_values(self.data)
        self._logger.info("Filled missing values in %s masked column(s).", count)
        return count

    def rename_columns(self, mapping: dict[str, str] | None = None) -> list[str]:
        """Rename photometric columns following the canonical COSMIC schema."""
        renamed = rename_columns(self.data, mapping or COLUMN_MAPPING)
        if renamed:
            self._logger.info("Renamed columns: %s.", ", ".join(renamed))
        else:
            self._logger.info("No columns required renaming.")
        return renamed

    def apply_zero_point_correction(self) -> dict[str, object]:
        """Apply Gaia zero-point parallax corrections."""
        result = apply_zero_point_correction(self.data, self.zero_point_module, logger=self._logger)
        if result.get("applied"):
            self._logger.info("Applied zero-point correction to parallax values.")
        return result

    def correct_proper_motion(self) -> None:
        """Transform proper motions into the ICRF frame."""
        correct_proper_motion(self.data)
        self._logger.info("Transformed proper motion to ICRF frame.")

    def add_photometric_errors(self) -> list[str]:
        """Compute magnitude uncertainties from flux measurements."""
        created = add_photometric_errors(self.data)
        self._logger.info("Added photometric errors (%s).", ", ".join(created))
        return created

    def drop_invalid_sources(self, columns_to_check: Sequence[str] | None = None) -> int:
        """Remove rows containing non-finite values in the selected columns."""
        columns_used = (
            list(columns_to_check) if columns_to_check is not None else DEFAULT_DROP_COLUMNS
        )
        filtered, invalid = drop_invalid_sources(self.data, columns_used)
        self.data = filtered
        self._logger.info(
            "Dropped %s sources with invalid values in %s.",
            invalid,
            columns_used,
        )
        return invalid

    def filter_data(
        self,
        fidelity_column: str = "fidelity_v2",
        fidelity_threshold: float = 0.5,
    ) -> tuple[QTable, QTable]:
        """Split the table into high and low fidelity subsets."""
        good, bad, stats = split_by_fidelity(self.data, fidelity_column, fidelity_threshold)
        pct = int(round(stats["good_fraction"] * 100))
        self._logger.info(
            "Filtered data with fidelity threshold %.2f: %s high, %s low (%s%% good).",
            fidelity_threshold,
            stats["high_fidelity"],
            stats["low_fidelity"],
            pct,
        )
        self._logger.info(
            "High-fidelity sources with 2MASS data: %s | Low-fidelity: %s",
            stats["tmass_good"],
            stats["tmass_bad"],
        )
        self._logger.info(
            "High-fidelity sources with WISE data: %s | Low-fidelity: %s",
            stats["wise_good"],
            stats["wise_bad"],
        )
        return good, bad


__all__ = ["DataPreprocessor"]
