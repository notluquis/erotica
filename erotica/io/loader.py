"""High-level data loader for EROTICA datasets."""

from __future__ import annotations

import logging

from astropy.table import QTable

from ._constants import ALIASES
from ._helpers import (
    apply_unit_corrections,
    build_available_systems,
    collect_requested_columns,
    compute_valid_source_counts,
    handle_masked_columns,
    map_requested_columns,
)


class DataLoader:
    """Load and filter EROTICA-compatible catalogues."""

    def __init__(self, file_path: str, verbose: int = logging.INFO, debug_mode: bool = False):
        self.file_path = file_path
        self.data: QTable | None = None
        self.aliases = ALIASES

        self.logger = logging.getLogger("DataLoader")
        self.logger.setLevel(verbose if not debug_mode else logging.DEBUG)

        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

        self.logger.info("DataLoader initialized with file path: %s", self.file_path)

        if debug_mode:
            self._enable_astropy_debug()

    def load_data(
        self,
        systems=None,
        *,
        include_distances=None,
        include_zp_cols: bool = False,
        include_flux_errors: bool = False,
        fidelity: str | None = None,
        file_format: str = "ascii.ecsv",
        probability: str | None = None,
        normalize_names: bool = True,
    ) -> QTable:
        """Load the table and optionally subset relevant columns."""
        include_distances = self._normalize_distance_list(include_distances)

        try:
            full = QTable.read(self.file_path, format=file_format)
        except Exception as exc:
            self.logger.error("Failed to read file: %s", exc)
            raise

        self.logger.info("File read successfully: %s", self.file_path)

        if not any(
            [
                systems,
                include_distances,
                include_zp_cols,
                include_flux_errors,
                fidelity,
                probability,
            ]
        ):
            self.data = full
            self._postprocess_loaded_table()
            return self.data

        requested = collect_requested_columns(
            systems,
            include_distances,
            include_zp_cols,
            include_flux_errors,
            fidelity,
            probability,
            full.colnames,
        )

        present_map, missing = map_requested_columns(
            requested,
            full.colnames,
            normalize_names=normalize_names,
            aliases=self.aliases,
        )
        if missing:
            self.logger.warning(
                "The following requested canonical columns were not found and will be skipped: %s",
                ", ".join(sorted(missing)),
            )

        selected_names = sorted(set(present_map.values()))
        if not selected_names:
            self.logger.warning(
                "No matching columns were found for the requested configuration; returning full table."
            )
            self.data = full
        else:
            self.data = full[selected_names].copy()

        self._postprocess_loaded_table()
        return self.data

    def count_valid_sources(self) -> dict[str, int]:
        """Summarise how many sources contain usable measurements per category."""
        table = self._require_data()
        counts = compute_valid_source_counts(table)
        for category, count in counts.items():
            self.logger.info("- %s: %s", category, count)
        return counts

    def check_available_photometric_systems(
        self, file_format: str = "ascii.ecsv"
    ) -> dict[str, dict[str, object]]:
        """Report which photometric systems are present in the source file."""
        try:
            metadata_table = QTable.read(self.file_path, format=file_format)
        except Exception as exc:
            self.logger.error("Error reading the file header: %s", exc)
            raise ValueError(f"Error reading the file header: {exc}") from exc

        self.logger.info("File successfully read for photometric system check: %s", self.file_path)
        return build_available_systems(metadata_table.colnames)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _enable_astropy_debug(self) -> None:
        astropy_logger = logging.getLogger("astropy")
        astropy_logger.setLevel(logging.DEBUG)
        if not astropy_logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            astropy_logger.addHandler(handler)
        self.logger.debug("Debugging mode enabled for DataLoader.")

    def _postprocess_loaded_table(self) -> None:
        table = self._require_data()
        handle_masked_columns(table)
        apply_unit_corrections(table, logger=self.logger)
        self.logger.info("Selected columns loaded: %s", ", ".join(table.colnames))

    def _require_data(self) -> QTable:
        if self.data is None:
            raise ValueError("Data has not been loaded. Use load_data() first.")
        return self.data

    @staticmethod
    def _normalize_distance_list(include_distances):
        if include_distances is None:
            return None
        if not isinstance(include_distances, list):
            raise TypeError("'include_distances' must be a list or None.")
        return include_distances


__all__ = ["DataLoader"]
