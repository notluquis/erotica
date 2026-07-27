"""Shared constants for EROTICA clustering routines."""

from __future__ import annotations

DEFAULT_PARAM_GRID = {"min_cluster_size": [5, 10, 20]}
DEFAULT_SCORE_METHOD = "relative_validity"
DEFAULT_SEARCH_METHOD = "grid"
SUPPORTED_SEARCH_METHODS = {"grid", "optuna"}

PLOT_COLOR_CYCLE = "tab10"

__all__ = [
    "DEFAULT_PARAM_GRID",
    "DEFAULT_SCORE_METHOD",
    "DEFAULT_SEARCH_METHOD",
    "SUPPORTED_SEARCH_METHODS",
    "PLOT_COLOR_CYCLE",
]
