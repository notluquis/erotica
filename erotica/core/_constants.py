"""Shared constants for EROTICA clustering routines."""

from __future__ import annotations

DEFAULT_PARAM_GRID = {"min_cluster_size": [5, 10, 20]}
# Optuna counterpart of DEFAULT_PARAM_GRID. Without a default the Optuna path suggests no
# hyper-parameters at all, so every trial calls HDBSCAN with min_cluster_size=None and raises
# "Min samples and min cluster size must be integers!".
DEFAULT_SEARCH_SPACE = {"min_cluster_size": {"low": 5, "high": 100, "type": "int"}}
# Seed the Optuna sampler by default so repeating a search reproduces it. Callers can override
# with sampler_kwargs={"seed": ...}; pass seed=None explicitly for a non-deterministic study.
DEFAULT_SAMPLER_SEED = 0
DEFAULT_SCORE_METHOD = "relative_validity"
DEFAULT_SEARCH_METHOD = "grid"
SUPPORTED_SEARCH_METHODS = {"grid", "optuna"}

PLOT_COLOR_CYCLE = "tab10"

__all__ = [
    "DEFAULT_PARAM_GRID",
    "DEFAULT_SEARCH_SPACE",
    "DEFAULT_SAMPLER_SEED",
    "DEFAULT_SCORE_METHOD",
    "DEFAULT_SEARCH_METHOD",
    "SUPPORTED_SEARCH_METHODS",
    "PLOT_COLOR_CYCLE",
]
