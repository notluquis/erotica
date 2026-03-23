"""High-level clustering utilities for COSMIC."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from astropy.table import QTable

from ._constants import DEFAULT_SCORE_METHOD, DEFAULT_SEARCH_METHOD, SUPPORTED_SEARCH_METHODS
from ._estimator import HDBSCANEstimator
from ._plots import (
    plot_cluster_members,
    plot_cluster_persistence,
    plot_grid_search_results,
    plot_members_vs_persistence,
    plot_pm_scatter,
    plot_probability_histogram,
)
from ._search import run_grid_search, run_optuna_search
from ._style import apply_default_style
from ._summary import build_cluster_summary, clustering_statistics, combine_datasets


class Clustering:
    """Perform HDBSCAN clustering with optional hyper-parameter search."""

    def __init__(
        self,
        data: QTable,
        bad_data: QTable | None = None,
        *,
        search_method: str = DEFAULT_SEARCH_METHOD,
        sqlite_path: str = "optuna_study.db",
        study_name: str | None = None,
    ):
        if search_method not in SUPPORTED_SEARCH_METHODS:
            raise ValueError("search_method must be 'grid' or 'optuna'.")
        self.search_method = search_method
        self.data = data
        self.bad_data = bad_data
        self.storage_url = f"sqlite:///{sqlite_path}" if sqlite_path else None
        self.study_name = study_name

        self.clusterer = None
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.combined_data = None
        self._study = None
        self._pareto_trials = []

        apply_default_style()

    # ------------------------------------------------------------------
    # Main search routine
    # ------------------------------------------------------------------
    def search(
        self,
        columns: Sequence[str],
        *,
        persistence_threshold: float = 0.0,
        param_grid: dict[str, list] | None = None,
        grid_kwargs: dict | None = None,
        optuna_search_space: dict[str, dict] | None = None,
        n_trials: int = 50,
        n_jobs: int = -1,
        sampler: str = "TPESampler",
        sampler_kwargs: dict | None = None,
        score_method: str | Iterable[str] = DEFAULT_SCORE_METHOD,
        hdbscan_kwargs: dict | None = None,
    ) -> None:
        """Run the requested hyper-parameter search and annotate results."""
        hdbscan_kwargs = hdbscan_kwargs or {}
        X = self.data[list(columns)].to_pandas().values

        if self.search_method == "grid":
            results = run_grid_search(
                X,
                persistence_threshold=persistence_threshold,
                param_grid=param_grid,
                grid_kwargs=grid_kwargs,
                hdbscan_kwargs=hdbscan_kwargs,
            )
            self.cv_results_ = results["cv_results"]
        else:
            methods = [score_method] if isinstance(score_method, str) else list(score_method)
            results = run_optuna_search(
                X,
                persistence_threshold=persistence_threshold,
                search_space=optuna_search_space,
                n_trials=n_trials,
                n_jobs=n_jobs,
                sampler_name=sampler,
                sampler_kwargs=sampler_kwargs,
                score_methods=methods,
                hdbscan_kwargs=hdbscan_kwargs,
                storage_url=self.storage_url,
                study_name=self.study_name,
            )
            self._study = results["study"]
            self._pareto_trials = results.get("pareto", [])

        self.clusterer = results["clusterer"]
        self.best_params_ = results.get("best_params")
        self.best_score_ = results.get("best_score")

        self._annotate_results()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def show_results(self) -> None:
        if self._study is None:
            print("Run .search(...) first.")
            return
        trial = self._pareto_trials[0] if self._pareto_trials else self._study.best_trial
        print("Params:", trial.params)
        if hasattr(trial, "values"):
            print("Objectives:", tuple(f"{v:.6f}" for v in trial.values))
        else:
            print("Objective:", f"{trial.value:.6f}")
        if self._pareto_trials:
            print(f"Pareto front size: {len(self._pareto_trials)}")

    def get_best_params(self) -> dict | None:
        return self.best_params_

    def save_results(self, filename: str, format: str = "csv") -> None:
        if self.combined_data is None:
            raise ValueError("No results to save; run search first.")
        self.combined_data.write(filename, format=format, overwrite=True)

    def clustering_statistics(self, show_outliers: bool = False) -> None:
        table = self.combined_data or self.data
        stats = clustering_statistics(table, include_outliers=show_outliers)
        if stats["clusters"] == 0 and not np.isfinite(stats["mean"]):
            print("No clusters found")
            return
        print("Clustering Statistics:", stats)

    # ------------------------------------------------------------------
    # Plot wrappers
    # ------------------------------------------------------------------
    def plot_grid_search_results(self) -> None:
        if not self.cv_results_:
            raise ValueError("No CV results; run grid search first.")
        plot_grid_search_results(self.cv_results_)

    def plot_pm_scatter(self, pm_columns=("pmra", "pmdec"), show_outliers=False, clusters=None):
        table = self.data.copy()
        plot_pm_scatter(
            table,
            pm_columns=pm_columns,
            show_outliers=show_outliers,
            clusters=clusters,
        )

    def plot_probability_histogram(self) -> None:
        table = self.data if self.combined_data is None else self.combined_data
        plot_probability_histogram(table)

    def plot_cluster_members(self, show_outliers=False) -> None:
        table = self.combined_data or self.data
        plot_cluster_members(table, show_outliers=show_outliers)

    def plot_cluster_persistence(self) -> None:
        summary = self.get_cluster_summary(include_noise=True)
        plot_cluster_persistence(summary)

    def plot_members_vs_persistence(self, show_outliers: bool = False) -> None:
        summary = self.get_cluster_summary(include_noise=show_outliers)
        plot_members_vs_persistence(summary)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def get_cluster_summary(
        self,
        pm_columns: Sequence[str] = ("pmra", "pmdec"),
        include_noise: bool = True,
    ):
        table = self.combined_data or self.data
        if table is None:
            raise ValueError("No data available. Run search() or assign .data first.")
        persistence = np.asarray(
            getattr(getattr(self, "clusterer", None), "cluster_persistence_", []),
            dtype=float,
        )
        return build_cluster_summary(
            table,
            pm_columns=pm_columns,
            include_noise=include_noise,
            persistence_array=persistence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _annotate_results(self) -> None:
        if self.clusterer is None:
            raise RuntimeError("No clustering model available.")
        self.data["cluster"] = self.clusterer.labels_
        self.data["probability_hdbscan"] = self.clusterer.probabilities_
        self.combined_data = combine_datasets(self.data, self.bad_data)


__all__ = ["Clustering", "HDBSCANEstimator"]
