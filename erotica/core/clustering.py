"""High-level clustering utilities for EROTICA."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from astropy.table import QTable
from tqdm.auto import tqdm

from ._constants import DEFAULT_SCORE_METHOD, DEFAULT_SEARCH_METHOD, SUPPORTED_SEARCH_METHODS
from ._estimator import HDBSCANEstimator
from ._plots import (
    plot_cluster_members,
    plot_cluster_persistence,
    plot_condensed_tree,
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
        legacy_cluster_selection: bool = False,
    ):
        """Construct a clustering run.

        Parameters
        ----------
        legacy_cluster_selection
            **Exists solely to reproduce results published before 2026-08-03**, and there
            is no other reason to set it. When ``True``, ``search_pseudoprobability``
            picks the reported cluster with :meth:`_cluster_label_for_size`, the defective
            size-matching selector of issue #7: it compares a count of *condensed-tree
            rows* against *flat-cluster point counts* and, when they fail to coincide,
            falls back to the largest non-noise cluster — which on a contaminated field is
            the field. The default ``False`` uses :meth:`_cluster_label_from_tree`, which
            resolves the tree node to the label its leaves actually carry.

            This is a keyword rather than a monkey-patched attribute because reproducing
            an old result must not require reaching inside the object.
        """
        if search_method not in SUPPORTED_SEARCH_METHODS:
            raise ValueError("search_method must be 'grid' or 'optuna'.")
        self.search_method = search_method
        self.data = data
        self.bad_data = bad_data
        self.storage_url = f"sqlite:///{sqlite_path}" if sqlite_path else None
        self.study_name = study_name
        self.legacy_cluster_selection = bool(legacy_cluster_selection)

        self.clusterer = None
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.combined_data = None
        self.pseudoprobability_results_ = None
        self.pseudoprobability_selected_ = None
        self.pseudoprobability_sweep_track_ = None
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
    # Pseudoprobability search
    # ------------------------------------------------------------------
    def search_pseudoprobability(
        self,
        columns: Sequence[str] = ("pmra", "pmdec"),
        *,
        min_cluster_size_samples: Iterable[int] = range(10, 300),
        min_samples: int | None = None,
        probability_threshold: float = 0.5,
        min_cluster_members: int | None = None,
        max_cluster_members: int | None = None,
        selection: str = "max_persistence",
        select_cluster: bool = True,
        hdbscan_kwargs: dict | None = None,
    ) -> None:
        """Sweep min_cluster_size, build pseudoprobability, select best cluster.

        probability_times = fraction of iterations each source was in any cluster.
        probability = probability_hdbscan * probability_times.
        Best mcs selected by ``selection`` ('max_members' or 'max_lambda').
        """
        base_kwargs = {
            "algorithm": "best",
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "metric": "euclidean",
            "match_reference_implementation": True,
            **(hdbscan_kwargs or {}),
        }
        # MST not needed during sweep — skip to save ~20% per iteration
        sweep_kwargs = {**base_kwargs, "gen_min_span_tree": False}
        final_kwargs = {**base_kwargs, "gen_min_span_tree": True}

        samples = list(min_cluster_size_samples)
        X = self.data[list(columns)].to_pandas().values
        n_sources = len(self.data)
        n_samples = len(samples)
        labels_matrix = np.full((n_sources, n_samples), -1, dtype=np.int32)
        results: list[dict] = []
        sweep_track: list[dict] = []  # all mcs → cluster size, for plotting

        for i, min_cluster_size in enumerate(tqdm(samples, desc="mcs sweep", unit="mcs")):
            estimator = HDBSCANEstimator(
                min_cluster_size=int(min_cluster_size),
                min_samples=min_samples,
                **sweep_kwargs,
            ).fit(X)
            model = estimator.model_
            labels = np.asarray(model.labels_, dtype=np.int32)
            labels_matrix[:, i] = labels

            probability_times = np.mean(labels_matrix[:, : i + 1] != -1, axis=1)
            probability = np.asarray(model.probabilities_, dtype=float) * probability_times
            tree = model.condensed_tree_.to_pandas()
            if tree.empty or "lambda_val" not in tree or "parent" not in tree:
                continue

            lambda_value = float(tree["lambda_val"].max())
            desired_len = self._desired_tree_branch_size(tree)
            sweep_track.append(
                {"min_cluster_size": int(min_cluster_size), "desired_len": int(desired_len)}
            )
            if min_cluster_members is not None and desired_len < min_cluster_members:
                continue
            if max_cluster_members is not None and desired_len > max_cluster_members:
                continue
            if len(np.unique(labels)) <= 1:
                continue
            if np.count_nonzero(probability > probability_threshold) < 1:
                continue

            # Score this sweep step by the persistence of the cluster the selector would
            # actually return, not by a count of condensed-tree rows. `cluster_persistence_`
            # is HDBSCAN's own stability measure, is indexed by flat-cluster label, and was
            # already being collected here and discarded. See `selection=` in the docstring.
            persistence = np.asarray(getattr(model, "cluster_persistence_", []), dtype=float)
            step_label = self._cluster_label_from_tree(tree, labels, len(labels))
            step_persistence = (
                float(persistence[step_label]) if 0 <= step_label < persistence.size else 0.0
            )

            results.append(
                {
                    "min_cluster_size": int(min_cluster_size),
                    "desired_len": int(desired_len),
                    "lambda_value": lambda_value,
                    "selected_label": int(step_label),
                    "selected_persistence": step_persistence,
                    "relative_validity": float(getattr(model, "relative_validity_", np.nan)),
                    "cluster_persistence": persistence,
                }
            )

        if not results:
            raise RuntimeError("Pseudo-probability search did not find candidate clusters.")

        selected = self._select_pseudoprobability_result(results, selection)
        final_probability_times = np.mean(labels_matrix != -1, axis=1)
        final_estimator = HDBSCANEstimator(
            min_cluster_size=selected["min_cluster_size"],
            min_samples=min_samples,
            **final_kwargs,
        ).fit(X)

        self.clusterer = final_estimator.model_
        self.best_params_ = {
            "min_cluster_size": selected["min_cluster_size"],
            "min_samples": min_samples,
            **final_kwargs,
        }
        self.best_score_ = selected["lambda_value"]
        self.pseudoprobability_results_ = results
        self.pseudoprobability_sweep_track_ = sweep_track
        self.pseudoprobability_selected_ = {
            **selected,
            "probability_times": final_probability_times.copy(),
        }
        self._annotate_pseudoprobability_results(
            probability_times=final_probability_times,
            desired_len=selected["desired_len"],
            probability_threshold=probability_threshold,
            select_cluster=select_cluster,
        )

    # ------------------------------------------------------------------
    # Error-aware pseudoprobability (Monte-Carlo over Gaia errors)
    # ------------------------------------------------------------------
    def search_pseudoprobability_error_aware(
        self,
        columns: Sequence[str] = ("pmra", "pmdec"),
        *,
        error_columns: Sequence[str] | None = None,
        corr_columns: dict[tuple[str, str], str] | None = None,
        n_mc: int = 100,
        min_cluster_size_samples: Iterable[int] = range(10, 300, 10),
        min_samples: int | None = None,
        random_state: int = 0,
        hdbscan_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Error-aware recovery frequency, folding Gaia astrometric errors into f_i.

        For each of ``n_mc`` Monte-Carlo draws every source is perturbed by its
        (correlated) Gaia covariance, the ``min_cluster_size`` sweep is re-run, and
        cluster membership is recorded. The returned frequency is the mean clustered
        fraction over draws x resolutions; its spread is the error-induced membership
        uncertainty. Faint sources with large parallax/PM errors flicker in and out
        across draws and earn a correctly lower, hedged frequency instead of a hard cut.

        Adds ``pFreqMC`` and ``pFreqMC_std`` columns to ``self.data`` and returns
        ``(f_mean, f_std)``. This is the frequentist, error-aware layer; the calibrated
        Bayesian posterior is a separate step (see the membership guide). The validated
        :meth:`search_pseudoprobability` is untouched.

        Note that, like the base sweep, features are used on their raw scale -- mixing
        units (parallax with PM) still requires standardizing ``columns`` beforehand.
        """
        from ._error_aware import error_aware_pseudoprobability, gaia_covariance

        cols = list(columns)
        X = self.data[cols].to_pandas().values
        cov = gaia_covariance(self.data, cols, error_columns, corr_columns)
        f_mean, f_std = error_aware_pseudoprobability(
            X,
            cov=cov,
            n_mc=n_mc,
            min_cluster_size_samples=min_cluster_size_samples,
            min_samples=min_samples,
            random_state=random_state,
            hdbscan_kwargs=hdbscan_kwargs,
        )
        self.pfreq_error_aware_ = f_mean
        self.pfreq_error_aware_std_ = f_std
        self.data["pFreqMC"] = f_mean
        self.data["pFreqMC_std"] = f_std
        return f_mean, f_std

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

    def plot_condensed_tree(
        self,
        figsize: tuple[float, float] = (8, 6),
        cmap: str = "viridis",
        select_clusters: bool = True,
        label_clusters: bool = False,
        save_path: str | None = None,
    ) -> None:
        if self.clusterer is None:
            raise RuntimeError("Run search or search_pseudoprobability first.")
        plot_condensed_tree(
            self.clusterer,
            figsize=figsize,
            cmap=cmap,
            select_clusters=select_clusters,
            label_clusters=label_clusters,
            save_path=save_path,
        )

    def plot_mcs_sweep(self, figsize=(7, 5), save_path: str | None = None) -> None:
        """Plot cluster size vs min_cluster_size from pseudoprobability sweep."""
        if self.pseudoprobability_sweep_track_ is None:
            raise RuntimeError("Run search_pseudoprobability first.")
        import matplotlib.pyplot as plt

        mcs_vals = [r["min_cluster_size"] for r in self.pseudoprobability_sweep_track_]
        sizes = [r["desired_len"] for r in self.pseudoprobability_sweep_track_]
        best_mcs = self.pseudoprobability_selected_["min_cluster_size"]
        max_size = max(sizes)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mcs_vals, sizes, color="steelblue")
        ax.axvline(
            best_mcs,
            color="steelblue",
            linestyle="--",
            label=f"Optimal Min Cluster Size: {best_mcs}",
        )
        ax.axhline(
            max_size, color="olivedrab", linestyle="--", label=f"Max Cluster Size: {max_size}"
        )
        ax.set_xlabel("Min Cluster Size")
        ax.set_ylabel("Cluster Size")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        plt.show()

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
    def _annotate_pseudoprobability_results(
        self,
        *,
        probability_times: np.ndarray,
        desired_len: int,
        probability_threshold: float,
        select_cluster: bool,
    ) -> None:
        if self.clusterer is None:
            raise RuntimeError("No clustering model available.")
        labels = np.asarray(self.clusterer.labels_, dtype=int)
        self.data["cluster_hdbscan"] = labels
        self.data["probability_hdbscan"] = self.clusterer.probabilities_
        self.data["probability_times"] = probability_times
        self.data["probability"] = (
            np.asarray(self.clusterer.probabilities_, dtype=float) * probability_times
        )
        self.data["outlier_score"] = self.clusterer.outlier_scores_

        if select_cluster:
            # Resolve the label from the condensed tree, not by matching a row count against a
            # cluster size. The size-matching path succeeded 23 times out of 83 and fell back to
            # "largest cluster" -- the field -- on the rest. See issue #7 and
            # _cluster_label_for_size's danger note. `Clustering(..., legacy_cluster_selection=True)`
            # restores the old behaviour for reproducing results published before 2026-08-03.
            if self.legacy_cluster_selection:
                selected_label = self._cluster_label_for_size(labels, desired_len)
            else:
                selected_label = self._cluster_label_from_tree(
                    self.clusterer.condensed_tree_.to_pandas(), labels, len(labels)
                )
            retained = np.asarray(self.data["probability"], dtype=float) > probability_threshold
            self.data["cluster"] = np.where((labels == selected_label) & retained, labels, -1)
            self.pseudoprobability_selected_ = {
                **self.pseudoprobability_selected_,
                "selected_cluster": selected_label,
                "probability_threshold": probability_threshold,
            }
        else:
            self.data["cluster"] = labels

        self.combined_data = combine_datasets(self.data, self.bad_data)

    @staticmethod
    def _build_pseudoprobability(labels_storage: list[list[int]]) -> np.ndarray:
        return np.array(
            [
                np.count_nonzero(np.asarray(labels, dtype=int) != -1) / len(labels)
                if labels
                else 0.0
                for labels in labels_storage
            ],
            dtype=float,
        )

    @staticmethod
    def _desired_tree_branch_size(tree) -> int:
        max_lambda_val_row = tree["lambda_val"].idxmax()
        desired_parent = tree.at[max_lambda_val_row, "parent"]
        return int(len(tree[tree["parent"] == desired_parent]))

    @staticmethod
    def _select_pseudoprobability_result(results: list[dict], selection: str) -> dict:
        """Choose which sweep step to keep.

        .. danger::
           **``"max_members"`` and ``"max_lambda"`` are both defective above ~0.8
           contamination, measured.** They are retained only to reproduce results published
           before 2026-08-04.

           ``"max_members"`` is the argmax of ``desired_len``, which is a count of
           **condensed-tree rows** — the same quantity behind issue #7. At high
           contamination it selects a ``min_cluster_size`` whose densest branch lies *inside
           the field*, and no label chosen from that branch can be right. Measured selected
           purity 0.638 against an oracle of 0.940.

           ``"max_lambda"`` is not the alternative: it chose ``mcs_range.start`` in **12 of
           12** cells, i.e. it ignores the sweep entirely and returns whatever the smallest
           ``min_cluster_size`` produced.

           ``"max_persistence"`` (the default) scores each step by
           ``cluster_persistence_`` **of the cluster the selector actually returns** —
           HDBSCAN's own stability measure for that specific cluster, rather than a proxy
           for how big something is. The value was already being collected here and thrown
           away.
        """
        if selection == "max_persistence":
            return max(
                results,
                key=lambda item: (item["selected_persistence"], item["lambda_value"]),
            )
        if selection == "max_members":
            return max(results, key=lambda item: (item["desired_len"], item["lambda_value"]))
        if selection == "max_lambda":
            return max(results, key=lambda item: (item["lambda_value"], item["desired_len"]))
        raise ValueError("selection must be 'max_persistence', 'max_members' or 'max_lambda'.")

    @staticmethod
    def _cluster_label_for_size(labels: np.ndarray, desired_len: int) -> int:
        """Match a flat cluster by size. **Kept only to reproduce pre-2026-08-03 results.**

        .. danger::
           **This is the defect measured in issue #7, and it selects the field.** ``desired_len``
           comes from :meth:`_desired_tree_branch_size`, which counts **rows of the condensed
           tree** whose parent is a given node — its immediate children, sub-clusters or falling
           points. That is not the number of points in the flat cluster. The two coincide only by
           accident: **23 matches out of 83** across the benchmark.

           On the other 60 the fallback below returns the **largest non-noise cluster**, which at
           contamination 0.8–0.95 *is the field*. Measured consequence: HDBSCAN isolates the
           cluster at ≥0.8 purity in 96% of cells, and this selector then returns purity
           0.394 ± 0.052, dragging AUC to 0.70–0.80 against pyUPMASK's 0.998.

           Use :meth:`_cluster_label_from_tree`. This is retained only because ``select_cluster``
           results produced before 2026-08-03 used it.
        """
        clusters, counts = np.unique(labels, return_counts=True)
        matching = clusters[counts == desired_len]
        if matching.size:
            return int(matching[0])
        non_noise = clusters[clusters != -1]
        if non_noise.size == 0:
            return -1
        return int(non_noise[np.argmax(counts[clusters != -1])])

    @staticmethod
    def _cluster_label_from_tree(tree, labels: np.ndarray, n_samples: int) -> int:
        """Label of the flat cluster that the densest branch of the condensed tree resolves to.

        The right question is *"which cluster does this tree node become?"*, and it is answered by
        descending the node to its leaves and reading their label — not by hoping a row count
        equals a cluster size (see :meth:`_cluster_label_for_size`).

        The starting node is the parent of the maximum-``lambda_val`` row: the branch whose points
        persist to the highest density. Descend it to leaf children (``child < n_samples`` are data
        points, anything larger is another tree node), then take the modal non-noise label.

        **The fallback is noise, deliberately.** If the branch resolves to no labelled points the
        answer is "no cluster selected", not "the largest cluster" — on a contaminated field the
        largest cluster *is* the contamination, which is precisely how the previous implementation
        failed.
        """
        if tree is None or len(tree) == 0:
            return -1
        labels = np.asarray(labels, dtype=int)
        parent = int(tree.at[tree["lambda_val"].idxmax(), "parent"])

        parents = tree["parent"].to_numpy(dtype=int)
        children = tree["child"].to_numpy(dtype=int)
        frontier, seen, leaves = [parent], set(), []
        while frontier:
            node = frontier.pop()
            if node in seen:  # condensed trees are acyclic; a malformed one would otherwise hang
                continue
            seen.add(node)
            for child in children[parents == node]:
                (leaves if child < n_samples else frontier).append(int(child))

        if not leaves:
            return -1
        member_labels = labels[np.asarray(leaves, dtype=int)]
        member_labels = member_labels[member_labels != -1]
        if member_labels.size == 0:
            return -1
        values, counts = np.unique(member_labels, return_counts=True)
        return int(values[np.argmax(counts)])

    def _annotate_results(self) -> None:
        if self.clusterer is None:
            raise RuntimeError("No clustering model available.")
        self.data["cluster"] = self.clusterer.labels_
        self.data["probability_hdbscan"] = self.clusterer.probabilities_
        self.combined_data = combine_datasets(self.data, self.bad_data)


__all__ = ["Clustering", "HDBSCANEstimator"]
