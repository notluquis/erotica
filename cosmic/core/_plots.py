"""Plotting helpers used by COSMIC clustering."""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.table import QTable

from ._constants import PLOT_COLOR_CYCLE
from ._style import apply_default_style


def plot_grid_search_results(cv_results) -> None:
    if not cv_results:
        raise ValueError("No CV results; run grid search first.")
    params = np.array(cv_results["param_min_cluster_size"].data, dtype=int)
    scores = np.array(cv_results["mean_test_score"], dtype=float)
    valid = ~np.isnan(scores)
    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 5), tight_layout=True)
    ax.plot(params[valid], scores[valid], marker="o", linestyle="-")
    ax.set(xlabel="min_cluster_size", ylabel="Mean Test Score", title="Grid Search Results")
    ax.grid(True)
    plt.show()


def plot_pm_scatter(
    table: QTable,
    pm_columns: Sequence[str] = ("pmra", "pmdec"),
    *,
    show_outliers: bool = False,
    clusters: Iterable[int] | None = None,
) -> None:
    df = table[list(pm_columns) + ["cluster"]].to_pandas()
    if not show_outliers:
        df = df[df["cluster"] != -1]
    if clusters is not None:
        df = df[df["cluster"].isin(list(clusters))]

    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap(PLOT_COLOR_CYCLE)
    for i, cl in enumerate(sorted(df["cluster"].unique())):
        subset = df[df["cluster"] == cl]
        ax.scatter(
            subset[pm_columns[0]],
            subset[pm_columns[1]],
            label=str(cl),
            s=40,
            alpha=0.8,
            color=cmap(i % 10),
            edgecolor="none",
        )
    ax.set(title="Proper Motion Scatter", xlabel=pm_columns[0], ylabel=pm_columns[1])
    ax.legend(title="Cluster", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def plot_probability_histogram(table: QTable) -> None:
    probs = np.array(table["probability_hdbscan"].data, dtype=float)
    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs, bins=25, histtype="stepfilled", alpha=0.7, edgecolor="k")
    ax.set(
        title="Membership Probability Distribution",
        xlabel="Probability",
        ylabel="Count",
    )
    plt.tight_layout()
    plt.show()


def plot_cluster_members(table: QTable, *, show_outliers: bool = False) -> None:
    df = table.to_pandas()
    if not show_outliers:
        df = df[df["cluster"] != -1]
    counts = df["cluster"].value_counts().sort_index()
    apply_default_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    counts.plot.bar(ax=ax, edgecolor="k")
    ax.set(title="Cluster Member Counts", xlabel="Cluster", ylabel="Count")
    plt.tight_layout()
    plt.show()


def plot_cluster_persistence(summary: pd.DataFrame) -> None:
    labels = summary["cluster"].astype(str)
    persistence = summary["persistence"].values
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    ax.bar(labels, persistence, edgecolor="k")
    ax.set(title="Cluster Persistence", xlabel="Cluster", ylabel="Persistence")
    ax.grid(axis="y")
    plt.show()


def plot_members_vs_persistence(summary: pd.DataFrame) -> None:
    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), tight_layout=True)
    counts = summary["count"].values
    persistence = summary["persistence"].values
    labels = summary["cluster"].values
    ax.scatter(counts, persistence, s=80, alpha=0.8)
    for cnt, pers, lbl in zip(counts, persistence, labels):
        ax.annotate(str(lbl), (cnt, pers), textcoords="offset points", xytext=(5, 5))
    ax.set(
        title="Cluster Persistence vs. Number of Members",
        xlabel="Number of Members",
        ylabel="Cluster Persistence",
    )
    ax.grid(True)
    plt.show()


__all__ = [
    "plot_grid_search_results",
    "plot_pm_scatter",
    "plot_probability_histogram",
    "plot_cluster_members",
    "plot_cluster_persistence",
    "plot_members_vs_persistence",
]
