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


def _cluster_colors(n: int) -> list:
    """Return n visually distinct colors, combining tab20 palettes."""
    if n <= 10:
        return [plt.get_cmap("tab10")(i / 10) for i in range(n)]
    if n <= 20:
        return [plt.get_cmap("tab20")(i / 20) for i in range(n)]
    # More than 20: interleave tab20, tab20b, tab20c
    cmaps = [plt.get_cmap("tab20"), plt.get_cmap("tab20b"), plt.get_cmap("tab20c")]
    return [cmaps[i % 3]((i // 3) / 20) for i in range(n)]


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

    unique_clusters = sorted(df["cluster"].unique())
    n = len(unique_clusters)
    colors = _cluster_colors(n)

    apply_default_style()
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for color, cl in zip(colors, unique_clusters):
        subset = df[df["cluster"] == cl]
        ax.scatter(
            subset[pm_columns[0]],
            subset[pm_columns[1]],
            label=str(cl),
            s=20,
            alpha=0.7,
            color=color,
            edgecolor="none",
        )
    ax.set(title="Proper Motion Scatter", xlabel=pm_columns[0], ylabel=pm_columns[1])

    ncols = max(1, n // 15)
    ax.legend(
        title="Cluster",
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        ncols=ncols,
        fontsize=7,
        title_fontsize=8,
        markerscale=1.5,
        handlelength=1,
    )
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


def plot_condensed_tree(
    clusterer,
    *,
    figsize: tuple[float, float] = (8, 6),
    cmap: str = "viridis",
    select_clusters: bool = True,
    label_clusters: bool = False,
    save_path: str | None = None,
) -> None:
    """Plot condensed cluster tree (icicle plot) from HDBSCAN.

    Delegates bars/lines/colorbar/axis styling to hdbscan's CondensedTree.plot()
    (with select_clusters=False), then overlays cluster ellipses ourselves.
    Workaround for hdbscan bug: np.diff(np.percentile(..., q=[10,90])) returns a
    shape-(1,) array that propagates into Ellipse width/height and crashes
    Affine2D.scale() in matplotlib. Fix: explicit float() on all Ellipse arguments.
    """
    from matplotlib.patches import Ellipse

    apply_default_style()
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")

    # hdbscan handles bars, split lines, colorbar, inverted y-axis, spine removal
    clusterer.condensed_tree_.plot(
        axis=ax, cmap=cmap, select_clusters=False, colorbar=True,
    )

    if select_clusters:
        chosen = clusterer.condensed_tree_._select_clusters()
        plot_data = clusterer.condensed_tree_.get_plot_data()
        cb_all = plot_data["cluster_bounds"]

        plot_range = np.array(plot_data["bar_tops"] + plot_data["bar_bottoms"], dtype=float)
        plot_range = plot_range[np.isfinite(plot_range)]
        mean_y = float(np.mean([plot_range.max(), plot_range.min()]))
        # hdbscan bug: np.diff returns shape-(1,) — add [0] to get scalar
        max_h = float(np.diff(np.percentile(plot_range, [10, 90]))[0])
        min_h = 0.1 * max_h

        palette = _cluster_colors(len(chosen))
        for i, c in enumerate(chosen):
            cb = cb_all[c]
            # CB order: [LEFT=0, RIGHT=1, BOTTOM=2, TOP=3]
            width  = float(cb[1] - cb[0])
            height = float(cb[3] - cb[2])
            cx     = float((cb[0] + cb[1]) / 2.0)
            cy     = float((cb[2] + cb[3]) / 2.0)

            if not np.isfinite(cy):
                cy = mean_y
            if not np.isfinite(height) or height < min_h:
                height = max(max_h, min_h)

            ax.add_artist(Ellipse(
                (cx, cy), 2.0 * width, 1.2 * height,
                facecolor="none", edgecolor=palette[i], linewidth=2,
            ))
            if label_clusters:
                ax.annotate(
                    str(i), xy=(cx, cy),
                    xytext=(cx - 4.0 * width, cy + 0.65 * height),
                    horizontalalignment="left", verticalalignment="bottom",
                )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
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
    "plot_condensed_tree",
    "plot_members_vs_persistence",
]
