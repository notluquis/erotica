"""Plotting utilities for EROTICA cluster analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from astropy.table import QTable
from matplotlib import ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable

from erotica.core._style import apply_default_style


def plot_persistence_vs_members(
    data: QTable,
    summary: pd.DataFrame,
    *,
    percentile: float,
    output_dir: Path,
    save: bool,
    filename: str | None,
    figsize: tuple[int, int],
    plot_kwargs: Mapping[str, object] | None = None,
) -> None:
    plot_kwargs = dict(plot_kwargs or {})

    df_all = data.to_pandas()
    pm_std = df_all.groupby("cluster")["pm"].std()
    summary = summary.copy()
    summary["pm_std"] = summary["cluster"].map(pm_std)

    threshold = summary["persistence"].quantile(percentile)
    working = summary.query("cluster != -1")
    top = working.query("persistence >= @threshold")

    if top.empty:
        top = working

    apply_default_style()
    cmap = plt.colormaps[plot_kwargs.get("cmap", "plasma")]
    pm_values = top.pm_std.to_numpy(dtype=float)
    finite = np.isfinite(pm_values)
    if finite.any():
        min_val = float(np.nanmin(pm_values[finite]))
        max_val = float(np.nanmax(pm_values[finite]))
        if min_val == max_val:
            norm = plt.Normalize(min_val - 1e-6, max_val + 1e-6)
        else:
            norm = plt.Normalize(min_val, max_val)
        colors = cmap(norm(np.where(finite, pm_values, min_val)))
    else:
        norm = plt.Normalize(0.0, 1.0)
        colors = [cmap(0.5)] * len(top)

    fig, ax = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(ax)
    ax_top = divider.append_axes("top", size="20%", pad=0, sharex=ax)
    ax_right = divider.append_axes("right", size="20%", pad=0, sharey=ax)

    for sub_ax in (ax, ax_top, ax_right):
        sub_ax.grid(False)

    bins = plot_kwargs.get("bins", "auto")
    ax_top.hist(working.persistence, bins=bins, color="lightgray", edgecolor="k")
    ax_top.axvline(threshold, color="C0", linestyle="--")
    ax_top.xaxis.set_visible(False)
    ax_top.spines["bottom"].set_visible(False)

    ax_right.hist(
        working["count"],
        bins=bins,
        orientation="horizontal",
        color="lightgray",
        edgecolor="k",
    )
    ax_right.axhline(working["count"].quantile(percentile), color="C0", linestyle="--")
    ax_right.yaxis.set_visible(False)
    ax_right.spines["left"].set_visible(False)

    ax.scatter(
        working.persistence,
        working["count"],
        c="lightgray",
        s=plot_kwargs.get("s_all", 40),
        alpha=plot_kwargs.get("alpha_all", 0.5),
        edgecolor="k",
        label="All",
    )
    ax.scatter(
        top.persistence,
        top["count"],
        c=colors,
        s=plot_kwargs.get("s_top", 80),
        alpha=plot_kwargs.get("alpha_top", 0.9),
        edgecolor="k",
        label="Top percentile",
    )

    texts = [
        ax.text(
            x,
            y,
            str(lbl),
            ha="center",
            va="center",
            zorder=5,
            fontsize=plot_kwargs.get("fontsize", 11),
        )
        for x, y, lbl in zip(top.persistence, top["count"], top.cluster)
    ]
    _, arrows = adjust_text(
        texts,
        x=working.persistence,
        y=working["count"],
        ax=ax,
        expand=plot_kwargs.get("expand", (2.0, 2.0)),
        force_static=plot_kwargs.get("force_static", (0.4, 0.4)),
        min_arrow_len=plot_kwargs.get("min_arrow_len", 0),
        arrowprops=plot_kwargs.get(
            "arrowprops",
            dict(
                arrowstyle="->",
                color="gray",
                lw=plot_kwargs.get("arrow_lw", 1.2),
                shrinkA=plot_kwargs.get("shrinkA", 10),
                shrinkB=plot_kwargs.get("shrinkB", 10),
                connectionstyle=plot_kwargs.get("connectionstyle", "arc3,rad=0.2"),
                zorder=1,
            ),
        ),
    )
    for arrow in arrows:
        arrow.set_alpha(plot_kwargs.get("arrow_alpha", 0.7))

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="horizontal",
        fraction=plot_kwargs.get("cbar_fraction", 0.046),
        pad=plot_kwargs.get("cbar_pad", 0.065),
    )
    cbar.set_label(
        plot_kwargs.get("cbar_label", "Std Dev of Total PM"),
        fontsize=plot_kwargs.get("cbar_fontsize", 12),
    )

    ax.set(
        xlabel=plot_kwargs.get("xlabel", "Persistence"),
        ylabel=plot_kwargs.get("ylabel", "Number of Members"),
    )

    if save:
        fname = filename or "persistence_vs_members.png"
        out_path = Path(output_dir) / fname
        fig.savefig(out_path, dpi=plot_kwargs.get("dpi", 300), bbox_inches="tight")
    plt.show()


def plot_probability_vs_gmag(
    data: QTable,
    *,
    cluster: int,
    probability_threshold: float,
    gmag_limit: float | None,
    figsize: tuple[int, int],
    output_dir: Path,
    save: bool,
    filename: str | None,
    plot_kwargs: Mapping[str, object] | None = None,
) -> None:
    plot_kwargs = dict(plot_kwargs or {})
    table = data.to_pandas()
    subset = table[table["cluster"] == cluster]
    if gmag_limit is not None:
        subset = subset[subset["Gmag"] <= gmag_limit]

    if subset.empty:
        raise ValueError(f"Cluster {cluster} has zero rows in the provided data.")

    apply_default_style()
    fig, ax = plt.subplots(figsize=figsize)
    sc = ax.scatter(
        subset["Gmag"],
        subset["probability_hdbscan"],
        s=plot_kwargs.get("s", 9),
        c=subset["fidelity_v2"],
        cmap=plot_kwargs.get("cmap", "viridis"),
        vmin=subset["fidelity_v2"].min(),
        vmax=subset["fidelity_v2"].max(),
    )

    cbar = fig.colorbar(sc, ax=ax, pad=plot_kwargs.get("cbar_pad", 0.01))
    cbar.set_label(
        plot_kwargs.get("cbar_label", "Astrometric fidelity"),
        fontsize=plot_kwargs.get("cbar_fontsize", 15),
    )
    cbar.ax.tick_params(labelsize=plot_kwargs.get("cbar_tick_size", 14))

    ax.axhline(
        probability_threshold,
        color="k",
        linestyle="--",
        label=plot_kwargs.get("prob_label", f"{int(probability_threshold*100)}% prob"),
    )
    gmag_unit = getattr(data["Gmag"], "unit", None)
    xlabel = plot_kwargs.get("xlabel", f"Gmag [{gmag_unit}]" if gmag_unit is not None else "Gmag")
    ax.set_xlabel(xlabel, fontsize=plot_kwargs.get("xlabel_size", 16))
    ax.set_ylabel(
        plot_kwargs.get("ylabel", "probability_hdbscan"),
        fontsize=plot_kwargs.get("ylabel_size", 16),
    )
    ax.legend(
        loc=plot_kwargs.get("legend_loc", "lower left"),
        fontsize=plot_kwargs.get("legend_size", 10),
        framealpha=plot_kwargs.get("legend_alpha", 0.4),
    )

    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(
        axis="both",
        which="both",
        direction=plot_kwargs.get("tick_direction", "in"),
        labelsize=plot_kwargs.get("tick_labelsize", 14),
    )

    fig.align_labels()
    if save:
        fname = filename or f"prob_vs_gmag_cluster{cluster}.pdf"
        out_path = Path(output_dir) / fname
        fig.savefig(out_path, dpi=plot_kwargs.get("dpi", "figure"), bbox_inches="tight")
    plt.show()


def plot_pms_panels(
    data: QTable,
    *,
    cluster: int,
    pms_threshold: float,
    figsize: tuple[int, int],
    layout: str,
    output_dir: Path,
    save: bool,
    filename: str | None,
    age_xlim: tuple[float, float] | None,
    av_xlim: tuple[float, float] | None,
) -> None:
    subset = data[data["cluster"] == cluster]
    if len(subset) == 0:
        raise ValueError(f"Cluster {cluster} has zero rows.")

    needed = ["pms_sagitta", "age", "av_sagitta"]
    missing = [col for col in needed if col not in subset.colnames]
    if missing:
        raise KeyError(
            "Missing columns in self.data for plotting PMS panels: "
            + ", ".join(missing)
            + ". Run `pms_characterization(...)` first."
        )

    apply_default_style()

    pms = np.array(subset["pms_sagitta"])
    age = np.array(subset["age"])
    av = np.array(subset["av_sagitta"])

    if "ks_m" in subset.colnames:
        has_2mass = np.isfinite(np.array(subset["ks_m"]))
    else:
        has_j = (
            np.isfinite(np.array(subset["j_m"]))
            if "j_m" in subset.colnames
            else np.zeros(len(subset), dtype=bool)
        )
        has_h = (
            np.isfinite(np.array(subset["h_m"]))
            if "h_m" in subset.colnames
            else np.zeros(len(subset), dtype=bool)
        )
        has_k = (
            np.isfinite(np.array(subset["ks_m"]))
            if "ks_m" in subset.colnames
            else np.zeros(len(subset), dtype=bool)
        )
        has_2mass = has_j | has_h | has_k

    finite_pms = np.isfinite(pms)
    pms_mask = (pms >= pms_threshold) & has_2mass & finite_pms
    nopms_mask = (pms < pms_threshold) & has_2mass & finite_pms
    no2mass_mask = ~has_2mass & finite_pms

    bins_pms = np.arange(0.0, 1.0 + 0.1, 0.1)

    finite_age = np.isfinite(age)
    if finite_age.any():
        age_min, age_max = np.floor(np.nanmin(age[finite_age])), np.ceil(np.nanmax(age[finite_age]))
        bins_age = np.arange(age_min, age_max + 0.1, 0.1)
    else:
        bins_age = np.arange(6.0, 8.1, 0.1)

    finite_av = np.isfinite(av)
    if finite_av.any():
        av_min, av_max = np.floor(np.nanmin(av[finite_av])), np.ceil(np.nanmax(av[finite_av]))
        bins_av = np.arange(av_min, av_max + 0.1, 0.1)
    else:
        bins_av = np.arange(0.0, 3.1, 0.1)

    fig, axes = plt.subplots(3, 1, layout=layout, figsize=figsize)

    axes[0].hist(
        pms[finite_pms],
        bins=bins_pms,
        color="gray",
        histtype="step",
        label="All Data",
        alpha=0.85,
    )
    axes[0].hist(
        pms[pms_mask],
        bins=bins_pms,
        color="orange",
        histtype="step",
        label="PMS",
        alpha=0.9,
    )
    axes[0].hist(
        pms[nopms_mask],
        bins=bins_pms,
        color="blue",
        histtype="step",
        label="Non-PMS",
        alpha=0.9,
        linestyle="--",
    )
    axes[0].hist(
        pms[no2mass_mask],
        bins=bins_pms,
        color="green",
        histtype="step",
        label="No 2MASS Info",
        alpha=0.9,
        linestyle=":",
    )
    axes[0].set_xlabel("PMS Probability", fontsize=16)

    axes[1].hist(
        age[finite_age],
        bins=bins_age,
        color="gray",
        histtype="step",
        label="All Data",
        alpha=0.85,
    )
    axes[1].hist(
        age[pms_mask & finite_age],
        bins=bins_age,
        color="orange",
        histtype="step",
        label="PMS",
        alpha=0.9,
    )
    axes[1].hist(
        age[nopms_mask & finite_age],
        bins=bins_age,
        color="blue",
        histtype="step",
        label="Non-PMS",
        alpha=0.9,
        linestyle="--",
    )
    axes[1].hist(
        age[no2mass_mask & finite_age],
        bins=bins_age,
        color="green",
        histtype="step",
        label="No 2MASS Info",
        alpha=0.9,
        linestyle=":",
    )
    axes[1].set_xlabel("log(Age)", fontsize=16)
    if age_xlim is not None:
        axes[1].set_xlim(age_xlim)

    axes[2].hist(
        av[finite_av],
        bins=bins_av,
        color="gray",
        histtype="step",
        label="All Data",
        alpha=0.85,
    )
    axes[2].hist(
        av[pms_mask & finite_av],
        bins=bins_av,
        color="orange",
        histtype="step",
        label="PMS",
        alpha=0.9,
    )
    axes[2].hist(
        av[nopms_mask & finite_av],
        bins=bins_av,
        color="blue",
        histtype="step",
        label="Non-PMS",
        alpha=0.9,
        linestyle="--",
    )
    axes[2].hist(
        av[no2mass_mask & finite_av],
        bins=bins_av,
        color="green",
        histtype="step",
        label="No 2MASS Info",
        alpha=0.9,
        linestyle=":",
    )
    axes[2].set_xlabel(r"$A_V$", fontsize=16)
    if av_xlim is not None:
        axes[2].set_xlim(av_xlim)

    for ax in axes:
        ax.set_ylabel("Count", fontsize=16)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.tick_params(axis="both", which="both", direction="in", labelsize=14)
        ax.grid(False)

    axes[1].legend(fontsize=12, framealpha=0.4)
    fig.align_labels()

    if save:
        fname = filename or f"pms_stats_cluster{cluster}.pdf"
        out_path = Path(output_dir) / fname
        fig.savefig(out_path, dpi="figure", bbox_inches="tight")
    plt.show()


__all__ = [
    "plot_persistence_vs_members",
    "plot_probability_vs_gmag",
    "plot_pms_panels",
]
