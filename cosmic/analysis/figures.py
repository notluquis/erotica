"""Paper-style diagnostic figures for COSMIC analysis results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, angular_separation
from astropy.stats import knuth_bin_width
from matplotlib.colors import ListedColormap, LogNorm
from scipy.stats import gaussian_kde, ks_2samp, norm

from cosmic.core._style import apply_default_style

from .debugging import diagnose_distance_comparison, extract_distance_samples
from .external.asteca import generate_isochrone_for_plot
from .structure import (
    ClusterStructureAnalyzer,
    RDP_bayesian,
    RDP_bayesian_log_space,
    king_profile,
)
from .units import quantity_values


def _density_grid(samples, *, n_grid=200):
    samples = np.asarray(samples, dtype=float).ravel()
    samples = samples[np.isfinite(samples)]
    grid = np.linspace(*np.nanpercentile(samples, [0.5, 99.5]), n_grid)
    density = gaussian_kde(samples)(grid)
    density /= np.nanmax(density)
    return grid, density


def _peak_normalized_kde(samples, grid):
    """Legacy-compatible peak-normalized KDE helper."""
    density = gaussian_kde(samples)(grid)
    peak = np.nanmax(density)
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("KDE density could not be normalized.")
    return density / peak


def _distance_samples_from_parallax_result(parallax_results=None, parallax_trace=None):
    """Legacy-compatible extraction of parallax/distance posterior samples."""
    if parallax_trace is not None:
        return extract_distance_samples(parallax_trace)
    return extract_distance_samples(parallax_results)


def distance_samples_from_dm_trace(asteca_trace, *, variable: str = "dm") -> np.ndarray:
    """Transform distance-modulus posterior samples to kpc."""
    if not hasattr(asteca_trace, "posterior") or variable not in asteca_trace.posterior:
        raise ValueError(f"ASteCA trace must contain posterior variable {variable!r}.")
    dm = np.asarray(asteca_trace.posterior[variable].values, dtype=float).ravel()
    dm = dm[np.isfinite(dm)]
    return 10 ** ((dm + 5) / 5) / 1000


def _distance_samples_from_asteca_trace(asteca_trace=None, asteca_trace_path="fit_parameters_trace_1777967244.nc"):
    """Legacy-compatible ASteCA distance-modulus posterior extraction."""
    if asteca_trace is None:
        try:
            import arviz as az
        except ImportError as exc:
            raise ImportError("ArviZ is required to load ASteCA NetCDF traces.") from exc
        asteca_trace = az.from_netcdf(asteca_trace_path)
    return distance_samples_from_dm_trace(asteca_trace)


def plot_distance_pathway_overlap(
    *,
    parallax_results=None,
    parallax_trace=None,
    asteca_trace=None,
    distance_a=None,
    distance_b=None,
    savefig=None,
    show=True,
    random_seed=42,
):
    """Plot distance posterior overlap plus posterior difference diagnostics."""
    if distance_a is None:
        distance_a = extract_distance_samples(parallax_results or parallax_trace)
    if distance_b is None:
        if asteca_trace is None:
            raise ValueError("Provide either distance_b samples or asteca_trace.")
        distance_b = distance_samples_from_dm_trace(asteca_trace)

    comparison = diagnose_distance_comparison(distance_a, distance_b, random_seed=random_seed)
    grid_a, density_a = _density_grid(distance_a)
    grid_b, density_b = _density_grid(distance_b)

    rng = np.random.default_rng(random_seed)
    n_delta = min(len(distance_a), len(distance_b), 100_000)
    delta_samples = (
        rng.choice(np.asarray(distance_b), size=n_delta, replace=False)
        - rng.choice(np.asarray(distance_a), size=n_delta, replace=False)
    )
    grid_delta, density_delta = _density_grid(delta_samples)

    apply_default_style({"axes.grid": False, "legend.frameon": True})
    fig, (ax, ax_delta) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    ax.plot(grid_a, density_a, color="tab:blue", linewidth=2, label=r"$\varpi$-hierarchical")
    ax.fill_between(grid_a, density_a, 0, color="tab:blue", alpha=0.10, linewidth=0)
    ax.plot(grid_b, density_b, color="orange", linewidth=2, label=r"isochronal $\mu_\mathrm{DM}$")
    ax.fill_between(grid_b, density_b, 0, color="orange", alpha=0.18, linewidth=0)
    ax.axvline(
        comparison.mode_a.value,
        color="tab:blue",
        linestyle="--",
        linewidth=1.5,
        label=rf"$d_\varpi^\mathrm{{mode}}={comparison.mode_a.value:.3f}$ kpc",
    )
    ax.axvline(
        comparison.mode_b.value,
        color="orange",
        linestyle="--",
        linewidth=1.5,
        label=rf"$d_\mathrm{{iso}}^\mathrm{{mode}}={comparison.mode_b.value:.3f}$ kpc",
    )
    ax.set_xlabel("Distance [kpc]", fontsize=16)
    ax.set_ylabel("Peak-normalized posterior density", fontsize=16)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right", framealpha=0.8, fontsize=11)

    footer = (
        rf"$\Delta d_\mathrm{{mode}}={comparison.delta.value:+.3f}\,\mathrm{{kpc}}$; "
        rf"$\Delta d/\sigma_\mathrm{{comb}}={comparison.delta_in_sigma_combined:.1f}$"
    )
    ax.text(
        0.04,
        0.06,
        footer,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12,
        color="0.2",
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.8),
    )

    ax_delta.plot(grid_delta, density_delta, color="tab:gray", linewidth=2, label=r"$\Delta d$ posterior")
    ax_delta.fill_between(grid_delta, density_delta, 0, color="tab:gray", alpha=0.18, linewidth=0)
    ax_delta.axvline(0, color="black", linestyle="-", linewidth=1.2, label=r"$\Delta d=0$")
    ax_delta.axvline(
        comparison.delta.value,
        color="tab:gray",
        linestyle="--",
        linewidth=1.5,
        label=rf"$\Delta d_\mathrm{{mode}}={comparison.delta.value:+.3f}$ kpc",
    )
    interval = comparison.delta_interval_16_84.value
    ax_delta.axvspan(interval[0], interval[1], color="tab:gray", alpha=0.12, label="68% interval")
    ax_delta.set_xlabel(r"$\Delta d = d_{\mu_\mathrm{DM}} - d_\varpi$ [kpc]", fontsize=16)
    ax_delta.set_ylabel("Peak-normalized posterior density", fontsize=16)
    ax_delta.set_ylim(0, 1.08)
    ax_delta.legend(loc="upper left", framealpha=0.8, fontsize=11)
    ax_delta.text(
        0.96,
        0.06,
        rf"$68\%\ \Delta d=[{interval[0]:+.3f},{interval[1]:+.3f}]$",
        transform=ax_delta.transAxes,
        ha="right",
        va="bottom",
        fontsize=12,
        color="0.2",
        bbox=dict(facecolor="white", edgecolor="0.8", alpha=0.8),
    )

    for axis in (ax, ax_delta):
        axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        axis.tick_params(axis="both", which="both", direction="in", labelsize=14)

    if savefig:
        path = Path(savefig)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return comparison


def plot_cumulative(
    data,
    centers,
    *,
    prob_number=(50, 60, 70, 80),
    probability_column="probability",
    magnitude_column="Gmag",
    ra_column="ra",
    dec_column="dec",
    savefig=None,
    show=True,
):
    """Plot cumulative integrated magnitude as a function of cluster radius."""
    prob_thresholds = _normalise_probabilities(prob_number)
    centers = [SkyCoord(ra=ra, dec=dec, frame="icrs", unit="deg") for ra, dec in centers]
    if len(centers) == 1 and len(prob_thresholds) > 1:
        centers = centers * len(prob_thresholds)
    fig, axes = plt.subplots(1, len(prob_thresholds), figsize=(5 * len(prob_thresholds), 5), squeeze=False)
    for threshold, center, ax in zip(prob_thresholds, centers, axes.ravel()):
        selected = data[data[probability_column] >= threshold]
        distances = angular_separation(selected[ra_column], selected[dec_column], center.ra, center.dec).to(u.arcmin)
        radii = np.linspace(0, np.nanmax(distances.value), 400)
        cumulative = []
        for radius in radii:
            flux_sum = np.sum(10 ** (np.asarray(selected[magnitude_column][distances <= radius * u.arcmin]) / -2.5))
            cumulative.append(np.nan if flux_sum <= 0 else -2.5 * np.log10(flux_sum))
        ax.scatter(distances, selected[magnitude_column], s=10, color="tab:blue")
        twin = ax.twinx()
        twin.plot(radii, cumulative, color="tab:red", label="Integrated magnitude")
        ax.set_xlabel("Radius [arcmin]", fontsize=16)
        ax.set_ylabel(r"$G_\mathrm{mag}$ [mag]", fontsize=16)
        twin.set_ylabel(r"$\Sigma G_\mathrm{mag}$ [mag]", fontsize=16)
        ax.invert_yaxis()
        twin.invert_yaxis()
        ax.set_title(f"{int(threshold * 100)}% probability members")
        twin.legend(framealpha=0.8)
    if savefig:
        fig.savefig(savefig, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_isochrone(
    isochrones,
    logAge,
    Zini,
    color_row,
    mag_row,
    ax,
    *,
    dm=0,
    A_V=0,
    R_V=3.1,
    color="blue",
    alpha=0.4,
    linestyle="--",
    label=True,
    type="plot",
    yl=None,
):
    """Plot a single legacy isochrone DataFrame on a CMD axis."""
    filtered = isochrones.query(f"logAge == {logAge} and Zini == {Zini}").copy()
    if filtered.empty:
        raise ValueError(f"No isochrone found for logAge={logAge} and Zini={Zini}.")
    ext = A_V / R_V
    mag = filtered[mag_row] + dm + A_V
    col = filtered[color_row] + ext
    if yl:
        mask = mag >= 0.95 * yl[0]
        mag = mag[mask]
        col = col[mask]
    label_text = rf"$\log(age/yr)={logAge}$" if label else None
    if type == "plot":
        return ax.plot(col, mag, lw=3, color=color, alpha=alpha, linestyle=linestyle, label=label_text)
    if type == "scatter":
        return ax.scatter(col, mag, color=color, alpha=alpha, marker=linestyle, label=label_text)
    raise ValueError("type must be 'plot' or 'scatter'.")


def plot_color_color(
    isochrone_1,
    isochrone_2,
    logAge,
    Zini,
    color_1_row,
    color_2_row,
    ax,
    *,
    A_V=0,
    R_V=3.1,
    color="blue",
    alpha=0.4,
    linestyle="--",
):
    """Plot a legacy color-color isochrone relation."""
    first = isochrone_1.query(f"logAge == {logAge} and Zini == {Zini}").copy()
    second = isochrone_2.query(f"logAge == {logAge} and Zini == {Zini}").copy()
    if first.empty or second.empty:
        raise ValueError(f"No isochrone found for logAge={logAge} and Zini={Zini}.")
    ext = A_V / R_V
    return ax.plot(first[color_1_row] + ext, second[color_2_row] + ext, lw=3, c=color, alpha=alpha, linestyle=linestyle, label=rf"logAge={logAge}")


def plot_isochrone_label(isochrones, logAge, Zini, color_row, mag_row, ax, **kwargs):
    """Compatibility wrapper for label-aware isochrone plotting."""
    return plot_isochrone(isochrones, logAge, Zini, color_row, mag_row, ax, **kwargs)


def plot_errors_bar(magnitudes, colors, magnitude_errors, color_errors, ax, loc=None):
    """Plot median CMD error bars in two-magnitude bins."""
    mag_values = quantity_values(magnitudes, u.mag)
    mag_err = quantity_values(magnitude_errors, u.mag)
    color_err = quantity_values(color_errors, u.mag)
    bins = np.arange(np.floor(np.nanmin(mag_values)), np.ceil(np.nanmax(mag_values)) + 2, 2)
    median_mag_errors = []
    median_color_errors = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (mag_values >= lo) & (mag_values < hi)
        median_mag_errors.append(float(np.nanmedian(mag_err[mask])) if mask.any() else np.nan)
        median_color_errors.append(float(np.nanmedian(color_err[mask])) if mask.any() else np.nan)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    if loc is None:
        loc = 0.5 * np.nanmin(quantity_values(colors, u.mag))
    loc_value = loc.value if hasattr(loc, "value") else float(loc)
    return ax.errorbar(
        [loc_value] * len(bin_centers),
        bin_centers,
        xerr=median_color_errors,
        yerr=median_mag_errors,
        fmt="none",
        color="k",
        ecolor="k",
    )


def plot_iso_mass_curves_across_isochrones(
    isochrones,
    logAge_range,
    Zini,
    ax,
    *,
    dm=0,
    A_V=0,
    R_V=3.1,
    color="black",
    alpha=1,
    linestyle="-",
    mass_values=None,
    zorder=1,
):
    """Plot mass-track curves across a legacy isochrone grid."""
    if mass_values is None:
        raise ValueError("mass_values must be provided.")
    ext = A_V / R_V
    ages = np.sort(isochrones[(isochrones["Zini"] == Zini) & (isochrones["logAge"].between(*logAge_range))]["logAge"].unique())
    lines = []
    for mass in mass_values:
        cols = []
        mags = []
        for age in ages:
            frame = isochrones[(isochrones["logAge"] == age) & (isochrones["Zini"] == Zini)]
            points = frame[frame["Mass"].round(1) == mass]
            if not points.empty:
                first = points.head(1)
                cols.append(first["BP_RP"].iloc[0] + ext)
                mags.append(first["Gmag"].iloc[0] + dm + A_V)
        if cols and mags:
            lines.extend(ax.plot(cols, mags, label=fr"Mass = {mass} $M_\odot$", color=color, alpha=alpha, linestyle=linestyle, zorder=zorder))
    return lines


def _generate_plot_isochrone(synthcl, params, return_masses=False):
    """Generate an isochrone through the ASteCA adapter."""
    return np.asarray(generate_isochrone_for_plot(synthcl, params, return_masses=return_masses))


def plot_hist2d(
    synthcl,
    param_means,
    param_stds,
    ax,
    *,
    n_samples=100,
    cmin=50,
    cmap=ListedColormap(plt.get_cmap("gray_r")(np.linspace(0, 1, 69))),
    alpha=0.8,
    bins=None,
    return_masses=False,
    random_seed=None,
):
    """Plot sampled ASteCA isochrones as a 2D density in CMD space."""
    rng = np.random.default_rng(random_seed)
    all_isochrones = []
    for _ in range(n_samples):
        sampled = {key: rng.normal(param_means[key], param_stds[key]) for key in param_means}
        isochrone = _generate_plot_isochrone(synthcl, sampled, return_masses=return_masses)
        if isochrone.size == 0:
            continue
        if return_masses:
            mass = isochrone[3] if isochrone.shape[0] > 3 else np.full_like(isochrone[0], np.nan)
            all_isochrones.append((isochrone[0], isochrone[1], isochrone[2], mass))
        else:
            all_isochrones.append((isochrone[0], isochrone[1]))
    if not all_isochrones:
        raise ValueError("No valid isochrones were generated for the sampled parameters.")
    colors = np.concatenate([iso[1] for iso in all_isochrones])
    mags = np.concatenate([iso[0] for iso in all_isochrones])
    if bins is None:
        _, color_bins = knuth_bin_width(colors[np.isfinite(colors)], return_bins=True)
        _, mag_bins = knuth_bin_width(mags[np.isfinite(mags)], return_bins=True)
        bins = [color_bins, mag_bins]
    mesh = ax.hist2d(colors, mags, bins=bins, cmap=cmap, norm=LogNorm(), zorder=-1, alpha=alpha, cmin=cmin)[3]
    return all_isochrones if return_masses else mesh


def _normalise_probabilities(prob_number):
    values = np.asarray(prob_number, dtype=float)
    return np.where(values > 1, values / 100, values)


def _centers_as_skycoord(centers, n):
    coords = [center if isinstance(center, SkyCoord) else SkyCoord(ra=center[0], dec=center[1], frame="icrs", unit="deg") for center in centers]
    if len(coords) == 1 and n > 1:
        coords *= n
    return coords


def _ks_pairwise(distance_groups):
    results = {}
    keys = list(distance_groups)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            a = np.asarray(distance_groups[left], dtype=float)
            b = np.asarray(distance_groups[right], dtype=float)
            if len(a) and len(b):
                results[(left, right)] = ks_2samp(a, b)
    return results


def plot_cumulative_by_brightness(
    data,
    centers,
    *,
    prob_number=(50, 60, 70, 80),
    brightness_ranges=None,
    savefig=None,
    R_c=None,
    R_t=None,
    normalize=False,
    ks=True,
    paper=False,
    show=True,
):
    """Plot cumulative radial distributions in G-magnitude bins."""
    thresholds = _normalise_probabilities(prob_number)
    centers = _centers_as_skycoord(centers, len(thresholds))
    if brightness_ranges is None:
        quartiles = np.nanpercentile(quantity_values(data["Gmag"], u.mag), [0, 25, 50, 75, 100])
        brightness_ranges = [(quartiles[i], quartiles[i + 1]) for i in range(4)]
    fig, axes = plt.subplots((len(thresholds) + 1) // 2, 2 if len(thresholds) > 1 else 1, figsize=(12, 8), squeeze=False)
    all_distances = {item: [] for item in brightness_ranges}
    for threshold, center, ax in zip(thresholds, centers, axes.ravel()):
        selected = data[data["probability"] >= threshold]
        distances = angular_separation(selected["ra"], selected["dec"], center.ra, center.dec).to(u.arcmin)
        radii = np.linspace(0, np.nanmax(distances.value), 400) * u.arcmin
        gmag = quantity_values(selected["Gmag"], u.mag)
        for mag_min, mag_max in brightness_ranges:
            mask = (gmag > mag_min) & (gmag <= mag_max)
            group_distances = distances[mask]
            all_distances[(mag_min, mag_max)].extend(group_distances.value)
            cumulative = np.array([np.count_nonzero(group_distances <= radius) for radius in radii], dtype=float)
            if normalize and cumulative.max() > 0:
                cumulative /= cumulative.max()
            ax.plot(radii.value, cumulative, label=rf"$G$: {mag_min:.2f}-{mag_max:.2f}")
        ax.set_xlabel("Radius [arcmin]")
        ax.set_ylabel("Normalized cumulative count" if normalize else "Cumulative count")
        if not paper:
            ax.set_title(f"{int(threshold * 100)}% probability members")
        ax.legend()
    ks_results = _ks_pairwise(all_distances) if ks else {}
    if savefig:
        suffix = "cumulative_by_brightness_paper.pdf" if paper else "cumulative_by_brightness.pdf"
        Path(savefig).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(savefig) / suffix, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"figure": fig, "ks_results": ks_results}


def plot_cumulative_by_mass(
    data,
    centers,
    *,
    prob_number=(50, 60, 70, 80),
    mass_ranges=None,
    savefig=None,
    normalize=False,
    ks=True,
    show=True,
    **_,
):
    """Plot cumulative radial distributions in mass bins."""
    thresholds = _normalise_probabilities(prob_number)
    centers = _centers_as_skycoord(centers, len(thresholds))
    masses_all = quantity_values(data["mass"], u.Msun)
    if mass_ranges is None:
        quartiles = np.nanpercentile(masses_all, [0, 25, 50, 75, 100])
        mass_ranges = [(quartiles[i], quartiles[i + 1]) for i in range(4)]
    fig, axes = plt.subplots((len(thresholds) + 1) // 2, 2 if len(thresholds) > 1 else 1, figsize=(12, 8), squeeze=False)
    ks_results = {}
    for threshold, center, ax in zip(thresholds, centers, axes.ravel()):
        selected = data[data["probability"] >= threshold]
        distances = angular_separation(selected["ra"], selected["dec"], center.ra, center.dec).to(u.arcmin)
        radii = np.linspace(0, np.nanmax(distances.value), 400) * u.arcmin
        masses = quantity_values(selected["mass"], u.Msun)
        groups = {}
        for mass_min, mass_max in mass_ranges:
            mask = (masses >= mass_min) & (masses < mass_max)
            group_distances = distances[mask]
            groups[(mass_min, mass_max)] = group_distances.value
            cumulative = np.array([np.count_nonzero(group_distances <= radius) for radius in radii], dtype=float)
            if normalize and cumulative.max() > 0:
                cumulative /= cumulative.max()
            ax.plot(radii.value, cumulative, label=rf"$M$: {mass_min:.2f}-{mass_max:.2f}")
        if ks:
            ks_results[float(threshold)] = _ks_pairwise(groups)
        ax.set_xlabel("Radius [arcmin]")
        ax.set_ylabel("Normalized cumulative distribution" if normalize else "Cumulative distribution")
        ax.legend()
    if savefig:
        Path(savefig).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(savefig) / "cumulative_by_mass.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"figure": fig, "ks_results": ks_results}


def plot_cumulative_by_mass_and_type(data, centers, *, prob_number=(70,), savefig=None, normalize=False, ks=True, show=True, binary_column="P_binar", **_):
    """Plot mass-binned cumulative distributions split by binary probability."""
    threshold = float(_normalise_probabilities(prob_number)[0])
    center = _centers_as_skycoord(centers, 1)[0]
    selected = data[data["probability"] >= threshold]
    distances = angular_separation(selected["ra"], selected["dec"], center.ra, center.dec).to(u.arcmin)
    masses = quantity_values(selected["mass"], u.Msun)
    quartiles = np.nanpercentile(masses, [0, 25, 50, 75, 100])
    mass_ranges = [(quartiles[i], quartiles[i + 1]) for i in range(4)]
    binary_prob = quantity_values(selected[binary_column]) if binary_column in selected.colnames else np.zeros(len(selected))
    datasets = [("Single Stars", binary_prob < 0.6), ("Binary Stars", binary_prob >= 0.6)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False, sharex=True)
    radii = np.linspace(0, np.nanmax(distances.value), 400) * u.arcmin
    ks_results = {}
    for ax, (title, type_mask) in zip(axes.ravel()[:2], datasets):
        groups = {}
        for mass_min, mass_max in mass_ranges:
            mask = type_mask & (masses >= mass_min) & (masses < mass_max)
            group_distances = distances[mask]
            groups[(mass_min, mass_max)] = group_distances.value
            cumulative = np.array([np.count_nonzero(group_distances <= radius) for radius in radii], dtype=float)
            if normalize and cumulative.max() > 0:
                cumulative /= cumulative.max()
            ax.plot(radii.value, cumulative, label=rf"$M$: {mass_min:.2f}-{mass_max:.2f}")
        if ks:
            ks_results[title] = _ks_pairwise(groups)
        ax.set_title(title)
        ax.set_xlabel("Radius [arcmin]")
        ax.legend()
    axes.ravel()[2].plot(radii.value, [np.count_nonzero(distances[binary_prob < 0.6] <= radius) for radius in radii], label="Single Stars")
    axes.ravel()[2].plot(radii.value, [np.count_nonzero(distances[binary_prob >= 0.6] <= radius) for radius in radii], label="Binary Stars")
    axes.ravel()[2].set_title("Single vs Binary Stars")
    axes.ravel()[2].legend()
    if savefig:
        Path(savefig).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(savefig) / "cumulative_by_mass_and_type.pdf", bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return {"figure": fig, "ks_results": ks_results}


def graph_center_determination(data, *, prob_number=(50, 60, 70, 80), savefig=None, show=True, **kwargs):
    """Legacy-compatible center-determination facade."""
    analyzer = ClusterStructureAnalyzer(data)
    thresholds = _normalise_probabilities(prob_number)
    centers = []
    best_params = []
    for threshold in thresholds:
        result = analyzer.center(threshold, return_bestparams=True, **kwargs)
        centers.append([result["center_coords"], result["center_coords_error"]])
        best_params.append(result.get("best_params", {}))
    if savefig:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(quantity_values(data["ra"], u.deg), quantity_values(data["dec"], u.deg), s=8)
        for center, _ in centers:
            ax.scatter(center[0].value, center[1].value, marker="x", s=80)
        fig.savefig(Path(savefig) / "center_determination.pdf", bbox_inches="tight")
        if show:
            plt.show()
        else:
            plt.close(fig)
    return centers, best_params


def graph_king(data, centers, *, prob_number=(0.5, 0.6, 0.7, 0.8), density_method="equip", log_space=False, return_results=True, **kwargs):
    """Legacy-compatible King-profile analysis facade."""
    analyzer = ClusterStructureAnalyzer(data)
    thresholds = _normalise_probabilities(prob_number)
    center_coords = _centers_as_skycoord([item[0] if isinstance(item, (list, tuple)) and len(item) and isinstance(item[0], (list, tuple)) else item for item in centers], len(thresholds))
    all_results = {}
    traces = []
    for threshold, center in zip(thresholds, center_coords):
        profile = analyzer.radial_density_profile(center, probability_threshold=float(threshold), method=density_method, width=kwargs.get("width"))
        fit = analyzer.fit_king_profile(profile, log_space=log_space, return_trace=kwargs.get("return_trace", False), progressbar=kwargs.get("progressbar_bayesian", False))
        for key, value in fit.items():
            if key == "king_trace":
                traces.append(value)
                continue
            all_results.setdefault(key, []).append(value)
    payload = {}
    if return_results:
        payload["bayesian_results"] = all_results
    if traces:
        payload["traces"] = traces
    return payload or None


def graph_real(data, image, *, dt=None, projection=None, savefig_dir=None, prob_number=(60, 70, 80, 90), show=True):
    """Overlay probability-selected sources on an image/WCS projection."""
    thresholds = _normalise_probabilities(prob_number)
    subplot_kw = {"projection": projection} if projection is not None else {}
    fig, axes = plt.subplots(1, len(thresholds), figsize=(6 * len(thresholds), 6), squeeze=False, subplot_kw=subplot_kw)
    dt = 0 if dt is None else dt
    for threshold, ax in zip(thresholds, axes.ravel()):
        subset = data[data["probability"] >= threshold]
        ra = subset["ra"] - subset["pmra"] * dt
        dec = subset["dec"] - subset["pmdec"] * dt
        ax.imshow(image, cmap="cubehelix", aspect="equal")
        transform = ax.get_transform("world") if projection is not None else None
        ax.scatter(ra, dec, color="lightcyan", transform=transform, alpha=0.6, s=20, facecolors="none")
        ax.set_title(f"{int(threshold * 100)}% probability members")
    if savefig_dir is not None:
        Path(savefig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(savefig_dir) / "real_sky_prob.pdf", dpi=500, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


class ClusterFigureBuilder:
    """Plotting facade around a cluster source table."""

    def __init__(self, data) -> None:
        self.data = data

    def cumulative(self, centers, **kwargs):
        return plot_cumulative(self.data, centers, **kwargs)

    def cumulative_by_brightness(self, centers, **kwargs):
        return plot_cumulative_by_brightness(self.data, centers, **kwargs)

    def cumulative_by_mass(self, centers, **kwargs):
        return plot_cumulative_by_mass(self.data, centers, **kwargs)

    def cumulative_by_mass_and_type(self, centers, **kwargs):
        return plot_cumulative_by_mass_and_type(self.data, centers, **kwargs)

    def centers(self, **kwargs):
        return graph_center_determination(self.data, **kwargs)

    def real_sky(self, image, **kwargs):
        return graph_real(self.data, image, **kwargs)


__all__ = [
    "ClusterFigureBuilder",
    "distance_samples_from_dm_trace",
    "graph_center_determination",
    "graph_king",
    "graph_real",
    "king_profile",
    "plot_cumulative",
    "plot_cumulative_by_brightness",
    "plot_cumulative_by_mass",
    "plot_cumulative_by_mass_and_type",
    "plot_color_color",
    "plot_distance_pathway_overlap",
    "plot_errors_bar",
    "plot_hist2d",
    "plot_iso_mass_curves_across_isochrones",
    "plot_isochrone",
    "plot_isochrone_label",
]
