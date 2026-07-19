#!/usr/bin/env python3
"""Standalone regeneration of plot_pair_trace.pdf (Fig. B.3), replacing the
notebook version. 5x5 corner plot of the ASteCA isochrone-fit posterior
(Av, dm, loga, met) plus the likelihood-scatter nuisance term sigma: blue
marginal KDE on the diagonal, viridis hexbin joint densities off-diagonal,
black lines at the per-parameter mode and a black square at the joint mode.
Shares the common figure styling with the other regen_*.py scripts."""
import numpy as np, arviz as az
from scipy.stats import gaussian_kde
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

B = "/Users/notluquis/COSMIC/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"
post = az.from_netcdf(f"{B}/data/40/fit_parameters_trace_1724708835.nc").posterior

names = ["Av", "dm", "loga", "met", "sigma"]
s = {n: post[n].values.ravel() for n in names}
mode = {}
for n in names:
    x = s[n]; k = gaussian_kde(x); g = np.linspace(x.min(), x.max(), 400)
    mode[n] = g[np.argmax(k(g))]

n = len(names)
fig, axes = plt.subplots(n, n, figsize=(12.5, 12.5))
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if j > i:
            ax.axis("off"); continue
        if i == j:
            x = s[names[i]]; k = gaussian_kde(x); g = np.linspace(x.min(), x.max(), 300)
            ax.plot(g, k(g), color="#1f77b4", lw=1.5)
            ax.axvline(mode[names[i]], color="black", lw=1.0)
            ax.set_yticks([])
        else:
            x = s[names[j]]; y = s[names[i]]
            ax.hexbin(x, y, gridsize=25, cmap="viridis", mincnt=1)
            ax.axvline(mode[names[j]], color="black", lw=0.8)
            ax.axhline(mode[names[i]], color="black", lw=0.8)
            ax.plot(mode[names[j]], mode[names[i]], "s", color="black", ms=4)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        if i != j:
            ax.yaxis.set_major_locator(MaxNLocator(4))   # diagonal keeps empty y (density scale hidden)
        else:
            ax.set_yticks([])
        if i == n - 1:
            ax.set_xlabel(names[j])
        else:
            ax.set_xticklabels([])
        if j == 0 and i != 0:
            ax.set_ylabel(names[i])
        elif i != j:
            ax.set_yticklabels([])
fig.subplots_adjust(wspace=0.06, hspace=0.06)
fig.savefig(FIG + "plot_pair_trace.pdf", bbox_inches="tight")
plt.close()
print("wrote plot_pair_trace.pdf  (params: %s)" % ", ".join(names))
