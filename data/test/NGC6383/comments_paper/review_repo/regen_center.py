#!/usr/bin/env python3
"""Regenerate center_determination_paper.pdf, faithful to the notebook's
graph_center_determination(..., paper_single=True, distance_scale=mu_r): sky
scatter of the likely candidates colour-coded by pseudo-probability (viridis),
KDE iso-density contours, dark-orange center crosshair, and a 5 pc physical scale bar
at the adopted distance (1.110 kpc). The notebook drew the bar via
astropy's add_scalebar in white on a WCS axes; here the axes are plain
RA/Dec degrees, so the bar is drawn in black (visible on the white background)
with the cos(dec) factor applied so 5 pc is correct along the RA axis."""
import numpy as np
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.stats import gaussian_kde

B = "/Users/notluquis/COSMIC/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"
D_KPC = 1.110                                   # adopted parallax distance (mu_r)

m = Table.read(f"{B}/comments_paper/cds_final/ngc6383_members.ecsv").to_pandas()
ra = m.RAdeg.values; dec = m.DEdeg.values; ptil = m.pMember.values
cra, cdec = 263.6826, -32.5838

fig, ax = plt.subplots(figsize=(7.5, 6.5))
xy = np.vstack([ra, dec]); kde = gaussian_kde(xy)
kde.set_bandwidth(kde.factor * 1.4)              # mild extra smoothing for cleaner contours
xx, yy = np.mgrid[ra.min():ra.max():120j, dec.min():dec.max():120j]
zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
# iso-density contours at the 1- and 2-sigma enclosed fractions (39.3, 86.5% of the KDE),
# the standard astronomical convention. We omit the 3-sigma (98.9%) level because the
# proper-motion-selected candidates have a broad sky halo, so that contour traces the
# whole field rather than the cluster concentration.
zf = zz.ravel(); o = np.argsort(zf)[::-1]; cs = np.cumsum(zf[o]); cs /= cs[-1]
levs = sorted({float(zf[o][min(np.searchsorted(cs, f), zf.size - 1)]) for f in (0.393, 0.865)})
ax.contour(xx, yy, zz, levels=levs, colors="0.4", linewidths=0.8, zorder=2)
sc = ax.scatter(ra, dec, c=ptil, cmap="viridis", s=12, zorder=3)
ax.axvline(cra, color="darkorange", ls="--", lw=1.6, zorder=4, label="Cluster center")
ax.axhline(cdec, color="darkorange", ls="--", lw=1.6, zorder=4)
cb = fig.colorbar(sc, ax=ax, pad=0.01)
cb.set_label(r"Membership pseudo-probability $\tilde{p}$", fontsize=13)
ax.invert_xaxis()
ax.set_xlabel(r"$\alpha$ [deg]", fontsize=14)
ax.set_ylabel(r"$\delta$ [deg]", fontsize=14)
# the KDE contour pins the limits via sticky edges; disable so the default 5%
# data margin applies (no candidate sits on the frame) before the scale bar is placed
ax.use_sticky_edges = False; ax.margins(0.05); ax.autoscale_view()
from matplotlib.lines import Line2D
_kdeproxy = Line2D([0], [0], color="0.4", lw=0.8, label=r"KDE $1\sigma,2\sigma$ iso-density")
h, _ = ax.get_legend_handles_labels()
ax.legend(handles=h + [_kdeproxy], loc="upper left", framealpha=0.9)

# 5 pc scale bar (bottom right). 5 pc / 1110 pc = 0.258 deg true angle;
# extent along the (uncorrected) RA axis needs the cos(dec) factor.
ang = np.degrees(5.0 / (D_KPC * 1000.0))
dx = ang / np.cos(np.radians(cdec))
x0, x1 = ax.get_xlim()                          # inverted: x0 > x1 (x1 = right edge)
y0, y1 = ax.get_ylim()
xs = x1 + 0.10 * (x0 - x1)
yb = y0 + 0.07 * (y1 - y0)
ax.plot([xs, xs + dx], [yb, yb], color="k", lw=2.5, solid_capstyle="butt", zorder=6,
        path_effects=[pe.Stroke(linewidth=4, foreground="white"), pe.Normal()])
ax.text(xs + dx / 2, yb + 0.02 * (y1 - y0), "5 pc", ha="center", va="bottom",
        fontsize=11, zorder=6,
        path_effects=[pe.Stroke(linewidth=2.5, foreground="white"), pe.Normal()])

fig.savefig(FIG + "center_determination_paper.pdf", bbox_inches="tight")
plt.close()
print("wrote center_determination_paper.pdf  (viridis + KDE + 5 pc scale bar)")
