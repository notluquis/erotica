#!/usr/bin/env python3
"""Regenerate real_sky.pdf faithful to the notebook (cell 73): probable members
and members on the DSS2-red image with the WCS projection, the cluster center,
and the seven scale radii as circles, plus a 5 pc scale bar. Differences vs the
previous published figure: the half-mass radius legend value is corrected to the
adopted 6.26 arcmin (= 2.02 pc; the old figure carried a stale 6.24), and every
radius legend label is formatted to the precision used in the text/Table 1
(R_t = 54 lies outside the DSS2 frame and is not drawn — see Fig. D.1; R_Hill = 33.6; R_c, R_hl, R_hm, R_bound at 2 dp)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization.wcsaxes import SphericalCircle, add_scalebar
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

B = "/Users/notluquis/COSMIC/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"

hdu = fits.open(f"{B}/NGC6383_DSS2-red.fits")[0]
wcs_full = WCS(hdu.header)
STEP = 2                                            # downsample the DSS raster to keep the PDF light
data = hdu.data[::STEP, ::STEP]
wcs = wcs_full.slice((slice(None, None, STEP), slice(None, None, STEP)))

m = Table.read(f"{B}/comments_paper/cds_final/ngc6383_members.ecsv")
ref = m[np.asarray(m["Ref"]) == 1]                 # probable members + members (p>=0.6)
ra = np.asarray(ref["RAdeg"], float); dec = np.asarray(ref["DEdeg"], float)

cra, cdec = 263.6826, -32.5838                     # cluster center
sra, sdec = 263.67148711, -32.57727207             # cone-search center
D_KPC = 1.11                                       # adopted hierarchical-model distance (kpc)

# radii (arcmin) and the precision each label is quoted to in the text/Table 1
radii = [
    ("Cone Search",                 40.00, "red",          "--",      None),
    (r"$R_c = {:.2f}\,$arcmin",       1.96, "darkblue",     "solid",   "Rc"),
    (r"$R_{{hl}} = {:.2f}\,$arcmin",  6.02, "darkred",      "-.",      "Rhl"),
    (r"$R_{{hm}} = {:.2f}\,$arcmin",  6.26, "magenta",      "solid",   "Rhm"),
    (r"$R_{{\mathrm{{Hill}}}} = {:.1f}\,$arcmin",33.6,"yellow",       "dashdot", "Rhill"),
    (r"$R_{{bound}} = {:.1f}\,$arcmin",42.45,"orangered",   "solid",   "Rbound"),
]

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw={"projection": wcs})
ax.imshow(data, cmap="viridis", aspect="equal")
tr = ax.get_transform("world")
ax.scatter(ra, dec, color="darkred", transform=tr, alpha=0.6, s=10, zorder=4,
           label="NGC 6383 sources")
ax.scatter([cra], [cdec], marker="1", s=400, color="blue", transform=tr, lw=1,
           alpha=0.85, label="Center")

for lab, val, col, ls, _ in radii:
    cx, cy = (sra, sdec) if lab == "Cone Search" else (cra, cdec)
    label = lab if lab == "Cone Search" else lab.format(val)
    circ = SphericalCircle((cx * u.deg, cy * u.deg), val * u.arcmin, edgecolor=col,
                           facecolor="none", transform=tr, label=label)
    circ.set(linestyle=ls, alpha=0.85, linewidth=1.6)
    ax.add_patch(circ)

ax.set_autoscale_on(False)
ax.set_xlabel(r"$\alpha$ [deg]", fontsize=16)
ax.set_ylabel(r"$\delta$ [deg]", fontsize=16)
ax.coords[0].set_major_formatter("d.dd")
ax.coords[1].set_major_formatter("d.dd")
ax.coords.grid(True, color="white", linestyle="dotted", alpha=0.5)  # WCS celestial grid (standard on sky charts)
ax.tick_params(axis="both", which="both", direction="in")
ax.set_aspect("equal")

scalebar_angle = (5 * u.pc / (D_KPC * u.kpc)).to(u.deg, equivalencies=u.dimensionless_angles())
add_scalebar(ax, scalebar_angle, label="5 pc", color="white", corner="bottom right")
ax.legend(loc="upper left")
fig.savefig(FIG + "real_sky.pdf", bbox_inches="tight", dpi=200)
plt.close()
print("wrote real_sky.pdf  (R_hm=6.26, R_t not drawn/outside frame, R_Hill 33.6)")
