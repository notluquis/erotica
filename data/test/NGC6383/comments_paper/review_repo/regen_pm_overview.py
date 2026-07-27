#!/usr/bin/env python3
"""Generate pm_radec_overview.pdf: the full HDBSCAN clustering panorama of the
40 arcmin preprocessed sample (15276 sources), in the proper-motion plane (left)
and on the sky (right). Three groups from the clusterer's own labels_: the
selected NGC 6383 branch (label 0), the other HDBSCAN branches, and the field /
noise (label -1). Shows that NGC 6383 is a compact proper-motion overdensity
clearly separated from the diffuse field, while on the sky all groups overlap.
Shares the common figure styling with the other regen_*.py scripts."""
import sys
sys.path.insert(0, "/Users/notluquis/erotica")
import dill, numpy as np
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B = "/Users/notluquis/erotica/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"
clu = dill.load(open(f"{B}/comments_paper/radius_robustness/generated/dill/ngc6383_40_paperfaithful.dill", "rb"))

d = clu.data.to_pandas()
lab = np.asarray(clu.clusterer.labels_)
NGC = clu.pseudoprobability_selected_.get("label", 0)          # selected NGC branch label (0)
noise = lab == -1
ngc = lab == NGC
other = (~noise) & (~ngc)

fig, (axp, axs) = plt.subplots(1, 2, figsize=(14, 6), layout="tight")

def draw(ax, x, y):
    ax.scatter(x[noise], y[noise], s=3, c="0.7", alpha=0.30, lw=0, label=f"Field / noise ({noise.sum()})", zorder=1)
    ax.scatter(x[other], y[other], s=11, c="#0077BB", alpha=0.8, lw=0, label=f"Other HDBSCAN branches ({other.sum()})", zorder=2)
    ax.scatter(x[ngc], y[ngc], s=13, c="#CC3311", lw=0, label=f"NGC 6383 (selected, {ngc.sum()})", zorder=3)

# proper-motion plane (robust limits so the cluster knot is visible amid the field)
draw(axp, d.pmra.values, d.pmdec.values)
xlo, xhi = np.percentile(d.pmra, [0.5, 99.5]); ylo, yhi = np.percentile(d.pmdec, [0.5, 99.5])
axp.set_xlim(xlo, xhi); axp.set_ylim(ylo, yhi)
axp.set_xlabel(r"$\mu_{\alpha}^{*}$ [mas yr$^{-1}$]"); axp.set_ylabel(r"$\mu_{\delta}$ [mas yr$^{-1}$]")
axp.legend(loc="upper left", markerscale=1.6, framealpha=0.9)

# on-sky distribution
draw(axs, d.ra.values, d.dec.values)
axs.invert_xaxis()
axs.set_xlabel(r"$\alpha$ [deg]"); axs.set_ylabel(r"$\delta$ [deg]")

fig.savefig(FIG + "pm_radec_overview.pdf", bbox_inches="tight")
plt.close()
print("wrote pm_radec_overview.pdf  (NGC=%d, other=%d, noise=%d)" % (ngc.sum(), other.sum(), noise.sum()))
