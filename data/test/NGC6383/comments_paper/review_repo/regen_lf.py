#!/usr/bin/env python3
"""Standalone regeneration of luminosity_function.pdf (Fig. 9), replacing the
notebook version. Single-panel histogram of absolute G magnitudes (G_abs =
Gmag - distance modulus) for the reference sample (p>=0.6), orange hatched.
Shares the common figure styling (tick/label/legend sizes) with the other
regen_*.py scripts so all paper figures are visually consistent."""
import numpy as np
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B = "/Users/notluquis/COSMIC/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"
DM = 10.30                                           # adopted distance modulus (Table 1)

t = Table.read(f"{B}/comments_paper/cds_final/ngc6383_members.ecsv").to_pandas()
ref = t[t.Ref == 1]                                  # reference sample (p>=0.6), N=254
gabs = ref.Gmag.values - DM

fig, ax = plt.subplots(figsize=(7, 6), layout="tight")
bins = np.arange(np.floor(gabs.min()), np.ceil(gabs.max()) + 1.0, 1.0)
ax.hist(gabs, bins=bins, histtype="stepfilled", facecolor="#EE7733", alpha=0.35,
        edgecolor="#EE7733", lw=1.4, hatch="//")
ax.set_xlabel(r"$G_{\mathrm{abs}}$ [mag]")
ax.set_ylabel("Counts")
fig.savefig(FIG + "luminosity_function.pdf", bbox_inches="tight")
plt.close()
print("wrote luminosity_function.pdf  (N=%d, DM=%.2f)" % (len(ref), DM))
