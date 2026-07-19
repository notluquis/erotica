#!/usr/bin/env python3
"""Standalone regeneration of pms_stats.pdf (Fig. C.2), replacing the notebook
version. Three stacked step-histogram panels of the Sagitta outputs (PMS
probability, log age, visual extinction A_V) for the reference sample (p>=0.6),
split into all / PMS (>=0.6) / non-PMS (<0.6) / no-2MASS. Shares the common
figure styling with the other regen_*.py scripts."""
import numpy as np
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B = "/Users/notluquis/COSMIC/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"

t = Table.read(f"{B}/comments_paper/cds_final/ngc6383_members.ecsv").to_pandas()
ref = t[t.Ref == 1]
no2m = ref.Jmag.isna() & ref.Hmag.isna() & ref.Ksmag.isna()
groups = [                                            # (label, mask, color, linestyle)
    ("All Data",     np.ones(len(ref), bool), "#666666", "-"),
    ("PMS",          ref.PMSProb >= 0.6,      "#EE7733", "-"),
    ("Non-PMS",      ref.PMSProb < 0.6,       "#0077BB", "--"),
    ("No 2MASS Info", no2m.values,            "#009988", ":"),
]
panels = [("PMSProb", "PMS Probability"), ("logAgeSag", r"$\log(\mathrm{Age})$"), ("AvSag", r"$A_V$")]

fig, axs = plt.subplots(3, 1, figsize=(7, 12), layout="tight")
for ax, (col, xl) in zip(axs, panels):
    allv = ref[col].dropna().values
    bins = np.histogram_bin_edges(allv, bins="auto")
    for lab, m, c, ls in groups:
        v = ref.loc[m, col].dropna().values
        if len(v) == 0:
            continue
        ax.hist(v, bins=bins, histtype="step", color=c, ls=ls, lw=1.8, label=lab)
    ax.set_xlabel(xl)
    ax.set_ylabel("Count")
axs[1].legend(loc="upper right")                      # middle-panel legend applies to all
fig.savefig(FIG + "pms_stats.pdf", bbox_inches="tight")
plt.close()
print("wrote pms_stats.pdf  (N=%d, no2MASS=%d)" % (len(ref), int(no2m.sum())))
