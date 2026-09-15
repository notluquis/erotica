#!/usr/bin/env python3
"""Standalone regeneration of pms_stats.pdf (Fig. C.4), replacing the notebook
version. Three step-histogram panels of the Sagitta outputs (PMS
probability, log age, visual extinction A_V) for the reference sample (p>=0.6),
split into all / PMS (>=0.6) / non-PMS (<0.6) / no-2MASS. Shares the common
figure styling with the other regen_*.py scripts.

2026-09-15 (post-acceptance layout pass): the three panels were stacked 3x1
(upper/middle/lower), which made the appendix figure run tall and nearly bare
on its own page. Laid out 1x3 (left/middle/right) instead, so the caption's
"upper/middle/lower panel" wording moved to "left/middle/right panel" -- see
aanda.tex. No hexbin or dense scatter here (step histograms only), so the
matplotlib 3.11 rasterized-hexbin bug documented in regen_corner.py does not
apply to this figure."""
import numpy as np
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B = "/Users/notluquis/erotica/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"

t = Table.read(f"{B}/comments_paper/cds_final/ngc6383_members.ecsv").to_pandas()
ref = t[t.Ref == 1]
no2m = ref.Jmag.isna() & ref.Hmag.isna() & ref.Ksmag.isna()
groups = [                                            # (label, mask, color, linestyle)
    ("All data",     np.ones(len(ref), bool), "#666666", "-"),
    ("PMS",          ref.PMSProb >= 0.6,      "#EE7733", "-"),
    ("Non-PMS",      ref.PMSProb < 0.6,       "#0077BB", "--"),
    ("No 2MASS data", no2m.values,            "#009988", ":"),
]
panels = [("PMSProb", "PMS probability"), ("logAgeSag", r"$\log(\mathrm{age})$"), ("AvSag", r"$A_V$")]

fig, axs = plt.subplots(1, 3, figsize=(15, 4.5), layout="tight")
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
