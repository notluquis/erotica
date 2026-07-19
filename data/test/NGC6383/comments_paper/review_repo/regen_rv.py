#!/usr/bin/env python3
"""Regenerate radial_velocity_amplitude.pdf: sequential viridis for amplitude
(was diverging coolwarm), open gray diamonds for missing amplitudes (was yellow
filled circles -> near-white in grayscale), RV error bars on all sources."""
import numpy as np
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt
B="/Users/notluquis/COSMIC/data/test/NGC6383"
FIG=f"{B}/comments_paper/submission_package/clean_source/Figures/"
t=Table.read(f"{B}/comments_paper/review_repo/rv_amplitude_dr3.ecsv").to_pandas()
amp=t.rv_amplitude_robust.values; has=np.isfinite(amp)
fig,ax=plt.subplots(figsize=(7,5.4),layout="tight")
ax.errorbar(t.phot_g_mean_mag[~has],t.radial_velocity[~has],yerr=t.radial_velocity_error[~has],
            fmt="D",mfc="none",mec="0.35",ecolor="0.6",ms=7,lw=1,label="No amplitude measurement")
sc=ax.scatter(t.phot_g_mean_mag[has],t.radial_velocity[has],c=amp[has],cmap="viridis",
              s=70,zorder=5,edgecolors="black",linewidths=0.4,label="With amplitude")
ax.errorbar(t.phot_g_mean_mag[has],t.radial_velocity[has],yerr=t.radial_velocity_error[has],
            fmt="none",ecolor="0.6",lw=1,zorder=4)
cb=fig.colorbar(sc,pad=0.01); cb.set_label(r"RV amplitude [km s$^{-1}$]",fontsize=13)
ax.set_xlabel(r"$G$ [mag]",fontsize=14); ax.set_ylabel(r"Radial velocity [km s$^{-1}$]",fontsize=14)
ax.legend(loc="upper left")
fig.savefig(FIG+"radial_velocity_amplitude.pdf",bbox_inches="tight")
print("wrote radial_velocity_amplitude.pdf  n=%d (amp: %d)"%(len(t),has.sum()))
