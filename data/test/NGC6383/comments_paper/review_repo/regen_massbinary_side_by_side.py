#!/usr/bin/env python3
"""Regenerate ngc6383_mass_binary.pdf as a 1x2 (left/right) layout instead of
the original 2x1 (upper/lower) stack, to shrink its printed height for the
appendix whitespace compaction (A&A aa52082-24, editor: "reduce figures when
needed"). Same data and colormaps as regen_prob_massbinary.py (the only
change is the subplot layout, figsize, and rasterizing the star markers).
Data from saved ASteCA output + masses CSV + mode-fit isochrone locus (no
pipeline re-run)."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B = "/Users/notluquis/erotica/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"

ast = pd.read_csv(f"{B}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv", sep=r"\s+").reset_index(drop=True)
ms = pd.read_csv(f"{B}/data/40/masses_asteca_069.csv")
bp = ms.binar_prob.values
mass = ms.m1.values + np.where(bp > 0.7, np.nan_to_num(ms.m2.values), 0.0)
G = ast.Gmag.values
BPRP = ast.BP_RP.values
_loc = np.load(f"{FIG}mode_locus.npz")
iso_g = _loc["G"]
iso_bprp = _loc["bprp"]

fig, axs = plt.subplots(1, 2, figsize=(16, 6.6), layout="tight")
for ax, (c, lab, cm) in zip(axs, [(mass, r"Total mass $[M_\odot]$", "viridis_r"), (bp, "Binary probability", "viridis_r")]):
    ax.plot(iso_bprp, iso_g, color="0.3", ls="--", lw=1.2, zorder=1, label="Best mode-fit isochrone")
    s = ax.scatter(BPRP, G, s=80, lw=0, marker="*", c=c, cmap=cm, zorder=5, rasterized=True)
    cb = fig.colorbar(s, ax=ax, pad=0.01)
    cb.set_label(lab, fontsize=13)
    ax.invert_yaxis()
    ax.set_xlabel(r"$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]", fontsize=14)
    ax.set_ylabel(r"$G$ [mag]", fontsize=14)
    ax.legend(loc="upper right")
fig.savefig(FIG + "ngc6383_mass_binary.pdf", bbox_inches="tight")
plt.close()
print("wrote ngc6383_mass_binary.pdf  (N=%d, viridis_r, 1x2 layout, rasterized markers)" % len(ast))
