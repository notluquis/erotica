#!/usr/bin/env python3
"""Regenerate ngc6383_cmd.pdf + ngc6383_cmd_various.pdf, CVD/grayscale-safe:
PMS = orange filled, non-PMS = blue filled, no-2MASS = open black (shape+fill
channel); members=stars, probable=diamonds; isochrones black w/ distinct styles
(cmd) and viridis-ordered + distinct styles (cmd_various); fixes the
'G_RP - G_RP' axis-label typo of the original."""
import numpy as np, pandas as pd, asteca
from astropy.table import Table
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt
from matplotlib import cm

B = "/Users/notluquis/COSMIC/data/test/NGC6383"
FIG = f"{B}/comments_paper/submission_package/clean_source/Figures/"
cds = Table.read(f"{B}/comments_paper/cds_final/ngc6383_members.ecsv").to_pandas()
ref = cds[cds.Ref == 1].reset_index(drop=True)            # 254
bprp = (ref.BPmag - ref.RPmag).values
g = ref.Gmag.values
rpj = (ref.RPmag - ref.Jmag).values
no2m = (ref.Jmag.isna() & ref.Hmag.isna() & ref.Ksmag.isna()).values
pms = (ref.PMSProb >= 0.6).values & ~no2m
non = (ref.PMSProb < 0.6).values & ~no2m
mem = (ref.pMember >= 0.8).values
prob = ~mem
ORANGE = "#EE7733"
BLUE = "#0077BB"

df = pd.read_csv(f"{B}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv", sep=r"\s+")
e2 = np.full(len(ref), 0.05)
clu = asteca.Cluster(ra=ref.RAdeg.values, dec=ref.DEdeg.values,
    mag=g, e_mag=df.e_Gmag.values, color=bprp, e_color=df.e_BP_RP.values,
    color2=rpj, e_color2=e2)
iso = asteca.Isochrones(model="MIST", isochs_path=f"{B}/MIST/UBVRIplus/",
    mag="Gaia_G_EDR3", color=("Gaia_BP_EDR3", "Gaia_RP_EDR3"),
    color2=("Gaia_RP_EDR3", "2MASS_J"),
    magnitude_effl=6390.0, color_effl=(5320.0, 7970.0), color2_effl=(7970.0, 12350.0),
    z_to_FeH=0.0152)
synth = asteca.Synthetic(iso, def_params={"met": 0.024, "loga": 6.55, "alpha": 0.09,
    "beta": 0.94, "Rv": 3.1, "DR": 0.0, "Av": 1.24, "dm": 10.3},
    IMF_name="chabrier_2014", gamma="D&K", seed=42)
synth.calibrate(clu)

def locus(params, idx=0):
    return synth.get_isochrone(params, color_idx=idx)

MODE = {"met": 0.025, "loga": 6.553, "alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0.0, "Av": 1.136, "dm": 10.282}
MEAN = {"met": 0.023, "loga": 6.491, "alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0.0, "Av": 1.254, "dm": 10.255}
MEDIAN = {"met": 0.024, "loga": 6.489, "alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0.0, "Av": 1.252, "dm": 10.260}
t = locus(MODE)
print("get_isochrone ->", np.shape(t))
# save the clean mode-fit locus so the mass/binary CMD can overlay the same smooth
# isochrone (avoids the zig-zag of connecting the scattered synthetic sample)
np.savez(FIG + "mode_locus.npz", G=np.asarray(t[0]), bprp=np.asarray(t[1]))

import arviz as az
ida = az.from_netcdf(f"{B}/data/40/fit_parameters_trace_1724708835.nc")
post = ida.posterior
rng = np.random.default_rng(9)
ii = rng.integers(0, post.sizes["chain"], 150)
jj = rng.integers(0, post.sizes["draw"], 150)
samples = [{"met": float(post["met"][i, j]), "loga": float(post["loga"][i, j]),
            "Av": float(post["Av"][i, j]), "dm": float(post["dm"][i, j]),
            "alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0.0} for i, j in zip(ii, jj)]

# ---------- Figure: ngc6383_cmd ----------
fig, ax = plt.subplots(figsize=(7, 7), layout="tight")
for p in samples:
    try:
        l = locus(p)
        ax.plot(l[1], l[0], color="0.55", alpha=0.05, lw=0.9, zorder=1)
    except Exception:
        pass
l = locus(MODE);   ax.plot(l[1], l[0], color="black", ls="-",  lw=1.9, zorder=3, label="Best mode fit")
l = locus(MEAN);   ax.plot(l[1], l[0], color="black", ls="--", lw=1.4, zorder=3, label="Best mean fit")
l = locus(MEDIAN); ax.plot(l[1], l[0], color="black", ls=":",  lw=1.6, zorder=3, label="Best median fit")
ax.scatter(bprp[pms & mem],   g[pms & mem],   s=95, marker="*", c=ORANGE, lw=0.4, edgecolors="black", zorder=5, label="PMS members")
ax.scatter(bprp[pms & prob],  g[pms & prob],  s=40, marker="D", c=ORANGE, lw=0.4, edgecolors="black", zorder=4, label="PMS probable members")
ax.scatter(bprp[non & mem],   g[non & mem],   s=95, marker="*", c=BLUE, lw=0.4, edgecolors="black", zorder=5, label="Non-PMS members")
ax.scatter(bprp[non & prob],  g[non & prob],  s=40, marker="D", c=BLUE, lw=0.4, edgecolors="black", zorder=4, label="Non-PMS probable members")
ax.scatter(bprp[no2m & mem],  g[no2m & mem],  s=95, marker="*", facecolors="none", edgecolors="black", lw=0.9, zorder=5, label="Members (no 2MASS)")
ax.scatter(bprp[no2m & prob], g[no2m & prob], s=40, marker="D", facecolors="none", edgecolors="black", lw=0.9, zorder=4, label="Probable members (no 2MASS)")
ax.invert_yaxis()
ax.set_xlabel(r"$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]", fontsize=15)
ax.set_ylabel(r"$G$ [mag]", fontsize=15)
from matplotlib.lines import Line2D
draws_proxy = Line2D([0], [0], color="0.55", lw=1.0, alpha=0.7,
                     label="Posterior draws ($N=150$)")
h, lbl = ax.get_legend_handles_labels()
ax.legend(handles=h[:3] + [draws_proxy] + h[3:], loc="upper right")
fig.savefig(FIG + "ngc6383_cmd.pdf", bbox_inches="tight")
plt.close()
print("wrote ngc6383_cmd.pdf")

# ---------- Figure: ngc6383_cmd_various ----------
ages = [6.2, 6.4, 6.6, 6.8, 7.0]
cols = [cm.viridis(x) for x in np.linspace(0.05, 0.85, 5)]
LS = ["-", "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))]
fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.4), layout="tight")
hasJ = ~ref.Jmag.isna().values
allm = np.ones(len(ref), bool)

def panel(ax, x, y, m, xl, yl, inv=True):
    ax.scatter(x[pms & mem & m],  y[pms & mem & m],  s=80, marker="*", c=ORANGE, lw=0.3, edgecolors="black", zorder=5)
    ax.scatter(x[pms & prob & m], y[pms & prob & m], s=32, marker="D", c=ORANGE, lw=0.3, edgecolors="black", zorder=4)
    ax.scatter(x[non & mem & m],  y[non & mem & m],  s=80, marker="*", c=BLUE, lw=0.3, edgecolors="black", zorder=5)
    ax.scatter(x[non & prob & m], y[non & prob & m], s=32, marker="D", c=BLUE, lw=0.3, edgecolors="black", zorder=4)
    if inv:
        ax.invert_yaxis()
    ax.set_xlabel(xl, fontsize=14)
    ax.set_ylabel(yl, fontsize=14)

# clip isochrones to the observed magnitude range (the brightest member is G~8.8).
# ASteCA returns the full isochrone at MIST grid-node ages (e.g. loga=6.6) but a
# high-mass-truncated isochrone at interpolated ages; clipping to G>=GCLIP makes all
# ages span a consistent range and removes the non-physical bright tail / empty axis.
GCLIP = 7.0
for a, cstyle, lst in zip(ages, cols, LS):
    P = {"met": 0.024, "loga": a, "alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0.0, "Av": 1.24, "dm": 10.30}
    l0 = locus(P, 0)
    l1 = locus(P, 1)
    G0 = np.asarray(l0[0]); mk = G0 >= GCLIP
    axs[0].plot(np.asarray(l0[1])[mk], G0[mk], color=cstyle, ls=lst, lw=1.7, zorder=3, label=rf"$\log(\mathrm{{age}})={a}$")
    axs[1].plot(np.asarray(l1[1])[mk], np.asarray(l1[0])[mk], color=cstyle, ls=lst, lw=1.7, zorder=3)
    axs[2].plot(np.asarray(l0[1])[mk], np.asarray(l1[1])[mk], color=cstyle, ls=lst, lw=1.7, zorder=3)
panel(axs[0], bprp, g, allm, r"$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]", r"$G$ [mag]")
panel(axs[1], rpj, g, hasJ, r"$G_{\mathrm{RP}}-J$ [mag]", r"$G$ [mag]")
panel(axs[2], bprp, rpj, hasJ, r"$G_{\mathrm{BP}}-G_{\mathrm{RP}}$ [mag]", r"$G_{\mathrm{RP}}-J$ [mag]", inv=False)
axs[0].legend(loc="upper right", title=r"$Z_{\mathrm{ini}}=0.024$")
fig.savefig(FIG + "ngc6383_cmd_various.pdf", bbox_inches="tight")
plt.close()
print("wrote ngc6383_cmd_various.pdf")
