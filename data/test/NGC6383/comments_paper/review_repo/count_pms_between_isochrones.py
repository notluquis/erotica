#!/usr/bin/env python3
"""Count the PMS candidates of Fig. 7 (left panel) between the log(age)=6.20 and 6.80 isochrones.

Round 3 (2026-09-14): backs "about 60% of the PMS candidates ... lie between them" in Sect. 6.1 (CMD paragraph).
Prints 48/80 (18 young side, 14 old side). An independent re-check varied the matching (+-0.05, +-0.30 mag,
median instead of max, interpolation over all 87): the fraction stays at 60-64%, the young/old split
ranges 14-23 / 9-16, so only the fraction is quoted. The 7 stars dropped here fall in a G gap of the
isochrone sampling (G ~ 14.7-15.5), not outside its range.
Same cluster, isochrones and Av/dm as regen_cmd_figs.py (whose header this reuses); writes nothing.
A star is on the young side of an isochrone if it is redder than the reddest isochrone point within
0.15 mag in G. Stars outside the isochrones' G range are not counted. Run with the cosmic env (asteca 0.6.9).

Original header of regen_cmd_figs.py follows:
Regenerate ngc6383_cmd.pdf + ngc6383_cmd_various.pdf, CVD/grayscale-safe:
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

B = "/Users/notluquis/erotica/data/test/NGC6383"
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

import arviz as az

# Young-side census: for each isochrone age, how many PMS candidates (Sagitta PMSProb>=0.6, with 2MASS)
# lie redder than the isochrone at their own G (i.e. on the young side), in G vs BP-RP.
def color_at(P, gmag):
    G, c = [np.asarray(v) for v in locus(P, 0)]
    # PMS/low-mass part: faint end, where color increases with G; use the reddest branch per G bin
    out = np.full(len(gmag), np.nan)
    for i, gi in enumerate(gmag):
        near = np.abs(G - gi) < 0.15
        if near.any(): out[i] = c[near].max()
    return out
sel = pms
P = lambda a: {"met": 0.024, "loga": a, "alpha": 0.09, "beta": 0.94, "Rv": 3.1, "DR": 0.0, "Av": 1.24, "dm": 10.30}
c62 = color_at(P(6.2), g[sel]); c68 = color_at(P(6.8), g[sel]); ok = np.isfinite(c62) & np.isfinite(c68)
x = bprp[sel][ok]
between = ((x <= c62[ok]) & (x >= c68[ok])).sum()
print("PMS plotted:", sel.sum(), "with both isochrones defined:", ok.sum())
print(f"between 6.20 and 6.80: {between}/{ok.sum()} = {100*between/ok.sum():.1f}%  | younger than 6.20: {(x > c62[ok]).sum()} | older than 6.80: {(x < c68[ok]).sum()}")
