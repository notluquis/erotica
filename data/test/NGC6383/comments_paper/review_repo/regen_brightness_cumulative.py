#!/usr/bin/env python3
"""Regenerate cumulative_by_brightness_paper.pdf with the CVD-safe standard
(Tol colours + distinct line styles), matching the layout of the original."""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B="/Users/notluquis/COSMIC/data/test/NGC6383"
FIG=f"{B}/comments_paper/submission_package/clean_source/Figures/"
t=Table.read(f"{B}/comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv")
c=SkyCoord(263.6826*u.deg,-32.5838*u.deg)
sc=SkyCoord(np.asarray(t["ra"])*u.deg,np.asarray(t["dec"])*u.deg)
dc=angular_separation(sc.ra,sc.dec,c.ra,c.dec).to(u.arcmin).value
g=np.asarray(t["Gmag"],float)
q=np.percentile(np.sort(g),[0,25,50,75,100])
print("quartiles:",np.round(q,2))
CB=["#0077BB","#EE7733","#009988","#CC3311"]; LS=["-","--",":","-."]
rad=np.linspace(0,dc.max(),400)
fig,ax=plt.subplots(figsize=(7.2,7),layout="tight")
for j in range(4):
    sel=(g>=q[j])&(g<q[j+1]) if j<3 else (g>=q[j])&(g<=q[j+1])
    di=dc[sel]
    cu=np.array([np.sum(di<=r) for r in rad],float); cu/=cu.max()
    ax.plot(rad,cu,color=CB[j],ls=LS[j],lw=1.9,label=fr"$G_{{\rm mag}}$: {q[j]:.2f} to {q[j+1]:.2f} mag")
ax.axvline(1.96,color="black",ls="-.",lw=1.3,label=r"$R_c = 1.96$ arcmin")
ax.axvline(33.6,color="black",ls=":",lw=1.6,label=r"$R_{\mathrm{Hill}} = 33.6$ arcmin")
ax.set_xlabel("Radius [arcmin]",fontsize=15); ax.set_ylabel("Normalized cumulative count",fontsize=15)
ax.legend(loc="lower right")
fig.savefig(FIG+"cumulative_by_brightness_paper.pdf",bbox_inches="tight")
print("wrote cumulative_by_brightness_paper.pdf")
