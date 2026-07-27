#!/usr/bin/env python3
"""Regenerate king_profile_logscale.pdf CVD-safe: black data points, blue King
curve + posterior band (from audited idata), black dashed/dotted Rc/Rt verticals,
gray dash-dot background level. Published values for lines/legend."""
import sys, numpy as np
sys.path.insert(0,"/Users/notluquis/erotica")
import astropy.units as u, arviz as az
from astropy.table import Table
from astropy.coordinates import SkyCoord
from erotica.analysis import structure
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt

B="/Users/notluquis/erotica/data/test/NGC6383"
FIG=f"{B}/comments_paper/submission_package/clean_source/Figures/"
t=Table.read(f"{B}/comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv")
center=SkyCoord(263.6826*u.deg,-32.5838*u.deg)
prof=structure.radial_density_profile(t,center,method="equip")
r=np.asarray(prof.radius.to(u.arcmin).value if hasattr(prof.radius,"to") else prof.radius,float)
rho=np.asarray(prof.density,float)

def king(rr,k,b,Rc,Rt):
    m=k*((1/np.sqrt(1+(rr/Rc)**2))-(1/np.sqrt(1+(Rt/Rc)**2)))**2+b
    return np.where(rr<=Rt,m,b)

Kp,Bp,Rcp,Rtp=4.918,0.0110,1.959,40.34   # published (posterior medians, 40-arcmin fit)
idata=az.from_netcdf(f"{B}/comments_paper/review_repo/idata_king.nc")
post=idata.posterior
ks=post["k"].values.ravel(); bs=post["b"].values.ravel()
rcs=post["R_c"].values.ravel(); rts=post["R_t"].values.ravel()
# use the full posterior (8000 draws) and a fine radial grid so the 16-84 band
# is smooth; with only ~200 samples the percentile envelope looked jagged/"bitten",
# especially where individual King curves truncate at their own R_t.
sel=np.arange(len(ks))
rr=np.geomspace(0.2,55,600)
curves=np.array([king(rr,ks[i],bs[i],rcs[i],rts[i]) for i in sel])
lo,hi=np.percentile(curves,[16,84],axis=0)

fig,ax=plt.subplots(figsize=(7,5.6),layout="tight")
ax.fill_between(rr,lo,hi,color="#0077BB",alpha=0.18,lw=0,label=r"King model $1\sigma$ band")
ax.plot(rr,king(rr,Kp,Bp,Rcp,Rtp),color="#0077BB",lw=2.0,label="King model (median)")
ax.errorbar(r,rho,yerr=np.sqrt(np.maximum(rho,1e-3)/ (np.pi*(np.gradient(r)*2*r))) if False else None,
            fmt="o",color="black",ms=5,zorder=5,label="Observed density")
ax.axvline(Rcp,color="black",ls="--",lw=1.4,label=rf"$R_c = {Rcp:.2f}\,\mathrm{{arcmin}}$")
ax.axvline(Rtp,color="black",ls=":",lw=1.6,label=rf"$R_t = {Rtp:.0f}\,\mathrm{{arcmin}}$ (this window)")
ax.axhline(Bp,color="0.45",ls="-.",lw=1.4,label=rf"$b = {Bp:.3f}\,\mathrm{{arcmin^{{-2}}}}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$r$ [arcmin]",fontsize=14)
ax.set_ylabel(r"$\rho$ [stars arcmin$^{-2}$]",fontsize=14)
ax.legend(loc="lower left")
fig.savefig(FIG+"king_profile_logscale.pdf",bbox_inches="tight")
print("wrote king_profile_logscale.pdf")
