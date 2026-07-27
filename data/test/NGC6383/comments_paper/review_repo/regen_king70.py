#!/usr/bin/env python3
"""Generate king_profile_cone70.pdf for Appendix D: King fit to the 70-arcmin
extraction (idata_king_cone70_modpriors.nc, N=628), showing that this window
contains a genuine background annulus beyond the cluster and fully encloses the
R_t posterior (16-84% interval inside the data). Same styling as regen_king.py."""
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
t=Table.read(f"{B}/comments_paper/radius_robustness/generated/70/paperfaithful_reference_p06.ecsv")
center=SkyCoord(263.6826*u.deg,-32.5838*u.deg)
prof=structure.radial_density_profile(t,center,method="equip")
r=np.asarray(prof.radius.to(u.arcmin).value if hasattr(prof.radius,"to") else prof.radius,float)
rho=np.asarray(prof.density,float)

def king(rr,k,b,Rc,Rt):
    m=k*((1/np.sqrt(1+(rr/Rc)**2))-(1/np.sqrt(1+(Rt/Rc)**2)))**2+b
    return np.where(rr<=Rt,m,b)

post=az.from_netcdf(f"{B}/comments_paper/review_repo/idata_king_cone70_modpriors.nc").posterior
ks=post["k"].values.ravel(); bs=post["b"].values.ravel()
rcs=post["R_c"].values.ravel(); rts=post["R_t"].values.ravel()
Kp,Bp,Rcp,Rtp=[float(np.median(x)) for x in (ks,bs,rcs,rts)]
rr=np.geomspace(0.2,75,600)
curves=np.array([king(rr,ks[i],bs[i],rcs[i],rts[i]) for i in range(len(ks))])
lo,hi=np.percentile(curves,[16,84],axis=0)

fig,ax=plt.subplots(figsize=(7,5.6),layout="tight")
ax.fill_between(rr,lo,hi,color="#0077BB",alpha=0.18,lw=0,label=r"King model $1\sigma$ band")
ax.plot(rr,king(rr,Kp,Bp,Rcp,Rtp),color="#0077BB",lw=2.0,label="King model (median)")
ax.errorbar(r,rho,fmt="o",color="black",ms=5,zorder=5,label="Observed density")
ax.axvline(Rtp,color="black",ls=":",lw=1.6,label=rf"$R_t = {Rtp:.0f}\,\mathrm{{arcmin}}$ (adopted)")
ax.axhline(Bp,color="0.45",ls="-.",lw=1.4,label=rf"$b = {Bp:.3f}\,\mathrm{{arcmin^{{-2}}}}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$r$ [arcmin]",fontsize=14)
ax.set_ylabel(r"$\rho$ [stars arcmin$^{-2}$]",fontsize=14)
ax.legend(loc="lower left")
fig.savefig(FIG+"king_profile_cone70.pdf",bbox_inches="tight")
print(f"wrote king_profile_cone70.pdf (medians: k={Kp:.2f} b={Bp:.4f} Rc={Rcp:.2f} Rt={Rtp:.1f})")
