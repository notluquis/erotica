#!/usr/bin/env python3
"""Rebuild parallax_determination_paper.pdf (3-panel) matching the paper caption,
with improved legend (#18: delta-omega/omega thresholds in the legend box).
Fit values are the paper's published results (no MCMC re-run needed)."""
import numpy as np, pandas as pd
from astropy.table import Table
from scipy.stats import norm
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt, matplotlib.ticker as ticker

B="/Users/notluquis/erotica/data/test/NGC6383"
FIG=f"{B}/comments_paper/submission_package/clean_source/Figures/"
# reference sample (254): designation, parallax, parallax_error, Gmag
ast=pd.read_csv(f"{B}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv",sep=r"\s+")
# Bailer-Jones geometric distances from the full catalog, matched by source_id
cl=Table.read(f"{B}/data/40/clustering_results.ecsv").to_pandas()[["source_id","r_med_geo"]]
ast["source_id"]=ast["designation"].astype(str).str.extract(r"(\d+)$")[0].astype("int64")
df=ast.merge(cl,on="source_id",how="left")
print("matched r_med_geo:",df["r_med_geo"].notna().sum(),"/",len(df))

# paper fit values
mu_plx, sig_plx, sig_mean = 0.908, 0.046, 0.004     # mas
mu_d, std_d = 1.110, 0.060                            # kpc (mode + std)

frac=df["parallax_error"]/df["parallax"]
use=frac<=0.1; non=~use
fig,ax=plt.subplots(1,3,figsize=(18,5.5),constrained_layout=True)

# --- panel 0: parallax histogram ---
a=ax[0]
a.axvline(mu_plx,color="black",ls="-.",lw=1,label=r"$\mu_\varpi={:.3f}\pm{:.3f}$ mas".format(mu_plx,sig_mean))
_,bins,_=a.hist(df.parallax[use],bins="auto",color="#0077BB",histtype="step",lw=2.0,ls="-",label=r"$\delta\varpi/\varpi<0.1$")
a.hist(df.parallax,bins=bins,color="#999999",histtype="step",lw=1.0,ls="--",label="all")
a.hist(df.parallax[non],bins=bins,color="#EE7733",histtype="step",lw=1.5,ls=":",label=r"$\delta\varpi/\varpi>0.1$")
bw=np.diff(bins)[0]; x=np.linspace(mu_plx-3*sig_plx,mu_plx+3*sig_plx,200)
a.plot(x,norm.pdf(x,mu_plx,sig_plx)*use.sum()*bw,color="black",lw=1.2,label="Gaussian fit")
# axvspan spans the full axes height (y in axes coords) regardless of ylim, so the
# band always reaches the top; fill_betweenx tied to a fixed/expanded ylim did not
a.axvspan(mu_plx-sig_mean,mu_plx+sig_mean,color="#999999",alpha=0.2,lw=0,label=r"$\mu_\varpi$ standard error")
a.set_ylim(0,a.get_ylim()[1]); a.set_xlabel(r"$\varpi$ [mas]",fontsize=15); a.set_ylabel("Counts",fontsize=15)
a.legend(loc="upper left",fontsize=11)

# --- panel 1: G vs parallax ---
a=ax[1]
a.scatter(df.parallax[use],df.Gmag[use],s=12,c="#0077BB",marker="o",label=r"$\delta\varpi/\varpi<0.1$")
a.scatter(df.parallax[non],df.Gmag[non],s=14,c="#EE7733",marker="^",label=r"$\delta\varpi/\varpi>0.1$")
a.errorbar(df.parallax,df.Gmag,xerr=df.parallax_error,fmt="none",alpha=0.3,elinewidth=0.5,color="gray",label="Parallax uncertainty")
a.axvline(mu_plx,color="black",ls="-.",lw=1,label=r"$\mu_\varpi={:.3f}\pm{:.3f}$ mas".format(mu_plx,sig_mean))
a.set_xlim(mu_plx-15*sig_plx,mu_plx+15*sig_plx)
a.axvspan(mu_plx-sig_plx,mu_plx+sig_plx,color="#999999",alpha=0.2,lw=0,label=r"$1\sigma$ parallax dispersion")
a.set_xlabel(r"$\varpi$ [mas]",fontsize=15); a.set_ylabel(r"$G_\mathrm{mag}$",fontsize=15)
a.invert_yaxis(); a.legend(loc="upper left",fontsize=11)

# --- panel 2: Bailer-Jones geometric distance histogram ---
a=ax[2]
d_use=df.r_med_geo[use]/1000.0; d_all=df.r_med_geo/1000.0; d_non=df.r_med_geo[non]/1000.0  # pc->kpc
sel=(d_all>0.8*np.nanmin(d_use))&(d_all<1.2*np.nanmax(d_use))
_,bins,_=a.hist(d_use.dropna(),bins="auto",color="#0077BB",histtype="step",lw=2.0,ls="-",label=r"$\delta\varpi/\varpi<0.1$")
a.hist(d_all.dropna(),bins=bins,color="#999999",histtype="step",lw=1.0,ls="--",label="all")
a.hist(d_non.dropna(),bins=bins,color="#EE7733",histtype="step",lw=1.5,ls=":",label=r"$\delta\varpi/\varpi>0.1$")
a.axvline(mu_d,color="black",ls="-.",lw=1,label=r"$\mu_d={:.3f}\pm{:.3f}$ kpc".format(mu_d,std_d))
a.axvspan(mu_d-std_d,mu_d+std_d,color="#999999",alpha=0.2,lw=0,label=r"$\mu_d$ dispersion")
a.set_ylim(0,a.get_ylim()[1]); a.set_xlim(0.8,1.5)
a.set_xlabel("Distance [kpc]",fontsize=15); a.set_ylabel("Counts",fontsize=15)
a.legend(loc="upper left",fontsize=11)

for a in ax:
    a.xaxis.set_minor_locator(ticker.AutoMinorLocator()); a.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    a.tick_params(axis="both",which="both",direction="in")
fig.savefig(FIG+"parallax_determination_paper.pdf",bbox_inches="tight")
print("wrote parallax_determination_paper.pdf | useful=%d nonuseful=%d"%(use.sum(),non.sum()))
