#!/usr/bin/env python3
"""Definitive recompute + figure regeneration with ASteCA 0.6.9 masses.
Regenerates cumulative_by_mass_and_type[_mseg].pdf + mass_segregation_mst.pdf
into clean_source/Figures/, and prints every number needed for the tex."""
import numpy as np, pandas as pd, astropy.units as u, astropy.constants as const
from astropy.coordinates import SkyCoord, angular_separation
from scipy.stats import ks_2samp
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams.update({"xtick.labelsize": 13, "ytick.labelsize": 13, "axes.labelsize": 15, "legend.fontsize": 11, "axes.titlesize": 14})
import matplotlib.pyplot as plt
B="/Users/notluquis/erotica/data/test/NGC6383"
FIG=f"{B}/comments_paper/submission_package/clean_source/Figures/"
rng=np.random.default_rng(7)
ast=pd.read_csv(f"{B}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv",sep=r"\s+")
m=pd.read_csv(f"{B}/data/40/masses_asteca_069.csv")
d=ast.reset_index(drop=True).copy(); d["bp"]=m.binar_prob.values
_bin=m.binar_prob.values>0.7
d["mass"]=m.m1.values+np.where(_bin,np.nan_to_num(m.m2.values),0.0)
d["mass_std"]=np.sqrt(m.m1_std.values**2+np.where(_bin,np.nan_to_num(m.m2_std.values)**2,0.0))
c=SkyCoord(263.6826*u.deg,-32.5838*u.deg); DIST,DERR=1.110,0.060; N=254; MC,MCE=902.27,92.3
sc=SkyCoord(d.ra.values*u.deg,d.dec.values*u.deg)
d["dc"]=angular_separation(sc.ra,sc.dec,c.ra,c.dec).to(u.arcmin).value

mmean=np.nanmean(d.mass); mmax=np.nanmax(d.mass)
# error on <m>: sampling SE (pop dispersion/sqrt N) combined with per-star measurement error
emmean=np.hypot(np.nanstd(d.mass)/np.sqrt(N), np.sqrt(np.nansum(d.mass_std.values**2))/N)
_im=int(np.nanargmax(d.mass.values)); emmax=d.mass_std.values[_im]  # per-star std of most massive
o=np.argsort(d.dc.values); cm=np.cumsum(d.mass.values[o]); i=np.searchsorted(cm,cm[-1]/2)
rhm=d.dc.values[o][i]
B_=4000
def hm(df):
    s=df.dc.values; mm=df.mass.values; oo=np.argsort(s); cc=np.cumsum(mm[oo]); j=np.searchsorted(cc,cc[-1]/2)
    return s[oo][min(j,len(s)-1)]
ehm=np.std([hm(d.sample(N,replace=True,random_state=rng.integers(1e9))) for _ in range(B_)])
rhm_pc=(rhm*u.arcmin*DIST*u.kpc).to(u.pc,u.dimensionless_angles())
erhm_pc=rhm_pc.value*np.hypot(ehm/rhm,DERR/DIST)
# self-consistent: M = N*<m> of the SAME observed system (NOT the Hunt+24 total mass,
# which mixes an IMF-extrapolated M with the Gaia-detected N and implies <m>=3.55)
MOBS=N*mmean; MOBSE=N*emmean
trh=(0.17*N/np.log(0.11*N)*np.sqrt(rhm_pc**3/(const.G*MOBS*u.Msun))).to(u.Myr).value
trel=np.hypot(1.5*erhm_pc/rhm_pc.value,0.5*MOBSE/MOBS)
print(f"M_obs=N<m>={MOBS:.0f}+-{MOBSE:.0f} Msun")
tseg=mmean/mmax*trh; cutoff=mmean*trh/(10**6.55/1e6)
tseg_frac=np.sqrt((emmean/mmean)**2+(emmax/mmax)**2+trel**2)  # full propagation: <m>, m_max, t_rh
print("=== ASteCA 0.6.9 final numbers ===")
print(f"<m>={mmean:.3f}+-{emmean:.3f}  m_max={mmax:.2f}+-{emmax:.2f}")
print(f"R_hm={rhm:.2f}+-{ehm:.2f}' = {rhm_pc.value:.2f}+-{erhm_pc:.2f} pc")
print(f"t_rh={trh:.1f}+-{trh*trel:.1f} | min t_seg={tseg:.2f}+-{tseg*tseg_frac:.2f} (full prop) | cutoff={cutoff:.2f}")

def quart_ks(sub):
    q=np.percentile(np.sort(sub.mass.values),[0,25,50,75,100]); rg=[(q[k],q[k+1]) for k in range(4)]
    sd=sub[sub.bp<0.6]; g=[sd[(sd.mass>=a)&(sd.mass<b)].dc.values for a,b in rg]
    ks=[ks_2samp(g[k],g[j]) for k in range(4) for j in range(k+1,4) if len(g[k])>1 and len(g[j])>1]
    Ds=[k[0] for k in ks]; ps=[k[1] for k in ks]
    s=sub[sub.bp<0.6].dc.values; b=sub[sub.bp>=0.6].dc.values; D,p=ks_2samp(s,b)
    return q,rg,min(Ds),max(Ds),min(ps),max(ps),D,p,len(s),len(b)
q,rg,Dl,Dh,pl,ph,Dsb,psb,ns,nb=quart_ks(d)
print(f"\nFULL quartiles: {np.round(q,2)}")
print(f"FULL single quartile KS: D={Dl:.2f}-{Dh:.2f} p={pl:.2f}-{ph:.2f}")
print(f"FULL single-vs-binary: D={Dsb:.3f} p={psb:.3f} (Ns={ns} Nb={nb})")
dm=d[d.mass<=cutoff]; qm,rgm,Dlm,Dhm,plm,phm,Dsbm,psbm,nsm,nbm=quart_ks(dm)
print(f"\nMSEG n={len(dm)} ({len(dm)/N*100:.1f}%) <m>={np.nanmean(dm.mass):.3f}")
print(f"MSEG quartiles: {np.round(qm,2)}")
print(f"MSEG single quartile KS: D={Dlm:.2f}-{Dhm:.2f} p={plm:.2f}-{phm:.2f} top {rgm[3][0]:.2f}-{rgm[3][1]:.2f}")

# figures
def plot(data,m_seg,fname):
    data=data.copy()
    if m_seg is not None: data=data[data.mass<=m_seg]
    q=np.percentile(np.sort(data.mass.values),[0,25,50,75,100]); rg=[(q[k],q[k+1]) for k in range(4)]
    mr=data.dc.max(); rad=np.linspace(0,mr,400)
    CB=["#0077BB","#EE7733","#009988","#CC3311"]; LS=["-","--",":","-."]  # Tol CVD-safe + line styles
    fig,axs=plt.subplots(1,3,figsize=(18,6),squeeze=False,sharex=True); axs=axs.flatten()
    bn=data[data.bp>=0.6]; sg=data[data.bp<0.6]
    for idx,(ti,sd) in enumerate([("Single Stars",sg),("Binary Stars",bn)]):
        ax=axs[idx]
        for j,(a,b) in enumerate(rg):
            di=sd[(sd.mass>=a)&(sd.mass<b)].dc.values
            if len(di)==0: continue
            cu=np.array([np.sum(di<=r) for r in rad],float); cu=cu/cu.max() if cu.max()>0 else cu
            ax.plot(rad,cu,color=CB[j],ls=LS[j],lw=1.8,label=fr"$M$: {a:.2f}-{b:.2f}")
        ax.set_title(ti); ax.set_xlabel("Radius [arcmin]"); ax.legend()
        if idx==0: ax.set_ylabel("Normalized cumulative count")
    ax=axs[2]
    for j,(arr,lab) in enumerate([(sg.dc.values,"Single Stars"),(bn.dc.values,"Binary Stars")]):
        cu=np.array([np.sum(arr<=r) for r in rad],float); cu=cu/cu.max() if cu.max()>0 else cu
        ax.plot(rad,cu,color=CB[j],ls=LS[j],lw=1.8,label=lab)
    ax.set_title("Single vs Binary Stars"); ax.set_xlabel("Radius [arcmin]"); ax.legend()
    for k,ax in enumerate(axs):
        if k!=0: ax.set_yticks([])
    plt.subplots_adjust(wspace=0,hspace=0); plt.savefig(FIG+fname,bbox_inches="tight"); plt.close()
    print(f"  wrote {fname}")
plot(d,None,"cumulative_by_mass_and_type.pdf")
plot(d,cutoff,"cumulative_by_mass_and_type_mseg.pdf")

# Lambda_MSR figure
dra=(d.ra.values-c.ra.deg)*np.cos(np.deg2rad(c.dec.deg)); ddec=d.dec.values-c.dec.deg
pts=np.column_stack([dra,ddec]); order=np.argsort(d.mass.values)[::-1]
mst=lambda P:minimum_spanning_tree(squareform(pdist(P))).toarray().sum()
Ns=[5,8,10,15,20,30,40,50]; lam=[]; sig=[]
for n in Ns:
    lm=mst(pts[order[:n]]); rnd=np.array([mst(pts[rng.choice(N,n,replace=False)]) for _ in range(500)])
    lam.append(rnd.mean()/lm); sig.append(rnd.std()/lm)
fig,ax=plt.subplots(figsize=(6.2,4.4))
lam=np.array(lam); sig=np.array(sig)
top=float((lam+sig).max())*1.08; bot=max(0.0,float((lam-sig).min())-0.2)
ax.errorbar(Ns,lam,yerr=sig,fmt="o-",color="purple",capsize=3,lw=1.6,ms=6,zorder=3,
            label=r"$\Lambda_{\mathrm{MSR}}$ (NGC 6383)")
ax.axhline(1,ls="--",color="gray",lw=1.2,zorder=2,label=r"$\Lambda_{\mathrm{MSR}}=1$ (no segregation)")
ax.set_ylim(bot,top)
ax.set_xlabel(r"$N_{\mathrm{MST}}$ (number of most massive members)",fontsize=12)
ax.set_ylabel(r"$\Lambda_{\mathrm{MSR}}$",fontsize=13)
ax.legend(loc="upper right")
fig.tight_layout()
fig.savefig(FIG+"mass_segregation_mst.pdf"); plt.close()
print("  wrote mass_segregation_mst.pdf  Lambda(N=10)=%.2f"%lam[2])
tt=d[d.designation.astype(str).str.contains("4054567805963940352")]
print("T-Tauri binar_prob:",tt.bp.values)
