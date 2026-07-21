import sys, numpy as np, jax, jax.numpy as jnp
sys.path.insert(0,"/Users/notluquis/COSMIC")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("/Users/notluquis/COSMIC/iso_grid2.py").read())
A_G,E_col=0.83,0.42
true=dict(log_age=6.60,feh=0.0,dm=10.30,Av=1.20)
# synthetic cluster (with binaries + scatter)
rng=np.random.default_rng(1); p=np.exp(np.asarray(imf_logw)); p/=p.sum()
m=rng.choice(np.asarray(mass_axis),200,p=p); G,C=isochrone(true['log_age'],true['feh'],jnp.asarray(m))
G=np.asarray(G)+true['dm']+A_G*true['Av']; C=np.asarray(C)+E_col*true['Av']
isb=rng.random(200)<0.3; G=np.where(isb,G-0.75,G)
Gobs=G+rng.normal(0,0.05,200); Cobs=C+rng.normal(0,0.05,200)
# isochrones at several ages (apparent frame, truth dm/Av)
mm=np.asarray(mass_axis)
fig,ax=plt.subplots(figsize=(6,7))
ax.scatter(Cobs,Gobs,s=14,c='k',alpha=0.6,label='synthetic cluster (200 stars, 30% binaries)')
for la,col,lw in [(6.6,'C1',2.5),(7.0,'C0',1.3),(7.5,'C2',1.3)]:
    Gi,Ci=isochrone(la,0.0,jnp.asarray(mm)); Gi=np.asarray(Gi)+true['dm']+A_G*true['Av']; Ci=np.asarray(Ci)+E_col*true['Av']
    ax.plot(Ci,Gi,col,lw=lw,label=f"MIST logAge={la}"+(" (TRUTH)" if la==6.6 else ""))
ax.set_xlabel("BP - RP (mag)"); ax.set_ylabel("G (apparent)"); ax.invert_yaxis()
ax.set_title("NGC 6383-like synthetic cluster + MIST isochrones\n(mass-grid fitter, dm=10.3 Av=1.2)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
out="/private/tmp/claude-501/-Users-notluquis-phd/9a0f9b71-83f6-46ec-ba5e-2fc967bc6851/scratchpad/cmd_fit.png"
plt.tight_layout(); plt.savefig(out,dpi=110); print("saved",out)
print(f"CMD: G {Gobs.min():.1f}..{Gobs.max():.1f}, BP-RP {Cobs.min():.2f}..{Cobs.max():.2f}, {isb.sum()} binaries")
