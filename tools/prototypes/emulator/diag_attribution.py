"""Is the +0.084 dex holdout bias EMULATOR error, or a sampling artefact?

`holdout age 2x` differs from `control` in two ways at once: the emulator must
interpolate across a 0.10 dex gap, AND its likelihood turned out to have very
different geometry (posterior sd 0.0038 vs 0.0840).  One converged run and one
unconverged run cannot separate those.

The profile likelihood can, and needs no sampler.  Profile BOTH emulators over
log age on the SAME injected data:
  * if the coarse profile peaks near 6.73, the sampler found that likelihood's
    true optimum and the displacement is genuine emulator error;
  * if it still peaks near 6.65, the +0.084 dex is a sampling artefact of a
    sharp well and must not be called emulator-induced.
Also reports the coarse emulator's held-out accuracy AT 6.65 itself, rather than
interpolating between the 6.45 and 6.85 entries of the accuracy table.
"""
import numpy as np, jax, jax.numpy as jnp
import mist_grid
from emulator import load
from model import make_quadrature, simulate, star_log_likelihood

T={"log_age":6.65,"feh":0.0,"dm":10.30,"Av":1.20}
oracle=mist_grid.raw_isochrone(T["log_age"],T["feh"])
full=mist_grid.build(); gen_em=load(full); gen_q=make_quadrature(gen_em,n_mass=128,n_q=4)
data,box=simulate(gen_em,gen_q,T,254,seed=0,oracle=oracle)     # identical to run_one
G,C,sG,sC=data

# --- held-out accuracy of the coarse emulator AT the injected node ----------
coarse=mist_grid.build(stride_age=2); em_c=load(coarse)
lm=np.asarray(full["logmass"]); m=10.0**lm
ok=(m>=oracle["mass"][0])&(m<=oracle["mass"][-1]); lm,m=lm[ok],m[ok]
Gt=np.interp(m,oracle["mass"],oracle["G"]); Ct=np.interp(m,oracle["mass"],oracle["BPRP"])
for tag,e in (("full grid",load(full)),("stride_age=2",em_c)):
    dG=np.asarray(e.absolute(6.65,0.0,jnp.asarray(lm))[0])-Gt
    dC=np.asarray(e.absolute(6.65,0.0,jnp.asarray(lm))[1])-Ct
    print(f"accuracy at the INJECTED node (log age 6.65, [Fe/H] 0.00), {tag:12s}: "
          f"RMS(G)={np.sqrt((dG**2).mean()):.4f} mean(dG)={dG.mean():+.4f} max={np.abs(dG).max():.4f} "
          f"RMS(col)={np.sqrt((dC**2).mean()):.4f}")

# --- profile both emulators over log age on the SAME data -------------------
ages=jnp.arange(6.55,6.851,0.01)
LO=jnp.array([-1.0,9.6,0.0]); HI=jnp.array([0.5,11.0,3.0])
STARTS=jnp.array([[0.0,10.30,1.20],[0.15,10.05,1.50],[-0.1,10.5,0.95]])
print(f"\n{'grid':>14} {'profile argmax':>15} {'bias vs truth':>14}")
for tag,gr in (("full (stride 1)",full),("coarse (stride 2)",coarse)):
    em=load(gr); q=make_quadrature(em,n_mass=128,n_q=4)
    def nll(p,a):
        return -star_log_likelihood(em,q,G,C,sG,sC,a,p[0],p[1],p[2],0.3,0.02,0.02,box).sum()
    g=jax.grad(nll)
    def descend(a,p0):
        def step(p,_):
            gg=g(p,a); gg=gg/(jnp.linalg.norm(gg)+1e-12)
            return jnp.clip(p-0.01*gg,LO,HI),None
        p,_=jax.lax.scan(step,p0,None,length=250)
        return nll(p,a)
    best=jax.jit(jax.vmap(lambda a: jnp.min(jax.vmap(lambda s: descend(a,s))(STARTS))))
    v=np.array(best(ages)); v=v-v.min()
    top=float(np.asarray(ages)[int(np.argmin(v))])
    print(f"{tag:>17} {top:15.3f} {top-6.65:+14.3f}")
    print("      curve:", " ".join(f"{float(a):.2f}:{x:.1f}" for a,x in zip(ages,v))[:300])
