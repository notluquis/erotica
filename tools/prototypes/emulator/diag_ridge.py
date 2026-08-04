"""Decisive test: is the slow mixing caused by the NUISANCE parameters or by the
age-distance-extinction RIDGE itself?

Run the identical data three ways:
  A  all 7 parameters free                       (the production model)
  B  f_bin, f_out, jitter FIXED at truth         (4 physical parameters only)
  C  4 physical, and dm fixed at truth           (the ridge cut in one direction)

If B converges, the nuisances are the problem.  If B fails and C converges, the
ridge is the problem and the fix is a reparametrisation, not more warmup.
"""
import numpy as np, jax, jax.numpy as jnp, arviz as az, numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
import mist_grid
from emulator import load
from model import make_quadrature, simulate, star_log_likelihood
numpyro.set_host_device_count(1)

grid=mist_grid.build(); em=load(grid); quad=make_quadrature(em,n_mass=128,n_q=4)
T={"log_age":6.65,"feh":0.0,"dm":10.30,"Av":1.20}
oracle=mist_grid.raw_isochrone(T["log_age"],T["feh"])
data,box=simulate(em,quad,T,254,seed=0,oracle=oracle)

def mk(mode):
    def model(G,C,sG,sC):
        la=numpyro.sample("log_age",dist.Uniform(em.age_lo,em.age_hi))
        fe=numpyro.sample("feh",dist.TruncatedNormal(0.,0.25,low=em.feh_lo,high=em.feh_hi))
        dm=T["dm"] if mode=="C" else numpyro.sample("dm",dist.Normal(10.30,0.20))
        av=numpyro.sample("Av",dist.Uniform(0.,3.))
        if mode=="A":
            fb=numpyro.sample("f_bin",dist.Beta(2.,3.)); fo=numpyro.sample("f_out",dist.Beta(1.5,20.))
            jt=numpyro.sample("jitter",dist.HalfNormal(0.05))
        else:
            fb,fo,jt=0.3,0.02,0.01
        numpyro.factor("ll",star_log_likelihood(em,quad,G,C,sG,sC,la,fe,dm,av,fb,fo,jt,box).sum())
    return model

for mode,desc in (("A","7 params (production)"),("B","4 physical, nuisances fixed"),
                  ("C","3 physical, dm also fixed")):
    grp=[p for p in ("log_age","feh","dm","Av") if not (mode=="C" and p=="dm")]
    k=NUTS(mk(mode),target_accept_prob=0.85,max_tree_depth=10,
           dense_mass=[tuple(grp)],init_strategy=init_to_median)
    mc=MCMC(k,num_warmup=800,num_samples=800,num_chains=2,chain_method="sequential",progress_bar=False)
    import time; t0=time.time()
    mc.run(jax.random.PRNGKey(0),*data,extra_fields=("diverging","num_steps"))
    dt=time.time()-t0
    idata=az.from_numpyro(mc); rh=az.rhat(idata); es=az.ess(idata)
    ns=float(np.mean(mc.get_extra_fields()["num_steps"]))
    print(f"\n[{mode}] {desc}  ({dt:.0f}s, mean tree steps {ns:.0f}, div {int(np.asarray(mc.get_extra_fields()['diverging']).sum())})")
    for p in grp:
        x=np.asarray(idata.posterior[p])
        print(f"   {p:8s} rhat {float(np.asarray(rh[p])):7.4f} ESSb {float(np.asarray(es[p])):7.0f} "
              f"mean {x.mean():9.4f} sd {x.std():.4f} truth {T[p]:.3f} chainmeans {x.mean(1)}")
