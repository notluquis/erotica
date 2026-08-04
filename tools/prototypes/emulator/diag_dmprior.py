"""Is the ridge tamed by the dm prior Gaia actually supplies?

sigma_dm = 0.20 mag (used in the first pass) is LOOSER than real data give.
NGC 6383's own parallax, 1.11 +/- 0.06 kpc, is dm = 10.30 +/- 0.09 mag
(aanda.tex Table 1).  Using 0.20 makes the age-distance ridge artificially long.
This is not prior-tightening to buy convergence: it is using the width the
measurement has.
"""
import time, numpy as np, jax, arviz as az, numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
import mist_grid
from emulator import load
from model import make_model, make_quadrature, simulate
numpyro.set_host_device_count(1)
grid=mist_grid.build(); em=load(grid); quad=make_quadrature(em,n_mass=128,n_q=4)
T={"log_age":6.65,"feh":0.0,"dm":10.30,"Av":1.20}
oracle=mist_grid.raw_isochrone(T["log_age"],T["feh"])
data,box=simulate(em,quad,T,254,seed=0,oracle=oracle)
for s in (0.09, 0.03):
    m=make_model(em,quad,box,dm_mu=10.30,dm_sigma=s)
    mc=MCMC(NUTS(m,target_accept_prob=0.85,max_tree_depth=10,
                 dense_mass=[("log_age","feh","dm","Av")],init_strategy=init_to_median),
            num_warmup=1000,num_samples=1000,num_chains=2,chain_method="sequential",progress_bar=False)
    t0=time.time(); mc.run(jax.random.PRNGKey(0),*data,extra_fields=("diverging","num_steps")); dt=time.time()-t0
    idata=az.from_numpyro(mc); rh=az.rhat(idata); es=az.ess(idata)
    nd=int(np.asarray(mc.get_extra_fields()["diverging"]).sum())
    wr=max(float(np.asarray(rh[p])) for p in ("log_age","feh","dm","Av"))
    we=min(float(np.asarray(es[p])) for p in ("log_age","feh","dm","Av"))
    print(f"\n=== dm_sigma={s} ({dt:.0f}s, div {nd}, mean tree steps {float(np.mean(mc.get_extra_fields()['num_steps'])):.0f}) "
          f"GATE {'PASS' if wr<1.01 and we>400 and nd==0 else 'FAIL'} (rhat {wr:.4f}, ESS {we:.0f})")
    for p in ("log_age","feh","dm","Av"):
        x=np.asarray(idata.posterior[p]).ravel()
        print(f"   {p:8s} truth {T[p]:8.3f} mean {x.mean():9.4f} sd {x.std(ddof=1):.4f} "
              f"bias {x.mean()-T[p]:+.4f} b/sd {(x.mean()-T[p])/x.std(ddof=1):+6.2f} "
              f"rhat {float(np.asarray(rh[p])):.4f} ESSb {float(np.asarray(es[p])):.0f}")
