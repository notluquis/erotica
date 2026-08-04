"""Why is ESS 3 with zero divergences?  Look at the chains, not the summary."""
import numpy as np, jax, arviz as az, numpyro
from numpyro.infer import MCMC, NUTS, init_to_median
import mist_grid
from emulator import load
from model import make_model, make_quadrature, simulate
numpyro.set_host_device_count(1)
grid=mist_grid.build(); em=load(grid); quad=make_quadrature(em,n_mass=128,n_q=4)
T={"log_age":6.65,"feh":0.0,"dm":10.30,"Av":1.20}
oracle=mist_grid.raw_isochrone(T["log_age"],T["feh"])
data,box=simulate(em,quad,T,254,seed=0,oracle=oracle)
m=make_model(em,quad,box,dm_mu=10.30,dm_sigma=0.20)
mcmc=MCMC(NUTS(m,target_accept_prob=0.85,max_tree_depth=12,
               dense_mass=[("log_age","feh","dm","Av")],init_strategy=init_to_median),
          num_warmup=800,num_samples=600,num_chains=2,chain_method="sequential",progress_bar=False)
mcmc.run(jax.random.PRNGKey(0),*data,extra_fields=("diverging","num_steps","adapt_state.step_size"))
ex=mcmc.get_extra_fields()
print("mean tree steps/iter:", float(np.mean(ex["num_steps"])), " max:", int(np.max(ex["num_steps"])))
print("final step_size:", np.asarray(ex["adapt_state.step_size"]).reshape(2,-1)[:,-1])
idata=az.from_numpyro(mcmc)
for p in ("log_age","feh","dm","Av","f_bin","f_out","jitter"):
    x=np.asarray(idata.posterior[p])
    print(f"{p:8s} chain means {x.mean(1)}  chain sds {x.std(1)}  rhat {float(np.asarray(az.rhat(idata)[p])):.3f} ess {float(np.asarray(az.ess(idata)[p])):.0f}")
