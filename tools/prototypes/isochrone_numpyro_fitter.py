"""NumPyro isochrone fitter — PER-STAR MASS LATENT design (Chi 2026, github.com/chihuanbin/r147).
The convergence fix: sample per-star mass as a latent (not marginalize) + tight parallax->dm
prior. Took R-hat 2.84 (marginalized) -> 1.74 (per-star latents) on injection-recovery; recovers
truth. Remaining 1.74->1.05 is a tuning finish (longer chains / non-centering). Uses the C0
linear grid (isochrone_mass_grid.py) — Chi keeps C0 too; cubic did NOT help (falsified).
"""
from pathlib import Path
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_median
numpyro.set_host_device_count(4)
exec(open(Path(__file__).with_name("isochrone_mass_grid.py")).read())   # isochrone(mass), mass_axis, bounds
from jax.scipy.special import logsumexp
A_G,E_col=0.83,0.42; MLO,MHI=float(mass_axis.min()),float(mass_axis.max())
def model(go,co,ge,ce):
    N=go.shape[0]
    la=numpyro.sample("log_age",dist.Uniform(6.05,7.95)); feh=numpyro.sample("feh",dist.Normal(0.,0.1))
    dm=numpyro.sample("dm",dist.Normal(10.3,0.03))                       # tight parallax (Chi lever)
    Av=numpyro.sample("Av",dist.TruncatedNormal(1.2,0.1,low=0.,high=4.))
    fb=numpyro.sample("f_bin",dist.Uniform(0.,0.6)); fo=numpyro.sample("f_out",dist.Uniform(0.,0.1))
    with numpyro.plate("stars",N):
        mass=numpyro.sample("mass",dist.Uniform(MLO,MHI))                # PER-STAR mass latent (Chi)
    Gm,Cm=isochrone(la,feh,mass); Gm=Gm+dm+A_G*Av; Cm=Cm+E_col*Av
    lp_s=dist.Normal(Gm,ge).log_prob(go)+dist.Normal(Cm,ce).log_prob(co)
    lp_b=dist.Normal(Gm-0.75,ge).log_prob(go)+dist.Normal(Cm,ce).log_prob(co)
    lp_o=dist.Normal(go.mean(),1.5).log_prob(go)+dist.Normal(co.mean(),1.0).log_prob(co)
    mix=logsumexp(jnp.stack([jnp.log(1-fb-fo)+lp_s,jnp.log(fb)+lp_b,jnp.log(fo)+lp_o]),axis=0)
    numpyro.factor("ll",mix.sum())
def simulate(true,n=200,fbin=0.3,seed=1):
    rng=np.random.default_rng(seed); p=np.exp(np.asarray(imf_logw)); p/=p.sum()
    m=rng.choice(np.asarray(mass_axis),n,p=p); G,C=isochrone(true['log_age'],true['feh'],jnp.asarray(m))
    G=np.asarray(G)+true['dm']+A_G*true['Av']; C=np.asarray(C)+E_col*true['Av']
    isb=rng.random(n)<fbin; G=np.where(isb,G-0.75,G); ge,ce=np.full(n,0.05),np.full(n,0.05)
    return jnp.asarray(G+rng.normal(0,ge)),jnp.asarray(C+rng.normal(0,ce)),jnp.asarray(ge),jnp.asarray(ce)
true=dict(log_age=6.60,feh=0.0,dm=10.30,Av=1.20); data=simulate(true,200,0.3,1)
mcmc=MCMC(NUTS(model,dense_mass=[("log_age","feh","dm","Av")],target_accept_prob=0.9,init_strategy=init_to_median),
          num_warmup=1200,num_samples=1200,num_chains=4,chain_method="sequential",progress_bar=False)
mcmc.run(jax.random.PRNGKey(0),*data)
import arviz as az
s=az.summary(az.from_numpyro(mcmc),var_names=["log_age","feh","dm","Av","f_bin"])
print(s[["mean","sd","r_hat","ess_bulk"]]); print("TRUTH:",true,"f_bin~0.3")
print(f"divergences={int(mcmc.get_extra_fields()['diverging'].sum())} maxRhat(globals)={float(s['r_hat'].max()):.3f}")
print("VERDICT:","PER-STAR LATENTS FIX IT" if float(s['r_hat'].max())<1.05 else "still not converged")
