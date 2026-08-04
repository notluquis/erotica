"""N floor without MCMC: how much information about log age does N stars carry?

The profile likelihood half-width is an *information* statement and is
independent of whether any sampler converges -- which matters here, because the
MCMC ladder did not complete at N = 61 (trajectories saturate max_tree_depth on
the wider, longer ridge that less data produces).

For each N: scan log age on a fine grid with (feh, dm, Av) re-optimised at every
step by gradient descent run inside JAX (not scipy -- the scipy/L-BFGS version of
this script was Python-bound and did not finish), then report where the profile
peaks and its Delta lnL = 0.5 half-width.
"""
import numpy as np, jax, jax.numpy as jnp
import mist_grid
from emulator import load
from model import make_quadrature, simulate, star_log_likelihood

grid=mist_grid.build(); em=load(grid); quad=make_quadrature(em,n_mass=128,n_q=4)
T={"log_age":6.65,"feh":0.0,"dm":10.30,"Av":1.20}
oracle=mist_grid.raw_isochrone(T["log_age"],T["feh"])
ages=jnp.arange(6.40,6.921,0.02)   # 27 nodes -- coarse on purpose, this is a width not a peak
LO=jnp.array([em.feh_lo,9.6,0.0]); HI=jnp.array([em.feh_hi,11.0,3.0])
STARTS=jnp.array([[0.0,10.30,1.20],[0.15,10.05,1.50]])

def profile_for(data,box):
    G,C,sG,sC=data
    def nll(p,a):
        return -star_log_likelihood(em,quad,G,C,sG,sC,a,p[0],p[1],p[2],0.3,0.02,0.02,box).sum()
    g=jax.grad(nll)
    def descend(a,p0):                      # projected gradient descent, fixed budget
        def step(p,_):
            gr=g(p,a); gr=gr/(jnp.linalg.norm(gr)+1e-12)
            p=jnp.clip(p-0.01*gr,LO,HI)
            return p,None
        p,_=jax.lax.scan(step,p0,None,length=120)
        return nll(p,a)
    best=jax.jit(jax.vmap(lambda a: jnp.min(jax.vmap(lambda s: descend(a,s))(STARTS))))
    return np.asarray(best(ages))

print(f"{'N':>5} {'argmax':>8} {'bias':>8} {'half-width':>11} {'local min':>10}  per-seed argmax")
for n in (30,61,128,254):
    tops=[];hws=[];nms=[]
    for seed in (0,1):
        data,box=simulate(em,quad,T,n,seed=seed,oracle=oracle)
        v=profile_for(data,box); v=v-v.min()
        top=float(np.asarray(ages)[int(np.argmin(v))]); tops.append(top)
        ins=np.asarray(ages)[v<0.5]
        hws.append((ins.max()-ins.min())/2 if len(ins)>1 else 0.0025)
        nms.append(int(np.sum((v[1:-1]<v[:-2])&(v[1:-1]<v[2:]))))
    print(f"{n:5d} {tops[0]:8.3f} {tops[0]-T['log_age']:+8.3f} {np.median(hws):11.4f} {int(np.median(nms)):10d}  {np.round(tops,3)}")
