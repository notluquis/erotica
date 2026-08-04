"""K=64 vs K=128: same likelihood shape along log age at the truth conditioning?"""
import numpy as np, jax, jax.numpy as jnp
import mist_grid
from emulator import load
from model import make_quadrature, simulate, star_log_likelihood
grid=mist_grid.build(); em=load(grid)
T={"log_age":6.65,"feh":0.0,"dm":10.30,"Av":1.20}
oracle=mist_grid.raw_isochrone(T["log_age"],T["feh"])
ages=np.arange(6.45,6.86,0.01)
curves={}
for K,J in ((192,5),(128,4),(64,3),(48,3)):
    q=make_quadrature(em,n_mass=K,n_q=J)
    data,box=simulate(em,q,T,254,seed=0,oracle=oracle); G,C,sG,sC=data
    f=jax.jit(jax.vmap(lambda a: star_log_likelihood(em,q,G,C,sG,sC,a,0.0,10.30,1.20,0.3,0.02,0.02,box).sum()))
    v=np.asarray(f(jnp.asarray(ages))); curves[K]=v-v.max()
    print(f"K={K:3d} J={J}: argmax={ages[np.argmax(v)]:.3f}  ll(truth)={v[np.argmin(abs(ages-6.65))]:.2f}  "
          f"halfwidth(dlnL=0.5)={np.ptp(ages[v>v.max()-0.5])/2:.4f} dex")
ref=curves[192]
for K in (128,64,48):
    print(f"  max |dlnL(K={K}) - dlnL(K=192)| over the window = {np.max(np.abs(curves[K]-ref)):.4f}")
