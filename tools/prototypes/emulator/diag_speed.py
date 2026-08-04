import time, numpy as np, jax, jax.numpy as jnp
import mist_grid
from emulator import load
from model import make_quadrature, simulate, star_log_likelihood
grid=mist_grid.build()
T={"log_age":6.65,"feh":0.0,"dm":10.30,"Av":1.20}
oracle=mist_grid.raw_isochrone(T["log_age"],T["feh"])
for order in ((3,3),(1,1)):
  for K,J in ((128,4),(64,3),(48,3)):
    em=load(grid,order_age=order[0],order_feh=order[1]); q=make_quadrature(em,n_mass=K,n_q=J)
    data,box=simulate(em,q,T,254,seed=0,oracle=oracle)
    G,C,sG,sC=data
    f=jax.jit(jax.grad(lambda th: star_log_likelihood(em,q,G,C,sG,sC,th[0],th[1],th[2],th[3],0.3,0.02,0.02,box).sum()))
    x=jnp.array([6.65,0.0,10.3,1.2]); f(x)[0].block_until_ready()
    t0=time.time()
    for _ in range(200): r=f(x)
    r.block_until_ready(); dt=(time.time()-t0)/200
    print(f"order={order} K={K} J={J}: {1e3*dt:7.3f} ms/grad")
