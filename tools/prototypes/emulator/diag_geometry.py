"""Is the control failure a sampler problem or an identifiability problem?"""
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import mist_grid
from emulator import load
from model import make_quadrature, simulate, star_log_likelihood

grid = mist_grid.build(); em = load(grid); quad = make_quadrature(em, n_mass=128, n_q=4)
T = {"log_age": 6.65, "feh": 0.00, "dm": 10.30, "Av": 1.20}
oracle = mist_grid.raw_isochrone(T["log_age"], T["feh"])
data, box = simulate(em, quad, T, 254, seed=0, oracle=oracle)
G, C, sG, sC = data

def ll(th):
    la, fe, dm, av = th
    return star_log_likelihood(em, quad, G, C, sG, sC, la, fe, dm, av,
                               0.3, 0.02, 0.02, box).sum()
llj = jax.jit(ll)
print(f"ll at TRUTH = {float(llj(jnp.array([6.65,0.0,10.30,1.20]))):.2f}")

# profile over log_age: optimise feh, dm, Av at each
def negprof(la):
    f = jax.jit(lambda p: -ll(jnp.array([la, p[0], p[1], p[2]])))
    g = jax.jit(jax.grad(lambda p: -ll(jnp.array([la, p[0], p[1], p[2]]))))
    best = None
    for x0 in ([0.0,10.30,1.20],[0.2,10.0,1.6],[-0.3,10.6,0.8]):
        r = minimize(lambda p: float(f(jnp.asarray(p))), x0,
                     jac=lambda p: np.asarray(g(jnp.asarray(p)), float), method="L-BFGS-B",
                     bounds=[(em.feh_lo,em.feh_hi),(9.5,11.0),(0.0,3.0)])
        if best is None or r.fun < best.fun: best = r
    return best

ages = np.arange(6.20, 7.11, 0.05)
prof = [negprof(float(a)) for a in ages]
v = np.array([p.fun for p in prof]); v -= v.min()
print(f"\n{'logage':>7} {'-dlnL':>9} {'feh':>7} {'dm':>8} {'Av':>7}")
for a,p,d in zip(ages, prof, v):
    star = " <<< PROFILE MAX" if d==0 else ""
    print(f"{a:7.2f} {d:9.2f} {p.x[0]:7.3f} {p.x[1]:8.3f} {p.x[2]:7.3f}{star}")
mins = int(np.sum((v[1:-1]<v[:-2])&(v[1:-1]<v[2:])))
print(f"\nlocal minima in the PROFILE: {mins}  | profile argmax age = {ages[np.argmin(v)]:.2f} (truth 6.65)")
