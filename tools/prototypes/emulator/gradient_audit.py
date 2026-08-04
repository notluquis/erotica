"""THE headline measurement: what fraction of the parameter space has an exactly
zero gradient, for the current binned-Hess path and for the emulator.

Everything here is measured by this script.  The 46.7% figure quoted in the
dossiers is **recomputed from the archived grid**, not cited, so that the
comparison is apples to apples: same definition, same tolerance, same code.

Three things are measured, because the first alone would not settle it:

1. **Zero-gradient fraction of the forward model** -- the Hess-grid slice test
   the design note used (adjacent slices bitwise identical), reproduced.
2. **Zero-gradient fraction of the log posterior**, per parameter, over draws
   from the prior.  This is the number that actually governs whether NUTS can
   move, and it has never been measured for either path.
3. **Local minima per dex along log age.**  A nonzero gradient that reverses
   sign 50 times per dex is not a usable gradient -- the dossier measured 45-55
   local minima per dex for every likelihood built on asteca's ``generate()``,
   and traced it to the generator rather than the likelihood.  If the emulator
   inherited that, it would have fixed nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import mist_grid
from emulator import load
from model import make_quadrature, physical_logp_fn, simulate

HGRID = Path("/Users/notluquis/erotica/data/test/NGC6383/data/40/hgrid_paper254.npz")
ZTOL = 0.0          # "exactly zero" means exactly zero.  No epsilon fudge.
OUT = Path(__file__).parent / "_results"


# --------------------------------------------------------------------------
# 1. The incumbent: the archived binned-Hess grid.
# --------------------------------------------------------------------------
def audit_hess_grid() -> dict:
    z = np.load(HGRID)
    H = z["H_grid"]                       # (M_met, M_loga, Nb_mag, Nb_col)
    out: dict[str, object] = {"shape": list(H.shape)}

    # (a) adjacent-slice identity -- the design note's own probe
    for axis, name in ((0, "met"), (1, "loga")):
        Hm = np.moveaxis(H, axis, 0)
        same = np.array([np.array_equal(Hm[i], Hm[i + 1]) for i in range(Hm.shape[0] - 1)])
        uniq = len({Hm[i].tobytes() for i in range(Hm.shape[0])})
        out[f"{name}_identical_adjacent"] = f"{same.sum()}/{len(same)}"
        out[f"{name}_zero_grad_frac"] = float(same.mean())
        out[f"{name}_unique_slices"] = f"{uniq}/{Hm.shape[0]}"

    # (b) gradient of a Poisson log-likelihood built on that grid, by finite
    #     difference in index space -- exactly what a bilinear interpolator's
    #     analytic derivative reduces to.
    rng = np.random.default_rng(0)
    obs = H[100, 100]                     # a mid-grid slice used as pseudo-data
    obs = rng.poisson(np.clip(obs, 0, None) * 254 / max(obs.sum(), 1e-9))

    def poisson_ll(i, j):
        lam = np.clip(H[i, j] * 254 / max(H[i, j].sum(), 1e-9), 1e-12, None)
        return float((obs * np.log(lam) - lam).sum())

    for axis, name in ((0, "met"), (1, "loga")):
        n = H.shape[axis]
        g = []
        for _ in range(400):
            i = rng.integers(0, H.shape[0] - 1)
            j = rng.integers(0, H.shape[1] - 1)
            a, b = ((i, j), (i + 1, j)) if axis == 0 else ((i, j), (i, j + 1))
            g.append(poisson_ll(*b) - poisson_ll(*a))
        g = np.array(g)
        out[f"{name}_logp_zero_grad_frac"] = float(np.mean(g == 0.0))
    return out


# --------------------------------------------------------------------------
# 2 + 3. The emulator.
# --------------------------------------------------------------------------
def audit_emulator(order_age: int, order_feh: int, n_draw: int = 2000,
                   n_star: int = 254, seed: int = 0,
                   n_mass: int = 96, n_q: int = 3, chunk: int = 50) -> dict:
    grid = mist_grid.build()
    em = load(grid, order_age=order_age, order_feh=order_feh)
    quad = make_quadrature(em, n_mass=n_mass, n_q=n_q)
    truth = {"log_age": 6.60, "feh": 0.0, "dm": 10.30, "Av": 1.20}
    data, box = simulate(em, quad, truth, n_star, seed=seed)
    logp = physical_logp_fn(em, quad, data, box, dm_mu=10.30, dm_sigma=0.20)

    # Chunked, not one big vmap: the unrolled 4x4x2-tap cubic makes a large
    # jaxpr, and vmapping 2000 draws over an (N, K, J) mixture materialises
    # ~5e8 elements.  Chunking keeps the trace small and the memory flat.
    gfn = jax.jit(jax.vmap(jax.grad(logp)))
    rng = np.random.default_rng(seed)
    th = np.column_stack([
        rng.uniform(em.age_lo, em.age_hi, n_draw),
        rng.uniform(em.feh_lo, em.feh_hi, n_draw),
        rng.normal(10.30, 0.20, n_draw),
        rng.uniform(0.0, 3.0, n_draw),
    ])
    g = np.concatenate([np.asarray(gfn(jnp.asarray(th[i:i + chunk])))
                        for i in range(0, n_draw, chunk)])
    finite = np.isfinite(g).all(1)

    names = ["log_age", "feh", "dm", "Av"]
    out = {
        "order_age": order_age, "order_feh": order_feh,
        "n_draw": n_draw, "n_star": n_star,
        "nonfinite_frac": float(1 - finite.mean()),
    }
    for k, nm in enumerate(names):
        out[f"{nm}_zero_grad_frac"] = float(np.mean(g[finite, k] == 0.0))
        out[f"{nm}_grad_median_abs"] = float(np.median(np.abs(g[finite, k])))
    out["any_zero_frac"] = float(np.mean((g[finite] == 0.0).any(1)))

    # local minima along log age, on the 0.005 dex grid the dossier used
    ages = np.arange(em.age_lo + 1e-6, em.age_hi - 1e-6, 0.005)
    scan = jax.jit(jax.vmap(lambda a: logp(jnp.array([a, 0.0, 10.30, 1.20]))))
    f = np.concatenate([np.asarray(scan(jnp.asarray(ages[i:i + chunk])))
                        for i in range(0, len(ages), chunk)])
    mins = int(np.sum((f[1:-1] < f[:-2]) & (f[1:-1] < f[2:])))
    out["local_minima_per_dex"] = mins / (em.age_hi - em.age_lo)
    out["argmax_log_age"] = float(ages[int(np.argmax(f))])
    out["truth_log_age"] = truth["log_age"]

    # Gradient continuity across an age node.  A C0 interpolant has a gradient
    # that is piecewise constant, so it *jumps* at every 0.05 dex node; a C1 one
    # does not.  NUTS integrates this field, so a discontinuous force is a real
    # (if second-order) defect, and the ratio jump/scale says how real.
    g1 = jax.jit(jax.grad(logp))
    h = 1e-4
    jumps, scales = [], []
    for k in (5, 10, 15, 20, 25):
        node = float(em.logage[k])
        gl = float(g1(jnp.array([node - h, 0.0, 10.30, 1.20]))[0])
        gr = float(g1(jnp.array([node + h, 0.0, 10.30, 1.20]))[0])
        jumps.append(abs(gr - gl))
        scales.append(0.5 * (abs(gr) + abs(gl)))
    out["grad_jump_at_node_median"] = float(np.median(jumps))
    out["grad_scale_median"] = float(np.median(scales))
    out["grad_jump_rel"] = float(np.median(np.array(jumps) / np.array(scales)))
    return out


def main() -> None:
    OUT.mkdir(exist_ok=True)
    res: dict[str, object] = {}

    print("=" * 74)
    print("1. INCUMBENT -- archived binned Hess grid (data/40/hgrid_paper254.npz)")
    print("=" * 74)
    h = audit_hess_grid()
    res["hess"] = h
    for k, v in h.items():
        print(f"  {k:34s} {v}")

    print()
    print("=" * 74)
    print("2. EMULATOR -- unbinned per-star likelihood over the MIST tensor")
    print("=" * 74)
    for oa, of in ((1, 1), (3, 3)):
        e = audit_emulator(oa, of)
        res[f"emulator_order{oa}"] = e
        tag = "C0 trilinear" if oa == 1 else "C1 tricubic"
        print(f"\n  -- {tag} (order_age={oa}, order_feh={of}) --")
        for k, v in e.items():
            print(f"  {k:34s} {v}")

    (OUT / "gradient_audit.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT / 'gradient_audit.json'}")


if __name__ == "__main__":
    main()
