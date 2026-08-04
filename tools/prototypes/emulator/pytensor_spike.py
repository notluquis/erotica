"""Part 2 spike: can the emulator live in PyTensor, so PyMC keeps its speedup?

The design note assumed the emulator must be JAX-native and therefore that the
isochrone module must step outside PyMC.  That would cost the shared
``SamplingConfig``, the ArviZ provenance record, and -- if a ``pt.Op`` wrapping a
JAX function breaks the whole-graph JAX compile -- the measured 2.71x speedup
from ``nuts_sampler="numpyro"`` that makes the whole thing viable.

This spike tests the third option nobody costed: **write the interpolation in
PyTensor itself**.  Multilinear interpolation on a regular grid is a fixed
number of gathers and multiply-adds; PyTensor has `set_subtensor`-free advanced
indexing and can express it directly.  If that compiles under
``nuts_sampler="numpyro"``, the PyMC-vs-JAX dilemma dissolves.

Three things are checked, in order of how much they would cost to be wrong:

1. ``pt.grad`` returns a finite, nonzero gradient (the published ``pt.Op``
   raised ``NotImplementedError: pullback not implemented``).
2. ``pm.sample(nuts_sampler="numpyro")`` compiles the graph and runs.
3. The posterior agrees with the same model sampled by the default PyTensor
   NUTS, and the wall-clock ratio is recorded.
"""

from __future__ import annotations

import time

import numpy as np
import pymc as pm
import pytensor.tensor as pt

import mist_grid


def pt_interp3(grid, xa, xf, xm):
    """Trilinear interpolation of a constant 3-D grid at continuous indices.

    ``grid`` is a NumPy array baked into the graph as a constant; ``xa/xf/xm``
    are PyTensor scalars/vectors.  Eight gathers, seven multiply-adds -- a
    closed-form expression with no adaptive work, differentiable by PyTensor's
    own autodiff because the only parameter-dependent quantities are the
    weights.
    """
    na, nf, nm = grid.shape
    Gc = pt.as_tensor_variable(grid)

    def _tap(x, n):
        xc = pt.clip(x, 0.0, n - 1.0)
        i0 = pt.clip(pt.floor(xc), 0, n - 2).astype("int64")
        return i0, xc - i0

    ia, ta = _tap(xa, na)
    jf, tf = _tap(xf, nf)
    km, tm = _tap(xm, nm)
    out = 0.0
    for p, wa in ((0, 1 - ta), (1, ta)):
        for q, wf in ((0, 1 - tf), (1, tf)):
            for r, wm in ((0, 1 - tm), (1, tm)):
                out = out + wa * wf * wm * Gc[
                    pt.clip(ia + p, 0, na - 1),
                    pt.clip(jf + q, 0, nf - 1),
                    pt.clip(km + r, 0, nm - 1),
                ]
    return out


def build_model(grid, data, imf_w, dm_mu=10.30, dm_sigma=0.20):
    """A stripped single/outlier CMD model -- enough to prove the mechanism."""
    G_obs, C_obs, sG, sC = data
    a0, da = grid["logage"][0], grid["logage"][1] - grid["logage"][0]
    f0, df = grid["feh"][0], grid["feh"][1] - grid["feh"][0]
    m0, dmg = grid["logmass"][0], grid["logmass"][1] - grid["logmass"][0]
    lm = grid["logmass"]

    with pm.Model() as model:
        logage = pm.Uniform("log_age", grid["logage"][0], grid["logage"][-1])
        feh = pm.Uniform("feh", grid["feh"][0], grid["feh"][-1])
        dm = pm.Normal("dm", dm_mu, dm_sigma)
        Av = pm.Uniform("Av", 0.0, 3.0)

        xa = (logage - a0) / da
        xf = (feh - f0) / df
        xm = pt.as_tensor_variable((lm - m0) / dmg)          # (K,) constant

        Gm = pt_interp3(grid["G"], xa, xf, xm) + dm + 0.83 * Av
        Cm = pt_interp3(grid["BPRP"], xa, xf, xm) + 0.42 * Av

        vG = (sG**2)[:, None]
        vC = (sC**2)[:, None]
        lp = (-0.5 * ((G_obs[:, None] - Gm[None, :]) ** 2 / vG + pt.log(2 * np.pi * vG))
              - 0.5 * ((C_obs[:, None] - Cm[None, :]) ** 2 / vC + pt.log(2 * np.pi * vC))
              + pt.as_tensor_variable(np.log(imf_w))[None, :])
        pm.Potential("obs", pm.math.logsumexp(lp, axis=1).sum())
    return model


def main() -> None:
    import jax.numpy as jnp

    import emulator as emu
    from model import make_quadrature, simulate

    grid = mist_grid.build()
    em = emu.load(grid)
    quad = make_quadrature(em, n_mass=96, n_q=3)
    truth = {"log_age": 6.60, "feh": 0.0, "dm": 10.30, "Av": 1.20}
    (G, C, sG, sC), _ = simulate(em, quad, truth, 120, seed=3, f_bin=0.0, f_out=0.0)
    data = tuple(np.asarray(x) for x in (G, C, sG, sC))

    # IMF weights on the emulator's own log-mass axis
    lw = np.asarray(emu.kroupa_log_pdf_logmass(jnp.asarray(grid["logmass"])))
    w = np.exp(lw - lw.max())
    w /= w.sum()

    print("=" * 74)
    print("PART 2 SPIKE -- PyTensor-native trilinear interpolation inside PyMC")
    print("=" * 74)

    model = build_model(grid, data, w)

    # --- (1) does pt.grad work? -------------------------------------------
    # The published ASteCA model raised
    #   NotImplementedError: pullback not implemented for SyntheticCluster
    # here, because its `pt.Op` defined only `perform()`.  That is the whole
    # reason a gradient-free sampler was forced.
    try:
        dlogp = model.compile_dlogp()
        pt0 = model.initial_point()
        gv = np.asarray(dlogp(pt0))
        names = [v.name for v in model.value_vars]
        print("\n(1) pt.grad via model.compile_dlogp():")
        for n, x in zip(names, gv):
            print(f"    d logp / d {n:12s} = {float(x):+.6g}")
        ok1 = bool(np.all(np.isfinite(gv)) and np.all(gv != 0.0))
        print(f"    -> {'GRADIENT AVAILABLE and nonzero' if ok1 else 'PROBLEM'}")
    except Exception as e:  # noqa: BLE001
        print(f"\n(1) pt.grad FAILED: {type(e).__name__}: {e}")
        return

    # --- (2)+(3) sample both backends -------------------------------------
    res = {}
    for backend in ("numpyro", "pymc"):
        t0 = time.time()
        try:
            with model:
                idata = pm.sample(draws=500, tune=500, chains=2, cores=1,
                                  target_accept=0.9, random_seed=1,
                                  nuts_sampler=backend, progressbar=False)
            dt = time.time() - t0
            import arviz as az
            rh = az.rhat(idata)
            summ = {v: (float(idata.posterior[v].mean()),
                        float(idata.posterior[v].std()),
                        float(rh[v]))
                    for v in ("log_age", "feh", "dm", "Av")}
            res[backend] = (dt, summ)
            print(f"\n(2) nuts_sampler={backend!r}: COMPILED AND RAN in {dt:.1f}s")
            for v, (m, s, r) in summ.items():
                print(f"    {v:8s} {m:9.4f} +/- {s:.4f}   rhat {r:.4f}   "
                      f"truth {truth[v if v != 'log_age' else 'log_age']:.3f}")
        except Exception as e:  # noqa: BLE001
            print(f"\n(2) nuts_sampler={backend!r} FAILED: {type(e).__name__}: {e}")

    if len(res) == 2:
        print(f"\n(3) wall clock: numpyro {res['numpyro'][0]:.1f}s vs "
              f"pymc {res['pymc'][0]:.1f}s  -> {res['pymc'][0] / res['numpyro'][0]:.2f}x")
        d = max(abs(res["numpyro"][1][v][0] - res["pymc"][1][v][0])
                for v in res["numpyro"][1])
        print(f"    max |posterior mean difference| between backends: {d:.5f}")


if __name__ == "__main__":
    main()
