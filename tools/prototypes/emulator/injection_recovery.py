"""Does emulator error bias the *posterior*?  And how few stars can it take?

This is the check the brief calls the theoretical core, and it is the one the
earlier prototype (``../isochrone_numpyro_fitter.py``) could not perform: its
``simulate()`` drew stars from the same ``isochrone()`` the model fitted, so
emulator error cancelled **exactly** and the test was structurally blind to it --
the same defect as the half-ensemble bootstrap the P01 dossiers convicted.

The fix is to inject from a **raw MIST isochrone that the emulator's grid does
not contain**.  Nothing in the truth path touches the emulator.

The controlled comparison
-------------------------
Every scenario injects the **same kind of truth** -- a raw MIST isochrone -- and
varies only the emulator that fits it.  ``control`` fits with the full grid, on
which the injected node is reproduced to ~1e-15 mag, so its bias measures the
sampler and the identifiability of the model and nothing else.  ``holdout``
fits the *same data* with a grid coarsened by a stride of 2, so the injected
node is genuinely absent.  The **difference** between the two is
emulator-induced posterior bias, isolated.

``control``       full grid; injected age is a grid node.  Emulator error ~ 0.
``holdout age``   ``stride_age=2`` (0.10 dex); injected age held out.
``model error``   0.03 mag of extra scatter the fit is not told about, to show
                  how much of the control's precision is an artefact of a
                  simulation in which the model happens to be exactly right.
``N=...``         the N floor: the census median is ~61 members, and the
                  comparable published fits are all of rich clusters.

Gate: Vehtari et al. (2021) -- rhat < 1.01, ESS_bulk > 400, zero divergences,
read from ``az.rhat``/``az.ess``.  **Never** ``az.summary``, which applies
``round_to="auto"`` and cannot decide 1.00996 against a < 1.01 threshold.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arviz as az
import jax
import numpy as np
import numpyro
from numpyro.infer import MCMC, NUTS, init_to_median

import mist_grid
from emulator import load
from model import make_model, make_quadrature, simulate

OUT = Path(__file__).parent / "_results"
PARAMS = ("log_age", "feh", "dm", "Av")

# Memory constraint from the brief: 4 chains has exhausted RAM on this machine.
CHAINS, CORES = 2, 1
numpyro.set_host_device_count(CORES)


def run_one(name, *, truth, n_star, stride_age=1, stride_feh=1,
            dm_sigma=0.20, order=(3, 3), warmup=1500, draws=1000, seed=0,
            n_mass=128, n_q=4, mag_limit=None, model_error=0.0):
    # n_mass=128 is not a convenience default.  With continuous simulated
    # masses, the quadrature resolution shifts the likelihood argmax:
    # K=192 -> 6.650, K=128 -> 6.650, K=64 -> 6.660, K=48 -> 6.630 against a
    # truth of 6.650.  The posterior sd is ~0.02 dex, so K=64's +0.01 dex is a
    # 0.5 sigma bias bought for a 3.5x speedup.  Set K by convergence, not by
    # wall clock.
    grid = mist_grid.build(stride_age=stride_age, stride_feh=stride_feh)
    em = load(grid, order_age=order[0], order_feh=order[1])
    quad = make_quadrature(em, n_mass=n_mass, n_q=n_q)

    # Truth is ALWAYS a raw MIST isochrone off disk -- never the emulator.  The
    # generator uses the full grid only to get a mass axis and IMF weights; the
    # photometry comes from `oracle`.
    oracle = mist_grid.raw_isochrone(truth["log_age"], truth["feh"])
    gen_em = load(mist_grid.build(), order_age=order[0], order_feh=order[1])
    gen_quad = make_quadrature(gen_em, n_mass=n_mass, n_q=n_q)
    data, box = simulate(gen_em, gen_quad, truth, n_star, seed=seed,
                         oracle=oracle, mag_limit=mag_limit,
                         model_error=model_error)

    model = make_model(em, quad, box, dm_mu=truth["dm"], dm_sigma=dm_sigma)
    # target_accept 0.85, not 0.95.  The posterior is a razor-thin *curved*
    # ridge (the profile scan moves dm from 10.99 to 10.07 as log age goes 6.20
    # -> 7.10).  A high target_accept forces tiny steps, tiny steps make deep
    # trees, and short warmup at that step size never traverses the ridge -- so
    # the dense mass matrix adapts to whichever basin each chain started in.
    # That is precisely the r-hat 1.84 / ESS 3 signature seen at 0.95.
    kernel = NUTS(model, target_accept_prob=0.85, max_tree_depth=12,
                  dense_mass=[("log_age", "feh", "dm", "Av")],
                  init_strategy=init_to_median)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=draws, num_chains=CHAINS,
                chain_method="sequential", progress_bar=False)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), *data, extra_fields=("diverging",))
    dt = time.time() - t0

    idata = az.from_numpyro(mcmc)
    rhat = az.rhat(idata)
    ess = az.ess(idata)
    ess_t = az.ess(idata, method="tail")
    ndiv = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())

    row = {"name": name, "n_star": int(np.asarray(data[0]).shape[0]),
           "dm_sigma": dm_sigma, "order": list(order), "seconds": round(dt, 1),
           "divergences": ndiv, "n_eff_params": 7, "model_error": model_error,
           "stride_age": stride_age, "stride_feh": stride_feh}
    worst_rhat, worst_ess = 0.0, np.inf
    for p in PARAMS:
        post = np.asarray(idata.posterior[p]).ravel()
        r = float(np.asarray(rhat[p]))
        e = float(np.asarray(ess[p]))
        et = float(np.asarray(ess_t[p]))
        mean, sd = float(post.mean()), float(post.std(ddof=1))
        bias = mean - truth[p]
        row[p] = {"truth": truth[p], "mean": mean, "sd": sd, "bias": bias,
                  "bias_over_sd": bias / sd if sd > 0 else np.nan,
                  "rhat": r, "ess_bulk": e, "ess_tail": et}
        worst_rhat = max(worst_rhat, r)
        worst_ess = min(worst_ess, e)
    row["worst_rhat"] = worst_rhat
    row["worst_ess_bulk"] = worst_ess
    row["gate"] = bool(worst_rhat < 1.01 and worst_ess > 400 and ndiv == 0)
    row["max_abs_bias_over_sd"] = max(abs(row[p]["bias_over_sd"]) for p in PARAMS)
    return row


def _print(row):
    g = "PASS" if row["gate"] else "FAIL"
    print(f"\n--- {row['name']}  (N={row['n_star']}, dm_sigma={row['dm_sigma']}, "
          f"{row['seconds']}s) --- GATE {g}")
    print(f"    {'param':8s} {'truth':>8} {'mean':>9} {'sd':>8} {'bias':>9} "
          f"{'bias/sd':>8} {'rhat':>7} {'ESSb':>8} {'ESSt':>8}")
    for p in PARAMS:
        d = row[p]
        print(f"    {p:8s} {d['truth']:8.3f} {d['mean']:9.4f} {d['sd']:8.4f} "
              f"{d['bias']:+9.4f} {d['bias_over_sd']:+8.2f} {d['rhat']:7.4f} "
              f"{d['ess_bulk']:8.0f} {d['ess_tail']:8.0f}")
    print(f"    divergences={row['divergences']}  worst rhat={row['worst_rhat']:.4f}  "
          f"worst ESS_bulk={row['worst_ess_bulk']:.0f}  "
          f"max|bias/sd|={row['max_abs_bias_over_sd']:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short chains, smoke test")
    ap.add_argument("--only", default=None, help="run one scenario by name")
    a = ap.parse_args()
    w, d = (600, 400) if a.quick else (1500, 1000)

    OUT.mkdir(exist_ok=True)
    # 6.65 is an ODD index on the full age axis, so stride_age=2 holds it out
    # while stride_age=1 reproduces it exactly.  That is what makes control and
    # holdout differ *only* by emulator error.
    T = {"log_age": 6.65, "feh": 0.00, "dm": 10.30, "Av": 1.20}
    rows = []

    scen = [
        # 1. must pass, or nothing downstream is interpretable
        ("control (full grid)",  dict(truth=T, n_star=254)),
        # 2. THE theoretical core: same data, emulator coarsened 2x -> the
        #    difference from (1) is emulator-induced posterior bias, isolated
        ("holdout age 2x",       dict(truth=T, n_star=254, stride_age=2)),
        # 3. how much of the precision in (1) is an artefact of a perfect model

        # 3-4. the N floor -- the brief's required number, run first because it is cheap
        ("N=61 census median",   dict(truth=T, n_star=61)),
        ("N=30",                 dict(truth=T, n_star=30)),
        # 5. how much of the precision above is an artefact of a perfect model
        ("model error 0.03 mag", dict(truth=T, n_star=254, model_error=0.03)),
    ]
    for name, kw in scen:
        if a.only and a.only not in name:
            continue
        try:
            r = run_one(name, warmup=w, draws=d, **kw)
            rows.append(r)
            _print(r)
        except Exception as e:  # noqa: BLE001
            print(f"\n--- {name} --- RAISED {type(e).__name__}: {e}")
        (OUT / "injection_recovery.json").write_text(json.dumps(rows, indent=2))

    print("\n" + "=" * 78)
    print(f"{'scenario':24s} {'N':>5} {'rhat':>7} {'ESSb':>7} {'div':>4} "
          f"{'bias/sd age':>12} {'max|b/sd|':>10} {'gate':>5}")
    print("=" * 78)
    for r in rows:
        print(f"{r['name']:24s} {r['n_star']:5d} {r['worst_rhat']:7.4f} "
              f"{r['worst_ess_bulk']:7.0f} {r['divergences']:4d} "
              f"{r['log_age']['bias_over_sd']:+12.2f} "
              f"{r['max_abs_bias_over_sd']:10.2f} "
              f"{'PASS' if r['gate'] else 'FAIL':>5}")


if __name__ == "__main__":
    main()
