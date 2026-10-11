#!/usr/bin/env python3
"""One NUTS chain of one fit, in resumable chunks (the mechanics of ``isochrone_c1_chunked.py``
``run --chain-index``, measured equivalent there to ``numpyro.infer.MCMC``), on any data table.

    nuts_chain.py run --data X.ecsv --starts S.json --chain c --outdir D [--warmup W --draws N]
    nuts_chain.py finalize --data X.ecsv --outdirs D0,D1,D2,D3 --json OUT.json [--fig OUT.png]

Kernel: numpyro NUTS on PyMC's jaxified log-density of ``IsochroneFitter.build_model`` (the same
graph ``fit(nuts_sampler="numpyro")`` samples), target_accept 0.9, diagonal mass adapted from a
seed = the Laplace marginal sd mapped to the unconstrained space (``_seeded_inverse_mass``, with its
guard at a bound). Starts: ``S.json["chain_starts"][c]`` (``iv_common.chain_starts``), no jitter.
One chain per process, ``NPROC=1`` (one core).

``finalize`` writes, per parameter: posterior median, equal-tailed 68 % and 90 % intervals (sample
quantiles, nothing parabolic), ``az.rhat`` (rank-normalised split R-hat) and ``az.ess`` bulk/tail
over the chains, and **per chain**: mean, sd, divergences, mean tree depth, R-hat of that chain
against the others pooled; and the per-chain mean of the log density.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iv_common as C  # noqa: E402

STATS = ("num_steps", "step_size", "accept_prob", "diverging", "potential_energy", "energy")


def _data(path):
    from astropy.table import QTable, Table

    return QTable(Table.read(path))


def _model(data_path):
    f = C.fitter(_data(data_path))
    return f, f.build_model()


def _flatten(state):
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(state)
    return [np.asarray(x) for x in leaves], treedef


def run(a) -> None:
    import jax
    from jax import lax
    from numpyro.infer import NUTS
    from pymc.sampling.jax import _get_batched_jittered_initial_points, get_jaxified_logp

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    f, model = _model(a.data)
    st = json.loads(Path(a.starts).read_text())
    start = dict(st["chain_starts"][a.chain])
    inv_mass = f._seeded_inverse_mass(model, st)
    with model:
        logp_fn = get_jaxified_logp(model, negative_logp=False)
        init = _get_batched_jittered_initial_points(
            model=model, chains=1, initvals=[start], random_seed=a.seed + a.chain,
            jitter=False, logp_fn=logp_fn,
        )
    init = [np.asarray(v) for v in init]
    key = jax.random.PRNGKey(a.seed * 10 + a.chain)
    kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=a.target_accept,
        inverse_mass_matrix=inv_mass,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
    )
    total = a.warmup + a.draws
    cfg = {
        "data": str(a.data), "starts": str(a.starts), "chain": a.chain, "warmup": a.warmup,
        "draws": a.draws, "chunk": a.chunk, "seed": a.seed, "target_accept": a.target_accept,
        "start": start, "inverse_mass_initial": np.asarray(inv_mass).tolist(),
        "value_vars": [v.name for v in model.value_vars], "nproc": os.environ.get("NPROC", ""),
        "setup_s": round(time.time() - t0, 1),
    }
    state0 = kernel.init(key, a.warmup, init, model_args=(), model_kwargs={})
    leaves0, treedef = _flatten(state0)
    sfile = out / "state.npz"
    if sfile.exists():
        old = json.loads((out / "config.json").read_text())
        for k in ("warmup", "draws", "chunk", "seed", "chain", "target_accept", "data"):
            if old[k] != cfg[k]:
                raise SystemExit(f"resume refused: {k} {old[k]} != {cfg[k]}")
        d = np.load(sfile)
        state = jax.tree_util.tree_unflatten(
            treedef, [jax.numpy.asarray(d[f"l{i}"]) for i in range(len(leaves0))]
        )
        done = int(d["done"])
    else:
        state, done = state0, 0
        (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
        (out / "progress.jsonl").write_text("")

    def one(s):
        s = kernel.sample(s, (), {})
        return s, (s.z, s.num_steps, s.adapt_state.step_size, s.accept_prob, s.diverging,
                   s.potential_energy, s.energy)

    compiled = jax.jit(lambda s: lax.scan(lambda c, _: one(c), s, None, length=a.chunk))
    while done < total:
        t = time.time()
        state, (z, ns, ss, acc, div, pe, en) = compiled(state)
        ns = np.asarray(ns)
        dt = time.time() - t
        np.savez(
            out / f"chunk_{done:05d}.npz",
            **{f"z{i}": np.asarray(v) for i, v in enumerate(z)},
            num_steps=ns, step_size=np.asarray(ss), accept_prob=np.asarray(acc),
            diverging=np.asarray(div), potential_energy=np.asarray(pe), energy=np.asarray(en),
        )
        done += a.chunk
        leaves, _ = _flatten(state)
        tmp = out / "state.tmp.npz"
        np.savez(tmp, done=done, **{f"l{i}": x for i, x in enumerate(leaves)})
        tmp.replace(sfile)
        rec = {
            "iter": done, "phase": "warmup" if done <= a.warmup else "sampling",
            "chunk_s": round(dt, 2), "leapfrog": int(ns.sum()),
            "depth_mean": round(float(np.mean(np.floor(np.log2(np.maximum(ns, 1))) + 1)), 2),
            "step_size": float(np.asarray(ss)[-1]), "div": int(np.asarray(div).sum()),
            "accept": round(float(np.asarray(acc).mean()), 3), "wall": C.now(),
        }
        with open(out / "progress.jsonl", "a") as log:
            log.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    (out / "DONE").write_text(C.now() + "\n")


def load_chains(outdirs):
    zs, cats, cfgs = [], [], []
    for od in outdirs:
        od = Path(od)
        cfg = json.loads((od / "config.json").read_text())
        parts = [np.load(p) for p in sorted(od.glob("chunk_*.npz"))]
        nvar = len(cfg["value_vars"])
        total = cfg["warmup"] + cfg["draws"]
        z = [np.concatenate([p[f"z{i}"] for p in parts])[cfg["warmup"]:total] for i in range(nvar)]
        cat = {k: np.concatenate([p[k] for p in parts])[cfg["warmup"]:total] for k in STATS}
        if len(z[0]) < cfg["draws"]:
            raise SystemExit(f"{od}: {len(z[0])} of {cfg['draws']} draws on disk")
        zs.append(z)
        cats.append(cat)
        cfgs.append(cfg)
    z = [np.stack([zz[i] for zz in zs]) for i in range(len(zs[0]))]
    cat = {k: np.stack([c[k] for c in cats]) for k in STATS}
    return z, cat, cfgs


def posterior(data_path, outdirs):
    import jax
    from arviz_base import from_dict
    from pymc.sampling.jax import get_jaxified_graph
    from pymc.util import get_default_varnames

    z, cat, cfgs = load_chains(outdirs)
    f, model = _model(data_path)
    outs = list(get_default_varnames(model.unobserved_value_vars, include_transformed=False))
    fn = get_jaxified_graph(inputs=model.value_vars, outputs=outs)
    res = jax.vmap(jax.vmap(fn))(*[jax.numpy.asarray(v) for v in z])
    post = {v.name: np.asarray(r) for v, r in zip(outs, res, strict=True)}
    stats = {
        "diverging": cat["diverging"], "n_steps": cat["num_steps"],
        "tree_depth": np.floor(np.log2(np.maximum(cat["num_steps"], 1))).astype(int) + 1,
        "lp": -cat["potential_energy"], "energy": cat["energy"],
    }
    return from_dict({"posterior": post, "sample_stats": stats}), post, stats, cfgs


def summarize(post, stats) -> dict:
    import arviz as az

    out = {"params": {}, "chains": {}}
    nch = next(iter(post.values())).shape[0]
    for p in C.PARAMS:
        x = post[p]
        flat = x.ravel()
        q = np.quantile(flat, [0.05, 0.16, 0.5, 0.84, 0.95])
        rh = float(az.rhat(x))
        eb = float(az.ess(x, method="bulk"))
        et = float(az.ess(x, method="tail"))
        out["params"][p] = {
            "median": q[2], "q05": q[0], "q16": q[1], "q84": q[3], "q95": q[4],
            "mean": float(flat.mean()), "sd": float(flat.std()),
            "rhat": rh, "ess_bulk": eb, "ess_tail": et,
            "chain_mean": x.mean(1).tolist(), "chain_sd": x.std(1).tolist(),
            # per chain: split R-hat of that chain alone (non-stationarity) and its mean's offset
            # from the other chains', in pooled sd (a chain stuck elsewhere shows here)
            "rhat_split_chain": [float(az.rhat(x[c][None])) for c in range(nch)],
            "offset_sd": [
                float((x[c].mean() - np.delete(x, c, 0).mean()) / max(flat.std(), 1e-12))
                for c in range(nch)
            ],
        }
    out["chains"] = {
        "divergences": stats["diverging"].sum(1).tolist(),
        "tree_depth_mean": stats["tree_depth"].mean(1).round(2).tolist(),
        "lp_mean": stats["lp"].mean(1).tolist(),
        "n_draws": int(stats["lp"].shape[1]),
    }
    P = out["params"]
    out["gate"] = {
        "rhat_max": max(P[p]["rhat"] for p in C.PARAMS),
        "ess_bulk_min": min(P[p]["ess_bulk"] for p in C.PARAMS),
        "ess_tail_min": min(P[p]["ess_tail"] for p in C.PARAMS),
        "divergences": int(stats["diverging"].sum()),
        "rhat_split_chain_max": max(max(P[p]["rhat_split_chain"]) for p in C.PARAMS),
    }
    g = out["gate"]
    g["converged"] = bool(
        g["rhat_max"] <= 1.01 and g["ess_bulk_min"] >= 400 and g["ess_tail_min"] >= 400
        and g["divergences"] == 0 and g["rhat_split_chain_max"] <= 1.01
    )
    return out


def pairfig(post, path, truth=None, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = ("feh", "loga", "dm", "Av", "sigma_int")
    n = len(ps)
    fig, ax = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    nch = post["loga"].shape[0]
    cols = plt.cm.tab10(np.arange(nch))
    for i, pi in enumerate(ps):
        for j, pj in enumerate(ps):
            a = ax[i, j]
            if j > i:
                a.axis("off")
                continue
            for c in range(nch):
                if i == j:
                    a.hist(post[pi][c], bins=40, histtype="step", color=cols[c], density=True)
                else:
                    a.plot(post[pj][c][::4], post[pi][c][::4], ".", ms=1, alpha=0.3, color=cols[c])
            if truth is not None:
                if i == j:
                    a.axvline(truth[pi], color="k", lw=1)
                else:
                    a.plot(truth[pj], truth[pi], "k+", ms=10, mew=1.5)
            if i == n - 1:
                a.set_xlabel(pj)
            if j == 0 and i > 0:
                a.set_ylabel(pi)
    fig.suptitle(title + "  (colour = chain; black = truth)" if truth else title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def tracefig(post, path, title=""):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ps = ("feh", "loga", "dm", "Av")
    fig, ax = plt.subplots(len(ps), 1, figsize=(8, 1.6 * len(ps)), sharex=True)
    for a, p in zip(ax, ps, strict=True):
        for c in range(post[p].shape[0]):
            a.plot(post[p][c], lw=0.4)
        a.set_ylabel(p)
    ax[0].set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def finalize(a) -> None:
    _, post, stats, cfgs = posterior(a.data, a.outdirs.split(","))
    s = summarize(post, stats)
    s["config"] = {k: cfgs[0][k] for k in ("warmup", "draws", "seed", "target_accept")}
    s["starts"] = [c["start"] for c in cfgs]
    s["sampling_s_per_chain"] = [
        round(sum(json.loads(x).get("chunk_s", 0)
                  for x in (Path(od) / "progress.jsonl").read_text().splitlines()))
        for od in a.outdirs.split(",")
    ]
    truth = json.loads(a.truth) if a.truth else None
    if a.fig:
        pairfig(post, a.fig, truth, a.title)
        tracefig(post, a.fig.replace(".png", "_trace.png"), a.title)
    if a.save_draws:
        np.savez_compressed(a.save_draws, **{p: post[p] for p in C.PARAMS})
    C.dump(Path(a.json), s)
    print(json.dumps(s["gate"]), json.dumps({p: [round(s["params"][p][k], 4) for k in ("q05", "q16", "median", "q84", "q95")] for p in ("feh", "loga")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "finalize"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--starts")
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--outdir")
    ap.add_argument("--outdirs")
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--chunk", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--json")
    ap.add_argument("--fig")
    ap.add_argument("--truth", default="")
    ap.add_argument("--title", default="")
    ap.add_argument("--save-draws", default="")
    a = ap.parse_args()
    run(a) if a.stage == "run" else finalize(a)
