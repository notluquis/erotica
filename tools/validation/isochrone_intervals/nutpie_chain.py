#!/usr/bin/env python3
"""One NUTS chain of one fit with **nutpie** (JAX backend), started at its own over-dispersed point,
and the four-chain summary in the schema of ``nuts_chain.summarize`` (hub finding
``isochrone-intervals-nuts.md`` §3; pre-registration ``r113-nutpie-rerun-prereg-2026-11-12.md``).

    nutpie_chain.py run --kind iv|kw --fit DIR --chain c --warmup W --draws N --seed S
    nutpie_chain.py finalize --kind iv|kw --fit DIR [--title T]

Why a separate runner. The cited fits used ``nuts_chain.py`` / ``nuts8.py``: numpyro's NUTS kernel on
PyMC's jaxified density, one chain per process, each chain from ``starts.json["chain_starts"][c]``
(over-dispersed along the age-[Fe/H] ridge, ``iv_common.chain_starts``), no jitter. The author's
standing decision (2026-10-19) is nutpie as the NUTS engine. nutpie takes ONE start per call
(``pm.sample`` refuses a list of ``initvals``), so the over-dispersed design is kept by running one
chain per process, each with its own start. Two settings differ from nutpie's defaults on purpose:

* ``jitter_rvs=set()``: nutpie's ``compile_pymc_model`` jitters EVERY free variable by U(-1, 1) in the
  transformed space when no strategy is given, which would replace the declared start by a random
  point (on log t's interval transform, +-1 is most of the prior range). The cited runs used no jitter.
* ``mode="JAX", gradient_backend="jax"``: the package's ``NUTPIE_COMPILE_KWARGS``. numba with
  fastmath gave "same seed, another fit" (decisions.md, 2026-10-21).

target_accept 0.9 as in the cited runs. The mass matrix is nutpie's adaptation (the cited runs seeded
it from the Laplace sd); that difference, and the engine itself, are what the sampler control of the
pre-registration measures (A_0 re-run with both engines on the same data). One core per process (``cores=1``, ``chains=1``).

``run`` writes ``chain<c>/{draws.npz, run.json, DONE}``: the posterior draws of every free variable
(constrained), the sampler stats (diverging, n_steps, tree depth, logp), the seed, the start, the
package versions, the wall time and the peak RSS. ``finalize`` stacks the four chains and writes
``nuts.json`` (``nuts_chain.summarize`` + config), ``draws.npz`` (what ``chain_z.py`` reads) and the
pair / trace figures.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "isochrone_modelfix"))


def _engine(kind: str):
    """(module with ``_model`` / ``summarize`` / ``pairfig`` / ``tracefig``, parameter names)."""
    if kind == "iv":
        import nuts_chain as N

        return N, tuple(N.C.PARAMS)
    os.environ.setdefault("ARM", "K")
    os.environ.setdefault("WINDOW", "W")
    import nuts8  # noqa: F401  (patches nuts_chain: K x W model, PARAMS + k, figures)
    import nuts_chain as N

    return N, tuple(N.C.PARAMS)


def run(a) -> None:
    import pymc as pm

    fit = Path(a.fit)
    out = fit / f"chain{a.chain}"
    if (out / "DONE").exists():
        print(f"{out}: DONE, skipped", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    N, params = _engine(a.kind)
    t0 = time.time()
    f, model = N._model(fit / "data.ecsv")
    st = json.loads((fit / "starts.json").read_text())
    start = {k: float(v) for k, v in st["chain_starts"][a.chain].items()}
    names = {v.name for v in model.free_RVs}
    start = {k: v for k, v in start.items() if k in names}
    t1 = time.time()
    with model:
        idata = pm.sample(
            draws=a.draws,
            tune=a.warmup,
            chains=1,
            cores=1,
            random_seed=a.seed,
            nuts={"target_accept": a.target_accept},
            nuts_sampler="nutpie",
            initvals=start,
            compile_kwargs={"mode": "JAX", "gradient_backend": "jax", "jitter_rvs": set()},
            progressbar=False,
        )
    t2 = time.time()
    post = {p: np.asarray(idata.posterior[p]).reshape(-1) for p in params if p in idata.posterior}
    ss = idata.sample_stats

    def stat(*keys):
        for k in keys:
            if k in ss:
                return np.asarray(ss[k]).reshape(-1)
        return None

    stats = {
        "diverging": stat("diverging"),
        "n_steps": stat("n_steps"),
        "tree_depth": stat("depth", "tree_depth"),
        "lp": stat("logp", "lp"),
        "step_size": stat("step_size"),
    }
    np.savez(
        out / "draws.npz", **post, **{f"stat_{k}": v for k, v in stats.items() if v is not None}
    )
    import jax
    import nutpie

    rec = {
        "kind": a.kind,
        "fit": str(fit),
        "chain": a.chain,
        "warmup": a.warmup,
        "draws": a.draws,
        "seed": a.seed,
        "target_accept": a.target_accept,
        "start": start,
        "engine": "nutpie",
        "compile_kwargs": {"mode": "JAX", "gradient_backend": "jax", "jitter_rvs": "set()"},
        "versions": {"pymc": pm.__version__, "nutpie": nutpie.__version__, "jax": jax.__version__},
        "build_s": round(t1 - t0, 1),
        "sample_s": round(t2 - t1, 1),
        # ru_maxrss is in bytes on macOS and in KiB on Linux
        "peak_rss_mb": round(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / (2**20 if sys.platform == "darwin" else 2**10),
            1,
        ),
        "first_draw": {p: float(v[0]) for p, v in post.items()},
    }
    (out / "run.json").write_text(json.dumps(rec, indent=1) + "\n")
    (out / "DONE").write_text(time.strftime("%F %T") + "\n")
    print(json.dumps({k: rec[k] for k in ("fit", "chain", "sample_s", "peak_rss_mb")}), flush=True)


def finalize(a) -> None:
    fit = Path(a.fit)
    N, params = _engine(a.kind)
    chains = sorted(fit.glob("chain*/draws.npz"))
    if len(chains) != 4 or not all((c.parent / "DONE").exists() for c in chains):
        raise SystemExit(f"{fit}: {len(chains)} finished chains, need 4")
    ds = [np.load(c) for c in chains]
    post = {p: np.stack([d[p] for d in ds]) for p in params if p in ds[0].files}
    stats = {
        "diverging": np.stack([d["stat_diverging"] for d in ds]).astype(bool),
        "tree_depth": np.stack([d["stat_tree_depth"] for d in ds]),
        "lp": np.stack([d["stat_lp"] for d in ds]),
    }
    N.C.PARAMS = tuple(p for p in params if p in post)
    res = N.summarize(post, stats)
    runs = [json.loads((c.parent / "run.json").read_text()) for c in chains]
    res["config"] = {
        "engine": "nutpie",
        "seeds": [r["seed"] for r in runs],
        "target_accept": runs[0]["target_accept"],
        "warmup": runs[0]["warmup"],
        "draws": runs[0]["draws"],
        "versions": runs[0]["versions"],
        "compile_kwargs": runs[0]["compile_kwargs"],
        "starts": [r["start"] for r in runs],
    }
    res["sampling_s_per_chain"] = [r["sample_s"] for r in runs]
    res["peak_rss_mb"] = [r["peak_rss_mb"] for r in runs]
    prep = json.loads((fit / "prep.json").read_text()) if (fit / "prep.json").exists() else {}
    truth = prep.get("truth_pymc") or prep.get("truth")
    from iv_common import dump

    dump(fit / "nuts.json", res)
    np.savez(fit / "draws.npz", **post)
    N.pairfig(post, fit / "pair.png", truth=truth, title=a.title or f"{fit.name} nutpie")
    N.tracefig(post, fit / "pair_trace.png", title=a.title or f"{fit.name} nutpie")
    g = res["gate"]
    print(fit.name, {k: g[k] for k in ("rhat_max", "ess_bulk_min", "ess_tail_min", "divergences")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "finalize"])
    ap.add_argument("--kind", choices=["iv", "kw"], required=True)
    ap.add_argument("--fit", required=True)
    ap.add_argument("--chain", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    run(a) if a.stage == "run" else finalize(a)
