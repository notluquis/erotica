#!/usr/bin/env python3
r"""C1 (the NGC 6383 NUTS certificate) run in resumable chunks, with its progress on disk.

WHY THIS EXISTS
---------------
C1 of the hub finding ``isochrone-nuts-convergence-2026-09.md`` §9.5 (4 chains x 2000 draws after
2000 tune, numpyro, target_accept 0.9, chains started from ``find_start``) was launched three
times through ``pm.sample``/``sample_jax_nuts`` and never finished: the last attempt ran 21.4 h
with numpyro's progress bar at ``0/4000`` in all four chains. Inside ``MCMC.run`` nothing reaches
the disk until the end, so a stopped run loses everything and a slow run cannot be told from a
hung one. This script runs **the same Markov chains** -- same jaxified log-density of the same
PyMC model, same numpyro ``NUTS`` kernel and settings, same initial points and the same PRNG keys
as ``sample_jax_nuts`` derives them from ``random_seed`` -- but calls ``kernel.sample`` itself in
chunks of ``--chunk`` iterations (``lax.scan``), and after every chunk writes:

* ``progress.jsonl``: iteration reached, wall time of the chunk, and per chain the leapfrog
  steps, max tree depth, step size, divergences and acceptance of that chunk;
* ``chunk_<k>.npz``: the unconstrained draws and sample stats of that chunk;
* ``state.npz``: the full kernel state (position, adaptation state, keys), so a killed run
  resumes from the last chunk instead of from zero.

``numpyro.infer.MCMC`` does the same thing internally (``fori_collect`` over ``kernel.sample``
from ``kernel.init``), so the chain is the same process; the only thing that changes is where
the loop lives. Warmup is part of the same state (``state.i`` and ``adapt_state``), so a chunk
boundary inside warmup does not restart adaptation.

STAGES
------
``search``   run ``find_start(4, default_rng(42), mode="JAX")`` once and cache it (it is
             deterministic and took 1299 s in the last attempt).
``gradbench`` time one log-density gradient of the model (compile excluded), per chain start.
``run``      the chunked sampler; ``--warmup/--draws`` give the run length.
``finalize`` transform to the constrained space as ``sample_jax_nuts`` does, and write the same
             summary as ``isochrone_unbinned_recovery.py run ngc 0`` (its ``_summ``).

``float32``  log L and gradient at the four starts in float32 against float64 (no sampling).
``equivcheck`` the driver against ``numpyro.infer.MCMC`` for one chain (see below).
``diagsummary`` the 2026-09-29 diagnosis arms, from ``~/.cache/erotica-c1``, into the committed
             sidecar ``isochrone_c1_diag.json``.

HOW IT IS RUN FOR C1 (measured 2026-09-29, hub finding §10.18)
--------------------------------------------------------------
Four processes, one chain each (``--chain-index 0..3``), each with ``NPROC=1`` and
``XLA_FLAGS=--xla_force_host_platform_device_count=1``. ``NPROC`` is what sizes XLA's CPU
thread pool here; ``--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`` left
235 threads and made the run slower. One gradient uses ~5 cores by default (0.065 s) and 1.0
core with ``NPROC=1`` (0.133 s alone, 0.21-0.25 s with four such processes): four of them use
4.0 cores. ``finalize --outdir d0,d1,d2,d3`` joins them in chain order.

WHAT WOULD FALSIFY THE EQUIVALENCE
----------------------------------
A chain that differs in distribution from ``sample_jax_nuts`` with the same arguments. Checked
by ``equivcheck``: a short run of both from the same starts and keys, compared draw by draw.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
CACHE = Path(os.path.expanduser("~/.cache/erotica-c1"))


def _head() -> dict:
    head = subprocess.run(
        ["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(HERE.parent.parent), "status", "--porcelain", "erotica"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {"head": head, "erotica_dirty": dirty}


def _fitter():
    from astropy.table import QTable, Table
    from isochrone_nuts_convergence import MIST, PRIORS, SAMPLE

    from erotica.analysis._isochrone import IsochroneFitter

    pri = {k: v for k, v in PRIORS.items() if k not in ("M_met", "M_loga")}
    f = IsochroneFitter(isochs_path=MIST, **pri)
    f.setup(QTable(Table.read(SAMPLE)), prob_threshold=0.0)
    return f


def stage_search(out: Path) -> None:
    import erotica

    t0 = time.time()
    f = _fitter()
    t1 = time.time()
    found = f.find_start(4, np.random.default_rng(42), mode="JAX")
    t2 = time.time()
    found = json.loads(json.dumps(found, default=float))
    found.update({"t_setup_s": round(t1 - t0, 1), "t_search_s": round(t2 - t1, 1), **_head()})
    found["erotica_file"] = erotica.__file__
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(found, indent=1) + "\n")
    print(json.dumps({k: found[k] for k in ("mode", "local_sd", "t_setup_s", "t_search_s")}))


def _inverse_mass(f, model, found: dict, variant: str) -> np.ndarray:
    """``seeded``: the package's (guarded at bounds since 2026-09-29); ``seeded_c1``: the
    unguarded map every earlier C1 attempt ran with, reproduced here for the before arm;
    ``identity``: numpyro's default."""
    seeded = f._seeded_inverse_mass(model, found)
    if variant == "seeded":
        return seeded
    if variant == "identity":
        return np.ones_like(seeded)
    if variant == "seeded_c1":
        bounds = {
            "met": (10.0 ** float(f._node_logz[0]), 10.0 ** float(f._node_logz[-1])),
            "loga": tuple(f.loga_range),
            "dm": tuple(f.dm_range),
            "Av": tuple(f.Av_range),
        }
        out = []
        for rv in model.free_RVs:
            if rv.name in bounds and rv.name in found["local_sd"]:
                a, b = bounds[rv.name]
                x = float(np.clip(found["mode"][rv.name], a + 1e-9 * (b - a), b - 1e-9 * (b - a)))
                sd = float(found["local_sd"][rv.name])
                out.append((sd * (b - a) / ((x - a) * (b - x))) ** 2)
            else:
                out.append(1.0)
        return np.asarray(out)
    raise ValueError(variant)


def _prepare(search: Path, mass: str, chains: int, seed: int):
    """Everything ``sample_jax_nuts`` builds before ``MCMC.run``, timed."""
    import jax
    from pymc.sampling.jax import _get_batched_jittered_initial_points, get_jaxified_logp
    from pymc.util import _get_seeds_per_chain

    T = {}
    t = time.time()
    f = _fitter()
    T["setup_s"] = time.time() - t
    found = json.loads(search.read_text())
    found["chain_starts"] = found["chain_starts"][:chains]
    t = time.time()
    model = f.build_model()
    T["build_model_s"] = time.time() - t
    inv_mass = _inverse_mass(f, model, found, mass)
    t = time.time()
    with model:
        logp_fn = get_jaxified_logp(model, negative_logp=False)
        (rs,) = _get_seeds_per_chain(seed, 1)
        init = _get_batched_jittered_initial_points(
            model=model,
            chains=chains,
            initvals=found["chain_starts"],
            random_seed=rs,
            jitter=False,
            logp_fn=logp_fn,
        )
    T["jaxify_and_initial_points_s"] = time.time() - t
    if chains == 1:
        init = [np.asarray(v)[None] for v in init]
    init = [np.asarray(v) for v in init]
    keys = jax.random.PRNGKey(rs)
    keys = jax.random.split(keys, chains) if chains > 1 else keys[None]
    return f, model, logp_fn, init, keys, inv_mass, T, found


def stage_gradbench(args) -> None:
    import jax

    f, model, logp_fn, init, keys, inv_mass, T, found = _prepare(
        Path(args.search), args.mass, 4, 42
    )
    vg = jax.jit(jax.value_and_grad(logp_fn))
    x0 = [v[0] for v in init]
    t = time.time()
    jax.block_until_ready(vg(x0))
    T["first_grad_incl_compile_s"] = time.time() - t
    import resource

    per, cpu = [], []
    for c in range(init[0].shape[0]):
        xc = [v[c] for v in init]
        jax.block_until_ready(vg(xc))
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t = time.time()
        for _ in range(args.reps):
            out = vg(xc)
            jax.block_until_ready(out)  # one gradient at a time, as a leapfrog step is
        dt = time.time() - t
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        per.append(dt / args.reps)
        cpu.append((r1.ru_utime - r0.ru_utime + r1.ru_stime - r0.ru_stime) / dt)
    T["s_per_grad_by_chain_start"] = per
    T["cores_busy_during_grads"] = cpu
    T["potential_at_starts"] = [float(vg([v[c] for v in init])[0]) for c in range(len(per))]
    T["inverse_mass_seeded"] = f._seeded_inverse_mass(model, found).tolist()
    T["value_vars"] = [v.name for v in model.value_vars]
    T["init_unconstrained"] = [np.asarray(v).tolist() for v in init]
    T.update(_head())
    T["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    print(json.dumps(T, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(T, indent=1) + "\n")


def stage_float32(args) -> None:
    """log L and its gradient at the four chain starts, float64 against float32. Only a
    comparison of the function: nothing is sampled in float32."""
    import jax
    import jax.numpy as jnp

    f, model, logp_fn, init, keys, inv_mass, T, found = _prepare(Path(args.search), "seeded", 4, 42)
    vg = jax.jit(jax.value_and_grad(logp_fn))
    out = []
    for c in range(4):
        x64 = [jnp.asarray(v[c], dtype=jnp.float64) for v in init]
        v64, g64 = vg(x64)
        x32 = [jnp.asarray(v[c], dtype=jnp.float32) for v in init]
        with jax.enable_x64(False):
            vg32 = jax.jit(jax.value_and_grad(logp_fn))
            v32, g32 = vg32(x32)
        g64 = np.array([float(g) for g in g64])
        g32 = np.array([float(g) for g in g32])
        out.append(
            {
                "potential64": float(v64),
                "potential32": float(v32),
                "dtype32": str(v32.dtype),
                "abs_diff": float(abs(float(v64) - float(v32))),
                "grad64": g64.tolist(),
                "grad32": g32.tolist(),
                "grad_rel_diff": (np.abs(g32 - g64) / np.maximum(np.abs(g64), 1e-12)).tolist(),
            }
        )
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


def _flatten(state):
    import jax

    leaves, treedef = jax.tree_util.tree_flatten(state)
    return [np.asarray(x) for x in leaves], treedef


def stage_run(args) -> None:
    import jax
    from jax import lax
    from numpyro.infer import NUTS

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    chains = args.chains
    f, model, logp_fn, init, keys, inv_mass, T, found = _prepare(
        Path(args.search), args.mass, chains, args.seed
    )
    if args.chain_index >= 0:
        # one chain of the ``chains``-chain run, with its own start and key, as derived for
        # all of them: four such processes are the four chains of one run
        c = args.chain_index
        init, keys, chains = [v[c : c + 1] for v in init], keys[c : c + 1], 1
    # the kernel exactly as pymc's _sample_numpyro_nuts builds it
    kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=args.target_accept,
        inverse_mass_matrix=inv_mass,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
    )
    total = args.warmup + args.draws
    t = time.time()
    # one init per chain with its own key, as MCMC(chain_method="parallel") does: a batched
    # key would make HMC.init vmap and replace kernel._sample_fn by a vmapped one, which the
    # pmap/vmap below would then map a second time
    per_chain = [
        kernel.init(keys[c], args.warmup, [v[c] for v in init], model_args=(), model_kwargs={})
        for c in range(chains)
    ]
    state0 = jax.tree_util.tree_map(lambda *xs: jax.numpy.stack(xs), *per_chain)
    state0 = jax.block_until_ready(state0)
    T["kernel_init_s"] = time.time() - t
    cfg = {
        "warmup": args.warmup,
        "draws": args.draws,
        "chains": chains,
        "chain_index": args.chain_index,
        "chunk": args.chunk,
        "mass": args.mass,
        "chain_method": args.chain_method,
        "target_accept": args.target_accept,
        "seed": args.seed,
        "inverse_mass_initial": np.asarray(inv_mass).tolist(),
        "value_vars": [v.name for v in model.value_vars],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "jax_cache": os.environ.get("JAX_COMPILATION_CACHE_DIR", ""),
        "nproc": os.environ.get("NPROC", ""),
        "devices": len(jax.devices()),
        **_head(),
    }
    leaves0, treedef = _flatten(state0)
    sfile = outdir / "state.npz"
    if sfile.exists() and not args.fresh:
        old = json.loads((outdir / "config.json").read_text())
        for k in (
            "warmup",
            "draws",
            "chains",
            "chain_index",
            "chunk",
            "mass",
            "target_accept",
            "seed",
        ):
            if old[k] != cfg[k]:
                raise SystemExit(f"resume refused: {k} {old[k]} != {cfg[k]}")
        d = np.load(sfile)
        leaves = [d[f"l{i}"] for i in range(len(leaves0))]
        state = jax.tree_util.tree_unflatten(treedef, [jax.numpy.asarray(x) for x in leaves])
        done = int(d["done"])
        print(f"resuming at iteration {done}", flush=True)
    else:
        state, done = state0, 0
        (outdir / "config.json").write_text(json.dumps({**cfg, "prep": T}, indent=1) + "\n")
        (outdir / "progress.jsonl").write_text("")

    def one(s):
        s = kernel.sample(s, (), {})
        return s, (
            s.z,
            s.num_steps,
            s.adapt_state.step_size,
            s.accept_prob,
            s.diverging,
            s.potential_energy,
            s.energy,
        )

    def chunk_fn(s):
        return lax.scan(lambda c, _: one(c), s, None, length=args.chunk)

    if args.chain_method == "parallel":
        runner = jax.pmap(chunk_fn)
    else:
        runner = jax.jit(jax.vmap(chunk_fn))
    t = time.time()
    compiled = runner.lower(state).compile()
    t_compile = time.time() - t
    with open(outdir / "progress.jsonl", "a") as log:
        log.write(json.dumps({"event": "compiled", "s": round(t_compile, 2), "at": done}) + "\n")
    stop = min(total, args.stop_at) if args.stop_at else total
    while done < stop:
        t = time.time()
        state, (z, ns, ss, acc, div, pe, en) = compiled(state)
        ns = np.asarray(ns)  # blocks: the chunk is done
        dt = time.time() - t
        k0 = done
        done += args.chunk
        np.savez(
            outdir / f"chunk_{k0:05d}.npz",
            **{f"z{i}": np.asarray(v) for i, v in enumerate(z)},
            num_steps=ns,
            step_size=np.asarray(ss),
            accept_prob=np.asarray(acc),
            diverging=np.asarray(div),
            potential_energy=np.asarray(pe),
            energy=np.asarray(en),
        )
        leaves, _ = _flatten(state)
        tmp = outdir / "state.tmp.npz"
        np.savez(tmp, done=done, **{f"l{i}": x for i, x in enumerate(leaves)})
        tmp.replace(sfile)
        depth = np.floor(np.log2(np.maximum(ns, 1))).astype(int) + 1
        rec = {
            "iter": done,
            "phase": "warmup" if done <= args.warmup else "sampling",
            "chunk_s": round(dt, 2),
            "s_per_iter": round(dt / args.chunk, 3),
            "leapfrog_per_chain": ns.sum(1).tolist(),
            "s_per_leapfrog_max_chain": round(dt / max(1, int(ns.sum(1).max())), 5),
            "tree_depth_max": depth.max(1).tolist(),
            "tree_depth_mean": np.round(depth.mean(1), 2).tolist(),
            "saturated_10": (depth >= 10).sum(1).tolist(),
            "step_size_last": np.asarray(ss)[:, -1].tolist(),
            "divergences": np.asarray(div).sum(1).tolist(),
            "accept_mean": np.round(np.asarray(acc).mean(1), 3).tolist(),
            "wall": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(outdir / "progress.jsonl", "a") as log:
            log.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)


STATS = ("num_steps", "step_size", "accept_prob", "diverging", "potential_energy", "energy")


def _load_draws(outdirs: str, keep_warmup: bool = False) -> tuple[list, dict]:
    """Post-warmup draws (unconstrained) and stats of one run, from one directory or from a
    comma-separated list of single-chain directories (``--chain-index``), in that order."""
    zs, cats = [], []
    for od in outdirs.split(","):
        od = Path(od)
        cfg = json.loads((od / "config.json").read_text())
        parts = [np.load(p) for p in sorted(od.glob("chunk_*.npz"))]
        nvar = len(cfg["value_vars"])
        cat = {k: np.concatenate([p[k] for p in parts], axis=1) for k in STATS}
        z = [np.concatenate([p[f"z{i}"] for p in parts], axis=1) for i in range(nvar)]
        total = cfg["warmup"] + cfg["draws"]
        if z[0].shape[1] < total:
            raise SystemExit(f"{od}: only {z[0].shape[1]} of {total} iterations on disk")
        w = 0 if keep_warmup else cfg["warmup"]
        zs.append([v[:, w:total] for v in z])
        cats.append({k: v[:, w:total] for k, v in cat.items()})
    z = [np.concatenate([zz[i] for zz in zs], axis=0) for i in range(len(zs[0]))]
    cat = {k: np.concatenate([c[k] for c in cats], axis=0) for k in STATS}
    return z, cat


def stage_equivcheck(args) -> None:
    """The chunked driver against ``numpyro.infer.MCMC`` -- what ``sample_jax_nuts`` calls --
    for chain ``--chain-index`` of the 4-chain run, from the same kernel, start and key, in the
    same process configuration as the relaunch (one chain per process). Pass: the same number
    of leapfrog steps in every iteration and draws equal to 1e-8. ``--mutate`` moves the
    driver's seed by one and must fail it."""
    from numpyro.infer import MCMC, NUTS

    W, S, c = args.warmup, args.draws, max(args.chain_index, 0)
    f, model, logp_fn, init, keys, inv_mass, T, found = _prepare(
        Path(args.search), args.mass, 4, 42
    )
    kernel = NUTS(
        potential_fn=logp_fn,
        target_accept_prob=args.target_accept,
        inverse_mass_matrix=inv_mass,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
    )
    mcmc = MCMC(kernel, num_warmup=W, num_samples=S, num_chains=1, progress_bar=False)
    t = time.time()
    mcmc.run(keys[c], init_params=[v[c] for v in init], extra_fields=("num_steps",))
    t_ref = time.time() - t
    ref = [np.asarray(v)[None] for v in mcmc.get_samples()]
    ref_ns = np.asarray(mcmc.get_extra_fields()["num_steps"])[None]
    ns = argparse.Namespace(**vars(args))
    ns.seed = 43 if args.mutate else 42
    ns.fresh, ns.chain_index, ns.stop_at, ns.chains = True, c, 0, 4
    stage_run(ns)
    z, cat = _load_draws(args.outdir)
    diff = max(float(np.max(np.abs(a - b))) for a, b in zip(ref, z, strict=True))
    same_steps = bool(np.array_equal(ref_ns, cat["num_steps"]))
    out = {
        "chain_index": c,
        "warmup": W,
        "draws": S,
        "mutate_seed": bool(args.mutate),
        "max_abs_diff_unconstrained": diff,
        "same_leapfrog_counts": same_steps,
        "leapfrog_ref": ref_ns.tolist(),
        "leapfrog_driver": cat["num_steps"].tolist(),
        "mcmc_seconds": round(t_ref, 1),
        "nproc": os.environ.get("NPROC", ""),
        "pass": bool(same_steps and diff < 1e-8),
        **_head(),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")


DIAG_ARMS = {
    # arm: (directory under CACHE, what changed)
    "seeded_c1_par4": (
        "probe_seeded",
        "C1 as run: unguarded seeded mass, pmap x4, default threads",
    ),
    "guarded_par4": ("probe_guarded_par", "guarded mass, pmap x4, default threads"),
    "guarded_par4_eigenflags": (
        "probe_guarded_par_1t",
        "guarded, pmap x4, --xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
    ),
    "guarded_vectorized": ("probe_guarded_vec", "guarded, vmap x4 on one device"),
    "guarded_1chain": ("probe_guarded_1ch", "guarded, one chain, one device, default threads"),
    "guarded_par2": ("probe_guarded_par2", "guarded, pmap x2 (another process at ~1 core)"),
    "guarded_par4_nproc4": ("probe_guarded_par_nproc4", "guarded, pmap x4, NPROC=4"),
    **{
        f"guarded_4proc_nproc1_c{c}": (f"probe_4proc_c{c}", f"guarded, chain {c} alone, NPROC=1")
        for c in range(4)
    },
}


def stage_diagsummary(args) -> None:
    out = {"arms": {}, "note": "chunk_s is wall time; see asleep_s for chunks the laptop slept in"}
    for arm, (d, what) in DIAG_ARMS.items():
        p = CACHE / d / "progress.jsonl"
        if not p.exists():
            continue
        rows = [json.loads(line) for line in p.read_text().splitlines()]
        out["arms"][arm] = {"what": what, "chunks": [r for r in rows if "iter" in r]}
        out["arms"][arm]["compile_s"] = [r["s"] for r in rows if r.get("event") == "compiled"]
        log = CACHE / "logs" / f"{d}.log"
        if log.exists():  # /usr/bin/time -l of the whole process (setup included)
            m = re.search(r"([\d.]+) real\s+([\d.]+) user\s+([\d.]+) sys", log.read_text())
            if m:
                real, user, sys_ = map(float, m.groups())
                out["arms"][arm]["process"] = {
                    "real_s": real,
                    "user_s": user,
                    "sys_s": sys_,
                    "cores": round((user + sys_) / real, 2),
                }
    for name in (
        "gradbench_default",
        "gradbench_background",
        "gradbench_nproc1",
        "gradbench_nproc2",
        "gradbench_cache_cold",
        "gradbench_cache_warm",
        "float32",
        "equiv_c0",
        "equiv_c0_mut",
        "equiv_c3",
    ):
        p = CACHE / f"{name}.json"
        if p.exists():
            d = json.loads(p.read_text())
            if isinstance(d, dict):
                d.pop("init_unconstrained", None)
            out[name] = d
    # the laptop slept 777 s inside the seeded arm's third chunk (pmset log, 12:17:18-12:30:54)
    out["asleep_s"] = {"seeded_c1_par4": {"chunk_ending_iter_15": 777}}
    target = HERE / "isochrone_c1_diag.json"
    target.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {target}")


def stage_finalize(args) -> None:
    import jax
    from arviz_base import from_dict
    from pymc.sampling.jax import get_jaxified_graph
    from pymc.util import get_default_varnames

    z, cat = _load_draws(args.outdir)
    outdir = Path(args.outdir.split(",")[0])
    cfg = json.loads((outdir / "config.json").read_text())
    f = _fitter()
    model = f.build_model()
    outs = list(get_default_varnames(model.unobserved_value_vars, include_transformed=False))
    fn = get_jaxified_graph(inputs=model.value_vars, outputs=outs)
    res = jax.vmap(jax.vmap(fn))(*[jax.numpy.asarray(v) for v in z])
    post = {v.name: np.asarray(r) for v, r in zip(outs, res, strict=True)}
    stats = {
        "diverging": cat["diverging"],
        "n_steps": cat["num_steps"],
        "tree_depth": np.log2(cat["num_steps"]).astype(int) + 1,
        "step_size": cat["step_size"],
        "acceptance_rate": cat["accept_prob"],
        "lp": cat["potential_energy"],
        "energy": cat["energy"],
    }
    idata = from_dict({"posterior": post, "sample_stats": stats})
    from isochrone_unbinned_recovery import OUT_DIR, _strict, _summ

    found = json.loads(Path(args.search).read_text())
    res = {
        "batch": "ngc",
        "N": int(f._N_obs),
        "search": {k: found[k] for k in ("mode", "loglike", "runner_up", "local_sd")},
        "driver": "isochrone_c1_chunked.py",
        "config": {k: cfg[k] for k in cfg if k != "prep"},
    }
    res.update(_summ(idata, None))
    prog = [json.loads(line) for line in (outdir / "progress.jsonl").read_text().splitlines()]
    res["seconds_sampling"] = round(sum(r.get("chunk_s", 0) for r in prog))
    res["search_seconds"] = found.get("t_search_s")
    res.update(_head())
    target = Path(args.json) if args.json else OUT_DIR / "ngc_0.json"
    target.write_text(json.dumps(_strict(res), indent=1) + "\n")
    print(json.dumps(_strict(res), indent=1)[:3000])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "stage",
        choices=["search", "gradbench", "float32", "run", "equivcheck", "diagsummary", "finalize"],
    )
    ap.add_argument("--search", default=str(CACHE / "search_seed42.json"))
    ap.add_argument("--outdir", default=str(CACHE / "c1"))
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--mass", default="seeded")
    ap.add_argument("--chain-method", default="parallel", choices=["parallel", "vectorized"])
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--stop-at", type=int, default=0, help="stop after this iteration (probe)")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--chain-index", type=int, default=-1, help="run only this chain of --chains")
    ap.add_argument("--mutate", action="store_true", help="equivcheck: driver seed moved by one")
    ap.add_argument("--out", default="")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    if a.stage == "search":
        stage_search(Path(a.search))
    elif a.stage == "gradbench":
        stage_gradbench(a)
    elif a.stage == "float32":
        stage_float32(a)
    elif a.stage == "run":
        stage_run(a)
    elif a.stage == "equivcheck":
        stage_equivcheck(a)
    elif a.stage == "diagsummary":
        stage_diagsummary(a)
    else:
        stage_finalize(a)
