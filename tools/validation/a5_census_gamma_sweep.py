#!/usr/bin/env python3
r"""A5 -- the ``gamma ~ 2`` identifiability sweep across the Gaia open-cluster census.

WHAT THIS ANSWERS
-----------------
Fit the EFF profile ``Sigma(r) ~ (1 + (r/a)^2)^(-gamma/2)`` to every Hunt & Reffert (2024)
``Type='o'`` open cluster and ask whether ``gamma`` piles up near 2 -- the value at which EFF and
an untruncated King profile coincide, so that a pile-up there would be a statement about how
Galactic open clusters are actually shaped.

THE BACKGROUND IS PINNED, AND THAT IS THE CENTRAL DESIGN DECISION
-----------------------------------------------------------------
``EFFPriors(b_scale=1e-6)``. A **free** flat background invents one where the truth is zero and
biases ``gamma`` upward, with a size that grows as ``N`` falls -- i.e. with an ``N``-dependence
that would read as physics across a census spanning four decades in ``N``. Measured in
``ellipticity_bias.py`` on a circular null at ``N = 15000`` (true ``b`` exactly 0): free ``b``
returns 0.0072 and ``+0.0116 +/- 0.0020`` on ``gamma`` (5.8 sigma); pinned ``b`` gives
``-0.0038 +/- 0.0034``, consistent with zero.

The effect is far larger at census geometry. Differencing ``eff_gamma_bias_lowN_ratios.json``
(free) against ``eff_gamma_bias_lowN_ratios_pinnedbg.json`` (pinned), at ``N = 60``,
``gamma_true = 2``:

    r_tot/a      free      pinned
       2       +1.653    +1.418 +/- 0.152
       4       +1.842    +1.152 +/- 0.175
       8       +1.566    +0.460 +/- 0.172
       16      +0.987    +0.170 +/- 0.122
       42      +0.417    +0.079 +/- 0.088

So the free-background term supplies **most** of the bias at wide footprints and a minority of it
at narrow ones. Using the free-background surface as a correction would inject a spurious
``N``-dependent, footprint-dependent artefact into every census ``gamma``. It is not used.

WHAT SURVIVES PINNING IS A LEVER ARM, AND IT IS THE GATE
--------------------------------------------------------
``gamma`` is a log--log slope, so measuring it needs dynamic range in ``log r``. At ``r_tot/a = 2``
there is 0.3 dex of it; at 42 there is 1.6 dex. The pinned column above is that effect alone, and
it is consistent with zero only from ``r_tot/a ~ 16`` upward. Clusters below that do not have a
measured ``gamma`` -- they have a prior plus a lever-arm bias -- and this script records
``rtot_over_a``, computed from **the fitted** ``a`` rather than from the catalogue's empirical
``rc``, so that gate can be applied downstream. (The two are not interchangeable: on NGC_6383,
``r_tot/rc = 13.6`` but ``r_tot/a_fit = 16.0``.)

GEOMETRY, AND THE UNIT TRAP
---------------------------
Hunt & Reffert's radii (``rc``, ``rt``, ``rtot``, ``r50``, ``rJ``) are in **degrees**. The EFF
priors inherited here (``a_scale = 5.0``) were calibrated in **arcmin** against ``a ~ 1.65'``.
Everything is therefore converted to arcmin before fitting; passing degrees would silently run an
uncalibrated prior and the convergence diagnostics would look perfect.

Member radii are great-circle separations (``SkyCoord.separation``) from HR24's own catalogue
centre -- not ``dRA*cos(dec)``, which is wrong at high declination. Verified on NGC_6383: maximum
member separation 33.738' against a catalogue ``rtot`` of 33.739', i.e. **HR24's published member
list is already truncated at** ``rtot``, so ``field_radius = rtot`` matches the sample passed in.
That match is what makes the point-process normalisation correct; a mismatch invalidates every
``gamma``.

DIAGNOSTICS ARE READ FROM az.rhat / az.ess, NEVER FROM az.summary
-----------------------------------------------------------------
``az.summary`` rounds, and rounding has previously caused converged cells in this project to be
misreported as failures. The recorded r-hat is the **maximum over all fitted parameters** and the
ESS the **minimum**, because ``a`` and ``gamma`` are strongly correlated (+0.90 measured on
NGC_6383) and a fit that converged on ``gamma`` alone is not a converged fit.

CHECKPOINTING -- ONE FILE, ONE WRITER
-------------------------------------
Workers compute and return; the **parent alone** appends to ``<out>.cells.jsonl``, one JSON object
per line, flushed and fsynced. Resume reads the file and skips any cluster ``Name`` already
present. A prior sweep in this project lost 16 cells to a dual-writer bug in which completed runs
were written without the resume keys; workers here never touch the sidecar.

USAGE
-----
    python tools/validation/fetch_hr24.py           # cluster table  (7167 rows)
    python tools/validation/fetch_hr24_members.py   # member table   (1291929 rows)
    python tools/validation/a5_census_gamma_sweep.py --workers 4
    python tools/validation/a5_census_gamma_sweep.py --workers 4 --with-king
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CLUSTERS = HERE / "hr24_clusters.ecsv"
MEMBERS = HERE / "hr24_members.parquet"
DEFAULT_OUT = HERE / "a5_census_gamma_sweep.json"

# One EFF fit is ~3.2 s warm (chains=4, draws=1500, tune=1000, numpyro) and flat in N over the
# census range, so the per-cluster cost is set by sampler startup, not by sample size.
DRAWS, TUNE, CHAINS = 1500, 1000, 4

_STATE: dict = {}


def _init_worker(with_king: bool, prob_min: float, free_background: bool = False):
    """Import the heavy stack once per worker, not once per cluster.

    Threads are pinned to 1 per worker BEFORE jax/XLA is imported. Measured on this 8-core
    machine: 4 unpinned workers each spawn a full-width XLA thread pool, so 4 workers contend for
    ~32 threads on 8 cores and a fit that takes 3.2 s standalone takes 73 s. Pinning trades
    intra-fit parallelism -- which this likelihood barely uses -- for clean process-level
    parallelism, which it uses perfectly.
    """
    warnings.filterwarnings("ignore")
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = "1"
    os.environ["XLA_FLAGS"] = (
        "--xla_force_host_platform_device_count=1 "
        "--xla_cpu_multi_thread_eigen=false "
        "intra_op_parallelism_threads=1"
    )
    from erotica.analysis.inference import SamplingConfig
    from erotica.analysis.structure import EFFPriors, KingPriors, eff_unbinned, king_unbinned

    _STATE["cfg"] = SamplingConfig(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        random_seed=5,
        progressbar=False,
        nuts_sampler="numpyro",
    )
    # b_scale=1e-6 pins the flat background at ~0. See the module docstring: this is the whole
    # point of the run, not a tuning knob.
    _STATE["eff_priors"] = EFFPriors(b_scale=1.0 if free_background else 1e-6)
    _STATE["king_priors"] = KingPriors(b_scale=1e-6) if with_king else None
    _STATE["eff"] = eff_unbinned
    _STATE["king"] = king_unbinned
    _STATE["with_king"] = with_king
    _STATE["prob_min"] = prob_min


def _diagnostics(idata):
    """Max r-hat and min ESS across every fitted parameter, plus the divergence count.

    Read from ``az.rhat`` / ``az.ess`` directly. ``az.summary`` rounds to 2-3 significant figures,
    which turns an r-hat of 1.0049 into "1.00" and an r-hat of 1.0149 into "1.01" -- the gate
    boundary. Never gate on a rounded number.
    """
    import arviz as az

    names = list(idata.posterior.data_vars)
    rhat, ess_b, ess_t = az.rhat(idata), az.ess(idata, method="bulk"), az.ess(idata, method="tail")
    return dict(
        rhat_max=float(max(float(np.nanmax(rhat[n].values)) for n in names)),
        ess_bulk_min=float(min(float(np.nanmin(ess_b[n].values)) for n in names)),
        ess_tail_min=float(min(float(np.nanmin(ess_t[n].values)) for n in names)),
        divergences=int(idata.sample_stats["diverging"].values.sum()),
        rhat_by_param={n: float(np.nanmax(rhat[n].values)) for n in names},
    )


def fit_one(task):
    """Fit one cluster. Runs in a worker; returns a plain dict and writes nothing."""
    meta, ra, dec, prob = task
    t0 = time.time()
    out = dict(meta)
    try:
        import astropy.units as u
        from astropy.coordinates import SkyCoord

        keep = prob >= _STATE["prob_min"]
        ctr = SkyCoord(meta["ra_deg"] * u.deg, meta["dec_deg"] * u.deg)
        stars = SkyCoord(ra[keep] * u.deg, dec[keep] * u.deg)
        r = ctr.separation(stars).to(u.arcmin).value

        field = meta["rtot_arcmin"]
        r = r[(r <= field) & np.isfinite(r)]
        out["n_fitted"] = int(r.size)
        if r.size < 10 or not np.isfinite(field) or field <= 0:
            out["status"] = "skipped_too_few"
            out["wall_s"] = time.time() - t0
            return out

        fit = _STATE["eff"](
            r,
            field_radius=field,
            priors=_STATE["eff_priors"],
            sampling=_STATE["cfg"],
            progressbar=False,
        )
        idata = fit["eff_trace"]
        post = idata.posterior
        g = np.asarray(post["gamma"].values).ravel()
        a = np.asarray(post["a"].values).ravel()

        out.update(
            gamma_median=float(np.median(g)),
            gamma_mean=float(g.mean()),
            gamma_sd=float(g.std(ddof=1)),
            gamma_q16=float(np.percentile(g, 16)),
            gamma_q84=float(np.percentile(g, 84)),
            gamma_q025=float(np.percentile(g, 2.5)),
            gamma_q975=float(np.percentile(g, 97.5)),
            a_median=float(np.median(a)),
            a_sd=float(a.std(ddof=1)),
            a_q16=float(np.percentile(a, 16)),
            a_q84=float(np.percentile(a, 84)),
            k_median=float(fit["k_median"]),
            b_median=float(fit["b_median"]),
            corr_a_gamma=float(np.corrcoef(a, g)[0, 1]),
            # The recoverability axis. Computed from the FITTED a, because the catalogue's rc is a
            # different quantity: on NGC_6383, rtot/rc = 13.6 while rtot/a_fit = 16.0.
            rtot_over_a=float(field / np.median(a)),
            rtot_over_a_q16=float(field / np.percentile(a, 84)),
            rtot_over_a_q84=float(field / np.percentile(a, 16)),
            **{f"eff_{k}": v for k, v in _diagnostics(idata).items()},
        )
        out["status"] = "ok"

        if _STATE["with_king"]:
            try:
                kfit = _STATE["king"](
                    r,
                    field_radius=field,
                    priors=_STATE["king_priors"],
                    sampling=_STATE["cfg"],
                    progressbar=False,
                )
                kidata = kfit["king_trace"]
                kpost = kidata.posterior
                rt = np.asarray(kpost["r_t"].values).ravel()
                rc = np.asarray(kpost["r_c"].values).ravel()
                out.update(
                    king_rc_median=float(np.median(rc)),
                    king_rc_sd=float(rc.std(ddof=1)),
                    king_rt_median=float(np.median(rt)),
                    king_rt_sd=float(rt.std(ddof=1)),
                    king_rt_q16=float(np.percentile(rt, 16)),
                    king_rt_q84=float(np.percentile(rt, 84)),
                    **{f"king_{k}": v for k, v in _diagnostics(kidata).items()},
                )
                out["king_status"] = "ok"
            except Exception as exc:  # noqa: BLE001 -- one bad cluster must not kill the sweep
                out["king_status"] = f"error: {type(exc).__name__}: {exc}"
    except Exception as exc:  # noqa: BLE001
        out["status"] = f"error: {type(exc).__name__}: {exc}"
    out["wall_s"] = time.time() - t0
    return out


def build_tasks(args):
    import pandas as pd
    from astropy.table import Table

    cl = Table.read(CLUSTERS)
    typ = np.asarray(cl["Type"], dtype=str)
    cl = cl[typ == args.type]
    print(f"[catalogue] {CLUSTERS.name}: {len(cl)} rows with Type={args.type!r}")

    logage = np.asarray(cl["logAge50"], dtype=float)
    young = logage < np.log10(args.young_myr * 1e6)
    print(
        f"[catalogue] {int(young.sum())} below {args.young_myr} Myr (logAge50 < "
        f"{np.log10(args.young_myr * 1e6):.4f})"
    )

    if args.young_only:
        cl = cl[young]
    if args.limit:
        cl = cl[: args.limit]

    mem = pd.read_parquet(MEMBERS)
    mem["Name"] = mem["Name"].astype(str)
    groups = {n: g for n, g in mem.groupby("Name", sort=False)}
    print(f"[members]   {MEMBERS.name}: {len(mem)} rows, {len(groups)} distinct clusters")

    tasks, missing = [], 0
    for row in cl:
        name = str(row["Name"])
        g = groups.get(name)
        if g is None:
            missing += 1
            continue
        meta = dict(
            name=name,
            n_catalogue=int(row["N"]),
            ra_deg=float(row["RA_ICRS"]),
            dec_deg=float(row["DE_ICRS"]),
            # degrees -> arcmin. The priors are calibrated in arcmin; see the module docstring.
            rtot_arcmin=float(row["rtot"]) * 60.0,
            rc_arcmin=float(row["rc"]) * 60.0,
            rt_arcmin=float(row["rt"]) * 60.0,
            r50_arcmin=float(row["r50"]) * 60.0,
            rJ_arcmin=float(row["rJ"]) * 60.0,
            rtot_over_rc=float(row["rtot"]) / float(row["rc"]) if row["rc"] else float("nan"),
            rtot_over_r50=float(row["rtot"]) / float(row["r50"]) if row["r50"] else float("nan"),
            logAge50=float(row["logAge50"]),
            dist50_pc=float(row["dist50"]),
            CST=float(row["CST"]),
            MassTot=float(row["MassTot"]),
        )
        tasks.append(
            (
                meta,
                g["RA_ICRS"].to_numpy(dtype=float),
                g["DE_ICRS"].to_numpy(dtype=float),
                g["Prob"].to_numpy(dtype=float),
            )
        )
    if missing:
        print(f"[members]   WARNING: {missing} catalogue clusters absent from the member table")
    return tasks


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument(
        "--limit", type=int, default=0, help="fit only the first N clusters (smoke test)"
    )
    ap.add_argument(
        "--type",
        default="o",
        help="HR24 Type; 'o' = open cluster. Moving groups "
        "('m') and globulars ('g') must stay separate.",
    )
    ap.add_argument("--young-myr", type=float, default=10.0)
    ap.add_argument("--young-only", action="store_true")
    ap.add_argument(
        "--prob-min",
        type=float,
        default=0.0,
        help="membership-probability cut. Default 0: HR24's published member list is "
        "already the selected sample (verified: NGC_6383 has N=322 in both).",
    )
    ap.add_argument(
        "--free-background",
        action="store_true",
        help="UNPIN the background. Not for producing census gammas -- it injects the "
        "artefact the pinning exists to remove. Its one legitimate use is the "
        "paired comparison: refit the same clusters both ways and report the "
        "difference, which is what king_model_validity.md's 'fit both ways' rule "
        "asks for and what measures the corona-vs-shape split on REAL data.",
    )
    ap.add_argument(
        "--with-king",
        action="store_true",
        help="also fit King. Roughly doubles the cost and R_t is often unconstrained.",
    )
    args = ap.parse_args()

    tasks = build_tasks(args)

    sidecar = args.out.with_suffix(".cells.jsonl")
    done = set()
    if sidecar.exists():
        with sidecar.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["name"])
                except (json.JSONDecodeError, KeyError):
                    continue  # a torn final line from a crash; it will simply be refitted
        print(f"[resume]    {len(done)} clusters already in {sidecar.name}; skipping them")

    todo = [t for t in tasks if t[0]["name"] not in done]
    print(
        f"[sweep]     {len(todo)} to fit, {args.workers} workers, "
        f"EFF background {'FREE (b_scale=1.0)' if args.free_background else 'PINNED (b_scale=1e-6)'}"
        f"{', King too' if args.with_king else ''}"
    )
    if not todo:
        print("[sweep]     nothing to do")
        return

    import multiprocessing as mp

    t0, n_ok = time.time(), 0
    ctx = mp.get_context("spawn")
    # THE SINGLE WRITER. Workers return dicts; only this loop touches the sidecar.
    with (
        sidecar.open("a") as fh,
        ctx.Pool(
            args.workers,
            initializer=_init_worker,
            initargs=(args.with_king, args.prob_min, args.free_background),
        ) as pool,
    ):
        for i, res in enumerate(pool.imap_unordered(fit_one, todo, chunksize=1), 1):
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            n_ok += res.get("status") == "ok"
            if i % 25 == 0 or i == len(todo):
                el = time.time() - t0
                print(
                    f"  [{i:>5d}/{len(todo)}] ok={n_ok} {el / i:.2f}s/cluster "
                    f"eta={(len(todo) - i) * el / i / 60:.1f} min",
                    flush=True,
                )

    rows = [json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()]
    args.out.write_text(
        json.dumps(
            dict(
                catalogue=str(CLUSTERS.name),
                members=str(MEMBERS.name),
                type=args.type,
                background=(
                    "free (EFFPriors(b_scale=1.0))"
                    if args.free_background
                    else "pinned (EFFPriors(b_scale=1e-6))"
                ),
                draws=DRAWS,
                tune=TUNE,
                chains=CHAINS,
                sampler="numpyro",
                prob_min=args.prob_min,
                with_king=args.with_king,
                n_clusters=len(rows),
                cells=rows,
            ),
            indent=1,
        )
    )
    print(f"\nwrote {args.out}  ({len(rows)} clusters)")


if __name__ == "__main__":
    sys.exit(main())
