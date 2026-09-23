#!/usr/bin/env python3
r"""Does the unbinned isochrone likelihood (2026-09-22/23) pass the criterion the Hess grid failed?

WHAT QUESTION THIS SETTLES
--------------------------
``isochrone_nuts_convergence.py`` measured that the precomputed-Hess likelihood depended on the
grid's internal reference and failed injection-recovery (dm outside the 90 % interval in 16/16).
The likelihood was rewritten as an unbinned per-star mixture over EEP-interpolated isochrones.
This script runs the success criterion pre-registered in the hub finding
``isochrone-nuts-convergence-2026-09.md`` §9.5, as amended in §9.6 *before* any run:

* ``A``   -- the exact failing configuration: 10 binaries (truth Z 0.014286, log t 6.55,
             dm 10.45, A_V 1.24, seeds 1-10) + 6 singles (dm 10.47, A_V 1.10, seeds 1-6).
             C3: truth inside the 90 % interval in >= 12 of 16 per parameter; a run failing the
             gate (R-hat >= 1.01, ESS_bulk <= 400, any divergence) counts as a miss everywhere.
* ``B``   -- diagnostic: 16 singles with truths drawn from the prior (seeds 101-116).
* ``loo`` -- C4: 6 singles (A's singles truth, seeds 1-6) fitted with the Z = 0.014286 file
             removed; pass if the pooled posterior fraction within 3e-4 of a remaining node < 5 %.
* ``refinv`` -- C2 on NGC 6383: log L at fixed parameters with the prior centre moved (the old
             grid's reference) by a full and by a half bin; must be exactly unchanged.
* ``ngc`` -- C1: NGC 6383, 4 chains x 2000 after 2000 tune, search start.
* ``ngc_prior`` -- 4 chains x 500 after 1500, started from PyMC's jittered prior centre: where
                   chains end up without the search.
* ``ngc_fine``  -- 2 chains x 1000 after 1500 with the representation refined (15 q nodes,
                   stride 1): a robustness check, ~3x the cost per gradient.

The generator is ``synthetic_cluster`` of ``isochrone_nuts_convergence.py``: star-level draws
from a MIST node with its own IMF inverse CDF, per-star D&K q draws and Gaussian errors, written
independently of the likelihood.

WHAT WOULD FALSIFY THE CONCLUSION
---------------------------------
C1-C4 not all met. Nothing here is tuned on these runs: the representation and the 0.01 mag width
floor were fixed from the D1 measurements and the approximation-error budget
(``isochrone_likelihood_d1/``) and committed to the finding before the first run.

USAGE
-----
    PYTHONPATH=~/erotica python tools/validation/isochrone_unbinned_recovery.py run A 0
    PYTHONPATH=~/erotica python tools/validation/isochrone_unbinned_recovery.py summarize

``run <batch> <i>`` writes ``isochrone_unbinned/<batch>_<i>.json``; ``summarize`` merges them
into ``isochrone_unbinned_recovery.json`` with the C1-C4 verdicts. Needs the ``bayes`` extra and
numpyro. Set ``XLA_FLAGS=--xla_force_host_platform_device_count=<chains>`` to run chains in
parallel. Data paths are this machine's (repo landmine: hardcoded paths).
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from astropy.table import QTable, Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from isochrone_nuts_convergence import MIST, PRIORS, SAMPLE, synthetic_cluster  # noqa: E402

from erotica.analysis import _isochrone as iso_mod  # noqa: E402
from erotica.analysis._isochrone import IsochroneFitter  # noqa: E402

OUT_DIR = HERE / "isochrone_unbinned"
SUMMARY = HERE / "isochrone_unbinned_recovery.json"
Z_NODES = np.array([2.540e-3, 4.518e-3, 8.033e-3, 1.4286e-2, 2.5404e-2, 4.5175e-2])
PARAMS = ["met", "loga", "dm", "Av", "sigma_int", "f_bg"]
PRIORS = {k: v for k, v in PRIORS.items() if k not in ("M_met", "M_loga")}


def _truths(batch: str) -> list[dict]:
    if batch == "A":
        return [
            dict(Z=0.014286, loga=6.55, dm=10.45, Av=1.24, seed=s, binaries=True)
            for s in range(1, 11)
        ] + [
            dict(Z=0.014286, loga=6.55, dm=10.47, Av=1.10, seed=s, binaries=False)
            for s in range(1, 7)
        ]
    if batch == "B":
        rng = np.random.default_rng(20260923)
        out = []
        ages = np.round(np.arange(6.05, 6.951, 0.05), 2)
        for s in range(101, 117):
            while True:
                dm = rng.normal(10.3, 0.2)
                if 9.5 <= dm <= 10.7:
                    break
            out.append(
                dict(
                    Z=float(rng.choice(Z_NODES[:5])),
                    loga=float(rng.choice(ages)),
                    dm=float(dm),
                    Av=float(rng.uniform(0.5, 2.0)),
                    seed=s,
                    binaries=False,
                )
            )
        return out
    if batch == "loo":
        return [
            dict(Z=0.014286, loga=6.55, dm=10.47, Av=1.10, seed=s, binaries=False)
            for s in range(1, 7)
        ]
    raise ValueError(batch)


def _mist_without(z_keep_out: float) -> Path:
    """A directory with every MIST file except the one whose Zinit is ``z_keep_out``."""
    probe = iso_mod.MISTIsochrones(MIST)
    tmp = Path(tempfile.mkdtemp())
    n = 0
    for f in sorted(MIST.glob("*.iso.cmd")):
        with open(f, encoding="utf-8", errors="ignore") as fh:
            head = [next(fh) for _ in range(8)]
        z = probe._z_from_header_lines([h.strip().lstrip("#").strip() for h in head])
        if z is not None and abs(round(z, 6) - z_keep_out) < 1e-7:
            continue
        (tmp / f.name).symlink_to(f)
        n += 1
    assert n == len(list(MIST.glob("*.iso.cmd"))) - 1, "the held-out file was not found"
    return tmp


def _summ(idata, truth: dict | None) -> dict:
    import arviz as az

    post, ss = idata.posterior, idata.sample_stats
    rh, eb = az.rhat(post), az.ess(post, method="bulk")
    res = {
        "divergences": int(ss["diverging"].values.sum()),
        "tree_depth_chain_mean": np.round(ss["tree_depth"].values.mean(1), 2).tolist(),
        "step_size_chain_mean": np.round(ss["step_size"].values.mean(1), 5).tolist(),
    }
    E = ss["energy"].values
    res["bfmi"] = np.round(np.mean(np.diff(E, axis=1) ** 2, axis=1) / np.var(E, axis=1), 3).tolist()
    for p in PARAMS:
        a = post[p].values
        q5, q16, q50, q84, q95 = np.percentile(a.ravel(), [5, 16, 50, 84, 95])
        row = {
            "rhat": round(float(rh[p]), 4),
            "ess_bulk": round(float(eb[p]), 0),
            "chain_means": np.round(a.mean(1), 5).tolist(),
            "q05_16_50_84_95": [round(float(x), 5) for x in (q5, q16, q50, q84, q95)],
        }
        if truth is not None and p in truth:
            row["truth"] = truth[p]
            row["in90"] = bool(q5 <= truth[p] <= q95)
        res[p] = row
    res["gate"] = bool(
        res["divergences"] == 0
        and all(res[p]["rhat"] < 1.01 and res[p]["ess_bulk"] > 400 for p in PARAMS)
    )
    return res


def run(batch: str, i: int) -> dict:
    OUT_DIR.mkdir(exist_ok=True)
    t0 = time.time()
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
    if batch.startswith("ngc"):
        data = QTable(Table.read(SAMPLE))
        f = IsochroneFitter(isochs_path=MIST, **PRIORS)
        if batch == "ngc_fine":
            iso_mod._Q_NODES = np.array(
                [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.87, 0.93, 0.97, 1.0]
            )
            f.BINARY_STRIDE = 1
        f.setup(data, prob_threshold=0.0)
        start = "prior" if batch == "ngc_prior" else "search"
        n_ch = 2 if batch == "ngc_fine" else 4
        found = (
            f.find_start(n_ch, np.random.default_rng(42), mode="JAX") if start == "search" else None
        )
        # C1 is the 4 x 2000 certificate; ngc_prior only reports where jittered chains end up,
        # and ngc_fine is a robustness check at ~3x the cost per gradient
        draws, tune, chains = {"ngc": (2000, 2000, 4), "ngc_prior": (500, 1500, 4)}.get(
            batch, (1000, 1500, 2)
        )
        idata = f.fit(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            nuts_sampler="numpyro",
            random_seed=42,
            progressbar=False,
            initvals=found["chain_starts"] if found else None,
            start=start,
        )
        res = {
            "batch": batch,
            "N": int(f._N_obs),
            "search": found and {k: found[k] for k in ("mode", "loglike", "runner_up", "local_sd")},
        }
        res.update(_summ(idata, None))
    else:
        tr = _truths(batch)[i]
        data = synthetic_cluster(
            tr["Z"], tr["loga"], tr["dm"], tr["Av"], tr["seed"], binaries=tr["binaries"]
        )
        kw = {} if tr["binaries"] else {"alpha": 0.0, "beta": 0.0}
        path = _mist_without(0.014286) if batch == "loo" else MIST
        f = IsochroneFitter(isochs_path=path, **{**PRIORS, **kw})
        f.setup(data, prob_threshold=0.0)
        found = f.find_start(2, np.random.default_rng(tr["seed"]), mode="JAX")
        idata = f.fit(
            draws=1000,
            tune=1500,
            chains=2,
            target_accept=0.9,
            nuts_sampler="numpyro",
            random_seed=tr["seed"],
            progressbar=False,
            initvals=found["chain_starts"],
        )
        truth = {"met": tr["Z"], "loga": tr["loga"], "dm": tr["dm"], "Av": tr["Av"]}
        # Diagnostic only (the chains start from the search, never from the truth): polish from
        # the truth too, so a search that missed the dominant mode shows up here, not in coverage.
        from scipy.optimize import minimize

        f6 = f._compiled_loglike("JAX")
        met_lo = 10 ** float(f._node_logz[0]) * (1 + 1e-9)
        met_hi = 10 ** float(f._node_logz[-1]) * (1 - 1e-9)
        bounds = [(met_lo, met_hi), f.loga_range, f.dm_range, f.Av_range, (1e-4, 0.5), (1e-4, 0.5)]
        y0 = np.clip([tr["Z"], tr["loga"], tr["dm"], tr["Av"], 0.01, 0.01], *np.array(bounds).T)
        r = minimize(
            lambda y: tuple(-np.asarray(v) for v in f6(y)),
            y0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
        )
        truth_polish = {"loglike": -float(r.fun), "x": [float(v) for v in r.x]}
        res = {
            "batch": batch,
            "index": i,
            "truth_cfg": tr,
            "search": {k: found[k] for k in ("mode", "loglike", "runner_up", "local_sd")},
            "truth_polish": truth_polish,
            "search_missed_mode": bool(truth_polish["loglike"] > found["loglike"] + 1.0),
        }
        res.update(_summ(idata, truth))
        if batch == "loo":
            met = idata.posterior["met"].values.ravel()
            nodes = np.array([n for n in Z_NODES if abs(n - 0.014286) > 1e-6])
            res["met_samples_thinned"] = np.round(met[::10], 6).tolist()
            res["frac_within_3e-4_of_remaining_node"] = float(
                np.mean(np.min(np.abs(met[:, None] - nodes[None]), axis=1) < 3e-4)
            )
    res.update({"seconds": round(time.time() - t0), "head": head, "erotica_dirty": dirty})
    (OUT_DIR / f"{batch}_{i}.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    return res


def refinv() -> dict:
    """The test that killed the grid likelihood, on NGC 6383: move the prior centre (the old grid
    reference) and evaluate log L at the three points of the old ``refinv`` stage. The old code
    changed by -10.45 / -10.59 / +4.49 under a half-bin move; this must be exactly zero."""
    data = QTable(Table.read(SAMPLE))
    pts = [(0.00536, 6.292, 10.01, 0.553), (0.0143, 6.55, 10.3, 1.24), (0.0143, 6.55, 10.1, 1.0)]
    out = {"points": pts}
    for name, kw in (
        ("default", {}),
        ("moved", {"dm_mu": 10.6, "Av_range": (0.6, 2.1)}),
        ("moved_half_bin", {"dm_mu": 10.3 + 0.349, "Av_range": (0.5 + 0.134, 2.0 + 0.134)}),
    ):
        f = IsochroneFitter(isochs_path=MIST, **{**PRIORS, **kw})
        f.setup(data, prob_threshold=0.0)
        out[name] = [f.loglike(*p, 0.02, 0.05) for p in pts]
    out["max_abs_diff_moved"] = max(
        abs(a - b) for a, b in zip(out["default"], out["moved"], strict=True)
    )
    out["max_abs_diff_half_bin"] = max(
        abs(a - b) for a, b in zip(out["default"], out["moved_half_bin"], strict=True)
    )
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "refinv.json").write_text(json.dumps(out, indent=1) + "\n")
    return out


def summarize() -> dict:
    out = {}
    for batch in ("A", "B", "loo"):
        runs = [json.loads(p.read_text()) for p in sorted(OUT_DIR.glob(f"{batch}_*.json"))]
        if not runs:
            continue
        s = {
            "n_runs": len(runs),
            "gate_pass": sum(r["gate"] for r in runs),
            "search_missed_mode": sum(r.get("search_missed_mode", False) for r in runs),
        }
        for p in ("met", "loga", "dm", "Av"):
            # a run failing the gate is a miss on every parameter (pre-registered, §9.5)
            s[f"{p}_in90"] = sum(r[p]["in90"] and r["gate"] for r in runs)
        if batch == "A":
            s["C3_pass"] = bool(
                len(runs) == 16 and all(s[f"{p}_in90"] >= 12 for p in ("met", "loga", "dm", "Av"))
            )
        if batch == "loo":
            met = np.concatenate([np.asarray(r["met_samples_thinned"]) for r in runs])
            nodes = np.array([n for n in Z_NODES if abs(n - 0.014286) > 1e-6])
            frac = float(np.mean(np.min(np.abs(met[:, None] - nodes[None]), axis=1) < 3e-4))
            s["pooled_frac_within_3e-4_of_remaining_node"] = round(frac, 4)
            s["met_medians"] = [r["met"]["q05_16_50_84_95"][2] for r in runs]
            s["C4_pass"] = bool(frac < 0.05)
        out[batch] = s
    for batch in ("ngc", "ngc_prior", "ngc_fine"):
        p = OUT_DIR / f"{batch}_0.json"
        if p.exists():
            r = json.loads(p.read_text())
            out[batch] = {
                "gate": r["gate"],
                "worst_rhat": max(r[q]["rhat"] for q in PARAMS),
                "worst_ess_bulk": min(r[q]["ess_bulk"] for q in PARAMS),
                "divergences": r["divergences"],
                "medians": {q: r[q]["q05_16_50_84_95"][2] for q in PARAMS},
                "chain_means": {q: r[q]["chain_means"] for q in PARAMS},
            }
    if (OUT_DIR / "refinv.json").exists():
        r = json.loads((OUT_DIR / "refinv.json").read_text())
        out["refinv"] = {k: r[k] for k in ("max_abs_diff_moved", "max_abs_diff_half_bin")}
    SUMMARY.write_text(json.dumps(out, indent=1) + "\n")
    return out


if __name__ == "__main__":
    if sys.argv[1] == "refinv":
        print(json.dumps(refinv(), indent=1))
    elif sys.argv[1] == "run":
        print(json.dumps(run(sys.argv[2], int(sys.argv[3])), default=float)[:2000])
    else:
        print(json.dumps(summarize(), indent=1))
