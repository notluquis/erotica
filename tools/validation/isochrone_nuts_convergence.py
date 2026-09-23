#!/usr/bin/env python3
r"""Why the isochrone NUTS fit of NGC 6383 did not converge, and what is still wrong after the fix.

WHAT QUESTION THIS SETTLES
--------------------------
The only 4 x 2000 NUTS run of ``IsochroneFitter`` on NGC 6383 (2026-06-11) gave R-hat 1.5-2.2 with
zero divergences. ``docs/design-notes/isochrone_sampler_fix.md`` (2026-07-21) blamed a staircase in
the precomputed Hess grid. This script measures, on the same data and priors:

1. ``frame``   -- where the shifted model Hess puts its mass, row by row, against the observed Hess,
                  with the pre-2026-09-22 code and with the current one.
2. ``gradient``-- identical adjacent grid slices and exact-zero d loglike / d(met, loga) at random
                  prior points, for both grids.
3. ``ablation``-- NUTS (numpyro) with frame {old, new} x grid {nearest node, node interpolation}, plus
                  the old model with every chain started at one point (multimodality check), plus the
                  current model with 4 chains.
4. ``refinv``  -- reference invariance: the same model with the grid's internal reference moved half a
                  bin. A correct likelihood cannot depend on it.
5. ``recovery``-- injection-recovery on star-level synthetic clusters drawn from MIST by a generator
                  written independently of ``_hess_for_isochrone`` / ``posterior_cmd``.

WHAT WOULD FALSIFY THE CONCLUSIONS
----------------------------------
* The frame bug is the dominant cause: falsified if the (old frame, node interpolation) variant
  converges, or the (new frame, nearest node) variant does not.
* Multimodality is not the cause: falsified if the one-start run of the old model converges.
* The current likelihood is still biased: falsified if log L is invariant (|dlogL| < 0.5) under the
  half-bin reference move, or if injected dm is inside the 90 % interval in most recovery runs.

Needs the ``bayes`` extra plus ``numpyro`` (the ``erotica-bench`` env has no pymc; this was run in the
``cosmic`` env with ``PYTHONPATH`` pointing at the repo). ``blackjax`` is not used: it is broken on
pymc 6 (see ``pyproject.toml``). Data paths are this machine's (repo landmine: hardcoded paths).

USAGE
-----
    PYTHONPATH=~/erotica python tools/validation/isochrone_nuts_convergence.py [stage ...]

Writes ``tools/validation/isochrone_nuts_convergence.json`` (merged per stage). Runtime ~1 h, one
process; RAM is small (the model has 6 parameters and ~200 bins).
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from astropy.table import QTable, Table

from erotica.analysis._isochrone import MISTIsochrones, _ccm89

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "isochrone_nuts_convergence.json"
B = Path("/Users/notluquis/erotica/data/test/NGC6383")
MIST = B / "MIST" / "UBVRIplus"
SAMPLE = B / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
PRE_FIX_COMMIT = "f4ce09a"  # dev head before the 2026-09-22 fix
# The five stages below measure the *precomputed-Hess* likelihood as fixed on 2026-09-22. It was
# replaced the same day by the unbinned one, so they import the module at that commit, exactly as
# they import the pre-fix module at PRE_FIX_COMMIT.
GRID_COMMIT = "507f779"
PRIORS = dict(
    loga_range=(6.0, 7.0),
    Av_range=(0.5, 2.0),
    dm_mu=10.3,
    dm_sigma=0.2,
    dm_range=(9.5, 10.7),
    M_met=200,
    M_loga=200,
)
PARAMS = ["met", "loga", "dm", "Av", "log_s", "bg"]


# ---------------------------------------------------------------------------- variants
def _module_at(commit, name):
    src = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{commit}:erotica/analysis/_isochrone.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tmp = Path(tempfile.mkdtemp()) / f"{name}.py"
    tmp.write_text(src)
    spec = importlib.util.spec_from_file_location(f"erotica.analysis.{name}", tmp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _old_module():
    return _module_at(PRE_FIX_COMMIT, "_isochrone_prefix")


OLD = _old_module()
new = _module_at(GRID_COMMIT, "_isochrone_grid")


def _bracket(nodes, x):
    k = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, nodes.size - 2))
    return k, k + 1, float(np.clip((x - nodes[k]) / (nodes[k + 1] - nodes[k]), 0, 1))


class OldFrameNodeInterp(OLD.IsochroneFitter):
    """Pre-fix frame, node-interpolated grid."""

    def _precompute_H_grid(self):
        iso = self._isochs
        fm, fl = iso._met_values, iso._loga_values
        self._met_grid = np.linspace(fm[0], fm[-1], self.M_met)
        self._loga_grid = np.linspace(fl[0], fl[-1], self.M_loga)
        cache = {}

        def node(mi, ai):
            if (mi, ai) not in cache:
                cache[(mi, ai)] = self._hess_for_isochrone(
                    *iso.get_isochrone(float(fm[mi]), float(fl[ai]))
                )
            return cache[(mi, ai)]

        H = np.zeros((self.M_met, self.M_loga, *self._Nbins))
        for i, m in enumerate(self._met_grid):
            m0, m1, wm = _bracket(fm, m)
            for j, a in enumerate(self._loga_grid):
                a0, a1, wa = _bracket(fl, a)
                H[i, j] = (
                    (1 - wm) * (1 - wa) * node(m0, a0)
                    + wm * (1 - wa) * node(m1, a0)
                    + (1 - wm) * wa * node(m0, a1)
                    + wm * wa * node(m1, a1)
                )
        self._H_grid = H


class NewFrameNearest(new.IsochroneFitter):
    """Current frame, pre-fix nearest-node grid."""

    def _precompute_H_grid(self):
        iso = self._isochs
        fm, fl = iso._met_values, iso._loga_values
        self._met_grid = np.linspace(fm[0], fm[-1], self.M_met)
        self._loga_grid = np.linspace(fl[0], fl[-1], self.M_loga)
        self._set_reference_frame()
        cache, H = {}, None
        for i, m in enumerate(self._met_grid):
            mi = int(np.argmin(abs(fm - m)))
            for j, a in enumerate(self._loga_grid):
                ai = int(np.argmin(abs(fl - a)))
                if (mi, ai) not in cache:
                    cache[(mi, ai)] = self._hess_for_isochrone(
                        *iso.get_isochrone(float(fm[mi]), float(fl[ai]))
                    )
                if H is None:
                    H = np.zeros((self.M_met, self.M_loga, *cache[(mi, ai)].shape))
                H[i, j] = cache[(mi, ai)]
        self._H_grid = H


class RefShifted(new.IsochroneFitter):
    """Current model; only the grid's internal reference moves by +0.5 bin in mag and colour."""

    def _set_reference_frame(self):
        super()._set_reference_frame()
        self._dmag_ref += 0.5 * self._binw_mag
        self._dcol_ref += 0.5 * self._binw_col
        self._pad = (self._pad[0] + 1, self._pad[1] + 1)


VARIANTS = {
    "old_frame_nearest": OLD.IsochroneFitter,
    "old_frame_interp": OldFrameNodeInterp,
    "new_frame_nearest": NewFrameNearest,
    "new_frame_interp": new.IsochroneFitter,
    "new_refshift": RefShifted,
}


def fitter(variant, data=None, **kw):
    f = VARIANTS[variant](isochs_path=MIST, **{**PRIORS, **kw})
    f.setup(
        QTable(Table.read(SAMPLE)) if data is None else data,
        prob_threshold=0.0,
        precompute_grid=True,
    )
    return f


def loglike_fn(f, bg=None):
    import pytensor
    import pytensor.tensor as pt

    met, loga, dm, Av, ls = (pt.dscalar(n) for n in ("met", "loga", "dm", "Av", "ls"))
    lam = pt.maximum(
        f._shift_histogram(f._interp_H(met, loga), dm + f._kG * Av, f._k_col1 * Av).reshape((-1,)),
        1e-6,
    )
    mu = pt.exp(ls) * lam + (0.0128 if bg is None else bg)
    return pytensor.function([met, loga, dm, Av, ls], pt.sum(f._obs_hess * pt.log(mu) - mu)), (
        met,
        loga,
        dm,
        Av,
        lam,
    )


# ---------------------------------------------------------------------------- sampling
def run_nuts(f, chains, seed=42, initvals=None, draws=2000, tune=2000):
    import arviz as az

    t = time.time()
    idata = f.fit(
        draws=draws,
        tune=tune,
        chains=chains,
        cores=1,
        target_accept=0.9,
        nuts_sampler="numpyro",
        random_seed=seed,
        progressbar=False,
        initvals=initvals,
    )
    post, ss = idata.posterior, idata.sample_stats
    E = ss["energy"].values
    res = {
        "seconds": round(time.time() - t, 1),
        "chains": chains,
        "N_in_hess": float(f._obs_hess.sum()),
        "divergences": int(ss["diverging"].values.sum()),
        "bfmi": np.round(np.mean(np.diff(E, axis=1) ** 2, axis=1) / np.var(E, axis=1), 3).tolist(),
        "tree_depth_chain_mean": np.round(ss["tree_depth"].values.mean(1), 2).tolist(),
        "step_size_chain_mean": np.round(ss["step_size"].values.mean(1), 4).tolist(),
        "lp_chain_mean": np.round(ss["lp"].values.mean(1), 2).tolist(),
    }
    rh, eb = az.rhat(post), az.ess(post, method="bulk")
    for v in PARAMS:
        a = post[v].values
        res[v] = {
            "rhat": round(float(rh[v]), 4),
            "ess_bulk": round(float(eb[v]), 0),
            "chain_means": np.round(a.mean(1), 4).tolist(),
            "q05_16_50_84_95": np.round(np.percentile(a.ravel(), [5, 16, 50, 84, 95]), 4).tolist(),
        }
    return res, idata


# ---------------------------------------------------------------------------- O2 generator
def chabrier_sample(rng, n, mlo, mhi):
    m = np.logspace(np.log10(mlo), np.log10(mhi), 20000)
    c = np.exp(-0.5 * ((0 - np.log10(0.2)) / 0.55) ** 2)
    pdf = np.where(
        m < 1.0, np.exp(-0.5 * ((np.log10(m) - np.log10(0.2)) / 0.55) ** 2) / m, c * m**-2.35
    )
    cdf = np.concatenate([[0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(m))])
    return np.interp(rng.uniform(size=n), cdf / cdf[-1], m)


def synthetic_cluster(Z, loga, dm, Av, seed, binaries=True, n_target=254, glim=19.8):
    """Star-level draw from a MIST node: own IMF inverse CDF, per-star D&K q draws, Gaussian errors."""
    rng = np.random.default_rng(seed)
    iso = MISTIsochrones(
        MIST, magnitude_col="Gaia_G_EDR3", color_col1="Gaia_BP_EDR3", color_col2="Gaia_RP_EDR3"
    )
    mass, G, BP, RP = iso.get_isochrone(Z, loga)
    kG, kBP, kRP = (_ccm89(lam, 3.1) for lam in (6390.7, 5182.6, 7825.1))
    flux = lambda mag: 10 ** (-0.4 * mag)  # noqa: E731
    out = []
    while sum(len(x[0]) for x in out) < n_target:
        m1 = chabrier_sample(rng, 4000, mass[0], mass[-1])
        g, bp, rp = (np.interp(m1, mass, a) for a in (G, BP, RP))
        isb = (rng.uniform(size=m1.size) < np.clip(0.09 + 0.94 / (1 + 1.4 / m1), 0, 1)) & binaries
        gam = np.where(
            m1 <= 0.1,
            4.2,
            np.where(m1 <= 0.6, 0.4, np.where(m1 <= 1.4, 0.3, np.where(m1 <= 6.5, -0.5, 0.0))),
        )
        m2 = np.clip(rng.uniform(size=m1.size) ** (1 / (gam + 1)) * m1, mass[0], None)
        for arr, band in ((g, G), (bp, BP), (rp, RP)):
            sec = np.interp(m2, mass, band)
            arr[isb] = -2.5 * np.log10(flux(arr[isb]) + flux(sec[isb]))
        Ga, BPa, RPa = g + dm + kG * Av, bp + dm + kBP * Av, rp + dm + kRP * Av
        eG = 0.0003 + 0.02 * 10 ** (0.4 * (Ga - 20.5))
        eC = 0.001 + 0.08 * 10 ** (0.4 * (Ga - 20.5))
        Go = Ga + rng.normal(0, 1, Ga.size) * eG
        Bo = BPa + rng.normal(0, 1, Ga.size) * eC
        Ro = RPa + rng.normal(0, 1, Ga.size) * eC
        keep = Go < glim
        out.append((Go[keep], Bo[keep], Ro[keep], eG[keep], eC[keep], eC[keep]))
    cols = [np.concatenate([x[k] for x in out])[:n_target] for k in range(6)]
    return QTable(
        {
            "Gmag": cols[0],
            "G_BPmag": cols[1],
            "G_RPmag": cols[2],
            "e_Gmag": cols[3],
            "e_G_BPmag": cols[4],
            "e_G_RPmag": cols[5],
            "probability_hdbscan": np.ones(n_target),
        }
    )


# ---------------------------------------------------------------------------- stages
def stage_frame():
    import pytensor.tensor as pt

    res = {}
    for v in ("old_frame_nearest", "new_frame_interp"):
        f = fitter(v)
        H = f._shift_histogram(
            f._interp_H(pt.constant(0.015), pt.constant(6.55)),
            pt.constant(10.3 + f._kG * 1.24),
            pt.constant(f._k_col1 * 1.24),
        ).eval()
        obs = f._obs_hess.reshape(f._Nbins)
        edges = np.linspace(*f._mag_range, f._Nbins[0] + 1)
        res[v] = {
            "params": {"met": 0.015, "loga": 6.55, "dm": 10.3, "Av": 1.24},
            "row_G_lo": np.round(edges[:-1], 2).tolist(),
            "obs_per_row": obs.sum(1).tolist(),
            "model_per_row_scaled_to_N": np.round(
                H.sum(1) / max(H.sum(), 1e-12) * obs.sum(), 2
            ).tolist(),
            "N_obs": int(f._N_obs),
            "N_in_hess": float(obs.sum()),
        }
    return res


def stage_gradient():
    res = {}
    for v in ("old_frame_nearest", "new_frame_interp"):
        f = fitter(v)
        H = f._H_grid
        fn = loglike_fn(f)[1]
        import pytensor
        import pytensor.tensor as pt

        met, loga, dm, Av, lam = fn
        mu = 254.0 * lam + 0.1
        g = pytensor.function(
            [met, loga, dm, Av], pytensor.grad(pt.sum(f._obs_hess * pt.log(mu) - mu), [met, loga])
        )
        rng = np.random.default_rng(0)
        zero = np.zeros(2, int)
        for _ in range(300):
            out = g(
                rng.uniform(1e-6, 0.045),
                rng.uniform(6, 7),
                rng.uniform(9.6, 10.6),
                rng.uniform(0.55, 1.95),
            )
            zero += np.array([float(x) == 0.0 for x in out])
        res[v] = {
            "identical_adjacent_met_slices": f"{sum(np.array_equal(H[i], H[i + 1]) for i in range(H.shape[0] - 1))}/{H.shape[0] - 1}",
            "identical_adjacent_loga_slices": f"{sum(np.array_equal(H[:, j], H[:, j + 1]) for j in range(H.shape[1] - 1))}/{H.shape[1] - 1}",
            "zero_grad_met_of_300": int(zero[0]),
            "zero_grad_loga_of_300": int(zero[1]),
        }
    return res


def stage_ablation():
    res = {}
    for v in ("old_frame_nearest", "old_frame_interp", "new_frame_nearest", "new_frame_interp"):
        res[f"{v}_2ch"] = run_nuts(fitter(v), 2)[0]
    one = {
        "met": 0.0143,
        "loga": 6.55,
        "dm": 10.3,
        "Av": 1.24,
        "log_s": float(np.log(254)),
        "bg": 0.05,
    }
    res["old_frame_nearest_2ch_one_start"] = run_nuts(fitter("old_frame_nearest"), 2, initvals=one)[
        0
    ]
    res["new_frame_interp_4ch"] = run_nuts(fitter("new_frame_interp"), 4)[0]
    return res


def stage_refinv():
    res = {"points": [], "loglike_new": [], "loglike_refshift": []}
    fns = {v: loglike_fn(fitter(v))[0] for v in ("new_frame_interp", "new_refshift")}
    for p in [
        (0.00536, 6.292, 10.01, 0.553, 5.543),
        (0.0143, 6.55, 10.3, 1.24, 5.5),
        (0.0143, 6.55, 10.1, 1.0, 5.5),
    ]:
        res["points"].append(p)
        res["loglike_new"].append(round(float(fns["new_frame_interp"](*p)), 2))
        res["loglike_refshift"].append(round(float(fns["new_refshift"](*p)), 2))
    res["new_refshift_4ch"] = run_nuts(fitter("new_refshift"), 4)[0]
    return res


def stage_recovery():
    import arviz as az

    nodes = np.array([4.518e-3, 8.033e-3, 1.4286e-2, 2.5404e-2])
    batches = {
        "binaries": dict(dm=10.45, Av=1.24, binaries=True, seeds=range(1, 11)),
        "singles": dict(dm=10.47, Av=1.10, binaries=False, seeds=range(1, 7)),
    }
    res = {}
    for name, b in batches.items():
        rows, pooled_met = [], []
        for seed in b["seeds"]:
            data = synthetic_cluster(0.014286, 6.55, b["dm"], b["Av"], seed, binaries=b["binaries"])
            kw = {} if b["binaries"] else {"alpha": 0.0, "beta": 0.0}
            f = fitter("new_frame_interp", data=data, **kw)
            idata = f.fit(
                draws=1000,
                tune=1500,
                chains=2,
                cores=1,
                target_accept=0.9,
                nuts_sampler="numpyro",
                random_seed=seed,
                progressbar=False,
            )
            post = idata.posterior
            rh = az.rhat(post)
            row = {"seed": seed, "divergences": int(idata.sample_stats["diverging"].values.sum())}
            truth = {"met": 0.014286, "loga": 6.55, "dm": b["dm"], "Av": b["Av"]}
            for p, t in truth.items():
                a = post[p].values.ravel()
                q5, med, q95 = np.percentile(a, [5, 50, 95])
                row[p] = {
                    "truth": t,
                    "median": round(float(med), 4),
                    "q05": round(float(q5), 4),
                    "q95": round(float(q95), 4),
                    "in90": bool(q5 <= t <= q95),
                    "rhat": round(float(rh[p]), 3),
                }
            rows.append(row)
            pooled_met.append(post["met"].values.ravel())
        met = np.concatenate(pooled_met)
        res[name] = {
            "truth": {"met": 0.014286, "loga": 6.55, "dm": b["dm"], "Av": b["Av"]},
            "reference": {"dm_mu": PRIORS["dm_mu"], "Av_mid": float(np.mean(PRIORS["Av_range"]))},
            "runs": rows,
            "dm_truth_in90": f"{sum(r['dm']['in90'] for r in rows)}/{len(rows)}",
            "runs_with_rhat_gt_1.01": sum(any(r[p]["rhat"] > 1.01 for p in truth) for r in rows),
            "pooled_met_frac_within_3e-4_of_node": round(
                float(np.mean(np.min(np.abs(met[:, None] - nodes[None]), axis=1) < 3e-4)), 3
            ),
        }
    return res


STAGES = {
    "frame": stage_frame,
    "gradient": stage_gradient,
    "ablation": stage_ablation,
    "refinv": stage_refinv,
    "recovery": stage_recovery,
}

if __name__ == "__main__":
    import pymc

    import erotica

    todo = sys.argv[1:] or list(STAGES)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}
    out["env"] = {
        "pymc": pymc.__version__,
        "erotica_file": erotica.__file__,
        "pre_fix_commit": PRE_FIX_COMMIT,
        "head": subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
    }
    for s in todo:
        print(f"== {s}", flush=True)
        out[s] = STAGES[s]()
        OUT.write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("wrote", OUT)
