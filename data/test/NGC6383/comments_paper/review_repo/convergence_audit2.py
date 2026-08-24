#!/usr/bin/env python3
"""Convergence audit (Vehtari et al. 2021 protocol) for the paper's Bayesian fits:
parallax, distance, proper-motion, and King-profile models, re-run with NUTS
(4 chains x 2000 draws, tune 2000) on the production 254-source reference sample.
Criteria: rank-normalized R-hat < 1.01, bulk/tail ESS > 400, 0 divergences,
E-BFMI > 0.3. Prints diagnostics + comparison to published values; saves idata."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/notluquis/erotica")
import astropy.units as u
from astropy.table import QTable, Table
from astropy.coordinates import SkyCoord, angular_separation
import arviz as az

from erotica.analysis.inference import (
    SamplingConfig, fit_parallax_model, distance_model, proper_motion_2d_gaussian,
)
from erotica.analysis import structure, write_metadata

B = "/Users/notluquis/erotica/data/test/NGC6383"
OUT = f"{B}/comments_paper/review_repo"
CFG = SamplingConfig(draws=2000, tune=2000, target_accept=0.9, chains=4,
                     random_seed=42, nuts_sampler="pymc", progressbar=False, extra_kwargs={"cores": 1})

SRC = [f"{B}/comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv",
       f"{B}/data/40/clustering_results.ecsv"]
ref = Table.read(SRC[0])
print(f"reference sample N={len(ref)}")
frac = np.abs(ref["parallax_error"] / ref["parallax"])
sub = ref[frac < 0.1]
print(f"frac plx err <0.1 subsample n={len(sub)}")

# Bailer-Jones geometric distances matched by source_id
cl = Table.read(SRC[1])
bj = pd.DataFrame({"source_id": np.asarray(cl["source_id"]),
                   "r_med_geo": np.asarray(cl["r_med_geo"], dtype=float)})
sid = pd.DataFrame({"source_id": np.asarray(sub["source_id"])})
rgeo = sid.merge(bj, on="source_id", how="left")["r_med_geo"].values
rgeo_kpc = rgeo[np.isfinite(rgeo)] / 1000.0
print(f"BJ distances matched: {len(rgeo_kpc)}/{len(sub)}")


def diag(idata, name, params):
    s = az.summary(idata, var_names=params, round_to=4)
    div = int(idata.sample_stats.diverging.sum()) if "diverging" in idata.sample_stats else -1
    try:
        bfmi = float(np.min(az.bfmi(idata)))
    except Exception:
        bfmi = float("nan")
    rmax = float(s["r_hat"].max()); bmin = float(s["ess_bulk"].min()); tmin = float(s["ess_tail"].min())
    ok = (rmax < 1.01) and (bmin > 400) and (tmin > 400) and (div == 0) and (bfmi > 0.3)
    print(f"\n===== {name} =====")
    print(s[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]])
    print(f"divergences={div}  min E-BFMI={bfmi:.3f}  max R-hat={rmax:.4f}  "
          f"min ESS bulk/tail={bmin:.0f}/{tmin:.0f}  -> {'PASS' if ok else 'FAIL'}")
    nc = f"{OUT}/idata_{name}.nc"
    idata.to_netcdf(nc)
    write_metadata(nc[: -len(".nc")] + "_provenance.json", inputs=SRC, seeds=CFG.random_seed,
                   script="convergence_audit2.py", model=name)
    return ok

# 3. proper-motion 2D Gaussian (published means 2.542/-1.713, dispersions 0.152/0.138)
r3 = proper_motion_2d_gaussian(np.asarray(ref["pmra"], float) * u.mas / u.yr,
                               np.asarray(ref["pmdec"], float) * u.mas / u.yr,
                               return_trace=True, sampling=CFG)
res = r3.results
print(f"\nPM: mu=({res['mu_RA_mean']:.4f},{res['mu_Dec_mean']:.4f}) "
      f"sigma=({res['sigma_RA_mean']:.4f},{res['sigma_Dec_mean']:.4f}) corr={res['corr_mean']:.3f}")
diag(r3.trace, "pm", ["mu_RA", "mu_Dec", "sigma_RA", "sigma_Dec", "corr"])

# 4. King profile (published Rc 1.95+-0.19', Rt 40.4+-14.3', k 4.91, b 0.0111)
center = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
prof = structure.radial_density_profile(ref, center, method="equip")
king = structure.RDP_bayesian(prof.density, prof.radius, d_center=42.45 * u.arcmin,
                              return_trace=True,
                              sampling=SamplingConfig(draws=2000, tune=2000, target_accept=0.95,
                                                      chains=4, random_seed=42,
                                                      nuts_sampler="pymc", progressbar=False,
                                                      extra_kwargs={"cores": 1}))
tr = king["king_trace"]
post = tr.posterior
for v in ["R_c", "R_t", "b", "k"]:
    arr = post[v].values.ravel(); print(f"King {v}: {arr.mean():.4f}+-{arr.std():.4f}")
C = np.log10(post["R_t"].values.ravel() / post["R_c"].values.ravel())
print(f"King C=log10(Rt/Rc): {C.mean():.3f}+-{C.std():.3f}  (published 1.32+-0.16)")
diag(tr, "king", ["R_c", "R_t", "b", "k", "sigma"])
print("\nAUDIT DONE")
