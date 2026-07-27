#!/usr/bin/env python3
"""Re-fit the NGC 6383 isochrone posterior with true NUTS (cosmic IsochroneFitter,
blackjax backend), using the PAPER priors: loga~U(6,7), Av~U(0.5,2.0),
dm~N(10.3,0.2), on the production 254-source reference sample (no PMS weighting).
Goal: (a) honest convergence diagnostics (the 2024 production trace was a
300-chain DEMetropolis-family run failing the Vehtari criteria), (b) check the
published mode values (loga 6.550+-0.145, dm 10.300+-0.262, Av 1.240+-0.262,
Z 0.024+-0.008) against a properly converged posterior."""
import sys, numpy as np
sys.path.insert(0, "/Users/notluquis/erotica")
from astropy.table import QTable, Table
import arviz as az
from scipy.stats import gaussian_kde

from pumps.analysis._isochrone import IsochroneFitter

B = "/Users/notluquis/erotica/data/test/NGC6383"
ref = Table.read(f"{B}/comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv")
print(f"reference sample N={len(ref)}")

fitter = IsochroneFitter(
    isochs_path=f"{B}/MIST/UBVRIplus",
    loga_range=(6.0, 7.0),
    Av_range=(0.5, 2.0),
    dm_mu=10.3,
    dm_sigma=0.2,
    dm_range=(9.5, 10.7),
    M_met=200,
    M_loga=200,
)
fitter.setup(QTable(ref), prob_threshold=0.0, precompute_grid=False)  # all 254 members
fitter.build_grid(grid_cache=f"{B}/data/40/hgrid_paper254.npz")

idata = fitter.fit(
    draws=2000, tune=2000, chains=4, target_accept=0.9,
    nuts_sampler="blackjax", random_seed=42,
)

params = ["met", "loga", "Av", "dm"]
avail = [p for p in params if p in idata.posterior]
s = az.summary(idata, var_names=avail, round_to=4)
print(s[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]])
div = int(idata.sample_stats.diverging.sum()) if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats else -1
try:
    bfmi = float(np.min(az.bfmi(idata)))
except Exception:
    bfmi = float("nan")
print(f"divergences={div}  min E-BFMI={bfmi:.3f}")

print("\nmode (KDE) per parameter vs published:")
pub = {"loga": (6.550, 0.145), "dm": (10.300, 0.262), "Av": (1.240, 0.262), "met": (0.024, 0.008)}
for p in avail:
    arr = idata.posterior[p].values.ravel()
    grid = np.linspace(arr.min(), arr.max(), 2000)
    mode = grid[np.argmax(gaussian_kde(arr)(grid))]
    lo, hi = np.percentile(arr, [16, 84])
    pm_, ps = pub.get(p, (np.nan, np.nan))
    print(f"  {p}: mode={mode:.4f}  mean={arr.mean():.4f}+-{arr.std():.4f}  [16,84]=({lo:.4f},{hi:.4f})  published={pm_}+-{ps}")

idata.to_netcdf(f"{B}/comments_paper/review_repo/idata_isochrone_nuts.nc")
print("saved idata_isochrone_nuts.nc\nISOCHRONE REFIT DONE")
