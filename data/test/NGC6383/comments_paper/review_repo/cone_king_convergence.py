#!/usr/bin/env python3
"""Referee point 1 follow-up: derived parameters for the 50-arcmin cone run
(the run whose automated branch coincides with the NGC-like branch) and a
King-profile refit on the 50-arcmin reference sample, using the same priors
as the paper (b~U(0,2rho_min), k~U(0,2rho_max), Rc~U(0,0.8Rt), Rt~U(Rc,1.5Tmax)).
Sanity: the same code refits the 40-arcmin sample to confirm it reproduces the
published values before trusting the 50-arcmin numbers."""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u

B = "/Users/notluquis/erotica/data/test/NGC6383/comments_paper/radius_robustness/generated"
CRA, CDEC = 263.6826, -32.5838
TMAX = 42.45  # arcmin: max(Hill 28.32, bound 42.45) as in the paper
c = SkyCoord(CRA * u.deg, CDEC * u.deg)
rng = np.random.default_rng(11)

def load(radius):
    t = Table.read(f"{B}/{radius}/paperfaithful_reference_p06.ecsv").to_pandas()
    sc = SkyCoord(t.ra.values * u.deg, t.dec.values * u.deg)
    t["dc"] = angular_separation(sc.ra, sc.dec, c.ra, c.dec).to(u.arcmin).value
    return t

def astrometry(t, tag):
    plx = t.parallax.values
    pmra, pmde = t.pmra.values, t.pmdec.values
    # parallax stats on the delta_varpi/varpi < 0.1 subsample, as in the paper
    frac = t.parallax_error.values / np.abs(plx)
    sel = frac < 0.1
    print(f"[{tag}] N={len(t)}  plx(all)={plx.mean():.3f}+-{plx.std():.3f}"
          f"  plx(frac<0.1, n={sel.sum()})={plx[sel].mean():.3f}+-{plx[sel].std():.3f} mas")
    print(f"[{tag}] pmRA={pmra.mean():.3f}+-{pmra.std():.3f}  pmDE={pmde.mean():.3f}+-{pmde.std():.3f} mas/yr")

def king_density(t):
    n = len(t)
    K = int(2 * n ** 0.4)
    q = np.quantile(np.sort(t.dc.values), np.linspace(0, 1, K + 1))
    counts, _ = np.histogram(t.dc.values, bins=q)
    area = np.pi * (q[1:] ** 2 - q[:-1] ** 2)
    rho = counts / area
    rmid = 0.5 * (q[1:] + q[:-1])
    return rmid, rho, K

def fit_king(t, tag):
    rmid, rho, K = king_density(t)
    import pymc as pm
    with pm.Model():
        b = pm.Uniform("b", 0, 2 * rho.min())
        k = pm.Uniform("k", 0, 2 * rho.max())
        Rt = pm.Uniform("Rt", 0, 1.5 * TMAX)
        Rc = pm.Uniform("Rc", 0, 0.8 * Rt)
        mod = b + k * (1 / np.sqrt(1 + (rmid / Rc) ** 2)
                       - 1 / np.sqrt(1 + (Rt / Rc) ** 2)) ** 2
        sigma = pm.HalfNormal("sigma", rho.std())
        pm.Normal("obs", mu=mod, sigma=sigma, observed=rho)
        idata = pm.sample(2000, tune=2000, chains=4, cores=4,
                          progressbar=False, random_seed=11, target_accept=0.99)
    post = idata.posterior
    out = {}
    for v in ["b", "k", "Rc", "Rt"]:
        s = post[v].values.ravel()
        out[v] = (np.mean(s), np.std(s))
    C = np.log10(post["Rt"].values.ravel() / post["Rc"].values.ravel())
    print(f"[{tag}] K={K} annuli | Rc={out['Rc'][0]:.2f}+-{out['Rc'][1]:.2f}'"
          f"  Rt={out['Rt'][0]:.1f}+-{out['Rt'][1]:.1f}'"
          f"  b={out['b'][0]:.5f}+-{out['b'][1]:.5f}  k={out['k'][0]:.2f}+-{out['k'][1]:.2f}"
          f"  C=log10(Rt/Rc)={C.mean():.2f}+-{C.std():.2f}")
    import arviz as az
    summ = az.summary(idata, var_names=["b","k","Rc","Rt","sigma"], round_to=4)
    div = int(idata.sample_stats.diverging.sum())
    bfmi = float(np.min(az.bfmi(idata)))
    rmax = float(summ["r_hat"].max()); bmin = float(summ["ess_bulk"].min()); tmin = float(summ["ess_tail"].min())
    passed = rmax < 1.01 and bmin > 400 and tmin > 400 and div == 0 and bfmi > 0.3
    print(f"[{tag}] DIAG: div={div} minBFMI={bfmi:.3f} maxRhat={rmax:.4f} minESS={bmin:.0f}/{tmin:.0f} -> {'PASS' if passed else 'FAIL'}")
    idata.to_netcdf(f"/Users/notluquis/erotica/data/test/NGC6383/comments_paper/review_repo/idata_king_cone{tag.split()[-1].strip(chr(39))}.nc")
    return out

for radius in [40, 50, 60, 70]:
    t = load(radius)
    astrometry(t, f"{radius}'")
for radius in [40, 50, 60, 70]:
    t = load(radius)
    fit_king(t, f"King {radius}'")
