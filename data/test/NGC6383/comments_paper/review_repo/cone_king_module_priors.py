#!/usr/bin/env python3
"""Cone King refits with the PAPER's exact prior parametrization (cosmic
structure.RDP_bayesian: b,k~U; Rc~U(0,0.8*maxd); Rt~U(Rc,1.5*maxd)) so the
manuscript's 'identical priors' claim is literally true. ta=0.99, diagnostics."""
import sys, numpy as np
sys.path.insert(0,"/Users/notluquis/erotica")
import astropy.units as u, arviz as az
from astropy.table import Table
from astropy.coordinates import SkyCoord
from pumps.analysis import structure
from pumps.analysis.inference import SamplingConfig

B="/Users/notluquis/erotica/data/test/NGC6383/comments_paper/radius_robustness/generated"
center=SkyCoord(263.6826*u.deg,-32.5838*u.deg)
for radius in [50,60,70]:
    t=Table.read(f"{B}/{radius}/paperfaithful_reference_p06.ecsv")
    prof=structure.radial_density_profile(t,center,method="equip")
    res=structure.RDP_bayesian(prof.density,prof.radius,d_center=42.45*u.arcmin,
        return_trace=True,
        sampling=SamplingConfig(draws=2000,tune=2000,target_accept=0.99,chains=4,
                                random_seed=42,nuts_sampler="pymc",progressbar=False,
                                extra_kwargs={"cores":1}))
    tr=res["king_trace"]; post=tr.posterior
    out={v:(float(post[v].values.mean()),float(post[v].values.std())) for v in ["R_c","R_t","b","k"]}
    C=np.log10(post["R_t"].values.ravel()/post["R_c"].values.ravel())
    s=az.summary(tr,var_names=["R_c","R_t","b","k","sigma"],round_to=4)
    div=int(tr.sample_stats.diverging.sum()); bfmi=float(np.min(az.bfmi(tr)))
    rmax=float(s["r_hat"].max()); bmin=float(s["ess_bulk"].min()); tmin=float(s["ess_tail"].min())
    ok=rmax<1.01 and bmin>400 and tmin>400 and div==0 and bfmi>0.3
    print(f"[{radius}'] N={len(t)} Rc={out['R_c'][0]:.2f}+-{out['R_c'][1]:.2f} Rt={out['R_t'][0]:.1f}+-{out['R_t'][1]:.1f} "
          f"C={C.mean():.2f}+-{C.std():.2f} b={out['b'][0]:.5f} | div={div} BFMI={bfmi:.2f} Rhat={rmax:.4f} ESS={bmin:.0f}/{tmin:.0f} {'PASS' if ok else 'FAIL'}")
    tr.to_netcdf(f"/Users/notluquis/erotica/data/test/NGC6383/comments_paper/review_repo/idata_king_cone{radius}_modpriors.nc")
