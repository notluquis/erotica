"""Checks added after review (finding §7): (i) EEM->EEM control, (ii) MIST with the EEM T_eff ceiling
(10 645 K), so that MIST - EEM is compared on the same prior volume and after subtracting each model's
own estimator bias, per true-T_eff bin; real fits with the MIST ceiling; A_V scale of the CMD layer."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data, sedgrid as sg
from run_synth import draw

R = 5
TB = [(3.52, 3.65), (3.65, 3.75), (3.75, 3.85), (3.85, 4.03)]
t, mag, err, have, flags = data.load()
lt_e, bc_e, rng_e = sg.eem_fine(0.0)
lt_m, bc_m = sg.mist_fine(0.0, t_hi=rng_e[1] * 1.0001)
print("EEM range", rng_e, "MIST10k grid", 10 ** lt_m[0], 10 ** lt_m[-1], bc_m.shape, bc_e.shape)
fits = {"eem": (lt_e, bc_e), "mist10k": (lt_m, bc_m)}
arms = [("ctl_eem", "eem", "eem"), ("ctl_m10", "mist", "mist10k"), ("mm_m10", "eem", "mist10k")]
rng = np.random.default_rng(20261012)
out = {"arms": {}}
for config in ("b", "a"):
    use_all = data.mask(have, config)
    for name, truth, fit in arms:
        lt_f, bc_f = fits[fit]
        res = []
        for i in range(len(t)):
            if use_all[i].sum() < 3:
                continue
            s = data.sigma(err[i:i + 1])[0]
            for _ in range(R):
                lt, lg, av = draw(rng, "eem-truth")  # truth capped at 10 500 K for every arm
                m = sg.truth_mags(truth, lt, lg, av) + 15 + rng.normal(0, 1, 6) * s
                p, c = sg.fit_star(bc_f, m, s, use_all[i])
                sm = sg.summarise(p, lt_f)
                res.append((lt, av, sm["logt50"], sm["av50"], sm["logt16"], sm["logt84"], sm["av16"], sm["av84"], sm["sd_av"]))
        a = np.array(res)
        o = {}
        for lo, hi in TB:
            s_ = (a[:, 0] >= lo) & (a[:, 0] < hi)
            o[f"T{10**lo:.0f}-{10**hi:.0f}"] = dict(
                n=int(s_.sum()), bias_av=round(float(np.median(a[s_, 3] - a[s_, 1])), 3),
                bias_teff_frac=round(float(np.median(10 ** (a[s_, 2] - a[s_, 0]) - 1)), 4),
                cov_av=round(float(np.mean((a[s_, 6] <= a[s_, 1]) & (a[s_, 1] <= a[s_, 7]))), 3),
                cov_logt=round(float(np.mean((a[s_, 4] <= a[s_, 0]) & (a[s_, 0] <= a[s_, 5]))), 3),
                med_sd_av=round(float(np.median(a[s_, 8])), 3))
        out["arms"][f"{name}_{config}"] = o
        print(f"{name}_{config}", json.dumps(o))
    # atmosphere systematic net of the estimator: mm_m10 - ctl_m10 per bin
    out[f"atmos_net_{config}"] = {b: dict(dav=round(out["arms"][f"mm_m10_{config}"][b]["bias_av"] - out["arms"][f"ctl_m10_{config}"][b]["bias_av"], 3),
                                          dteff_frac=round(out["arms"][f"mm_m10_{config}"][b]["bias_teff_frac"] - out["arms"][f"ctl_m10_{config}"][b]["bias_teff_frac"], 4))
                                  for b in out["arms"][f"ctl_m10_{config}"]}
    print("atmos_net", config, out[f"atmos_net_{config}"])

# real fits with MIST10k, and comparison to Zhang+23 / BossNet / EEM per real star
from astropy.table import Table, join
r = Table.read(HERE / "sed_real_perstar.ecsv")
z = Table.read(HERE / "members_zhang23.ecsv")
spec = json.loads(Path("/Users/notluquis/phd/agent-findings/scripts/isochrone_external_spectra.json").read_text())["stars"]
G = mag[:, 0]
real = {}
for config in ("a", "b"):
    use_all = data.mask(have, config); s = data.sigma(err)
    te, av = np.full(len(t), np.nan), np.full(len(t), np.nan)
    for i in range(len(t)):
        if use_all[i].sum() < 3:
            continue
        p, c = sg.fit_star(bc_m, mag[i], s[i], use_all[i]); sm = sg.summarise(p, lt_m)
        te[i], av[i] = 10 ** sm["logt50"], sm["av50"]
    real[config] = (te, av)
sid = np.asarray(t["source_id"], np.int64)
zi = {int(x): k for k, x in enumerate(z["source_id"])}
zt = np.array([z["z23_teff"][zi[s_]] if s_ in zi and z["z23_quality_flags"][zi[s_]] < 8 else np.nan for s_ in sid])
za = np.array([z["z23_A550"][zi[s_]] if s_ in zi and z["z23_quality_flags"][zi[s_]] < 8 else np.nan for s_ in sid])
bt = np.array([spec.get(str(s_), {}).get("bossnet_dr20", {}).get("teff", np.nan) for s_ in sid])
ri = {int(x): k for k, x in enumerate(r["source_id"])}
pms = G > 14
def st(d):
    d = d[np.isfinite(d)]
    return dict(n=int(d.size), median=round(float(np.median(d)), 3), rscatter=round(float(1.4826 * np.median(np.abs(d - np.median(d)))), 3)) if d.size else dict(n=0)
cmp = {}
for config in ("a", "b"):
    te, av = real[config]
    ee = np.array([r[f"eem_{config}_teff50"][ri[s_]] for s_ in sid]); ea = np.array([r[f"eem_{config}_av50"][ri[s_]] for s_ in sid])
    cmp[config] = dict(
        m10_vs_eem=dict(dteff_frac=st((te / ee - 1)[pms]), dav=st((av - ea)[pms])),
        m10_vs_zhang=dict(dteff_frac=st((te / zt - 1)[pms]), dav=st((av - za)[pms])),
        eem_vs_zhang=dict(dteff_frac=st((ee / zt - 1)[pms]), dav=st((ea - za)[pms])),
        m10_vs_boss=dict(dteff_frac=st((te / bt - 1)[pms])), eem_vs_boss=dict(dteff_frac=st((ee / bt - 1)[pms])))
    print("real", config, json.dumps(cmp[config]))
out["real_pms"] = cmp

# A_V scale: k_col = E(BP-RP)/A_V and A_G/A_V from MIST at A_V ~ 0.7, log g 4.0, per T_eff
T, g, A, bc = sg.load_mist_bc(0.0)
ig = list(g).index(4.0); ia = list(A).index(0.6); ib = list(A).index(0.8)
scale = {}
for TT in (3500, 4000, 4500, 5000, 6000, 8000, 15000, 25000):
    it = list(T).index(TT) if TT in T else int(np.argmin(np.abs(T - TT)))
    ext = lambda k: ((bc[it, ig, 0, k] - bc[it, ig, ia, k]) / A[ia] + (bc[it, ig, 0, k] - bc[it, ig, ib, k]) / A[ib]) / 2
    kc = ext(1) - ext(2); kg = ext(0)
    scale[int(T[it])] = dict(k_col=round(float(kc), 4), k_G=round(float(kg), 4),
                             cmd_066_as_column=round(0.66 * 0.449 / kc, 3), cmd_059=round(0.59 * 0.449 / kc, 3), cmd_071=round(0.71 * 0.449 / kc, 3))
out["av_scale"] = scale
print("AV scale (CMD layer k_col=0.449, k_G=0.844):", json.dumps(scale, indent=0))
(HERE / "extra_checks.json").write_text(json.dumps(out, indent=1))
