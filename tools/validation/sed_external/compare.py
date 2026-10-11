"""Control of the SED T_eff / A_V against published values on the same stars, the usable-star
selection and the hierarchical cluster A_V level (pre-registered §0.5).

Published sources and their independence class (§0.2): Zhang+23 (isochrone-free, XP), GSP-Phot and
StarHorse (isochrone priors), BossNet and XCSAO (SDSS-V DR20 BOSS spectra), spectral-type anchors.
Writes compare.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table, join

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402

SPEC = Path("/Users/notluquis/phd/agent-findings/scripts/isochrone_external_spectra.json")
SYNTH = HERE / "synth_summary.json"


def f(col):
    return np.ma.asarray(col).astype(float).filled(np.nan)


def stats(d):
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0)
    return dict(n=int(d.size), median=round(float(np.median(d)), 3),
                rscatter=round(float(1.4826 * np.median(np.abs(d - np.median(d)))), 3))


def hier(marg, av, mu_grid=np.arange(0.0, 3.0001, 0.005), tau_grid=np.arange(0.0, 1.0001, 0.005)):
    """A_V,i ~ N(mu, tau^2), per-star likelihoods on the A_V grid (flat prior, so posterior = L).
    Flat priors on mu and tau. Returns marginal summaries of mu and tau."""
    dA = av[1] - av[0]
    logL = np.zeros((mu_grid.size, tau_grid.size))
    for p in marg:
        p = p / p.sum()
        for j, tau in enumerate(tau_grid):
            if tau < dA:  # delta: evaluate the per-star likelihood at mu
                logL[:, j] += np.log(np.interp(mu_grid, av, p) / dA + 1e-300)
            else:
                k = np.exp(-0.5 * ((av[None, :] - mu_grid[:, None]) / tau) ** 2) / (np.sqrt(2 * np.pi) * tau)
                logL[:, j] += np.log((k * p[None, :]).sum(1) + 1e-300)
    P = np.exp(logL - logL.max())
    P /= P.sum()

    def q(x, w):
        c = np.cumsum(w) / w.sum()
        return [round(float(np.interp(v, c, x)), 3) for v in (0.16, 0.5, 0.84)]

    return dict(n=len(marg), mu=q(mu_grid, P.sum(1)), tau=q(tau_grid, P.sum(0)))


def main():
    r = Table.read(HERE / "sed_real_perstar.ecsv")
    g = Table.read(HERE / "members_gaia_ap.ecsv")
    z = Table.read(HERE / "members_zhang23.ecsv")
    t = join(r, g, keys="source_id", join_type="left")
    t = join(t, z, keys="source_id", join_type="left")
    spec = json.loads(SPEC.read_text())["stars"]
    G = f(t["G"])
    pms = G > 14
    out = {}

    # ---- usable stars (config b) and the hierarchical level
    synth = json.loads(SYNTH.read_text())
    gate_ok = {}
    for lo, hi in ((-np.inf, 14), (14, 17), (17, np.inf)):
        k = f"G{lo}-{hi}"
        c = synth["ctl_b"][k]
        gate_ok[k] = bool(c and abs(c["bias_av"]) <= 0.05 and c["cov_av"] >= 0.55)
    out["gate_ctl_b"] = gate_ok
    bin_ok = np.zeros(len(t), bool)
    for lo, hi in ((-np.inf, 14), (14, 17), (17, np.inf)):
        bin_ok |= (G >= lo) & (G < hi) & gate_ok[f"G{lo}-{hi}"]
    sd_b = f(t["mist_b_sd_av"])
    dme = f(t["mist_b_av50"]) - f(t["eem_b_av50"])
    usable = bin_ok & (sd_b <= 0.15) & np.isfinite(dme) & (np.abs(dme) <= 0.10)
    usable_noeem = bin_ok & (sd_b <= 0.15)
    out["usable_b"] = dict(n=int(usable.sum()), n_pms=int((usable & pms).sum()),
                           n_sd_only=int(usable_noeem.sum()), n_sd_only_pms=int((usable_noeem & pms).sum()),
                           n_eem_defined=int(np.isfinite(dme).sum()),
                           n_eem_ok=int((np.abs(dme) <= 0.10).sum()))
    m = np.load(HERE / "sed_real_av_marginals.npz")
    sid = m["source_id"]
    idx = {s: i for i, s in enumerate(sid)}
    rows = np.array([idx[s] for s in np.asarray(t["source_id"])])
    for name in ("mist_b", "eem_b", "mist_a"):
        sel = usable & (m[name][rows].sum(1) > 0)
        out[f"hier_{name}_usable"] = hier(m[name][rows][sel], m["av"])
        selp = sel & pms
        out[f"hier_{name}_usable_pms"] = hier(m[name][rows][selp], m["av"]) if selp.sum() > 2 else None
        selm = sel & ~pms
        out[f"hier_{name}_usable_ms"] = hier(m[name][rows][selm], m["av"]) if selm.sum() > 2 else None
    # anchors (spectral types), config b and a
    for c in ("a", "b"):
        a = f(t[f"anchor_{c}_av50"])
        ok = np.isfinite(a)
        sel = ok & (m["mist_b"][rows].sum(1) > 0)
        out[f"anchors_{c}"] = dict(n=int(ok.sum()), median=round(float(np.nanmedian(a)), 3),
                                   values=[(int(s), round(float(v), 3), str(src))
                                           for s, v, src in zip(t["source_id"][ok], a[ok], t["anchor_src"][ok])])

    # ---- control against published (config a and b, MIST and EEM)
    pub = {
        "zhang23": (f(t["z23_teff"]), f(t["z23_A550"]), f(t["z23_quality_flags"]) < 8),
        "gspphot": (f(t["ap_Teff"]), f(t["ap_A0"]), np.isfinite(f(t["ap_Teff"]))),
        "starhorse": (f(t["sh_teff50"]), f(t["sh_av50"]), f(t["sh_sh_outflag"]) if False else np.isfinite(f(t["sh_teff50"]))),
        "esphs": (f(t["ap_Teff-HS"]), f(t["ap_A0-HS"]), np.isfinite(f(t["ap_Teff-HS"]))),
    }
    boss_t = np.full(len(t), np.nan)
    xc_t = np.full(len(t), np.nan)
    for i, s in enumerate(np.asarray(t["source_id"])):
        v = spec.get(str(int(s)), {}).get("bossnet_dr20")
        if v:
            boss_t[i], xc_t[i] = v["teff"], v["xcsao_teff"]
    pub["bossnet"] = (boss_t, np.full(len(t), np.nan), np.isfinite(boss_t))
    pub["xcsao"] = (xc_t, np.full(len(t), np.nan), np.isfinite(xc_t))
    for v in ("mist_a", "mist_b", "eem_a", "eem_b"):
        tt, aa = f(t[f"{v}_teff50"]), f(t[f"{v}_av50"])
        for src, (pt, pa, ok) in pub.items():
            for lab, sel in (("all", ok), ("pms", ok & pms), ("ms", ok & ~pms)):
                out.setdefault(f"cmp_{v}", {})[f"{src}_{lab}"] = dict(
                    dteff_frac=stats((tt - pt)[sel] / pt[sel]), dav=stats((aa - pa)[sel]))
    # internal: MIST - EEM, config b and a; a - b
    out["mist_minus_eem"] = {c: dict(dav=stats(f(t[f"mist_{c}_av50"]) - f(t[f"eem_{c}_av50"])),
                                     dteff_frac=stats(f(t[f"mist_{c}_teff50"]) / f(t[f"eem_{c}_teff50"]) - 1))
                             for c in ("a", "b")}
    out["a_minus_b_mist"] = dict(dav=stats(f(t["mist_a_av50"]) - f(t["mist_b_av50"])),
                                 dteff_frac=stats(f(t["mist_a_teff50"]) / f(t["mist_b_teff50"]) - 1))
    out["sweeps_b"] = dict(
        feh_m025=dict(dav=stats(f(t["mist_b_fehm025_av50"]) - f(t["mist_b_av50"])),
                      dteff_frac=stats(f(t["mist_b_fehm025_teff50"]) / f(t["mist_b_teff50"]) - 1)),
        floor2_sd_ratio=stats(f(t["mist_b_floor2_sd_av"]) / f(t["mist_b_sd_av"])))
    # precision per G bin
    prec = {}
    for v in ("mist_a", "mist_b", "mist_c"):
        for lo, hi in ((-np.inf, 14), (14, 17), (17, np.inf)):
            sel = (G >= lo) & (G < hi) & np.isfinite(f(t[f"{v}_sd_av"]))
            prec[f"{v}_G{lo}-{hi}"] = dict(
                n=int(sel.sum()), med_sd_av=round(float(np.median(f(t[f"{v}_sd_av"])[sel])), 3),
                med_sd_logt=round(float(np.median(f(t[f"{v}_sd_logt"])[sel])), 4),
                n_sd_logt_le_002=int((f(t[f"{v}_sd_logt"])[sel] <= 0.02).sum()),
                n_sd_av_le_015=int((f(t[f"{v}_sd_av"])[sel] <= 0.15).sum()),
                med_rho=round(float(np.median(f(t[f"{v}_rho"])[sel])), 2),
                med_chi2_per_band=round(float(np.median((f(t[f"{v}_chi2"]) / f(t[f"{v}_nb"]))[sel])), 2),
                n_far_mass_gt_01=int((f(t[f"{v}_far_mass"])[sel] > 0.1).sum()))
    out["precision"] = prec
    (HERE / "compare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("cmp_") and not k.startswith("anchors")},
                     indent=1))
    for k in ("cmp_mist_a", "cmp_mist_b", "cmp_eem_b"):
        print(k)
        for kk, vv in out[k].items():
            if vv["dteff_frac"].get("n", 0):
                print(f"   {kk:16s} dTeff/T {vv['dteff_frac']}  dAV {vv['dav']}")
    for c in ("a", "b"):
        print("anchors", c, out[f"anchors_{c}"]["n"], out[f"anchors_{c}"]["median"])
        for row in out[f"anchors_{c}"]["values"]:
            print("    ", row)


if __name__ == "__main__":
    main()
