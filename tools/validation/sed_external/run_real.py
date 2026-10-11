"""SED fit of the 254 real members (pre-registered §0.3), after the synthetic arms were read.

Variants (model, config, [Fe/H], floor scale):
  mist a / b / c at [Fe/H] 0, floor x1     (a: control only; b: the only prior candidate)
  eem  a / b at [Fe/H] 0                  (atmosphere systematic; EEM covers 2800-10 645 K only)
  mist b at [Fe/H] -0.25; mist b floor x2  (sweeps)
Spectral-type anchors: for members with a spectroscopic type, A_V with T_eff restricted to the
type's EEM T_eff +- one subtype (configuration a and b, MIST).
Writes sed_real_perstar.ecsv and sed_real.json.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from astropy.table import Table

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import sedgrid as sg  # noqa: E402

SPEC = Path("/Users/notluquis/phd/agent-findings/scripts/isochrone_external_spectra.json")
VARIANTS = [
    ("mist", "a", 0.0, 1.0), ("mist", "b", 0.0, 1.0), ("mist", "c", 0.0, 1.0),
    ("eem", "a", 0.0, 1.0), ("eem", "b", 0.0, 1.0),
    ("mist", "b", -0.25, 1.0), ("mist", "b", 0.0, 2.0),
]


def vname(model, config, feh, fl):
    s = f"{model}_{config}"
    if feh != 0.0:
        s += "_fehm025"
    if fl != 1.0:
        s += "_floor2"
    return s


def spt_teff():
    """Spectroscopic types of members -> (T_lo, T_hi) from the EEM table, +- one subtype."""
    e = sg.load_eem()
    spt, T = e["SpT"], e["Teff"]
    d = json.loads(SPEC.read_text())["stars"]
    out = {}
    for sid, v in d.items():
        types = []
        if "aidelman2018" in v:
            tt = float(v["aidelman2018"]["teff"])
            et = float(v["aidelman2018"]["e_teff"])
            out[int(sid)] = (tt - et, tt + et, "aidelman2018:" + v["aidelman2018"]["sptype"])
            continue
        if "rauw2010_spt" in v:
            lo, hi = v["rauw2010_spt"]["teff_range"]
            out[int(sid)] = (lo, hi, "rauw2010:" + v["rauw2010_spt"]["sptype"])
            continue
        for k in ("skiff_bmk", "simbad_spt"):
            for x in v.get(k, []):
                types.append((k, x["sptype"].strip()))
        for k, s in types:
            mm = re.match(r"([OBAFGKM])(\d(?:\.\d)?)", s)
            if not mm:
                continue
            code = mm.group(1) + (mm.group(2) if "." in mm.group(2) else mm.group(2))
            idx = [j for j, x in enumerate(spt) if re.match(rf"^{code}(\.0)?V", x)]
            if not idx:
                continue
            j = idx[0]
            lo, hi = sorted([T[min(j + 1, len(T) - 1)], T[max(j - 1, 0)]])
            out[int(sid)] = (lo, hi, f"{k}:{s}")
            break
    return out


def main():
    t, mag, err, have, flags = data.load()
    sid = np.asarray(t["source_id"], np.int64)
    res = Table()
    res["source_id"] = sid
    res["G"] = mag[:, 0]
    res["cstar_bad"] = flags["cstar_bad"]
    res["irx_flag"] = flags["irx_flag"]
    grids = {}
    marg = {}
    for model, config, feh, fl in VARIANTS:
        key = (model, feh)
        if key not in grids:
            grids[key] = sg.mist_fine(feh) if model == "mist" else sg.eem_fine(feh)[:2]
        logt, bc = grids[key]
        use_all = data.mask(have, config)
        s = data.sigma(err, fl)
        name = vname(model, config, feh, fl)
        cols = {k: np.full(len(t), np.nan) for k in
                ("teff16", "teff50", "teff84", "sd_logt", "av16", "av50", "av84", "sd_av", "rho",
                 "far_mass", "edge_t", "chi2", "nb")}
        marg[name] = np.zeros((len(t), sg.AV.size))
        for i in range(len(t)):
            if use_all[i].sum() < 3:
                continue
            p, cmin = sg.fit_star(bc, mag[i], s[i], use_all[i])
            marg[name][i] = p.sum(0)
            sm = sg.summarise(p, logt)
            for q in ("16", "50", "84"):
                cols["teff" + q][i] = 10 ** sm["logt" + q]
            for k in ("sd_logt", "av16", "av50", "av84", "sd_av", "rho", "far_mass", "edge_t"):
                cols[k][i] = sm[k]
            cols["chi2"][i], cols["nb"][i] = cmin, use_all[i].sum()
        for k, v in cols.items():
            res[f"{name}_{k}"] = v
        ok = np.isfinite(cols["av50"])
        print(f"{name}: {ok.sum()} fitted; median A_V {np.nanmedian(cols['av50']):.3f}; "
              f"median sd(A_V) {np.nanmedian(cols['sd_av']):.3f}; median sd(logT) "
              f"{np.nanmedian(cols['sd_logt']):.4f}; median chi2/nb {np.nanmedian(cols['chi2']/cols['nb']):.2f}")

    # spectral-type anchors (MIST, T_eff restricted to the type's range)
    anchors = spt_teff()
    logt, bc = grids[("mist", 0.0)]
    for config in ("a", "b"):
        use_all = data.mask(have, config)
        s = data.sigma(err)
        a16, a50, a84 = (np.full(len(t), np.nan) for _ in range(3))
        tlo, thi = np.full(len(t), np.nan), np.full(len(t), np.nan)
        src = np.full(len(t), "", dtype="U40")
        for i, sd in enumerate(sid):
            if sd not in anchors or use_all[i].sum() < 2:
                continue
            lo, hi, lab = anchors[sd]
            sel = (logt >= np.log10(lo)) & (logt <= np.log10(hi))
            if sel.sum() == 0:
                sel = np.zeros_like(sel)
                sel[np.argmin(np.abs(logt - np.log10(0.5 * (lo + hi))))] = True
            p, _ = sg.fit_star(bc[sel], mag[i], s[i], use_all[i])
            sm = sg.summarise(p, logt[sel])
            a16[i], a50[i], a84[i] = sm["av16"], sm["av50"], sm["av84"]
            src[i], tlo[i], thi[i] = lab, lo, hi
        res[f"anchor_{config}_av16"], res[f"anchor_{config}_av50"], res[f"anchor_{config}_av84"] = a16, a50, a84
        res["anchor_src"], res["anchor_tlo"], res["anchor_thi"] = src, tlo, thi
        n = np.isfinite(a50).sum()
        print(f"anchors {config}: {n} stars; A_V median {np.nanmedian(a50):.3f}")
    res.write(HERE / "sed_real_perstar.ecsv", overwrite=True)
    np.savez_compressed(HERE / "sed_real_av_marginals.npz", av=sg.AV, source_id=sid, **marg)
    print("wrote sed_real_perstar.ecsv")


if __name__ == "__main__":
    main()
