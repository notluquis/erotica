#!/usr/bin/env python3
"""External oracle of the grid layer: lambda Ori (Cao et al. 2022, 2022ApJ...924...84C).

WHY THIS EXISTS
---------------
The PMS controls (SPOTS, BHAC15) enter the grid layer to measure how much the age of a young
cluster moves with the stellar physics. Cao+22 measured exactly that in lambda Ori: Class III mean
ages 2.4 Myr (MIST, SPOTS f=0, DSEP), 2.5 (BHAC15), 3.9 +- 0.2 (SPOTS f=0.34) [their Table 2, read
2026-10-04]. If the module, fed the same stars, does not reproduce the DIFFERENCE between grids, the
NGC 6383 control would hide a defect. The pre-registration (tolerances, sample, primary/secondary)
is the hub file ``agent-findings/isochrone-grids-lambda-ori-prereg.md``, committed before this ran.

DESIGN (as pre-registered)
--------------------------
* Sample: Cao+22 table1 (J/ApJ/924/84) Class III stars with a published age, x 2MASS PSC (J, Ks) and
  Gaia DR3 (parallax, Lindegren+21 zero point) -- ``agent-findings/scripts/isochrone_grids_lambda_ori_data.py``.
* Each star dereddened with ITS published A_V (Cao's SED / G-RP methods, independent of the
  evolutionary grids) and moved to the cluster's reference distance with its own parallax when
  plx/e_plx > 10 and RUWE < 1.4 (Cao use per-star parallaxes); the fit keeps a free A_V in
  [-0.3, 0.3] and dm ~ N(dm_ref, 0.03) for the residuals.
* Bands 2MASS J, J-Ks (SPOTS has J/Ks down to 0.10 Msun; its Gaia colours stop at 0.55 Msun at
  f=0.34). Window: ``safe_window`` over all grids and the priors, never from the stars.
* Grids: MIST v1.2 ([Fe/H] = 0 node), BHAC15, SPOTS f = 0, 0.17, 0.34. Same likelihood, same
  window, same stars. Primary without binaries (Cao do not model them); with binaries secondary.
* Estimates: maximum likelihood (``find_start``) with Laplace widths (labelled as such).

WHAT WOULD FALSIFY (pre-registered; see the hub file for the numbers)
--------------------------------------------------------------------
P1: Delta log t (SPOTS 0.34 - SPOTS 0) off the published 0.211 dex by more than 2 sigma.
N1/N2: SPOTS 0 - MIST, BHAC15 - MIST off the published 0.000 / 0.018 by more than 2 sigma.
S1 (secondary): any absolute age off its published value by > 0.5 Myr.

Output: ``lambda_ori_oracle.json`` (``_binaries`` for the secondary).

POST-UNBLINDING DIAGNOSTICS (added after the primary failed; labelled as such)
-----------------------------------------------------------------------------
D1 ``--stage profile``: log L(log t) with A_V = 0 and dm = dm_ref fixed (are the bounds deciding?).
D2 ``--stage hrd``: the same likelihood and stars in Cao's HR space (their log L, log T_eff), with
   "bands" built from each grid's log L / log T_eff -- no colour table between models and stars.
D3 ``lambda_ori_colour_check.py``: observed J-Ks minus each grid's J-Ks at the star's published
   T_eff (log t ~ 6.5 node).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
CACHE = Path.home() / ".cache/erotica-grids"
DATA = CACHE / "oracles/lambda_ori/lambda_ori_2mass_dr3.ecsv"
MIST12 = Path.home() / "erotica/data/test/NGC6383/MIST/UBVRIplus"
EFFL = (12350.0, 12350.0, 21590.0)  # 2MASS J, J, Ks isophotal wavelengths (Cohen+03) [I]
PRI = dict(loga_range=(6.0, 7.0), Av_range=(-0.3, 0.3), dm_sigma=0.03)


def grids() -> dict:
    from erotica.analysis import grids as G

    lr = (5.9, 7.1)
    return {
        "MIST v1.2": (
            G.MISTGrid(MIST12, bands=("2MASS_J", "2MASS_Ks"), loga_range=lr).select(feh=0.0),
            ("2MASS_J", "2MASS_J", "2MASS_Ks"),
        ),
        "BHAC15": (
            G.BHAC15Grid(CACHE / "bhac15/BHAC15_iso.2mass", bands=("Mj", "Mk"), loga_range=lr),
            ("Mj", "Mj", "Mk"),
        ),
        "SPOTS f=0": (
            G.SPOTSGrid(CACHE / "spots", 0.0, bands=("J_mag", "K_mag"), loga_range=lr),
            ("J_mag", "J_mag", "K_mag"),
        ),
        "SPOTS f=0.17": (
            G.SPOTSGrid(CACHE / "spots", 0.17, bands=("J_mag", "K_mag"), loga_range=lr),
            ("J_mag", "J_mag", "K_mag"),
        ),
        "SPOTS f=0.34": (
            G.SPOTSGrid(CACHE / "spots", 0.34, bands=("J_mag", "K_mag"), loga_range=lr),
            ("J_mag", "J_mag", "K_mag"),
        ),
    }


def sample() -> tuple:
    from astropy.table import QTable, Table

    from erotica.analysis._isochrone import _ccm89

    t = Table.read(DATA)
    cls = np.asarray(t["Class"].filled("") if hasattr(t["Class"], "filled") else t["Class"])
    age = t["logAge"]
    has_age = ~np.asarray(age.mask) if hasattr(age, "mask") else np.isfinite(age)
    t = t[(cls == "III") & has_age]
    kJ, kK = _ccm89(EFFL[0]), _ccm89(EFFL[2])
    plx = np.asarray(t["plx_zpc"], float)
    eplx = np.asarray(
        t["e_Plx"].filled(np.nan) if hasattr(t["e_Plx"], "filled") else t["e_Plx"], float
    )
    ruwe = np.asarray(
        t["RUWE"].filled(np.nan) if hasattr(t["RUWE"], "filled") else t["RUWE"], float
    )
    good = (plx / eplx > 10) & (ruwe < 1.4)
    dm_ref = float(5 * np.log10(100.0 / np.median(plx[good])))
    dm_i = np.where(good, 5 * np.log10(100.0 / np.where(good, plx, 1.0)), dm_ref)
    e_dm = np.where(good, 5 / np.log(10) * eplx / np.where(good, plx, 1.0), 0.0)
    av = np.asarray(t["Av"], float)
    eav = np.asarray(t["e_Av"].filled(0.0) if hasattr(t["e_Av"], "filled") else t["e_Av"], float)
    J = np.asarray(t["Jmag"], float) - kJ * av - (dm_i - dm_ref)
    K = np.asarray(t["Kmag"], float) - kK * av - (dm_i - dm_ref)
    eJ = np.hypot(np.hypot(np.asarray(t["e_Jmag"], float), e_dm), kJ * eav)
    eK = np.hypot(np.asarray(t["e_Kmag"], float), (kK) * eav)
    ecol = np.hypot(
        np.hypot(np.asarray(t["e_Jmag"], float), np.asarray(t["e_Kmag"], float)), (kJ - kK) * eav
    )
    q = QTable(
        {
            "J0": J,
            "K0": K,
            "e_J": eJ,
            "e_K": eK,
            "e_BP_RP": ecol,
            "probability_hdbscan": np.ones(len(J)),
            "logAge_cao": np.asarray(age[...], float)[(cls == "III") & has_age]
            if False
            else np.asarray(t["logAge"], float),
            "Mstar_cao": np.asarray(t["Mstar"], float),
        }
    )
    meta = {
        "n_classIII_with_age": int(len(t)),
        "n_per_star_parallax": int(good.sum()),
        "dm_ref": dm_ref,
        "kJ": kJ,
        "kK": kK,
    }
    return q, meta


def hrd_view(g):
    """Post-unblinding diagnostic D2: the same grid with "bands" built from log L and log T_eff, so
    the population likelihood runs in Cao's HR-diagram space with no colour table or BC between
    the models and the stars: Mbol = 4.74 - 2.5 log L, colour = -10 log T_eff."""
    import copy

    from erotica.analysis.grids.base import GridNode

    h = copy.copy(g)
    h._nodes = {
        k: GridNode(
            n.eep,
            n.mass,
            {
                "Mbol": 4.74 - 2.5 * n.extra["logl"],
                "T": -10.0 * n.extra["logte"],
                "zero": 0.0 * n.mass,
            },
            n.extra,
        )
        for k, n in g._nodes.items()
    }
    h.bands = h.default_bands = ("Mbol", "T", "zero")
    h.name = g.name + " [HRD]"
    return h


def main() -> None:
    from erotica.analysis._isochrone import IsochroneFitter, _ccm89
    from erotica.analysis.grids import safe_window

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["window", "fit", "profile", "hrd"], default="window")
    ap.add_argument("--binaries", action="store_true")
    a = ap.parse_args()
    q, meta = sample()
    G = grids()
    win = safe_window(
        [g for g, _ in G.values()],
        [b[0] for _, b in G.values()],
        loga_range=PRI["loga_range"],
        dm_range=(meta["dm_ref"] - 0.15, meta["dm_ref"] + 0.15),
        Av_range=PRI["Av_range"],
        k_mag=_ccm89(EFFL[0]),
        data_faint=float(np.max(q["J0"])),
    )
    inw = (q["J0"] >= win[0]) & (q["J0"] <= win[1])
    la = np.asarray(q["logAge_cao"])[inw]
    pub_in = float(10 ** la.mean() / 1e6)
    meta.update(
        {
            "window_J": list(win),
            "n_in_window": int(inw.sum()),
            "cao_spots034_geomean_in_window_Myr": pub_in,
            "cao_spots034_sem_dex_in_window": float(la.std() / np.sqrt(len(la))),
            "mass_range_in_window": [
                float(np.min(np.asarray(q["Mstar_cao"])[inw])),
                float(np.max(np.asarray(q["Mstar_cao"])[inw])),
            ],
        }
    )
    print(json.dumps(meta, indent=1), flush=True)
    if a.stage == "window":
        return
    if a.stage == "profile":  # D1, post-unblinding: log L(log t) with A_V = 0 and dm = dm_ref fixed
        res = {}
        for name, (g, bands) in G.items():
            f = IsochroneFitter(
                grid=g,
                magnitude=bands[0],
                color=(bands[1], bands[2]),
                magnitude_effl=EFFL[0],
                color_effl=(EFFL[1], EFFL[2]),
                obs_columns=("J0", "J0", "K0"),
                obs_error_columns=("e_J", "e_J", "e_K"),
                alpha=0.0,
                beta=0.0,
                dm_mu=meta["dm_ref"],
                dm_range=(meta["dm_ref"] - 0.15, meta["dm_ref"] + 0.15),
                **PRI,
            )
            f.setup(q, prob_threshold=0.0, mag_window=win)
            ages = np.linspace(6.0, 7.0, 101)
            ll = [
                max(f.loglike(0.0, x, meta["dm_ref"], 0.0, s, 0.03) for s in (0.02, 0.04, 0.08))
                for x in ages
            ]
            res[name] = {
                "loga_best": float(ages[int(np.argmax(ll))]),
                "loglike": ll,
                "ages": ages.tolist(),
            }
            print(name, "D1 best log t at A_V=0, dm=dm_ref:", res[name]["loga_best"], flush=True)
        (HERE / "lambda_ori_oracle_D1_profile.json").write_text(json.dumps(res) + "\n")
        return
    if (
        a.stage == "hrd"
    ):  # D2, post-unblinding: Cao's own logL, logTeff, same stars, same likelihood
        from astropy.table import QTable, Table

        t = Table.read(DATA)
        cls = np.asarray(t["Class"].filled("") if hasattr(t["Class"], "filled") else t["Class"])
        has = (
            ~np.asarray(t["logAge"].mask) if hasattr(t["logAge"], "mask") else np.ones(len(t), bool)
        )
        t = t[(cls == "III") & has]
        t = t[np.asarray(inw)]
        lL, lT = np.asarray(t["logLstar"], float), np.asarray(t["logTeff"], float)
        elL = np.asarray(t["e_logLstar"].filled(0.05), float)
        elT = np.asarray(t["e_logTeff"].filled(0.01), float)
        h = QTable(
            {
                "M": 4.74 - 2.5 * lL,
                "T": -10.0 * lT,
                "Z": np.zeros(len(t)),
                "e_M": 2.5 * elL,
                "e_T": 10.0 * elT,
                "e_Z": np.zeros(len(t)),
                "probability_hdbscan": np.ones(len(t)),
            }
        )
        h["e_BP_RP"] = h["e_T"]
        hw = safe_window(
            [hrd_view(g) for g, _ in G.values()],
            "Mbol",
            loga_range=PRI["loga_range"],
            dm_range=(-0.01, 0.01),
            Av_range=(0.0, 1e-6),
            k_mag=0.0,
            data_faint=float(np.max(h["M"])),
        )
        res = {
            "window_Mbol": list(hw),
            "n": int(((h["M"] >= hw[0]) & (h["M"] <= hw[1])).sum()),
            "fits": {},
        }
        for name, (g, _) in G.items():
            f = IsochroneFitter(
                grid=hrd_view(g),
                obs_columns=("M", "T", "Z"),
                obs_error_columns=("e_M", "e_T", "e_Z"),
                alpha=0.0,
                beta=0.0,
                loga_range=PRI["loga_range"],
                Av_range=(0.0, 1e-6),
                dm_mu=0.0,
                dm_sigma=0.005,
                dm_range=(-0.01, 0.01),
            )
            f.setup(h, prob_threshold=0.0, mag_window=hw)
            s = f.find_start(2, np.random.default_rng(1))
            res["fits"][name] = {
                "loga": s["mode"]["loga"],
                "age_Myr": 10 ** s["mode"]["loga"] / 1e6,
                "loga_laplace_sd": s["local_sd"]["loga"],
                "sigma_int": s["mode"]["sigma_int"],
                "f_bg": s["mode"]["f_bg"],
                "n_stars": int(f._N_obs),
            }
            print(name, "D2 HRD", json.dumps(res["fits"][name]), flush=True)
        (HERE / "lambda_ori_oracle_D2_hrd.json").write_text(json.dumps(res, indent=1) + "\n")
        return
    out = {"meta": meta, "binaries": a.binaries, "fits": {}}
    for name, (g, bands) in G.items():
        t0 = time.time()
        kw = dict(
            PRI, dm_mu=meta["dm_ref"], dm_range=(meta["dm_ref"] - 0.15, meta["dm_ref"] + 0.15)
        )
        if not a.binaries:
            kw.update(alpha=0.0, beta=0.0)
        f = IsochroneFitter(
            grid=g,
            magnitude=bands[0],
            color=(bands[1], bands[2]),
            magnitude_effl=EFFL[0],
            color_effl=(EFFL[1], EFFL[2]),
            obs_columns=("J0", "J0", "K0"),
            obs_error_columns=("e_J", "e_J", "e_K"),
            **kw,
        )
        f.setup(q, prob_threshold=0.0, mag_window=win)
        s = f.find_start(2, np.random.default_rng(1))
        m = s["mode"]
        out["fits"][name] = {
            "loga": m["loga"],
            "age_Myr": 10 ** m["loga"] / 1e6,
            "loga_laplace_sd": s["local_sd"]["loga"],
            "dm": m["dm"],
            "Av": m["Av"],
            "sigma_int": m["sigma_int"],
            "f_bg": m["f_bg"],
            "loglike": s["loglike"],
            "runner_up": s["runner_up"],
            "n_stars": int(f._N_obs),
            "sigma_floor": f.SIGMA_FLOOR,
            "grid": g.describe(),
            "seconds": round(time.time() - t0, 1),
        }
        print(
            name,
            json.dumps(
                {k: v for k, v in out["fits"][name].items() if k not in ("grid", "runner_up")}
            ),
            flush=True,
        )
    sfx = "_binaries" if a.binaries else ""
    (HERE / f"lambda_ori_oracle{sfx}.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n"
    )


if __name__ == "__main__":
    main()
