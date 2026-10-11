"""Per-member photometry, errors and band masks for the SED fit (pre-registered rules, §0.3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.table import Table, join

from sedgrid import BANDS, cstar

HERE = Path(__file__).resolve().parent
MEMBERS = Path("/Users/notluquis/phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
FLOOR = np.array([0.02, 0.02, 0.02, 0.03, 0.03, 0.03])
CONFIGS = {"a": [1, 1, 1, 1, 1, 1], "b": [1, 0, 0, 1, 1, 1], "c": [0, 0, 0, 1, 1, 1]}


def f(col):
    return np.ma.asarray(col).astype(float).filled(np.nan)


def load():
    m = Table.read(MEMBERS)
    m.rename_column("GaiaDR3", "source_id")
    g = Table.read(HERE / "members_gaia_ap.ecsv")
    t = join(m, g, keys="source_id", join_type="left")
    n = len(t)
    mag = np.full((n, 6), np.nan)
    err = np.full((n, 6), np.nan)
    mag[:, 0], err[:, 0] = f(t["Gmag"]), f(t["e_Gmag"])
    mag[:, 1], err[:, 1] = f(t["BPmag"]), f(t["e_BPmag"])
    mag[:, 2], err[:, 2] = f(t["RPmag"]), f(t["e_RPmag"])
    for k, b in enumerate(["J", "H", "Ks"]):
        src = np.asarray(t[f"src_{b}"]).astype(str)
        ok = ~np.isin(src, ["none", "sat"])
        mag[:, 3 + k] = np.where(ok, f(t[f"{b}_best"]), np.nan)
        err[:, 3 + k] = np.where(ok, f(t[f"e{b}_best"]), np.nan)
    have = np.isfinite(mag) & np.isfinite(err) & (err > 0)

    # IR excess: Ks out for any flag; H out too for Class II / flat spectrum (pre-registered)
    spicy = np.asarray(t["spicy_class"]).astype(str)
    yq = np.asarray(t["yso_Q"]).astype(str) == "True"
    ya = np.asarray(t["yso_alpha"]).astype(str) == "True"
    disk = np.isin(spicy, ["ClassII", "FS"])
    have[:, 5] &= ~(yq | ya | disk)
    have[:, 4] &= ~disk

    # BP/RP excess: |C*| > 3 sigma_C* -> add |C*| in quadrature to BP and RP
    cs, sc = cstar(mag[:, 0], mag[:, 1] - mag[:, 2], f(t["E(BP/RP)"]))
    bad = np.abs(cs) > 3 * sc
    infl = np.where(bad, np.abs(cs), 0.0)
    err[:, 1] = np.hypot(err[:, 1], infl)
    err[:, 2] = np.hypot(err[:, 2], infl)
    flags = dict(disk=disk, irx_flag=yq | ya | disk, cstar=cs, cstar_bad=bad)
    return t, mag, err, have, flags


def sigma(err, floor_scale=1.0):
    return np.sqrt(err ** 2 + (floor_scale * FLOOR) ** 2)


def mask(have, config):
    return have & np.array(CONFIGS[config], bool)[None, :]


__all__ = ["BANDS", "load", "sigma", "mask", "CONFIGS", "FLOOR"]
