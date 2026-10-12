#!/usr/bin/env python3
"""Interval phase on the K x W model (hub finding ``isochrone-intervals-nuts.md`` §2, pre-registered
before any of these fits): prepare one fit, i.e. the synthetic table, the control estimator
(``aniso.search8`` + Laplace sd of the arm's log posterior) and the NUTS chain starts on THIS table.

    kw_cells.py prep --cell A --rep k      truth drawn around the K x W mode (fits/K_W.json)
    kw_cells.py prep --cell C --rep k      ctl-off's age, [Fe/H] and A_V, sigma_int 0.04, f_bg 0, k interior
    kw_cells.py prep --cell real           the 201 members in G 14-19
    kw_cells.py selftest                   draw_table reproduces aniso.draw draw for draw
    kw_cells.py zstarts --feh X            the Z arm: kw_runs/realZ_<X>/ from real_0 without feh
                                           (then ZFIX=X run_kw_nuts.sh realZ_<X>)

Writes ``kw_runs/<cell>_<rep>/{data.ecsv, prep.json, starts.json}``. The table carries the real
per-star errors UNSCALED (the likelihood multiplies them by k); ``aniso.draw`` returns them scaled,
so ``draw_table`` is ``aniso.draw`` with both error columns kept before scaling (checked in
``selftest``). Cell C also runs the control with the lattice's A_V start at 0.5 instead of 1.0
(R32-21: ``search`` starts (dm, A_V) at (dm_mu, 1.0), which is C's true A_V).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402
import starts8 as S8  # noqa: E402

C = A.C
RUNS = HERE / "kw_runs"
ARM, WIN = "K", "W"
KW = json.loads((HERE / "fits" / "K_W.json").read_text())["mode"]
PYMC = S8.PYMC
SEED0 = {"A": 6000, "C": 7000}


def truth_for(cell: str, rep: int, rng) -> dict:
    dm = float(rng.normal(10.223, 0.026))
    if cell == "A":  # around the K x W mode, off every MIST node (0.05 dex, 0.25 in [Fe/H]) a.s.
        t = {"feh": float(rng.uniform(-0.60, -0.30)), "loga": float(rng.uniform(5.87, 6.07)),
             "dm": dm, "Av": KW["Av"], "sG": KW["sG"], "k": KW["k"], "f_bg": KW["f_bg"]}
    elif cell == "C":  # ctl-off's world in the PMS window; k kept interior (U(1, 6) prior)
        t = {"feh": 0.15, "loga": 6.43, "dm": dm, "Av": 1.0, "sG": 0.04, "k": KW["k"], "f_bg": 0.0}
    else:
        raise ValueError(cell)
    return A._fill(ARM, t, 0.0)


def draw_table(f, gen, th: dict, n: int, rng, real, window):
    """``aniso.draw`` (same random stream, same catalogue) returning the QTable with the per-star
    errors before the k scaling, as the likelihood expects them."""
    Greal = np.asarray(real["Gmag"], float)
    Creal = np.asarray(real["G_BPmag"], float) - np.asarray(real["G_RPmag"], float)
    eGr = np.asarray(real["e_Gmag"], float)
    eCr = np.asarray(real["e_BP_RP"], float)
    lo, hi = window
    inw = (Greal >= lo) & (Greal <= hi)
    Gw, Cw = Greal[inw], Creal[inw]
    kk = th.get("k", 1.0)
    beta = th.get("beta", 0.0)
    n_bg = int(rng.binomial(n, th["f_bg"]))
    n_cl = n - n_bg
    fl2 = f.SIGMA_FLOOR**2
    kG, kc = f._kG, f._k_col1
    Gs, Cs, eGs, eCs = [], [], [], []
    got = 0
    for _ in range(400):
        if got >= n_cl:
            break
        Gt, Ct = C._true_stars(gen, th["feh"], th["loga"], th["dm"], th["Av"], 4 * n, rng)
        eG0, eC0 = C._real_errors(Greal, eGr, eCr, Gt, rng)
        eG, eC = kk * eG0, kk * eC0
        dA = rng.normal(size=Gt.size) * th["tau"]
        Go = Gt + kG * dA + rng.normal(size=Gt.size) * np.sqrt(eG**2 + fl2 + th["sG"] ** 2)
        Co = Ct + kc * dA + rng.normal(size=Gt.size) * np.sqrt(eC**2 + fl2 + th["sC"] ** 2)
        keep = (Go >= lo) & (Go <= hi)
        Gs.append(Go[keep])
        Cs.append(Co[keep])
        eGs.append(eG0[keep])
        eCs.append(eC0[keep])
        got += int(keep.sum())
    G = np.concatenate(Gs)[:n_cl]
    Cc = np.concatenate(Cs)[:n_cl]
    eGc = np.concatenate(eGs)[:n_cl]
    eCc = np.concatenate(eCs)[:n_cl]
    span = Gw.max() - Gw.min()
    u = rng.uniform(size=n_bg)
    if abs(beta) < 1e-4:
        Gb = Gw.min() + u * span
    else:
        Gb = Gw.min() + np.log1p(u * np.expm1(beta * span)) / beta
    Cb = rng.uniform(Cw.min(), Cw.max(), n_bg)
    eGb, eCb = C._real_errors(Greal, eGr, eCr, Gb, rng)
    tab = C.table(np.concatenate([G, Gb]), np.concatenate([Cc, Cb]), np.concatenate([eGc, eGb]),
                  np.concatenate([eCc, eCb]))
    return tab, {"n_bg": n_bg, "n_cluster": n_cl}


def logpost(f, x):
    ll = f.compiled8(ARM)(np.asarray(x, float))[0]
    return ll, ll - 0.5 * ((x[2] - f.dm_mu) / f.dm_sigma) ** 2


def control(f, start_av=1.0, seed=11):
    res = A.search8(f, ARM, verbose=False, start_av=start_av)
    st = S8.laplace_starts(f, ARM, res["x"], seed=seed)
    sd = st["laplace_sd_all"]
    ctrl = {"mode": {PYMC[k]: v for k, v in zip(A.free_names(ARM), res["x"], strict=True)},
            "x": res["x"], "laplace_sd": sd, "at_bound": res["at_bound"], "logpost": res["logpost"],
            "runner_up": res["runner_up"], "seconds": res["seconds"]}
    return ctrl, st


def prep(cell: str, rep: int) -> None:
    import erotica

    t0 = time.time()
    out = RUNS / (f"{cell}_{rep}" if cell != "real" else "real_0")
    out.mkdir(parents=True, exist_ok=True)
    real = C.real_table()
    info = {"cell": cell, "rep": rep, "arm": ARM, "window": WIN, "erotica_file": erotica.__file__,
            "started": C.now()}
    names = A.free_names(ARM)
    if cell == "real":
        data, truth = real, None
    else:
        seed = SEED0[cell] + rep
        rng = np.random.default_rng(seed)
        truth = truth_for(cell, rep, rng)
        gen = A.make_fitter(real, "F")
        fw = A.make_fitter(real, WIN)
        data, gi = draw_table(fw, gen, truth, int(fw._N_obs), rng, real, A.WINDOW[WIN])
        info.update({"seed": seed, "truth": truth,
                     "truth_pymc": {PYMC[k]: truth[k] for k in names}, **gi})
    data.write(out / "data.ecsv", overwrite=True)
    f = A.make_fitter(data, WIN)
    info["n_window"] = int(f._N_obs)
    ctrl, st = control(f, 1.0, 11 + rep)
    info["control"] = ctrl
    if cell == "C":  # R32-21: the same control with the lattice's A_V start away from the truth
        alt, _ = control(f, 0.5, 11 + rep)
        info["control_startav05"] = {k: alt[k] for k in ("mode", "x", "logpost", "at_bound")}
    if truth is not None:
        xt = [truth[k] for k in names]
        lt, pt_ = logpost(f, xt)
        lm, pm_ = logpost(f, ctrl["x"])
        info["truth_vs_mode"] = {"loglike_truth": lt, "loglike_mode": lm,
                                 "logpost_truth": pt_, "logpost_mode": pm_}
    C.dump(out / "starts.json", st)
    info["seconds"] = round(time.time() - t0, 1)
    C.dump(out / "prep.json", info)
    m = ctrl["mode"]
    print(cell, rep, {k: round(v, 4) for k, v in m.items()},
          {k: round(ctrl["laplace_sd"][k], 4) for k in ("feh", "loga")}, ctrl["at_bound"],
          info.get("truth_vs_mode"), info["seconds"], flush=True)


def selftest() -> dict:
    real = C.real_table()
    gen = A.make_fitter(real, "F")
    fw = A.make_fitter(real, WIN)
    th = truth_for("A", 0, np.random.default_rng(1))
    n = int(fw._N_obs)
    G1, C1, ncl1, eC1 = A.draw(fw, gen, th, n, np.random.default_rng(5), real, A.WINDOW[WIN], True)
    tab, gi = draw_table(fw, gen, th, n, np.random.default_rng(5), real, A.WINDOW[WIN])
    out = {"n": n, "max_dG": float(np.max(np.abs(np.asarray(tab["Gmag"]) - G1))),
           "max_dC": float(np.max(np.abs(np.asarray(tab["G_BPmag"]) - C1))),
           "max_d_scaled_eC": float(np.max(np.abs(th["k"] * np.asarray(tab["e_BP_RP"]) - eC1))),
           "n_cluster_equal": bool(ncl1 == gi["n_cluster"])}
    print(out)
    return out


def zstarts(feh: float) -> Path:
    """The Z arm's run dir: the data and real_0's chain starts with the "feh" entry dropped (the
    chains then find the conditional mode along the other parameters; starts stay overdispersed)."""
    src = RUNS / "real_0"
    out = RUNS / f"realZ_{feh:+.3f}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.ecsv").write_bytes((src / "data.ecsv").read_bytes())
    st = json.loads((src / "starts.json").read_text())
    st["chain_starts"] = [{k: v for k, v in c.items() if k != "feh"} for c in st["chain_starts"]]
    st["local_sd"] = {k: v for k, v in st["local_sd"].items() if k != "feh"}
    st["zfix"] = feh
    C.dump(out / "starts.json", st)
    C.dump(out / "prep.json", {"cell": "realZ", "rep": 0, "zfix": feh, "from": str(src),
                               "started": C.now()})
    print(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "selftest", "zstarts"])
    ap.add_argument("--feh", type=float)
    ap.add_argument("--cell", choices=["A", "C", "real"])
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "selftest":
        selftest()
    elif a.stage == "zstarts":
        zstarts(a.feh)
    else:
        prep(a.cell, a.rep)
