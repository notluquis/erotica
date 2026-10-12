#!/usr/bin/env python3
"""POST HOC (hub finding isochrone-intervals-nuts.md §1.6): does the sigma_int HalfNormal(0.05) and
f_bg Beta(1, 19) prior explain cell A's NUTS under-coverage? Importance-reweight each replicate's
draws to flat priors on sigma_int and f_bg (w = 1 / (p_HN(sigma) p_Beta(f))), weighted quantiles,
Kish ESS. Does not change the pre-registered verdict. Writes reweight_flat.json."""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def wq(x, w, q):
    o = np.argsort(x)
    c = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, c, x[o]))


out = {}
for k in range(8):
    r = HERE / "runs" / f"A_{k}"
    d = np.load(r / "draws.npz")
    t = json.loads((r / "prep.json").read_text())["truth"]
    s = d["sigma_int"].ravel(); f = d["f_bg"].ravel()
    lw = s**2 / (2 * 0.05**2) - 18 * np.log1p(-f)
    w = np.exp(lw - lw.max())
    kish = float(w.sum() ** 2 / (w**2).sum())
    row = {"kish_ess": kish, "n": int(w.size)}
    for p in ("loga", "feh"):
        x = d[p].ravel()
        q = {qq: wq(x, w, qq) for qq in (0.05, 0.16, 0.5, 0.84, 0.95)}
        q0 = {qq: float(np.quantile(x, qq)) for qq in (0.16, 0.5, 0.84)}
        row[p] = {"truth": t[p], "med_prior": q0[0.5], "med_flat": q[0.5],
                  "in68_prior": bool(q0[0.16] <= t[p] <= q0[0.84]),
                  "in68_flat": bool(q[0.16] <= t[p] <= q[0.84]), "in90_flat": bool(q[0.05] <= t[p] <= q[0.95])}
    row["sigma_med_prior"], row["sigma_med_flat"] = float(np.median(s)), wq(s, w, 0.5)
    row["fbg_med_prior"], row["fbg_med_flat"] = float(np.median(f)), wq(f, w, 0.5)
    out[f"A_{k}"] = row
    print(k, round(kish), {p: (round(row[p]["med_prior"] - row[p]["truth"], 3), round(row[p]["med_flat"] - row[p]["truth"], 3), row[p]["in68_prior"], row[p]["in68_flat"], row[p]["in90_flat"]) for p in ("loga", "feh")}, round(row["sigma_med_prior"], 3), round(row["sigma_med_flat"], 3))
for p in ("loga", "feh"):
    out[f"cov68_flat_{p}"] = sum(out[f"A_{k}"][p]["in68_flat"] for k in range(8))
    out[f"cov90_flat_{p}"] = sum(out[f"A_{k}"][p]["in90_flat"] for k in range(8))
    print(p, "flat 68:", out[f"cov68_flat_{p}"], "/8  90:", out[f"cov90_flat_{p}"], "/8")
(HERE / "reweight_flat.json").write_text(json.dumps(out, indent=1) + "\n")
