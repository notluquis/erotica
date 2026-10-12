#!/usr/bin/env python3
"""POST HOC per-chain convergence clause (replaces the split R-hat clause that gate_noise.json
showed cannot pass): for each chain c and parameter, z = (mean_c - mean_rest) / sqrt(MCSE_c^2 +
MCSE_rest^2), MCSE = sd / sqrt(ESS) with ESS from az.ess (bulk) of chain c (split in two halves)
and of the other chains. Fails if any |z| > Z_MAX over 6 parameters x 4 chains (Bonferroni,
24 tests, ~0.02 two-sided); with more parameters (K x W: + k, 28 tests) the threshold is
recomputed at the same family level (``z_max``). Validated here both ways: a perfect sampler (stationary AR(1) at the
measured per-chain ESS) and one chain shifted by 0.5 sd.

    chain_z.py selftest           -> chain_z_selftest.json
    chain_z.py runs/A_0 runs/A_1  -> prints and writes runs/<r>/chain_z.json
"""
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
from scipy.stats import norm

PARAMS = ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")
Z_MAX = 3.3
# the family level of the original 24 tests (6 parameters x 4 chains), ~0.023 two-sided; with the
# K x W arm's 7 parameters (+ k) the threshold is recomputed at the same family level
ALPHA_FAM = 24 * 2 * norm.sf(Z_MAX)


def z_max(n_tests):
    return float(norm.isf(ALPHA_FAM / (2 * n_tests)))


def zs(x):
    out = []
    for c in range(x.shape[0]):
        xc = x[c]
        rest = np.delete(x, c, 0)
        e_c = float(az.ess(xc[: len(xc) // 2 * 2].reshape(2, -1), method="bulk"))
        e_r = float(az.ess(rest, method="bulk"))
        m_c, m_r = xc.mean(), rest.mean()
        se = np.sqrt(xc.var() / e_c + rest.var() / e_r)
        out.append(float((m_c - m_r) / se))
    return out


def gate(post, params=PARAMS):
    z = {p: zs(post[p]) for p in params}
    zmax = max(abs(v) for p in params for v in z[p])
    thr = z_max(len(params) * next(iter(post.values())).shape[0])
    return {"z": z, "z_abs_max": zmax, "z_thr": thr, "pass": bool(zmax <= thr)}


def ar1(rng, phi, n=500, k=4):
    e = rng.normal(size=(k, n))
    x = np.empty_like(e)
    x[:, 0] = e[:, 0] / np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[:, t] = phi * x[:, t - 1] + e[:, t]
    return x * np.sqrt(1 - phi**2)  # unit marginal sd


def selftest(params=PARAMS, out="chain_z_selftest.json"):
    rng = np.random.default_rng(3)
    res = {"params": list(params), "z_thr": z_max(4 * len(params))}
    for ess in (210, 450):
        r = ess / 500
        phi = (1 - r) / (1 + r)
        ff, det = [], []
        for _ in range(300):
            post = {p: ar1(rng, phi) for p in params}
            ff.append(not gate(post, params)["pass"])
            post["loga"][1] += 0.5  # one chain off by 0.5 marginal sd
            det.append(not gate(post, params)["pass"])
        res[str(ess)] = {"false_fail": float(np.mean(ff)), "detects_0p5sd_shift": float(np.mean(det)), "n": 300}
        print(ess, res[str(ess)], flush=True)
    Path(__file__).with_name(out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "selftest":
        selftest()
    elif sys.argv[1] == "selftest7":  # the K x W arm: PARAMS + k
        selftest(PARAMS + ("k",), "chain_z_selftest_k.json")
    else:
        for r in sys.argv[1:]:
            d = np.load(Path(r) / "draws.npz")
            g = gate({p: d[p] for p in d.files})  # every parameter the fit saved (K x W: + k)
            (Path(r) / "chain_z.json").write_text(json.dumps(g, indent=1) + "\n")
            print(r, round(g["z_abs_max"], 2), g["pass"], {p: [round(v, 2) for v in g["z"][p]] for p in ("loga", "feh")})
