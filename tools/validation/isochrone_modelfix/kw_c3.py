#!/usr/bin/env python3
"""C3' of the hub finding ``isochrone-intervals-nuts.md`` §2.17 (pre-registered there before any
replicate was prepared): do unresolved binaries bias log t under the K x W model when the generator
does NOT share the likelihood's numerical treatment of them?

Why a new generator. ``iv_common._true_stars`` (used by cells A and C) draws binaries from the
likelihood's own ``_deposit`` sheet: the same EEP stride, the same q nodes, the same smoothed
gamma, so a coverage test with it cannot see an error in that treatment (§2.12, "Binarias (C3)").
This generator shares the PHYSICS (Chabrier 2014 IMF, Offner b(m), Duchene & Kraus step gamma(m),
companion floor at the lowest mass of the isochrone, flux sum in G, BP, RP) and none of the
NUMERICS: stars are drawn at a continuous primary mass (inverse CDF of the IMF over the isochrone's
mass range), the isochrone is read at fixed MASS (not fixed EEP) and blended bilinearly in
([Fe/H], log t) between the four bracketing nodes, q is continuous (``u**(1/(gamma+1))``) and the
pair's bands are summed per star. It is the construction of ``synthetic_cluster`` of C3 (§10.17,
H1-H3), on the MIST v1.2 grid layer of K x W.

Arms (same truths and seeds as cell A, 6000 + rep, so every replicate is paired with an A replicate):
    IB   binaries in the generator, binaries in the model (the production model)
    IS   singles in the generator, singles in the model (alpha = beta = 0): the control of the
         generator itself; it must show no offset, or the IB offset is not attributed to binaries
    IN   singles in the generator, the production model (binaries): how much a model that expects
         the Offner fraction moves log t when there are none (sensitivity, MAP only)
    IBL, ISL   IB and IS with 8 x 201 stars: the sampling scatter of the mode falls as 1/sqrt(8), so
         the offset of the estimator (the quantity C3 measured) is resolved (MAP only)

    kw_c3.py selftest                     closed-form and oracle checks of the generator
    kw_c3.py prep --arm IB --rep k        data + search8 control + Laplace + chain starts
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402
import kw_cells as K  # noqa: E402
import starts8 as S8  # noqa: E402

from erotica.analysis._isochrone import _chabrier2014_xi, _dk_gamma, _mag_combine  # noqa: E402

C = A.C
RUNS = HERE / "kw_runs"
# arm -> (binaries in the generator, binaries in the model, catalogue size in units of the real 201)
ARMS = {"IB": (True, True, 1), "IS": (False, False, 1), "IN": (False, True, 1),
        "IBL": (True, True, 8), "ISL": (False, False, 8)}


def _corner(iso, feh, loga):
    """(mass, G, BP, RP) at a node, ordered by mass."""
    _, mass, G, BP, RP = iso.get_isochrone_eep(float(feh), float(loga))
    o = np.argsort(mass, kind="stable")
    return mass[o], G[o], BP[o], RP[o]


def _bracket(nodes, x):
    nodes = np.asarray(nodes, float)
    k = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, len(nodes) - 2))
    w = float(np.clip((x - nodes[k]) / (nodes[k + 1] - nodes[k]), 0.0, 1.0))
    return k, w


class MassIsochrone:
    """Absolute G, BP, RP as functions of MASS at (feh, loga): the four bracketing nodes read at
    fixed mass (``np.interp``) and blended bilinearly. Defined on the mass range common to the four
    nodes."""

    def __init__(self, iso, feh, loga):
        zs, ages = np.asarray(iso._met_values, float), np.asarray(iso._loga_values, float)
        i, wz = _bracket(zs, feh)
        j, wa = _bracket(ages, loga)
        self.corners = []
        for di, wi in ((0, 1 - wz), (1, wz)):
            for dj, wj in ((0, 1 - wa), (1, wa)):
                if wi * wj > 0:
                    self.corners.append((wi * wj, _corner(iso, zs[i + di], ages[j + dj])))
        self.lo = max(c[1][0][0] for c in self.corners)
        self.hi = min(c[1][0][-1] for c in self.corners)

    def bands(self, m):
        m = np.clip(m, self.lo, self.hi)
        out = np.zeros((3, np.size(m)))
        for w, (mass, G, BP, RP) in self.corners:
            for r, band in enumerate((G, BP, RP)):
                out[r] += w * np.interp(m, mass, band)
        return out


def imf_sample(rng, n, lo, hi):
    """Inverse-CDF draw from the Chabrier (2014) dN/dm on [lo, hi] (cumulative trapezoid, 20001
    log-spaced points)."""
    m = np.geomspace(lo, hi, 20001)
    xi = _chabrier2014_xi(m)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(m))])
    return np.interp(rng.uniform(size=n), cdf / cdf[-1], m)


def ind_true_stars(iso, kG, kBP, kRP, feh, loga, dm, Av, n, rng, alpha, beta, detail=False):
    """``n`` noiseless apparent (G, colour) from the independent generator (module docstring)."""
    mi = MassIsochrone(iso, feh, loga)
    m1 = imf_sample(rng, n, mi.lo, mi.hi)
    b1 = mi.bands(m1)
    is_bin = rng.uniform(size=n) < np.clip(alpha + beta / (1.0 + 1.4 / m1), 0.0, 1.0)
    g1 = _dk_gamma(m1) + 1.0
    q = rng.uniform(size=n) ** (1.0 / g1)
    m2 = np.clip(q * m1, mi.lo, None)  # companion floor: the isochrone's lowest mass
    b2 = mi.bands(m2)
    bands = np.where(is_bin[None, :], _mag_combine(b1, b2), b1)
    G = bands[0] + dm + kG * Av
    col = (bands[1] - bands[2]) + (kBP - kRP) * Av
    return (G, col, m1, is_bin, q) if detail else (G, col)


@contextlib.contextmanager
def independent_generator(binaries: bool):
    """Replace ``iv_common._true_stars`` (the likelihood's own sheet) by the independent generator
    for the duration of a draw; ``gen`` is the full-window fitter that carries the grid, the
    extinction coefficients and the Offner (alpha, beta)."""
    orig = C._true_stars

    def patched(gen, feh, loga, dm, Av, n, rng):
        a, b = (gen.alpha, gen.beta) if binaries else (0.0, 0.0)
        return ind_true_stars(gen._isochs, gen._kG, gen._kBP, gen._kRP, feh, loga, dm, Av, n, rng, a, b)

    C._true_stars = patched
    try:
        yield
    finally:
        C._true_stars = orig


def make_fitter(data, binaries: bool):
    """The K x W fitter; ``binaries=False`` is the singles model (alpha = beta = 0), set before the
    first segment/deposit call so ``_segments`` returns ``binary = None``."""
    f = A.make_fitter(data, K.WIN)
    if not binaries:
        f.alpha = f.beta = 0.0
    return f


def prep(arm: str, rep: int, search8: bool = True) -> None:
    import erotica

    gen_bin, mod_bin, mult = ARMS[arm]
    t0 = time.time()
    out = RUNS / f"{arm}_{rep}"
    out.mkdir(parents=True, exist_ok=True)
    real = C.real_table()
    seed = K.SEED0["A"] + rep  # the truth of cell A's replicate `rep`, drawn from the same stream
    rng = np.random.default_rng(seed)
    truth = K.truth_for("A", rep, rng)
    gen = A.make_fitter(real, "F")
    fw = A.make_fitter(real, K.WIN)
    with independent_generator(gen_bin):
        data, gi = K.draw_table(fw, gen, truth, mult * int(fw._N_obs), rng, real, A.WINDOW[K.WIN])
    data.write(out / "data.ecsv", overwrite=True)
    names = A.free_names(K.ARM)
    info = {"cell": arm, "rep": rep, "arm": K.ARM, "window": K.WIN, "generator": "independent",
            "binaries_in_generator": gen_bin, "binaries_in_model": mod_bin, "mult": mult,
            "erotica_file": erotica.__file__, "started": C.now(), "seed": seed, "truth": truth,
            "truth_pymc": {K.PYMC[k]: truth[k] for k in names}, **gi}
    f = make_fitter(data, mod_bin)
    info["n_window"] = int(f._N_obs)
    names_ = names
    if search8:  # as cells A and C: the control and the NUTS starts (Laplace at search8's mode)
        res = A.search8(f, K.ARM, verbose=False, start_av=1.0)
        st = S8.laplace_starts(f, K.ARM, res["x"], seed=11 + rep)
        info["control"] = {"mode": {K.PYMC[k]: v for k, v in zip(names_, res["x"], strict=True)},
                           "x": res["x"], "laplace_sd": st["laplace_sd_all"],
                           "at_bound": res["at_bound"], "logpost": res["logpost"],
                           "runner_up": res["runner_up"], "seconds": res["seconds"]}
        fl = f.compiled8(K.ARM)
        info["truth_vs_mode"] = {"loglike_truth": fl(np.asarray([truth[k] for k in names_], float))[0],
                                 "loglike_mode": fl(np.asarray(res["x"], float))[0]}
        C.dump(out / "starts.json", st)
    info["seconds"] = round(time.time() - t0, 1)
    C.dump(out / "prep.json", info)
    print(arm, rep, "prepared", info["n_window"], "stars", info["seconds"], "s", flush=True)


def glob(arm: str, rep: int) -> None:
    """Amended estimator (``search_global``) on a prepared replicate -> ``global.json``: the mode, log L
    there and at the truth (a mode with log L(truth) > log L(mode) + 0.5 is flagged, §2.17 rule 5)."""
    from astropy.table import Table

    out = RUNS / f"{arm}_{rep}"
    prep_ = json.loads((out / "prep.json").read_text())
    f = make_fitter(__import__("astropy.table", fromlist=["QTable"]).QTable(Table.read(out / "data.ecsv")),
                    prep_["binaries_in_model"])
    names = A.free_names(K.ARM)
    res = search_global(f)
    fl = f.compiled8(K.ARM)
    t = prep_["truth"]
    llt = fl(np.asarray([t[k] for k in names], float))[0]
    res.update({"arm": arm, "rep": rep, "loglike_truth": llt, "truth_loga": t["loga"],
                "err_loga": res["mode"]["loga"] - t["loga"], "flag_search_missed": bool(llt > res["loglike"] + 0.5)})
    C.dump(out / "global.json", res)
    print(arm, rep, "global mode loga", round(res["mode"]["loga"], 4), "err", round(res["err_loga"], 4),
          "logL", round(res["loglike"], 2), "truth", round(llt, 2), "flag", res["flag_search_missed"],
          res["seconds"], flush=True)


def selftest() -> dict:
    """Oracles of the generator, none of them the generator's own code path:
    (1) equal-mass pair = single - 2.5 log10(2) in every band (closed form);
    (2) primary masses follow the Chabrier CDF obtained by ``scipy.integrate.quad`` (KS);
    (3) the binary fraction is the integral of b(m) dN/dm (``quad``), within 5 sigma;
    (4) alpha = beta = 0 draws no binary and every star lies on the blended isochrone;
    (5) at a node, the blend is that node's isochrone (corners = 1)."""
    from scipy import integrate, stats

    real = C.real_table()
    gen = A.make_fitter(real, "F")
    iso, kG, kBP, kRP = gen._isochs, gen._kG, gen._kBP, gen._kRP
    feh, loga = -0.43, 5.96
    out = {}
    out["equal_pair_offset"] = float(_mag_combine(np.array([5.0]), np.array([5.0]))[0] - 5.0)
    out["equal_pair_expected"] = float(-2.5 * np.log10(2.0))
    rng = np.random.default_rng(3)
    mi = MassIsochrone(iso, feh, loga)
    m = imf_sample(rng, 20000, mi.lo, mi.hi)
    norm = integrate.quad(lambda x: float(_chabrier2014_xi(x)), mi.lo, mi.hi, limit=200)[0]
    cdf = lambda x: integrate.quad(lambda y: float(_chabrier2014_xi(y)), mi.lo, x, limit=200)[0] / norm  # noqa: E731
    out["imf_ks_p"] = float(stats.kstest(m[::8], np.vectorize(cdf)).pvalue)
    G, col, m1, isb, q = ind_true_stars(iso, kG, kBP, kRP, feh, loga, 10.2, 0.5, 40000, rng,
                                        gen.alpha, gen.beta, detail=True)
    fb = integrate.quad(lambda x: float(np.clip(gen.alpha + gen.beta / (1 + 1.4 / x), 0, 1)
                                        * _chabrier2014_xi(x)), mi.lo, mi.hi, limit=200)[0] / norm
    out["binary_fraction"] = {"measured": float(isb.mean()), "expected": float(fb),
                              "z": float((isb.mean() - fb) / np.sqrt(fb * (1 - fb) / isb.size))}
    G0, c0, _, isb0, _ = ind_true_stars(iso, kG, kBP, kRP, feh, loga, 10.2, 0.5, 4000, rng, 0.0, 0.0, True)
    out["singles_only"] = {"n_binary": int(isb0.sum())}
    zs = np.asarray(iso._met_values, float)
    ages = np.asarray(iso._loga_values, float)
    node = MassIsochrone(iso, float(zs[2]), float(ages[10]))
    mass, Gn, _, _ = _corner(iso, zs[2], ages[10])
    mm = np.linspace(max(mass[0], node.lo), min(mass[-1], node.hi), 50)
    out["node_blend_max_abs"] = float(np.max(np.abs(node.bands(mm)[0] - np.interp(mm, mass, Gn))))
    out["lo_hi"] = [float(mi.lo), float(mi.hi)]
    print(json.dumps(out, indent=1))
    return out


def search_global(f, polish=12, wide=None, feh_lattice=(-1.0, -0.5, 0.0, 0.5), age_stride=2,
                  start_av=1.0, verbose=False) -> dict:
    """``aniso.search8`` with two changes (hub §2.17, amendment 1, chosen on cell A replicate 0 and on
    nothing else): the lattice stage's nuisance parameters sit at ``wide`` (default the K x W mode of the
    REAL data, sigma_int 0.094, k 2.446, f_bg 0.067, which a real analysis has) instead of
    (0.05, 1, 0.02), and the best ``polish`` lattice points are polished instead of 4. The objective and
    bounds are search8's. Never seeded at a truth."""
    from scipy.optimize import minimize

    t0 = time.time()
    names = A.free_names("K")
    fl = f.compiled8("K")
    zlo, zhi = f._met_bounds()
    blo = {"feh": zlo + 1e-9, "loga": f.loga_range[0], "dm": f.dm_range[0], "Av": f.Av_range[0]}
    bhi = {"feh": zhi - 1e-9, "loga": f.loga_range[1], "dm": f.dm_range[1], "Av": f.Av_range[1]}
    for k, (a, b) in A.WBOUNDS.items():
        blo[k], bhi[k] = a, b
    lo = np.array([blo[n] for n in names])
    hi = np.array([bhi[n] for n in names])
    wide = wide or (K.KW["sG"], K.KW["k"], K.KW["f_bg"])

    def fn(y):
        v, g = fl(y)
        z = (y[2] - f.dm_mu) / f.dm_sigma
        g = g.copy()
        g[2] -= z / f.dm_sigma
        return v - 0.5 * z * z, g

    ages = [a for a in f._node_loga if lo[1] <= a <= hi[1]][::age_stride]
    fehs = [x for x in feh_lattice if zlo <= x <= zhi]
    cands = []
    for z in fehs:
        for a in ages:

            def nll2(y, z=float(z), a=float(a)):
                v, g = fn(np.array([z, a, y[0], y[1], *wide]))
                return -v, -g[2:4]

            r = minimize(nll2, [f.dm_mu, start_av], jac=True, method="L-BFGS-B",
                         bounds=list(zip(lo[2:4], hi[2:4], strict=True)))
            cands.append((float(r.fun), float(z), float(a), *map(float, r.x)))
    cands.sort(key=lambda c: c[0])

    def nll(y):
        v, g = fn(y)
        return -v, -g

    pol = []
    for c in cands[:polish]:
        y0 = np.clip(np.array([*c[1:], *wide]), lo, hi)
        r = minimize(nll, y0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)))
        pol.append((-float(r.fun), np.asarray(r.x, float)))
        if verbose:
            print(f"  polish lp={-r.fun:.2f} loga={r.x[1]:.3f} t={time.time() - t0:.0f}s", flush=True)
    pol.sort(key=lambda p: -p[0])
    lp, b = pol[0]
    at = [n for n, x, a, z in zip(names, b, lo, hi, strict=True) if min(x - a, z - x) < 1e-3 * (z - a)]
    return {"arm": "K", "x": b.tolist(), "mode": A.expand("K", b), "logpost": lp, "loglike": fl(b)[0],
            "runner_up": [(p[0], p[1].tolist()) for p in pol[1:]], "at_bound": at,
            "n": int(f._N_obs), "seconds": round(time.time() - t0, 1)}



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "selftest", "glob"])
    ap.add_argument("--no-search8", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--rep", type=int, default=0)
    a = ap.parse_args()
    if a.stage == "selftest":
        selftest()
    elif a.stage == "glob":
        glob(a.arm, a.rep)
    else:
        prep(a.arm, a.rep, not a.no_search8)
