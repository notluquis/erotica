#!/usr/bin/env python3
"""Profile of the K x W log posterior (and of log L alone) in log t on the 201 real members, G 14-19
(hub finding ``isochrone-intervals-nuts.md`` §2.16, pre-registered before any profile value was read).

At each fixed log t, maximise over (feh, dm, Av, sigma_int, k, f_bg) with L-BFGS-B from >= 8 starts,
then continuation sweeps from the neighbours' optima until nothing improves by more than 0.01.
The likelihood is ``aniso.WidthFitter.compiled8("K")`` unchanged (the one NUTS used, nuts8.py).

Objectives (terms stored separately at every optimum):
  post = log L - 0.5 z_dm^2 - sigma_int^2 / (2 0.05^2) + 18 log(1 - f_bg)      (NUTS priors, no constants;
         the uniforms on feh, log t, Av, k are bounds only) -- density in the model's own parameters
  like = log L                                                                  (dm free in [9.5, 10.7])
  s8   = log L - 0.5 z_dm^2                                                     (search8's metric, oracle only)

    profile_loga.py run T [T ...]      one JSON per point in kw_runs/real_0/profile/
    profile_loga.py cost               3 points, prints timing and nfev ONLY (blind)
    profile_loga.py sweep              continuation sweeps over the written points
    profile_loga.py laplace            nuisance log-det (6-D Hessian of post) at each point's optimum
    profile_loga.py runnerup           polish search8's runner-up (log t 6.62) to convergence, s8 metric
    profile_loga.py summary            profile.json + profile.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402

RUN = HERE / "kw_runs" / "real_0"
OUT = RUN / "profile"
NAMES = A.free_names("K")  # feh, loga, dm, Av, sG, k, f_bg
NUIS = [0, 2, 3, 4, 5, 6]
PREP = json.loads((RUN / "prep.json").read_text())["control"]
XMODE = np.array(PREP["x"], float)
TMODE = float(XMODE[1])
RUNNER = np.array(PREP["runner_up"][-1][1], float)  # log t 6.623, s8 -298.83
RIDGE = np.array(json.loads((HERE / "starts_K_W.json").read_text())["ridge_vector"], float)
FEH_LATTICE = (-0.9, -0.5, -0.1, 0.2, 0.45)


def grid() -> list[float]:
    """5.70-7.00: step 0.02 outside [6.00, 6.50], 0.01 inside, plus the search8 mode."""
    g = set(np.round(np.arange(5.70, 7.0001, 0.02), 4)) | set(np.round(np.arange(6.00, 6.5001, 0.01), 4))
    return sorted(float(t) for t in g | {round(TMODE, 5)})


_F = None


def fitter():
    global _F
    if _F is None:
        f = A.make_fitter(A.C.real_table(), "W")
        _F = (f, f.compiled8("K"))
    return _F


def bounds(f):
    zlo, zhi = f._met_bounds()
    lo = {"feh": zlo + 1e-9, "loga": f.loga_range[0], "dm": f.dm_range[0], "Av": f.Av_range[0]}
    hi = {"feh": zhi - 1e-9, "loga": f.loga_range[1], "dm": f.dm_range[1], "Av": f.Av_range[1]}
    for k, (a, b) in A.WBOUNDS.items():
        lo[k], hi[k] = a, b
    return np.array([lo[n] for n in NAMES]), np.array([hi[n] for n in NAMES])


def terms(f, ll, x):
    z = (x[2] - f.dm_mu) / f.dm_sigma
    return {"logL": ll, "dm_prior": -0.5 * z * z, "sig_prior": -x[4] ** 2 / (2 * 0.05**2),
            "fbg_prior": 18.0 * np.log1p(-x[6])}


def objective(kind):
    f, fl = fitter()

    def fn(x):
        ll, g = fl(x)
        g = np.array(g, float).copy()
        if kind == "like":
            return ll, g
        z = (x[2] - f.dm_mu) / f.dm_sigma
        v = ll - 0.5 * z * z
        g[2] -= z / f.dm_sigma
        if kind == "s8":
            return v, g
        v += -x[4] ** 2 / (2 * 0.05**2) + 18.0 * np.log1p(-x[6])
        g[4] -= x[4] / 0.05**2
        g[6] -= 18.0 / (1.0 - x[6])
        return v, g

    return fn


def maximise(kind, t, x0):
    f, fl = fitter()
    lo, hi = bounds(f)
    fn = objective(kind)
    x0 = np.clip(np.asarray(x0, float), lo + 1e-6, hi - 1e-6)
    x0[1] = t

    def nll(y):
        x = x0.copy()
        x[NUIS] = y
        v, g = fn(x)
        return -v, -g[NUIS]

    t0 = time.time()
    r = minimize(nll, x0[NUIS], jac=True, method="L-BFGS-B", options={"maxiter": 1000},
                 bounds=list(zip(lo[NUIS], hi[NUIS], strict=True)))
    x = x0.copy()
    x[NUIS] = r.x
    at = [NAMES[i] for i in NUIS if min(x[i] - lo[i], hi[i] - x[i]) < 1e-3 * (hi[i] - lo[i])]
    return {"value": -float(r.fun), "x": x.tolist(), "nfev": int(r.nfev), "success": bool(r.success),
            "seconds": round(time.time() - t0, 1), "at_bound": at,
            "terms": {k: float(v) for k, v in terms(f, float(fl(x)[0]), x).items()}}


def c_ridge():
    """[Fe/H] vs log t of the 32 cell-C chain means (the plateaus), and their mean A_V."""
    lt, fe, av = [], [], []
    for c in range(8):
        d = np.load(HERE / "kw_runs" / f"C_{c}" / "draws.npz")
        lt += list(d["loga"].mean(1))
        fe += list(d["feh"].mean(1))
        av += list(d["Av"].mean(1))
    b, a = np.polyfit(lt, fe, 1)
    return float(a), float(b), float(np.mean(av))


def starts(t):
    """The 8 pre-registered starts at log t = t (nuisances sG, k, f_bg at the search8 mode)."""
    ca, cb, cav = c_ridge()
    s = []
    r = XMODE.copy()
    r[[0, 2, 3]] = XMODE[[0, 2, 3]] + (t - TMODE) * RIDGE[[0, 2, 3]]
    s.append(("ridge", r))
    c = XMODE.copy()
    c[0], c[2], c[3] = ca + cb * t, XMODE[2], cav
    s.append(("C_plateaus", c))
    u = RUNNER.copy()
    u[[0, 2, 3]] = RUNNER[[0, 2, 3]] + (t - RUNNER[1]) * RIDGE[[0, 2, 3]]
    s.append(("runner_up", u))
    for z in FEH_LATTICE:
        m = XMODE.copy()
        m[0], m[2], m[3] = z, XMODE[2], 0.6
        s.append((f"feh{z:+.2f}", m))
    return s


def path(t):
    return OUT / f"t_{t:.5f}.json"


def run_point(t, quiet=False):
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"t": t, "post": {}, "like": {}}
    for name, x0 in starts(t):
        rec["post"][name] = maximise("post", t, x0)
    bp = max(rec["post"].values(), key=lambda r: r["value"])
    for name, x0 in starts(t) + [("warm_post", np.array(bp["x"]))]:
        rec["like"][name] = maximise("like", t, x0)
    if abs(t - TMODE) < 1e-4:
        rec["s8_oracle"] = maximise("s8", t, XMODE)
    for k in ("post", "like"):
        rec[f"best_{k}"] = max(rec[k].values(), key=lambda r: r["value"])
    path(t).write_text(json.dumps(rec, indent=1))
    if not quiet:
        print(f"t={t:.4f} done", flush=True)
    return rec


def cost():
    """Blind cost test: timing and nfev only, no profile value printed."""
    for t in (5.97, 6.30, 6.62):
        t0 = time.time()
        rec = run_point(t, quiet=True)
        n = [r["nfev"] for k in ("post", "like") for r in rec[k].values()]
        print(f"t={t}: {time.time() - t0:.0f} s, {len(n)} optimisations, nfev median {np.median(n):.0f} "
              f"max {max(n)}", flush=True)


def sweep(tol=0.01, max_pass=6):
    """Continuation: re-optimise each point from its neighbours' optima until no gain > tol."""
    ts = sorted(float(p.stem[2:]) for p in OUT.glob("t_*.json"))
    for npass in range(max_pass):
        gain = 0.0
        order = ts if npass % 2 == 0 else ts[::-1]
        for i, t in enumerate(order):
            rec = json.loads(path(t).read_text())
            for nb in (order[i - 1] if i else None, order[i + 1] if i + 1 < len(order) else None):
                if nb is None:
                    continue
                nrec = json.loads(path(nb).read_text())
                for k in ("post", "like"):
                    r = maximise(k, t, nrec[f"best_{k}"]["x"])
                    rec[k][f"cont_{nb:.5f}_p{npass}"] = r
                    g = r["value"] - rec[f"best_{k}"]["value"]
                    if g > 0:
                        gain = max(gain, g)
                        rec[f"best_{k}"] = r
            path(t).write_text(json.dumps(rec, indent=1))
        print(f"pass {npass}: max gain {gain:.4f}", flush=True)
        if gain <= tol:
            return npass
    return max_pass


def hessian_nuis(kind, x, h=None):
    fn = objective(kind)
    x = np.asarray(x, float)
    h = h if h is not None else np.array([1e-3, 0, 1e-3, 1e-3, 1e-4, 1e-4, 1e-4])
    H = np.empty((6, 6))
    for a, i in enumerate(NUIS):
        e = np.zeros(7)
        e[i] = h[i]
        H[a] = (fn(x + e)[1][NUIS] - fn(x - e)[1][NUIS]) / (2 * h[i])
    return -0.5 * (H + H.T)


def laplace():
    for p in sorted(OUT.glob("t_*.json")):
        rec = json.loads(p.read_text())
        H = hessian_nuis("post", rec["best_post"]["x"])
        sgn, ld = np.linalg.slogdet(H)
        rec["laplace_post"] = {"sign": float(sgn), "logdet_negH": float(ld),
                               "min_eig": float(np.linalg.eigvalsh(H).min())}
        p.write_text(json.dumps(rec, indent=1))
    print("laplace done", flush=True)


def runnerup():
    f, fl = fitter()
    lo, hi = bounds(f)
    fn = objective("s8")
    t0 = time.time()
    r = minimize(lambda y: tuple(-np.asarray(v) for v in fn(y)), np.clip(RUNNER, lo, hi), jac=True,
                 method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True)), options={"maxiter": 2000})
    out = {"start_s8": fn(RUNNER)[0], "polished_s8": -float(r.fun), "x": r.x.tolist(), "nfev": int(r.nfev),
           "mode_s8": fn(XMODE)[0], "seconds": round(time.time() - t0, 1)}
    (OUT / "runnerup.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out), flush=True)


def summary():
    recs = [json.loads(p.read_text()) for p in sorted(OUT.glob("t_*.json"))]
    recs.sort(key=lambda r: r["t"])
    t = np.array([r["t"] for r in recs])
    P = np.array([r["best_post"]["value"] for r in recs])
    L = np.array([r["best_like"]["value"] for r in recs])
    i0, j0 = int(P.argmax()), int(L.argmax())
    dP, dL = P - P[i0], L - L[j0]
    rows = []
    for r, a, b in zip(recs, dP, dL, strict=True):
        bp = r["best_post"]
        spread = sorted(v["value"] for k, v in r["post"].items() if not k.startswith("cont_"))
        rows.append({"t": r["t"], "dpost": float(a), "dlike": float(b), "x_post": bp["x"],
                     "terms_post": bp["terms"], "at_bound_post": bp["at_bound"],
                     "x_like": r["best_like"]["x"], "at_bound_like": r["best_like"]["at_bound"],
                     "n_starts_within_0.01": int(sum(v >= spread[-1] - 0.01 for v in spread)),
                     "laplace_logdet": r.get("laplace_post", {}).get("logdet_negH"),
                     "s8_oracle": r.get("s8_oracle", {}).get("value")})
    (OUT.parent / "profile.json").write_text(json.dumps({"tmax_post": float(t[i0]), "tmax_like": float(t[j0]),
                                                         "rows": rows}, indent=1))
    print(f"{len(rows)} points; argmax post {t[i0]:.4f}, like {t[j0]:.4f}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "run":
        for tt in sys.argv[2:]:
            run_point(float(tt))
    elif cmd == "grid":
        print(" ".join(f"{x:.5f}" for x in grid()))
    else:
        {"cost": cost, "sweep": sweep, "laplace": laplace, "runnerup": runnerup, "summary": summary}[cmd]()
