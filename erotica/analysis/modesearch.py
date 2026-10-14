r"""Is there another solution far from where the chains started?  (prototype, branch ``feat/mode-search``)

WHY THIS MODULE EXISTS
----------------------
R-hat, ESS and tau look at the draws. A mode that no chain ever visited leaves no draw, so no
diagnostic computed from draws can see it (hub ``methodology.md`` §A.1.2; Vehtari et al. 2021 say the
same of R-hat). Two cases in this programme:

* HD 159176 outer orbit: 2/32 emcee walkers sat in a period alias P ~ 3.95 yr about 14 log-units below
  the main mode and never crossed; tau passed, R-hat almost passed. Their weight was set by the
  initialisation, not by the posterior (hub ``state/findings.yaml`` R26-04).
* NGC 6383 isochrone, K x W, synthetic cell C: NUTS chains sat on plateaus along an age-[Fe/H] ridge,
  0/8 replicates converged (hub ``agent-findings/isochrone-intervals-nuts.md`` §2.15).

WHAT IT DOES
------------
:func:`mode_search` is the cheap, sampler-agnostic half: local optimisations from starts spread over the
prior, then starts placed along the lowest-curvature axis of the best mode (the ridge), clustering of the
optima into modes, a Laplace approximation at each mode, and -- if posterior draws are given -- which
modes the chains visited and with what weight. :func:`verdict` turns that into the mandatory output line.

WHAT IT DOES NOT DO
-------------------
It finds basins of attraction of a local optimiser, not probability mass. A thin, high mode with a small
basin can be missed by every start; a broad plateau's mass is not its height (the Laplace mass is a
local quadratic guess). Passing it is evidence that ``n_starts`` optimisations from the prior found no
competing basin -- not proof that none exists. Nested sampling or tempering are the heavier checks.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

# A mode the chains never visited is a failure if its Laplace mass is within this many log-units of the
# best mode's (e^-5 ~ 0.7 %). A visited mode is a failure if the fraction of draws in it exceeds
# MASS_RATIO_TOL times its Laplace share AND is above MIN_FRACTION: that is the R26-04 shape (weight set
# by the initialisation). Both are first guesses to be calibrated, not doctrine.
UNVISITED_DLOGMASS = -5.0
MASS_RATIO_TOL = 10.0
MIN_FRACTION = 0.01


def _wrap(d, periods):
    for i, p in periods.items():
        d[..., i] = (d[..., i] + 0.5 * p) % p - 0.5 * p
    return d


def _hessian(f, x, h):
    """Central-difference Hessian of a scalar function."""
    n = len(x)
    H = np.empty((n, n))
    f0 = f(x)
    for i in range(n):
        ei = np.zeros(n)
        ei[i] = h[i]
        H[i, i] = (f(x + ei) - 2 * f0 + f(x - ei)) / h[i] ** 2
        for j in range(i):
            ej = np.zeros(n)
            ej[j] = h[j]
            H[i, j] = H[j, i] = (
                f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)
            ) / (4 * h[i] * h[j])
    return H


def _laplace(logp, x, scale, bounds, grad=None):
    h = 1e-4 * scale
    if bounds is not None:  # one-sided room at a bound would bias the Hessian: shrink the step
        lo, hi = bounds
        h = np.minimum(h, 0.5 * np.maximum(np.minimum(x - lo, hi - x), 1e-12))
    if grad is not None:  # 2d gradient calls instead of ~2d^2 values (the isochrone costs 0.5 s each)
        H = np.array([(grad(x + hi_ * e) - grad(x - hi_ * e)) / (2 * hi_) for hi_, e in zip(h, np.eye(len(x)), strict=True)])
    else:
        H = _hessian(logp, x, h)
    try:
        cov = np.linalg.inv(-0.5 * (H + H.T))
        sign, logdet = np.linalg.slogdet(cov)
        ok = bool(sign > 0 and np.all(np.diag(cov) > 0))
    except np.linalg.LinAlgError:
        cov, logdet, ok = None, np.nan, False
    return cov, (float(logdet) if ok else np.nan), ok


def _optimise(neglogp, x0, grad, bounds, maxiter):
    jac = (lambda y: -grad(y)) if grad is not None else None
    b = None if bounds is None else list(zip(*bounds, strict=True))
    r = minimize(neglogp, x0, jac=jac, method="L-BFGS-B", bounds=b, options={"maxiter": maxiter})
    return np.asarray(r.x, float), -float(r.fun), int(r.nfev)


def mode_search(
    logp,
    starts,
    *,
    grad=None,
    bounds=None,
    scale=None,
    periods=None,
    ridge_fracs=(0.125, 0.25, 0.5, 0.9),
    same_mode=3.0,
    draws=None,
    maxiter=2000,
    names=None,
):
    """Multi-start optimisation + ridge starts + mode clustering + Laplace per mode.

    Parameters
    ----------
    logp : callable(x) -> float
        Log posterior (any constant) in the coordinates the optimiser moves in. ``-inf`` outside support.
    starts : (n, d) array
        Starting points **drawn from the prior** (never seeded at a truth or at the chains).
    grad : callable(x) -> (d,), optional
        Gradient of ``logp``; finite differences otherwise.
    bounds : (lo, hi), optional
    scale : (d,), optional
        Typical width per coordinate (prior sd or range); sets Hessian steps and the fallback metric.
    periods : {index: period}, optional
        Periodic coordinates (a phase), wrapped before distances.
    ridge_fracs : sequence of float
        Starts along the best mode's lowest-curvature axis, both ways, at these fractions of the way to
        the bounds (10 x ``scale`` where unbounded), plus +-4 Laplace sd. Cell C's plateaus sat 25-100
        Laplace sd apart, out of reach of steps counted in sd.
    same_mode : float
        Two optima are one mode when their Mahalanobis distance (best mode's Laplace covariance) is
        below this.
    draws : (m, c, d) array, optional
        Posterior draws (draw, chain/walker, dim) to compare against the modes found.
    """
    starts = np.atleast_2d(np.asarray(starts, float))
    d = starts.shape[1]
    scale = np.ones(d) if scale is None else np.asarray(scale, float)
    periods = periods or {}
    nlp = lambda y: -lp if np.isfinite(lp := float(logp(y))) else 1e300  # noqa: E731
    nfev = {"prior": 0, "chain": 0, "ridge": 0}
    optima = []
    # the last draw of every chain is also polished, so the modes the chains found are in the list and
    # each draw can be assigned to one; a mode reached ONLY from "chain" starts is not news, one reached
    # only from "prior"/"ridge" starts is what the chains may have missed
    extra = [] if draws is None else [("chain", x0) for x0 in np.asarray(draws, float)[-1]]
    for src, x0 in [("prior", x0) for x0 in starts] + extra:
        x, v, n = _optimise(nlp, x0, grad, bounds, maxiter)
        nfev[src] += n
        if np.isfinite(v) and v > -1e299:
            optima.append((v, _wrap(x.copy()[None], periods)[0], src))

    def cluster(optima):
        optima = sorted(optima, key=lambda o: -o[0])
        modes = []
        for v, x, src in optima:
            for m in modes:
                dx = _wrap((x - m["x"])[None], periods)[0]
                dist = np.sqrt(dx @ m["prec"] @ dx) if m["prec"] is not None else np.max(np.abs(dx / scale)) / 1e-3
                if dist < same_mode:
                    m["n_starts"] += 1
                    m["sources"].add(src)
                    break
            else:
                cov, logdet, ok = _laplace(logp, x, scale, bounds, grad)
                at = [] if bounds is None else [
                    i for i in range(d) if min(x[i] - bounds[0][i], bounds[1][i] - x[i]) < 1e-6 * scale[i]]
                modes.append(
                    dict(x=x, logp=v, cov=cov, prec=np.linalg.inv(cov) if ok else None, logdet=logdet,
                         laplace_ok=ok, at_bound=at, n_starts=1, sources={src})
                )
        return modes

    modes = cluster(optima)
    # ridge: the best mode's lowest-curvature axis, where the plateaus of cell C lay
    if modes and modes[0]["laplace_ok"] and ridge_fracs:
        w, V = np.linalg.eigh(modes[0]["cov"])
        axis, sd = V[:, -1], np.sqrt(w[-1])
        x_b = modes[0]["x"]
        lo = bounds[0] if bounds is not None else np.full(d, -np.inf)
        hi = bounds[1] if bounds is not None else np.full(d, np.inf)
        lo = np.where(np.isfinite(lo), lo, x_b - 10 * scale)
        hi = np.where(np.isfinite(hi), hi, x_b + 10 * scale)
        pts = []
        for sgn in (1.0, -1.0):
            ax = sgn * axis
            with np.errstate(divide="ignore", invalid="ignore"):
                room = np.where(ax > 0, (hi - x_b) / ax, np.where(ax < 0, (lo - x_b) / ax, np.inf))
            tmax = float(np.min(room))
            pts += [(f"ridge{sgn * f:+g}", x_b + f * tmax * ax) for f in ridge_fracs]
            pts.append((f"ridge{sgn * 4:+g}sd", x_b + 4 * sd * ax))
        for k, x0 in pts:
            if bounds is not None:
                x0 = np.clip(x0, bounds[0], bounds[1])
            x, v, n = _optimise(nlp, x0, grad, bounds, maxiter)
            nfev["ridge"] += n
            if np.isfinite(v) and v > -1e299:
                optima.append((v, _wrap(x.copy()[None], periods)[0], k))
        modes = cluster(optima)

    best = modes[0]
    for m in modes:
        m["dlogp"] = m["logp"] - best["logp"]
        m["log_mass"] = m["logp"] + 0.5 * m["logdet"] if m["laplace_ok"] else np.nan
    lm = np.array([m["log_mass"] for m in modes])
    ref = np.nanmax(lm) if np.isfinite(lm).any() else np.nan
    for m in modes:
        m["dlog_mass"] = m["log_mass"] - ref
        # a mode at a bound or with a non-definite Hessian has no Laplace mass; it still counts (the bound
        # is part of the model, hub isochrone-intervals-nuts §2.16), ranked by its height instead
        m["dlog_key"] = m["dlog_mass"] if np.isfinite(m["dlog_mass"]) else m["dlogp"]
    out = dict(modes=modes, n_starts=len(starts), nfev=nfev, names=names)
    if draws is not None:
        out["chains"] = _assign(np.asarray(draws, float), modes, scale, periods)
    return out


def _assign(draws, modes, scale, periods):
    """Nearest mode (Mahalanobis, each mode's own Laplace covariance) for every draw."""
    m_, c_, d = draws.shape
    flat = draws.reshape(-1, d)
    D = np.empty((len(flat), len(modes)))
    for j, m in enumerate(modes):
        dx = _wrap(flat - m["x"], periods)
        # Gaussian classification (Mahalanobis + log det) so a broad mode does not swallow a narrow one
        D[:, j] = (np.einsum("ni,ij,nj->n", dx, m["prec"], dx) + m["logdet"] if m["prec"] is not None
                   else np.sum((dx / scale) ** 2, 1) * 1e6)
    lab = D.argmin(1).reshape(m_, c_)
    frac = np.array([(lab == j).mean() for j in range(len(modes))])
    per_chain = np.array([[(lab[:, c] == j).mean() for j in range(len(modes))] for c in range(c_)])
    home = per_chain.argmax(1)
    crossings = int(np.sum(lab[1:] != lab[:-1]))
    return dict(fraction=frac.tolist(), chains_whole_in=[int(np.sum((per_chain[:, j] == 1.0))) for j in range(len(modes))],
                chain_home=home.tolist(), crossings=crossings, n_chains=c_)


def verdict(res, *, unvisited=UNVISITED_DLOGMASS, ratio=MASS_RATIO_TOL, min_fraction=MIN_FRACTION):
    """The mandatory output: modes found, delta log posterior, where, and PASS/FAIL with the reason."""
    modes, names = res["modes"], res["names"]
    lines = [f"mode search: {res['n_starts']} prior starts + ridge, evaluations {res['nfev']}, "
             f"{len(modes)} mode(s)"]
    fails = []
    ch = res.get("chains")
    lm = np.array([m["dlog_key"] for m in modes])
    share = np.exp(lm - np.nanmax(lm)) / np.nansum(np.exp(lm - np.nanmax(lm))) if np.isfinite(lm).any() else None
    for j, m in enumerate(modes):
        where = ", ".join(f"{n}={v:.4g}" for n, v in zip(names or range(len(m["x"])), m["x"], strict=True))
        s = (f"  [{j}] logp={m['logp']:.2f} dlogp={m['dlogp']:+.2f} dlog_mass={m['dlog_mass']:+.2f}{' AT BOUND ' + str(m['at_bound']) if m['at_bound'] else ''} starts={m['n_starts']} "
             f"from={sorted(m['sources'])} at {where}")
        if ch is not None:
            f = ch["fraction"][j]
            s += f" | draws {100 * f:.2f} %, chains wholly in it {ch['chains_whole_in'][j]}/{ch['n_chains']}"
            if f == 0 and m["dlog_key"] >= unvisited:
                fails.append(f"mode {j} has dlog_mass {m['dlog_key']:+.2f} >= {unvisited} and no draw visited it")
            if share is not None and np.isfinite(share[j]) and f > min_fraction and f > ratio * share[j]:
                fails.append(f"mode {j} holds {100 * f:.2f} % of draws against a Laplace share of "
                             f"{100 * share[j]:.2g} %: weight set by the initialisation, not the posterior")
        lines.append(s)
    if ch is None:
        lines.append("  no draws given: modes reported, chain coverage not checked")
    if ch is not None and len(modes) > 1:
        lines.append(f"  crossings between modes along the chains: {ch['crossings']}")
    lines.append("VERDICT: " + ("FAIL -- " + "; ".join(fails) if fails else "PASS (within the limits in the docstring)"))
    return "\n".join(lines), not fails
