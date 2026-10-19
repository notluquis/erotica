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
    pmap=None,
    embed=None,
    embed_periods=None,
    draws_to_x=None,
    per_chain=5,
    bound_tol=1e-3,
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
        Two optima are one mode when their Mahalanobis distance, under the Laplace covariance of the mode
        already accepted, is below this.
    draws : (m, c, d) array, optional
        Posterior draws (draw, chain/walker, dim) to compare against the modes found. ``per_chain`` draws
        of every chain, evenly spaced, are polished; the modes they reach are where that chain lives.
    per_chain : int
        How many draws per chain are polished (hub finding R69-05: one last draw misses a chain that
        spent most of its run in another mode).
    bound_tol : float
        A mode within ``bound_tol`` x (upper - lower) of a bound of that coordinate counts as AT BOUND
        (no Laplace mass, ranked by height). Relative to the bound range, not to ``scale``: the
        NGC 6383 likelihood mode at dm = 9.500196 sat 2e-4 from the bound 9.5 and the old
        ``1e-6 * scale`` tolerance (scale = the prior sd, 0.026) did not flag it (hub finding §3.5).
    pmap : callable(list of x0) -> list of (x, logp, nfev), optional
        Runs the local optimisations elsewhere (a process pool); serial ``L-BFGS-B`` otherwise.
    embed, embed_periods, draws_to_x : optional
        Optimise in one parametrisation and compare in another. Needed when the density in the sampler's
        coordinates has an integrable singularity an optimiser climbs (HD 159176: the physical prior
        written in Thiele-Innes elements, -2 log sin i at face-on orbits; hub finding §3): optimise
        where the density is bounded, and give ``embed(x)`` -> the draws' coordinates (where modes are
        clustered and draws assigned, with the Laplace covariance pushed through its Jacobian), the
        periods there, and ``draws_to_x(y)`` -> a starting point from a draw.
    """
    starts = np.atleast_2d(np.asarray(starts, float))
    d = starts.shape[1]
    n_lap = [0]  # log-density or gradient calls spent on Laplace Hessians (R69-10: they were not counted)
    _logp, _grad = logp, grad

    def lp_count(x):
        n_lap[0] += 1
        return _logp(x)

    gr_count = None if grad is None else (lambda x: (n_lap.__setitem__(0, n_lap[0] + 1), _grad(x))[1])
    lap_cache = {}
    scale = np.ones(d) if scale is None else np.asarray(scale, float)
    periods = periods or {}
    nlp = lambda y: -lp if np.isfinite(lp := float(logp(y))) else 1e300  # noqa: E731
    nfev = {"prior": 0, "chain": 0, "ridge": 0}
    kind = lambda src: "chain" if src.startswith("chain") else src  # noqa: E731
    optima = []
    # the last draw of every chain is also polished, so the modes the chains found are in the list and
    # each draw can be assigned to one; a mode reached ONLY from "chain" starts is not news, one reached
    # only from "prior"/"ridge" starts is what the chains may have missed
    to_x = draws_to_x or (lambda y: y)
    emb = embed or (lambda x: x)
    eper = periods if embed is None else (embed_periods or {})
    extra = []
    if draws is not None:
        dr = np.asarray(draws, float)
        idx = np.unique(np.linspace(0, len(dr) - 1, max(1, per_chain)).round().astype(int))
        extra = [(f"chain{k}_{i}", to_x(dr[i, k])) for k in range(dr.shape[1]) for i in idx]
    run = pmap or (lambda xs: [_optimise(nlp, x0, grad, bounds, maxiter) for x0 in xs])
    jobs = [("prior", x0) for x0 in starts] + extra
    for (src, _), (x, v, n) in zip(jobs, run([x0 for _, x0 in jobs]), strict=True):
        nfev[kind(src)] += n
        if np.isfinite(v) and v > -1e299:
            optima.append((v, _wrap(np.array(x, float)[None], periods)[0], src))

    def cluster(optima):
        optima = sorted(optima, key=lambda o: -o[0])
        modes = []
        for v, x, src in optima:
            y = np.asarray(emb(x), float)
            for m in modes:
                dy = _wrap((y - m["y"])[None], eper)[0]
                if m["prec_y"] is not None:
                    dist = np.sqrt(dy @ m["prec_y"] @ dy)
                elif embed is not None:  # R69-08: compare where the draws live, not in x
                    dist = np.max(np.abs(dy) / (np.abs(m["y"]) + 1e-3)) / 1e-3
                else:
                    dist = np.max(np.abs(_wrap((x - m["x"])[None], periods)[0] / scale)) / 1e-3
                if dist < same_mode:
                    m["n_starts"] += 1
                    m["sources"].add(src)
                    break
            else:
                key = x.tobytes()
                if key not in lap_cache:
                    lap_cache[key] = _laplace(lp_count, x, scale, bounds, gr_count)
                cov, logdet, ok = lap_cache[key]
                at = [] if bounds is None else [
                    i for i in range(d)
                    if min(x[i] - bounds[0][i], bounds[1][i] - x[i])
                    < max(1e-6 * scale[i], bound_tol * (bounds[1][i] - bounds[0][i]))]
                if at:  # R69-07: a Gaussian cut by a bound has no full Laplace mass; rank it by height
                    cov, logdet, ok = cov, np.nan, False
                prec_y, logdet_y = (np.linalg.inv(cov), logdet) if ok else (None, np.nan)
                if ok and embed is not None:  # push the Laplace covariance through the embedding
                    h = 1e-6 * scale
                    J = np.array([_wrap((np.asarray(emb(x + hi_ * e), float) - np.asarray(emb(x - hi_ * e), float))[None], eper)[0]
                                  / (2 * hi_) for hi_, e in zip(h, np.eye(d), strict=True)]).T
                    cy = J @ cov @ J.T
                    sgn, logdet_y = np.linalg.slogdet(cy)
                    prec_y, logdet_y = (np.linalg.inv(cy), float(logdet_y)) if sgn > 0 else (None, np.nan)
                modes.append(
                    dict(x=x, y=y, logp=v, cov=cov, prec=np.linalg.inv(cov) if ok else None, logdet=logdet,
                         prec_y=prec_y, logdet_y=logdet_y, laplace_ok=ok, at_bound=at, n_starts=1, sources={src})
                )
        return modes

    modes = cluster(optima)
    # ridge: the best mode's lowest-curvature axis, where the plateaus of cell C lay
    if modes and ridge_fracs:
        if modes[0]["laplace_ok"]:
            w, V = np.linalg.eigh(modes[0]["cov"])
            axis, sd = V[:, -1], np.sqrt(w[-1])
        else:
            axis, sd = np.eye(d)[int(np.argmax(scale))], float(np.max(scale)) * 1e-3
        # post hoc (hub finding §3, arm T2): on a rippled ridge the local Hessian is stiff along the ridge
        # and the softest axis is a nuisance; when >= 3 modes sit within 10 of the best, the ridge is the
        # first principal component of their positions (in units of ``scale``) instead
        near = np.array([_wrap((m["x"] - modes[0]["x"])[None], periods)[0] / scale
                         for m in modes if m["logp"] >= modes[0]["logp"] - 10])
        if len(near) >= 3:
            axis = np.linalg.svd(near - near.mean(0), full_matrices=False)[2][0] * scale
            axis /= np.linalg.norm(axis)
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
        if bounds is not None:
            pts = [(k, np.clip(x0, bounds[0], bounds[1])) for k, x0 in pts]
        for (k, _), (x, v, n) in zip(pts, run([x0 for _, x0 in pts]), strict=True):
            nfev["ridge"] += n
            if np.isfinite(v) and v > -1e299:
                optima.append((v, _wrap(np.array(x, float)[None], periods)[0], k))
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
    nfev["laplace"] = n_lap[0]
    out = dict(modes=modes, n_starts=len(starts), nfev=nfev, names=names)
    for m in modes:
        hits = [s_[5:].split("_") for s_ in m["sources"] if s_.startswith("chain")]
        m["chain_hits"] = {}
        for c, _ in hits:
            m["chain_hits"][int(c)] = m["chain_hits"].get(int(c), 0) + 1
        m["sources"] = {kind(s_) for s_ in m["sources"]}
    if draws is not None:
        c_ = np.asarray(draws).shape[1]
        out["chains"] = _assign(np.asarray(draws, float), modes, eper)
        # where a chain lives = the modes its polished draws reach (their basins): the per-walker count of
        # hub methodology §A.1.2. Post hoc (hub finding §3, E1b seed 2): per-draw Laplace assignment let a
        # broad junk mode take 20-99 % of the draws when the main mode was not Gaussian
        tot = np.zeros(c_)
        for m in modes:
            for c, k in m["chain_hits"].items():
                tot[c] += k
        out["chains"]["homeless"] = [int(c) for c in range(c_) if tot[c] == 0]
        # occupancy: per chain, the fraction of its polished draws in each mode; averaged over chains
        out["chains"]["occupancy"] = [float(np.mean([m["chain_hits"].get(c, 0) / tot[c] if tot[c] else 0.0
                                                     for c in range(c_)])) for m in modes]
        out["chains"]["homes"] = [len(m["chain_hits"]) for m in modes]
    return out


def _assign(draws, modes, periods):
    """Nearest mode (Mahalanobis, each mode's own Laplace covariance) for every draw."""
    m_, c_, d = draws.shape
    flat = draws.reshape(-1, d)
    D = np.empty((len(flat), len(modes)))
    for j, m in enumerate(modes):
        dx = _wrap(flat - m["y"], periods)
        # responsibility in the Laplace mixture: component weight exp(logp) sqrt(det) times the Gaussian
        # exp(-D/2) / sqrt(det), so the score is logp - D/2 and det cancels. Post hoc (hub finding §3,
        # E1b seed 2): with "D + log det" a broad junk mode 24 below the best took 99.5 % of the draws
        D[:, j] = (np.einsum("ni,ij,nj->n", dx, m["prec_y"], dx) - 2 * m["logp"] if m["prec_y"] is not None
                   else np.sum(dx**2, 1) * 1e6)
    lab = D.argmin(1).reshape(m_, c_)
    frac = np.array([(lab == j).mean() for j in range(len(modes))])
    per_chain = np.array([[(lab[:, c] == j).mean() for j in range(len(modes))] for c in range(c_)])
    home = per_chain.argmax(1)
    crossings = int(np.sum(lab[1:] != lab[:-1]))
    return dict(fraction=frac.tolist(), chains_whole_in=[int(np.sum(per_chain[:, j] == 1.0)) for j in range(len(modes))],
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
            f, h = ch["fraction"][j], ch["occupancy"][j]
            s += (f" | chains in it {ch['homes'][j]}/{ch['n_chains']}, occupancy {100 * h:.2f} %, draws "
                  f"(Laplace assignment, indicative) {100 * f:.2f} %")
            # R69-01: the per-draw assignment is no veto; only polished chain draws count as a visit
            if ch["homes"][j] == 0 and m["dlog_key"] >= unvisited:
                fails.append(f"mode {j} has dlog_mass {m['dlog_key']:+.2f} >= {unvisited} and no chain visited it")
            if share is not None and np.isfinite(share[j]) and h > min_fraction and h > ratio * share[j]:
                fails.append(f"mode {j} holds {100 * h:.2f} % of the chains' occupancy against a Laplace "
                             f"share of {100 * share[j]:.2g} %: weight set by the initialisation, not the posterior")
        lines.append(s)
    if ch is None:
        lines.append("  no draws given: modes reported, chain coverage not checked")
    if ch is not None and len(modes) > 1:
        lines.append(f"  crossings between modes along the chains: {ch['crossings']}")
    if ch is not None and ch["homeless"]:  # R69-09: a chain whose polished draws all failed says nothing
        lines.append(f"  WARNING: chains with no finite polished draw (not counted): {ch['homeless']}")
    lines.append("VERDICT: " + ("FAIL -- " + "; ".join(fails) if fails else "PASS (within the limits in the docstring)"))
    return "\n".join(lines), not fails


def like_vs_post(res_like, res_post, logpost, *, tol=2.0, names=None):
    """Does a prior choose the mode? Mode search on log L alone against the one on the posterior.

    Hub finding §3.5 (NGC 6383 isochrone): the posterior had one mode, the likelihood two humps 0.44
    apart with dm sliding from 10.57 to 9.51 between them, and the dm prior picked the age. A posterior
    mode search cannot see that. Here: if a likelihood mode other than the one holding the posterior's
    best point is within ``tol`` of the best likelihood, the data do not separate them and the prior does.
    ``logpost(x)`` evaluates the posterior at each likelihood mode (same coordinates).
    """
    lm = res_like["modes"]
    xb = res_post["modes"][0]["x"]
    home = int(np.argmin([np.sum(((m["x"] - xb) / (np.abs(m["x"]) + 1e-3)) ** 2) for m in lm]))
    best_post = res_post["modes"][0]["logp"]
    lines, flag = [f"likelihood vs posterior: {len(lm)} likelihood mode(s); posterior best is in likelihood mode {home}"], []
    for j, m in enumerate(lm):
        dpost = float(logpost(m["x"])) - best_post
        where = ", ".join(f"{n}={v:.4g}" for n, v in zip(names or range(len(m["x"])), m["x"], strict=True))
        lines.append(f"  L[{j}] dlogL={m['dlogp']:+.2f} dlogpost_there={dpost:+.2f} at {where}")
        if j != home and m["dlogp"] >= -tol:
            flag.append(f"likelihood mode {j} is {m['dlogp']:+.2f} from the best likelihood but {dpost:+.2f} in the "
                        f"posterior: the prior, not the data, chooses between them")
    shown = flag[:3] + ([f"(+{len(flag) - 3} more likelihood modes in the same state)"] if len(flag) > 3 else [])
    lines.append("PRIOR CHECK: " + ("PRIOR DECIDES -- " + "; ".join(shown) if flag else "the data separate the modes"))
    return "\n".join(lines), not flag


# ---------------------------------------------------------------------------------------------------
# PyMC adapter and the report the fitting paths attach to their output (hub finding §3.8)
# ---------------------------------------------------------------------------------------------------


class PyMCTarget:
    """A ``pm.Model`` as the plain functions :func:`mode_search` takes.

    Coordinates
    -----------
    ``z`` is the concatenation of the model's *value variables* (the unconstrained, transformed space NUTS
    samples).  ``jacobian`` is declared, not hidden:

    * ``jacobian=True`` (default): ``logp`` is the density NUTS samples, in ``z``.  Modes are those of the
      sampler's own target, so the chains' occupancy is compared with the same density.  Hierarchical
      models have a funnel neck whose density grows without bound in ``z`` (hub finding §4, step 2): an
      optimiser climbs it.  There is no PyMC option that bounds it; the neck shows up as a mode with
      ``logp`` far above the rest and no Laplace mass, and the report says "jacobian=True".
    * ``jacobian=False``: the density in the *constrained* parameters, evaluated at ``z``.  A mode at a
      bound of the support (a scale at 0) runs to ``|z| -> inf`` and is reported as a degenerate optimum.

    ``loglike`` is the PyMC ``datalogp`` (observed variables and potentials, no priors) in the same ``z``.
    Continuous free variables only.
    """

    def __init__(self, model, *, jacobian=True):
        import pymc as pm
        import pytensor
        import pytensor.tensor as pt
        from pytensor.graph.replace import graph_replace

        self.model, self.jacobian = model, bool(jacobian)
        ip = model.initial_point()
        self._vv = list(model.value_vars)
        self._rvs = list(model.free_RVs)
        if any(np.asarray(ip[v.name]).dtype.kind != "f" for v in self._vv):
            raise ValueError("mode search needs continuous free variables only")
        shapes = [np.asarray(ip[v.name]).shape for v in self._vv]
        sizes = [int(np.prod(s, dtype=int)) for s in shapes]
        edges = np.r_[0, np.cumsum(sizes)]
        self.dim = int(edges[-1])
        self.names = []
        for v, s, n in zip(self._vv, shapes, sizes, strict=True):
            self.names += [v.name] if n == 1 and not s else [f"{v.name}[{i}]" for i in range(n)]
        z = pt.dvector("z")
        sub = {v: z[edges[i]:edges[i + 1]].reshape(shapes[i]) for i, v in enumerate(self._vv)}
        comp = getattr(pm.pytensorf, "compile", None) or pytensor.function

        def build(expr):
            e = graph_replace(expr, sub, strict=False)
            return comp([z], [e, pt.grad(e, z, disconnected_inputs="ignore")])

        self._lp = build(model.logp(jacobian=self.jacobian, sum=True))
        self._ll = build(model.datalogp)
        # constrained -> unconstrained, for the draws and for the prior starts
        cs = [rv.type(f"c_{rv.name}") for rv in self._rvs]
        rv2c = dict(zip(self._rvs, cs, strict=True))
        parts = []
        for rv, c in zip(self._rvs, cs, strict=True):
            tr = model.rvs_to_transforms.get(rv)
            if tr is None:
                parts.append(pt.reshape(c, (-1,)))
            else:
                ins = [graph_replace(i, rv2c, strict=False) if hasattr(i, "owner") else i for i in rv.owner.inputs]
                parts.append(pt.reshape(tr.forward(c, *ins), (-1,)))
        self._fwd = comp(cs, pt.concatenate(parts))

    @staticmethod
    def _quiet(f, *a):
        with np.errstate(all="ignore"):
            return f(*a)

    def logp(self, z):
        v = float(self._quiet(self._lp, np.asarray(z, float))[0])
        return v if np.isfinite(v) else -np.inf

    def grad(self, z):
        return np.nan_to_num(np.asarray(self._quiet(self._lp, np.asarray(z, float))[1], float), nan=0.0)

    def loglike(self, z):
        v = float(self._quiet(self._ll, np.asarray(z, float))[0])
        return v if np.isfinite(v) else -np.inf

    def loglike_grad(self, z):
        return np.nan_to_num(np.asarray(self._quiet(self._ll, np.asarray(z, float))[1], float), nan=0.0)

    def to_z(self, constrained):
        """``constrained``: one value per free RV, in ``model.free_RVs`` order -> the ``z`` vector."""
        return np.asarray(self._quiet(self._fwd, *[np.asarray(c, float) for c in constrained]), float)

    def prior_starts(self, n, seed=0):
        """``n`` starting points drawn from the prior (never from the chains or a truth)."""
        import pymc as pm

        with self.model:
            pri = pm.sample_prior_predictive(draws=n, var_names=[rv.name for rv in self._rvs],
                                             random_seed=seed, return_inferencedata=False)
        z = np.array([self.to_z([pri[rv.name][i] for rv in self._rvs]) for i in range(n)])
        return z[np.all(np.isfinite(z), 1)]

    def draws_z(self, idata, max_draws=500):
        """Posterior draws as ``(draw, chain, dim)`` in ``z``, thinned evenly to at most ``max_draws``."""
        post = idata.posterior
        n = post.sizes["draw"]
        sel = np.unique(np.linspace(0, n - 1, min(n, max_draws)).round().astype(int))
        out = np.empty((len(sel), post.sizes["chain"], self.dim))
        vals = [np.asarray(post[rv.name].values) for rv in self._rvs]
        for a, i in enumerate(sel):
            for c in range(out.shape[1]):
                out[a, c] = self.to_z([v[c, i] for v in vals])
        return out


class ModeSearchConfig:
    """What the fitting paths run after sampling (``SamplingConfig.mode_search``).  ``None`` there = off."""

    def __init__(self, n_starts=16, jacobian=True, likelihood=False, seed=0, per_chain=5, max_draws=500):
        self.n_starts, self.jacobian, self.likelihood = int(n_starts), bool(jacobian), bool(likelihood)
        self.seed, self.per_chain, self.max_draws = int(seed), int(per_chain), int(max_draws)

    def __repr__(self):
        return ("ModeSearchConfig(" + ", ".join(f"{k}={v!r}" for k, v in vars(self).items()) + ")")


def search_pymc(model, idata=None, config=None):
    """Run the §A.1.8 mode search on a ``pm.Model`` (+ its ``InferenceData``); never raises.

    Returns a plain dict (JSON-serialisable, so it can ride in ``idata.attrs``): ``ok`` (True/False, or
    None when the search itself failed), ``text`` (the mandatory output lines), ``n_modes``, ``jacobian``,
    ``prior_decides`` (None unless ``config.likelihood``), ``nfev``, ``error``.  A FAIL does not block
    anything: the caller warns and keeps the result.
    """
    cfg = config or ModeSearchConfig()
    out = dict(ok=None, text="", n_modes=0, jacobian=cfg.jacobian, prior_decides=None, nfev=None, error=None,
               config=repr(cfg))
    try:
        tg = PyMCTarget(model, jacobian=cfg.jacobian)
        starts = tg.prior_starts(cfg.n_starts, cfg.seed)
        if len(starts) == 0:
            raise ValueError("no finite prior start (improper prior?)")
        sc = np.nanstd(starts, 0)
        sc = np.where(np.isfinite(sc) & (sc > 0), sc, 1.0)
        draws = None if idata is None else tg.draws_z(idata, cfg.max_draws)
        res = mode_search(tg.logp, starts, grad=tg.grad, scale=sc, draws=draws, names=tg.names,
                          per_chain=cfg.per_chain)
        text, ok = verdict(res)
        text = f"[density in unconstrained z, jacobian={cfg.jacobian}]\n" + text
        if cfg.likelihood:
            res_l = mode_search(tg.loglike, starts, grad=tg.loglike_grad, scale=sc, names=tg.names)
            t2, prior_ok = like_vs_post(res_l, res, tg.logp, names=tg.names)
            text += "\n" + t2
            out["prior_decides"] = not prior_ok
        out.update(ok=bool(ok), text=text, n_modes=len(res["modes"]), nfev=res["nfev"])
    except Exception as exc:  # a gate is never the reason a fit fails; the error stays visible in the result
        out.update(ok=None, error=f"{type(exc).__name__}: {exc}", text=f"mode search ERROR: {type(exc).__name__}: {exc}")
    return out
