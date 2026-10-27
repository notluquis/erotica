r"""The K x W isochrone model: :class:`IsochroneFitter` with an error-scale parameter, fitted in a
magnitude window.


**Scope: read this before quoting a number from this class.**

* This is a model of the **pre-main-sequence in a window, G 14-19**, with the catalogue errors **scaled
  by a free factor k**. It is **not the age of the cluster**. The log t it returns is the age at which
  the stars *in that window* are best described by this model.
* **Why a window.** One isochrone does not describe the main sequence and the pre-main-sequence of
  NGC 6383 at once (the MS vs PMS tension: a single-age fit of the whole CMD is falsified, and K x F, the
  scaled-error model on all stars, fails the predictive check). The window is what removes the tension;
  it does not resolve it.
* **The log t is conditional on the model and on the prior on dm.** The likelihood alone has a second
  hump at log t ~ 6.45, which is the bound dm >= 9.50 of the prior range; the cluster parallax excludes
  that distance (hub finding D15). Over log t 5.80-6.56 the likelihood alone is flat to within -2.7,
  so the prior on dm is what fixes the age. A wrong centre of that prior is not in the interval.
* **k absorbs the model mismatch.** Whether it is an error underestimate or a spread that scales with the
  error is not identified, and a k > 1 is what makes the check pass; it is not a measurement of the errors.
* **The coverage was calibrated on simulations of the same model**, with n = 8 replicates per cell. It is
  "no gross under-coverage detected", not "calibrated", and it says nothing about a different truth.

What changes with respect to :class:`~erotica.analysis.IsochroneFitter` (everything else, the
EEP-interpolated isochrones, the Chabrier IMF, the Offner/Duchene-Kraus binaries, the extinction law,
the field component and the conditional likelihood of the window, is inherited unchanged):

* **K**: the width of star :math:`i` is :math:`s_i^2 = (k\,e_i)^2 + \sigma_{\rm floor}^2 +
  \sigma_{\rm int}^2` in both axes, with :math:`k \ge 1` a free scale of the catalogue's errors and
  :math:`\sigma_{\rm int}` the isotropic intrinsic width. On NGC 6383's PMS window the fitted
  :math:`k` is 2.45 and :math:`\sigma_{\rm int}` 0.094 mag. What :math:`k` *is* (an underestimate of the
  Gaia DR3 photometric errors, or a physical spread that scales with the error) is **not**
  identified by this model; the age does not depend on that reading (hub finding
  ``isochrone-model-fix.md``, R42-03), but the model does depend on having the term: without it the
  ridge residuals at G 17.5-19 fail the posterior predictive check.
* **W**: pass ``mag_window=(14.0, 19.0)`` to :meth:`~IsochroneFitter.setup`. The whole-CMD single-age
  fit is falsified for this cluster (a PMS and an MS that do not share an age under one isochrone), so
  the age this model returns is **the age of the stars in the window**, never "the age of the
  cluster".

Parameters, in the order the compiled function takes them: ``(met, loga, dm, Av, sigma_int, k, f_bg)``.
Priors: those of the parent plus :math:`k \sim U(1, 6)`; the upper bound is the search bound, not a
physical statement.

**Convention inherited and measured**: the detected fraction :math:`F(\theta)` of the window
(:meth:`~IsochroneFitter._detected_fraction`) is computed with the intrinsic and floor widths and with
the error at the segment midpoint from the polynomial fitted to the catalogue's **unscaled** errors
(``_e_mag_coef``), not with :math:`k\,e_i`. At the window's faint edge that term is the smaller one
(0.01 mag against :math:`\sigma_{\rm int}` 0.094 for NGC 6383), so its effect on :math:`F` is small
`[I]`, not measured. The coverage numbers below were obtained with it.

**What was measured with this likelihood** (hub finding ``isochrone-intervals-nuts.md`` §2.10-§2.16,
research code ``tools/validation/isochrone_modelfix`` on branch ``isochrone-model-fix``; this class is the
same function, checked to 1e-10 in ``tests/test_isochrone_kw.py``):

* injection-recovery, cell A (8 replicates, 201 stars, G 14-19, truths off every MIST node): log t inside
  the 68 % interval in 6/8 and the 90 % in 7/8 (6 of 8 fits converged); [Fe/H], :math:`A_V` and
  :math:`k` under-cover (1/8, 4/8, 3/8). Not "calibrated": a rule that passes with probability 0.58 even
  at true coverage 0.50 / 0.75.
* cell C (log t 6.43, [Fe/H] +0.15): NUTS does **not** converge (0/8, R-hat of log t 1.76-2.76) with the
  chain starts of :meth:`find_start`. A fit there is not to be reported without passing the gate.
* NGC 6383, 201 stars: log t = 5.965 (+0.034 / -0.032), 0.92 (+0.08 / -0.07) Myr, 4 x 2000, R-hat <= 1.003.
  Conditional on the model, MIST v1.2, and the prior on dm (which fixes the age along an age-distance
  ridge: log L alone is flat within -2.7 over log t 5.80-6.56).

The width term is the error-scale generalisation of the parent's. Whether a published isochrone fitter
carries the same term was not surveyed here (see the isochrone row of the hub's ``model-landscape.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._isochrone import IsochroneFitter, _require_pymc

# Hard bounds of the nuisance parameters in the search (L-BFGS-B) and the chain-start clipping.
_NUISANCE_BOUNDS = {"sigma_int": (1e-4, 2.0), "k": (1.0, 6.0), "f_bg": (1e-4, 0.5)}


class IsochroneFitterKW(IsochroneFitter):
    r"""Isochrone fitting with a free scale :math:`k` of the catalogue errors (see the module
    docstring).

    **A model of the PMS in the window G 14-19 with errors scaled by k; not the age of the cluster.**
    Its log t is conditional on the model and on the prior on dm; ``k`` absorbs the mismatch; coverage
    was calibrated on simulations of the same model with n = 8. The MS vs PMS tension is the reason for
    the window. See the module docstring, "Scope".

    Workflow, as :class:`IsochroneFitter` with the window::

        f = IsochroneFitterKW(grid=grid, magnitude=..., color=..., dm_mu=..., dm_sigma=...)
        f.setup(members, prob_threshold=0.0, mag_window=(14.0, 19.0))
        idata = f.fit(draws=2000, tune=1000, chains=4)

    ``find_start`` is the mode search that was measured (a [Fe/H] x log t lattice with ``(dm, A_V)``
    optimised at a wide width, the best ``SEARCH_POLISH`` polished on all seven parameters, the
    objective including the ``dm`` prior); the chain starts are the measured over-dispersed ones, along
    the age ridge of the Laplace covariance at the mode. ``SEARCH_*`` and ``START_DLOGA`` are the
    measured values, not tuned here.
    """

    LIKELIHOOD_VERSION = "unbinned-eep-kw-v1"
    #: Prior bounds of the error scale ``k`` (U(1, 6)).
    K_RANGE = (1.0, 6.0)
    #: [Fe/H] values of the search lattice (those inside the grid's range); ``None`` = every node.
    SEARCH_FEH_LATTICE: tuple[float, ...] | None = (-1.0, -0.5, 0.0, 0.5)
    #: Every ``SEARCH_AGE_STRIDE``-th age node inside the prior is a lattice point.
    SEARCH_AGE_STRIDE = 2
    SEARCH_POLISH = 4
    #: Where each lattice point's ``(dm, A_V)`` optimisation starts in ``A_V``.
    SEARCH_START_AV = 1.0
    #: Chain starts: offsets in log t along the ridge of the Laplace covariance (cycled).
    START_DLOGA = (-0.17, -0.06, 0.06, 0.17)

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def _star_loglike(  # type: ignore[override]
        self,
        met: Any,
        loga: Any,
        dm: Any,
        Av: Any,
        sigma_int: Any,
        f_bg: Any,
        xp: Any = np,
        k: Any = 1.0,
    ) -> Any:
        """Per-star log-likelihood ``(N,)``; ``k = 1`` is the parent's likelihood."""
        return super()._star_loglike(met, loga, dm, Av, sigma_int, f_bg, xp, error_scale=k)

    def loglike(  # type: ignore[override]
        self,
        met: float,
        loga: float,
        dm: float,
        Av: float,
        sigma_int: float = 0.0,
        f_bg: float = 0.0,
        k: float = 1.0,
    ) -> float:
        """Total (weighted) log-likelihood at fixed parameters, in NumPy."""
        v = self._star_loglike(met, loga, dm, Av, sigma_int, f_bg, np, k)
        return float(np.sum(self._star_weights * v))

    def _compiled_loglike(self, mode: str | None = None) -> Any:  # type: ignore[override]
        """Compiled ``(log L, d log L / d params)`` of ``(met, loga, dm, Av, sigma_int, k, f_bg)``."""
        import importlib.util

        import pytensor
        import pytensor.tensor as pt

        v = [pt.dscalar(n) for n in ("met", "loga", "dm", "Av", "sigma_int", "k", "f_bg")]
        met, loga, dm, Av, s, k, f = v
        ll = pt.sum(self._star_weights * self._star_loglike(met, loga, dm, Av, s, f, pt, k))
        if mode == "JAX" and importlib.util.find_spec("jax") is None:
            mode = None
        fn = pytensor.function(v, [ll, *pytensor.grad(ll, v)], mode=mode)
        return lambda x: (lambda out: (float(out[0]), np.array(out[1:], float)))(fn(*x))

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def build_model(self) -> Any:
        """The parent's PyMC model plus ``k ~ U(K_RANGE)``; ``k`` is created after ``f_bg``."""
        pm = _require_pymc()
        import pytensor.tensor as pt

        if self._nodes is None or self._obs_mag is None:
            raise RuntimeError(
                "Call setup() (or setup(precompute_grid=False) + build_grid()/load_grid()) first."
            )
        loga_min, loga_max = self.loga_range
        if loga_min < self._node_loga[0] - 1e-9 or loga_max > self._node_loga[-1] + 1e-9:
            raise ValueError(
                f"loga_range {self.loga_range} reaches beyond the node table "
                f"[{self._node_loga[0]}, {self._node_loga[-1]}]; rebuild it with build_grid()."
            )
        met_min, met_max = self._met_bounds()
        weights = self._star_weights

        with pm.Model() as model:
            mname = self._met_name
            if len(self._node_logz) > 1:
                met = pm.Uniform(mname, lower=met_min, upper=met_max)
            else:
                met = pm.Deterministic(mname, pt.as_tensor_variable(met_min))
            loga = pm.Uniform("loga", lower=loga_min, upper=loga_max)
            dm = pm.TruncatedNormal(
                "dm",
                mu=self.dm_mu,
                sigma=self.dm_sigma,
                lower=self.dm_range[0],
                upper=self.dm_range[1],
            )
            Av = pm.Uniform("Av", lower=self.Av_range[0], upper=self.Av_range[1])
            sigma_int = pm.HalfNormal("sigma_int", sigma=0.05)
            f_bg = pm.Beta("f_bg", alpha=1.0, beta=19.0)
            # k >= 1 because the hypothesis is under-estimated errors; 6 is the search bound
            k = pm.Uniform("k", lower=self.K_RANGE[0], upper=self.K_RANGE[1])

            def _logp(
                value: Any, met: Any, loga: Any, dm: Any, Av: Any, s: Any, f: Any, k: Any
            ) -> Any:
                return weights * self._star_loglike(met, loga, dm, Av, s, f, pt, k)

            pm.CustomDist(
                "y",
                met,
                loga,
                dm,
                Av,
                sigma_int,
                f_bg,
                k,
                logp=_logp,
                observed=np.zeros(self._N_obs),
            )
        return model

    # ------------------------------------------------------------------
    # Mode search and chain starts
    # ------------------------------------------------------------------

    def find_start(  # type: ignore[override]
        self, n_chains: int = 4, rng: np.random.Generator | None = None, mode: str | None = None
    ) -> dict:
        r"""Locate the dominant mode of the log posterior and draw over-dispersed chain starts.

        1. A lattice of ``SEARCH_FEH_LATTICE`` x every ``SEARCH_AGE_STRIDE``-th age node, with
           ``(dm, A_V)`` optimised at the wide width (:math:`\sigma_{\rm int}` 0.05, :math:`k` 1,
           :math:`f_{\rm bg}` 0.02), where the likelihood is smooth. The objective is
           :math:`\ln L - \tfrac12 z_{dm}^2`: the ``dm`` prior is part of what is maximised.
        2. The ``SEARCH_POLISH`` best points are polished (L-BFGS-B, bounded) on all seven parameters;
           the best is the mode. ``runner_up`` holds the others.
        3. The Laplace covariance :math:`S` at the mode (central differences of the gradient of that
           objective). Each chain starts at the mode moved along the regression line of
           :math:`({\rm [Fe/H]}, dm, A_V)` on :math:`\log t` (:math:`S_{\cdot 1}/S_{11}`) by
           ``START_DLOGA`` in log t, the nuisance parameters at the mode times U(0.7, 1.4).

        **A mode search is not a proof of uniqueness**: it found the old stretch of the age ridge, 0.2
        dex from the truth, with log L 0.6 to 46 units below the truth's, in 3 of 8 synthetic replicates of cell A and
        in 3 of 3 replicates of the binary experiment prepared with it (hub finding
        ``isochrone-intervals-nuts.md`` §2.9, §2.17.1), and NUTS did not converge in the harder cell. ``mode`` is the search's maximum, and a
        reported age needs the profile of the posterior in log t as well (§2.16).

        Returns ``mode``, ``loglike`` (:math:`\ln L` at the mode), ``logpost`` (with the ``dm``
        prior), ``runner_up``, ``at_bound``, ``local_sd`` (marginal Laplace sd of the first four),
        ``laplace_sd`` (all seven, NaN where :math:`-H` is not positive definite), ``laplace_cov``,
        ``ridge_vector`` and ``chain_starts`` (a list of dicts for ``initvals``).
        """
        from scipy.optimize import minimize

        rng = np.random.default_rng() if rng is None else rng
        if self._nodes is None:
            raise RuntimeError("Call setup() or build_grid() first.")
        zlo, zhi = self._met_bounds()
        names = (self._met_name, "loga", "dm", "Av", "sigma_int", "k", "f_bg")
        eps = 1e-9 if self._met_kind == "feh" and zhi - zlo > 2e-9 else 0.0  # one metallicity: lo == hi
        lo = np.array(
            [
                zlo + eps,
                self.loga_range[0],
                self.dm_range[0],
                self.Av_range[0],
                *[_NUISANCE_BOUNDS[n][0] for n in names[4:]],
            ]
        )
        hi = np.array(
            [
                zhi - eps,
                self.loga_range[1],
                self.dm_range[1],
                self.Av_range[1],
                *[_NUISANCE_BOUNDS[n][1] for n in names[4:]],
            ]
        )
        lo[5], hi[5] = self.K_RANGE
        fl = self._compiled_loglike(mode)

        def objective(y: np.ndarray) -> tuple[float, np.ndarray]:
            v, g = fl(y)
            z = (y[2] - self.dm_mu) / self.dm_sigma
            g = g.copy()
            g[2] -= z / self.dm_sigma
            return v - 0.5 * z * z, g

        ages = [a for a in self._node_loga if lo[1] <= a <= hi[1]][:: self.SEARCH_AGE_STRIDE]
        zs_all = self._met_from_coord(self._node_logz)
        if self.SEARCH_FEH_LATTICE is not None and self._met_kind == "feh":
            zs = [x for x in self.SEARCH_FEH_LATTICE if zlo <= x <= zhi]
        else:
            zs = []
        if not zs:  # no lattice, or none of it inside a narrow grid: every node
            zs = [float(z) for z in np.clip(zs_all, lo[0], hi[0])]
        wide = [0.05, 1.0, 0.02]  # sigma_int, k, f_bg of the lattice stage
        cands = []
        for z in zs:
            for a in ages:

                def nll2(y: np.ndarray, z: float = float(z), a: float = float(a)) -> tuple:
                    v, g = objective(np.array([z, a, y[0], y[1], *wide]))
                    return -v, -g[2:4]

                r = minimize(
                    nll2,
                    [self.dm_mu, self.SEARCH_START_AV],
                    jac=True,
                    method="L-BFGS-B",
                    bounds=list(zip(lo[2:4], hi[2:4], strict=True)),
                )
                cands.append((float(r.fun), float(z), float(a), *map(float, r.x)))
        cands.sort(key=lambda c: c[0])

        def nll(y: np.ndarray) -> tuple:
            v, g = objective(y)
            return -v, -g

        polished = []
        start_tail = [0.03, 1.5, 0.02]
        for c in cands[: self.SEARCH_POLISH]:
            y0 = np.clip(np.array([*c[1:], *start_tail]), lo, hi)
            r = minimize(
                nll, y0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True))
            )
            polished.append((-float(r.fun), np.asarray(r.x, float)))
        polished.sort(key=lambda p: -p[0])
        best_lp, x = polished[0]
        at_bound = [
            n
            for n, v, a, b in zip(names, x, lo, hi, strict=True)
            if min(v - a, b - v) < 1e-3 * (b - a)
        ]

        n = len(x)
        h = np.array([1e-3] * 4 + [1e-4] * (n - 4))
        H = np.empty((n, n))
        for i in range(n):
            e = np.zeros(n)
            e[i] = h[i]
            H[i] = (objective(x + e)[1] - objective(x - e)[1]) / (2 * h[i])
        try:
            S = np.linalg.inv(-0.5 * (H + H.T))
        except np.linalg.LinAlgError:  # a flat direction (a one-metallicity grid): zero variance there
            S = np.linalg.pinv(-0.5 * (H + H.T))
        v = S[:4, 1] / S[1, 1]
        margin = 1e-3 * (hi[:4] - lo[:4])
        starts = []
        for i in range(n_chains):
            d = self.START_DLOGA[i % len(self.START_DLOGA)]
            s = np.clip(x[:4] + d * v, lo[:4] + margin, hi[:4] - margin)
            st = {nm: float(val) for nm, val in zip(names[:4], s, strict=True)}
            for nm, val in zip(names[4:], x[4:], strict=True):
                a, b = _NUISANCE_BOUNDS[nm] if nm != "k" else self.K_RANGE
                st[nm] = float(np.clip(val * rng.uniform(0.7, 1.4), a + 1e-3, b - 1e-3))
            if len(self._node_logz) == 1:
                st.pop(self._met_name)
            starts.append(st)
        with np.errstate(invalid="ignore"):
            sd = np.sqrt(np.diag(S))
        return {
            "mode": dict(zip(names, x.tolist(), strict=True)),
            "loglike": float(fl(x)[0]),
            "logpost": float(best_lp),
            "runner_up": [(float(lp), xx.tolist()) for lp, xx in polished[1:]],
            "at_bound": at_bound,
            "local_sd": {
                nm: float(s_) for nm, s_ in zip(names[:4], sd[:4], strict=True) if np.isfinite(s_)
            },
            "laplace_sd": dict(zip(names, sd.tolist(), strict=True)),
            "laplace_cov": S.tolist(),
            "ridge_vector": v.tolist(),
            "chain_starts": starts,
        }

    # ------------------------------------------------------------------
    # Posterior predictive
    # ------------------------------------------------------------------

    def posterior_cmd(  # type: ignore[override]
        self, idata: Any, *, num_samples: int = 20, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Synthetic CMDs from posterior draws, with each draw's own error scale ``k``.

        The parent's ``posterior_cmd`` would draw with the catalogue errors unscaled (:math:`k = 1`),
        stars 2.4 times too sharp for this model, so it is overridden rather than inherited.
        """
        import arviz as az

        rng = np.random.default_rng() if rng is None else rng
        vn = [self._met_name, "loga", "dm", "Av", "sigma_int", "k"]
        ds = az.extract(idata, num_samples=num_samples, var_names=vn)
        cols = [np.asarray(ds[v]).ravel() for v in vn]
        N_out = self._N_obs if self._N_obs else 500
        cmds = []
        for met_v, loga_v, dm_v, Av_v, s_v, k_v in zip(*cols, strict=True):
            cmds.append(
                self._draw_stars(
                    float(met_v),
                    float(loga_v),
                    float(dm_v),
                    float(Av_v),
                    float(s_v),
                    N_out,
                    rng,
                    error_scale=float(k_v),
                )
            )
        return self._obs_mag, self._obs_col, cmds  # type: ignore[return-value]
