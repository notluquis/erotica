r"""Wilson (1975) lowered-isothermal model: projected surface density and point-process helpers.

The Wilson model is the :math:`g = 2` member of the family of lowered isothermal distribution
functions (Gomez-Leyton & Velazquez 2014, `2014JSMTE..04..006G`; Gieles & Zocchi 2015,
`2015MNRAS.454..576G`),

.. math::
    f(\varepsilon) \propto e^{\varepsilon} - \sum_{n=0}^{g-1} \frac{\varepsilon^n}{n!}
    \quad (\varepsilon > 0),

where :math:`\varepsilon = \psi - v^2/2` is the (dimensionless) binding energy per unit mass.
:math:`g = 1` is the King (1966) model, :math:`g = 2` is Wilson (1975, `1975AJ.....80..175W`),
whose envelope is more extended but still has a finite tidal cutoff. McLaughlin & van der Marel
(2005, `2005ApJS..161..304M`) fit King, Wilson and power-law families to a large sample of
clusters and find Wilson fits clusters of any age as well as or better than King in the majority
of cases.

**This is a dynamical model, not a fitting formula.** Unlike :func:`~erotica.analysis.structure.king_profile`
(King 1962, empirical), the profile is the solution of Poisson's equation with the density integrated
from the distribution function above,

.. math::
    \hat\rho(\psi) = \int_0^{\sqrt{\psi}} f(\psi - s^2)\, 2 s^2\, ds, \qquad
    \hat\psi'' + \frac{2}{\hat r}\hat\psi' = -9\,\frac{\hat\rho(\hat\psi)}{\hat\rho(W_0)},

with :math:`\hat\psi(0) = W_0` and :math:`\hat\psi'(0) = 0`. Radii are in units of the scale radius
:math:`r_0` (the same convention as LIMEPY, so :math:`\hat r_t = r_t/r_0`). The projected
surface density is

.. math:: \Sigma(R) = 2 \int_R^{r_t} \frac{\rho(r)\, r\, dr}{\sqrt{r^2 - R^2}}.

There is **no closed form**: thread G7 described the Wilson :math:`\Sigma(r)` as "closed-form
integral", which is false for any :math:`g \ge 1` except by a numerical solve. The fit therefore
uses a table (below), and the test suite measures what that costs.

Why not LIMEPY
--------------
LIMEPY (Gieles & Zocchi 2015, MIT, ``astro-limepy`` on PyPI; the unrelated PyPI project named
``limepy`` is a LimeSurvey client) implements the same model. It is **not used as a dependency**:
its last release (1.2) is from 2023 and, measured on 2026-10-27, it fails on current SciPy
(``ode.set_integrator(nsteps=1e6)`` needs an ``int``, and ``dopri5._solout`` rejects its callback
signature on SciPy 1.18). It runs, after the first of those two patches, on SciPy 1.14. It is used
here as the **external oracle** of the tests (frozen reference values, plus an ``importorskip``
cross-check), not as a runtime dependency.

Table
-----
For fitting, :math:`S(s; W_0) = \Sigma(s\, r_t)/\Sigma(0)` and its cumulative
:math:`C(s) = \int_0^s S\, s'\, ds'` are tabulated once (lazily, ~seconds) on a grid
:math:`W_0 \in [W_{0,\min}, W_{0,\max}]` and a common grid in :math:`s = R/r_t`, and interpolated
linearly in :math:`W_0`. Working in :math:`s` rather than in :math:`R/r_0` keeps the truncation
edge at the same grid index for every :math:`W_0`.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import gammainc

W0_MIN = 1.0
W0_MAX = 10.0
W0_STEP = 0.25

_GL_X, _GL_W = np.polynomial.legendre.leggauss(64)


def lowered_isothermal_density(psi, g: int = 2):
    r"""Dimensionless density :math:`\hat\rho(\psi)` of the lowered isothermal model.

    Computed as :math:`\int_0^{\sqrt\psi} f(\psi - s^2)\,2 s^2\,ds` with
    :math:`f(\varepsilon) = e^{\varepsilon} P(g, \varepsilon)` (:math:`P` the regularised lower
    incomplete gamma function, which equals :math:`e^\varepsilon - \sum_{n<g}\varepsilon^n/n!`
    without the cancellation at small :math:`\varepsilon`). Normalisation is arbitrary; only
    ratios enter the Poisson equation. Returns 0 for :math:`\psi \le 0`.
    """
    psi = np.asarray(psi, dtype=float)
    out = np.zeros_like(psi)
    pos = psi > 0
    p = psi[pos]
    if p.size:
        half = 0.5 * np.sqrt(p)[:, None]
        s = half * (_GL_X[None, :] + 1.0)
        eps = p[:, None] - s**2
        f = np.exp(eps) * gammainc(float(g), np.maximum(eps, 0.0))
        out[pos] = np.sum(half * _GL_W[None, :] * f * 2.0 * s**2, axis=1)
    return out


@lru_cache(maxsize=256)
def solve_lowered_isothermal(W0: float, g: int = 2):
    """Solve Poisson's equation for the :math:`g` model with central potential ``W0``.

    Returns ``(r_t, psi_of_r)`` with ``r_t`` in units of :math:`r_0` and ``psi_of_r`` a vectorised
    callable (zero beyond ``r_t``). Tolerances are 1e-10 relative.
    """
    rho0 = float(lowered_isothermal_density(np.array([W0]), g)[0])

    def rhs(r, y):
        psi, dpsi = y
        rho = lowered_isothermal_density(np.array([psi]), g)[0]
        return [dpsi, -9.0 * rho / rho0 - 2.0 * dpsi / r]

    def hit(r, y):
        return y[0]

    hit.terminal = True
    hit.direction = -1
    r_start = 1e-6
    y0 = [W0 - 1.5 * r_start**2, -3.0 * r_start]
    sol = solve_ivp(
        rhs,
        (r_start, 1e5),
        y0,
        method="DOP853",
        events=hit,
        rtol=1e-10,
        atol=1e-12,
        dense_output=True,
    )
    if sol.t_events[0].size == 0:
        raise RuntimeError(f"Wilson/lowered-isothermal model with W0={W0}, g={g} did not truncate")
    r_t = float(sol.t_events[0][0])

    def psi_of_r(r):
        r = np.asarray(r, dtype=float)
        out = np.zeros_like(r)
        inside = r < r_t
        rr = np.maximum(r[inside], r_start)
        out[inside] = np.maximum(sol.sol(rr)[0], 0.0)
        return out

    return r_t, psi_of_r


_S_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 1.0, 999)])
_U_X, _U_W = np.polynomial.legendre.leggauss(96)


def projected_profile(W0: float, g: int = 2, s=None):
    r"""Exact (per-call ODE solve) projected profile.

    Returns ``(r_t, s, S, C)``: truncation radius in units of :math:`r_0`; the grid
    :math:`s = R/r_t`; :math:`S = \Sigma(R)/\Sigma(0)`; and
    :math:`C(s) = \int_0^s S s' ds'` (so the number inside :math:`R` is
    :math:`2\pi\,k\,r_t^2\,C`).
    """
    r_t, psi = solve_lowered_isothermal(float(W0), int(g))
    s = _S_GRID if s is None else np.asarray(s, dtype=float)
    R = s * r_t
    umax = np.sqrt(np.maximum(r_t**2 - R**2, 0.0))
    # Line-of-sight nodes u = sinh(x * asinh(umax)) : dense near u = 0, where the integrand varies
    # on the scale r_0 = 1, and sparse out to umax ~ r_t >> r_0. Uniform Gauss-Legendre nodes in u
    # were off by 6 % at W0 = 9 (r_t/r_0 = 2132), caught against scipy.integrate.quad.
    top = np.arcsinh(umax)[:, None]
    x = 0.5 * (_U_X[None, :] + 1.0)
    u = np.sinh(x * top)
    wgt = 0.5 * _U_W[None, :] * top * np.cosh(x * top)
    r = np.sqrt(R[:, None] ** 2 + u**2)
    rho = lowered_isothermal_density(psi(r.ravel()), g).reshape(r.shape)
    sigma = 2.0 * np.sum(wgt * rho, axis=1)
    if s[0] != 0.0:
        raise ValueError("the grid s must start at 0 (it normalises to the central value)")
    sigma0 = sigma[0]
    S = sigma / sigma0
    C = CubicSpline(s, S * s).antiderivative()(s)
    return r_t, s, S, C


_X_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 5e4, 2000)])


def _profile_on_x_grid(W0: float):
    """``(r_t, S(x), N(x))`` on ``_X_GRID`` (x = R/r_0); S = 0 and N = N_total beyond r_t."""
    r_t, _ = solve_lowered_isothermal(float(W0), 2)
    inside = _X_GRID < r_t
    s = np.concatenate([_X_GRID[inside] / r_t, [1.0]])
    _, _, S_s, C_s = projected_profile(W0, 2, s=s)
    S = np.zeros_like(_X_GRID)
    N = np.full_like(_X_GRID, r_t**2 * C_s[-1])
    S[inside] = S_s[:-1]
    N[inside] = r_t**2 * C_s[:-1]
    return r_t, S, N


@lru_cache(maxsize=1)
def _table():
    w0 = np.arange(W0_MIN, W0_MAX + 0.5 * W0_STEP, W0_STEP)
    rt, S, N = zip(*(_profile_on_x_grid(float(w)) for w in w0), strict=False)
    return w0, np.log(np.asarray(rt)), np.asarray(S), np.asarray(N)


def _interp_rows(W0: float, rows: np.ndarray) -> np.ndarray:
    w0 = _table()[0]
    if not (W0_MIN <= W0 <= W0_MAX):
        raise ValueError(f"W0={W0} outside the tabulated range [{W0_MIN}, {W0_MAX}]")
    x = (W0 - W0_MIN) / W0_STEP
    i = min(int(np.floor(x)), len(w0) - 2)
    t = x - i
    return (1.0 - t) * rows[i] + t * rows[i + 1]


def wilson_truncation_radius(W0, r_0):
    """Tidal (truncation) radius ``r_t = r_0 * rhat_t(W0)`` of the Wilson model."""
    return r_0 * float(np.exp(_interp_rows(float(W0), _table()[1][:, None])[0]))


def wilson_surface_density(radius, *, k, r_0, W0, b=0.0):
    r"""Wilson (1975) projected surface density :math:`k\,S(R/r_0; W_0) + b`.

    Parameters
    ----------
    radius : array_like
        Projected radius, same unit as ``r_0`` (arcmin by the package convention).
    k : float
        Central surface density (stars per square unit of ``radius``): :math:`\Sigma(0)` of the
        cluster term.
    r_0 : float
        Scale radius (the LIMEPY convention); the truncation radius is
        :func:`wilson_truncation_radius`.
    W0 : float
        Dimensionless central potential, in ``[W0_MIN, W0_MAX]``.
    b : float, optional
        Uniform background added everywhere (also beyond :math:`r_t`).
    """
    x = np.asarray(radius, dtype=float) / r_0
    S = np.interp(x, _X_GRID, _interp_rows(float(W0), _table()[2]))
    cluster = np.where(x < wilson_truncation_radius(W0, 1.0), S, 0.0)
    return k * cluster + b


def wilson_expected_count(k, b, r_0, W0, field_radius):
    r"""Expected count in a circular field, :math:`\Lambda = 2\pi k\,r_0^2\,N(R_f/r_0) + \pi b R_f^2`.

    ``N(x) = \int_0^x S x' dx'`` is held at its total beyond :math:`r_t`.
    """
    N = np.interp(field_radius / r_0, _X_GRID, _interp_rows(float(W0), _table()[3]))
    return 2.0 * np.pi * k * r_0**2 * float(N) + np.pi * b * field_radius**2


def wilson_log_likelihood(radii, *, k, r_0, W0, b, field_radius):
    r"""Unbinned point-process log-likelihood :math:`\sum_i \ln[2\pi r_i \Sigma(r_i)] - \Lambda`."""
    sigma = wilson_surface_density(radii, k=k, r_0=r_0, W0=W0, b=b)
    if np.any(sigma <= 0):
        return -np.inf
    return float(
        np.sum(np.log(2.0 * np.pi * radii * sigma))
        - wilson_expected_count(k, b, r_0, W0, field_radius)
    )
