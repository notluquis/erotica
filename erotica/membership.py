r"""Cluster membership with a rule fixed in advance: generative mixture, calibration, two-line rule, propagation.

WHY THIS MODULE EXISTS
----------------------
Four different quantities are called "membership probability" in the open-cluster literature (hub
``agent-findings/membership-rule-literature-2026-11-03.md`` §2): the posterior of a generative mixture,
the stability frequency under resampling (UPMASK, the ``f_i`` of :mod:`erotica.calibration`), the
HDBSCAN proximity, and the probability of appearing in a list. Only the first is P(member | data), and
only under its model. This module computes that one, and nothing else.

It implements step 2-4 of the programme rule (hub ``methodology.md`` §A.3, plan in the finding §6):
assign ``p_i`` with a cluster + field mixture whose selection sits in the likelihood, calibrate it by
injection, classify with two independent lines and fixed thresholds, and propagate ``p_i`` -- never a
cut -- downstream. Detection (step 1: HDBSCAN plus an empty-field control) is *not* here; the mixture
only shares probability inside a cluster that detection already found.

THE MODEL
---------
Per star ``i`` with observed proper motion ``μ_i`` (2-vector, measurement covariance ``C_i``) and
parallax ``ϖ_i`` (error ``e_i``), conditioned on its errors and on having been selected:

.. math::

    L_i = (1 - η)\, \frac{N(μ_i; m_c, C_i + σ_c^2 I)}{P_A^{c}(C_i)} \frac{N(ϖ_i; ϖ_c, e_i^2 + s_c^2)}{W^{c}(e_i)}
        + η\, \frac{N(μ_i; m_f, C_i + Σ_f)}{P_A^{f}(C_i)} \frac{N(ϖ_i; ϖ_f, e_i^2 + s_f^2)}{W^{f}(e_i)}

* the cluster's intrinsic proper-motion dispersion ``σ_c`` is isotropic (internal velocity dispersion
  in the tangent plane); its parallax depth ``s_c`` is free;
* the field is a broad 2-D Gaussian in proper motion (full covariance ``Σ_f``) and a Gaussian in
  parallax. This is the minimal parametric field; the empirical local template of the rule
  (``EmpiricalField.from_sideband``) is pending, see below;
* ``P_A`` is the probability that a star of that component, *with that star's errors*, lands inside the
  proper-motion acceptance region (QMC: through the Gaussian for the cluster, over the region for the
  field, see :class:`Truncation`), and ``W`` the
  same for the parallax window (closed form). **Every component is divided by its own P_A and W**
  (hub ``methodology.md`` §A.1.4; C5 §15). Dividing only one component, or none, biases the fraction
  and the cluster dispersion -- the ``nopa`` mutation in the tests;
* the acceptance probability of the observed point, common to both components, cancels;
* ``η`` is the field fraction *among selected stars*, constant over the sample (the C5 §15 choice).
  A population where the selection depends on the errors differently for members and field would make
  the observed fraction depend on ``e_i``; that is not modelled.

Ported in structure from the hub analysis script ``agent-findings/scripts/c5_mixture.py::Model``
(JAX likelihood with exact gradient and Hessian, P_A by QMC on a smooth acceptance frontier), without
its NGC 6383 specifics (the κ/shear design matrix, the HR24 flag, the CMD axis, Theia 1645).

FITTING AND THE MODE SEARCH
---------------------------
:meth:`MixtureModel.fit` maximises the log posterior (the likelihood plus an optional log-normal prior
on the number of field stars, the C5 ``prior_Nf``) through :func:`erotica.analysis.modesearch.mode_search`:
``n_starts`` L-BFGS-B optimisations from starts drawn uniformly inside the parameter box (the box *is*
the prior here), ridge starts, clustering into modes and a Laplace mass per mode. The verdict text and the
mode table are kept on the fit (hub ``methodology.md`` §A.1.8). Uniform starts in the box are a
convention, not a measured choice.

THE TWO-LINE RULE
-----------------
:class:`TwoLineRule` is hub finding §3.5. Proper motion and parallax together are **one** line (A):
they come from the same astrometric solution. A second line (photometry, RV, youth) must be supplied as
a per-star log-likelihood ratio until its axis is implemented here; with Gaia astrometry alone no star
can be "established". The defaults 0.9 / 0.1 / ln 20 are the conventions written in §3.5 before
looking. For NGC 6383 the calibration of §11 adopted ``p_hi = 0.99`` (``p_lo = 0.01``) because the
specific risk at 0.9 was 0.127 at G >= 18.5 -- pass those explicitly there.

STATUS AND WHAT IS PENDING (hub finding §6, order of the plan)
--------------------------------------------------------------
Done in this module, with tests in ``tests/test_membership_mixture.py``:

* the mixture with PM (per-star covariance) + parallax, a parametric field, and the truncation in every
  component; V1 (construction control: without the field it reproduces the single-component truncated
  MLE, computed independently by a radial Rice integral);
* :func:`calibrate_by_injection` and :func:`propagate` in their minimal form;
* the mutations ``nopa``, ``nofield``, ``nosel`` as tests that must break recovery.

Pending, not run here:

* **V0** -- the mixture on the three empty control fields of B12 (N_c compatible with 0);
* **V2** -- injection into the real catalogue of the cone, >= 300 replicates per cell, CORP with
  MCB <= 0.01 global and <= 0.02 per subgroup, 68/95 % coverage of the cluster parameters;
* **V3** -- CORP against independent labels (own RVs, X-rays, IGAPS) per subgroup;
* **V4** -- global and specific risks of the two-line rule per subgroup (the machinery is
  :func:`calibrate_by_injection` with ``rule=``; the run is pending);
* the empirical local field (``EmpiricalField.from_sideband``, conditioned on G), the CMD axis
  conditioned on G, RV and youth axes, overlapping groups (Theia 1645, Antalova 2) as explicit
  components, the HR24 flag as a modelled feature with its own rate, the per-G-bin mixing fraction
  (hub ``agent-findings/membership-rule-faint-fix-2026-11-04.md``), the full systematic covariance;
* the baseline on the same split (P01's p̃ and HR24's probability on the same replicates);
* the Hessian intervals are not recalibrated: C5 §15.2 measured them short by a factor 1.075-1.145;
* the QMC error of the cluster's P_A at a sharp edge: V1 measured 3.9e-3 in the PM block with 4096 normals
  and 4e-4 with 16384 for an edge of width 0.03 mas/yr; the default 512 is C5's and is not re-measured on
  a real frontier.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np
from scipy import stats
from scipy.stats import qmc

from .analysis.modesearch import mode_search, verdict
from .calibration import corp

# float64: the Hessian and the per-star log-densities need it. PyTensor's JAX linker sets the same flag
# whenever floatX is float64 (pytensor/link/jax/dispatch/basic.py), which is EROTICA's sampling default.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.scipy.special import log_ndtr as _jlog_ndtr  # noqa: E402
from jax.scipy.special import logsumexp as _jlse  # noqa: E402

__all__ = [
    "MembershipData",
    "CircularRegion",
    "Truncation",
    "GaussianField",
    "MixtureModel",
    "MixtureFit",
    "TwoLineRule",
    "PropagationResult",
    "sample_members",
    "sample_field",
    "calibrate_by_injection",
    "propagate",
    "auc",
]

_LOG2PI = float(np.log(2 * np.pi))


# ====================================================================== data
@dataclass
class MembershipData:
    """Per-star astrometry: ``pm`` (N, 2), ``pm_cov`` (N, 2, 2), ``plx`` (N,), ``plx_err`` (N,).

    ``aux`` carries any other per-star columns (e.g. ``G``, radius) through injection and padding, for
    subgroups; it never enters the likelihood.
    """

    pm: np.ndarray
    pm_cov: np.ndarray
    plx: np.ndarray
    plx_err: np.ndarray
    aux: dict[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self):
        self.pm = np.asarray(self.pm, float).reshape(-1, 2)
        n = len(self.pm)
        self.pm_cov = np.asarray(self.pm_cov, float).reshape(n, 2, 2)
        self.plx = np.asarray(self.plx, float).reshape(n)
        self.plx_err = np.asarray(self.plx_err, float).reshape(n)
        self.aux = {k: np.asarray(v).reshape(n, *np.shape(v)[1:]) for k, v in self.aux.items()}
        for nm in ("pm", "pm_cov", "plx", "plx_err"):
            if not np.all(np.isfinite(getattr(self, nm))):
                raise ValueError(f"{nm} has non-finite values")
        if np.any(self.plx_err <= 0):
            raise ValueError("plx_err must be positive")
        det = self.pm_cov[:, 0, 0] * self.pm_cov[:, 1, 1] - self.pm_cov[:, 0, 1] * self.pm_cov[:, 1, 0]
        if np.any(self.pm_cov[:, 0, 0] <= 0) or np.any(det <= 0):
            raise ValueError("pm_cov must be positive definite for every star")

    @property
    def n(self) -> int:
        return len(self.pm)

    def take(self, idx) -> MembershipData:
        idx = np.asarray(idx)
        return MembershipData(self.pm[idx], self.pm_cov[idx], self.plx[idx], self.plx_err[idx],
                              {k: v[idx] for k, v in self.aux.items()})

    @staticmethod
    def concat(a: MembershipData, b: MembershipData) -> MembershipData:
        keys = set(a.aux) & set(b.aux)
        return MembershipData(np.r_[a.pm, b.pm], np.r_[a.pm_cov, b.pm_cov], np.r_[a.plx, b.plx],
                              np.r_[a.plx_err, b.plx_err], {k: np.r_[a.aux[k], b.aux[k]] for k in keys})


# ====================================================================== selection
@dataclass(frozen=True)
class CircularRegion:
    """Proper-motion acceptance region: a circle with a logistic edge.

    ``prob(μ) = 1 / (1 + exp(-(R - |μ - c|) / softness))``. The edge must be smooth (``softness > 0``):
    P_A is a QMC average of this function, and a hard indicator would give P_A a zero gradient almost
    everywhere. The simulators accept a star with this probability, so model and generator share it.
    """

    center: tuple[float, float]
    radius: float
    softness: float

    def __post_init__(self):
        if not (self.radius > 0 and self.softness > 0):
            raise ValueError("radius and softness must be positive")

    def _z(self, pm, xp):
        c = xp.asarray(self.center)
        return (self.radius - xp.sqrt(xp.sum((pm - c) ** 2, axis=-1))) / self.softness

    def prob(self, pm, xp=np):
        return 0.5 * (1.0 + xp.tanh(0.5 * self._z(pm, xp)))

    def logprob(self, pm):
        """log of :meth:`prob` in JAX, without underflow far outside the region."""
        return jax.nn.log_sigmoid(self._z(pm, jnp))

    def support(self, n: int, seed: int = 0) -> tuple[np.ndarray, float]:
        """``n`` scrambled-Sobol points filling the disc where ``prob`` > ~1e-5 (radius R + 12 softness),
        and log of the area each one stands for: the nodes of ∫ prob(μ) g(μ) dμ ≈ (A/n) Σ prob(μ_k) g(μ_k)."""
        rmax = self.radius + 12.0 * self.softness
        m = int(np.ceil(np.log2(n)))
        u = qmc.Sobol(d=2, scramble=True, seed=seed).random_base2(m)[:n]
        r, ph = rmax * np.sqrt(u[:, 0]), 2 * np.pi * u[:, 1]
        pts = np.asarray(self.center, float) + np.c_[r * np.cos(ph), r * np.sin(ph)]
        return pts, float(np.log(np.pi * rmax * rmax / n))


@dataclass(frozen=True)
class Truncation:
    """The selection, put in the expected number of **every** component (hub ``methodology.md`` §A.1.4).

    ``region`` is the proper-motion acceptance (a :class:`CircularRegion`, or any object with
    ``prob(pm, xp=np)`` for the simulators and ``logprob(pm)`` in JAX for the likelihood); ``plx_window = (lo, hi)`` a hard parallax window
    or ``None``.

    P_A is one integral, ∫ prob(μ) N(μ; m, S) dμ, estimated two ways because no single node set serves
    both components. The cluster's Gaussian is narrow next to the region, so it is sampled through the
    Gaussian (``n_qmc`` scrambled-Sobol normals, 512 in C5 §13/§15). The field's is broad, often
    elongated, and puts few of its own points inside the region; sampling it that way let the optimiser
    find field shapes whose P_A the QMC underestimated by ~e^19 per star (first smoke fit, 2026-11-05).
    So the field integral uses ``n_support`` nodes spread over the region itself (``region.support``).
    """

    region: Any = None
    plx_window: tuple[float, float] | None = None
    n_qmc: int = 512
    n_support: int = 1024
    seed: int = 0

    def normals(self) -> np.ndarray:
        m = int(np.ceil(np.log2(self.n_qmc)))
        u = qmc.Sobol(d=2, scramble=True, seed=self.seed).random_base2(m)[: self.n_qmc]
        return stats.norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))

    def accept_prob(self, pm: np.ndarray, plx: np.ndarray) -> np.ndarray:
        a = np.ones(len(pm)) if self.region is None else np.asarray(self.region.prob(pm, np), float)
        if self.plx_window is not None:
            a = a * ((plx >= self.plx_window[0]) & (plx <= self.plx_window[1]))
        return a


@dataclass(frozen=True)
class GaussianField:
    """Parametric field: 2-D Gaussian in proper motion (full covariance) and Gaussian in parallax, both
    convolved with each star's errors and truncated by the same selection as the cluster."""


# ====================================================================== likelihood (JAX)
PARAM_NAMES = ("mu_c_x", "mu_c_y", "log_sig_c", "plx_c", "log_splx_c",
               "mu_f_x", "mu_f_y", "log_sf_1", "log_sf_2", "atanh_rho_f", "plx_f", "log_splx_f", "logit_eta")
_CLUSTER = slice(0, 5)


def _chol2(S00, S01, S11):
    L00 = jnp.sqrt(S00)
    L10 = S01 / L00
    L11 = jnp.sqrt(S11 - L10 * L10)
    return L00, L10, L11


def _logn2(r0, r1, L00, L10, L11):
    y0 = r0 / L00
    y1 = (r1 - L10 * y0) / L11
    return -0.5 * (y0 * y0 + y1 * y1) - jnp.log(L00 * L11) - _LOG2PI


def _log_pa_pm(mean, L00, L10, L11, region, nodes, component):
    """log P_A: probability that a star of ``component`` with this per-star covariance (Cholesky L) is
    accepted by ``region``. 'cluster' samples the Gaussian, 'field' integrates over the region's support
    nodes (see :class:`Truncation`). ``component`` is also the hook the mutation tests patch."""
    if component == "field":
        pts, logw, lpr = nodes["support"], nodes["support_logw"], nodes["support_logprob"]
        ln = _logn2(pts[None, :, 0] - mean[:, 0:1], pts[None, :, 1] - mean[:, 1:2],
                    L00[:, None], L10[:, None], L11[:, None])
        return _jlse(ln + lpr[None, :], axis=1) + logw
    normals = nodes["normals"]
    g0 = normals[None, :, 0]
    g1 = normals[None, :, 1]
    p0 = mean[..., 0:1] + L00[:, None] * g0
    p1 = mean[..., 1:2] + L10[:, None] * g0 + L11[:, None] * g1
    # log-space average: a plain mean of probabilities underflows to 0 for a component far outside the
    # region, and a clipped log(0) then hands that component a +690 nats bonus per star (seen in the
    # first smoke fit, 2026-11-05: a cluster parked outside the circle took the whole likelihood)
    lp = region.logprob(jnp.stack([p0, p1], -1))
    return _jlse(lp, axis=1) - jnp.log(lp.shape[1])


def _log_window(mu, sd, window):
    """log[Φ(b) − Φ(a)], a = (lo − μ)/sd, b = (hi − μ)/sd, without cancellation on either side."""
    lo, hi = window
    a, b = (lo - mu) / sd, (hi - mu) / sd
    up = _jlog_ndtr(-a) + jnp.log1p(-jnp.exp(jnp.minimum(_jlog_ndtr(-b) - _jlog_ndtr(-a), -1e-16)))
    dn = _jlog_ndtr(b) + jnp.log1p(-jnp.exp(jnp.minimum(_jlog_ndtr(a) - _jlog_ndtr(b), -1e-16)))
    return jnp.where(a > 0, up, dn)


def _components(th, d, *, selection, has_field, nodes):
    """Per-star log densities (cluster, field) of the full astrometric line, each normalised by its P_A and W."""
    pm, C, plx, e = d["pm"], d["pm_cov"], d["plx"], d["plx_err"]
    region = None if selection is None else selection.region
    window = None if selection is None else selection.plx_window
    # cluster
    s2 = jnp.exp(2 * th[2])
    L = _chol2(C[:, 0, 0] + s2, C[:, 0, 1], C[:, 1, 1] + s2)
    lc = _logn2(pm[:, 0] - th[0], pm[:, 1] - th[1], *L)
    if region is not None:
        lc = lc - _log_pa_pm(jnp.broadcast_to(th[0:2], pm.shape), *L, region, nodes, "cluster")
    sd = jnp.sqrt(e * e + jnp.exp(2 * th[4]))
    lcp = -0.5 * ((plx - th[3]) / sd) ** 2 - jnp.log(sd) - 0.5 * _LOG2PI
    if window is not None:
        lcp = lcp - _log_window(th[3], sd, window)
    lc = lc + lcp
    if not has_field:
        return lc, None
    a, b, rho = jnp.exp(th[7]), jnp.exp(th[8]), jnp.tanh(th[9])
    L = _chol2(C[:, 0, 0] + a * a, C[:, 0, 1] + rho * a * b, C[:, 1, 1] + b * b)
    lf = _logn2(pm[:, 0] - th[5], pm[:, 1] - th[6], *L)
    if region is not None:
        lf = lf - _log_pa_pm(jnp.broadcast_to(th[5:7], pm.shape), *L, region, nodes, "field")
    sd = jnp.sqrt(e * e + jnp.exp(2 * th[11]))
    lfp = -0.5 * ((plx - th[10]) / sd) ** 2 - jnp.log(sd) - 0.5 * _LOG2PI
    if window is not None:
        lfp = lfp - _log_window(th[10], sd, window)
    return lc, lf + lfp


def _per_star_loglike(th, d, **kw):
    lc, lf = _components(th, d, **kw)
    if lf is None:
        return lc
    eta = jax.nn.sigmoid(th[12])
    return jnp.logaddexp(jnp.log1p(-eta) + lc, jnp.log(eta) + lf)


def _logpost(th, d, prior_field, **kw):
    ll = jnp.sum(d["w"] * _per_star_loglike(th, d, **kw))
    if prior_field is not None and kw["has_field"]:
        med, sdlog = prior_field
        nf = jax.nn.sigmoid(th[12]) * jnp.sum(d["w"])
        ll = ll - 0.5 * ((jnp.log(nf) - np.log(med)) / sdlog) ** 2
    return ll


# ====================================================================== the rule
@dataclass(frozen=True)
class TwoLineRule:
    """Hub finding §3.5, for one star. ``ℓ_k = log L_c,k − log L_f,k`` per independent line k.

    * **member**: ``p >= p_hi``, at least ``min_lines`` lines with ``ℓ_k > 0``, and none with ``ℓ_k < −veto``;
    * **field**: ``p <= p_lo``, at least ``min_lines`` lines with ``ℓ_k < 0``, and none with ``ℓ_k > veto``;
    * **undecided**: everything else. It enters downstream only through ``p``.

    A line with NaN for a star (no data) is not counted for that star. The defaults are the §3.5
    conventions; NGC 6383's calibration (§11) adopted ``p_hi = 0.99``, ``p_lo = 0.01``.
    """

    p_hi: float = 0.9
    p_lo: float = 0.1
    veto: float = float(np.log(20.0))
    min_lines: int = 2

    MEMBER = "member"
    FIELD = "field"
    UNDECIDED = "undecided"

    def apply(self, p: np.ndarray, llr: Mapping[str, np.ndarray]) -> np.ndarray:
        p = np.asarray(p, float)
        L = np.vstack([np.asarray(v, float).reshape(p.shape) for v in llr.values()]) if llr else np.empty((0, p.size))
        ok = np.isfinite(L)
        pos = np.sum(ok & (L > 0), 0)
        neg = np.sum(ok & (L < 0), 0)
        strong_neg = np.any(ok & (L < -self.veto), 0)
        strong_pos = np.any(ok & (L > self.veto), 0)
        out = np.full(p.shape, self.UNDECIDED, dtype=object)
        out[(p >= self.p_hi) & (pos >= self.min_lines) & ~strong_neg] = self.MEMBER
        out[(p <= self.p_lo) & (neg >= self.min_lines) & ~strong_pos] = self.FIELD
        return out


# ====================================================================== the model
@dataclass
class MixtureFit:
    """Result of :meth:`MixtureModel.fit`: the best mode, its Laplace covariance, and the mode search."""

    theta: np.ndarray            # full parameter vector (unconstrained), length len(PARAM_NAMES)
    free: np.ndarray             # indices of the free parameters
    logpost: float
    n_starts: int                # every optimisation start, the detection start included
    init_start: bool             # whether one of them came from ``init`` (mode_search labels it 'prior')
    modes: list[dict]
    verdict: str
    verdict_ok: bool
    at_bound: list[str]
    _hessian: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _cov: Any = field(default=False, repr=False)

    def params(self) -> dict[str, float]:
        return _natural(self.theta)

    @property
    def cov(self) -> np.ndarray | None:
        """Laplace covariance over the free parameters (inverse negative Hessian of the log posterior), or
        ``None`` if the Hessian is not negative definite. Computed on first access: compiling the exact
        Hessian costs more than a fit, and most fits (every injection replicate) never need it."""
        if self._cov is False:
            cov = None
            if self._hessian is not None:
                try:
                    cov = np.linalg.inv(-self._hessian())
                    if not np.all(np.isfinite(cov)) or np.any(np.diag(cov) <= 0):
                        cov = None
                except np.linalg.LinAlgError:
                    cov = None
            self._cov = cov
        return self._cov


def _natural(th) -> dict[str, float]:
    th = np.asarray(th, float)
    return dict(mu_c=(th[0], th[1]), sigma_c=float(np.exp(th[2])), plx_c=float(th[3]), s_plx_c=float(np.exp(th[4])),
                mu_f=(th[5], th[6]), sf_1=float(np.exp(th[7])), sf_2=float(np.exp(th[8])), rho_f=float(np.tanh(th[9])),
                plx_f=float(th[10]), s_plx_f=float(np.exp(th[11])), eta=float(1 / (1 + np.exp(-th[12]))))


class MixtureModel:
    """Cluster + field mixture over Gaia astrometry, with the selection in every component.

    Parameters
    ----------
    data : MembershipData
    selection : Truncation or None
        ``None`` means the sample was not cut in proper motion or parallax.
    field : GaussianField or None
        ``None`` drops the field component (the construction control V1, and the ``nofield`` mutation).
    """

    def __init__(self, data: MembershipData, selection: Truncation | None = None,
                 field: GaussianField | None = GaussianField(), *, _shared: dict | None = None):
        self.data = data
        self.selection = selection
        self.field = field
        self.has_field = field is not None
        self.free = np.arange(len(PARAM_NAMES)) if self.has_field else np.arange(_CLUSTER.stop)
        self.fit_result: MixtureFit | None = None
        if _shared is not None:
            self._fns = _shared
        else:
            nodes = None
            if selection is not None and selection.region is not None:
                sp, slw = selection.region.support(selection.n_support, selection.seed)
                nodes = dict(normals=jnp.asarray(selection.normals()), support=jnp.asarray(sp), support_logw=slw,
                             support_logprob=selection.region.logprob(jnp.asarray(sp)))
            kw = dict(selection=selection, has_field=self.has_field, nodes=nodes)

            def lp(th, d, prior_field):
                return _logpost(th, d, prior_field, **kw)

            self._fns = dict(
                lp=jax.jit(lp, static_argnums=2),
                vg=jax.jit(jax.value_and_grad(lp), static_argnums=2),
                hess=jax.jit(jax.hessian(lp), static_argnums=2),
                comp=jax.jit(lambda th, d: _components(th, d, **kw)),
            )
        self._d = self._device_data(data)

    # -------------------------------------------------------------- plumbing
    @staticmethod
    def _device_data(data: MembershipData, w=None) -> dict:
        return dict(pm=jnp.asarray(data.pm), pm_cov=jnp.asarray(data.pm_cov), plx=jnp.asarray(data.plx),
                    plx_err=jnp.asarray(data.plx_err),
                    w=jnp.asarray(np.ones(data.n) if w is None else w, dtype=float))

    def with_data(self, data: MembershipData, weights: np.ndarray | None = None) -> MixtureModel:
        """Same model and compiled functions, new data (no recompilation when the shape is unchanged)."""
        m = MixtureModel(data, self.selection, self.field, _shared=self._fns)
        if weights is not None:
            m._d = self._device_data(data, weights)
        return m

    def default_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Box over the free parameters. Conventions (C), not measurements -- override with ``bounds=``.

        Cluster mean inside the region (within 0.85 R of its centre: the region was drawn around the
        detected cluster), cluster dispersion up to R, field mean within 2 R and field scales from R to
        50 R: the field is the broad component. With 3 starts and the field allowed down to R/2, one smoke
        fit (2026-11-05) landed in the swapped mode (σ_c → 0.002, field shaped like the cluster). Without a region, R is 3 robust PM spreads.
        """
        pm, plx = self.data.pm, self.data.plx
        sel = self.selection
        if sel is not None and sel.region is not None and hasattr(sel.region, "radius"):
            c, R = np.asarray(sel.region.center, float), float(sel.region.radius)
        else:
            c = np.median(pm, 0)
            R = float(np.max(1.4826 * np.median(np.abs(pm - c), 0))) * 3 + 1e-3
        if sel is not None and sel.plx_window is not None:
            plo, phi = sel.plx_window
        else:
            plo, phi = np.quantile(plx, [0.005, 0.995])
        pr = max(phi - plo, 1e-3)
        lo = np.array([c[0] - 0.85 * R, c[1] - 0.85 * R, np.log(1e-3), plo, np.log(1e-4),
                       c[0] - 2 * R, c[1] - 2 * R, np.log(R), np.log(R), -2.0, plo - pr, np.log(0.05 * pr), -9.0])
        hi = np.array([c[0] + 0.85 * R, c[1] + 0.85 * R, np.log(R), phi, np.log(0.5 * pr),
                       c[0] + 2 * R, c[1] + 2 * R, np.log(50 * R), np.log(50 * R), 2.0, phi + pr, np.log(10 * pr), 9.0])
        return lo[self.free], hi[self.free]

    def _start_from(self, init: Mapping[str, Any], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        th = self._full(0.5 * (lo + hi))
        tr = dict(mu_c=lambda v: (0, np.asarray(v, float)), sigma_c=lambda v: (2, np.log(v)), plx_c=lambda v: (3, v),
                  s_plx_c=lambda v: (4, np.log(v)), eta=lambda v: (12, np.log(v) - np.log1p(-v)))
        for k, v in init.items():
            if k not in tr:
                raise ValueError(f"init accepts {sorted(tr)}; got {k!r}")
            i, val = tr[k](v)
            if k == "mu_c":
                th[0:2] = val
            else:
                th[i] = val
        return np.clip(th[self.free], lo, hi)

    def _full(self, x, base=None):
        th = np.zeros(len(PARAM_NAMES)) if base is None else np.array(base, float)
        th[self.free] = x
        return th

    def logpost(self, theta, prior_field=None) -> float:
        """Log posterior at a full parameter vector (for checks and tests)."""
        return float(self._fns["lp"](jnp.asarray(theta, float), self._d, prior_field))

    # -------------------------------------------------------------- fit
    def fit(self, n_starts: int = 9, *, seed: int = 0, prior_field: tuple[float, float] | None = None,
            init: Mapping[str, Any] | None = None, bounds: tuple[np.ndarray, np.ndarray] | None = None,
            maxiter: int = 2000, ridge: bool = True) -> MixtureFit:
        """Maximise the log posterior with an explicit mode search (hub ``methodology.md`` §A.1.8).

        ``n_starts`` starts are drawn uniformly in the box. ``init`` adds one more start built from the
        detection stage (natural units: ``mu_c``, ``sigma_c``, ``plx_c``, ``s_plx_c``, ``eta``; the rest at
        the box centre) -- the mixture only shares probability inside a cluster detection already found
        (§A.3 step 1), so the detected cluster is a legitimate start. It is one start among the prior ones,
        never a replacement: the verdict still reports every mode, and its "n prior starts" counts the
        detection start among them (``MixtureFit.init_start`` says so). Measured 2026-11-05 on the test fixture
        (400 stars, 13 parameters): 6 uniform starts alone missed the global mode in 1 of 4 seeds.

        ``prior_field = (median, sdlog)``: log-normal prior on the number of field stars ``η Σw``
        (C5's ``prior_Nf`` from the sideband E1).
        """
        lo, hi = self.default_bounds() if bounds is None else (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        rng = np.random.default_rng(seed)
        starts = rng.uniform(lo, hi, (n_starts, lo.size))
        if init is not None:
            x0 = self._start_from(init, lo, hi)
            starts = np.vstack([x0[None], starts])
        names = [PARAM_NAMES[k] for k in self.free]
        cache = {}

        def vg(x):
            key = np.asarray(x, float).tobytes()
            if key not in cache:
                v, g = self._fns["vg"](jnp.asarray(self._full(x)), self._d, prior_field)
                cache.clear()
                cache[key] = (float(v), np.asarray(g)[self.free])
            return cache[key]

        def logp(x):
            v = vg(x)[0]
            return v if np.isfinite(v) else -np.inf

        res = mode_search(logp, starts, grad=lambda x: vg(x)[1], bounds=(lo, hi), scale=(hi - lo) / 10,
                          names=names, maxiter=maxiter, ridge_fracs=(0.125, 0.25, 0.5, 0.9) if ridge else ())
        text, ok = verdict(res)
        best = res["modes"][0]
        th = self._full(best["x"])
        d, free = self._d, self.free.copy()

        def hessian():
            return np.asarray(self._fns["hess"](jnp.asarray(th), d, prior_field))[np.ix_(free, free)]

        self.fit_result = MixtureFit(theta=th, free=free, logpost=float(best["logp"]), n_starts=len(starts),
                                     init_start=init is not None,
                                     modes=res["modes"], verdict=text, verdict_ok=bool(ok),
                                     at_bound=[names[i] for i in best["at_bound"]], _hessian=hessian)
        return self.fit_result

    # -------------------------------------------------------------- outputs
    def _need_fit(self) -> MixtureFit:
        if self.fit_result is None:
            raise RuntimeError("call fit() first")
        return self.fit_result

    def _comp(self, theta=None):
        th = self._need_fit().theta if theta is None else np.asarray(theta, float)
        lc, lf = self._fns["comp"](jnp.asarray(th), self._d)
        return th, np.asarray(lc), None if lf is None else np.asarray(lf)

    @property
    def params(self) -> dict[str, float]:
        return self._need_fit().params()

    @property
    def p(self) -> np.ndarray:
        """Posterior membership probability of every star under the fitted model."""
        th, lc, lf = self._comp()
        if lf is None:
            return np.ones_like(lc)
        logit = (np.log1p(-_sig(th[12])) + lc) - (np.log(_sig(th[12])) + lf)
        return _sig(logit)

    @property
    def axis_llr(self) -> dict[str, np.ndarray]:
        """``ℓ_k`` per independent line. Proper motion and parallax are one line, ``"A"`` (hub §3.5)."""
        _, lc, lf = self._comp()
        if lf is None:
            raise RuntimeError("no field component: the log-likelihood ratio is undefined")
        return {"A": lc - lf}

    def classify(self, rule: TwoLineRule = TwoLineRule(), extra_lines: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Apply the two-line rule. ``extra_lines``: per-star ``ℓ_k`` of lines not modelled here
        (photometry ``"B"``, RV ``"C"``, youth ``"D"``), NaN where a star has no data."""
        llr = dict(self.axis_llr)
        for k, v in (extra_lines or {}).items():
            if k in llr:
                raise ValueError(f"line {k!r} is already modelled")
            llr[k] = np.asarray(v, float)
        return rule.apply(self.p, llr)


def _sig(x):
    x = np.asarray(x, float)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))), np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


# ====================================================================== generators
def _sample_conditional(n, draw, accept, rng, max_iter=100_000):
    """Per-star rejection: star i is redrawn with ITS errors until accepted (C5 §13 'estrella a estrella')."""
    out = None
    todo = np.ones(n, bool)
    it = 0
    while todo.any():
        k = np.where(todo)[0]
        pm, plx = draw(k)
        acc = rng.random(k.size) < accept(pm, plx)
        if out is None:
            out = [np.empty((n, 2)), np.empty(n)]
        out[0][k[acc]] = pm[acc]
        out[1][k[acc]] = plx[acc]
        todo[k[acc]] = False
        it += 1
        if it > max_iter:
            raise RuntimeError("rejection sampling did not converge: the selection rejects this component")
    return out


def sample_members(params: Mapping[str, Any], donors: MembershipData, rng: np.random.Generator,
                   selection: Truncation | None = None) -> MembershipData:
    """Synthetic members with the errors (and ``aux``) of ``donors``, row by row, conditioned on selection.

    ``params``: ``mu_c``, ``sigma_c``, ``plx_c``, ``s_plx_c`` (natural units, as :meth:`MixtureFit.params`).
    """
    Lc = np.linalg.cholesky(donors.pm_cov)
    mu = np.asarray(params["mu_c"], float)

    def draw(k):
        e = np.einsum("kab,kb->ka", Lc[k], rng.standard_normal((k.size, 2)))
        pm = mu + params["sigma_c"] * rng.standard_normal((k.size, 2)) + e
        plx = params["plx_c"] + np.hypot(donors.plx_err[k], params["s_plx_c"]) * rng.standard_normal(k.size)
        return pm, plx

    acc = (lambda pm, plx: selection.accept_prob(pm, plx)) if selection is not None else (lambda pm, plx: np.ones(len(pm)))
    pm, plx = _sample_conditional(donors.n, draw, acc, rng)
    return MembershipData(pm, donors.pm_cov, plx, donors.plx_err, dict(donors.aux))


def sample_field(params: Mapping[str, Any], donors: MembershipData, rng: np.random.Generator,
                 selection: Truncation | None = None) -> MembershipData:
    """Synthetic field stars from the parametric :class:`GaussianField` (``mu_f``, ``sf_1``, ``sf_2``,
    ``rho_f``, ``plx_f``, ``s_plx_f``), with the donors' errors, conditioned on selection."""
    a, b, r = params["sf_1"], params["sf_2"], params["rho_f"]
    Lf = np.linalg.cholesky(np.array([[a * a, r * a * b], [r * a * b, b * b]]))
    Lc = np.linalg.cholesky(donors.pm_cov)
    mu = np.asarray(params["mu_f"], float)

    def draw(k):
        e = np.einsum("kab,kb->ka", Lc[k], rng.standard_normal((k.size, 2)))
        pm = mu + rng.standard_normal((k.size, 2)) @ Lf.T + e
        plx = params["plx_f"] + np.hypot(donors.plx_err[k], params["s_plx_f"]) * rng.standard_normal(k.size)
        return pm, plx

    acc = (lambda pm, plx: selection.accept_prob(pm, plx)) if selection is not None else (lambda pm, plx: np.ones(len(pm)))
    pm, plx = _sample_conditional(donors.n, draw, acc, rng)
    return MembershipData(pm, donors.pm_cov, plx, donors.plx_err, dict(donors.aux))


def auc(p, y) -> float:
    """Area under the ROC curve by ranks (Mann-Whitney); NaN if one class is empty."""
    y = np.asarray(y, bool)
    p = np.asarray(p, float)
    if y.all() or (~y).all():
        return float("nan")
    r = stats.rankdata(p)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


# ====================================================================== calibration by injection
def calibrate_by_injection(model: MixtureModel, field_catalog: MembershipData, n_rep: int, *,
                           subgroups: Mapping[str, Callable[[MembershipData], np.ndarray]] | None = None,
                           rule: TwoLineRule | None = None,
                           extra_lines: Callable[[MembershipData, np.ndarray, np.random.Generator], Mapping[str, np.ndarray]] | None = None,
                           n_starts: int = 3, init: Mapping[str, Any] | None = None, ridge: bool = True, seed: int = 0,
                           prior_field: tuple[float, float] | None = None) -> dict:
    """N2 of hub ``methodology.md`` §A.3: synthetic members injected into a real field catalogue, refit, score.

    ``field_catalog`` must hold **no cluster members** (an empty control field or a sideband) and already
    satisfy the selection; every row is labelled ``y = 0``. Each replicate draws ``Poisson((1 − η̂) N)``
    members from the fitted cluster of ``model`` with the errors and ``aux`` of random catalogue rows,
    refits with ``n_starts`` and scores ``p`` against the truth: CORP (pooled and per subgroup), AUC,
    ``Σp − N_true`` and, with ``rule``, its global and specific risks (§A.2 row e): the false-member rate
    among stars it calls "member" and the completeness among true members. ``extra_lines(data, y, rng)``
    supplies the other lines' ``ℓ_k`` for the rule. Replicates are padded to one shape (zero weight), so
    the likelihood compiles once.

    Each replicate is fitted **the way the real fit was**: the same ``n_starts``, ``prior_field`` and the
    same detection-stage ``init`` and ``ridge`` -- never the fitted values, which would seed every replicate at its own
    truth and hide the modes the real fit can fall into (hub ``state/findings.yaml`` R26-13).
    """
    fit = model._need_fit()
    par = fit.params()
    rng = np.random.default_rng(seed)
    n_fit = model.data.n
    n_mem = rng.poisson((1 - par["eta"]) * n_fit, n_rep) if model.has_field else np.full(n_rep, n_fit)
    n_pad = field_catalog.n + int(n_mem.max())
    P, Y, W, groups, sums, cls = [], [], [], {k: [] for k in (subgroups or {})}, [], []
    for b in range(n_rep):
        donors = field_catalog.take(rng.integers(field_catalog.n, size=n_mem[b]))
        mem = sample_members(par, donors, rng, model.selection)
        data = MembershipData.concat(field_catalog, mem)
        y = np.r_[np.zeros(field_catalog.n, bool), np.ones(mem.n, bool)]
        pad = n_pad - data.n
        w = np.r_[np.ones(data.n), np.zeros(pad)]
        padded = MembershipData.concat(data, data.take(np.zeros(pad, int))) if pad else data
        m = model.with_data(padded, weights=w)
        m.fit(n_starts=n_starts, seed=seed + 1 + b, prior_field=prior_field, init=init, ridge=ridge)
        p = m.p[: data.n]
        P.append(p)
        Y.append(y)
        sums.append(float(p.sum() - y.sum()))
        for k, f in (subgroups or {}).items():
            groups[k].append(np.asarray(f(data), bool))
        if rule is not None:
            llr = {k: v[: data.n] for k, v in m.axis_llr.items()}
            if extra_lines is not None:
                llr.update(extra_lines(data, y, rng))
            cls.append(rule.apply(p, llr))
        W.append(b)
    p_all, y_all = np.concatenate(P), np.concatenate(Y)
    out = dict(n_rep=n_rep, n_members_injected=n_mem.tolist(), corp=corp(p_all, y_all), auc=auc(p_all, y_all),
               sum_p_minus_true=dict(mean=float(np.mean(sums)), sd=float(np.std(sums, ddof=1)) if n_rep > 1 else float("nan"),
                                     per_rep=sums))
    if subgroups:
        out["corp_by_subgroup"] = {}
        for k, masks in groups.items():
            msk = np.concatenate(masks)
            out["corp_by_subgroup"][k] = corp(p_all[msk], y_all[msk]) if msk.sum() and np.unique(y_all[msk]).size == 2 else None
    if rule is not None:
        c_all = np.concatenate(cls)

        def risks(msk):
            called = msk & (c_all == TwoLineRule.MEMBER)
            true = msk & y_all
            return dict(n_called_member=int(called.sum()),
                        false_member_rate=float(np.mean(~y_all[called])) if called.any() else float("nan"),
                        completeness=float(np.mean(c_all[true] == TwoLineRule.MEMBER)) if true.any() else float("nan"))
        out["rule_risk"] = risks(np.ones_like(y_all))
        if subgroups:
            out["rule_risk_by_subgroup"] = {k: risks(np.concatenate(m_)) for k, m_ in groups.items()}
    out["arrays"] = dict(p=p_all, y=y_all)
    return out


# ====================================================================== propagation
@dataclass(frozen=True)
class PropagationResult:
    values: np.ndarray
    mode: str
    n_draws: int

    @property
    def mean(self) -> float:
        return float(np.mean(self.values))

    @property
    def sd(self) -> float:
        return float(np.std(self.values, ddof=1))

    def quantiles(self, q=(0.16, 0.5, 0.84)) -> np.ndarray:
        return np.quantile(self.values, q)


def propagate(statistic: Callable, model: MixtureModel, mode: str = "bernoulli", n_draws: int = 1000,
              *, seed: int = 0) -> PropagationResult:
    """Distribution of a downstream statistic with the membership uncertainty -- never a cut (§A.3 step 5).

    * ``mode="bernoulli"``: ``statistic(data, members)`` with ``members ~ Bernoulli(p_i)`` independently,
      ``n_draws`` times. Membership uncertainty only; the fit parameters are held at the best mode.
    * ``mode="in_model"``: ``statistic(params)`` on draws of the parameters from the Laplace approximation
      of the fit (free parameters, unconstrained space), with ``params`` as :meth:`MixtureFit.params`.
      Use it when the quantity is a model parameter or a function of them. The Laplace intervals are not
      recalibrated (C5 §15.2 measured them short by 1.075-1.145).
    """
    rng = np.random.default_rng(seed)
    fit = model._need_fit()
    if mode == "bernoulli":
        p = model.p
        vals = np.array([float(statistic(model.data, rng.random(p.size) < p)) for _ in range(n_draws)])
    elif mode == "in_model":
        if fit.cov is None:
            raise RuntimeError("the fit has no Laplace covariance (Hessian not negative definite)")
        xs = rng.multivariate_normal(fit.theta[fit.free], fit.cov, n_draws)
        vals = np.array([float(statistic(_natural(model._full(x, fit.theta)))) for x in xs])
    else:
        raise ValueError("mode must be 'bernoulli' or 'in_model'")
    return PropagationResult(values=vals, mode=mode, n_draws=n_draws)
