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

Three optional pieces (hub ``agent-findings/membership-cmd-axis-2026-11-05.md``), each off by default:

* ``lines=("B",)``: further lines multiply the cluster term by ``exp(ℓ_k)``, with ``ℓ_k`` per star in
  ``MembershipData.lines``. The CMD axis is :func:`photometric_llr`: colour conditioned on G under the cluster
  (:class:`MemberSequence`, from a reference list) against the field (:class:`SidebandTemplate`). This is the form
  of the hub's assignment model and C5 §15 -- an empirical density ratio, not an isochrone, so the age fit stays out
  of the membership; keep ``lines=()`` when ``p`` will weight a CMD analysis;
* ``field=EmpiricalField()``: flat PM over the acceptance region and the sideband's ``p_f(ϖ | G)``;
* ``n_groups``: one field fraction per ``MembershipData.group`` (G bins), each with its own prior on its count.

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
specific risk at 0.9 was 0.127 at G >= 18.5 with the astrometry-only model. With :func:`cone_mixture` the lowest
``p_hi`` whose specific risk stays <= 5 % in all six subgroups is ``CONE_P_HI = 0.8`` (V4 on the new replicates).

STATUS AND WHAT IS PENDING (hub finding §6, order of the plan)
--------------------------------------------------------------
Done in this module, with tests in ``tests/test_membership_mixture.py``:

* the mixture with PM (per-star covariance) + parallax, a parametric field, and the truncation in every
  component; V1 (construction control: without the field it reproduces the single-component truncated
  MLE, computed independently by a radial Rice integral);
* :func:`calibrate_by_injection` and :func:`propagate` in their minimal form;
* the mutations ``nopa``, ``nofield``, ``nosel`` as tests that must break recovery.

VALIDATED ON NGC 6383, 2026-11-05 -- AND NOT YET FIT FOR USE ALONE (hub
``agent-findings/membership-rule-validation-2026-11-05.md``; the 300 injection replicates of the hub's earlier
validation, on the real catalogues of three empty control fields, so every arm is scored on the same stars):

* C0 passes (field responsibility 36.2 in [25; 45] on the 254) and **V0 passes** in the three empty fields,
  one at the edge (Σp = 4.93 against 5 in C357.50+0.50; σ_c collapses onto one star in two fields).
* **V2 fails.** CORP MCB 0.0145 > 0.01 overall and > 0.02 in four of six subgroups (0.29 at G >= 18.5); AUC
  0.727. The fraction η is recovered (median 0.139 against 0.136 true): what is missing is discrimination. With
  this field model the astrometric line alone has AUC 0.728, against 0.794 for the hub's line A (empirical field
  conditioned on G) and 0.928 for the photometric line B. Adding ℓ_B to the logit (not done here) gives MCB
  0.0029 overall and still fails G >= 18.5 (0.082), like the hub's model.
* Laplace coverage of the cluster parameters: 0.54-0.56 at 68 % and 0.85-0.90 at 95 % -- short.
* **So p from the astrometric model alone is not a calibrated membership probability on a real young-cluster cone**;
  it needs the CMD axis and the per-G mixing fraction (below). V3 (the
  X-ray/YSO line) does not contradict it, but its log score is no better than a constant p.

WITH THE THREE PIECES, 2026-11-05 (hub ``agent-findings/membership-cmd-axis-2026-11-05.md``, preregistered in hub
8b236c3 before running; the same 300 replicates, V0 and V3):

* ``lines=("B",)`` alone: MCB 0.0023 overall, AUC 0.943, still 0.079 at G >= 18.5. The empirical field adds AUC
  (0.964) but not the faint bins. ``n_groups=3`` (G < 16 / 16-18.5 / >= 18.5) is what closes them.
* **CMD line + per-G fraction (Gaussian field) and CMD line + empirical field + per-G fraction pass V2 in all six
  subgroups** (MCB 0.0016 / 0.0010 overall, 0.0163 / 0.0155 at G >= 18.5; AUC 0.944 / 0.978), V0 in the three empty
  fields and C0, and the X-ray/YSO line's log score beats a constant p by ~5 nats. No arm without the CMD line passes.
* Calibrated by the gate, not bin by bin: faint mid-range p is under-confident, because the per-bin prior on the faint
  field count (the hub's A1 prior) sits ~32 % above the true count in these replicates.
* Against pyUPMASK (Pera+2021, the bench configuration) on the same 300 replicates, Brier difference with a
  replicate bootstrap: both arms beat it overall and in all six subgroups (all three pieces: -0.320 [-0.328; -0.311]
  overall, -0.171 [-0.184; -0.158] at G >= 18.5). pyUPMASK gives p ~ 0 beyond 20' here (AUC 0.515) and p = 1 to
  every star of the empty fields at n = 19-33 (hub finding §4.6).
* **Confirmed on 300 new replicates** (preregistered in hub 62ec43c before running; generator seeds 4101-4103, fit
  seeds 1300-1599, V0 on ten acceptances per field): all three pieces pass V2 in all six subgroups (MCB 0.0012
  overall, 0.0174 at G >= 18.5; AUC 0.977) and V0, and beat CMD line + per-G fraction in Brier overall
  (-0.0047 [-0.0054; -0.0041]) with no subgroup against. So the validated cone configuration is
  :func:`cone_mixture` (empirical field + CMD line + three G bins) with ``p_hi = CONE_P_HI = 0.8``. One line of
  evidence (the simulation, new draws of the same generator and the same three field catalogues); the X-ray/YSO
  line on the real cone does not decide between the two (Delta log score -0.42 [-1.98; +1.52]).

Pending:

* RV and youth axes, overlapping groups (Theia 1645, Antalova 2) as explicit components, the HR24 flag as a modelled
  feature with its own rate, a radial prior on the field fraction, the full systematic covariance;
* the Hessian intervals are not recalibrated: C5 §15.2 measured them short by a factor 1.075-1.145, and the
  validation above measured 68 % intervals covering 0.54-0.56;
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
    "EmpiricalField",
    "SidebandTemplate",
    "MemberSequence",
    "photometric_llr",
    "MixtureModel",
    "MixtureFit",
    "cone_mixture",
    "cone_groups",
    "CONE_G_EDGES",
    "CONE_P_HI",
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
    #: per-star log-likelihood ratio ``ℓ_k = log L_c,k − log L_f,k`` of further lines (photometry ``"B"``, ...),
    #: conditionally independent of the astrometry given the component. NaN = no data for that star (the line
    #: then carries no information: 0 in ``p``, not counted by the rule). They enter ``p`` only for the lines a
    #: :class:`MixtureModel` is built with (``lines=``); see :func:`photometric_llr` for line B.
    lines: dict[str, np.ndarray] = field(default_factory=dict)
    #: per-star log density of the parallax under the field, already normalised on the parallax window and
    #: conditioned on whatever the caller conditions it on (G, with :class:`SidebandTemplate`). Used by
    #: :class:`EmpiricalField`; ignored by :class:`GaussianField`.
    field_plx_logpdf: np.ndarray | None = None
    #: per-star integer group (e.g. a G bin) for a mixing fraction per group (``MixtureModel(n_groups=...)``).
    group: np.ndarray | None = None

    def __post_init__(self):
        self.pm = np.asarray(self.pm, float).reshape(-1, 2)
        n = len(self.pm)
        self.pm_cov = np.asarray(self.pm_cov, float).reshape(n, 2, 2)
        self.plx = np.asarray(self.plx, float).reshape(n)
        self.plx_err = np.asarray(self.plx_err, float).reshape(n)
        self.aux = {k: np.asarray(v).reshape(n, *np.shape(v)[1:]) for k, v in self.aux.items()}
        self.lines = {k: np.asarray(v, float).reshape(n) for k, v in self.lines.items()}
        for k, v in self.lines.items():
            if np.any(np.isinf(v)):
                raise ValueError(f"line {k!r} has infinite values (NaN marks a star without data)")
        if self.field_plx_logpdf is not None:
            self.field_plx_logpdf = np.asarray(self.field_plx_logpdf, float).reshape(n)
            if not np.all(np.isfinite(self.field_plx_logpdf)):
                raise ValueError("field_plx_logpdf has non-finite values")
        if self.group is not None:
            self.group = np.asarray(self.group).reshape(n)
            if not np.issubdtype(self.group.dtype, np.integer) or np.any(self.group < 0):
                raise ValueError("group must hold non-negative integers")
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
                              {k: v[idx] for k, v in self.aux.items()}, {k: v[idx] for k, v in self.lines.items()},
                              None if self.field_plx_logpdf is None else self.field_plx_logpdf[idx],
                              None if self.group is None else self.group[idx])

    @staticmethod
    def concat(a: MembershipData, b: MembershipData) -> MembershipData:
        """Rows of ``a`` then ``b``. ``aux`` keeps the keys both have; ``lines``, ``field_plx_logpdf`` and ``group``
        must be present in both or in neither (a modelled quantity silently dropped would change ``p``)."""
        keys = set(a.aux) & set(b.aux)
        if set(a.lines) != set(b.lines):
            raise ValueError(f"lines differ: {sorted(a.lines)} vs {sorted(b.lines)}")
        for nm in ("field_plx_logpdf", "group"):
            if (getattr(a, nm) is None) != (getattr(b, nm) is None):
                raise ValueError(f"{nm} present in only one of the two")

        def cat(nm):
            return None if getattr(a, nm) is None else np.r_[getattr(a, nm), getattr(b, nm)]
        return MembershipData(np.r_[a.pm, b.pm], np.r_[a.pm_cov, b.pm_cov], np.r_[a.plx, b.plx],
                              np.r_[a.plx_err, b.plx_err], {k: np.r_[a.aux[k], b.aux[k]] for k in keys},
                              {k: np.r_[a.lines[k], b.lines[k]] for k in a.lines}, cat("field_plx_logpdf"), cat("group"))


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


@dataclass(frozen=True)
class EmpiricalField:
    """Field taken from the data, not fitted: flat in proper motion over the acceptance region, and the
    parallax density ``MembershipData.field_plx_logpdf`` (normalised on the parallax window, e.g. conditioned on
    G by :class:`SidebandTemplate`).

    The proper-motion density of the selected field is ``prob(μ) / A`` with ``A = ∫ prob(μ) dμ`` (the region's
    support nodes); ``prob(μ_i)`` is common to both components and cancels, so each star gets ``−log A``. Flat
    is the field the hub's assignment model uses (``membership_rule_v0_v2.py``, ``1/A_F``): inside an
    acceptance of ~0.3 mas/yr the field's PM gradient is small next to the cluster's, and a Gaussian fitted to
    the ~40 field stars there runs to the box edges (hub ``agent-findings/membership-rule-validation-2026-11-05.md``
    §3.3, scales at the edge in ~30 % of replicates). Measured before adopting it, with the Gaussian fits frozen
    (hub ``agent-findings/membership-cmd-axis-2026-11-05.md`` §1): AUC of line A at G < 16 goes 0.747 → 0.826 with
    a flat PM field and → 0.870 with the parallax conditioned on G as well. Needs a selection region (a flat
    density on the whole plane is improper); the field then has no free parameter but the fraction."""

    pm: str = "flat"

    def __post_init__(self):
        if self.pm != "flat":
            raise ValueError("EmpiricalField supports pm='flat' only")


def _scott(x, d):
    x = np.asarray(x, float)
    return float(np.std(x) * len(x) ** (-1.0 / (d + 4)))


class SidebandTemplate:
    """Kernel density of the field in (G, colour, parallax) from a sideband: stars of the same cone outside the
    cluster's proper-motion region (hub ``membership_rule_v0_v2.Template``; C5 ``lf_feat``).

    Product Gaussian kernels with Scott bandwidths (``d = 3``; the G bandwidth capped at ``max_mag_bandwidth``),
    the parallax kernel renormalised on ``plx_window`` so every conditional is a density on the selected volume.
    Conditioning on G is what makes the template local in magnitude: the field's parallax and colour
    distributions change with G far more than within a 0.33 mas/yr PM acceptance.
    """

    def __init__(self, mag, colour, plx, plx_window, max_mag_bandwidth: float = 0.3):
        self.G, self.C, self.P = (np.asarray(v, float).ravel() for v in (mag, colour, plx))
        if not (len(self.G) == len(self.C) == len(self.P)) or len(self.G) < 5:
            raise ValueError("the sideband needs at least 5 stars with mag, colour and plx")
        self.window = (float(plx_window[0]), float(plx_window[1]))
        self.hg = min(_scott(self.G, 3), max_mag_bandwidth)
        self.hc = _scott(self.C, 3)
        self.hp = _scott(self.P, 3)
        self.fwin = stats.norm.cdf((self.window[1] - self.P) / self.hp) - stats.norm.cdf((self.window[0] - self.P) / self.hp)
        self.n = len(self.G)

    def _lf(self, mag, colour, plx, use_colour, use_plx):
        from scipy.special import logsumexp
        g = np.atleast_1d(np.asarray(mag, float))
        lg = stats.norm.logpdf((g[:, None] - self.G[None, :]) / self.hg)
        num = lg.copy()
        den = lg + np.log(self.fwin)[None, :] if use_plx else lg.copy()
        if use_colour:
            c = np.atleast_1d(np.asarray(colour, float))
            num = num + stats.norm.logpdf((c[:, None] - self.C[None, :]) / self.hc) - np.log(self.hc)
        if use_plx:
            p = np.atleast_1d(np.asarray(plx, float))
            num = num + stats.norm.logpdf((p[:, None] - self.P[None, :]) / self.hp) - np.log(self.hp)
        return logsumexp(num, 1) - logsumexp(den, 1)

    def log_plx(self, mag, plx):
        """log p_f(ϖ | G), a density on the parallax window: the ``field_plx_logpdf`` of :class:`EmpiricalField`."""
        return self._lf(mag, None, plx, False, True)

    def log_colour(self, mag, colour, plx=None):
        """log p_f(c | G), or log p_f(c | G, ϖ) = log p_f(c, ϖ | G) − log p_f(ϖ | G) when ``plx`` is given."""
        if plx is None:
            return self._lf(mag, colour, None, True, False)
        return self._lf(mag, colour, plx, True, True) - self.log_plx(mag, plx)


class MemberSequence:
    """The cluster's sequence in the CMD from a reference list of members: a ridge (running median of colour in
    G over ``window`` stars, then a running mean over ``smooth`` ridge points) plus a kernel density of the
    colour residuals (Scott, ``d = 1``) -- hub ``membership_rule_v0_v2.Ridge``, C5 ``lm_col``.

    It is empirical on purpose: a young cluster's pre-main sequence is broad (age spread, extinction, binaries,
    accretion), and an isochrone would put the age fit inside the membership (hub
    ``agent-findings/membership-cmd-axis-2026-11-05.md`` §1, decision D1). A star of the reference list is scored
    leave-one-out when its id is passed, so it does not vote for itself.
    """

    def __init__(self, mag, colour, ids=None, window: int = 31, smooth: int = 9):
        from scipy.ndimage import uniform_filter1d
        mag, colour = np.asarray(mag, float).ravel(), np.asarray(colour, float).ravel()
        if len(mag) < max(window, 5):
            raise ValueError("the reference list is shorter than the ridge window")
        o = np.argsort(mag)
        gm, cm = mag[o], colour[o]
        rid = np.array([np.median(cm[max(0, k - window // 2):k + window // 2 + 1]) for k in range(len(gm))])
        self.gm, self.rid = gm, uniform_filter1d(rid, smooth, mode="nearest")
        self.mres = colour - self.ridge(mag)
        self.h = _scott(self.mres, 1)
        self.ids = None if ids is None else np.asarray(ids).ravel()

    def ridge(self, mag):
        return np.interp(mag, self.gm, self.rid)

    def log_colour(self, mag, colour, ids=None):
        """log p_m(c | G); ``ids`` (same kind as the reference ids) switches on leave-one-out for listed stars."""
        from scipy.special import logsumexp
        mag, colour = np.atleast_1d(np.asarray(mag, float)), np.atleast_1d(np.asarray(colour, float))
        lk = stats.norm.logpdf(((colour - self.ridge(mag))[:, None] - self.mres[None, :]) / self.h) - np.log(self.h)
        n = np.full(len(mag), float(len(self.mres)))
        if ids is not None and self.ids is not None:
            pos = {s: k for k, s in enumerate(self.ids.tolist())}
            for i, s in enumerate(np.asarray(ids).ravel().tolist()):
                if s in pos:
                    lk[i, pos[s]] = -np.inf
                    n[i] -= 1
        return logsumexp(lk, 1) - np.log(n)

    def draw(self, mag, rng):
        """Synthetic colours of members of magnitude ``mag`` (ridge + a resampled residual + kernel noise)."""
        mag = np.asarray(mag, float)
        return self.ridge(mag) + rng.choice(self.mres, mag.size) + rng.normal(0, self.h, mag.size)


def photometric_llr(sequence: MemberSequence, template: SidebandTemplate, mag, colour, plx=None, ids=None):
    """Line B, ``ℓ_B = log p_m(c | G) − log p_f(c | G[, ϖ])``: the CMD axis conditioned on G. Pass ``plx`` to
    condition the field colour on parallax too (the hub's form, consistent with :class:`EmpiricalField`, whose
    field parallax is conditioned on G from the same template). Stars with a non-finite colour get NaN (no
    data). Put the result in ``MembershipData.lines["B"]`` and build the model with ``lines=("B",)``."""
    mag, colour = np.asarray(mag, float), np.asarray(colour, float)
    ok = np.isfinite(mag) & np.isfinite(colour)
    out = np.full(mag.shape, np.nan)
    if ok.any():
        lm = sequence.log_colour(mag[ok], colour[ok], None if ids is None else np.asarray(ids)[ok])
        lf = template.log_colour(mag[ok], colour[ok], None if plx is None else np.asarray(plx, float)[ok])
        out[ok] = lm - lf
    return out


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


def _components(th, d, *, selection, has_field, nodes, empirical=False, n_groups=1):
    """Per-star log densities (cluster, field) of the full astrometric line, each normalised by its P_A and W.

    With ``empirical`` the field is :class:`EmpiricalField`: ``−log A`` (flat PM over the region) plus the
    caller's ``field_plx_logpdf`` (already a density on the window, so no W here)."""
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
    if empirical:
        return lc, d["fplx"] - nodes["log_area"]
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


_ETA = 12   # first index of the mixing-fraction block (one logit per group)


def _per_star_loglike(th, d, **kw):
    """log L_i. The modelled lines multiply the cluster term by exp(ℓ_k): with the line's field density as the
    common factor, L_c,k = L_f,k e^{ℓ_k}, and the common factor drops out of p and of the fit."""
    lc, lf = _components(th, d, **kw)
    lc = lc + d["lsum"]
    if lf is None:
        return lc
    eta = jax.nn.sigmoid(th[_ETA + d["group"]])
    return jnp.logaddexp(jnp.log1p(-eta) + lc, jnp.log(eta) + lf)


def _logpost(th, d, prior_field, **kw):
    """``prior_field``: ``(median, sdlog)`` on the number of field stars, or with ``n_groups > 1`` one such pair per
    group, on that group's number ``η_g Σ_{i∈g} w_i``."""
    ll = jnp.sum(d["w"] * _per_star_loglike(th, d, **kw))
    if prior_field is not None and kw["has_field"]:
        ng = kw["n_groups"]
        pf = (prior_field,) if ng == 1 else tuple(prior_field)
        wg = jnp.zeros(ng).at[d["group"]].add(d["w"])
        for g, (med, sdlog) in enumerate(pf):
            nf = jax.nn.sigmoid(th[_ETA + g]) * wg[g]
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
    conventions; NGC 6383's first calibration (§11, astrometry-only model) adopted ``p_hi = 0.99``, ``p_lo = 0.01``;
    with :func:`cone_mixture` use ``p_hi = CONE_P_HI``.
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
    param_names: tuple[str, ...] = PARAM_NAMES
    _hessian: Callable[[], np.ndarray] | None = field(default=None, repr=False)
    _cov: Any = field(default=False, repr=False)

    def params(self) -> dict[str, float]:
        return _natural(self.theta, len(self.param_names) - _ETA)

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


def _natural(th, n_groups: int = 1) -> dict[str, float]:
    """Natural parameters. ``eta`` is the first group's field fraction; ``eta_groups`` has all of them."""
    th = np.asarray(th, float)
    eta = 1 / (1 + np.exp(-th[_ETA:_ETA + n_groups]))
    return dict(mu_c=(th[0], th[1]), sigma_c=float(np.exp(th[2])), plx_c=float(th[3]), s_plx_c=float(np.exp(th[4])),
                mu_f=(th[5], th[6]), sf_1=float(np.exp(th[7])), sf_2=float(np.exp(th[8])), rho_f=float(np.tanh(th[9])),
                plx_f=float(th[10]), s_plx_f=float(np.exp(th[11])), eta=float(eta[0]),
                eta_groups=tuple(float(e) for e in eta))


def _param_names(n_groups: int) -> tuple[str, ...]:
    return PARAM_NAMES if n_groups == 1 else PARAM_NAMES[:_ETA] + tuple(f"logit_eta_{g}" for g in range(n_groups))


class MixtureModel:
    """Cluster + field mixture over Gaia astrometry, with the selection in every component.

    Parameters
    ----------
    data : MembershipData
    selection : Truncation or None
        ``None`` means the sample was not cut in proper motion or parallax.
    field : GaussianField, EmpiricalField or None
        ``None`` drops the field component (the construction control V1, and the ``nofield`` mutation).
        :class:`EmpiricalField` takes the field from the data (flat PM over the region, ``field_plx_logpdf``) and
        leaves the fraction as its only parameter.
    lines : tuple of str
        Names of ``data.lines`` that enter ``p`` (the CMD axis is ``"B"``, from :func:`photometric_llr`). Each is a
        per-star log-likelihood ratio, conditionally independent of the astrometry given the component.
        Default ``()``: astrometry only, which is the right choice when ``p`` will weight a CMD analysis (an
        isochrone fit), so the CMD is not used twice.
    n_groups : int
        Number of mixing fractions, one per value of ``data.group`` (e.g. G bins: the field fraction of a young
        cluster's cone grows steeply to faint G, hub ``agent-findings/membership-rule-faint-fix-2026-11-04.md``).
        With ``n_groups > 1``, ``prior_field`` in :meth:`fit` is one ``(median, sdlog)`` per group.
    """

    def __init__(self, data: MembershipData, selection: Truncation | None = None,
                 field: GaussianField | EmpiricalField | None = GaussianField(), *, lines: tuple[str, ...] = (),
                 n_groups: int = 1, _shared: dict | None = None):
        self.data = data
        self.selection = selection
        self.field = field
        self.lines = tuple(lines)
        self.n_groups = int(n_groups)
        self.has_field = field is not None
        self.empirical = isinstance(field, EmpiricalField)
        self.param_names = _param_names(self.n_groups)
        missing = [k for k in self.lines if k not in data.lines]
        if missing:
            raise ValueError(f"lines {missing} are not in data.lines ({sorted(data.lines)})")
        if self.n_groups < 1:
            raise ValueError("n_groups must be >= 1")
        if self.n_groups > 1:
            if data.group is None:
                raise ValueError("n_groups > 1 needs data.group")
            if np.any(data.group >= self.n_groups):
                raise ValueError("data.group has values >= n_groups")
            if not self.has_field:
                raise ValueError("n_groups > 1 needs a field component")
        if self.empirical:
            if selection is None or selection.region is None:
                raise ValueError("EmpiricalField needs a selection region: a flat PM density on the plane is improper")
            if data.field_plx_logpdf is None:
                raise ValueError("EmpiricalField needs data.field_plx_logpdf")
        eta_idx = np.arange(_ETA, _ETA + self.n_groups)
        if not self.has_field:
            self.free = np.arange(_CLUSTER.stop)
        elif self.empirical:
            self.free = np.r_[np.arange(_CLUSTER.stop), eta_idx]
        else:
            self.free = np.r_[np.arange(_ETA), eta_idx]
        self.fit_result: MixtureFit | None = None
        if _shared is not None:
            self._fns = _shared
        else:
            nodes = None
            if selection is not None and selection.region is not None:
                sp, slw = selection.region.support(selection.n_support, selection.seed)
                lpr = selection.region.logprob(jnp.asarray(sp))
                nodes = dict(normals=jnp.asarray(selection.normals()), support=jnp.asarray(sp), support_logw=slw,
                             support_logprob=lpr, log_area=float(_jlse(lpr)) + slw)
            kw = dict(selection=selection, has_field=self.has_field, nodes=nodes, empirical=self.empirical,
                      n_groups=self.n_groups)

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
    def _device_data(self, data: MembershipData, w=None) -> dict:
        """Arrays for the likelihood. The modelled lines are summed with NaN → 0 here, before any weight: a
        padded row (weight 0) with a NaN line would otherwise make 0 × NaN = NaN and poison every fit."""
        lsum = np.zeros(data.n)
        for k in self.lines:
            lsum = lsum + np.nan_to_num(data.lines[k], nan=0.0)
        fplx = np.zeros(data.n) if data.field_plx_logpdf is None else data.field_plx_logpdf
        group = np.zeros(data.n, int) if (data.group is None or self.n_groups == 1) else data.group
        return dict(pm=jnp.asarray(data.pm), pm_cov=jnp.asarray(data.pm_cov), plx=jnp.asarray(data.plx),
                    plx_err=jnp.asarray(data.plx_err), lsum=jnp.asarray(lsum), fplx=jnp.asarray(fplx),
                    group=jnp.asarray(group, dtype=jnp.int32),
                    w=jnp.asarray(np.ones(data.n) if w is None else w, dtype=float))

    def with_data(self, data: MembershipData, weights: np.ndarray | None = None) -> MixtureModel:
        """Same model and compiled functions, new data (no recompilation when the shape is unchanged)."""
        m = MixtureModel(data, self.selection, self.field, lines=self.lines, n_groups=self.n_groups, _shared=self._fns)
        if weights is not None:
            m._d = m._device_data(data, weights)
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
        lo = np.r_[lo[:_ETA], np.full(self.n_groups, lo[_ETA])]
        hi = np.r_[hi[:_ETA], np.full(self.n_groups, hi[_ETA])]
        return lo[self.free], hi[self.free]

    def _start_from(self, init: Mapping[str, Any], lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
        th = self._full(0.5 * (lo + hi))
        tr = dict(mu_c=lambda v: (0, np.asarray(v, float)), sigma_c=lambda v: (2, np.log(v)), plx_c=lambda v: (3, v),
                  s_plx_c=lambda v: (4, np.log(v)), eta=lambda v: (_ETA, np.log(v) - np.log1p(-v)))
        for k, v in init.items():
            if k not in tr:
                raise ValueError(f"init accepts {sorted(tr)}; got {k!r}")
            i, val = tr[k](v)
            if k == "mu_c":
                th[0:2] = val
            elif k == "eta":                       # the detection stage gives one fraction: every group starts there
                th[_ETA:_ETA + self.n_groups] = val
            else:
                th[i] = val
        return np.clip(th[self.free], lo, hi)

    def _full(self, x, base=None):
        th = np.zeros(len(self.param_names)) if base is None else np.array(base, float)
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
        names = [self.param_names[k] for k in self.free]
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
                                     init_start=init is not None, param_names=self.param_names,
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
        eta = _sig(th[_ETA + np.asarray(self._d["group"])])
        logit = (np.log1p(-eta) + lc + np.asarray(self._d["lsum"])) - (np.log(eta) + lf)
        return _sig(logit)

    @property
    def axis_llr(self) -> dict[str, np.ndarray]:
        """``ℓ_k`` per independent line. Proper motion and parallax are one line, ``"A"`` (hub §3.5); the
        modelled ``lines`` follow with their own values (NaN where a star has no data)."""
        _, lc, lf = self._comp()
        if lf is None:
            raise RuntimeError("no field component: the log-likelihood ratio is undefined")
        out = {"A": lc - lf}
        out.update({k: np.asarray(self.data.lines[k], float) for k in self.lines})
        return out

    def classify(self, rule: TwoLineRule = TwoLineRule(), extra_lines: Mapping[str, np.ndarray] | None = None) -> np.ndarray:
        """Apply the two-line rule. ``extra_lines``: per-star ``ℓ_k`` of lines not modelled here
        (photometry ``"B"``, RV ``"C"``, youth ``"D"``), NaN where a star has no data."""
        llr = dict(self.axis_llr)
        for k, v in (extra_lines or {}).items():
            if k in llr:
                raise ValueError(f"line {k!r} is already modelled")
            llr[k] = np.asarray(v, float)
        return rule.apply(self.p, llr)


#: G edges of the per-G mixing fractions validated on the NGC 6383 cone (hub membership-cmd-axis §1.2).
CONE_G_EDGES: tuple[float, float] = (16.0, 18.5)
#: Lowest p_hi whose specific risk is <= 5 % in all six subgroups for :func:`cone_mixture` (V4, hub finding §4.7).
CONE_P_HI: float = 0.8


def cone_groups(mag) -> np.ndarray:
    """The per-star group index of :func:`cone_mixture`: 0 for G < 16, 1 for 16 <= G < 18.5, 2 for G >= 18.5."""
    return np.digitize(np.asarray(mag, float), CONE_G_EDGES).astype(int)


def cone_mixture(data: MembershipData, selection: Truncation) -> MixtureModel:
    """The configuration validated on a real young-cluster cone (hub ``agent-findings/membership-cmd-axis-2026-11-05.md``).

    Empirical field (flat PM over the region, ``data.field_plx_logpdf``), the CMD line ``"B"`` in p
    (:func:`photometric_llr`) and one mixing fraction per G bin (``data.group`` from :func:`cone_groups`). It passed V2
    in all six subgroups, V0, C0 and the X-ray/YSO line on the 300 replicates of the validation, beat pyUPMASK in all
    six, and was confirmed against CMD line + per-G fraction on 300 new replicates by a rule fixed before running.
    The constructor defaults of :class:`MixtureModel` stay off (``lines=()``): a p that will weight an isochrone fit must
    not carry the CMD. Fit with one ``prior_field`` per group; classify with ``TwoLineRule(p_hi=CONE_P_HI,
    p_lo=1 - CONE_P_HI)``. The faint bin's field prior inherited from the hub (A1) sits ~32 % high (hub thread B16).
    """
    if "B" not in data.lines:
        raise ValueError("cone_mixture needs the CMD line: data.lines['B'] from photometric_llr")
    if data.field_plx_logpdf is None:
        raise ValueError("cone_mixture needs data.field_plx_logpdf (e.g. SidebandTemplate.log_plx)")
    g = np.asarray(data.group) if data.group is not None else None
    if g is None or g.min() < 0 or g.max() > len(CONE_G_EDGES):
        raise ValueError("cone_mixture needs data.group in {0, 1, 2} (cone_groups of G)")
    return MixtureModel(data, selection, EmpiricalField(), lines=("B",), n_groups=len(CONE_G_EDGES) + 1)


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
                           prior_field: tuple[float, float] | None = None,
                           member_donors: MembershipData | None = None) -> dict:
    """N2 of hub ``methodology.md`` §A.3: synthetic members injected into a real field catalogue, refit, score.

    ``field_catalog`` must hold **no cluster members** (an empty control field or a sideband) and already
    satisfy the selection; every row is labelled ``y = 0``. Each replicate draws ``Poisson((1 − η̂) N)``
    members from the fitted cluster of ``model`` with the errors and ``aux`` of random rows of
    ``member_donors`` -- the detected members, so the injected stars have the cluster's magnitudes and errors --
    or, if it is ``None``, of the field catalogue. The second choice gives the injected members the field's G
    distribution (faint where a young cluster is bright), and an injection that does not look like the cluster
    fails its predictive check (hub ``methodology.md`` §A.1.3; found validating AD-I48, hub
    ``agent-findings/membership-rule-validation-2026-11-05.md``). Pass ``member_donors`` on real data. It
    refits with ``n_starts`` and scores ``p`` against the truth: CORP (pooled and per subgroup), AUC,
    ``Σp − N_true`` and, with ``rule``, its global and specific risks (§A.2 row e): the false-member rate
    among stars it calls "member" and the completeness among true members. ``extra_lines(data, y, rng)``
    supplies the other lines' ``ℓ_k`` for the rule. Replicates are padded to one shape (zero weight), so
    the likelihood compiles once.

    Each replicate is fitted **the way the real fit was**: the same ``n_starts``, ``prior_field`` and the
    same detection-stage ``init`` and ``ridge`` -- never the fitted values, which would seed every replicate at its own
    truth and hide the modes the real fit can fall into (hub ``state/findings.yaml`` R26-13).
    """
    if model.lines or model.empirical or model.n_groups > 1:
        raise NotImplementedError(
            "calibrate_by_injection draws members in astrometry only: it cannot give them a CMD line, a sideband "
            "parallax density or a group. Inject with a harness that draws those too (hub "
            "agent-findings/scripts/membership_cmd_axis.py) instead of letting a modelled axis drop silently.")
    fit = model._need_fit()
    par = fit.params()
    rng = np.random.default_rng(seed)
    n_fit = model.data.n
    n_mem = rng.poisson((1 - par["eta"]) * n_fit, n_rep) if model.has_field else np.full(n_rep, n_fit)
    n_pad = field_catalog.n + int(n_mem.max())
    pool = field_catalog if member_donors is None else member_donors
    P, Y, W, groups, sums, cls = [], [], [], {k: [] for k in (subgroups or {})}, [], []
    for b in range(n_rep):
        donors = pool.take(rng.integers(pool.n, size=n_mem[b]))
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
        vals = np.array([float(statistic(_natural(model._full(x, fit.theta), model.n_groups))) for x in xs])
    else:
        raise ValueError("mode must be 'bernoulli' or 'in_model'")
    return PropagationResult(values=vals, mode=mode, n_draws=n_draws)
