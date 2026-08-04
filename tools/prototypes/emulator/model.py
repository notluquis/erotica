"""Unbinned per-star CMD likelihood over the differentiable emulator.

Design choices, each of which is a departure from something that failed
=======================================================================

**Unbinned, not binned.**  The package's Hess path histograms the CMD, which is
piecewise constant in the bin interiors -- 46.7% exactly-zero gradient in log age
and 95.0% in metallicity.  No smoothing of a histogram removes that; the fix is
to not histogram.

**Per-star mass marginalised over a fixed quadrature, not sampled as a latent.**
Chi et al. (2026) and the earlier prototype in ``../isochrone_numpyro_fitter.py``
both sample one mass latent per star, giving ``N + 7`` parameters.  Here

.. math::
   \\mathcal{L}_i = \\sum_k w_k \\, p(d_i \\mid m_k, \\theta),

with the log-mass nodes :math:`m_k` and IMF weights :math:`w_k` **fixed at build
time**.  This is a closed-form finite sum -- not adaptive quadrature -- so it is
exactly differentiable and costs a fixed number of operations.  Two consequences
that matter:

1. The parameter count is **7 regardless of N**.  At the census median of ~61
   members, per-star latents give 68 parameters for 122 numbers; marginalising
   gives 7.  This is what makes the N floor a question about information rather
   than about identifiability.
2. The emulator is evaluated on the mass nodes **once per log-density call**, not
   once per star: cost is ``K + K*J`` evaluations, independent of N.

**The amplitude is not a parameter.**  An unbinned per-star likelihood is
already conditioned on N, so the free ``log_s`` that coupled global scale to CMD
shape in the Hess path has no analogue here.  The scale-shape ridge is gone by
construction, not by a prior.

**Binaries are flux-summed with the mass ratio marginalised**, not modelled as a
fixed 0.75 mag offset.  ``q`` runs over a fixed Gauss-Legendre grid.

Known omission, stated rather than hidden
-----------------------------------------
There is **no completeness/selection term**.  For a magnitude-limited sample the
correct likelihood divides by :math:`\\int S(m,\\theta) p_{\\rm IMF}(m)\\,dm`,
which depends on :math:`\\theta`; SIESTA (Ferreira et al. 2024) carries exactly
this term.  Omitting it is defensible only when the fit is conditioned on a
sample whose selection is independent of the parameters, which a magnitude cut
is not.  ``injection_recovery.py`` measures the size of the resulting bias
rather than assuming it is small.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.special import logsumexp

from emulator import K_BPRP, K_G, TrackEmulator, kroupa_log_pdf_logmass

LN10 = np.log(10.0)


@dataclass(frozen=True)
class Quadrature:
    """Fixed mass / mass-ratio quadrature.  Built once, never parameter-dependent."""

    logmass: jnp.ndarray      # (K,) log-mass nodes
    logw: jnp.ndarray         # (K,) normalised log IMF weights
    q: jnp.ndarray            # (J,) mass ratios
    logwq: jnp.ndarray        # (J,) normalised log weights


def make_quadrature(em: TrackEmulator, n_mass: int = 192, n_q: int = 5,
                    q_lo: float = 0.10) -> Quadrature:
    """Trapezoid nodes in log-mass with IMF weights; Gauss-Legendre in ``q``."""
    lm = jnp.linspace(em.logmass_lo, em.logmass_hi, n_mass)
    lw = kroupa_log_pdf_logmass(lm)
    # trapezoid weights on a uniform log-mass grid
    dw = jnp.ones(n_mass).at[0].set(0.5).at[-1].set(0.5)
    lw = lw + jnp.log(dw)
    lw = lw - logsumexp(lw)

    xq, wq = np.polynomial.legendre.leggauss(n_q)
    q = 0.5 * (xq + 1.0) * (1.0 - q_lo) + q_lo
    lwq = jnp.log(jnp.asarray(wq) / np.sum(wq))
    return Quadrature(lm, lw, jnp.asarray(q), lwq)


def star_log_likelihood(em, quad, G_obs, C_obs, sG, sC, logage, feh, dm, Av,
                        f_bin, f_out, jitter, box):
    """log L_i for every star, as a (N,) vector.  Pure JAX, no sampling."""
    Gm, Cm = em.absolute(logage, feh, quad.logmass)               # (K,)
    Gs = Gm + dm + K_G * Av
    Cs = Cm + K_BPRP * Av

    lm2 = quad.logmass[:, None] + jnp.log10(quad.q)[None, :]      # (K,J)
    lm2 = jnp.clip(lm2, em.logmass_lo, em.logmass_hi)
    Gb, Cb = em.flux_sum(logage, feh,
                         jnp.broadcast_to(quad.logmass[:, None], lm2.shape),
                         lm2, dm, Av)                              # (K,J)

    vG = sG[:, None] ** 2 + jitter**2
    vC = sC[:, None] ** 2 + jitter**2

    def _gauss(x, mu, var):
        return -0.5 * ((x - mu) ** 2 / var + jnp.log(2 * jnp.pi * var))

    # single: (N,K)
    ls = (_gauss(G_obs[:, None], Gs[None, :], vG)
          + _gauss(C_obs[:, None], Cs[None, :], vC)
          + quad.logw[None, :])
    ls = logsumexp(ls, axis=1)

    # binary: (N,K,J)
    lb = (_gauss(G_obs[:, None, None], Gb[None, :, :], vG[:, :, None])
          + _gauss(C_obs[:, None, None], Cb[None, :, :], vC[:, :, None])
          + quad.logw[None, :, None] + quad.logwq[None, None, :])
    lb = logsumexp(lb.reshape(lb.shape[0], -1), axis=1)

    # outlier: uniform over the CMD box
    lo = jnp.full_like(ls, -jnp.log(box[0]) - jnp.log(box[1]))

    return logsumexp(jnp.stack([
        jnp.log(jnp.clip((1 - f_out) * (1 - f_bin), 1e-300)) + ls,
        jnp.log(jnp.clip((1 - f_out) * f_bin, 1e-300)) + lb,
        jnp.log(jnp.clip(f_out, 1e-300)) + lo,
    ]), axis=0)


def make_model(em, quad, box, *, dm_mu, dm_sigma, feh_mu=0.0, feh_sigma=0.25,
               Av_lo=0.0, Av_hi=3.0):
    """Return a NumPyro model closing over the emulator and quadrature.

    ``dm_sigma`` is deliberately an argument: the earlier prototype used 0.03 mag,
    which pins the dm-log age-Av ridge by fiat and buys convergence with the
    prior.  ``injection_recovery.py`` runs the gate at both a tight and a
    defensible width and reports both.
    """

    def model(G_obs, C_obs, sG, sC):
        logage = numpyro.sample("log_age", dist.Uniform(em.age_lo, em.age_hi))
        feh = numpyro.sample("feh", dist.TruncatedNormal(
            feh_mu, feh_sigma, low=em.feh_lo, high=em.feh_hi))
        dm = numpyro.sample("dm", dist.Normal(dm_mu, dm_sigma))
        Av = numpyro.sample("Av", dist.Uniform(Av_lo, Av_hi))
        f_bin = numpyro.sample("f_bin", dist.Beta(2.0, 3.0))
        f_out = numpyro.sample("f_out", dist.Beta(1.5, 20.0))
        jitter = numpyro.sample("jitter", dist.HalfNormal(0.05))
        ll = star_log_likelihood(em, quad, G_obs, C_obs, sG, sC,
                                 logage, feh, dm, Av, f_bin, f_out, jitter, box)
        numpyro.factor("obs", ll.sum())

    return model


def physical_logp_fn(em, quad, data, box, *, dm_mu, dm_sigma):
    """``(log_age, feh, dm, Av) -> log posterior`` with nuisances fixed.

    This is the function the gradient audit differentiates.  Nuisances are held
    at plausible constants so that the measured zero-gradient fraction is a
    property of the **forward model**, which is what is being compared against
    the binned Hess path -- not of the mixture weights.
    """
    G_obs, C_obs, sG, sC = data

    def logp(theta):
        logage, feh, dm, Av = theta
        ll = star_log_likelihood(em, quad, G_obs, C_obs, sG, sC,
                                 logage, feh, dm, Av,
                                 0.3, 0.02, 0.02, box).sum()
        lp = (dist.Uniform(em.age_lo, em.age_hi).log_prob(logage)
              + dist.Normal(dm_mu, dm_sigma).log_prob(dm)
              + dist.Uniform(0.0, 3.0).log_prob(Av))
        return ll + lp

    return logp


def simulate(em, quad, truth, n, *, seed=0, f_bin=0.3, f_out=0.02,
             err_floor=0.01, mag_limit=None, oracle=None, model_error=0.0):
    """Draw a synthetic cluster.

    ``oracle`` -- if given, a dict with keys ``mass``/``G``/``BPRP`` from
    :func:`mist_grid.raw_isochrone` -- generates the stars from the **raw MIST
    track**, not from the emulator.  That is the whole point: if the truth comes
    out of the same interpolator the fit uses, emulator error cancels exactly and
    the test is structurally blind to it.
    """
    rng = np.random.default_rng(seed)
    # CONTINUOUS masses by inverse-CDF, *not* `rng.choice(quad.logmass)`.
    # Drawing the simulated masses from the model's own quadrature nodes puts
    # every star exactly on a node, so the marginalisation sum is evaluated at
    # its own support points and the likelihood is inflated by an amount that
    # depends on K -- measured: max |d lnL| of 233 between K=64 and K=192 over
    # a 0.4 dex window, which vanished once the draw was made continuous.  That
    # is the same class of circularity as generating the truth from the
    # emulator: the test tells you about your own discretisation.
    fine = np.linspace(em.logmass_lo, em.logmass_hi, 4096)
    pdf = np.exp(np.asarray(kroupa_log_pdf_logmass(jnp.asarray(fine))))
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    lm = np.interp(rng.random(n), cdf, fine)

    if oracle is None:
        G0, C0 = em.absolute(truth["log_age"], truth["feh"], jnp.asarray(lm))
        G0, C0 = np.asarray(G0), np.asarray(C0)
    else:
        m = 10.0**lm
        G0 = np.interp(m, oracle["mass"], oracle["G"])
        C0 = np.interp(m, oracle["mass"], oracle["BPRP"])

    is_bin = rng.random(n) < f_bin
    q = rng.uniform(0.1, 1.0, n)
    lm2 = np.clip(lm + np.log10(q), em.logmass_lo, em.logmass_hi)
    if oracle is None:
        G2, C2 = em.absolute(truth["log_age"], truth["feh"], jnp.asarray(lm2))
        G2, C2 = np.asarray(G2), np.asarray(C2)
    else:
        m2 = 10.0**lm2
        G2 = np.interp(m2, oracle["mass"], oracle["G"])
        C2 = np.interp(m2, oracle["mass"], oracle["BPRP"])
    bp1, rp1, bp2, rp2 = G0 + 0.5 * C0, G0 - 0.5 * C0, G2 + 0.5 * C2, G2 - 0.5 * C2

    def _add(a, b):
        return -2.5 * np.log10(10 ** (-0.4 * a) + 10 ** (-0.4 * b))

    Gt = np.where(is_bin, _add(G0, G2), G0)
    Ct = np.where(is_bin, _add(bp1, bp2) - _add(rp1, rp2), C0)

    G = Gt + truth["dm"] + K_G * truth["Av"]
    C = Ct + K_BPRP * truth["Av"]

    # magnitude-dependent Gaia-like errors
    sG = err_floor * (1 + np.exp(0.6 * (G - 18.5)))
    sC = 2.0 * sG
    G = G + rng.normal(0, sG)
    C = C + rng.normal(0, sC)

    # Unmodelled *model* error: scatter the photometry by an amount the fitted
    # model is not told about.  Without this the simulation has no discrepancy
    # at all -- truth is raw MIST and the emulator reproduces raw MIST to 1e-15
    # at its nodes -- so the posterior comes out ~0.007 dex wide in log age,
    # which is not a credible width for a real cluster age.  Everything that
    # divides by a posterior sd (bias/sd, the N floor) is optimistic until this
    # is nonzero.  Reported `sG`/`sC` are left untouched, which is the point:
    # the fit believes its errors are smaller than they are.
    if model_error > 0:
        G = G + rng.normal(0, model_error, G.shape)
        C = C + rng.normal(0, model_error, C.shape)

    n_out = int(round(f_out * n))
    if n_out:
        idx = rng.choice(n, n_out, replace=False)
        G[idx] = rng.uniform(G.min(), G.max(), n_out)
        C[idx] = rng.uniform(C.min(), C.max(), n_out)

    if mag_limit is not None:
        keep = G < mag_limit
        G, C, sG, sC = G[keep], C[keep], sG[keep], sC[keep]

    box = (float(G.max() - G.min()), float(C.max() - C.min()))
    return (jnp.asarray(G), jnp.asarray(C), jnp.asarray(sG), jnp.asarray(sC)), box
