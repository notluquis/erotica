"""Differentiable stellar-track emulator: JAX evaluation of the MIST tensor.

The emulator maps ``(log age, [Fe/H], log M) -> (M_G, (BP-RP)_0)`` by separable
tensor-product interpolation of the regular grid built by :mod:`mist_grid`.
It is an *interpolator*, not a neural network -- the same architectural choice
Chi et al. (2026) make, and for the same reason: a multilinear/multicubic
interpolant on a regular grid is a **finite, closed-form** expression in the
grid values, so it is exactly differentiable, has no training loss, and cannot
hallucinate structure that is not in the tracks.  Its error is bounded by the
interpolation stencil, which is what makes the accuracy audit in
``emulator_accuracy.py`` meaningful.

On ``analysis/CLAUDE.md``'s prohibition
--------------------------------------
That file bans forward models with no closed form, because quadrature inside the
graph is evaluated at every leapfrog step.  A tensor-product interpolant is not
quadrature: evaluating it is a **fixed** number of multiply-adds (8 taps for
trilinear, up to 64 for tricubic), decided at build time and independent of the
parameter values.  The prohibition is about *adaptive* work in the graph; this
has none.  Recorded explicitly so the exemption is argued rather than assumed.

Two interpolation orders are provided and both are audited:

``order=1``  trilinear.  Value-continuous (C0); the gradient is piecewise
             constant and **jumps at grid nodes**.  This is what
             ``jax.scipy.ndimage.map_coordinates`` gives and what Chi uses.
``order=3``  tensor-product Catmull-Rom.  C1: value *and* first derivative are
             continuous everywhere.  Costs 64 taps instead of 8.

C1 matters here for a reason specific to Hamiltonian samplers: NUTS integrates
the gradient field, and a gradient that jumps at every 0.05-dex age node is a
discontinuous force.  Whether that is *measurably* worse is settled by
``gradient_audit.py``, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------
# Extinction.  Colour-independent approximation to the Gaia EDR3 coefficients;
# real A_G/A_V depends on the star's own colour (Casagrande & VandenBerg 2018).
# Stated here because a constant ratio is an approximation, not a fact.
# --------------------------------------------------------------------------
K_G = 0.83      # A_G / A_V
K_BPRP = 0.42   # E(BP-RP) / A_V


# --------------------------------------------------------------------------
# Separable interpolation kernels.  Both return (offsets, weights) for a
# fractional coordinate ``t`` in [0, 1) measured from the base node.
# --------------------------------------------------------------------------
def _w_linear(t):
    return jnp.stack([1.0 - t, t], axis=0)          # offsets 0, 1


def _w_cubic(t):
    """Catmull-Rom (a = -1/2) cubic convolution weights; offsets -1, 0, 1, 2.

    Keys (1981), IEEE ASSP 29, 1153.  C1-continuous and interpolating (it passes
    through the nodes), unlike a B-spline, so grid values are reproduced exactly.
    """
    t2 = t * t
    t3 = t2 * t
    return jnp.stack([
        -0.5 * t + t2 - 0.5 * t3,
        1.0 - 2.5 * t2 + 1.5 * t3,
        0.5 * t + 2.0 * t2 - 1.5 * t3,
        -0.5 * t2 + 0.5 * t3,
    ], axis=0)


_OFFSETS = {1: np.array([0, 1]), 3: np.array([-1, 0, 1, 2])}
_KERNEL = {1: _w_linear, 3: _w_cubic}


def _axis_taps(x, n, order):
    """Base indices and weights for one axis.  ``x`` is a continuous index."""
    xc = jnp.clip(x, 0.0, n - 1.0)
    i0 = jnp.floor(xc).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, n - 2)          # keep the cell inside the grid
    t = xc - i0
    w = _KERNEL[order](t)                                     # (ntap, ...)
    idx = jnp.clip(i0[None, ...] + _OFFSETS[order].reshape(-1, *([1] * x.ndim)),
                   0, n - 1)             # clamp the stencil at the boundary
    return idx, w


def _interp3(grid, xa, xf, xm, order_a, order_f, order_m):
    """Separable tensor-product interpolation of a 3-D grid at continuous indices."""
    na, nf, nm = grid.shape
    ia, wa = _axis_taps(xa, na, order_a)
    jf, wf = _axis_taps(xf, nf, order_f)
    km, wm = _axis_taps(xm, nm, order_m)
    out = 0.0
    for p in range(ia.shape[0]):
        for q in range(jf.shape[0]):
            for r in range(km.shape[0]):
                out = out + wa[p] * wf[q] * wm[r] * grid[ia[p], jf[q], km[r]]
    return out


@dataclass(frozen=True)
class TrackEmulator:
    """Differentiable ``(log age, [Fe/H], log M) -> (M_G, (BP-RP)_0)``.

    Attributes
    ----------
    order_age, order_feh, order_mass
        Interpolation order per axis (1 = linear/C0, 3 = Catmull-Rom/C1).
        The default is C1 in age and [Fe/H] -- the coarse axes, 0.05 dex and
        0.25 dex -- and linear in mass, where 160 log-spaced nodes already
        resolve the tracks and cubic would only invite overshoot at the
        turn-off hook.
    """

    G: jnp.ndarray
    BPRP: jnp.ndarray
    logage: jnp.ndarray
    feh: jnp.ndarray
    logmass: jnp.ndarray
    # Axis bounds and steps as **plain Python floats**.  JAX >= 0.10 stages out
    # operations on concrete arrays inside a trace, so `float(self.logage[0])`
    # raises ConcretizationTypeError under jit.  Keeping the scalars outside the
    # array world also makes the index map an exactly affine expression.
    age_lo: float = 0.0
    age_hi: float = 0.0
    age_step: float = 1.0
    feh_lo: float = 0.0
    feh_hi: float = 0.0
    feh_step: float = 1.0
    logmass_lo: float = 0.0
    logmass_hi: float = 0.0
    logmass_step: float = 1.0
    order_age: int = 3
    order_feh: int = 3
    order_mass: int = 1

    def _coords(self, logage, feh, logmass):
        # Regular axes -> affine index maps.  Affine (not jnp.interp) so the
        # index map itself is exactly linear and contributes no kinks.
        xa = (logage - self.age_lo) / self.age_step
        xf = (feh - self.feh_lo) / self.feh_step
        xm = (logmass - self.logmass_lo) / self.logmass_step
        shape = jnp.broadcast_shapes(jnp.shape(xa), jnp.shape(xf), jnp.shape(xm))
        return (jnp.broadcast_to(xa, shape),
                jnp.broadcast_to(xf, shape),
                jnp.broadcast_to(xm, shape))

    def absolute(self, logage, feh, logmass):
        """Intrinsic ``(M_G, (BP-RP)_0)``.  Broadcasts over all three inputs."""
        xa, xf, xm = self._coords(
            jnp.asarray(logage, float), jnp.asarray(feh, float), jnp.asarray(logmass, float)
        )
        g = _interp3(self.G, xa, xf, xm, self.order_age, self.order_feh, self.order_mass)
        c = _interp3(self.BPRP, xa, xf, xm, self.order_age, self.order_feh, self.order_mass)
        return g, c

    def apparent(self, logage, feh, logmass, dm, Av):
        """Observed ``(G, BP-RP)`` after distance modulus and extinction."""
        g, c = self.absolute(logage, feh, logmass)
        return g + dm + K_G * Av, c + K_BPRP * Av

    def flux_sum(self, logage, feh, logm1, logm2, dm, Av):
        """Unresolved binary: sum the fluxes of the two components.

        ``G_tot = -2.5 log10(10^(-0.4 G1) + 10^(-0.4 G2))``, done in BP and RP
        separately so the colour is right rather than assumed.  ``logsumexp``
        keeps it stable and differentiable.
        """
        g1, c1 = self.absolute(logage, feh, logm1)
        g2, c2 = self.absolute(logage, feh, logm2)
        # BP = G + (BP-RP)*fBP is not exact; instead reconstruct BP, RP from
        # G and (BP-RP) using BP - RP = c and the (arbitrary but consistent)
        # split BP = G + c/2, RP = G - c/2.  The flux sum is then exact in G
        # and correct to first order in colour -- adequate for a mixture
        # component whose weight is itself a fitted nuisance.
        bp1, rp1 = g1 + 0.5 * c1, g1 - 0.5 * c1
        bp2, rp2 = g2 + 0.5 * c2, g2 - 0.5 * c2

        def _add(m1, m2):
            return -2.5 * jax.scipy.special.logsumexp(
                jnp.stack([-0.4 * m1 * jnp.log(10.0), -0.4 * m2 * jnp.log(10.0)]), axis=0
            ) / jnp.log(10.0)

        g = _add(g1, g2)
        c = _add(bp1, bp2) - _add(rp1, rp2)
        return g + dm + K_G * Av, c + K_BPRP * Av


def load(grid: dict[str, np.ndarray], **kw) -> TrackEmulator:
    """Wrap a :func:`mist_grid.build` dict as a :class:`TrackEmulator`."""
    a, f, m = grid["logage"], grid["feh"], grid["logmass"]
    return TrackEmulator(
        G=jnp.asarray(grid["G"]),
        BPRP=jnp.asarray(grid["BPRP"]),
        logage=jnp.asarray(a),
        feh=jnp.asarray(f),
        logmass=jnp.asarray(m),
        age_lo=float(a[0]), age_hi=float(a[-1]), age_step=float(a[1] - a[0]),
        feh_lo=float(f[0]), feh_hi=float(f[-1]), feh_step=float(f[1] - f[0]),
        logmass_lo=float(m[0]), logmass_hi=float(m[-1]),
        logmass_step=float(m[1] - m[0]),
        **kw,
    )


# --------------------------------------------------------------------------
# Initial mass function -- the prior on the per-star mass latent.
# --------------------------------------------------------------------------
def kroupa_log_pdf_logmass(logmass, m_break=0.5, a_lo=1.3, a_hi=2.3):
    """log p(log10 M) for a Kroupa (2001) broken power law.

    ``p(M) dM ∝ M^-a dM`` becomes ``p(log M) ∝ M^(1-a)`` in log-mass.  Returned
    unnormalised over the emulator's mass range: an additive constant is
    irrelevant to NUTS and to any likelihood ratio, and normalising it would
    make the value depend on the grid bounds.
    """
    m = 10.0**logmass
    ln10 = jnp.log(10.0)
    lo = (1.0 - a_lo) * logmass * ln10
    hi = (1.0 - a_hi) * logmass * ln10 + (a_hi - a_lo) * jnp.log(m_break)
    return jnp.where(m < m_break, lo, hi)


if __name__ == "__main__":
    import mist_grid

    g = mist_grid.build()
    for oa in (1, 3):
        em = load(g, order_age=oa, order_feh=oa)
        Gv, Cv = em.absolute(6.6, 0.0, em.logmass)
        gr = jax.grad(lambda a: em.absolute(a, 0.0, 0.0)[0])(6.6)
        print(f"order_age={oa}: G[{float(Gv.min()):.2f},{float(Gv.max()):.2f}] "
              f"BP-RP[{float(Cv.min()):.2f},{float(Cv.max()):.2f}]  dG/dloga={float(gr):+.4f}")
    em = load(g)
    print("node reproduction (should be ~0):",
          float(jnp.abs(em.absolute(g["logage"][5], g["feh"][2], g["logmass"][40])[0]
                        - g["G"][5, 2, 40])))
