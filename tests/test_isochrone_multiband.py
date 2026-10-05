"""Tests for the multiband isochrone likelihood (erotica.analysis._isochrone_multiband).

Oracles (tests/AGENTS.md: an oracle exists independently of the code under test):

* **The 2D fitter** (:class:`IsochroneFitter`, the code that ran C1 and the grid fits): with no
  near-infrared band the n-dimensional likelihood must equal it to rounding -- NumPy value, compiled
  PyTensor value and gradient -- with binaries, off-node, f_bg > 0, sigma_int > 0 and a two-sided
  window. The multiband code never calls the 2D density, so this is a construction oracle, not a
  tautology.
* **Numerical quadrature**: the closed-form n-dimensional segment integral against
  ``scipy.integrate.quad`` of ``scipy.stats.multivariate_normal`` along the segment, with two rank-1
  covariance terms and a missing coordinate (dropping the row and column by hand).
* **Marginal of a Gaussian**: a band missing for every star equals the fitter without that band.
* **Invariance**: shifting the zero-point group's magnitudes and the zero point by the same amount.
* **Finite differences** of the NumPy likelihood for the compiled gradient, with every switch on.

The toy family has closed-form photometry (as ``test_grids.py``) with 2MASS-like columns added.
"""

from __future__ import annotations

import importlib.util
import textwrap
from pathlib import Path

import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis._isochrone import IsochroneFitter
from erotica.analysis._isochrone_multiband import (
    FITZ19_BPRP,
    MultibandIsochroneFitter,
    NIRBand,
    _segment_density_nd,
    fitz19_k,
)
from erotica.analysis.grids import MISTGrid

requires_bayes = pytest.mark.skipif(
    importlib.util.find_spec("pytensor") is None, reason="requires the bayes extra (pytensor)"
)

NGC = Path.home() / "erotica" / "data" / "test" / "NGC6383"
MIST12 = NGC / "MIST" / "UBVRIplus"
C1_SAMPLE = NGC / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
C1_PRIORS = dict(
    loga_range=(6.0, 7.0), Av_range=(0.5, 2.0), dm_mu=10.3, dm_sigma=0.2, dm_range=(9.5, 10.7)
)

_M = np.geomspace(0.1, 8.0, 90)
_FEH = (-0.5, -0.25, 0.0, 0.25)
_AGES = (6.3, 6.4, 6.5, 6.6, 6.7)
TMASS = ("2MASS_J", "2MASS_H", "2MASS_Ks")


def _phot(mass, loga, feh):
    """Closed-form toy photometry: G, BP-RP and three NIR magnitudes."""
    lm = np.log10(mass)
    G = 4.6 - 6.5 * lm - 2.0 * (7.0 - loga) / (1.0 + mass**2) + 0.6 * feh * (1.0 - lm)
    col = 0.8 - 1.4 * lm + 0.3 * feh * (1.0 + lm) + 0.1 * (7.0 - loga)
    J = G - 1.1 * col - 0.2 * (7.0 - loga) * mass / (1 + mass)
    H = J - 0.35 * col - 0.05 * feh
    K = H - 0.12 * col - 0.03 * (7.0 - loga)
    return G, col, J, H, K


def _write_family(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    for feh in _FEH:
        Z = 0.0142857 * 10**feh
        Y = 0.249 + 1.49 * Z
        head = textwrap.dedent(f"""\
            # MIST version number  = 1.2
            #  Yinit        Zinit   [Fe/H]   [a/Fe]  v/vcrit
            # {Y:.4f}  {Z:.5E}  {feh:7.2f}     0.00     0.40
            # EEP log10_isochrone_age_yr initial_mass [Fe/H]_init Gaia_G_EDR3 Gaia_BP_EDR3 Gaia_RP_EDR3 2MASS_J 2MASS_H 2MASS_Ks
        """)
        rows = []
        for a in _AGES:
            G, c, J, H, K = _phot(_M, a, feh)
            for i in range(len(_M)):
                rows.append(
                    f"{i + 1} {a:.4f} {_M[i]:.8f} {feh:.2f} {G[i]:.8f} "
                    f"{G[i] + 0.6 * c[i]:.8f} {G[i] - 0.4 * c[i]:.8f} {J[i]:.8f} {H[i]:.8f} {K[i]:.8f}"
                )
        tag = f"{'p' if feh >= 0 else 'm'}{abs(feh):.2f}"
        (d / f"MIST_v1.2_feh_{tag}_afe_p0.0_vvcrit0.4_toy.iso.cmd").write_text(
            head + "\n".join(rows) + "\n"
        )
    return d


def _stars(n=120, loga=6.52, feh=-0.1, dm=10.0, Av=0.3, seed=3, missing=True):
    rng = np.random.default_rng(seed)
    m = np.geomspace(0.15, 6.0, n)
    G, c, J, H, K = _phot(m, loga, feh)
    e = 0.01
    t = QTable(
        {
            "Gmag": G + dm + 0.84 * Av + rng.normal(0, e, n),
            "G_BPmag": G + 0.6 * c + dm + 1.07 * Av + rng.normal(0, e, n),
            "G_RPmag": G - 0.4 * c + dm + 0.62 * Av + rng.normal(0, e, n),
            "e_Gmag": np.full(n, e),
            "e_G_BPmag": np.full(n, e),
            "e_G_RPmag": np.full(n, e),
            "probability_hdbscan": np.ones(n),
        }
    )
    for name, x, k in (("J", J, 0.29), ("H", H, 0.18), ("Ks", K, 0.12)):
        v = x + dm + k * Av + rng.normal(0, 0.03, n)
        if missing:
            v[rng.uniform(size=n) < 0.1] = np.nan
        t[f"{name}_fit"] = v
        t[f"e{name}_fit"] = np.full(n, 0.03)
        t[f"src_{name}"] = np.where(rng.uniform(size=n) < 0.7, "VIRAC2->2MASS", "2MASS")
    return t


NIR = (
    NIRBand("2MASS_J", "J_fit", "eJ_fit", 12350.0, "J", "src_J", ("VIRAC2->2MASS",)),
    NIRBand("2MASS_H", "H_fit", "eH_fit", 16620.0, "H", "src_H", ("VIRAC2->2MASS",)),
    NIRBand("2MASS_Ks", "Ks_fit", "eKs_fit", 21590.0, "K", "src_Ks", ("VIRAC2->2MASS",)),
)
PRI = dict(
    loga_range=(6.3, 6.7), Av_range=(0.0, 1.0), dm_mu=10.0, dm_sigma=0.3, dm_range=(9.5, 10.5)
)
BIN = dict(alpha=0.09, beta=0.94)
SINGLE = dict(alpha=0.0, beta=0.0)
POINTS = [(-0.13, 6.45, 10.02, 0.31, 0.02, 0.01), (0.1, 6.52, 9.98, 0.1, 0.0, 0.0)]


def _grid(tmp_path, bands=None):
    d = _write_family(tmp_path / "fam")
    return MISTGrid(d, bands=bands or ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3", *TMASS))


def _pair(tmp_path, pri, window=None, **kw):
    g = _grid(tmp_path)
    f2 = IsochroneFitter(grid=g, **pri)
    f2.setup(_stars(), prob_threshold=0.0, mag_window=window)
    fn = MultibandIsochroneFitter(grid=g, **pri, **kw)
    fn.setup(_stars(), prob_threshold=0.0, mag_window=window)
    return f2, fn


# ---------------------------------------------------------------------------------------------
# Control: Gaia-only multiband == the 2D fitter
# ---------------------------------------------------------------------------------------------


@requires_bayes
@pytest.mark.parametrize("pri", [SINGLE, BIN], ids=["single", "binaries"])
@pytest.mark.parametrize("window", [None, (11.0, 17.5)], ids=["one-sided", "two-sided"])
def test_gaia_only_reproduces_the_2d_likelihood(tmp_path, pri, window):
    """Oracle: :class:`IsochroneFitter`. Mutations seen red (2026-10-05): floor added per band
    (colour gets two), background over one coordinate, Woodbury correction dropped (binaries)."""
    f2, fn = _pair(tmp_path, {**PRI, **pri}, window)
    for p in POINTS:
        a, b = f2.loglike(*p), fn.loglike(*p)
        assert abs(a - b) < 1e-10 * abs(a), (p, a, b)


@requires_bayes
def test_gaia_only_compiled_value_and_gradient_equal_the_2d(tmp_path):
    """Oracle: the 2D fitter's compiled graph. NaN gradients hide in the safe-branch ``where``."""
    f2, fn = _pair(tmp_path, {**PRI, **BIN}, (11.0, 17.5))
    c2, cn = f2._compiled_loglike(), fn._compiled_loglike_nd()
    for p in POINTS:
        v2, g2 = c2(list(p))
        vn, gn = cn(list(p))
        assert np.isfinite(gn).all()
        assert abs(v2 - vn) < 1e-10 * abs(v2)
        np.testing.assert_allclose(gn, g2, rtol=1e-8, atol=1e-8 * abs(v2))


@requires_bayes
def test_c1_sample_mist_v12_gaia_only_equals_2d():
    """The same control on the 254 members of C1 with MIST v1.2 (skips without the files)."""
    for p in (MIST12, C1_SAMPLE):
        if not p.exists():
            pytest.skip(f"data not present on this machine: {p}")
    from astropy.table import Table

    g = MISTGrid(MIST12, loga_range=(5.95, 7.05)).select(feh=[-0.5, -0.25, 0.0])
    data = QTable(Table.read(C1_SAMPLE))
    f2 = IsochroneFitter(grid=g, **C1_PRIORS)
    f2.setup(data)
    fn = MultibandIsochroneFitter(grid=g, **C1_PRIORS)
    fn.setup(data)
    for p in [(-0.37, 6.07, 10.26, 0.64, 0.12, 0.08), (-0.1, 6.4, 10.2, 1.0, 0.05, 0.02)]:
        a, b = f2.loglike(*p), fn.loglike(*p)
        assert abs(a - b) < 1e-10 * abs(a), (p, a, b)


# ---------------------------------------------------------------------------------------------
# The n-dimensional segment integral
# ---------------------------------------------------------------------------------------------


def test_segment_integral_matches_quadrature_with_two_rank1_terms_and_a_missing_band():
    """Oracle: ``quad`` over t of ``multivariate_normal.pdf`` with the full covariance built by
    hand, the missing coordinate's row and column deleted. Mutation seen red: the Woodbury
    capacitance's off-diagonal sign flipped."""
    from scipy.integrate import quad
    from scipy.stats import multivariate_normal

    rng = np.random.default_rng(7)
    n, N, S = 4, 3, 5
    X = rng.normal(0, 0.3, (N, n))
    sig = rng.uniform(0.05, 0.2, (N, n))
    has = np.ones((N, n), bool)
    has[1, 2] = False
    has[2, 0] = False
    A = rng.normal(0, 0.3, (S, n))
    B = A + rng.normal(0, 0.4, (S, n))
    B[3] = A[3] + 1e-6  # a near-zero segment takes the point branch
    U1 = rng.normal(0, 0.1, (S, n))
    U2 = rng.normal(0, 0.05, (S, n))
    sw = [np.where(has[:, c], 1 / sig[:, c], 0.0) for c in range(n)]
    J0, J1 = _segment_density_nd(
        [X[:, c] for c in range(n)],
        sw,
        [A[:, c] for c in range(n)],
        [B[:, c] for c in range(n)],
        [[U1[:, c] for c in range(n)], [U2[:, c] for c in range(n)]],
        first_moment=True,
    )
    for i in range(N):
        o = has[i]
        for k in range(S):
            C = np.diag(sig[i] ** 2) + np.outer(U1[k], U1[k]) + np.outer(U2[k], U2[k])
            C = C[np.ix_(o, o)]

            def pdf(t, i=i, k=k, o=o, C=C):
                return multivariate_normal.pdf(X[i, o], A[k, o] + t * (B[k, o] - A[k, o]), C)

            ref0 = quad(pdf, 0, 1, epsabs=0, epsrel=1e-11)[0]
            ref1 = quad(lambda t, pdf=pdf: t * pdf(t), 0, 1, epsabs=0, epsrel=1e-11)[0]
            tol = (
                1e-4 if k == 3 else 1e-8
            )  # point branch: evaluated at A, exact only at zero length
            assert abs(J0[i, k] - ref0) <= tol * ref0 + 1e-300, (i, k, J0[i, k], ref0)
            assert abs(J1[i, k] - ref1) <= tol * ref1 + 1e-300, (i, k, J1[i, k], ref1)


# ---------------------------------------------------------------------------------------------
# Missing bands, zero points, sigma_av, gradient
# ---------------------------------------------------------------------------------------------


@requires_bayes
def test_a_band_missing_for_every_star_is_the_fitter_without_it(tmp_path):
    """Oracle: the marginal of a Gaussian drops the row and column. Fitter A has J, H, Ks with H
    blanked for every star; fitter B has J and Ks only. Background spans: H contributes nothing."""
    g = _grid(tmp_path)
    data = _stars()
    data["H_fit"] = np.nan
    fa = MultibandIsochroneFitter(grid=g, nir_bands=NIR, **PRI, **BIN)
    fa.setup(data, prob_threshold=0.0)
    fb = MultibandIsochroneFitter(grid=g, nir_bands=(NIR[0], NIR[2]), **PRI, **BIN)
    fb.setup(data, prob_threshold=0.0)
    p = (-0.13, 6.45, 10.02, 0.31, 0.02, 0.01)
    a = fa.loglike(*p, zp=(0.01, 0.0, -0.02))
    b = fb.loglike(*p, zp=(0.01, -0.02))
    assert abs(a - b) < 1e-10 * abs(a), (a, b)


@requires_bayes
def test_zero_point_shifts_only_its_group_and_is_invariant(tmp_path):
    """Oracle: moving the group's observed J by delta and the zero point by delta changes nothing;
    moving it with the zero point fixed does. Mutation seen red: zero point applied to all stars."""
    g = _grid(tmp_path)
    base = _stars()
    shifted = _stars()
    grp = np.asarray(base["src_J"]) == "VIRAC2->2MASS"
    shifted["J_fit"] = np.where(grp, base["J_fit"] + 0.05, base["J_fit"])
    f0 = MultibandIsochroneFitter(grid=g, nir_bands=NIR, **PRI, **BIN)
    f0.setup(base, prob_threshold=0.0)
    f1 = MultibandIsochroneFitter(grid=g, nir_bands=NIR, **PRI, **BIN)
    f1.setup(shifted, prob_threshold=0.0)
    p = (-0.13, 6.45, 10.02, 0.31, 0.02, 0.01)
    a = f0.loglike(*p, zp=(0.0, 0.0, 0.0))
    b = f1.loglike(*p, zp=(0.05, 0.0, 0.0))
    c = f1.loglike(*p, zp=(0.0, 0.0, 0.0))
    assert abs(a - b) < 1e-10 * abs(a)
    assert abs(a - c) > 1.0


@requires_bayes
def test_sigma_av_at_zero_is_the_model_without_it(tmp_path):
    g = _grid(tmp_path)
    fa = MultibandIsochroneFitter(grid=g, nir_bands=NIR, ext="fitz19", sigma_av=True, **PRI, **BIN)
    fa.setup(_stars(), prob_threshold=0.0)
    fb = MultibandIsochroneFitter(grid=g, nir_bands=NIR, ext="fitz19", **PRI, **BIN)
    fb.setup(_stars(), prob_threshold=0.0)
    p = (-0.13, 6.45, 10.02, 0.31, 0.02, 0.01)
    a = fa.loglike(*p, sigma_av=0.0)
    assert abs(a - fb.loglike(*p)) < 1e-10 * abs(a)
    assert fa.loglike(*p, sigma_av=0.2) != a


@requires_bayes
@pytest.mark.slow  # ~280 s: the compile of the fitz19 + sigma_av graph
def test_compiled_gradient_matches_finite_differences_with_every_switch(tmp_path):
    """Oracle: central differences of the NumPy likelihood (an independent evaluation path)."""
    g = _grid(tmp_path)
    f = MultibandIsochroneFitter(grid=g, nir_bands=NIR, ext="fitz19", sigma_av=True, **PRI, **BIN)
    f.setup(_stars(), prob_threshold=0.0)
    th = np.array([-0.13, 6.45, 10.02, 0.31, 0.03, 0.02, 0.08, 0.01, -0.005, 0.02])
    v, gr = f._compiled_loglike_nd()(th)
    assert abs(v - f.loglike_theta(th)) < 1e-9 * abs(v)
    for k in range(len(th)):
        h = 1e-6
        tp, tm = th.copy(), th.copy()
        tp[k] += h
        tm[k] -= h
        fd = (f.loglike_theta(tp) - f.loglike_theta(tm)) / (2 * h)
        assert abs(fd - gr[k]) < 1e-4 * max(1.0, abs(gr[k])), (f.param_names[k], fd, gr[k])


# ---------------------------------------------------------------------------------------------
# Extinction per point
# ---------------------------------------------------------------------------------------------


def test_fitz19_coefficients_are_the_esa_table_rows():
    """Oracle: values of ``Fitz19_EDR3_MainSequence.csv`` (BPRP rows), typed here from the file:
    the intercepts, i.e. k at X = 0 with the A terms evaluated at A0 -> 0 (the code clips A0 at
    0.01, so compare at 0.01 by the formula)."""
    intercepts = {"G": 0.995969721536602, "BP": 1.15363197483424, "RP": 0.66320787941067,
                  "J": 0.340345410744913, "H": 0.25505435103948, "K": 0.19404852029171}  # fmt: skip
    for b, a0 in intercepts.items():
        a = FITZ19_BPRP[b]
        ref = a0 + a[4] * 0.01 + a[5] * 1e-4 + a[6] * 1e-6
        assert abs(fitz19_k(b, 0.0, 0.0) - ref) < 1e-12
    # the near infrared is far less extinguished than G, and the order J > H > Ks holds
    kg, kj, kh, kk = (fitz19_k(b, 1.0, 1.0) for b in ("G", "J", "H", "K"))
    assert kg > kj > kh > kk > 0.15


@requires_bayes
def test_fitz19_extinction_is_applied_per_point(tmp_path):
    """The apparent single-star coordinates are M + dm + k(X0, A_V) A_V with X0 each point's own
    intrinsic colour (oracle: ``fitz19_k`` evaluated by hand on the interpolated node)."""
    g = _grid(tmp_path)
    f = MultibandIsochroneFitter(grid=g, nir_bands=NIR, ext="fitz19", **PRI, **BIN)
    f.setup(_stars(), prob_threshold=0.0)
    X = f._interp_isochrone(0.0, 6.5, np)
    d = f._deposit_nd(0.0, 6.5, 10.0, 1.2, np)
    col0 = X[2]
    np.testing.assert_allclose(d["single"][0], X[1] + 10.0 + fitz19_k("G", col0, 1.2) * 1.2)
    kc = fitz19_k("BP", col0, 1.2) - fitz19_k("RP", col0, 1.2)
    np.testing.assert_allclose(d["single"][1], X[2] + kc * 1.2)
    np.testing.assert_allclose(d["single"][4], X[5] + 10.0 + fitz19_k("K", col0, 1.2) * 1.2)
    assert np.ptp(d["k_single"][0]) > 0.05  # varies along the track


@requires_bayes
def test_generator_transfers_the_real_missing_pattern(tmp_path):
    g = _grid(tmp_path)
    f = MultibandIsochroneFitter(grid=g, nir_bands=NIR, **PRI, **BIN)
    f.setup(_stars(), prob_threshold=0.0)
    pat = {"G": f._obs_mag, "has": f._nir_has, "group": f._nir_group}
    cols = f.draw_stars_nd(
        [0.0, 6.5, 10.0, 0.3, 0.02, 0.0, 0.0, 0.0, 0.0],
        200,
        np.random.default_rng(1),
        nir_pattern=pat,
    )
    for j, b in enumerate(NIR):
        frac_real = 1 - f._nir_has[j].mean()
        frac_syn = np.isnan(cols[b.obs]).mean()
        assert abs(frac_real - frac_syn) < 0.08, (b.obs, frac_real, frac_syn)
        assert set(np.unique(cols[b.group_column])) <= {"VIRAC2->2MASS", "ref"}
