"""Tests for ``erotica.analysis.IsochroneFitterKW`` (the K x W model: an error-scale parameter ``k`` and
a magnitude window; hub finding ``isochrone-intervals-nuts.md`` §2).

Structure
---------
TestErrorScale        the parent's likelihood with ``k = 1``; ``k`` against the parent fed pre-scaled
                      errors (a different code path); the compiled gradient; the PyMC model's slot for ``k``
TestToyRecovery       the mode search on a closed-form toy family: stationarity, chain starts, ``k``
TestPosteriorCmd      the predictive draws carry each draw's ``k``
TestFrozenResearchOracle
                      numbers computed by the research implementation
                      (``tools/validation/isochrone_modelfix/aniso.py``, branch ``isochrone-model-fix``,
                      which has its own segment density, field term and mode search) on NGC 6383's
                      201 PMS-window stars and on a synthetic replicate, for the real MIST v1.2 grid.
                      Skipped where the MIST files are absent (CI).
TestAgainstTheNutsRun slow: the package's own NUTS on the real stars against the committed run.

What these tests do and do not transfer. The coverage of cell A (log t inside the 68 % / 90 % interval in
6/8 and 7/8) was measured with the research likelihood and NUTS; it is not re-run here, 8 NUTS fits are
hours. What carries it to this class is that **the likelihood is the same function** (frozen values to 1e-9
below, on two data sets) and the prior and starts are the same. A different sampler version or machine
changes the draws, not that statement.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from astropy.table import QTable

from erotica.analysis import IsochroneFitterKW
from erotica.analysis._isochrone import IsochroneFitter
from tests.test_isochrone import _toy_stars, _write_toy_family

requires_bayes = pytest.mark.skipif(
    any(importlib.util.find_spec(n) is None for n in ("pymc", "pytensor", "arviz")),
    reason="requires EROTICA's optional bayes extra",
)

HERE = Path(__file__).resolve().parent
NGC = Path.home() / "erotica" / "data" / "test" / "NGC6383"
MIST12 = NGC / "MIST" / "UBVRIplus"
requires_mist = pytest.mark.skipif(not MIST12.is_dir(), reason=f"MIST v1.2 files absent: {MIST12}")

PRIORS_NGC6383 = dict(
    loga_range=(5.7, 7.0), Av_range=(0.0, 3.0), dm_mu=10.223, dm_sigma=0.026, dm_range=(9.5, 10.7)
)


# ---------------------------------------------------------------------------------------------
# Toy family (closed form, legacy Z path): no local data needed
# ---------------------------------------------------------------------------------------------


def _toy(tmp_path, data, cls=IsochroneFitterKW, setup_kw=None, **kw):
    """The toy family of ``test_isochrone`` (singles) fitted with ``cls``; the family is written once
    per ``tmp_path``."""
    ages = (6.3, 6.4, 6.5, 6.6, 6.7, 6.8)
    if not any(tmp_path.glob("toy_*.iso.cmd")):
        _write_toy_family(tmp_path, (0.010, 0.015, 0.020), ages)
    args = dict(
        loga_range=(6.3, 6.8),
        Av_range=(0.2, 1.0),
        dm_mu=10.0,
        dm_sigma=0.3,
        dm_range=(9.6, 10.4),
        alpha=0.0,
        beta=0.0,
    )
    args.update(kw)
    f = cls(tmp_path, **args)
    f.setup(data, prob_threshold=0.0, **(setup_kw or {}))
    return f


_kw_toy = _toy


def _toy_with_errors(n, rng, k_true, loga=6.5, Z=0.015, dm=10.0, Av=0.6):
    """Star-level toy draw whose TRUE noise is ``k_true`` times the errors it reports; the reported
    error varies with magnitude so that ``k`` and ``sigma_int`` are separable."""
    t = _toy_stars(n, loga, Z, dm, Av, rng, e=1e-6)
    G = np.asarray(t["Gmag"], float)
    e = 0.004 + 0.0008 * (G - G.min()) ** 2
    for col in ("Gmag", "G_BPmag", "G_RPmag"):
        t[col] = np.asarray(t[col], float) + rng.normal(0, 1, n) * k_true * e
    for col in ("e_Gmag", "e_G_BPmag", "e_G_RPmag"):
        t[col] = e
    return t


def _theta(f):
    return dict(met=0.0152, loga=6.52, dm=10.05, Av=0.62, sigma_int=0.02, f_bg=0.03)


class TestErrorScale:
    def test_k_equal_one_is_the_parent_likelihood_bit_for_bit(self, tmp_path):
        """Oracle: the parent class on the same data. Same arguments, ``k = 1.0`` (an exact product)."""
        data = _toy_with_errors(120, np.random.default_rng(1), 1.0)
        f = _kw_toy(tmp_path, data)
        parent = _toy(tmp_path, data, cls=IsochroneFitter)
        th = _theta(f)
        a = f._star_loglike(**th, xp=np, k=1.0)
        b = IsochroneFitter._star_loglike(parent, **th, xp=np)
        assert np.array_equal(a, b)

    def test_k_is_the_parent_fed_errors_multiplied_by_k(self, tmp_path):
        """Oracle: a second path to the same number. The parent's likelihood on a table whose reported
        errors are ``k`` times the originals (so the scale never goes through ``k``). The two axes get
        different errors, so scaling only one of them is a different number.
        Mutation: ``error_scale`` applied to the magnitude error only -> red."""
        data = _toy_with_errors(150, np.random.default_rng(2), 2.0)
        f = _kw_toy(tmp_path, data)
        k = 2.4
        scaled = QTable(data, copy=True)
        for c in ("e_Gmag", "e_G_BPmag", "e_G_RPmag"):
            scaled[c] = k * np.asarray(data[c], float)
        parent = _toy(tmp_path, scaled, cls=IsochroneFitter)
        # F(theta) takes the error at the segment midpoint from the polynomial fitted to the
        # catalogue's UNSCALED errors (``_e_mag_coef``) in K x W, as measured: give the parent the same
        parent._e_mag_coef = f._e_mag_coef
        th = _theta(f)
        a = f._star_loglike(**th, xp=np, k=k)
        b = IsochroneFitter._star_loglike(parent, **th, xp=np)
        # the parent builds the colour error as hypot(e_BP, e_RP): same on both sides
        assert f._box_area == pytest.approx(parent._box_area)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=0)
        assert not np.allclose(a, f._star_loglike(**th, xp=np, k=1.0), rtol=1e-3)

    def test_loglike_method_carries_k(self, tmp_path):
        data = _toy_with_errors(100, np.random.default_rng(3), 1.0)
        f = _kw_toy(tmp_path, data)
        th = _theta(f)
        assert f.loglike(**th, k=2.0) == pytest.approx(
            float(np.sum(f._star_weights * f._star_loglike(**th, xp=np, k=2.0))), rel=1e-14
        )
        assert f.loglike(**th, k=2.0) != pytest.approx(f.loglike(**th, k=1.0), rel=1e-6)

    @requires_bayes
    def test_compiled_gradient_matches_finite_differences(self, tmp_path):
        """Oracle: central differences of the NumPy likelihood, in all seven parameters. Mutation:
        a compiled function whose ``k`` slot is another parameter's -> red on ``k`` or on that one."""
        data = _toy_with_errors(120, np.random.default_rng(4), 2.0)
        f = _kw_toy(tmp_path, data)
        fn = f._compiled_loglike()
        x = np.array([0.0152, 6.52, 10.05, 0.62, 0.02, 2.2, 0.03])
        v, g = fn(x)
        names = ("met", "loga", "dm", "Av", "sigma_int", "k", "f_bg")
        assert v == pytest.approx(f.loglike(**dict(zip(names, x, strict=True))), rel=1e-12)
        for i, h in enumerate((2e-5, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-4)):
            xp_, xm_ = x.copy(), x.copy()
            xp_[i] += h
            xm_[i] -= h
            fd = (
                f.loglike(**dict(zip(names, xp_, strict=True)))
                - f.loglike(**dict(zip(names, xm_, strict=True)))
            ) / (2 * h)
            assert g[i] == pytest.approx(fd, rel=2e-4, abs=2e-3), names[i]

    @requires_bayes
    def test_model_puts_k_in_its_slot_and_bounds_it(self, tmp_path):
        """Oracle: the model's observed-variable log-probability at a chosen point equals
        ``loglike`` there (``k`` and ``f_bg`` swapped would not), and the prior of ``k`` is
        U(K_RANGE). Mutation: swap ``k`` and ``f_bg`` in the ``CustomDist`` arguments -> red;
        ``K_RANGE`` (0, 6) -> red on the support check."""
        import pymc as pm

        data = _toy_with_errors(100, np.random.default_rng(5), 2.0)
        f = _kw_toy(tmp_path, data)
        model = f.build_model()
        point = {
            "met": 0.0152,
            "loga": 6.52,
            "dm": 10.05,
            "Av": 0.62,
            "sigma_int": 0.02,
            "k": 2.2,
            "f_bg": 0.03,
        }
        for name, val in point.items():
            model.set_initval(model[name], val)
        y_logp = model.compile_logp(vars=[model.observed_RVs[0]], jacobian=False)
        assert float(y_logp(model.initial_point())) == pytest.approx(f.loglike(**point), rel=1e-9)
        draws = pm.draw(model["k"], draws=4000, random_seed=1)
        assert draws.min() >= 1.0 and draws.max() <= 6.0 and draws.max() > 5.5 and draws.min() < 1.5


class TestWeightsPriorsAndStarts:
    @requires_bayes
    def test_weights_enter_the_model_and_the_compiled_function(self, tmp_path):
        """Oracle: with ``ms_weight = 0.5`` on half the stars, the compiled value and the model's
        observed log-probability equal ``loglike`` (which applies the weights in NumPy) and differ from
        the unweighted one. Mutation: dropping ``_star_weights`` from ``build_model`` or from
        ``_compiled_loglike`` -> red (no other fixture has weights other than 1)."""
        rng = np.random.default_rng(11)
        data = _toy_with_errors(120, rng, 2.0)
        data["pms_sagitta"] = np.where(np.arange(120) % 2 == 0, 0.9, 0.1)
        f = _toy(tmp_path, data, setup_kw=dict(pms_column="pms_sagitta", ms_weight=0.5))
        assert np.sum(f._star_weights == 0.5) == 60
        point = {
            "met": 0.0152,
            "loga": 6.52,
            "dm": 10.05,
            "Av": 0.62,
            "sigma_int": 0.02,
            "k": 2.2,
            "f_bg": 0.03,
        }
        want = f.loglike(**point)
        unweighted = float(np.sum(f._star_loglike(**point, xp=np)))
        assert abs(want - unweighted) > 1.0
        got = f._compiled_loglike()(np.array(list(point.values())))[0]
        assert got == pytest.approx(want, rel=1e-12)
        model = f.build_model()
        for name, val in point.items():
            model.set_initval(model[name], val)
        y_logp = model.compile_logp(vars=[model.observed_RVs[0]], jacobian=False)
        assert float(y_logp(model.initial_point())) == pytest.approx(want, rel=1e-9)

    @requires_bayes
    def test_priors_of_the_nuisance_parameters(self, tmp_path):
        """Oracle: closed-form moments of the declared priors, from 40 000 prior draws: sigma_int
        HalfNormal(0.05) has median 0.05 * 0.67449 and f_bg Beta(1, 19) has mean 0.05 and median
        1 - 0.5**(1/19). Mutation: HalfNormal(0.5) or Beta(1, 1) -> red."""
        import pymc as pm

        f = _toy(tmp_path, _toy_with_errors(60, np.random.default_rng(12), 1.0))
        model = f.build_model()
        s = pm.draw(model["sigma_int"], draws=40000, random_seed=2)
        b = pm.draw(model["f_bg"], draws=40000, random_seed=3)
        assert np.median(s) == pytest.approx(0.05 * 0.67449, rel=0.03)
        assert b.mean() == pytest.approx(0.05, rel=0.03)
        assert np.median(b) == pytest.approx(1 - 0.5 ** (1 / 19), rel=0.04)

    @requires_bayes
    def test_chain_starts_follow_the_laplace_ridge(self, tmp_path):
        """Oracle: the Laplace covariance at the returned mode computed here by central differences of
        the NumPy likelihood (a different path from the compiled gradient the search uses), and the ridge
        vector ``S[:4, 1] / S[1, 1]`` from it; each start's (met, dm, A_V) is the mode plus
        ``START_DLOGA * v`` (inside the clip). Mutation: ridge vector replaced by [0, 1, 0, 0] -> red."""
        data = _toy_with_errors(400, np.random.default_rng(8), 2.5, loga=6.55)
        f = _toy(tmp_path, data, dm_mu=10.0, dm_sigma=0.1, dm_range=(9.6, 10.4))
        st = f.find_start(4, np.random.default_rng(0))
        names = ("met", "loga", "dm", "Av", "sigma_int", "k", "f_bg")
        x = np.array([st["mode"][n] for n in names])

        def post(y):
            return f.loglike(*y[:5], f_bg=y[6], k=y[5]) - 0.5 * ((y[2] - f.dm_mu) / f.dm_sigma) ** 2

        h = np.array([2e-5, 1e-3, 1e-3, 1e-3, 1e-3, 1e-2, 1e-3])
        H = np.empty((7, 7))
        for i in range(7):
            for j in range(7):
                ei, ej = np.eye(7)[i] * h[i], np.eye(7)[j] * h[j]
                H[i, j] = (
                    post(x + ei + ej) - post(x + ei - ej) - post(x - ei + ej) + post(x - ei - ej)
                ) / (4 * h[i] * h[j])
        S = np.linalg.inv(-H)
        v = S[:4, 1] / S[1, 1]
        np.testing.assert_allclose(
            np.asarray(st["ridge_vector"])[[0, 1, 2]], v[[0, 1, 2]], rtol=0.15, atol=0.02
        )
        assert np.sign(st["ridge_vector"][3]) == np.sign(
            v[3]
        )  # A_V is weakly determined: sign only
        assert abs(v[2]) > 0.3  # a ridge that is not the loga axis itself
        for d, s in zip((-0.17, -0.06, 0.06, 0.17), st["chain_starts"], strict=True):
            for i, n in ((2, "dm"),):
                lo_, hi_ = {0: f._met_bounds(), 2: f.dm_range, 3: f.Av_range}[i]
                want = np.clip(x[i] + d * v[i], lo_ + 1e-3 * (hi_ - lo_), hi_ - 1e-3 * (hi_ - lo_))
                assert s[n] == pytest.approx(want, rel=0.2, abs=2e-3 * (hi_ - lo_)), (d, n)


class TestToyRecovery:
    @requires_bayes
    def test_find_start_is_a_stationary_point_of_the_log_posterior(self, tmp_path):
        """Oracle: the objective the search maximises, ``log L + log N(dm)`` (the compiled likelihood and
        the closed-form prior), does not increase when any free parameter is moved 0.1 % of its range
        either way from the returned mode. Mutation: a search that drops the ``dm`` prior from its
        objective stops where the prior still pulls (dm_mu 10.2, sd 0.05, mode dm ~ 10.0:
        gain ~ 0.16 per step) -> red."""
        rng = np.random.default_rng(7)
        data = _toy_with_errors(300, rng, 2.5)
        f = _kw_toy(tmp_path, data, dm_mu=10.2, dm_sigma=0.05, dm_range=(9.6, 10.4))
        st = f.find_start(4, np.random.default_rng(0))
        names = ("met", "loga", "dm", "Av", "sigma_int", "k", "f_bg")
        x = np.array([st["mode"][n] for n in names])
        fl = f._compiled_loglike()

        def objective(y):
            return fl(y)[0] - 0.5 * ((y[2] - f.dm_mu) / f.dm_sigma) ** 2

        lo = np.array([0.01, *f.loga_range[:1], *f.dm_range[:1], *f.Av_range[:1], 1e-4, 1.0, 1e-4])
        hi = np.array([0.02, *f.loga_range[1:], *f.dm_range[1:], *f.Av_range[1:], 2.0, 6.0, 0.5])
        base = objective(x)
        # the toy has sigma_int = 0 and no field: those two sit on their lower bounds by construction;
        # nothing else may be excused from the check
        assert set(st["at_bound"]) <= {"sigma_int", "f_bg"}, st["at_bound"]
        for i, n in enumerate(names):
            for sign in (-1.0, 1.0):
                y = x.copy()
                y[i] = np.clip(y[i] + sign * 1e-3 * (hi[i] - lo[i]), lo[i], hi[i])
                assert objective(y) <= base + 1e-6, (n, sign, objective(y) - base)
        assert st["logpost"] == pytest.approx(
            st["loglike"] - 0.5 * ((x[2] - f.dm_mu) / f.dm_sigma) ** 2, rel=1e-12
        )

    @requires_bayes
    def test_find_start_recovers_the_error_scale_and_the_chain_starts_lie_along_the_ridge(
        self, tmp_path
    ):
        """Oracle: the toy was drawn with noise = 2.5 x its reported errors and ``sigma_int`` = 0. The
        mode's ``k`` is within 0.5 of 2.5. The four chain starts sit at the measured offsets (-0.17, -0.06, +0.06, +0.17; written out here, not read from the class)
        in log t (the clip is far away), in the order given; and every start has all seven names.
        Mutation: ``START_DLOGA`` sign flipped -> red."""
        rng = np.random.default_rng(8)
        data = _toy_with_errors(400, rng, 2.5, loga=6.55)
        f = _kw_toy(tmp_path, data, dm_mu=10.0, dm_sigma=0.1, dm_range=(9.6, 10.4))
        st = f.find_start(4, np.random.default_rng(0))
        assert abs(st["mode"]["k"] - 2.5) < 0.5, st["mode"]
        assert st["mode"]["sigma_int"] < 0.03
        lo, hi = f.loga_range
        want = np.clip(
            st["mode"]["loga"] + np.array([-0.17, -0.06, 0.06, 0.17]),
            lo + 1e-3 * (hi - lo),
            hi - 1e-3 * (hi - lo),
        )
        np.testing.assert_allclose([s["loga"] for s in st["chain_starts"]], want, atol=1e-9)
        for s in st["chain_starts"]:
            assert set(s) == {"met", "loga", "dm", "Av", "sigma_int", "k", "f_bg"}
            assert 1.0 < s["k"] < 6.0


class TestPosteriorCmd:
    def test_draws_scale_with_k_in_closed_form(self, tmp_path):
        """Oracle: with a constant error model and no completeness cut, a star's noise is
        ``z * sqrt((k c)^2 + s2)`` for the same normal ``z`` whatever ``k`` is (same seed), so
        ``(Go_2 - Go_1) / (Go_3 - Go_1)`` is the constant ``(e_2 - e_1) / (e_3 - e_1)``. Mutation:
        ``error_scale`` not applied in ``_draw_stars`` -> the ratio is 0/0."""
        data = _toy_with_errors(100, np.random.default_rng(9), 1.0)
        f = _kw_toy(tmp_path, data)
        c = 0.05
        f._e_mag_fn = lambda m: np.full(np.shape(m), c)
        f._e_col_fn = lambda m: np.full(np.shape(m), c)
        f._mag_lim, f._mag_bright = np.inf, None
        s2 = f.SIGMA_FLOOR**2 + 0.02**2
        out = {}
        for k in (1.0, 2.0, 3.0):
            out[k] = f._draw_stars(
                0.0152, 6.5, 10.0, 0.6, 0.02, 60, np.random.default_rng(3), error_scale=k
            )
        e = {k: np.sqrt((k * c) ** 2 + s2) for k in out}
        for axis in (0, 1):
            num = out[2.0][axis] - out[1.0][axis]
            den = out[3.0][axis] - out[1.0][axis]
            np.testing.assert_allclose(num / den, (e[2.0] - e[1.0]) / (e[3.0] - e[1.0]), rtol=1e-9)
        # absolute amplitude: Go_3 - Go_1 = z (e_3 - e_1) with z ~ N(0, 1), so its sd is e_3 - e_1
        big = {
            k: f._draw_stars(
                0.0152, 6.5, 10.0, 0.6, 0.02, 3000, np.random.default_rng(5), error_scale=k
            )
            for k in (1.0, 3.0)
        }
        for axis in (0, 1):
            assert np.std(big[3.0][axis] - big[1.0][axis]) == pytest.approx(
                e[3.0] - e[1.0], rel=0.05
            )

    @requires_bayes
    def test_posterior_cmd_uses_each_draws_k(self, tmp_path):
        """Oracle: for a one-draw posterior, ``posterior_cmd`` is ``_draw_stars`` with that draw's ``k``
        (same seed). Mutation: ``posterior_cmd`` not passing ``k`` -> red."""
        import arviz as az

        data = _toy_with_errors(80, np.random.default_rng(10), 1.0)
        f = _kw_toy(tmp_path, data)
        post = {
            n: np.array([[v]])
            for n, v in (
                ("met", 0.0152),
                ("loga", 6.5),
                ("dm", 10.0),
                ("Av", 0.6),
                ("sigma_int", 0.02),
                ("k", 2.7),
            )
        }
        idata = az.from_dict({"posterior": post})
        _, _, cmds = f.posterior_cmd(idata, num_samples=1, rng=np.random.default_rng(4))
        ref = f._draw_stars(
            0.0152, 6.5, 10.0, 0.6, 0.02, f._N_obs, np.random.default_rng(4), error_scale=2.7
        )
        np.testing.assert_array_equal(cmds[0][0], ref[0])
        np.testing.assert_array_equal(cmds[0][1], ref[1])
        wrong = f._draw_stars(0.0152, 6.5, 10.0, 0.6, 0.02, f._N_obs, np.random.default_rng(4))
        assert not np.array_equal(cmds[0][0], wrong[0])


# ---------------------------------------------------------------------------------------------
# The real MIST v1.2 grid: values frozen from the research implementation
# ---------------------------------------------------------------------------------------------

#: ``tools/validation/isochrone_modelfix/fits/K_W.json`` (aniso.search8, NGC 6383, 201 stars in G 14-19)
_X_REAL = [
    -0.45178684650223266,
    5.971281287453579,
    10.22220877343697,
    0.4657360240327589,
    0.09433738800570059,
    2.4456073299081154,
    0.0668623597974588,
]
_LOGL_REAL = -278.58511134967847
_LOGPOST_REAL = -278.5855743966267
#: kw_runs/A_0/prep.json (cell A, replicate 0): log L at the injected truth and at the control's mode
_X_A0_TRUTH = [
    -0.30525315859357866,
    5.965749069117828,
    10.174690870818564,
    0.4657360240327589,
    0.09433738800570059,
    2.4456073299081154,
    0.0668623597974588,
]
_LOGL_A0_TRUTH = -279.4775311978356
_X_A0_MODE = [
    -0.05849639208773076,
    6.191052927418514,
    10.20200339089059,
    0.4752942315693778,
    0.08959059535478102,
    1.8579410095182398,
    0.06742836343771298,
]
_LOGL_A0_MODE = -287.5913348243316
_LOGPOST_A0_MODE = -287.9174129264717


def _mist_grid():
    from erotica.analysis.grids import MISTGrid

    edr3 = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
    return MISTGrid(MIST12, bands=edr3, loga_range=(5.6, 7.1)).select(
        feh=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5]
    )


def _kw_fitter(table_path):
    from astropy.table import Table

    f = IsochroneFitterKW(
        grid=_mist_grid(),
        magnitude="Gaia_G_EDR3",
        color=("Gaia_BP_EDR3", "Gaia_RP_EDR3"),
        # the frozen research values were computed with the constant-shift CCM89 law, not the
        # Gaia EDR3 colour-dependent default that #37 introduced
        extinction_law="ccm89",
        **PRIORS_NGC6383,
    )
    f.setup(QTable(Table.read(table_path)), prob_threshold=0.0, mag_window=(14.0, 19.0))
    return f


@requires_mist
class TestFrozenResearchOracle:
    @pytest.fixture(scope="class")
    @classmethod
    def real(cls):
        return _kw_fitter(HERE / "data" / "ngc6383_members254.ecsv")

    @pytest.fixture(scope="class")
    @classmethod
    def a0(cls):
        return _kw_fitter(HERE / "data" / "kw_cell_A0.ecsv")

    def test_window_keeps_the_201_stars(self, real):
        assert real._N_obs == 201

    def test_loglike_on_ngc6383_matches_the_research_value(self, real):
        """Oracle: ``aniso.WidthFitter.compiled8('K')`` at the K x W mode, -278.58511134967847 (its
        own segment density with a G-colour covariance, its own field term, its own k). Mutation:
        ``k`` not scaling the colour error moves it by 10s of units -> red."""
        ll = real.loglike(*[_X_REAL[i] for i in (0, 1, 2, 3, 4)], f_bg=_X_REAL[6], k=_X_REAL[5])
        assert ll == pytest.approx(_LOGL_REAL, abs=1e-8)

    def test_loglike_on_a_synthetic_replicate_matches_the_research_values(self, a0):
        """Oracle: the same research function on cell A replicate 0, at the injected truth and at the
        control's mode (two points 8 units of log L apart; a second data set, a second regime)."""
        for x, want in ((_X_A0_TRUTH, _LOGL_A0_TRUTH), (_X_A0_MODE, _LOGL_A0_MODE)):
            got = a0.loglike(x[0], x[1], x[2], x[3], x[4], x[6], x[5])
            assert got == pytest.approx(want, abs=1e-8)

    @pytest.mark.slow
    def test_find_start_reproduces_the_research_mode_on_ngc6383(self, real):
        """Oracle: ``aniso.search8`` on the same stars (the same lattice, polish and objective; the
        port is the same algorithm). Mode to 1e-3 in every parameter, log posterior to 1e-3.
        ~10 min on a loaded machine."""
        st = real.find_start(4, np.random.default_rng(0))
        names = (real._met_name, "loga", "dm", "Av", "sigma_int", "k", "f_bg")
        got = np.array([st["mode"][n] for n in names])
        np.testing.assert_allclose(got, _X_REAL, atol=1e-3)
        assert st["logpost"] == pytest.approx(_LOGPOST_REAL, abs=1e-3)
        # the measured Laplace sd of log t at that mode (starts_K_W.json): 0.0577
        assert st["local_sd"]["loga"] == pytest.approx(0.0577, abs=2e-3)

    @pytest.mark.slow
    def test_find_start_reproduces_the_known_failure_on_cell_a0(self, a0):
        """Oracle: the research control stays on the OLD stretch of the age ridge in replicate A_0
        (log t 6.191 against the truth 5.966; hub §2.9). The port is the same algorithm, so it
        reproduces the failure: a test that the mode search is what was measured, not a better one.
        A search that "fixed" this would change the measured procedure silently."""
        st = a0.find_start(4, np.random.default_rng(0))
        names = (a0._met_name, "loga", "dm", "Av", "sigma_int", "k", "f_bg")
        got = np.array([st["mode"][n] for n in names])
        np.testing.assert_allclose(got, _X_A0_MODE, atol=2e-3)
        assert st["logpost"] == pytest.approx(_LOGPOST_A0_MODE, abs=2e-3)


# Reference run (committed in the hub finding §2.11 and kw_runs/real_0/nuts.json): 4 chains x 1000 after
# 1000 warmup, seed 7, target_accept 0.9; log t median 5.965288, 16-84 % [5.933019, 5.999678],
# 5-95 % [5.912724, 6.025393], sd 0.034778, ESS bulk 2247, R-hat 1.003, 0 divergences.
_REF = dict(median=5.965288, q16=5.933019, q84=5.999678, q05=5.912724, q95=6.025393, sd=0.034778)


@requires_mist
@requires_bayes
@pytest.mark.slow
class TestAgainstTheNutsRun:
    def test_nuts_on_ngc6383_reproduces_the_committed_log_t(self):
        """Oracle: the committed K x W run (research code, hub §2.11). The tolerance is the Monte Carlo
        error of the DIFFERENCE of two independent runs, not a fraction of the posterior sd: the median's
        standard error is ``1.2533 sd / sqrt(ESS)`` ~ 0.0009 for the reference (ESS 2247) and
        ~0.0014 for a run with ESS 1000; their difference ~0.0017, so 0.005 is ~3 sigma. The quartile
        quantiles move more, 0.007. The package's own gate must pass too (R-hat <= 1.01, ESS > 400, 0
        divergences). Takes ~1-2 h on 4 cores at load 30-40."""
        import arviz as az

        f = _kw_fitter(HERE / "data" / "ngc6383_members254.ecsv")
        idata = f.fit(
            draws=1000,
            tune=1000,
            chains=4,
            cores=4,
            target_accept=0.9,
            random_seed=7,
            progressbar=False,
        )
        loga = idata.posterior["loga"].values
        assert float(az.rhat(loga)) <= 1.01
        assert float(az.ess(loga)) > 400
        assert int(idata.sample_stats["diverging"].sum()) == 0
        flat = loga.ravel()
        q = np.percentile(flat, [5, 16, 50, 84, 95])
        assert q[2] == pytest.approx(_REF["median"], abs=0.005)
        assert q[1] == pytest.approx(_REF["q16"], abs=0.007)
        assert q[3] == pytest.approx(_REF["q84"], abs=0.007)
        assert q[0] == pytest.approx(_REF["q05"], abs=0.009)
        assert q[4] == pytest.approx(_REF["q95"], abs=0.009)
