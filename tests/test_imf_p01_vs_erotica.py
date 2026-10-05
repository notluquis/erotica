"""The IMF that fitted P01 (ASteCA 0.6.9) and EROTICA's ``_chabrier2014_xi`` are two different functions.

Hub node ``iso-imf-p01-vs-erotica`` (finding ``agent-findings/p01-thesis-harvest-models-2026-10-04.md``, 3.1).
Both are called "Chabrier (2014)" and neither is the other:

    ASteCA 0.6.9 ``chabrier_2014`` (Eq. 34): n_c = 11, m_c = 0.18 -> m_0 = n_c m_c = 1.98 Msun, x = 1.35,
                                             sigma^2 = log10(n_c) / (x ln 10)  (sigma = 0.579)
    EROTICA ``_chabrier2014_xi``:             m_c = 0.20, sigma = 0.55, m_0 = 1 Msun, slope -2.35

Any EROTICA-vs-ASteCA comparison on NGC 6383 (total mass, <m>, fraction above 2 Msun) carries this
difference as a confounder, so the test pins both functions and the size of the effect on the 254 members.

What each part fixes
--------------------
* The EROTICA constants and the restated ASteCA 0.6.9 constants, at five masses each (a change to either
  ``mc``, ``sigma`` or ``m0`` moves a pinned value).
* The restated ASteCA function against the installed ``asteca`` when it is importable. The hub measured the
  function identical in 0.6.9 and 0.7.0 (max relative difference 0 over 20 001 masses), so a 0.7.0 install
  is a valid oracle for the 0.6.9 constants. Skips when ``asteca`` is absent.
* The effect on the follow-up quantities, two ways:
    (a) the quantities each IMF *implies* on the range the 254 members span (0.186-13.56 Msun, system masses);
    (b) first-order re-weighting of the observed P01 masses by xi_EROTICA / xi_ASteCA. (b) is a prior swap
        applied to point masses, **not** a re-fit: the likelihood of each star's photometry dominates its mass in
        ASteCA, so (b) is an upper bound on the prior's pull and is labelled as such [inferred].

The measured numbers are the oracle; they were obtained with the script cited above, not copied from prose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erotica.analysis._isochrone import _chabrier2014_xi

MASSES = Path(__file__).resolve().parents[1] / "data/test/NGC6383/data/40/masses_asteca_069.csv"
GRID = np.array([0.2, 0.5, 1.0, 1.98, 5.0])
# Pinned values of the unnormalised xi(m) (hub measurement 2026-10-05, this file's first run).
EROTICA_XI = np.array(
    [5.0, 1.5394132109468501, 0.4459557050747009, 0.08956297830652733, 0.010155754969644935]
)
ASTECA_XI = np.array(
    [
        2.8189247358958047,
        0.8431345647262043,
        0.24716267654151625,
        0.05660637304123406,
        0.006418728644324134,
    ]
)


def xi_asteca_069(m):
    """ASteCA 0.6.9 ``get_imf("chabrier_2014", m)``, restated (asteca/modules/imfs.py:101-125, Eq. 34 of
    Chabrier+2014). Restated rather than imported so the pin does not depend on an install; the next test
    cross-checks it against the installed package when there is one."""
    m = np.asarray(m, float)
    nc, mc = 11, 0.18
    m0 = nc * mc
    ah, x = 0.649, 1.35
    al = ah * nc ** (x / 2)
    sigma2 = np.log10(nc) / (x * np.log(10))
    c = 0.434294 / m
    low = c * al * m0 ** (-x) * np.exp(-((np.log10(m) - np.log10(mc)) ** 2) / (2 * sigma2))
    return np.where(m <= m0, low, c * ah * m ** (-x))


def _normalised(f, lo, hi, n=40001):
    m = np.logspace(np.log10(lo), np.log10(hi), n)
    y = f(m)
    return m, y / np.trapezoid(y, m)


def _stats(f, lo, hi):
    m, y = _normalised(f, lo, hi)
    hi_mask = m >= 2.0
    return {
        "mean": float(np.trapezoid(y * m, m)),
        "f2": float(np.trapezoid(y[hi_mask], m[hi_mask])),
    }


@pytest.fixture(scope="module")
def system_masses():
    d = pd.read_csv(MASSES)
    # system mass: m1 plus m2 when ASteCA calls the star a binary (binar_prob > 0.7), as in P01 (tex:330)
    return (d.m1 + d.m2.fillna(0).where(d.binar_prob > 0.7, 0)).to_numpy()


def test_erotica_imf_constants_are_pinned():
    np.testing.assert_allclose(_chabrier2014_xi(GRID), EROTICA_XI, rtol=1e-6)
    # continuous at m_0 = 1 Msun (the log-normal and the power law are joined there)
    eps = 1e-9
    np.testing.assert_allclose(
        _chabrier2014_xi(np.array(1.0 - eps)), _chabrier2014_xi(np.array(1.0 + eps)), rtol=1e-6
    )


def test_p01_imf_constants_are_pinned():
    np.testing.assert_allclose(xi_asteca_069(GRID), ASTECA_XI, rtol=1e-6)
    m0 = 11 * 0.18
    assert m0 == pytest.approx(1.98)
    # ASteCA joins the pieces at m_0 = 1.98 with the same density up to its 0.434294 rounding of log10(e)
    np.testing.assert_allclose(
        xi_asteca_069(np.array(m0)), xi_asteca_069(np.array(m0 + 1e-9)), rtol=2e-3
    )


def test_restated_p01_imf_matches_the_installed_asteca():
    imfs = pytest.importorskip("asteca.modules.imfs", reason="asteca not installed")
    m = np.logspace(np.log10(0.08), 2, 2001)
    np.testing.assert_allclose(xi_asteca_069(m), imfs.get_imf("chabrier_2014", m), rtol=1e-12)


def test_the_two_imfs_are_not_the_same_function():
    """Normalised over 0.08-100 Msun (ASteCA's sampling range): EROTICA is 1.01-1.04x ASteCA between 0.2 and 1
    Msun and 0.90x between 2 and 5. A test that the difference is *there*, with its size."""
    m, ya = _normalised(xi_asteca_069, 0.08, 100.0)
    _, ye = _normalised(_chabrier2014_xi, 0.08, 100.0)
    ratio = np.interp([0.2, 1.0, 5.0], m, ye / ya)
    assert 1.01 <= ratio[0] <= 1.05 and 1.0 <= ratio[1] <= 1.05
    assert 0.88 <= ratio[2] <= 0.93
    sa, se = _stats(xi_asteca_069, 0.08, 100.0), _stats(_chabrier2014_xi, 0.08, 100.0)
    assert sa["mean"] == pytest.approx(0.7286, abs=5e-4) and se["mean"] == pytest.approx(
        0.6968, abs=5e-4
    )
    assert sa["f2"] == pytest.approx(0.0586, abs=2e-4) and se["f2"] == pytest.approx(
        0.0529, abs=2e-4
    )


def test_effect_on_what_the_followup_compares(system_masses):
    """(a) what each IMF implies on the 254 members' mass range; (b) the first-order re-weighting of P01's own masses."""
    assert len(system_masses) == 254
    mean_obs, f2_obs, total_obs = (
        system_masses.mean(),
        (system_masses > 2).mean(),
        system_masses.sum(),
    )
    assert mean_obs == pytest.approx(1.3052, abs=5e-4)  # P01 quotes 1.31 (tex:330)
    assert f2_obs == pytest.approx(0.1614, abs=5e-4)

    lo, hi = float(system_masses.min()), float(system_masses.max())
    assert (lo, hi) == (pytest.approx(0.186, abs=1e-3), pytest.approx(13.56, abs=1e-2))
    sa, se = _stats(xi_asteca_069, lo, hi), _stats(_chabrier2014_xi, lo, hi)
    assert sa["mean"] == pytest.approx(0.8487, abs=1e-3) and se["mean"] == pytest.approx(
        0.8097, abs=1e-3
    )
    assert sa["f2"] == pytest.approx(0.0810, abs=5e-4) and se["f2"] == pytest.approx(
        0.0723, abs=5e-4
    )
    # total mass N<m> for N = 254 under each IMF: the follow-up's M = N<m> moves by this
    assert 254 * sa["mean"] == pytest.approx(215.6, abs=0.3) and 254 * se["mean"] == pytest.approx(
        205.7, abs=0.3
    )
    assert se["mean"] / sa["mean"] == pytest.approx(
        0.954, abs=2e-3
    )  # EROTICA's IMF: 4.6 % lighter on average
    assert se["f2"] / sa["f2"] == pytest.approx(
        0.893, abs=3e-3
    )  # and 10.7 % fewer stars above 2 Msun

    # (b) re-weight the observed masses by xi_E / xi_A, each normalised on 0.08-100 Msun
    ma = np.logspace(np.log10(0.08), 2, 40001)
    na, ne = np.trapezoid(xi_asteca_069(ma), ma), np.trapezoid(_chabrier2014_xi(ma), ma)
    w = (_chabrier2014_xi(system_masses) / ne) / (xi_asteca_069(system_masses) / na)
    assert w.min() == pytest.approx(0.9036, abs=1e-3) and w.max() == pytest.approx(1.0429, abs=1e-3)
    ess = w.sum() ** 2 / (w**2).sum()
    assert ess > 250  # the weights are mild: the swap is a perturbation, not a different population
    mean_w = (w * system_masses).sum() / w.sum()
    f2_w = (w * (system_masses > 2)).sum() / w.sum()
    assert mean_w == pytest.approx(1.2483, abs=1e-3)
    assert 254 * mean_w == pytest.approx(317.1, abs=0.5)  # 331.5 -> 317.1 Msun
    assert f2_w == pytest.approx(0.1467, abs=5e-4)
    # direction and size, so a sign error in the ratio cannot pass
    assert 0.03 < 1 - mean_w / mean_obs < 0.06
    assert 0.07 < 1 - f2_w / f2_obs < 0.12
    assert total_obs == pytest.approx(331.51, abs=0.05)
