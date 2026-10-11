"""Tests del nucleo del clasificador. Corre: ~/miniforge3/envs/erotica/bin/python -m pytest -q test_core.py"""
import numpy as np

import core


def _toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(core.FEATURES)))
    y = (X[:, 0] + 0.5 * rng.normal(size=n)) > 0
    return X, y


def test_fit_sin_pesos_es_el_modelo_del_f2():
    X, y = _toy()
    p0 = core.fit(X, y, params=dict(core.XGB_PARAMS, n_estimators=20)).predict_proba(X)[:, 1]
    p1 = core.fit(X, y, params=dict(core.XGB_PARAMS, n_estimators=20), sample_weight=None).predict_proba(X)[:, 1]
    assert np.array_equal(p0, p1)


def test_pesos_no_triviales_cambian_el_modelo():
    """Leccion 29: el test pasa el parametro NO-default. Pesar a cero las etiquetas 1 con X0 < 1 debe bajar P en esa franja."""
    X, y = _toy()
    pr = dict(core.XGB_PARAMS, n_estimators=50)
    w = np.where(y & (X[:, 0] < 1.0), 0.0, 1.0)
    band = (X[:, 0] > 0) & (X[:, 0] < 1.0)
    p_u = core.fit(X, y, params=pr).predict_proba(X)[:, 1]
    p_w = core.fit(X, y, params=pr, sample_weight=w).predict_proba(X)[:, 1]
    assert p_w[band].mean() < p_u[band].mean() - 0.3
