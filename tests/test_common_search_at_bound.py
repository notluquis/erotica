"""``tools/validation/isochrone_colour/common.search`` flags every free parameter that ends on its bound
(hub finding R32-19).

``search`` polishes six parameters, (feh, log t, dm, A_V, sigma_int, f_bg), but its ``at_bound`` list only
looked at the first four. In the synthetic arms of the external-constraints experiment f_bg sat on its lower bound
(1e-4) in 6-7 of 8 replicates and the pre-registered "replicates at bound" metric read 0 by construction. The toy
below puts the maximum of the log posterior below the lower bound of sigma_int and of f_bg, so the polish stops
on both bounds; the four physical parameters have interior maxima."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/validation/isochrone_colour"))
common = pytest.importorskip("common")

# interior optimum for feh, loga, dm, A_V; below the 1e-4 floor for sigma_int and f_bg
TARGET = np.array([-0.2, 6.3, 10.2, 1.1, -0.05, -0.05])
SCALE = np.array([0.1, 0.05, 0.05, 0.1, 0.02, 0.02])


class _Toy:
    loga_range = (5.7, 7.0)
    dm_range = (9.5, 10.7)
    Av_range = (0.0, 3.0)
    dm_mu = 10.2
    dm_sigma = 10.0  # the search adds this Gaussian prior on dm; wide, so it does not move the toy's optimum
    _node_logz = np.array([-1.0, 0.5])
    _node_loga = np.round(np.arange(5.7, 7.0001, 0.05), 2)
    _N_obs = 1

    def _met_bounds(self):
        return (-1.0, 0.5)

    def _compiled_loglike(self, mode):
        def fl(y):
            z = (np.asarray(y, float) - TARGET) / SCALE
            return float(-0.5 * z @ z), -z / SCALE

        return fl


def test_search_flags_sigma_int_and_f_bg_on_their_bounds():
    res = common.search(_Toy(), verbose=False)
    m = res["mode"]
    assert m["sigma_int"] == pytest.approx(1e-4, abs=1e-8)
    assert m["f_bg"] == pytest.approx(1e-4, abs=1e-8)
    assert "sigma_int" in res["at_bound"] and "f_bg" in res["at_bound"]
    for idx, k in enumerate(("feh", "loga", "dm", "Av")):  # interior: not flagged
        assert k not in res["at_bound"]
        assert m[k] == pytest.approx(TARGET[idx], abs=1e-3)
