"""Tests for physically motivated synthetic clusters.

WHAT THESE ARE CHECKED AGAINST
------------------------------
The Goodwin & Whitworth (2004) box-fractal construction has properties fixed by
its definition, and those are the oracles here:

* the survival probability is ``2**(D-3)``, so **D = 3 fills the cube uniformly**
  and smaller D leaves substructure. A uniform sphere is therefore an exact,
  parameter-free special case to test against.
* structure must be **monotone in D**, measured by a statistic that was not used
  to build it (Cartwright & Whitworth's Q).

Why this module exists at all: drawing positions from a smooth King or EFF profile
begs the question when validating a profile fit, because it assumes the smoothness
the fit assumes. See the module docstring.
"""

from __future__ import annotations

import numpy as np
import pytest

from erotica.analysis.synthetic import fractal_cluster, radial_profile_of


def _q_parameter(xy):
    """Cartwright & Whitworth (2004) Q = normalised MST edge length / normalised separation."""
    from scipy.sparse.csgraph import minimum_spanning_tree

    n = len(xy)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    edges = minimum_spanning_tree(d).toarray()
    edges = edges[edges > 0]
    r_cluster = np.linalg.norm(xy - xy.mean(axis=0), axis=1).max()
    m_bar = edges.mean() / (np.sqrt(n * np.pi * r_cluster**2) / n)
    s_bar = d[np.triu_indices(n, 1)].mean() / r_cluster
    return m_bar / s_bar


def test_uniform_sphere_is_the_exact_special_case():
    """Oracle: at D = 3 the survival probability is 2**0 = 1, so no sub-cube is
    ever pruned and the result must fill the sphere uniformly.

    Checked by the shell-volume law: for a uniform sphere the enclosed count
    grows as r**3, so the counts in equal-volume shells are equal.
    """
    pos = fractal_cluster(20000, fractal_dimension=3.0, rng=7)
    r = np.linalg.norm(pos, axis=1)
    assert r.max() <= 1.0 + 1e-9
    # Ten equal-WIDTH shells against the exact r**3 law, not equal-volume shells:
    # equal-volume bins put 4/5 of the sample in the outer half and are blind to a
    # central cusp, which is exactly the defect this has to catch.
    edges = np.linspace(0.0, 1.0, 11)
    counts, _ = np.histogram(r, bins=edges)
    expected = len(r) * (edges[1:] ** 3 - edges[:-1] ** 3)
    # Poisson, not a flat percentage: the innermost bin holds ~20 stars, where a
    # 20% relative tolerance is 0.9 sigma and would fail on shot noise alone.
    pull = (counts - expected) / np.sqrt(expected)
    assert np.all(np.abs(pull) < 4.0), np.round(pull, 2)
    assert (pull**2).mean() < 3.0, np.round(pull, 2)


def test_per_level_jitter_manufactures_a_central_cusp():
    """The Goodwin & Whitworth jitter is a knob with a known bias, so pin it.

    It displaces each block coherently with all its descendants; blocks that move
    inward pile up where they meet at the centre. This is why ``noise`` defaults
    to 0 -- the docstring quotes these numbers and this keeps them honest.
    """
    edges = np.linspace(0.0, 1.0, 11)

    def inner_excess(noise):
        r = np.linalg.norm(fractal_cluster(20000, fractal_dimension=3.0, rng=5, noise=noise), axis=1)
        counts, _ = np.histogram(r, bins=edges)
        expected = 20000 * (edges[1:] ** 3 - edges[:-1] ** 3)
        return (counts / expected)[:4].max()

    assert inner_excess(0.0) < 1.20
    assert inner_excess(0.10) > 1.25
    assert inner_excess(0.20) > 1.5


def test_structure_is_monotone_in_the_fractal_dimension():
    """Oracle: Q, a statistic not used in the construction, must rise with D.

    Lower D means more substructure, and Cartwright & Whitworth's Q is small for
    substructured distributions and large for centrally concentrated ones.
    """
    q = {
        d: float(np.mean([_q_parameter(fractal_cluster(300, fractal_dimension=d, rng=s)[:, :2])
                          for s in range(3)]))
        for d in (1.6, 2.0, 2.5, 3.0)
    }
    values = [q[d] for d in (1.6, 2.0, 2.5, 3.0)]
    assert values == sorted(values), q
    assert q[3.0] - q[1.6] > 0.25, q
    # D = 3 is the projection of a uniform BALL, whose surface density falls as
    # sqrt(1 - R**2) -- not a uniform disc. Cartwright & Whitworth (2004) put Q for
    # a uniform sphere at ~0.8, which is the external number to land on.
    assert 0.72 < q[3.0] < 0.88, q[3.0]


def test_returns_exactly_the_requested_number_without_duplicates():
    """Top-up must not sample with replacement -- coincident stars would read as
    infinite density to any density-based estimator."""
    pos = fractal_cluster(250, fractal_dimension=1.4, rng=3)
    assert pos.shape == (250, 3)
    assert len(np.unique(pos, axis=0)) == 250


def test_radius_scales_linearly():
    a = fractal_cluster(200, fractal_dimension=2.0, radius=1.0, rng=11)
    b = fractal_cluster(200, fractal_dimension=2.0, radius=5.0, rng=11)
    np.testing.assert_allclose(b, 5.0 * a)


def test_radial_profile_of_recovers_a_uniform_surface_density():
    """Oracle: a uniform sphere projects to a surface density that is NOT flat --
    it falls as sqrt(1 - (R/R0)^2). Checking against the right closed form, not
    against 'looks constant'."""
    pos = fractal_cluster(20000, fractal_dimension=3.0, rng=5)
    centres, density = radial_profile_of(pos, bins=10, project=True)
    r_max = np.linalg.norm(pos[:, :2], axis=1).max()
    expected = np.sqrt(np.clip(1.0 - (centres / r_max) ** 2, 0, None))
    expected = expected / expected[0] * density[0]
    # inner bins only; the outermost is edge-dominated
    np.testing.assert_allclose(density[:7], expected[:7], rtol=0.15)


@pytest.mark.parametrize("bad", [0.0, -1.0, 3.5])
def test_invalid_fractal_dimension_is_rejected(bad):
    with pytest.raises(ValueError, match="fractal_dimension"):
        fractal_cluster(100, fractal_dimension=bad)


def test_zero_stars_is_rejected():
    with pytest.raises(ValueError, match="n_stars"):
        fractal_cluster(0)
