"""Regression checks for nonuniform hybrid-NLO quark GPD matching.

Run from the repository root:
    python -m pytest tests/unit/test_quark_gpd_gi_gt_hybrid_nlo.py -q

These checks cover the finite-window numerical prescription and its interface.
They do not establish agreement with a separate Mathematica calculation that
uses different coupling, interpolation, cutoff, or integration-window choices.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from lamet_agent.kernels import quark_gpd_gi_gt_hybrid_nlo as gpd


PARAMS = dict(momentum_gev=2.0, scale_gev=2.0, zs_fm=0.3, skewness=0.3)
GRID = np.unique(np.r_[np.linspace(-2, 2, 25), np.linspace(-0.4, -0.2, 21), np.linspace(0.2, 0.4, 21)])


def test_coefficient_matches_original_away_from_singularities():
    # Keep the pre-refactor analytic expression as an independent algebra check.
    def original(x, y):
        xi, pz, mu = 0.3, 2.0, 2.0
        d = x - y
        a, b, c, e = xi + x, xi - x, xi + y, xi - y

        def piece(v):
            return abs(v) * (np.log(4 * pz**2 * v**2 / mu**2) - 1)

        terms = piece(a) / (c * (c + e)) + piece(b) / (e * (c + e)) - piece(d) / (c * e)
        terms += piece(a) / (c * (c - a)) + piece(b) / (e * (e - b))
        terms += (a / c + b / e) * (np.log(4 * pz**2 * d**2 / mu**2) - 1) / abs(d)
        terms += 6 * gpd._sine_integral(d * 0.3 * pz / gpd.GEV_FM) / (np.pi * d)
        return gpd._alphas_nloop(mu) * gpd.CF / (4 * np.pi) * terms

    for x, y in ((0.1, 0.7), (-0.6, 0.2), (0.35, 0.31), (0.9, -0.7)):
        actual = gpd._c_gpd_gt_hybrid(x, y, 0.3, 2, 2, 0.6 / gpd.GEV_FM, 1e-12)
        np.testing.assert_allclose(actual, original(x, y), rtol=1e-11, atol=1e-12)


def test_nonuniform_input_and_uniform_output():
    result = gpd.kernel(np.linspace(-0.8, 0.8, 5), GRID, **PARAMS)
    assert result.shape == (5, len(GRID))
    assert np.all(np.isfinite(result))


def test_output_rows_are_independent_of_other_requested_points():
    together = gpd.kernel(np.array([-0.4, 0.0, 0.4]), GRID, **PARAMS)
    alone = gpd.kernel(np.array([0.0]), GRID, **PARAMS)
    np.testing.assert_allclose(together[1], alone[0], rtol=0, atol=1e-12)


def test_adding_collinear_input_knots_does_not_change_matching():
    # Adding points on the same linear interpolant must not alter its convolution.
    coarse = np.linspace(-2, 2, 17)
    refined = np.unique(np.r_[coarse, (coarse[:-1] + coarse[1:]) / 2])
    f = np.exp(-(coarse**2))
    x = np.array([-0.4, 0.0, 0.4])
    first = gpd.kernel(x, coarse, **PARAMS) @ f
    second = gpd.kernel(x, refined, **PARAMS) @ np.interp(refined, coarse, f)
    np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-7)


def test_matrix_matches_direct_scalar_convolution():
    # Check matrix/basis assembly against an independent scalar quadrature path.
    x, cutoff = 0.4, 1e-4
    f = np.exp(-(GRID**2))
    fx = np.interp(x, GRID, f)

    def c(a, b):
        return gpd._c_gpd_gt_hybrid(a, b, 0.3, 2, 2, 0.6 / gpd.GEV_FM, 1e-12)

    def integrand(y):
        return c(x, y) * np.interp(y, GRID, f) - c(y, x) * fx

    correction = 0.0
    for lo, hi in ((GRID[0], x - cutoff), (x + cutoff, GRID[-1])):
        points = np.unique(np.r_[GRID, -0.3, 0.3])
        points = points[(points > lo) & (points < hi)]
        correction += quad(integrand, lo, hi, points=points, epsabs=1e-9, limit=1000)[0]
    actual = (gpd.kernel(np.array([x]), GRID, epsilon2=cutoff, **PARAMS) @ f)[0]
    np.testing.assert_allclose(actual, fx - correction, rtol=1e-7, atol=1e-8)


def test_cutoff_convergence_at_fixed_output():
    # A fixed smooth input and fixed outputs isolate the epsilon2 convergence.
    x = np.array([-0.32, -0.28, 0.28, 0.32])
    f = np.exp(-(GRID**2))
    values = [gpd.kernel(x, GRID, epsilon2=e, **PARAMS) @ f for e in (1e-4, 2e-5, 4e-6)]
    first = np.max(abs(values[1] - values[0]))
    second = np.max(abs(values[2] - values[1]))
    assert second < first
    assert second < 5e-4


@pytest.mark.parametrize(
    "grid, message",
    [
        (np.array([-1.0, 0.0, 0.0, 1.0]), "strictly increasing"),
        (np.array([1.0, 0.0, -1.0]), "strictly increasing"),
        (np.array([-1.0, np.nan, 1.0]), "finite"),
    ],
)
def test_invalid_input_grid(grid, message):
    with pytest.raises(ValueError, match=message):
        gpd.kernel(np.array([0.0]), grid, **PARAMS)


def test_hierarchy_and_exact_crossover_are_rejected():
    with pytest.raises(ValueError, match="hierarchy_ratio"):
        gpd.kernel(np.array([0.3001]), GRID, epsilon2=1e-5, **PARAMS)
    with pytest.raises(ValueError, match="one-sided"):
        gpd.kernel(np.array([0.3]), GRID, **PARAMS)


def test_input_grid_may_include_crossover_knots():
    grid = np.unique(np.r_[GRID, -0.3, 0.3])
    result = gpd.kernel(np.array([-0.4, 0.4]), grid, **PARAMS)
    assert np.all(np.isfinite(result))
