"""Fixed-order benchmarks and public H_F input/grid conventions."""

import numpy as np
import pytest

from lamet_agent.kernels.tmd import cg_nll, cg_nlo
from lamet_agent.kernels.tmd.coefficients import form_factor_kernel


def test_nlo_natural_scale_contains_both_finite_boundary_terms():
    assert cg_nlo.sudakov_correction(2.0) == pytest.approx(-0.197567977321501, abs=1e-14)
    assert form_factor_kernel(0.5, 0.5, momentum_gev=2.0) == pytest.approx(
        0.604864045356998, abs=2e-14
    )
    assert form_factor_kernel(0.5, 0.5, momentum_gev=2.0, order="nll") == 1.0


def test_nlo_truncation_is_linear_and_preserves_signed_endpoint_values():
    x = np.array([0.01, 0.3, 0.8])
    zero, first, second = [
        form_factor_kernel(x[:, None], x, momentum_gev=2.58, alpha_s=alpha)
        for alpha in (0.0, 0.1, 0.2)
    ]
    np.testing.assert_array_equal(zero, np.ones((3, 3)))
    np.testing.assert_allclose(second - 1, 2 * (first - 1), atol=2e-16, rtol=2e-15)
    assert second[0, 0] < 0


def test_one_loop_scale_derivative_matches_the_sudakov_rge():
    q, mu, alpha, step = np.array([0.7, 1.5, 3.0, 5.0]), 2.0, 0.293, 1e-5
    derivative = (
        cg_nlo.sudakov_correction(q, mu=mu * np.exp(step), alpha_s=alpha)
        - cg_nlo.sudakov_correction(q, mu=mu * np.exp(-step), alpha_s=alpha)
    ) / (2 * step)
    expected = -alpha * (4 / 3) / (4 * np.pi) * (4 * np.log(mu**2 / q**2) + 6)
    np.testing.assert_allclose(derivative, expected, atol=2e-11, rtol=2e-10)


def test_resummed_weak_coupling_recovers_nlo_logs_without_the_boundary_constant():
    q, alpha = np.array([0.7, 1.5, 3.0, 5.0]), 1e-6
    one_loop_logs = (
        cg_nlo.sudakov_correction(q, alpha_s=alpha)
        - cg_nlo.sudakov_correction(2.0, alpha_s=alpha)
    ) / alpha
    resummed = np.log(cg_nll.sudakov_factor(q, alpha_s=alpha)) / alpha
    np.testing.assert_allclose(resummed, one_loop_logs, rtol=4e-6, atol=1e-9)


@pytest.mark.parametrize("order", ["nlo", "nll", "nnll"])
def test_rectangular_grid_broadcasting_and_quark_antiquark_symmetry(order):
    x, y = np.array([0.2, 0.4, 0.8]), np.array([0.3, 0.7])
    kwargs = dict(momentum_gev=2.58, order=order)
    matrix = form_factor_kernel(x[:, None], y[None, :], **kwargs)
    assert matrix.shape == (3, 2)
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(matrix, form_factor_kernel(y[:, None], x, **kwargs).T, atol=2e-14)
    np.testing.assert_allclose(matrix, form_factor_kernel(1 - x[:, None], 1 - y, **kwargs), atol=2e-14)
    np.testing.assert_allclose(
        form_factor_kernel(x[:2], y, **kwargs), np.diag(matrix[:2]), atol=2e-14
    )


@pytest.mark.parametrize("order", ["nlo", "nll", "nnll"])
@pytest.mark.parametrize("invalid", [0.0, 1.0, -0.1, 1.1, np.nan, np.inf])
def test_invalid_fractions_are_rejected_on_either_leg(order, invalid):
    for x, y in ((invalid, 0.5), (0.5, invalid)):
        with pytest.raises(ValueError, match="0 < x1,x2 < 1"):
            form_factor_kernel(x, y, momentum_gev=2.58, order=order)


@pytest.mark.parametrize("kwargs", [
    {"momentum_gev": 0}, {"momentum_gev": -2}, {"mu": 0},
    {"alpha_s": -0.1}, {"alpha_s": np.nan}, {"order": "nnlo"},
    {"infrared_policy": "freeze", "freeze_scale_gev": 1.0},
])
def test_invalid_controls_and_nlo_infrared_options_are_rejected(kwargs):
    params = {"momentum_gev": 2.58, **kwargs}
    with pytest.raises(ValueError):
        form_factor_kernel(0.4, 0.6, **params)
