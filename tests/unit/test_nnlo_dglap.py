from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import lamet_agent.kernels.implementation as impl


@pytest.mark.parametrize("nf, expected", [
    (3, [26147.47962176508, 3341.986519583362, 1438.6494492769586, 613.7776354480475,
         561.9249432639435, 1044.2321890088558, 4311.583947849373, 58541.19620543744]),
    (4, [29294.011007273126, 3303.6680162135417, 1283.3900154048567, 436.7837936845772,
         354.7436750683496, 629.757029979005, 2792.6370853995168, 40233.514902441166]),
])
def test_nnlo_minus_matches_mathematica_notebook_reference(nf, expected):
    x = np.array([1e-6, .001, .01, .1, .3, .7, .9, .99])
    np.testing.assert_allclose(impl._p_nnlo_ns_minus(x, nf), expected, rtol=2e-12, atol=1e-10)


def test_nnlo_support_and_scalar_evaluation():
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(impl._p_nnlo_ns_minus(np.array([-2., 0., 1., 2.])), 0)
        assert np.isfinite(impl._p_nnlo_ns_minus(np.array([1e-12, 1 - 1e-12]))).all()
    np.testing.assert_allclose(impl._p_nnlo_ns_minus(0.3, 3), 561.9249432639435)


@pytest.mark.parametrize("nf", [-1, 3.5, 7, np.nan, np.inf])
def test_nnlo_rejects_invalid_flavour_count(nf):
    with pytest.raises(ValueError, match="nf must"):
        impl._p_nnlo_ns_minus(0.3, nf)


def test_nnlo_matrix_preserves_existing_orders_and_discrete_number():
    x = np.linspace(.1, .9, 9)
    lo, nlo = impl._dglap_evolution_matrices(x, impl._p_nlo_valence)
    matrices = impl._dglap_evolution_matrices(x, impl._p_nlo_valence, p_nnlo=impl._p_nnlo_ns_minus)
    np.testing.assert_array_equal(matrices[0], lo)
    np.testing.assert_array_equal(matrices[1], nlo)
    np.testing.assert_allclose(matrices[2].sum(axis=0), 0, atol=2e-12)
    np.testing.assert_array_equal(np.tril(matrices[2], -1), 0)
    assert np.linalg.norm(matrices[2]) > 0


@pytest.mark.parametrize("start,end", [(1., 3.), (3., 1.)])
def test_nnlo_evolution_matches_independent_ode(start, end):
    x = np.linspace(.1, .9, 5)
    lo, nlo, nnlo = impl._dglap_evolution_matrices(x, impl._p_nlo_valence, p_nnlo=impl._p_nnlo_ns_minus)

    def rhs(t, flat):
        a = impl.alpha_s(3, np.exp(t / 2), "running") / (4 * np.pi)
        return ((a * lo + a**2 * nlo + a**3 * nnlo) @ flat.reshape(5, 5)).ravel()

    sol = solve_ivp(rhs, np.log(np.array([start, end])**2), np.eye(5).ravel(),
                    method="DOP853", rtol=2e-12, atol=2e-13)
    assert sol.success
    actual = impl._evolution_operator(start, end, lo, nlo, 160, evo_nnlo=nnlo)
    np.testing.assert_allclose(actual, sol.y[:, -1].reshape(5, 5), atol=2e-5, rtol=2e-5)


def test_nnlo_cubic_normalization_and_coupling_order(monkeypatch):
    orders = []

    def coupling(order, mu, nf):
        orders.append(order)
        return 0.4

    monkeypatch.setattr(impl, "alpha_s", coupling)
    zero = np.zeros((2, 2))
    nnlo = np.array([[-20., 3.], [20., -3.]])
    actual = impl._evolution_operator(1., 2., zero, zero, 7, evo_nnlo=nnlo)
    expected = expm((0.4 / (4 * np.pi))**3 * nnlo * np.log(4))
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    assert orders == [3] * 7


def test_rgr_helper_can_consume_nnlo_without_changing_independent_matching(monkeypatch):
    monkeypatch.setattr(impl, "alpha_s", lambda *args: 0.3)
    x = np.array([.25, .5, .75])
    lo, nlo, nnlo = impl._dglap_evolution_matrices(x, impl._p_nlo_valence, p_nnlo=impl._p_nnlo_ns_minus)
    calls = []

    def fixed(grid, *, mu, **kwargs):
        calls.append(mu)
        return np.eye(3) * mu

    actual = impl._rgr_from_fixed_order(
        fixed, impl._p_nlo_valence, x, 2., 2., x, 1e-12, None,
        needs_zspz=False, takes_zspz=False, needs_zpsi=False, p_nnlo=impl._p_nnlo_ns_minus,
    )
    a = .3 / (4 * np.pi)
    generator = a * lo + a*a * nlo + a**3 * nnlo
    expected = np.stack([(expm(generator * np.log((2 / scale)**2)) * scale)[i]
                         for i, scale in enumerate(4 * x)])
    np.testing.assert_allclose(actual, expected, atol=2e-14)
    np.testing.assert_allclose(calls, 4 * x)
