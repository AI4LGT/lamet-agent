from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

import lamet_agent.kernels.implementation as impl


FIXTURES = Path(__file__).with_name("fixtures")


@pytest.mark.parametrize("index", range(5))
def test_all_five_matrices_match_archive_mathematica(index):
    reference = np.asarray(json.loads((FIXTURES / "jinxin-dglap-matrices.json").read_text()))
    actual = impl._dglap_nnlo_isovector_matrices(np.arange(1, 6) / 5)
    np.testing.assert_allclose(actual[index], reference[index], rtol=5e-12, atol=3e-12)


@pytest.mark.parametrize("index", range(6))
def test_coefficients_match_archive_mathematica(index):
    ref = np.asarray(json.loads((FIXTURES / "jinxin-dglap-coefficients.json").read_text()))
    x = ref[:, 0]
    p, m = impl._p_nnlo_ns_plus(x, 3), impl._p_nnlo_ns_minus(x, 3)
    values = [impl._p_nlo_full_unpolarized(x, 3), impl._p_nlo_valence(x, 3), p, m, (p + m) / 2, (p - m) / 2]
    np.testing.assert_allclose(values[index], ref[:, index + 1], rtol=5e-12, atol=2e-11)


def test_coupled_blocks_have_plus_and_minus_eigenchannels(monkeypatch):
    x = np.arange(1, 6) / 5
    d0, q1, b1, q2, b2 = impl._dglap_nnlo_isovector_matrices(x)
    blocks = impl._dglap_nnlo_isovector_blocks(x)
    monkeypatch.setattr(impl, "alpha_s", lambda *args: 0.3)
    operator = impl._evolution_operator(1., 2., *blocks[:2], 20, evo_nnlo=blocks[2])
    a = .3 / (4 * np.pi)
    q, qb = x * (1 - x), .2 * x * (1 - x)**2
    plus = expm(np.log(4) * (a*d0 + a*a*(q1+b1) + a**3*(q2+b2))) @ (q + qb)
    minus = expm(np.log(4) * (a*d0 + a*a*(q1-b1) + a**3*(q2-b2))) @ (q - qb)
    expected = np.concatenate(((plus + minus) / 2, (plus - minus) / 2))
    np.testing.assert_allclose(operator @ np.concatenate((q, qb)), expected, atol=2e-14)
    assert np.linalg.norm(operator[5:, :5]) > 0


@pytest.mark.parametrize("scales", [(1., 2.), (2., 1.)])
def test_coupled_evolution_matches_simultaneous_ode(scales):
    blocks = impl._dglap_nnlo_isovector_blocks(np.arange(1, 6) / 5)
    initial = np.linspace(.1, 1., 10)

    def rhs(t, state):
        a = impl.alpha_s(3, np.exp(t / 2), "running") / (4 * np.pi)
        return (a*blocks[0] + a*a*blocks[1] + a**3*blocks[2]) @ state

    sol = solve_ivp(rhs, np.log(np.array(scales)**2), initial, method="DOP853", rtol=2e-12, atol=2e-13)
    assert sol.success
    actual = impl._evolution_operator(*scales, *blocks[:2], 160, evo_nnlo=blocks[2]) @ initial
    np.testing.assert_allclose(actual, sol.y[:, -1], rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("grid", [[0, .5, 1], [.2, .4, .6], [.1, .4, 1], [1], [1, .5], [np.nan, 1]])
def test_archive_matrix_grid_validation(grid):
    with pytest.raises(ValueError, match="grid"):
        impl._dglap_nnlo_isovector_matrices(np.asarray(grid))


@pytest.mark.parametrize("nf", [-1, 3.5, 7, np.inf])
def test_plus_flavour_validation(nf):
    with pytest.raises(ValueError, match="nf must"):
        impl._p_nnlo_ns_plus(.3, nf)


def test_plus_support():
    with np.errstate(all="raise"):
        np.testing.assert_array_equal(impl._p_nnlo_ns_plus(np.array([-1., 0., 1., 2.])), 0)
