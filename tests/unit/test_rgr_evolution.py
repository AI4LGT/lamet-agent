from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import lamet_agent.kernels.implementation as impl
from lamet_agent.kernels import list_kernel_ids, load_kernel


def ode_operator(mu_initial, mu_final, lo, nlo):
    size = lo.shape[0]

    def rhs(t, flat):
        a = impl.alpha_s(impl.ALPHAS_N_LOOP, np.exp(t / 2), "running") / (4 * np.pi)
        return ((a * lo + a**2 * nlo) @ flat.reshape(size, size)).ravel()

    result = solve_ivp(
        rhs, np.log(np.array([mu_initial, mu_final]) ** 2), np.eye(size).ravel(),
        method="DOP853", rtol=2e-12, atol=2e-13,
    )
    assert result.success
    return result.y[:, -1].reshape(size, size)


@pytest.mark.parametrize("scales", [(0.8, 3.0), (3.0, 0.8)])
def test_evolution_matches_column_vector_ode(scales):
    grid = np.linspace(0.1, 0.9, 9)
    lo, nlo = impl._dglap_evolution_matrices(grid, impl._p_nlo_valence)
    assert np.linalg.norm(lo @ nlo - nlo @ lo) > 1
    expected = ode_operator(*scales, lo, nlo)
    coarse = impl._evolution_operator(*scales, lo, nlo, 40)
    fine = impl._evolution_operator(*scales, lo, nlo, 160)
    assert np.max(np.abs(fine - expected)) < np.max(np.abs(coarse - expected)) / 10
    np.testing.assert_allclose(fine, expected, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("needs_zpsi", [False, True])
def test_rgr_matches_independent_scale_ode_solutions(needs_zpsi):
    x = np.linspace(0.2, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 7)
    pz, target = 2.0, 2.0
    scales = 2 * pz * x
    lo, nlo = impl._dglap_evolution_matrices(x, impl._p_nlo_valence)

    def fixed(grid, *, mu, quasi_y_ls, **kwargs):
        return np.exp(-((grid[:, None] - quasi_y_ls[None, :]) ** 2)) * (1 + np.log(mu) * grid[:, None])

    expected = np.stack([
        (ode_operator(scale, target, lo, nlo) @ fixed(x, mu=scale, quasi_y_ls=y))[i]
        * (impl._zpsi_msbar_ratio(scale) if needs_zpsi else 1)
        for i, scale in enumerate(scales)
    ])
    actual = impl._rgr_from_fixed_order(
        fixed, impl._p_nlo_valence, x, pz, target, y, 1e-12, None,
        needs_zspz=False, takes_zspz=False, needs_zpsi=needs_zpsi, steps=160,
    )
    np.testing.assert_allclose(actual, expected, atol=5e-5, rtol=5e-5)


@pytest.mark.parametrize("kernel_id", [name for name in list_kernel_ids() if "_rgr_" in name])
def test_public_rgr_kernels_keep_cutoff_and_rectangular_shape(kernel_id):
    x = np.linspace(-0.75, 0.75, 4)
    y = np.linspace(-0.9, 0.9, 6)
    kwargs = {"momentum_gev": 2.0, "scale_gev": 2.0}
    if "hybrid" in kernel_id:
        kwargs["zs_fm"] = 0.2
    matrix = load_kernel(kernel_id)(x, y, **kwargs)
    assert matrix.shape == (4, 6)
    assert np.isfinite(matrix).all()
    np.testing.assert_array_equal(matrix[x < 0.15], 0)
    assert np.any(matrix[x > 0.15] != 0)


@pytest.mark.parametrize("grid", [[], [np.nan], [[0.1, 0.2]], [0.2, 0.1], [0.1, 0.1], [0.1, 0.2, 0.4]])
def test_rgr_rejects_unsupported_evolution_grids(grid):
    with pytest.raises(ValueError, match="RGR output grid"):
        impl._dglap_evolution_matrices(np.asarray(grid), impl._p_nlo_valence)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True])
def test_evolution_rejects_invalid_step_counts(steps):
    with pytest.raises(ValueError, match="positive integer"):
        impl._evolution_operator(1, 2, np.eye(2), np.eye(2), steps)


@pytest.mark.parametrize("name", ["momentum_gev", "scale_gev", "kappa", "mu_min_gev"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf])
def test_public_rgr_rejects_invalid_controls(name, value):
    kwargs = {"momentum_gev": 2.0, "scale_gev": 2.0, "zs_fm": 0.2, name: value}
    with pytest.raises(ValueError, match="finite and positive"):
        load_kernel("quark_pdf_cg_gt_hybrid_nlo_rgr_re")(np.array([0.25, 0.75]), np.array([0.25, 0.75]), **kwargs)
