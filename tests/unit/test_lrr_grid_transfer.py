from __future__ import annotations

import importlib

import numpy as np
import pytest
from scipy.linalg import expm
from scipy.integrate import quad

from lamet_agent.kernels import implementation as impl


IDS = [
    f"quark_pdf_gi_{op}_hybrid_nlo_lrr"
    for op in ("gt", "gz", "gtg5", "gzg5", "gtgpg5")
] + [f"quark_da_gi_{op}_hybrid_nlo_lrr" for op in ("gtg5", "gzg5")]


@pytest.mark.parametrize("kernel_id", IDS)
def test_off_grid_lrr_matches_independent_square_assembly(kernel_id):
    module = importlib.import_module(f"lamet_agent.kernels.{kernel_id}")
    y = np.linspace(0.05, 0.95, 10) if "_da_" in kernel_id else np.linspace(-1.95, 1.95, 40)
    x = np.array([0.17, 0.38, 0.76])
    pz, mu, zs = 2.0, 2.0, 0.2
    zspz = zs * pz / impl.GEV_FM
    fixed = module._fixed_order(y, pz, mu, y, 1e-12, zspz)
    shape = np.zeros((y.size, y.size))
    for j, source in enumerate(y):
        for i, target in enumerate(y):
            if i != j:
                shape[i, j] = impl.C_z_lrr(target / source, source, pz, zspz) / abs(source)
        shape[j, j] = -sum(shape[i, j] for i in range(y.size) if i != j)
    shape *= y[1] - y[0]
    coupling = impl.alpha_s(impl.ALPHAS_N_LOOP, mu, "running")
    r0 = impl.rnasym(0, 1., mu, 3) * coupling
    pv = impl.dPVasym(1., mu, 3, coupling)
    square = (fixed + r0 * shape) @ expm(-pv * shape)
    expected = np.column_stack([np.interp(x, y, column) for column in square.T])
    actual = module.kernel(x, y, momentum_gev=pz, scale_gev=mu, zs_fm=zs)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        module.kernel(y, y, momentum_gev=pz, scale_gev=mu, zs_fm=zs),
        square, rtol=2e-12, atol=2e-12,
    )


@pytest.mark.parametrize("nf", [3, 4, 5])
@pytest.mark.parametrize("coupling", [0.15, 0.25, 0.35])
def test_pv_against_integrable_borel_quadrature(nf, coupling):
    beta0, b, c1, c2 = impl._renormalon_params(nf)
    k = 2 * np.pi / (coupling * beta0)

    def integral(power):
        left = quad(lambda t: np.exp(-k * t), 0., 1., weight="alg", wvar=(0., -power))[0]
        right = np.exp(-k) * quad(lambda t: np.exp(-k * t) * t**(-power), 0., np.inf)[0]
        return left + np.cos(np.pi * power) * right

    jb = integral(b)
    value = (-1 + k * jb) / b + c1 * jb + c2 * integral(b - 1)
    expected = impl._NM_RENORMALON[nf] * 2 * np.pi / beta0 * value
    np.testing.assert_allclose(impl.dPVasym(1., 1., nf, coupling), expected, rtol=2e-8, atol=2e-10)


@pytest.mark.parametrize("x", [[], [[0.2]], [np.nan], [-3.], [3.]])
def test_invalid_output_grid_rejected(x):
    module = importlib.import_module("lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_lrr")
    with pytest.raises(ValueError, match="LRR"):
        module.kernel(np.asarray(x), np.array([-1.5, -0.5, 0.5, 1.5]),
                      momentum_gev=2., scale_gev=2., zs_fm=0.2)


@pytest.mark.parametrize("ksi", [-2., 0.2, 0.9, 1.1, 2.])
def test_shape_against_complex_fourier_integral(ksi):
    pz, source, zs, regulator = 2., -0.4, 0.8, 0.03
    parton_pz = abs(source) * pz
    frequency = (ksi - 1.) * parton_pz
    decay = regulator - 1j * frequency
    integral = np.exp(-decay * zs) * (zs / decay + 1 / decay**2)
    integral -= zs * np.exp(-regulator * zs) * np.exp(1j * frequency * zs) / (-1j * frequency)
    expected = parton_pz / np.pi * integral.real
    actual = impl.C_z_lrr(ksi, source, pz, zs * pz, regulator)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
