from __future__ import annotations

import importlib

import numpy as np
import pytest

from lamet_agent.kernels import implementation as impl, load_kernel
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_rgr import _fixed_order


@pytest.mark.parametrize("size", [4, 12, 40])
@pytest.mark.parametrize("eps", [1e-12, 1e-6])
def test_nlo_tables_against_scalar_reference(size, eps):
    y = np.linspace(-2., 2., size)
    pz, zs = 1.7220027554991584, .18
    constant, slope = impl._gi_gt_hybrid_nlo_tables(y, zs * pz / impl.GEV_FM, eps)
    np.testing.assert_allclose(constant.sum(axis=0), 0., atol=1e-12)
    np.testing.assert_allclose(slope.sum(axis=0), 0., atol=1e-12)
    for mu in (.7, 2., 4.5):
        expected = _fixed_order(y, y, momentum_gev=pz, scale_gev=mu, zs_fm=zs, eps=eps)
        a = impl.alpha_s(2, mu, "running") / (2 * np.pi)
        actual = np.eye(size) - a * (constant + 2 * np.log(mu / pz) * slope)
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("suffix", ["rgr", "rgr_lrr"])
def test_nlo_public_prepares_tables_only_once(monkeypatch, suffix):
    module = importlib.import_module(f"lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_{suffix}")
    original = module._gi_gt_hybrid_nlo_tables
    calls = []

    def counted(y, *args):
        calls.append(y.copy())
        return original(y, *args)

    monkeypatch.setattr(module, "_gi_gt_hybrid_nlo_tables", counted)
    y = np.linspace(-2., 2., 8)
    module.kernel(np.array([-.9, -.5, 0., .5, .9]), y, momentum_gev=2., scale_gev=2.,
                  zs_fm=.2, evolution_points=4, evolution_steps=4, y_max=4.)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], y)


@pytest.mark.parametrize("suffix", ["rgr", "rgr_lrr"])
def test_nlo_evolved_matrix_matches_scalar_matching_at_each_scale(suffix):
    y = np.linspace(-2., 2., 8)
    x = np.array([-.8, -.5, .5, .8])
    fixed = _fixed_order if suffix == "rgr" else load_kernel("quark_pdf_gi_gt_hybrid_nlo_lrr")

    def prepare(rows, columns):
        return lambda mu: fixed(rows, columns, momentum_gev=2., scale_gev=mu, zs_fm=.2)

    controls = dict(momentum_gev=2., scale_gev=2., kappa=1., mu_min_gev=.6,
                    evolution_points=4, evolution_steps=4, y_max=4., eps=1e-12)
    expected = impl._rgr_nonsinglet(x, y, prepare_matching=prepare, order="nlo", **controls)
    actual = load_kernel(f"quark_pdf_gi_gt_hybrid_nlo_{suffix}")(x, y, zs_fm=.2, **controls)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)
