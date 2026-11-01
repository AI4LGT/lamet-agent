from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from lamet_agent.kernels import load_kernel, matching_kernel_id
from lamet_agent.kernels import implementation as impl
from lamet_agent.kernels import quark_pdf_gi_gt_hybrid_nnlo as nnlo
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_rgr import _fixed_order


Y = np.array([-1.5, -.5, .5, 1.5])
ATTRS = dict(parton="quark", target_observable="pdf", gfix="gi", kernel_operator="gt",
             renormalization_scheme="hybrid", source_component="both")


def parameters(kernel):
    return dict(momentum_gev=2., scale_gev=2., zs_fm=.2, evolution_points=2, evolution_steps=4, y_max=4.)


def independent_matching(order, mu):
    coupling = impl.alpha_s(3 if order == "nnlo" else 2, mu, "running")
    nlo = _fixed_order(Y, Y, momentum_gev=2., scale_gev=mu, zs_fm=.2)
    forward = (np.eye(4) - nlo) * coupling / impl.alpha_s(2, mu, "running")
    fixed = np.eye(4) - forward
    if order == "nnlo":
        fixed = nnlo.kernel(Y, Y, momentum_gev=2., scale_gev=mu, zs_fm=.2, coupling_loops=3)
    shape = np.zeros((4, 4))
    for j, y in enumerate(Y):
        for i, x in enumerate(Y):
            if i != j:
                shape[i, j] = impl.C_z_lrr(x / y, y, 2., .4 / impl.GEV_FM) / abs(y)
        shape[j, j] = -sum(shape[i, j] for i in range(4) if i != j)
    r0 = impl.rnasym(0, 1., mu, 3) * coupling
    pv = impl.dPVasym(1., mu, 3, coupling)
    prefactor = fixed + r0 * shape
    if order == "nnlo":
        prefactor += impl.rnasym(1, 1., mu, 3) * coupling**2 * shape
        prefactor -= r0 * forward @ shape
        prefactor += .5 * r0**2 * shape @ shape
    return prefactor @ expm(-pv * shape)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("component", ["re", "im", "both"])
def test_combined_selection(order, component):
    assert matching_kernel_id({**ATTRS, "source_component": component}, order=order, resummation="rgr_lrr") == (
        f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr"
    )


@pytest.mark.parametrize("changes", [{"gfix": "cg"}, {"kernel_operator": "gz"},
                                    {"target_observable": "gpd"}, {"source_component": "bad"}])
def test_combined_rejects_unsupported_channels(changes):
    with pytest.raises(ValueError):
        matching_kernel_id({**ATTRS, **changes}, order="nlo", resummation="rgr_lrr")


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
def test_combined_no_evolution_matches_independent_square(order):
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr")
    x = np.array([-.5, .5])
    matrix = kernel(x, Y, **parameters(kernel))
    square = independent_matching(order, 2.)
    expected = np.column_stack([np.interp(x, Y, col) for col in square.T])
    np.testing.assert_allclose(matrix, expected, rtol=2e-11, atol=2e-11)
    np.testing.assert_allclose(square.sum(axis=0), 1., atol=3e-13)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
def test_combined_evolution_against_separate_ode(order):
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr")
    params = {**parameters(kernel), "evolution_steps": 80}
    grid = np.array([.5, 1.])
    signed = np.array([-1., -.5, .5, 1.])
    square = independent_matching(order, 3.)
    matched = np.column_stack([np.interp(signed, Y, col) for col in square.T])
    state = np.concatenate((matched[2:], -matched[:2][::-1]))
    d0, d1, d2 = impl._dglap_nnlo_isovector_blocks(grid, nf=3, y_max=4.)
    def rhs(t, flat):
        a = impl.alpha_s(3 if order == "nnlo" else 2, np.exp(t / 2), "running") / (4 * np.pi)
        evolution = a * d0 + a*a * d1 + (a**3 * d2 if order == "nnlo" else 0.)
        return (evolution @ flat.reshape(4, 4)).ravel()
    evolved = solve_ivp(rhs, (2 * np.log(3.), 2 * np.log(2.)), state.ravel(),
                       rtol=1e-11, atol=1e-13).y[:, -1].reshape(4, 4)
    expected = np.array([-(evolved[2] + evolved[3]) / 2, (evolved[0] + evolved[1]) / 2])
    np.testing.assert_allclose(kernel(np.array([-.75, .75]), Y, **params), expected, rtol=1e-5, atol=2e-7)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("eps_m", [0., -1., np.nan, True])
def test_combined_invalid_regulator(order, eps_m):
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr")
    with pytest.raises(ValueError, match="eps_m"):
        kernel(np.array([.5]), Y, **parameters(kernel), eps_m=eps_m)


def test_combined_reporting_names_both_resummations():
    from lamet_agent.stages.perturbative_matching.reporting import _kernel_structure
    assert _kernel_structure("quark_pdf_gi_gt_hybrid_nnlo_rgr_lrr")["resummation"] == "RGR+LRR"


@pytest.mark.parametrize("complex_samples", [False, True])
@pytest.mark.parametrize("order", ["nlo", "nnlo"])
def test_combined_stage_publishes_samples_and_provenance(tmp_path, complex_samples, order):
    from lamet_agent.agent import ToolContext
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.perturbative_matching._apply import run
    from lamet_agent.stages.perturbative_matching._inspection import run as inspect_kernel

    grid = Y
    output_grid = np.array([-0.5, 0.5, 1.])
    samples = [np.exp(-(grid**2)), 1.01 * np.exp(-(grid**2))]
    if complex_samples:
        samples = [sample + 0.2j * grid for sample in samples]
    quasi = EnsembleData(
        None,
        "bootstrap",
        samples,
        ["x"],
        {"x": grid.tolist()},
        attrs={
            "parton": "quark",
            "target_observable": "pdf",
            "gfix": "gi",
            "kernel_operator": "gt",
            "renormalization_scheme": "hybrid",
            "momentum_gev": 2.0,
            "zs_fm": 0.2,
            "source_component": "both",
            "output_component": "both" if complex_samples else "re",
        },
        name="quasi_distribution",
    )
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "perturbative_matching",
        "match",
        {
            "order": order,
            "resummation": "rgr_lrr",
            "mu": 2.0,
            "lc_x_ls": output_grid.tolist(),
            "kernel_parameters": {"evolution_points": 2, "evolution_steps": 4, "y_max": 4.,
                                  **({"quadrature_order": 16} if order == "nnlo" else {})},
        },
        {"quasi": quasi},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    inspect_kernel(context)
    assert context.params["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr"
    assert "antiquark" in context.state["kernel_inspection"]["document"]
    run(context)
    assert context.output is not None
    assert context.output.attrs["order"] == order
    assert context.output.attrs["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr"
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr")
    matrix = kernel(output_grid, grid, **parameters(kernel))
    np.testing.assert_allclose(context.output.array.values, np.asarray(samples) @ matrix.T)
    restored = EnsembleData.from_netcdf(tmp_path / "output.nc")
    np.testing.assert_allclose(restored.array.values, context.output.array.values)
    assert restored.attrs["order"] == order
    assert restored.attrs["output_component"] == ("both" if complex_samples else "re")


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
def test_combined_signed_cutoff_and_rectangular_grid(order):
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr")
    x = np.array([-.75, -.1, 0., .1, .75])
    matrix = kernel(x, Y, **parameters(kernel))
    assert matrix.shape == (5, 4)
    assert np.isfinite(matrix).all()
    np.testing.assert_array_equal(matrix[1:4], 0.)
    assert np.any(matrix[0] != 0.)
    assert np.any(matrix[-1] != 0.)


@pytest.mark.parametrize("xi", [-100., -10., -2., -1.0001, -1., -.9999, -.5, -.011, -.001,
                                .001, .02, .5, .99, 1.001, 1.02, 2., 9.9, 10., 100.])
def test_precomputed_log_parts_match_high_precision_reference(xi):
    for y in (-min(.3, 1 / abs(xi)), min(.7, 1 / abs(xi))):
        parts = nnlo._hybrid_log_parts(xi, y, 1.7, eps=1e-12, order=64)
        for ell in (-2., .4, 3.):
            expected = nnlo.hybrid_two_loop(xi, np.log(4.) - ell, y, 1.7, eps=1e-12, order=64)
            actual = parts @ np.array([1., ell, ell**2])
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-9)


def test_nnlo_preparation_builds_expensive_parts_once_per_grid_pair(monkeypatch):
    original = nnlo._hybrid_log_parts
    calls = []

    def counted(xi, source, *args, **kwargs):
        calls.append((xi, source))
        return original(xi, source, *args, **kwargs)

    monkeypatch.setattr(nnlo, "_hybrid_log_parts", counted)
    prepared = nnlo._prepare_nnlo_rgr_matching(Y, Y, momentum_gev=2., zs_fm=.2,
                                              eps=1e-12, quadrature_order=64)
    count = len(calls)
    assert count == len(set(calls)) == len(Y) * (len(Y) - 1)
    for mu in (.7, 2., 3.):
        expected = nnlo.kernel(Y, Y, momentum_gev=2., scale_gev=mu, zs_fm=.2, coupling_loops=3)
        np.testing.assert_allclose(prepared(mu), expected, rtol=2e-11, atol=2e-11)
    assert len(calls) == count
