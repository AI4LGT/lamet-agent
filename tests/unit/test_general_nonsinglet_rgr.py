from __future__ import annotations

import inspect

import numpy as np
import pytest

from lamet_agent.kernels import load_kernel, matching_kernel_id
from lamet_agent.kernels import implementation as impl
from lamet_agent.kernels import quark_pdf_gi_gt_hybrid_nnlo as nnlo


NLO_IDS = [
    f"quark_pdf_{gauge}_{op}_{scheme}_nlo_rgr"
    for gauge in ("gi", "cg") for op in ("gt", "gz")
    for scheme in (("hybrid", "ratio") if gauge == "gi" else ("hybrid", "ratio", "msbar"))
]
IDS = NLO_IDS + ["quark_pdf_gi_gt_hybrid_nnlo_rgr"]
Y = np.array([-1.5, -0.5, 0.5, 1.5])


def parameters(kernel):
    kw = dict(momentum_gev=2., scale_gev=2., evolution_points=2, evolution_steps=4, y_max=4.)
    if "zs_fm" in inspect.signature(kernel).parameters:
        kw["zs_fm"] = 0.2
    if "quadrature_order" in inspect.signature(kernel).parameters:
        kw["quadrature_order"] = 16
    return kw


@pytest.mark.parametrize("kernel_id", IDS)
def test_signed_output_and_no_evolution_limit(kernel_id):
    kernel = load_kernel(kernel_id)
    kw = parameters(kernel)
    x = np.array([-0.5, 0.5])
    actual = kernel(x, Y, **kw)
    fixed = load_kernel(kernel_id.removesuffix("_rgr"))
    fixed_kw = {key: value for key, value in kw.items() if key in inspect.signature(fixed).parameters}
    if "nnlo" in kernel_id:
        fixed_kw["coupling_loops"] = 3
    square = fixed(Y, Y, **fixed_kw)
    expected = np.array([np.interp(x, Y, column) for column in square.T]).T
    if "msbar" in kernel_id:
        expected *= impl._zpsi_msbar_ratio(2.)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    assert np.linalg.norm(actual[0]) > 0
    np.testing.assert_allclose(actual, actual[::-1, ::-1], atol=2e-12)


@pytest.mark.parametrize("kernel_id", IDS)
def test_off_grid_transfer_preserves_complete_matching(kernel_id):
    kernel = load_kernel(kernel_id)
    kw = parameters(kernel)
    y = np.linspace(-2., 2., 8)
    x = np.array([-0.5, 0.5])
    fixed = load_kernel(kernel_id.removesuffix("_rgr"))
    fixed_kw = {key: value for key, value in kw.items() if key in inspect.signature(fixed).parameters}
    if "nnlo" in kernel_id:
        fixed_kw["coupling_loops"] = 3
    samples = np.array([np.exp(-y*y), y * np.exp(-y*y) + 0.3j * np.exp(-y*y)])
    matched_samples = samples @ fixed(y, y, **fixed_kw).T
    expected = np.array([np.interp(x, y, sample) for sample in matched_samples])
    if "msbar" in kernel_id:
        expected *= impl._zpsi_msbar_ratio(2.)
    actual = samples @ kernel(x, y, **kw).T
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-12)


def test_fourier_grid_transfer_against_independent_column_subtraction():
    y = np.linspace(-2., 2., 200)
    q = np.exp(-y*y)
    density = impl._pdf_density(
        lambda xi, log_scale, yy: impl.C_hybrid_gi(xi, log_scale, yy, 0.4 / impl.GEV_FM, 1e-12),
        2., 2.,
    )
    regular = np.array([
        [0. if i == j else density(xx, yy) for j, yy in enumerate(y)]
        for i, xx in enumerate(y)
    ])
    # Assemble the square-grid plus distribution independently of either
    # production matching builder or the RGR grid-transfer helper.
    plus = regular - np.diag(regular.sum(axis=0))
    matched = q - impl.alpha_s(2, 2., "running") * impl.CF / (2*np.pi) * (y[1]-y[0]) * (plus @ q)
    x = np.array([-0.5, 0.5])
    expected = np.interp(x, y, matched)
    actual = load_kernel(NLO_IDS[0])(
        x, y, momentum_gev=2., scale_gev=2., zs_fm=0.2,
        evolution_points=2, evolution_steps=1, y_max=4.,
    ) @ q
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
def test_transferred_state_evolves_against_independent_ode(order):
    from scipy.integrate import solve_ivp

    y = np.linspace(-2., 2., 8)
    grid = np.arange(1, 5) / 4
    signed = np.r_[-grid[::-1], grid]
    q = np.exp(-y*y) * (1 + 0.2*y)
    fixed = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}")
    kw = dict(momentum_gev=2., scale_gev=2., zs_fm=0.2)
    if order == "nnlo":
        kw.update(coupling_loops=3, quadrature_order=16)
    matched = fixed(y, y, **kw) @ q
    projected = np.interp(signed, y, matched)
    initial = np.r_[projected[4:], -projected[:4][::-1]]
    d0, d1, d2 = impl._dglap_nnlo_isovector_blocks(grid, y_max=4.)

    def rhs(t, state):
        a = impl.alpha_s(3 if order == "nnlo" else 2, np.exp(t/2), "running") / (4*np.pi)
        generator = a*d0 + a*a*d1
        if order == "nnlo":
            generator += a**3*d2
        return generator @ state

    final = solve_ivp(rhs, (2*np.log(2.), 2*np.log(3.)), initial, rtol=1e-11, atol=1e-13).y[:, -1]
    expected = np.array([-final[5], final[1]])
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr")
    actual = kernel(
        np.array([-0.5, 0.5]), y, momentum_gev=2., scale_gev=3., zs_fm=0.2,
        evolution_points=4, evolution_steps=80, y_max=4.,
        **({"quadrature_order": 16} if order == "nnlo" else {}),
    ) @ q
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize("kernel_id", IDS)
def test_both_parities_and_complex_samples(kernel_id):
    kernel = load_kernel(kernel_id)
    kw = {**parameters(kernel), "scale_gev": 3.}
    x = np.array([-1., -0.5, 0., 0.5, 1.])
    matrix = kernel(x, Y, **kw)
    np.testing.assert_allclose(matrix, matrix[::-1, ::-1], atol=2e-12)
    np.testing.assert_array_equal(matrix[2], 0)
    even = np.array([1., 2., 2., 1.])
    odd = np.array([-1., -3., 3., 1.])
    result = matrix @ (even + 1j * odd)
    np.testing.assert_allclose(result.real, result.real[::-1], atol=2e-12)
    np.testing.assert_allclose(result.imag, -result.imag[::-1], atol=2e-12)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_coupled_evolution_reduces_to_plus_minus(order, sign):
    grid = np.arange(1, 5) / 4
    d0, q1, b1, q2, b2 = impl._dglap_nnlo_isovector_matrices(grid, y_max=6.)
    b0, b1full, b2full = impl._dglap_nnlo_isovector_blocks(grid, y_max=6.)
    scalar = impl._evolution_operator(
        1.3, 2.7, d0, q1 + sign * b1, 12,
        evo_nnlo=q2 + sign * b2 if order == "nnlo" else None,
    )
    coupled = impl._evolution_operator(
        1.3, 2.7, b0, b1full, 12, evo_nnlo=b2full if order == "nnlo" else None,
    )
    q = np.array([2., 1., 0.3, 0.])
    expected = scalar @ q
    np.testing.assert_allclose(coupled @ np.r_[q, sign * q], np.r_[expected, sign * expected], atol=2e-13)


@pytest.mark.parametrize("scale", [0.9, 1.7, 3.4])
def test_prepared_nnlo_matches_direct_formula(scale):
    x = np.array([-1., -0.5, 0.5, 1.])
    kw = dict(momentum_gev=2., zs_fm=0.2, eps=1e-12, quadrature_order=16)
    prepared = nnlo._prepare_nnlo_rgr_matching(x, Y, **kw)
    direct = nnlo.kernel(x, Y, scale_gev=scale, coupling_loops=3, **kw)
    np.testing.assert_allclose(prepared(scale), direct, rtol=3e-12, atol=3e-12)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("component", ["re", "im", "both"])
def test_selection_accepts_both_source_components(order, component):
    attrs = dict(parton="quark", target_observable="pdf", gfix="gi", kernel_operator="gt",
                 renormalization_scheme="hybrid", source_component=component)
    name = matching_kernel_id(attrs, order=order, resummation="rgr")
    assert name == f"quark_pdf_gi_gt_hybrid_{order}_rgr"
    assert callable(load_kernel(name))


@pytest.mark.parametrize("changes", [
    {"evolution_points": 1}, {"evolution_steps": 0}, {"kappa": 0},
    {"mu_min_gev": float("nan")}, {"zs_fm": 0}, {"y_max": 0.5},
])
def test_rejects_invalid_controls(changes):
    kernel = load_kernel(IDS[-1])
    with pytest.raises(ValueError):
        kernel(np.array([0.5]), Y, **{**parameters(kernel), **changes})


def test_cutoff_and_arbitrary_output_order():
    kernel = load_kernel(NLO_IDS[0])
    kw = parameters(kernel)
    x = np.array([1., -0.5, 0.1, -0.1, 0., 0.5])
    matrix = kernel(x, Y, **kw)
    np.testing.assert_array_equal(matrix[2:5], 0)
    np.testing.assert_allclose(matrix[[5, 1, 0]], kernel(x[[5, 1, 0]], Y, **kw), atol=1e-13)
    with pytest.raises(ValueError, match="below the evolution grid"):
        kernel(np.array([0.25]), Y, **kw)


@pytest.mark.parametrize("y", [np.array([-0.5, 0.5]), np.array([-1., 0., 1.]),
                                  np.array([-1.5, -0.4, 0.5, 1.5]), Y[::-1]])
def test_rejects_invalid_input_grids(y):
    kernel = load_kernel(NLO_IDS[0])
    with pytest.raises(ValueError):
        kernel(np.array([0.5]), y, **parameters(kernel))


@pytest.mark.parametrize("complex_samples", [False, True])
@pytest.mark.parametrize("order", ["nlo", "nnlo"])
def test_general_rgr_stage_publishes_samples_and_provenance(tmp_path, complex_samples, order):
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
            "resummation": "rgr",
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
    assert context.params["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr"
    assert "antiquark" in context.state["kernel_inspection"]["document"]
    run(context)
    assert context.output is not None
    assert context.output.attrs["order"] == order
    assert context.output.attrs["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr"
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr")
    matrix = kernel(output_grid, grid, **parameters(kernel))
    np.testing.assert_allclose(context.output.array.values, np.asarray(samples) @ matrix.T)
    restored = EnsembleData.from_netcdf(tmp_path / "output.nc")
    np.testing.assert_allclose(restored.array.values, context.output.array.values)
    assert restored.attrs["order"] == order
    assert restored.attrs["output_component"] == ("both" if complex_samples else "re")
