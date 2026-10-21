"""Scientific and workflow checks for fixed-order nonsinglet NNLO matching."""

from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import sici

from lamet_agent.kernels import load_kernel, matching_kernel_id
from lamet_agent.kernels import implementation as impl
from lamet_agent.kernels import quark_pdf_gi_gt_hybrid_nnlo as nnlo
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo import (
    Gamma2,
    hybrid_two_loop,
    msbar_two_loop,
    smooth_conversion,
)


@pytest.mark.parametrize("xi,b,ls", [(0.3, 1.5, 0.7), (1.0, 1.5, 0.7), (-2.0, 0.3, -0.4)])
def test_smooth_term_against_direct_coordinate_integral(xi, b, ls):
    def outer(t):
        logt = 2 * np.log(t)

        def inner(v):
            u = 1 - v
            diff = np.cos((xi - u) * b * t) - np.cos((xi - 1) * b * t)
            return (-(1 + u * u) * (ls + logt) + 1 - 4 * u + u * u - 4 * np.log(v)) * diff / v

        return logt * quad(inner, 0, 1, epsabs=1e-9)[0]

    reference = -1.5 * impl.CF**2 * b / np.pi * quad(outer, 0, 1, epsabs=1e-8)[0]
    assert smooth_conversion(xi, b, ls, 64) == pytest.approx(reference, rel=1e-9, abs=1e-10)


@pytest.mark.parametrize("xi", [-2.0, -0.4, 0.3, 1.5])
def test_hybrid_conversion_against_uncombined_eq41(xi):
    y, zspz, log_scale = 0.4, 2.1, 0.7
    b = abs(y * zspz)
    d = 1 - xi
    ell = np.log(4) - log_scale
    ls = np.log(b * b) + ell + 2 * np.euler_gamma - np.log(4)
    a = 11 * impl.CA * impl.CF / 8 - impl.CF * impl.NF * impl.TF / 2 + 9 * impl.CF**2 / 8
    c = (
        (53 / 8 - np.pi**2 / 6) * impl.CA * impl.CF
        + (25 / 8 + 2 * np.pi**2 / 3) * impl.CF**2
        - 5 * impl.CF * impl.NF * impl.TF / 2
    )
    delta = -1 / abs(d) + 2 * sici(d * b)[0] / (np.pi * d)
    hyper = float(mp.re(mp.hyper([1, 1, 1], [2, 2, 2], 1j * d * b)))
    reference = msbar_two_loop(xi, ell) - a * 2 * (np.log(4 * d * d) - ell) / abs(d) + c / abs(d)
    reference += a * (
        2 * np.log(b * b * np.exp(2 * np.euler_gamma) * d * d) / abs(d) - 8 * b / np.pi * hyper + 2 * delta * ls
    )
    reference += c * delta - impl.CF**2 * (1.5 * ls + 2.5) * (1.5 * delta + impl.C_ratio_gi(xi, log_scale))
    reference += smooth_conversion(xi, b, ls, 96)
    assert hybrid_two_loop(xi, log_scale, y, zspz, eps=1e-12, order=64) == pytest.approx(
        reference, rel=2e-11, abs=2e-10
    )


@pytest.mark.parametrize("xi", [-2.0, -0.4, 0.3, 1.5])
def test_msbar_scale_logs_have_published_double_log_coefficient(xi):
    # The second finite difference removes all finite and single-log pieces.
    second = msbar_two_loop(xi, 1) + msbar_two_loop(xi, -1) - 2 * msbar_two_loop(xi, 0)
    assert second == pytest.approx(2 * float(Gamma2(mp.mpf(xi))), rel=1e-11, abs=1e-11)


def test_removable_minus_one_boundary():
    from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo import _msbar

    left, right = msbar_two_loop(-1 - 1e-7, 0.4), msbar_two_loop(-1 + 1e-7, 0.4)
    assert _msbar(-1, 0.4) == pytest.approx((left + right) / 2, abs=1e-8)


@pytest.mark.parametrize("descending", [False, True])
def test_current_conservation_and_parity(descending):
    grid = np.array([-0.75, -0.25, 0.25, 0.75])
    if descending:
        grid = grid[::-1]
    matrix = nnlo.kernel(grid, grid, momentum_gev=2, scale_gev=2, zs_fm=0.2)
    np.testing.assert_allclose(matrix.sum(axis=0), 1, atol=1e-13)
    np.testing.assert_allclose(matrix, matrix[::-1, ::-1], atol=1e-12)


def test_inverse_expansion_cancels_forward_to_second_order(monkeypatch):
    grid = np.array([-0.75, -0.25, 0.25, 0.75])
    density = impl._pdf_density(
        lambda xi, log_scale, y: hybrid_two_loop(xi, log_scale, y, 0.4 / impl.GEV_FM, eps=1e-12, order=64), 2, 2
    )
    k2 = impl._plus_prescription_matrix(grid, grid, density, 1e-12)
    residuals = []
    for alpha in [0.02, 0.01]:
        monkeypatch.setattr(impl, "alpha_s", lambda *args: alpha)
        monkeypatch.setattr(nnlo, "alpha_s", lambda *args: alpha)
        inverse = nnlo.kernel(grid, grid, momentum_gev=2, scale_gev=2, zs_fm=0.2)
        nlo = load_kernel("quark_pdf_gi_gt_hybrid_nlo")(grid, grid, momentum_gev=2, scale_gev=2, zs_fm=0.2)
        forward = 2 * np.eye(grid.size) - nlo + (alpha / (2 * np.pi)) ** 2 * k2
        residuals.append(np.linalg.norm(inverse @ forward - np.eye(grid.size)))
    assert 7.5 < residuals[0] / residuals[1] < 8.5


def test_rectangular_output_subset():
    grid = np.array([-0.75, -0.25, 0.25, 0.75])
    kw = dict(momentum_gev=2, scale_gev=2, zs_fm=0.2)
    np.testing.assert_allclose(nnlo.kernel(grid[[0, 2]], grid, **kw), nnlo.kernel(grid, grid, **kw)[[0, 2]], atol=1e-13)


@pytest.mark.parametrize(
    "changes",
    [{"gfix": "cg"}, {"kernel_operator": "gz"}, {"target_observable": "gpd"}, {"renormalization_scheme": "ratio"}],
)
def test_nnlo_channel_selection_is_restricted(changes):
    attrs = dict(
        parton="quark", target_observable="pdf", gfix="gi", kernel_operator="gt", renormalization_scheme="hybrid"
    )
    assert matching_kernel_id(attrs, order="nnlo") == "quark_pdf_gi_gt_hybrid_nnlo"
    with pytest.raises(ValueError, match="NNLO is available only"):
        matching_kernel_id({**attrs, **changes}, order="nnlo")
    with pytest.raises(ValueError, match="NNLO is available only"):
        matching_kernel_id(attrs, order="nnlo", resummation="rgr")


@pytest.mark.parametrize(
    "kwargs", [{"quadrature_order": True}, {"quadrature_order": 15}, {"zs_fm": 0}, {"scale_gev": float("nan")}]
)
def test_invalid_controls(kwargs):
    params = dict(momentum_gev=2, scale_gev=2, zs_fm=0.2)
    with pytest.raises(ValueError):
        nnlo.kernel(np.array([0.25, 0.75]), np.array([0.25, 0.75]), **{**params, **kwargs})


@pytest.mark.parametrize("complex_samples", [False, True])
def test_nnlo_matching_stage_publishes_samples_and_provenance(tmp_path, complex_samples):
    from lamet_agent.agent import ToolContext
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.perturbative_matching._apply import run
    from lamet_agent.stages.perturbative_matching._inspection import run as inspect_kernel

    grid = np.array([-0.75, -0.25, 0.25, 0.75])
    samples = [np.exp(-(grid**2)), 1.01 * np.exp(-(grid**2))]
    if complex_samples:
        samples = [sample + 0.2j * grid for sample in samples]
    output_grid = grid[[0, 2, 3]] if complex_samples else grid
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
            "source_component": "re",
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
            "order": "nnlo",
            "resummation": "",
            "mu": 2.0,
            "lc_x_ls": output_grid.tolist(),
            "kernel_parameters": {"quadrature_order": 64},
        },
        {"quasi": quasi},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    inspect_kernel(context)
    assert context.params["kernel_id"] == "quark_pdf_gi_gt_hybrid_nnlo"
    assert "alpha_s squared" in context.state["kernel_inspection"]["document"]
    run(context)
    assert context.output is not None
    assert context.output.attrs["order"] == "nnlo"
    assert context.output.attrs["kernel_id"] == "quark_pdf_gi_gt_hybrid_nnlo"
    matrix = nnlo.kernel(output_grid, grid, momentum_gev=2, scale_gev=2, zs_fm=0.2)
    np.testing.assert_allclose(context.output.array.values, np.asarray(samples) @ matrix.T)
    restored = EnsembleData.from_netcdf(tmp_path / "output.nc")
    np.testing.assert_allclose(restored.array.values, context.output.array.values)
    assert restored.attrs["order"] == "nnlo"
    assert restored.attrs["output_component"] == ("both" if complex_samples else "re")
