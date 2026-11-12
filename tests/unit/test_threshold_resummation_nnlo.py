import json
from pathlib import Path

import mpmath as mp
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from lamet_agent.kernels import load_kernel, matching_kernel_id
from lamet_agent.kernels import quark_pdf_gi_gt_hybrid_nnlo_rgr_tr as tr
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo_rgr import kernel as rgr


@pytest.mark.parametrize("key", ["hard", "soft_i", "soft_h", "evolution", "integrals", "correction", "fractional"])
def test_fresh_mathematica_reference(key, monkeypatch):
    ref = json.loads((Path(__file__).parent / "fixtures/tr-nnlo-reference.json").read_text())
    y = np.arange(-19, 20, 2) / 10
    monkeypatch.setattr(tr, "alpha_s", lambda loops, mu, mode: .32 if mu < 1.5 else .24)
    s, ag, aj = tr._evolution_integrals(.32, .24)
    actual = {
        "hard": lambda: tr._hard_nnlo(y, 2, 2, .18, .24),
        "soft_i": lambda: tr._soft_nnlo(y, 2, 1, .32),
        "soft_h": lambda: tr._soft_nnlo(y, 2, 2, .24),
        "evolution": lambda: np.exp(-2*s+aj)*tr._star_matrix(y, 2, 1, 2*ag),
        "integrals": lambda: np.array([s, ag, aj]),
        "correction": lambda: tr._threshold_correction(y, 2, 2, 1, .18, .37, "point"),
        "fractional": lambda: tr._threshold_correction(y, 2, 2, 1, .18, .37, "fractional"),
    }[key]()
    np.testing.assert_allclose(actual, ref[key], atol=3e-13, rtol=3e-13)


@pytest.mark.parametrize("i,j", [(0, 0), (1, 1), (0, 1), (0, 3)])
def test_quartic_soft_against_high_precision_fourier_derivatives(i, j):
    y = [-1.5, -.5, .5, 1.5]
    with mp.workdps(45):
        def entry(e):
            f = mp.sin(mp.pi*e/2)*mp.gamma(1-e)*mp.exp(-mp.euler*e)/mp.pi
            def off(k):
                d = abs(mp.mpf(y[k])-y[j])
                return f*(4*d)**e/d
            if i != j:
                return off(i)
            mass = f/e*((4*(2-y[j]))**e+(4*(2+y[j]))**e) if e else mp.mpf(1)
            return mass-sum(off(k) for k in range(4) if k != j)
        c = tr._soft_coefficients(.32)
        expected = sum(mp.mpf(float(c[n]))*(-2)**n*mp.diff(entry, 0, n) for n in range(5))
    actual = tr._soft_nnlo(np.array(y), 2, 1, .32)[i, j]
    np.testing.assert_allclose(actual, float(expected), atol=5e-13)


@pytest.mark.parametrize("mode", ["point", "fractional"])
def test_no_tr_exactly_recovers_nnlo_rgr(mode):
    y = np.array([-1.5, -.5, .5, 1.5])
    x = np.array([-.5, -.3, .3, .5])
    args = dict(momentum_gev=2., scale_gev=2., zs_fm=.18, evolution_points=10,
                evolution_steps=4, quadrature_order=16)
    np.testing.assert_array_equal(tr.kernel(x, y, cut_quadrature=mode, **args), rgr(x, y, **args))


@pytest.mark.parametrize("mode", ["point", "fractional"])
def test_signed_reflection_and_rectangular_transfer(mode):
    from lamet_agent.kernels.implementation import _lo_interp_matrix
    y = np.array([-1.5, -.5, .5, 1.5])
    x = np.array([-.7, .7])
    args = dict(momentum_gev=2., scale_gev=2.8, zs_fm=.18, evolution_points=10,
                evolution_steps=4, quadrature_order=16)
    actual = tr.kernel(x, y, cut_quadrature=mode, **args)
    fixed = tr._prepare_nnlo_rgr_matching(y, y, momentum_gev=2., zs_fm=.18, eps=1e-12, quadrature_order=16)
    expected = _lo_interp_matrix(x, y) @ (fixed(2.8)+tr._threshold_correction(y, 2, 2.8, 1.2, .18, .4, mode))
    np.testing.assert_allclose(actual, expected, atol=2e-13)
    np.testing.assert_allclose(actual[0], actual[1, ::-1], atol=2e-13)


def test_fractional_boundary_weights_and_switch():
    y = np.linspace(-2, 2, 40)
    mask = tr._cut_weights(y, .4, "fractional")
    assert np.any((mask > 0) & (mask < 1))
    np.testing.assert_allclose(mask, mask[::-1, ::-1], atol=1e-13)
    np.testing.assert_array_equal(np.diag(mask), 1)
    for mode in ("point", "fractional"):
        np.testing.assert_array_equal(tr._threshold_correction(y, 2, 2, 2, .18, .4, mode), 0)
        near = tr._threshold_correction(y, 2, 2+1e-7, 2-1e-7, .18, .4, mode)
        assert np.max(np.abs(near)) < 2e-6


def test_documented_finite_cut_residual_is_quadratic(monkeypatch):
    y = np.linspace(-2, 2, 80)
    b, _, _ = tr._anomalous_dimensions()
    results = []
    for ah in (.01, .005, .0025):
        ai = solve_ivp(lambda t, a: -(b[0]*a*a+b[1]*a**3+b[2]*a**4),
                       (np.log(2.8), np.log(1.2)), [ah], rtol=1e-12, atol=1e-15).y[0, -1]
        monkeypatch.setattr(tr, "alpha_s", lambda loops, mu, mode: ai if mu < 2 else ah)
        delta = tr._threshold_correction(y, 2, 2.8, 1.2, .18, .4)
        results.append(np.interp(.7, y, delta @ np.exp(-y*y)))
    assert all(3.7 < a/b < 4.2 for a, b in zip(results, results[1:]))


@pytest.mark.parametrize("extra", [dict(cut_quadrature="bad"), dict(threshold_cut=1), dict(threshold_cut=0),
                                   dict(spectator_kappa=True), dict(zs_fm=np.nan), dict(quadrature_order=8)])
def test_invalid_controls(extra):
    args = dict(momentum_gev=2, scale_gev=2, zs_fm=.18)
    args.update(extra)
    with pytest.raises(ValueError):
        tr.kernel([.7], np.array([-1.5, -.5, .5, 1.5]), **args)


@pytest.mark.parametrize("x", [.9, 1., -1.])
def test_spectator_cutoff(x):
    with pytest.raises(ValueError, match="spectator scale"):
        tr.kernel([x], np.array([-1.5, -.5, .5, 1.5]), momentum_gev=2, scale_gev=2, zs_fm=.18)


@pytest.mark.parametrize("component", ["re", "im", "both"])
def test_discovery(component):
    attrs = dict(parton="quark", observable="pdf", gfix="gi", operator="gt",
                 renormalization_scheme="hybrid", source_component=component)
    name = matching_kernel_id(attrs, order="nnlo", resummation="rgr_tr")
    assert name == "quark_pdf_gi_gt_hybrid_nnlo_rgr_tr"
    assert load_kernel(name).__name__ == "kernel"


@pytest.mark.parametrize("complex_samples", [False, True])
def test_nnlo_tr_stage_publishes_samples_and_provenance(tmp_path, complex_samples):
    order = "nnlo"
    from lamet_agent.agent import ToolContext
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.matching._apply import run
    from lamet_agent.stages.matching._inspection import run as inspect_kernel

    grid = np.linspace(-2, 2, 8)
    output_grid = np.array([-0.7, 0.3, 0.7])
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
        "matching",
        "match",
        {
            "order": order,
            "resummation": "rgr_tr",
            "mu": 2.0,
            "lc_x_ls": output_grid.tolist(),
            "kernel_parameters": {"evolution_points": 10, "evolution_steps": 4, "y_max": 4.,
                                  **({"quadrature_order": 16} if order == "nnlo" else {})},
        },
        {"quasi": quasi},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    inspect_kernel(context)
    assert context.params["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_tr"
    assert "antiquark" in context.state["kernel_inspection"]["document"]
    run(context)
    assert context.output is not None
    assert context.output.attrs["order"] == order
    assert context.output.attrs["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_tr"
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_tr")
    matrix = kernel(output_grid, grid, momentum_gev=2., scale_gev=2., zs_fm=.2,
                    evolution_points=10, evolution_steps=4, y_max=4., quadrature_order=16)
    np.testing.assert_allclose(context.output.array.values, np.asarray(samples) @ matrix.T)
    restored = EnsembleData.from_netcdf(tmp_path / "output.nc")
    np.testing.assert_allclose(restored.array.values, context.output.array.values)
    assert restored.attrs["order"] == order
    assert restored.attrs["output_component"] == ("both" if complex_samples else "re")
