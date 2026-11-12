import json
from pathlib import Path

import numpy as np
import pytest

from lamet_agent.kernels import load_kernel, matching_kernel_id
from lamet_agent.kernels import quark_pdf_gi_gt_hybrid_nlo_rgr_lrr_tr as tr


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("key", ["hard", "soft_i", "soft_h", "point", "fractional"])
def test_mathematica_lrr_tr_reference(order, key, monkeypatch):
    ref = json.loads((Path(__file__).parent / "fixtures/tr-lrr-reference.json").read_text())
    y = np.arange(-19, 20, 2)/10
    shape = tr._lrr_shape(y, 2., .02, 1e-12, .01)
    monkeypatch.setattr(tr, "alpha_s", lambda loops, mu, mode: .32 if mu < 1.5 else .24)
    if key == "hard":
        actual = tr._hard_lrr(y, 2, 2, .18, .24, order)
    elif key.startswith("soft"):
        mu, a = (1., .32) if key == "soft_i" else (2., .24)
        actual = tr._soft_lrr(y, 2, mu, a, shape, order)
    else:
        actual = tr._threshold_correction(y, 2, 2, 1, .18, .37, key, shape, order)
    np.testing.assert_allclose(actual, ref[order][key], atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(shape, ref["shape"], atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("mode", ["point", "fractional"])
def test_tr_off_recovers_existing_lrr_and_active_is_signed(order, mode):
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr_tr")
    baseline = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr")
    y = np.linspace(-2, 2, 8)
    args = dict(momentum_gev=2., scale_gev=2., zs_fm=.18, evolution_points=10, evolution_steps=4)
    if order == "nnlo":
        args["quadrature_order"] = 16
    x = np.array([-.5, -.3, .3, .5])
    np.testing.assert_array_equal(kernel(x, y, cut_quadrature=mode, **args), baseline(x, y, **args))
    x = np.array([-.7, .7])
    actual = kernel(x, y, cut_quadrature=mode, **args)
    np.testing.assert_allclose(actual[0], actual[1, ::-1], atol=2e-12)
    assert np.isfinite(actual).all()
    assert np.max(np.abs(actual-baseline(x, y, **args))) > 1e-6


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
def test_lrr_zero_recovers_tr_ingredients(order, monkeypatch):
    from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_rgr_tr import _hard_nlo
    from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo_rgr_tr import _hard_nnlo
    monkeypatch.setattr(tr, "rnasym", lambda *a: 0.)
    monkeypatch.setattr(tr, "dPVasym", lambda *a: 0.)
    y = np.linspace(-2, 2, 8)
    shape = tr._lrr_shape(y, 2, .02, 1e-12, .01)
    hard = _hard_nlo if order == "nlo" else _hard_nnlo
    soft = tr._soft_nlo if order == "nlo" else tr._soft_nnlo
    np.testing.assert_allclose(tr._hard_lrr(y, 2, 2, .18, .24, order), hard(y, 2, 2, .18, .24), atol=2e-14)
    np.testing.assert_allclose(tr._soft_lrr(y, 2, 2, .24, shape, order), soft(y, 2, 2, .24), atol=2e-14)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("params", [dict(soft_eps_m=0), dict(soft_zs_gev_inv=-1), dict(eps_m=False),
                                    dict(threshold_cut=1), dict(cut_quadrature="bad"), dict(spectator_kappa=.1)])
def test_invalid_controls_and_spectator_guard(order, params):
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr_tr")
    with pytest.raises(ValueError):
        kernel(np.array([.7]), np.linspace(-2, 2, 8), momentum_gev=2., scale_gev=2., zs_fm=.18, **params)


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("component", ["re", "im", "both"])
def test_selector(order, component):
    attrs = dict(parton="quark", target_observable="pdf", gfix="gi", kernel_operator="gt",
                 renormalization_scheme="hybrid", source_component=component)
    assert matching_kernel_id(attrs, order=order, resummation="rgr_lrr_tr") == (
        f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr_tr")
    attrs["gfix"] = "cg"
    with pytest.raises(ValueError):
        matching_kernel_id(attrs, order=order, resummation="rgr_lrr_tr")


@pytest.mark.parametrize("order", ["nlo", "nnlo"])
@pytest.mark.parametrize("complex_samples", [False, True])
def test_lrr_tr_stage_publishes_samples_and_provenance(tmp_path, complex_samples, order):
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
            "resummation": "rgr_lrr_tr",
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
    assert context.params["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr_tr"
    assert "antiquark" in context.state["kernel_inspection"]["document"]
    run(context)
    assert context.output is not None
    assert context.output.attrs["order"] == order
    assert context.output.attrs["kernel_id"] == f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr_tr"
    kernel = load_kernel(f"quark_pdf_gi_gt_hybrid_{order}_rgr_lrr_tr")
    matrix = kernel(output_grid, grid, momentum_gev=2., scale_gev=2., zs_fm=.2,
                    evolution_points=10, evolution_steps=4, y_max=4.,
                    **({"quadrature_order": 16} if order == "nnlo" else {}))
    np.testing.assert_allclose(context.output.array.values, np.asarray(samples) @ matrix.T)
    restored = EnsembleData.from_netcdf(tmp_path / "output.nc")
    np.testing.assert_allclose(restored.array.values, context.output.array.values)
    assert restored.attrs["order"] == order
    assert restored.attrs["output_component"] == ("both" if complex_samples else "re")
