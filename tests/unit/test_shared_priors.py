"""Verify configurable priors across the independent numerical fit paths."""

import importlib
import json

import gvar as gv
import numpy as np
import pytest

from lamet_agent.contract import evaluate_rules
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.stages._priors import merge_priors, prior_diagnostics


def test_merge_preserves_defaults_and_broadcasts_independent_array_priors():
    correlated = gv.gvar([1.0, 2.0], [[1.0, 0.5], [0.5, 1.0]])
    defaults = {"fixed": correlated, "h0": gv.gvar([0.0, 0.0], [3.0, 3.0])}
    overrides = {"h0": "1.0(0.2)", "unused": "9.0(1.0)"}
    prior = merge_priors(defaults, overrides)
    assert set(prior) == set(defaults)
    np.testing.assert_array_equal(gv.evalcov(prior["fixed"]), gv.evalcov(correlated))
    np.testing.assert_allclose(gv.evalcov(prior["h0"]), np.eye(2) * 0.04)
    np.testing.assert_array_equal(gv.mean(defaults["h0"]), [0.0, 0.0])
    assert overrides["h0"] == "1.0(0.2)"
    recorded = json.loads(json.dumps(prior_diagnostics(prior, overrides)))
    assert recorded["inactive_prior_overrides"] == ["unused"]
    assert recorded["effective_priors"]["h0"] == {"mean": [1.0, 1.0], "sdev": [0.2, 0.2]}


@pytest.mark.parametrize(
    "stage,params",
    [
        ("renormalization", {"strategy": "self_renormalization", "type": "fit"}),
        ("fourier_transform", {}),
        ("extrapolation", {"operation": "fit"}),
    ],
)
def test_stage_contracts_accept_extra_names_but_reject_malformed_gaussians(stage, params):
    contract = importlib.import_module(f"lamet_agent.stages.{stage}.contract")
    for value, valid in [
        ({"unused": "2.0(0.5)"}, True),
        ({"unused": "2.0(0)"}, False),
        ({"unused": {"mean": True, "sdev": 1}}, False),
        ({"unused": {"mean": [1.0], "sdev": 1}}, False),
    ]:
        issues = evaluate_rules({**params, "priors": value}, contract.PARAM_RULES)
        assert (not any(issue.path.endswith("priors") for issue in issues)) is valid


@pytest.mark.parametrize("x_covariance", [False, True])
@pytest.mark.parametrize("independent", [False, True])
def test_extrapolation_named_priors_cover_both_covariance_and_coefficient_layouts(x_covariance, independent):
    from lamet_agent.stages.extrapolation.physics import fit_candidate

    rng = np.random.default_rng(203)
    data = []
    for index, spacing in enumerate([0.05, 0.07, 0.09, 0.11]):
        ensemble = EnsembleInfo("toy", str(index), spacing, spacing, 32, 64, 0.14)
        values = np.array([0.8, 1.0]) + 2 * spacing + rng.normal(0, 0.03, (24, 2))
        data.append(EnsembleData(
            ensemble, "bootstrap", list(values), ["x"], {"x": [0.2, 0.4]},
            attrs={"momentum_gev": 2.0, "resample_id": str(index)},
        ))
    kwargs = dict(x_range=(0.2, 0.4), x_covariance=x_covariance, x_independent_terms=["a"] if independent else [])
    baseline, _ = fit_candidate(data, ["a"], None, "0(3)", **kwargs)
    unchanged, unchanged_fit = fit_candidate(data, ["a"], None, {"unused": "9(1)"}, **kwargs)
    np.testing.assert_array_equal(unchanged.values, baseline.values)
    assert unchanged_fit["inactive_prior_overrides"] == ["unused"]
    changed, fit = fit_candidate(data, ["a"], None, {"h0": "0.5(0.001)"}, **kwargs)
    assert fit["effective_priors"] == {"h0": {"mean": 0.5, "sdev": 0.001}, "a": {"mean": 0.0, "sdev": 3.0}}
    assert np.max(np.abs(np.asarray(changed.mean) - 0.5)) < 0.01
    assert np.max(np.abs(np.asarray(changed.mean) - baseline.mean)) > 0.1


def test_self_renormalization_public_api_applies_overrides_to_both_fits(monkeypatch):
    from lamet_agent.stages.renormalization import physics

    z = np.array([0.0, 0.1, 0.2])
    spacings = [0.06, 0.12, 0.18]
    grids = [np.exp(physics.log_m(z, a, k=0.4, lambda_qcd_gev=0.1, d=0.0, n_f=3, scale_gev=2.0)
                    + 0.15 * z / 0.1973269804 + 0.4 * a) for a in spacings]
    rng = np.random.default_rng(90)
    samples = np.asarray(grids)[None, :, :] * (1 + rng.normal(0, 0.002, (12, 3, 3)))
    reference = EnsembleData(None, "bootstrap", list(samples), ["a", "z"], {"a": spacings, "z": z.tolist()})
    calls = []
    actual = physics.nonlinear_fit

    def capture(*args, **kwargs):
        calls.append(prior_diagnostics(kwargs["prior"])["effective_priors"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    kwargs = dict(short_distance_max_fm=0.2, k=0.4, lambda_qcd_gev=0.1, d=0.0, n_f=3, scale_gev=2.0,
                  zms_kernel=load_renormalization_kernel("z_msbar_pdf_nlo"), kernel_id="z_msbar_pdf_nlo", svdcut=1e-12)
    baseline = physics.fit_factor(reference, **kwargs)
    unchanged = physics.fit_factor(reference, priors={"unused": "9(1)"}, **kwargs)
    np.testing.assert_array_equal(unchanged.values, baseline.values)
    changed = physics.fit_factor(
        reference, priors={"g0.1": "0.4(0.1)", "m0": "0.2(0.01)"}, **kwargs
    )
    assert calls[-2]["g0.1"] == {"mean": 0.4, "sdev": 0.1}
    assert calls[-2]["f10.1"] == {"mean": 0.0, "sdev": 5.0}
    assert calls[-1] == {"m0": {"mean": 0.2, "sdev": 0.01}, "b": {"mean": 0.0, "sdev": 100.0}}
    assert not np.allclose(changed.values, baseline.values)


@pytest.fixture
def tail_input():
    from lamet_agent.stages.fourier_transform.physics import tail_model_values

    z = np.linspace(-1.0, 1.0, 21)
    parameters = {"A2": 0.8, "A2p": -0.05, "phi2": 0.2, "phi2p": -0.1, "Lambda": 0.7}
    center = tail_model_values(np.where(np.isclose(z, 0), 1e-6, z), "gi_nla", parameters, hadron="proton")
    center[np.isclose(z, 0)] = 1.0
    rng = np.random.default_rng(81)
    samples = center + rng.normal(0, 0.002, (8, len(z))) + 1j * rng.normal(0, 0.002, (8, len(z)))
    data = EnsembleData(None, "bootstrap", list(samples), ["z"], {"z": z.tolist()}, attrs={
        "coord_unit": "fm", "momentum_gev": 2.0, "hadron": "proton",
        "symmetry": '{"imag":"odd","real":"even"}', "sample_error_mode": "variance",
    })
    return data


def test_fourier_scan_applies_named_priors_to_ranges_and_la_nla_models(tail_input):
    from lamet_agent.stages.fourier_transform.physics import scan_fourier_transform

    data = tail_input
    result = scan_fourier_transform(
        data, [-0.5, 0, 0.5],
        transform={"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        tail={"models": ["gi_nla"], "z_min_fm": [0.3, 0.4], "z_max_fm": [1.0],
              "extent_fm": 1.2, "smoothing_method": "linear"},
        scan={"orders": ["LA", "NLA"], "sector": "full", "lambda0_gev": 0.1, "prior_widths": [1.0],
              "model_average": False, "max_schemes": 3, "source_component": "both", "output_scale": 1, "q_min": 0},
        priors={"Lambda": "0.6(0.2)", "A2p": "-0.05(0.5)",
                "unused": "9(1)"},
    )
    assert len(result["range_candidates"]) == 2
    assert len(result["model_candidates"]) == 2
    for candidate in [*result["range_candidates"], *result["model_candidates"]]:
        effective = candidate["effective_priors"]
        assert effective["Lambda"] == {"mean": 0.6, "sdev": 0.2}
        assert effective["A2"] == {"mean": 1.0, "sdev": 3.0}
        assert candidate["prior_convention"] == "physical_mean_internal_sdev"
        # Lambda is last for GI; the pre-existing log transform and internal width are preserved.
        internal = candidate["base_fit_priors"][f"u{len(effective) - 1}"]
        assert internal["mean"] == pytest.approx(np.log(0.6))
        assert internal["sdev"] == 0.2
        if candidate["order"] == "LA":
            assert candidate["inactive_prior_overrides"] == ["A2p", "unused"]
        else:
            assert effective["A2p"] == {"mean": -0.05, "sdev": 0.5}
            assert candidate["inactive_prior_overrides"] == ["unused"]


@pytest.mark.parametrize("model,name,value", [
    ("gi_nla", "A2", "25.0(0.1)"),
    ("gi_nla", "A2", "20.0(0.1)"),
    ("gi_nla", "phi2", "4.0(0.1)"),
    ("cg_nla", "n", "5.0(0.1)"),
])
def test_fourier_rejects_active_prior_means_outside_model_bounds(tail_input, monkeypatch, model, name, value):
    from lamet_agent.stages.fourier_transform import physics

    def unexpected_fit(*args, **kwargs):
        pytest.fail("out-of-range prior reached the fitter")

    monkeypatch.setattr(physics, "nonlinear_fit", unexpected_fit)
    means, widths = physics._scan_tail_priors(model_id=model, order="LA", lambda0_gev=0.1, hadron="proton")
    with pytest.raises(ValueError, match=f"prior '{name}' mean must lie strictly inside"):
        physics.fit_tail_parameters(
            tail_input, model_id=model, order="LA", z_min_fm=0.3, z_max_fm=1.0,
            prior_means=means, prior_widths=widths, priors={name: value}, hadron="proton", mode="center",
        )


@pytest.mark.parametrize("value,valid", [
    ("0.0(3.0)", True),
    ("0 ± 3.0", True),
    ({"h0": "1.0(0.3)"}, True),
    ({"mean": 0.0, "sdev": 3.0}, False),
    ("0.0(0)", False),
])
def test_extrapolation_contract_accepts_common_and_named_gvar_strings(value, valid):
    from lamet_agent.stages.extrapolation.contract import PARAM_RULES

    issues = evaluate_rules({"operation": "fit", "priors": value}, PARAM_RULES)
    assert (not any(issue.path.endswith("priors") for issue in issues)) is valid
