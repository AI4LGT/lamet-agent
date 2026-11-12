"""Physical Gaussian endpoint priors and their conditioning/resampling semantics."""

from types import SimpleNamespace

import gvar as gv
import lsqfit
import numpy as np
import pytest

from lamet_agent.contract import evaluate_rules
from lamet_agent.stages._priors import validate_prior_overrides
from lamet_agent.stages.soft_analysis.contract import JOB_RULES, check_settings
from lamet_agent.stages.soft_analysis.physics import endpoint_fit, inclusive_window


PRIORS = {"c": "1.0(3.0)", "log(n)": "0.0(3.0)"}
WINDOWS = [[.3, .45], [.55, .7]]


def _weak_shape():
    x = np.linspace(.3, .7, 41)
    mean = 50 * (x * (1 - x))**3
    noise = np.random.default_rng(917).normal(0, .3, (48, len(x)))
    noise -= noise.mean(axis=0)
    return x, mean + noise


def test_omitted_and_empty_endpoint_priors_preserve_identical_fit_results():
    x, values = _weak_shape()
    omitted, omitted_record = endpoint_fit(x, values, WINDOWS, "bootstrap")
    empty, empty_record = endpoint_fit(x, values, WINDOWS, "bootstrap", priors={})
    explicit_none, none_record = endpoint_fit(x, values, WINDOWS, "bootstrap", priors=None)
    np.testing.assert_array_equal(omitted, empty)
    np.testing.assert_array_equal(omitted, explicit_none)
    assert omitted_record == empty_record == none_record
    assert "effective_priors" not in omitted_record


def test_reference_priors_regularize_a_weak_shape_and_report_augmented_fit_quality():
    x, values = _weak_shape()
    _, baseline = endpoint_fit(x, values, WINDOWS, "bootstrap")
    parameters, fitted = endpoint_fit(x, values, WINDOWS, "bootstrap", priors=PRIORS)
    assert baseline["c"] == pytest.approx(50., rel=1e-5)
    assert baseline["n"] == pytest.approx(3., rel=1e-5)
    assert fitted["c"] < 15
    assert fitted["n"] < baseline["n"]
    assert fitted["effective_priors"] == validate_prior_overrides(PRIORS)
    assert fitted["chi2"] == pytest.approx(fitted["data_chi2"] + fitted["prior_chi2"])
    assert fitted["dof"] == fitted["fit_points"]
    assert fitted["data_dof"] == fitted["fit_points"] - 2
    assert fitted["prior_chi2"] == pytest.approx(((fitted["c"] - 1) / 3)**2 + (np.log(fitted["n"]) / 3)**2)
    assert np.isfinite(parameters).all()


def test_endpoint_gaussians_agree_with_reference_lsqfit_at_center_and_each_replica():
    x, values = _weak_shape()
    parameters, diagnostic = endpoint_fit(x, values, WINDOWS, "bootstrap", priors=PRIORS)
    selected = inclusive_window(x, .3, .45) | inclusive_window(x, .55, .7)
    xx, yy = x[selected], values[:, selected]
    eigenvalues, vectors = np.linalg.eigh(np.cov(yy, rowvar=False))
    covariance = (vectors * np.maximum(eigenvalues, eigenvalues[-1] * 1e-6)) @ vectors.T
    original_prior = {"c": gv.gvar(1, 3), "log(n)": gv.gvar(0, 3)}

    def model(x, p):
        return p["c"] * (x * (1 - x))**p["n"]

    for sample in (None, 0, 23, 47):
        y = yy.mean(axis=0) if sample is None else yy[sample]
        reference = lsqfit.nonlinear_fit(data=(xx, gv.gvar(y, covariance)), fcn=model,
                                         prior=original_prior, svdcut=0)
        expected = [gv.mean(reference.p["c"]), gv.mean(reference.p["n"])]
        actual = [diagnostic["c"], diagnostic["n"]] if sample is None else parameters[sample]
        # The two solvers optimize linear c versus log(c), with their normal
        # stopping tolerances; compare well below the parameter uncertainty.
        np.testing.assert_allclose(actual, expected, rtol=5e-4)
        assert expected[0] > 0  # The existing positive model excludes no optimum here.
        if sample is None:
            assert diagnostic["chi2"] == pytest.approx(reference.chi2, rel=1e-8)
            assert diagnostic["dof"] == reference.dof
            assert diagnostic["Q"] == pytest.approx(reference.Q, rel=1e-8)


@pytest.mark.parametrize("factor", [1e-6, 1e6])
def test_physical_amplitude_priors_are_conditioned_exactly_once(factor):
    x, values = _weak_shape()
    original, original_record = endpoint_fit(x, values, WINDOWS, "bootstrap", priors=PRIORS)
    scaled_priors = {"c": f"{factor!r} +- {3 * factor!r}", "log(n)": PRIORS["log(n)"]}
    scaled, scaled_record = endpoint_fit(x, values * factor, WINDOWS, "bootstrap", priors=scaled_priors)
    np.testing.assert_allclose(scaled / np.array([factor, 1]), original, rtol=2e-5)
    assert scaled_record["c"] / factor == pytest.approx(original_record["c"], rel=2e-6)
    assert scaled_record["n"] == pytest.approx(original_record["n"], rel=2e-6)
    for name in PRIORS:
        for key in ("mean", "sdev"):
            assert scaled_record["conditioned_priors"][name][key] == pytest.approx(
                original_record["conditioned_priors"][name][key], rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("priors", [
    {"log(b)": {"mean": 0., "sdev": 3.}},
    {"c": {"mean": 1., "sdev": 0.}},
    {"c": {"mean": True, "sdev": 3.}},
    {"c": {"mean": np.inf, "sdev": 3.}},
    {"c": {"mean": 1., "sdev": 3., "extra": 1.}},
    {"c": [1., 3.]},
])
def test_invalid_reference_endpoint_priors_are_rejected(priors):
    with pytest.raises(ValueError):
        validate_prior_overrides(priors)
    issue = check_settings(SimpleNamespace(params={"type": "intrinsic_soft", "priors": priors}, inputs={}))
    assert issue.path == "priors"


def test_endpoint_prior_manifest_uses_named_gaussians_without_changing_other_defaults():
    inputs = {"quasi": ["q1", "q2"], "form_factor": "ff", "cs_kernel": "cs"}
    job = {"id": "si", "type": "intrinsic_soft", "inputs": inputs, "priors": PRIORS}
    assert evaluate_rules({"jobs": [job]}, JOB_RULES) == []
    assert check_settings(SimpleNamespace(params=job, inputs=inputs)) is None
    assert job["priors"] == PRIORS
    legacy = {"id": "si", "type": "intrinsic_soft", "inputs": inputs}
    assert evaluate_rules({"jobs": [legacy]}, JOB_RULES) == []
    assert legacy["priors"] == {}


def test_unused_endpoint_names_follow_shared_prior_policy_without_regularizing_fit():
    x, values = _weak_shape()
    baseline, baseline_record = endpoint_fit(x, values, WINDOWS, "bootstrap")
    result, record = endpoint_fit(x, values, WINDOWS, "bootstrap", priors={"unused": "1(2)"})
    np.testing.assert_array_equal(result, baseline)
    assert record.pop("inactive_prior_overrides") == ["unused"]
    assert record == baseline_record
