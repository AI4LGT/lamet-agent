"""Native TMD prior conditioning, common replica modes, and published evidence."""

import gvar as gv
import numpy as np
import pytest

from lamet_agent.agent import ToolContext, _review_summary
from lamet_agent.stages.correlator_analysis import _fit_ff, _publish, _tmdwf
from lamet_agent.stages.correlator_analysis._priors import declared_prior_names
from lamet_agent.stages.correlator_analysis.physics import fit_qda_samples
from test_ff_reference import _inputs as wall_inputs, _options as wall_options
from test_tmd_correlator import _axial_inputs


def _options(rest=False):
    return dict(
        fit_scope=["qda"] if rest else ["2pt+qda"],
        qda_output="amplitude" if rest else "tmdwf",
        components="real",
        tmin=3,
        tmax=11,
        qda_window={"tmin": 5, "tmax": 10},
        n_states={"qda": 1} if rest else {"2pt": 1, "qda": 1},
        prior_width=1.0,
        svdcut=1e-7,
        posterior_prior_error_scale=10.0,
        fit_samples=False,
        tune_z=2,
    )


@pytest.mark.parametrize("rest", [False, True])
@pytest.mark.parametrize("raw_scale", [1e-8, 1.0, 1e8])
def test_native_wf_defaults_are_exact_and_keep_center_posterior_mode(monkeypatch, rest, raw_scale):
    data, _ = _axial_inputs(rest=rest, raw_scale=raw_scale)
    calls = []
    actual = _tmdwf.nonlinear_fit

    def capture(data, fcn, prior, **kwargs):
        key = "amplitude_re" if rest else "log(overlap0)"
        assert gv.mean(prior[key]) == (0.0 if rest else -1.0)
        assert gv.sdev(prior[key]) == (1.0 if rest else 3.0)
        if not rest:
            assert gv.mean(prior["phi0_re"]) == 0.0
            assert gv.sdev(prior["phi0_re"]) == 1.0
        assert kwargs["sample_prior_scale"] == 10.0
        calls.append(True)
        return actual(data, fcn, prior, **kwargs)

    monkeypatch.setattr(_tmdwf, "nonlinear_fit", capture)
    _, diagnostic = fit_qda_samples(data, **_options(rest))
    assert calls == [True]
    assert diagnostic["sample_prior_mode"] == "center_posterior"


@pytest.mark.parametrize("rest", [False, True])
def test_authored_wf_priors_condition_raw_units_once_and_original_mode_disables_widening(monkeypatch, rest):
    data, _ = _axial_inputs(rest=rest)
    key = "amplitude_re" if rest else "log(overlap0)"
    mean, width = (2e-12, 1e-12) if rest else (np.log(2e-10), 0.2)
    priors = {key: f"{float(mean)!r} +- {float(width)!r}"}
    if not rest:
        priors["phi0_re"] = "0.5(0.2)"
    captured = []
    actual = _tmdwf.nonlinear_fit

    def capture(data, fcn, prior, **kwargs):
        captured.append(prior)
        assert kwargs["sample_prior_scale"] is None
        return actual(data, fcn, prior, **kwargs)

    monkeypatch.setattr(_tmdwf, "nonlinear_fit", capture)
    _, diagnostic = fit_qda_samples(
        data, **{**_options(rest), "prior_width": 2.0}, priors=priors, sample_prior_mode="original"
    )
    scale = diagnostic["qda_conditioning"]
    prior = captured[0]
    assert gv.mean(prior[key]) == pytest.approx(mean * scale if rest else mean + np.log(scale))
    assert gv.sdev(prior[key]) == pytest.approx(2 * width * scale if rest else 2 * width)
    if not rest:
        assert gv.mean(prior["phi0_re"]) == 0.5
        assert gv.sdev(prior["phi0_re"]) == 0.4
    assert diagnostic["fits"][0]["effective_priors"][key]["sdev"] == gv.sdev(prior[key])


@pytest.mark.parametrize("mode, expected_scale", [("original", None), ("center_posterior", 3.0)])
def test_wall_ff_overrides_and_common_replica_mode_use_independent_width_factors(monkeypatch, mode, expected_scale):
    data = wall_inputs("bootstrap")
    calls = []
    actual = _fit_ff.nonlinear_fit

    def capture(data, fcn, prior, **kwargs):
        calls.append(prior)
        assert kwargs["sample_prior_scale"] == expected_scale
        return actual(data, fcn, prior, **kwargs)

    monkeypatch.setattr(_fit_ff, "nonlinear_fit", capture)
    _, diagnostics = _fit_ff.fit_ff_samples(
        data,
        **{
            **wall_options(),
            "fit_samples": False,
            "prior_width": 2.0,
            "posterior_prior_error_scale": 3.0,
            "sample_prior_mode": mode,
            "priors": {"ff": "-1e-8 +- 2e-8", "c1": "-0.4(0.2)"},
        },
    )
    assert len(calls) == 4
    for prior, record in zip(calls, diagnostics["fits"], strict=True):
        assert gv.mean(prior["ff"]) == pytest.approx(-1e-8 * record["conditioning"])
        assert gv.sdev(prior["ff"]) == pytest.approx(4e-8 * record["conditioning"])
        assert gv.mean(prior["c1"]) == -0.4
        assert gv.sdev(prior["c1"]) == 0.4


def test_tmd_prior_names_preserve_native_models_and_ignore_unused_parameters():
    base = {"component": "both", "fitting_form": "Breit", "fit_scope": ["2pt+qda"], "nstate": {"2pt": [2], "qda": [2]}}
    assert declared_prior_names({**base, "qda_output": "tmdwf"}) == {
        "log(E0)",
        "log(dE1)",
        "log(overlap0)",
        "log(overlap1)",
        "phi0_re",
        "phi0_im",
        "phi1_re",
        "phi1_im",
    }
    assert declared_prior_names({**base, "fit_scope": ["ff"], "nstate": {"ff": [2]}}) == {"ff", "log(dE)", "c1", "c2"}
    data, _ = _axial_inputs()
    baseline, baseline_diagnostic = fit_qda_samples(data, **_options())
    result, diagnostic = fit_qda_samples(data, **_options(), priors={"overlap0": "2e-10 +- 1e-10"})
    assert result is baseline is None
    for key in ("value", "chi2", "Q", "effective_priors"):
        if key in diagnostic["fits"][0]:
            assert diagnostic["fits"][0][key] == baseline_diagnostic["fits"][0][key]
    assert diagnostic["fits"][0]["inactive_prior_overrides"] == ["overlap0"]


def test_rotated_axial_publication_retains_tmd_identity_and_review_fit_quality(tmp_path):
    data, _ = _axial_inputs(rotated=True)
    settings = {
        "component": "re",
        "fitting_form": "Breit",
        "fit_scope": ["2pt+qda"],
        "nstate": {"2pt": [1], "qda": [1]},
        "qda_output": "tmdwf",
        "qda_window": {"tmin": 5, "tmax": 10},
        "tmd_phase_rotation": True,
        "svdcut": 1e-7,
        "posterior_prior_error_scale": 10.0,
        "sample_prior_mode": "original",
        "priors": {"phi0_re": "0.5(0.5)"},
    }
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "covariance", "target_observable": "tmd"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "wf",
        settings,
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
    )
    candidate = {"id": "wf", "window": {"tmin": 3, "tmax": 11}, "nstate": {"2pt": 1, "qda": 1}, "prior_width": 1.0}
    result, _, diagnostic = _publish._apply_qda_candidate(context, candidate, settings=settings, correlators=data)
    assert result.attrs["observable"] == "tmdwf"
    assert result.attrs["target_observable"] == "tmd"
    assert result.attrs["phase_rotation_applied"] == "true"
    assert result.attrs["sample_prior_mode"] == "original"
    context.output = result
    evidence = _review_summary(context, {"diagnostics": {"selected_application_fit": diagnostic}})
    assert evidence["tmd_diagnostics"]["center_fit_count"] == 2
    assert evidence["tmd_diagnostics"]["minimum_Q"] == min(fit["Q"] for fit in diagnostic["fits"])
    assert evidence["physical_metadata"]["target_observable"] == "tmd"
