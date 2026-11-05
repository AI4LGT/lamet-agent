"""Verify authored priors survive fitting, conditioning, retries, and publication."""

from types import SimpleNamespace
import json

import gvar as gv
import numpy as np
import pytest

from lamet_agent.agent import ToolContext
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.stages.correlator_analysis import physics


def _correlators():
    ensemble = EnsembleInfo("toy", "toy", 0.1, 0.1, 32, 32, 0.2)
    times = np.arange(14)
    tseps = np.asarray([8, 10])
    tau = np.arange(11)
    rng = np.random.default_rng(881)
    c2, c3, qda = [], [], []
    for _ in range(16):
        energy, overlap = 0.3 + rng.normal(0, 0.002), 1.2 + rng.normal(0, 0.01)
        two = overlap**2 / (2 * energy) * (np.exp(-energy * times) + np.exp(-energy * (32 - times)))
        two += rng.normal(0, 1e-4, len(times))
        three = np.zeros((2, 11, 1))
        for i, tsep in enumerate(tseps):
            three[i, : tsep + 1, 0] = two[tsep] * (0.7 / (2 * energy) + rng.normal(0, 0.003, tsep + 1))
        c2.append(two)
        c3.append(three)
        qda.append((two * (0.6 + rng.normal(0, 0.003, len(times))))[:, None])
    attrs = {"source_momentum": "[1,0,0]", "sink_momentum": "[1,0,0]", "resample_id": "shared"}
    return {
        "c2": EnsembleData(
            ensemble, "bootstrap", c2, ["t"], {"t": times.tolist()}, attrs={**attrs, "correlator_type": "two_point"}
        ),
        "c3": EnsembleData(
            ensemble,
            "bootstrap",
            c3,
            ["tsep", "tau", "z"],
            {"tsep": tseps.tolist(), "tau": tau.tolist(), "z": [1]},
            attrs={**attrs, "correlator_type": "three_point"},
        ),
        "qda": EnsembleData(
            ensemble,
            "bootstrap",
            qda,
            ["t", "z"],
            {"t": times.tolist(), "z": [1]},
            attrs={**attrs, "correlator_type": "qda"},
        ),
    }


def _context(tmp_path, params, correlators):
    return ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "variance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "prior_test",
        params,
        {},
        {},
        {"correlators": correlators, "correlator_rescale": 100.0},
        tmp_path,
        np.random.default_rng(4),
    )


def _settings(qda=False):
    scope = ["2pt+qda"] if qda else ["2pt+3pt"]
    return {
        "analysis_method": "lsqfit",
        "fit_scope": scope,
        "fitting_form": "Breit",
        "component": "re",
        "nstate": {"2pt": [1], "qda" if qda else "3pt": [1]},
        "pt2_windows": [{"tmin": 2, "tmax": 7}],
        "pt3_windows": [{"tsep_ls": [8, 10], "tau_cut": 2}],
        "prior_width": [2.0],
        "posterior_prior_error_scale": 4.0,
        "chained_prior_error_scale": 1.5,
        "sample_prior_mode": "original",
        "svdcut": 1e-8,
        "q_min": 0.0,
        "model_average": False,
        "priors": {"z0": {"mean": 1.2, "sdev": 0.2}, "O00_re": {"mean": 0.7, "sdev": 0.3}},
    }


@pytest.mark.parametrize("qda", [False, True])
def test_authored_priors_reach_tuning_and_final_application_in_physical_units(tmp_path, monkeypatch, qda):
    from lamet_agent.stages.correlator_analysis import _fit_matrix, _fit_qda, _publish

    records = _correlators()
    records = {key: value for key, value in records.items() if key in (("c2", "qda") if qda else ("c2", "c3"))}
    settings = _settings(qda)
    context = _context(tmp_path, settings, records)
    calls = []
    actual = physics.nonlinear_fit

    def capture(data, fcn, prior, **kwargs):
        calls.append((gv.mean(prior), gv.sdev(prior), kwargs.get("sample_prior_scale")))
        return actual(data, fcn, prior, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    (_fit_qda if qda else _fit_matrix).run(context, tune_z_values=[1])
    candidate = context.state["matrix_element_candidates"][0]
    apply = _publish._apply_qda_candidate if qda else _publish._apply_ordinary_candidate
    data, _, diagnostics = apply(context, candidate, settings=settings, correlators=records)
    assert data is not None
    scale = physics._qda_correlator_rescale(records["c2"].values[:, 2:7]) if qda else 100.0
    for means, widths, sample_scale in calls:
        assert means["z0"] == pytest.approx(1.2 * np.sqrt(scale))
        assert widths["z0"] == pytest.approx(0.2 * 2 * np.sqrt(scale))
        matrix_scale = np.sqrt(scale) if qda else 1.0
        assert means["O00_re"] == pytest.approx(0.7 * matrix_scale)
        assert widths["O00_re"] == pytest.approx(0.3 * 2 * matrix_scale)
        assert sample_scale is None
    recorded = diagnostics["fits"][0]["stages"][0]["effective_priors"]
    assert recorded["z0"]["sdev"] == pytest.approx(0.4 * np.sqrt(scale))
    assert diagnostics["sample_prior_mode"] == "original"


@pytest.mark.parametrize("chain_scale", [None, 1.5])
def test_chained_transfer_and_replica_widths_are_independent_of_prior_width(monkeypatch, chain_scale):
    records = {key: value for key, value in _correlators().items() if key != "qda"}
    calls = []
    actual = physics.nonlinear_fit

    def capture(data, fcn, prior, **kwargs):
        result = actual(data, fcn, prior, **kwargs)
        calls.append((prior, kwargs, result))
        return result

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    physics.fit_matrix_element_samples(
        records,
        fitting_form="Breit",
        fit_scope=["2pt", "3pt_over_2pt"],
        components="real",
        tmin=2,
        tmax=7,
        tsep_values=[8, 10],
        tau_min=2,
        n_states=1,
        prior_width=2.0,
        correlator_rescale=100.0,
        svdcut=1e-8,
        posterior_prior_error_scale=4.0,
        chained_prior_error_scale=chain_scale,
        workers=1,
        sample_error_mode="variance",
        priors={"z0": {"mean": 1.2, "sdev": 0.2}},
    )
    assert len(calls) == 2
    assert calls[0][1]["sample_prior_scale"] == 4.0  # never multiplied by prior_width again
    assert calls[1][1]["sample_prior_scale"] is None
    expected_scale = 4.0 if chain_scale is None else chain_scale
    assert gv.sdev(calls[1][0]["z0"]) == pytest.approx(gv.sdev(calls[0][2].p["z0"]) * expected_scale)
    for prior, posterior in zip(
        calls[1][1]["sample_priors"], physics._sample_posteriors_from_result(calls[0][2]), strict=True
    ):
        assert gv.mean(prior["z0"]) == pytest.approx(gv.mean(posterior["z0"]))
        assert gv.sdev(prior["z0"]) == pytest.approx(gv.sdev(posterior["z0"]) * expected_scale)


def test_nonbreit_overlap_conditioning_keeps_physical_matrix_prior():
    prior = physics.matrix_element_prior(
        1,
        form="NonBreit",
        scope=["2pt", "3pt"],
        components=("re",),
        width_scale=2.0,
        priors={
            "z0_i": {"mean": 0.8, "sdev": 0.2},
            "z0_f": {"mean": 0.9, "sdev": 0.1},
            "O00_re": {"mean": 0.6, "sdev": 0.3},
        },
        conditioning={"z0_i": 10.0, "z0_f": 10.0},
    )
    assert gv.mean(prior["z0_i"]) == 8.0
    assert gv.sdev(prior["z0_f"]) == 2.0
    assert gv.mean(prior["O00_re"]) == 0.6
    assert gv.sdev(prior["O00_re"]) == 0.6


def test_public_qda_wrapper_routes_authored_priors_and_sample_mode(monkeypatch):
    records = {key: value for key, value in _correlators().items() if key != "c3"}
    settings = _settings(True)
    calls = []
    actual = physics.nonlinear_fit

    def capture(*args, **kwargs):
        calls.append(kwargs["sample_prior_scale"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    _, _, diagnostics = physics.matrix_element_samples(
        records,
        method="qda",
        tmin=2,
        tmax=7,
        tau_min=None,
        lsqfit=settings,
        fit_scope=["2pt+qda"],
        n_states=1,
        prior_width=2.0,
        tune_z=1,
        fit_samples=False,
        sample_error_mode="variance",
    )
    assert calls == [None]
    assert diagnostics["chained_prior_error_scale"] == 1.5
    effective = diagnostics["fits"][0]["stages"][0]["effective_priors"]
    assert effective["O00_re"]["sdev"] / effective["O00_re"]["mean"] == pytest.approx(0.6 / 0.7)


def _spectrum_settings():
    return {
        "analysis_method": "lsqfit",
        "fit_scope": ["2pt"],
        "fitting_form": "Breit",
        "component": "re",
        "nstate": {"2pt": [1]},
        "pt2_windows": [{"tmin": 2, "tmax": 7}],
        "q_min": 0.0,
        "prior_width": [2.0],
        "sample_prior_mode": "center_posterior",
        "posterior_prior_error_scale": 4.0,
        "priors": {"E0": {"mean": 0.3, "sdev": 0.05}},
        "model_average": False,
    }


def test_direct_spectrum_overrides_recommended_priors_and_persists_diagnostics(tmp_path, monkeypatch):
    from lamet_agent.stages.correlator_analysis import _fit_spectrum, _publish

    context = _context(tmp_path, _spectrum_settings(), {"c2": _correlators()["c2"]})
    calls = []
    actual = physics.nonlinear_fit

    def capture(*args, **kwargs):
        calls.append(kwargs["sample_prior_scale"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    _fit_spectrum.run(
        context,
        tmin=2,
        tmax=7,
        n_states=1,
        prior_width=2.0,
        prior_means={"E0": 0.8, "z0": 1.2},
        prior_widths={"E0": 0.4, "z0": 0.2},
    )
    assert calls == [4.0]
    candidate = context.state["spectrum_candidates"][0]
    candidate["plot_artifact"] = "already-plotted.pdf"
    monkeypatch.setattr(ToolContext, "finish", lambda *args: None)
    _publish.run(context, candidate_id=candidate["id"])
    recorded = json.loads((tmp_path / "diagnostics/candidates.json").read_text())["candidates"][0]
    assert recorded["effective_priors"]["E0"] == {"mean": 0.3, "sdev": 0.1}
    assert recorded["effective_priors"]["z0"] == {"mean": 1.2, "sdev": 0.4}
    assert recorded["fit_priors"]["log(E0)"]["sdev"] == pytest.approx(1 / 3)
    assert recorded["sample_prior_mode"] == "center_posterior"
    assert recorded["nstate"] == {"2pt": 1}
    assert recorded["inactive_prior_overrides"] == []


def test_complete_spectrum_priors_bypass_model_for_full_authored_grid(tmp_path):
    from lamet_agent.stages.correlator_analysis.ask import initial

    params = _spectrum_settings()
    params.update(
        nstate={"2pt": [1, 2]}, prior_width=[1.0, 2.0], pt2_windows=[{"tmin": 2, "tmax": 7}, {"tmin": 3, "tmax": 8}]
    )
    params["priors"].update(z0={"mean": 1.2, "sdev": 0.2}, E1={"mean": 0.7, "sdev": 0.1}, z1={"mean": 0.4, "sdev": 0.2})
    context = _context(tmp_path, params, {})
    suggestion = initial(context, object())  # has no model methods: any model call fails
    assert len(suggestion["candidates"]) == 8
    assert {(item["tmin"], item["n_states"], item["prior_width"]) for item in suggestion["candidates"]} == {
        (time, count, width) for time in (2, 3) for count in (1, 2) for width in (1.0, 2.0)
    }
    assert context.state["fit_parameter_provenance"][0]["source"] == "manifest"


def test_spectrum_recommendations_and_retries_scan_authored_widths(tmp_path, monkeypatch):
    from lamet_agent.stages.correlator_analysis import workflow

    settings = _spectrum_settings()
    settings.update(prior_width=[1.0, 2.0], q_min=0.05)
    context = _context(tmp_path, settings, {})
    session = SimpleNamespace(recommendation_calls=0, max_recommendation_calls=2)
    calls = []

    def recommend(context, session, *_args):
        session.recommendation_calls += 1
        return {
            "tmin": 1 + session.recommendation_calls,
            "tmax": 7,
            "n_states": 1,
            "prior_means": {"E0": 0.9, "z0": 1.2},
            "prior_widths": {"E0": 0.2, "z0": 0.3},
        }

    def fit(context, **kwargs):
        calls.append((kwargs["tmin"], kwargs["prior_width"], context.params["priors"]["E0"]))
        candidate = {
            "id": f"spectrum_{len(calls):03d}",
            "window": {"tmin": kwargs["tmin"], "tmax": 7},
            "Q": 0.01 if len(calls) <= 2 else 0.9,
            "chi2_dof": 1.0,
        }
        context.state.setdefault("spectrum_candidates", []).append(candidate)
        return {"metrics": candidate}

    monkeypatch.setattr(workflow, "inspect", lambda _: None)
    monkeypatch.setattr(workflow, "initial", recommend)
    monkeypatch.setattr(workflow, "revise", recommend)
    monkeypatch.setattr(workflow, "fit_spectrum", fit)
    monkeypatch.setattr(workflow, "publish", lambda *args, **kwargs: None)
    workflow.run(context, session)
    assert [(time, width) for time, width, _ in calls] == [(2, 1.0), (2, 2.0), (3, 1.0), (3, 2.0)]
    assert all(prior == {"mean": 0.3, "sdev": 0.05} for _, _, prior in calls)


@pytest.mark.parametrize(
    "override",
    [
        {"E1": {"mean": 0.7, "sdev": 0.1}},
        {"typo": {"mean": 0.3, "sdev": 0.1}},
        {"E0": {"mean": 0.3, "sdev": float("nan")}},
        {"E0": {"mean": 0.3, "sdev": 0}},
    ],
)
def test_direct_spectrum_rejects_wrong_state_unknown_and_invalid_width(override):
    source = _correlators()["c2"]
    with pytest.raises(ValueError):
        physics.fit_spectrum_samples(
            source.values[:, 2:7],
            np.arange(2, 7),
            1,
            extent=32,
            resample="bootstrap",
            prior_means={"E0": 0.3, "z0": 1.2},
            prior_widths={"E0": 0.1, "z0": 0.2},
            priors=override,
        )


def test_log_conditioning_preserves_default_prior_and_transforms_authored_mean():
    kwargs = dict(n_states=1, form="Breit", scope=["2pt"], components=("re",), width_scale=1.0)
    baseline = physics.matrix_element_prior(**kwargs)
    conditioned = physics.matrix_element_prior(**kwargs, conditioning={"log(E0)": 2.0})
    assert gv.mean(conditioned["log(E0)"]) == pytest.approx(gv.mean(baseline["log(E0)"]))
    assert gv.sdev(conditioned["log(E0)"]) == pytest.approx(gv.sdev(baseline["log(E0)"]))
    authored = physics.matrix_element_prior(
        **kwargs, conditioning={"log(E0)": 2.0}, priors={"log(E0)": {"mean": -1.0, "sdev": 0.2}}
    )
    assert gv.mean(authored["log(E0)"]) == pytest.approx(-1.0 + np.log(2.0))
    assert gv.sdev(authored["log(E0)"]) == pytest.approx(0.2)


def test_qda_rejects_local_denominator_priors_with_explicit_two_point_data():
    records = {key: value for key, value in _correlators().items() if key != "c3"}
    with pytest.raises(ValueError, match="unknown prior parameter 'zprime0'"):
        physics.fit_qda_samples(
            records,
            fit_scope=["2pt+qda"],
            components="real",
            tmin=2,
            tmax=7,
            n_states=1,
            prior_width=1.0,
            svdcut=1e-8,
            posterior_prior_error_scale=4.0,
            priors={"zprime0": {"mean": 1.2, "sdev": 0.1}},
            prior_allowed_names={"log(E0)", "z0", "zprime0", "O00_re", "O00_im"},
        )


def test_direct_python_spectrum_api_preserves_original_prior_default(monkeypatch):
    source = _correlators()["c2"]
    calls = []
    actual = physics.nonlinear_fit

    def capture(*args, **kwargs):
        calls.append(kwargs["sample_prior_scale"])
        return actual(*args, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    _, diagnostics = physics.fit_spectrum_samples(
        source.values[:, 2:7],
        np.arange(2, 7),
        1,
        extent=32,
        resample="bootstrap",
        prior_means={"E0": 0.3, "z0": 1.2},
        prior_widths={"E0": 0.1, "z0": 0.2},
    )
    assert calls == [None]
    assert diagnostics["sample_prior_mode"] == "original"
