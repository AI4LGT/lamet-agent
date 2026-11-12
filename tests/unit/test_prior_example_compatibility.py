"""Freeze the prior behavior of the four existing PDF/DA example manifests.

These tests need no external example datasets or historical git checkout. The
legacy Gaussian values and replica factors are recorded explicitly below.
"""

import copy
import json
from pathlib import Path

import gvar as gv
import numpy as np
import pytest

from lamet_agent.agent import ToolContext
from lamet_agent.contract import evaluate_rules
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.stages.correlator_analysis import _fit_matrix, _fit_qda, physics
from lamet_agent.stages.correlator_analysis.contract import PARAM_RULES
import lamet_agent.parallel._pool as pool_module
from lamet_agent.parallel._pool import _ParallelPool


_EXAMPLES = ("pion_pdf_cg", "pion_pdf_gi", "pion_da_gi", "kaon_da_gi")
_ROOT = Path(__file__).resolve().parents[2]
_PDF_PRIORS = {
    "log(E0)": (0.0, 3.0),
    "log(dE1)": (0.0, 1.0),
    "z0": (1.0, 10.0),
    "z1": (1.0 / 3.0, 10.0 * (1.0 / 3.0)),
    "O00_re": (1.0, 10.0),
    "O00_im": (1.0, 10.0),
    "O01_re": (1.0, 10.0),
    "O01_im": (1.0, 10.0),
    "O11_re": (1.0, 10.0),
    "O11_im": (1.0, 10.0),
}
_DA_PRIORS = {
    "log(E0)": (0.0, 3.0),
    "z0": (1.0, 10.0),
    "zprime0": (1.0, 10.0),
    "O00_re": (1.0, 10.0),
    "O00_im": (0.0, 10.0),
}


def _jobs():
    jobs = []
    for example in _EXAMPLES:
        document = json.loads((_ROOT / "examples" / f"{example}_manifest.json").read_text())
        block = document["stages"]["correlator_analysis"]
        for job in block["jobs"]:
            params = {**copy.deepcopy(block["defaults"]), **copy.deepcopy(job)}
            params.pop("inputs")
            job_id = params.pop("id")
            jobs.append(pytest.param(example, job_id, params, id=f"{example}-{job_id}"))
    return jobs


def _resolved(params):
    params = copy.deepcopy(params)
    assert evaluate_rules(params, PARAM_RULES) == []
    assert params["prior_width"] == [1.0]
    assert params["priors"] == {}
    assert params["sample_prior_mode"] == "center_posterior"
    assert params.get("chained_prior_error_scale") is None
    return params


def _data(example):
    """Generate stable positive samples covering every authored example window."""
    ensemble = EnsembleInfo("prior", "prior", 0.1, 0.1, 32, 96, 0.14)
    t = np.arange(32)
    tseps = [8, 10, 12]
    tau = np.arange(13)
    rng = np.random.default_rng(1703)
    two, three, qda = [], [], []
    for _ in range(16):
        energy = 0.23 + rng.normal(0, 0.001)
        overlap = 1.1 + rng.normal(0, 0.005)
        c2 = overlap**2 / (2 * energy) * (np.exp(-energy * t) + np.exp(-energy * (96 - t)))
        c2 += 0.3**2 / (2 * (energy + 0.45)) * np.exp(-(energy + 0.45) * t)
        c2 *= 1 + rng.normal(0, 0.001, len(t))
        two.append(c2)
        c3 = np.zeros((len(tseps), len(tau), 1))
        for index, tsep in enumerate(tseps):
            c3[index, : tsep + 1, 0] = c2[tsep] * (0.65 / (2 * energy) + rng.normal(0, 0.002, tsep + 1))
        three.append(c3)
        denominator = 1e-8 * c2
        ratio = 0.72 + 0.13j + rng.normal(0, 0.001, len(t)) + 1j * rng.normal(0, 0.001, len(t))
        qda.append(np.column_stack([denominator, denominator * ratio]))
    attrs = {"source_momentum": "[0,0,1]", "sink_momentum": "[0,0,1]", "resample_id": "prior"}
    if "pdf" in example:
        return {
            "c2": EnsembleData(
                ensemble, "bootstrap", two, ["t"], {"t": t.tolist()}, attrs={**attrs, "correlator_type": "two_point"}
            ),
            "c3": EnsembleData(
                ensemble,
                "bootstrap",
                three,
                ["tsep", "tau", "z"],
                {"tsep": tseps, "tau": tau.tolist(), "z": [1]},
                attrs={**attrs, "correlator_type": "three_point"},
            ),
        }
    return {
        "qda": EnsembleData(
            ensemble,
            "bootstrap",
            qda,
            ["t", "z"],
            {"t": t.tolist(), "z": [0, 1]},
            attrs={**attrs, "correlator_type": "qda"},
        )
    }


def _assert_legacy_prior(prior, expected):
    assert set(prior.keys()) == set(expected)
    for key, (mean, sdev) in expected.items():
        # Exact comparisons catch avoidable conditioning round trips as well as
        # changes to a Gaussian mean, width, or inactive component's parameter.
        assert gv.mean(prior[key]) == mean, key
        assert gv.sdev(prior[key]) == sdev, key


@pytest.mark.parametrize("example, job_id, authored", _jobs())
def test_all_22_example_jobs_keep_legacy_candidate_priors_and_replica_scale(
    tmp_path, monkeypatch, example, job_id, authored
):
    params = _resolved(authored)
    is_pdf = "pdf" in example
    expected = _PDF_PRIORS if is_pdf else _DA_PRIORS
    expected_scale = 10.0 if is_pdf else 3.0
    assert params["posterior_prior_error_scale"] == expected_scale
    assert {count for counts in params["nstate"].values() for count in counts} == ({2} if is_pdf else {1})
    context = ToolContext(
        {"metadata": {"workers": 1, "sample_error_mode": "variance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        job_id,
        params,
        {},
        {},
        {"correlators": _data(example), "correlator_rescale": 1e5},
        tmp_path,
        np.random.default_rng(1),
    )
    calls = []
    actual = physics.nonlinear_fit

    def capture(data, fcn, prior, **kwargs):
        _assert_legacy_prior(prior, expected)
        assert kwargs["sample_prior_scale"] == expected_scale
        assert kwargs["sample_priors"] is None
        calls.append(tuple(data[0]["atoms"]))
        return actual(data, fcn, prior, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    (_fit_matrix if is_pdf else _fit_qda).run(context, tune_z_values=[1])
    expected_windows = len(params["pt2_windows"]) * (len(params["pt3_windows"]) if is_pdf else 1)
    assert len(calls) == expected_windows
    assert len(context.state["matrix_element_candidates"]) == expected_windows


@pytest.mark.parametrize("scale", [1e-21, 1e-8, 1.0, 1e5, 1e24])
@pytest.mark.parametrize("qda", [False, True])
def test_unmodified_base_priors_are_exact_at_every_conditioning_scale(scale, qda):
    expected = _DA_PRIORS if qda else _PDF_PRIORS
    prior = physics.matrix_element_prior(
        1 if qda else 2,
        form="Breit",
        scope=["2pt", "qda"] if qda else ["2pt", "3pt_over_2pt"],
        components=("re", "im"),
        width_scale=1.0,
        denominator_kind="qda_z0" if qda else "external_2pt",
        conditioning={key: np.sqrt(scale) for key in expected if key.startswith(("z", "O") if qda else ("z",))},
    )
    _assert_legacy_prior(prior, expected)


@pytest.mark.parametrize("example", _EXAMPLES)
def test_example_replica_outputs_match_frozen_legacy_prior_recipe(monkeypatch, example):
    first_job = next(item.values[2] for item in _jobs() if item.values[0] == example)
    params = _resolved(first_job)
    pdf = "pdf" in example
    window = params["pt2_windows"][0]
    kwargs = dict(
        fit_scope=params["fit_scope"],
        components={"re": "real", "both": "both"}[params["component"]],
        tmin=window["tmin"],
        tmax=window["tmax"],
        n_states={key: values[0] for key, values in params["nstate"].items()},
        prior_width=params["prior_width"][0],
        svdcut=params["svdcut"],
        posterior_prior_error_scale=params["posterior_prior_error_scale"],
        workers=1,
        sample_error_mode="variance",
        priors=params["priors"],
        sample_prior_mode=params["sample_prior_mode"],
        chained_prior_error_scale=params.get("chained_prior_error_scale"),
    )
    if pdf:
        three = params["pt3_windows"][0]
        kwargs.update(
            fitting_form=params["fitting_form"],
            tsep_values=three["tsep_ls"],
            tau_min=three["tau_cut"],
            correlator_rescale=1e5,
        )
    fit = physics.fit_matrix_element_samples if pdf else physics.fit_qda_samples
    records = _data(example)
    current, diagnostics = fit(records, **kwargs)
    actual = physics.nonlinear_fit
    expected = _PDF_PRIORS if pdf else _DA_PRIORS

    def legacy_recipe(data, fcn, prior, **fit_kwargs):
        # The old width-one recipe used these exact conditioned Gaussians and
        # a 10x (PDF) or 3x (DA) center-posterior replica prior.
        frozen = gv.BufferDict({key: gv.gvar(*values) for key, values in expected.items()})
        fit_kwargs["sample_prior_scale"] = 10.0 if pdf else 3.0
        return actual(data, fcn, frozen, **fit_kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", legacy_recipe)
    legacy, baseline = fit(records, **kwargs)
    np.testing.assert_allclose(current.values, legacy.values, rtol=1e-10, atol=1e-12)
    assert diagnostics["min_Q"] == pytest.approx(baseline["min_Q"], rel=1e-10, abs=1e-12)


@pytest.mark.parametrize("example", _EXAMPLES)
def test_example_replica_outputs_survive_worker_pool_recycling(monkeypatch, example):
    params = _resolved(next(item.values[2] for item in _jobs() if item.values[0] == example))
    pdf = "pdf" in example
    window = params["pt2_windows"][0]
    kwargs = dict(
        fit_scope=params["fit_scope"],
        components={"re": "real", "both": "both"}[params["component"]],
        tmin=window["tmin"],
        tmax=window["tmax"],
        n_states={key: values[0] for key, values in params["nstate"].items()},
        prior_width=params["prior_width"][0],
        svdcut=params["svdcut"],
        posterior_prior_error_scale=params["posterior_prior_error_scale"],
        workers=2,
        sample_error_mode="variance",
        priors=params["priors"],
        sample_prior_mode=params["sample_prior_mode"],
        chained_prior_error_scale=params.get("chained_prior_error_scale"),
    )
    if pdf:
        three = params["pt3_windows"][0]
        kwargs.update(
            fitting_form=params["fitting_form"],
            tsep_values=three["tsep_ls"],
            tau_min=three["tau_cut"],
            correlator_rescale=1e5,
        )
    fit = physics.fit_matrix_element_samples if pdf else physics.fit_qda_samples
    records = _data(example)
    monkeypatch.setattr(pool_module, "_MAX_MAPS_PER_EXECUTOR", 128)
    with _ParallelPool(2) as pool:
        persistent, baseline = fit(records, _parallel=pool, **kwargs)
        assert pool._executor is not None
    monkeypatch.setattr(pool_module, "_MAX_MAPS_PER_EXECUTOR", 1)
    with _ParallelPool(2) as pool:
        recycled, diagnostics = fit(records, _parallel=pool, **kwargs)
        assert pool._executor is None
    np.testing.assert_array_equal(recycled.values, persistent.values)
    assert diagnostics["min_Q"] == baseline["min_Q"]
