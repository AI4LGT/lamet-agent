"""Physical amplitude, wall normalization, and offline TMD fitting regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.stages.correlator_analysis._fit_ff import CURRENTS, fit_ff_samples, wall_ratios
from lamet_agent.stages.correlator_analysis.ask import initial
from lamet_agent.stages.correlator_analysis.physics import fit_qda_samples
from lamet_agent.stages.correlator_analysis.workflow import _can_revise


def _ensemble():
    return EnsembleInfo("test", "axial", 0.06, 0.06, 48, 64, 0.67)


def _axial_inputs(*, rest=False, raw_scale=1.0, rotated=False):
    rng = np.random.default_rng(71)
    n = 48
    t, z = np.arange(20), np.array([0, 2])
    energy, overlap, phi = 0.4, 2e-10, np.array([0.8, 0.5])
    periodic = np.exp(-energy * t) + np.exp(-energy * (64 - t))
    q = periodic[:, None] * (3e-12 * phi if rest else overlap * phi / 2)
    q = np.broadcast_to(q, (n, *q.shape)).copy() * (1 + 0.008 * rng.normal(size=(n, len(t), len(z))))
    momentum = [0, 0, 0] if rest else [8, 0, 0]
    if rotated:
        q = q * 1j * np.exp(1j * 8 * 2 * np.pi / 48 * z / 2)[None, None, :]
    attrs = {
        "resample_id": "aligned",
        "source_momentum": json.dumps(momentum),
        "sink_momentum": json.dumps(momentum),
        "kernel_operator": "g7" if rest else "g14",
        "longitudinal_direction": "[1,0,0]",
    }
    qdata = EnsembleData(
        _ensemble(),
        "bootstrap",
        list(q * np.sqrt(raw_scale)),
        ["t", "z"],
        {"t": list(t), "z": list(z)},
        attrs={**attrs, "correlator_type": "qda"},
    )
    data = {"q": qdata}
    if not rest:
        ss = overlap**2 / (2 * energy) * periodic
        ss = ss[None, :] * (1 + 0.008 * rng.normal(size=(n, len(t))))
        data["ss"] = EnsembleData(
            _ensemble(),
            "bootstrap",
            list(ss * raw_scale),
            ["t"],
            {"t": list(t)},
            attrs={**attrs, "correlator_type": "two_point"},
        )
    return data, phi


def _fit(data, **kwargs):
    return fit_qda_samples(
        data,
        components="real",
        tmin=3,
        tmax=11,
        prior_width=1.0,
        svdcut=1e-7,
        posterior_prior_error_scale=10.0,
        qda_window={"tmin": 5, "tmax": 10},
        **kwargs,
    )


def test_axial_amplitude_recovers_O00_over_E0_and_is_conditioning_invariant():
    data, phi = _axial_inputs(rotated=True)
    out, diag = _fit(
        data, fit_scope=["2pt+qda"], n_states={"2pt": 1, "qda": 1}, qda_output="tmdwf", tmd_phase_rotation=True
    )
    np.testing.assert_allclose(np.mean(out.values, axis=0), phi, rtol=0.015)
    assert out.attrs["amplitude_convention"] == "O00/E0"
    assert out.attrs["phase_rotation_applied"] == "true"
    assert diag["qda_conditioning"] ** 2 == pytest.approx(diag["pt2_conditioning"])
    scaled, _ = _axial_inputs(raw_scale=1e8, rotated=True)
    other, _ = _fit(
        scaled, fit_scope=["2pt+qda"], n_states={"2pt": 1, "qda": 1}, qda_output="tmdwf", tmd_phase_rotation=True
    )
    np.testing.assert_allclose(other.values, out.values, rtol=1e-6)
    data["q"].array.attrs["phase_rotation_applied"] = "true"
    with pytest.raises(ValueError, match="already rotated"):
        _fit(data, fit_scope=["2pt+qda"], n_states=1, qda_output="tmdwf", tmd_phase_rotation=True)


def test_rest_reference_is_fitted_absolute_amplitude_not_a_forced_unit_origin():
    data, phi = _axial_inputs(rest=True)
    out, diag = _fit(data, fit_scope=["qda"], n_states={"qda": 1}, qda_output="amplitude")
    np.testing.assert_allclose(np.mean(out.values, axis=0), 3e-12 * phi, rtol=0.02)
    assert out.attrs["amplitude_convention"] == "periodic_amplitude"
    assert diag["fits"][0]["E0"] == pytest.approx(0.4, rel=0.01)


def test_axial_fits_preserve_replica_coordinates_and_reject_mismatched_input_order():
    data, _ = _axial_inputs(rotated=True)
    for value in data.values():
        value.array = value.array.isel(resample=slice(None, None, -1))
    options = {"fit_scope": ["2pt+qda"], "n_states": 1, "qda_output": "tmdwf", "tmd_phase_rotation": True}
    result, _ = _fit(data, **options)
    np.testing.assert_array_equal(result.array.coords["resample"], data["q"].array.coords["resample"])
    data["ss"].array = data["ss"].array.isel(resample=slice(None, None, -1))
    with pytest.raises(ValueError, match="resampling"):
        _fit(data, **options)


def _wall_inputs():
    rng = np.random.default_rng(12)
    n, tsep, tau, t = 40, [6, 8], list(range(9)), list(range(10))
    attrs = {"resample_id": "wall", "bT_fm": 0.24, "source_momentum": "[0,0,6]", "sink_momentum": "[0,0,-6]"}
    den = 3 * np.exp(-0.5 * np.asarray(t))[None, :] * (1 + 0.01 * rng.normal(size=(n, len(t))))
    data = {}
    for current, fraction in [("gZ5", 0.7), ("gX5", 0.3)]:
        data[current] = EnsembleData(
            _ensemble(),
            "bootstrap",
            list(den * fraction),
            ["t"],
            {"t": t},
            attrs={**attrs, "kernel_operator": current, "bT_fm": 0.0},
        )
    for i, current in enumerate(CURRENTS):
        values = np.empty((n, len(tsep), len(tau)))
        for j, separation in enumerate(tsep):
            values[:, j, :] = (-2e-6 * (1 + 0.1 * i) * den[:, separation // 2] ** 2)[:, None]
        values *= 1 + 0.01 * rng.normal(size=values.shape)
        data[current] = EnsembleData(
            _ensemble(),
            "bootstrap",
            list(values),
            ["tsep", "tau"],
            {"tsep": tsep, "tau": tau},
            attrs={**attrs, "kernel_operator": current},
        )
    return data


def test_four_current_wall_fit_uses_midpoint_denominator_and_correct_normalization():
    data = _wall_inputs()
    source, _, _, ratios = wall_ratios(data, [6, 8], 3)
    np.testing.assert_allclose(np.mean(ratios[CURRENTS[0]]), -2e-6, rtol=0.005)
    result, diagnostic = fit_ff_samples(
        data,
        tsep_values=[6, 8],
        tau_cut=3,
        n_states=1,
        prior_width=1.0,
        energy_gev=2.67,
        svdcut=1e-7,
        posterior_prior_error_scale=10.0,
    )
    p = 6 * source.ensemble.k_s
    expected = 2e-6 * 1.15 * 6 * 48**3 / ((p**2 + 2.67**2) / (2 * p**2))
    assert float(np.mean(result.values)) == pytest.approx(expected, rel=0.005)
    assert result.dims == ["bT"] and result.coords["bT"] == [0.24]
    assert result.attrs["observable"] == "form_factor"
    assert len(diagnostic["fits"]) == 4
    assert all(len(f["sample_diagnostics"]) == 40 for f in diagnostic["fits"])
    data["gX5"].array.attrs["resample_id"] = "wrong"
    with pytest.raises(ValueError, match="resampling plan"):
        wall_ratios(data, [6, 8], 3)


def test_wall_fits_preserve_replica_coordinates_and_reject_mismatched_input_order():
    data = _wall_inputs()
    for value in data.values():
        value.array = value.array.isel(resample=slice(None, None, -1))
    result, _ = fit_ff_samples(data, tsep_values=[6, 8], tau_cut=3, n_states=1, prior_width=1.0,
                               energy_gev=2.67, svdcut=1e-7, posterior_prior_error_scale=10.0)
    np.testing.assert_array_equal(result.array.coords["resample"], data["gZ5"].array.coords["resample"])
    data["gX5"].array = data["gX5"].array.isel(resample=slice(None, None, -1))
    with pytest.raises(ValueError, match="resampling plan"):
        wall_ratios(data, [6, 8], 3)


def test_authored_windows_and_tuning_coordinates_skip_llm():
    context = SimpleNamespace(
        state={}, params={"fit_scope": ["2pt+qda"], "pt2_windows": [{"tmin": 3, "tmax": 11}], "tune_z_values": [0]}
    )
    # An object without any session methods proves no LLM/context operation occurs.
    selected = initial(context, object())
    assert selected["tune_z_values"] == [0]
    assert context.state["fit_parameter_provenance"][0]["source"] == "manifest"
    assert not _can_revise(SimpleNamespace(recommendation_calls=0, max_recommendation_calls=1), context)
    assert _can_revise(SimpleNamespace(recommendation_calls=0, max_recommendation_calls=2), context)


@pytest.mark.parametrize("fit_samples", [True, False])
def test_chained_spectrum_reused_locally_without_changing_any_z_fit(monkeypatch, fit_samples):
    """The shared SS fit runs once; every result matches independent one-z calls."""
    from lamet_agent.stages.correlator_analysis import physics

    rng = np.random.default_rng(52)
    t, tau, tseps, z = np.arange(16), np.arange(11), [8, 10], [0, 2, 4]
    energy, overlap = 0.4, 0.8
    ss = overlap**2 / (2 * energy) * (np.exp(-energy * t) + np.exp(-energy * (64 - t)))
    ss = ss[None, :] * (1 + 0.004 * rng.normal(size=(16, len(t))))
    ratio = np.array([0.8, 0.5, 0.2])[None, None, None, :] * (
        1 + 0.005 * rng.normal(size=(16, len(tseps), len(tau), len(z)))
    )
    three = ratio * ss[:, tseps, None, None]
    common = {"resample_id": "shared", "source_momentum": "[4,4,0]", "sink_momentum": "[4,4,0]"}
    data = {
        "ss": EnsembleData(
            _ensemble(), "bootstrap", list(ss), ["t"], {"t": list(t)}, attrs={**common, "correlator_type": "two_point"}
        ),
        "three": EnsembleData(
            _ensemble(),
            "bootstrap",
            list(three),
            ["tsep", "tau", "z"],
            {"tsep": tseps, "tau": list(tau), "z": z},
            attrs={**common, "correlator_type": "three_point"},
        ),
    }
    options = dict(
        fitting_form="Breit",
        fit_scope=["2pt", "3pt_over_2pt"],
        components="real",
        tmin=3,
        tmax=10,
        tsep_values=tseps,
        tau_min=3,
        n_states={"2pt": 1, "3pt_over_2pt": 1},
        prior_width=1.0,
        correlator_rescale=1.0,
        svdcut=1e-6,
        posterior_prior_error_scale=10.0,
        chained_prior_error_scale=1.0,
        workers=1,
        fit_samples=fit_samples,
    )
    calls = []
    original = physics.nonlinear_fit

    def spy(*args, **kwargs):
        calls.append(tuple(args[0][0]["atoms"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", spy)
    output, diagnostics = physics.fit_matrix_element_samples(data, **options)
    assert calls == [("2pt",), *([("3pt_over_2pt",)] * len(z))]
    expected_samples = []
    for i in range(len(z)):
        single = data["three"].copy()
        single.array = single.array.isel(z=[i])
        before = len(calls)
        independent, independent_diagnostic = physics.fit_matrix_element_samples(
            {"ss": data["ss"], "three": single}, **options
        )
        assert calls[before:] == [("2pt",), ("3pt_over_2pt",)]
        assert diagnostics["fits"][i] == independent_diagnostic["fits"][0]
        if fit_samples:
            expected_samples.append(independent.values)
    assert diagnostics["fits"][0]["stages"][0] is not diagnostics["fits"][1]["stages"][0]
    if fit_samples:
        np.testing.assert_array_equal(output.values, np.concatenate(expected_samples, axis=1))
    else:
        assert output is None
