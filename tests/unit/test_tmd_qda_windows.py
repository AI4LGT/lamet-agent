"""Static reference time windows retain one axial output per (P,bT) job."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from lamet_agent.stages.correlator_analysis._input import _finite_fit_configurations
from lamet_agent.stages.correlator_analysis._qda_windows import qda_time_masks, validate_qda_windows
from lamet_agent.stages.correlator_analysis.contract import check_tmd_modes
from lamet_agent.stages.correlator_analysis.physics import fit_qda_samples
from test_tmd_correlator import _axial_inputs


def _options():
    return dict(
        fit_scope=["2pt+qda"], qda_output="tmdwf", components="real",
        tmin=3, tmax=11, qda_window={"tmin": 5, "tmax": 10},
        n_states={"2pt": 1, "qda": 1}, prior_width=1.0, svdcut=1e-7,
        posterior_prior_error_scale=10.0, sample_prior_mode="original",
    )


def test_per_z_windows_match_independent_fits_and_record_actual_domains():
    data, phi = _axial_inputs()
    overrides = {"2": {"tmin": 7, "tmax": 12}}
    # A value outside z=2's authored window must never enter that fit.
    data["q"].array.values[:, 6, 1] = np.nan
    result, diagnostic = fit_qda_samples(data, **_options(), qda_windows_by_z=overrides)
    assert result.coords["z"] == [0.0, 2.0]
    assert json.loads(result.attrs["qda_windows_by_z"]) == overrides
    assert [fit["qda_window"] for fit in diagnostic["fits"]] == [
        {"tmin": 5, "tmax": 10}, {"tmin": 7, "tmax": 12},
    ]
    np.testing.assert_allclose(result.mean, phi, rtol=0.015)
    for index, window in enumerate([{"tmin": 5, "tmax": 10}, overrides["2"]]):
        single = {"ss": data["ss"], "q": data["q"].at("z", [2 * index])}
        direct, _ = fit_qda_samples(single, **{**_options(), "qda_window": window})
        np.testing.assert_allclose(result.values[:, index], direct.values[:, 0], rtol=1e-12)


def test_tuning_uses_the_same_per_z_override_as_replica_publication():
    data, _ = _axial_inputs()
    _, diagnostic = fit_qda_samples(
        data, **_options(), qda_windows_by_z={"2.0": {"tmin": 7, "tmax": 13}},
        fit_samples=False, tune_z=2,
    )
    assert diagnostic["fits"][0]["qda_window"] == {"tmin": 7, "tmax": 13}
    assert diagnostic["n_data"] == 8 + 6


@pytest.mark.parametrize("overrides", [
    {"2": {"tmin": 5, "tmax": 10}, "2.0": {"tmin": 6, "tmax": 10}},
    {"nan": {"tmin": 5, "tmax": 10}},
    {"2": {"tmin": True, "tmax": 10}},
    {"2": {"tmin": 10, "tmax": 5}},
    {"2": {"tmin": 5, "tmax": 10, "unused": 1}},
])
def test_malformed_or_duplicate_coordinate_windows_are_rejected(overrides):
    with pytest.raises(ValueError, match="qda_windows_by_z"):
        validate_qda_windows(overrides)
    context = SimpleNamespace(params={
        "analysis_method": "lsqfit", "fit_scope": ["2pt+qda"],
        "qda_output": "tmdwf", "qda_windows_by_z": overrides,
    })
    issue = check_tmd_modes(context)
    assert issue is not None and issue.path == "qda_windows_by_z"


def test_coordinate_windows_require_native_tmd_and_existing_coordinates():
    overrides = {"2": {"tmin": 7, "tmax": 12}}
    context = SimpleNamespace(params={
        "analysis_method": "lsqfit", "fit_scope": ["qda"],
        "qda_output": "ratio", "qda_windows_by_z": overrides,
    })
    assert check_tmd_modes(context).path == "qda_windows_by_z"
    data, _ = _axial_inputs()
    with pytest.raises(ValueError, match="available z"):
        fit_qda_samples(data, **_options(), qda_windows_by_z={"4": overrides["2"]})
    with pytest.raises(ValueError, match="at z=2"):
        fit_qda_samples(data, **_options(), qda_windows_by_z={"2": {"tmin": 19, "tmax": 20}})


def test_finite_preflight_excludes_only_the_exact_authored_time_z_domain():
    data, _ = _axial_inputs()
    source = data["q"]
    source.array.values[0, 6, 1] = np.nan   # outside z=2 override
    source.array.values[1, 11, 0] = np.nan  # outside z=0 fallback
    source.array.values[2, 11, 1] = np.nan  # inside z=2 override
    source.array.values[3, 6, 0] = np.nan   # inside z=0 fallback
    params = {
        "fit_scope": ["2pt+qda"], "pt2_windows": [{"tmin": 3, "tmax": 11}],
        "qda_window": {"tmin": 5, "tmax": 10},
        "qda_windows_by_z": {"2": {"tmin": 7, "tmax": 12}},
    }
    valid = _finite_fit_configurations(source, params)
    assert np.flatnonzero(~valid).tolist() == [2, 3]
    mask = qda_time_masks(source.coords["t"], source.coords["z"],
                          params["qda_window"], params["qda_windows_by_z"])
    assert mask.shape == (20, 2) and mask.sum() == 10
