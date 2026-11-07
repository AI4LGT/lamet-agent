"""Shared bootstrap lineage, geometry, and TMD ratio regression checks."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from lamet_agent.contract import evaluate_rules
from lamet_agent.data import EnsembleData, EnsembleInfo, merge_sample_lineage
from lamet_agent.stages.correlator_analysis._input import (
    _finite_fit_configurations,
    ensure_raw_correlators,
    load_descriptor,
    prepare_resampling_groups,
    resample_correlators,
)
from lamet_agent.stages.renormalization.contract import JOB_RULES
from lamet_agent.stages.renormalization.physics import physical_z_coordinates, tmd_ratio


def _sample(group, count=20, seed=3):
    raw = EnsembleData(None, "raw", list(np.arange(1, count + 1, dtype=float)), [], {})
    return resample_correlators(
        {"correlators": {"value": raw}, "configuration_ids": list(map(str, range(count)))},
        mode="bootstrap", group=group, bin_size=1, n_boot=128, seed=seed, tracked=True,
    )["value"]


def test_independent_groups_share_frame_but_not_random_stream_and_keep_cancellations(tmp_path):
    wf, ff = _sample("wf553", 20), _sample("ff100", 20)
    np.testing.assert_array_equal(wf.values, _sample("wf553", 20).values)
    assert wf.attrs["replica_frame_id"] == ff.attrs["replica_frame_id"]
    assert not np.array_equal(wf.values, ff.values)
    soft = ff.div(wf.mul(wf))
    soft.array.attrs.update(merge_sample_lineage(ff, wf))
    matched = wf.mul(soft)
    matched.array.data = wf.values * np.sqrt(soft.values)
    matched.array.attrs.update(merge_sample_lineage(wf, soft))
    np.testing.assert_allclose(matched.values, np.sqrt(ff.values), rtol=2e-15)
    assert set(json.loads(matched.attrs["resample_lineage"])) == {"wf553", "ff100"}
    matched.to_netcdf(tmp_path / "combined.nc")
    restored = EnsembleData.from_netcdf(tmp_path / "combined.nc")
    assert merge_sample_lineage(restored, wf) == merge_sample_lineage(matched, wf)


def test_lineage_rejects_inconsistent_shared_groups_and_replica_order():
    first = _sample("shared", 20)
    with pytest.raises(ValueError, match="conflicting resampling plans"):
        merge_sample_lineage(first, _sample("shared", 21))
    with pytest.raises(ValueError, match="replica frames"):
        merge_sample_lineage(first, _sample("other", 20, seed=4))
    permuted = first.copy()
    permuted.array = permuted.array.isel(resample=slice(None, None, -1))
    with pytest.raises(ValueError, match="replica order"):
        merge_sample_lineage(first, permuted)


def test_real_ratio_preflight_checks_imaginary_operands_before_complex_division():
    values = np.ones((3, 5), dtype=complex)
    values.imag[1, 2] = np.nan
    values.imag[2, 4] = np.nan  # Outside the selected domain.
    data = EnsembleData(None, "raw", list(values), ["t"], {"t": list(range(5))},
                        attrs={"correlator_type": "two_point"})
    params = {"component": "re", "fit_scope": ["2pt", "3pt_over_2pt"],
              "pt2_windows": [{"tmin": 1, "tmax": 4}]}
    assert _finite_fit_configurations(data, params).tolist() == [True, False, True]
    assert _finite_fit_configurations(data, {**params, "fit_scope": ["2pt"]}).tolist() == [True, True, True]


@pytest.mark.parametrize("scope", ["3pt_over_2pt", "FH"])
@pytest.mark.parametrize("form", ["Breit", "NonBreit"])
def test_ratio_preflight_includes_two_point_denominators_outside_spectrum_window(scope, form):
    values = np.ones((5, 16), dtype=complex)
    values[1, 12] = np.nan  # C2(tsep), outside the spectrum fit.
    values[2, 3] = np.nan  # NonBreit C2(tau), outside the spectrum fit.
    values[3, 9] = np.nan  # NonBreit C2(tsep-tau), outside the spectrum fit.
    values[4, 14] = np.nan  # Used by neither likelihood.
    data = EnsembleData(None, "raw", list(values), ["t"], {"t": list(range(16))},
                        attrs={"correlator_type": "two_point"})
    params = {"component": "re", "fit_scope": ["2pt", scope], "fitting_form": form,
              "pt2_windows": [{"tmin": 4, "tmax": 8}],
              "pt3_windows": [{"tsep_ls": [10, 12], "tau_cut": 3}]}
    expected = [True, False, form == "Breit", form == "Breit", True]
    assert _finite_fit_configurations(data, params).tolist() == expected
    params["fit_scope"] = ["2pt", "3pt_over_3pt"]
    assert _finite_fit_configurations(data, params).tolist() == [True] * 5


def _descriptor(tmp_path):
    ensemble = EnsembleInfo("toy", "same_lattice", .06, .06, 48, 64, .67)
    records = []
    with h5py.File(tmp_path / "raw.h5", "w") as handle:
        for name, bad_config, bad_time in (("a", 1, 2), ("b", 2, 3)):
            values = np.ones((6, 10, 2), dtype=complex)
            values[bad_config, bad_time, 1] = np.nan
            values[3, 8, 1] = np.nan  # Outside both authored likelihood windows.
            handle[name] = values
            records.append({
                "id": name, "ensemble": ensemble._asdict(), "count": 6,
                "format": "hdf5", "path": "raw.h5", "dataset": name,
                "dataset_dims": ["configuration", "t", "z"],
                "dims": ["configuration", "t", "z"], "coords": {"t": list(range(10)), "z": [0, 1]},
                "selectors": {"resample_group": "shared", "longitudinal_direction": [1, 1, 0],
                              "bT_fm": .12, "observable": "tmdpdf"},
                "correlator_type": "qda", "hadron": {"name": "pion"},
                "source_momentum": [3, 3, 0], "sink_momentum": [3, 3, 0],
                "current": {"kernel_operator": "g5", "parton": "quark", "renormalization_scheme": "ratio"},
                "source_sink_separation": None,
            })
    path = tmp_path / "correlators.json"
    path.write_text(json.dumps({"correlators": records}))
    return path, ensemble


def test_group_preflight_unions_invalid_configs_only_over_all_authored_fit_domains(tmp_path):
    path, ensemble = _descriptor(tmp_path)
    jobs = [{"id": name, "inputs": {"correlators": [{"json": path.name, "id": name}]}}
            for name in ("a", "b")]
    document = {"metadata": {"bin_size": 1}, "stages": {"correlator_analysis": {
        "defaults": {"fit_scope": ["qda"], "component": "both", "pt2_windows": [{"tmin": 2, "tmax": 5}]},
        "jobs": jobs,
    }}}
    group = prepare_resampling_groups(document, tmp_path)["shared"]
    assert group["retained_indices"] == [0, 3, 4, 5]
    assert group["excluded_indices"] == [1, 2]
    assert group["exclusion_reasons"] == {
        "1": [{"configuration_id": "1", "job_id": "a", "correlator_id": "a",
               "reason": "nonfinite_in_authored_fit_domain"}],
        "2": [{"configuration_id": "2", "job_id": "b", "correlator_id": "b",
               "reason": "nonfinite_in_authored_fit_domain"}],
    }
    jobs.reverse()
    assert prepare_resampling_groups(document, tmp_path)["shared"] == group
    context = SimpleNamespace(
        inputs={"correlators": [{"json": path, "id": "a"}]},
        state={"stage_shared": {"resampling_groups": {"shared": group}}},
    )
    masked = ensure_raw_correlators(context)["a"]
    assert json.loads(masked.attrs["excluded_configuration_reasons"]) == group["exclusion_reasons"]
    masked.to_netcdf(tmp_path / "masked.nc")
    restored = EnsembleData.from_netcdf(tmp_path / "masked.nc")
    assert json.loads(restored.attrs["excluded_configuration_reasons"]) == group["exclusion_reasons"]
    data = load_descriptor(path, selected_ids={"a"})["correlators"]["a"]
    assert data.ensemble.a_s == .06
    assert data.attrs["bz_step_fm"] == pytest.approx(.06 * np.sqrt(2))
    assert data.attrs["momentum_gev"] == pytest.approx(3 * np.sqrt(2) * ensemble.k_s)
    converted = physical_z_coordinates(data)
    assert converted.coords["z"] == pytest.approx([0, .06 * np.sqrt(2)])
    assert converted.ensemble == ensemble


def test_tmd_reference_and_application_keep_shared_denominator_covariance():
    samples = _sample("p0", 20)
    values = np.stack((2 * samples.values, 4 * samples.values), axis=1)
    target = EnsembleData(None, "bootstrap", list(values), ["z"], {"z": [0, .06]},
                          attrs={**samples.attrs, "bT_fm": .12, "momentum_gev": 0., "coord_unit": "fm",
                                 "observable": "tmdwf"})
    local = EnsembleData(None, "bootstrap", list(samples.values[:, None]), ["z"], {"z": [0.]},
                         attrs={**samples.attrs, "bT_fm": 0., "momentum_gev": 0., "coord_unit": "fm",
                                "observable": "tmdwf"})
    reference = tmd_ratio(target, local, operation="reference")
    assert reference.dims == []
    np.testing.assert_allclose(reference.values, 2.)
    target.array.attrs["momentum_gev"] = 3.44
    local.array.attrs["momentum_gev"] = 3.44
    normalized = tmd_ratio(target, local, operation="normalize_wf")
    applied = tmd_ratio(normalized, reference, operation="apply")
    np.testing.assert_allclose(applied.values, np.tile([1., 2.], (128, 1)))
    assert applied.attrs["resample_lineage"] == samples.attrs["resample_lineage"]


@pytest.mark.parametrize("operation", ["normalize_wf", "reference"])
def test_tmd_local_normalization_uses_real_current_with_replica_errors(operation):
    samples = _sample("shared", 20)
    fluctuation = samples.values
    momentum = 3.44 if operation == "normalize_wf" else 0.0
    attrs = {**samples.attrs, "momentum_gev": momentum, "coord_unit": "fm", "observable": "tmdwf"}
    target = EnsembleData(None, "bootstrap", list(((2 + .3j) * fluctuation)[:, None]),
                          ["z"], {"z": [0.]}, attrs={**attrs, "bT_fm": .24})
    local = EnsembleData(None, "bootstrap", list((fluctuation * (1 + .2j))[:, None]),
                         ["z"], {"z": [0.]}, attrs={**attrs, "bT_fm": 0.})
    result = tmd_ratio(target, local, operation=operation)
    np.testing.assert_allclose(result.values, 2 + .3j if operation == "normalize_wf" else 2.)
    assert result.attrs["local_normalization_component"] == "re"
    assert result.attrs["resample_lineage"] == samples.attrs["resample_lineage"]
    local.array.values[:] = 1j
    with pytest.raises(ValueError, match="zero sample"):
        tmd_ratio(target, local, operation=operation)


@pytest.mark.parametrize("operation,target_momentum,local_momentum", [
    ("normalize_wf", 3.44, 3.87),
    ("normalize_wf", 0.0, 0.0),
    ("normalize_wf", -3.44, -3.44),
    ("normalize_wf", 3.44, None),
    ("normalize_wf", 3.44, "3.44"),
    ("normalize_wf", np.nan, 3.44),
    ("normalize_wf", 3.44, np.inf),
    ("normalize_wf", True, True),
    ("reference", 0.0, 3.44),
    ("reference", 3.44, 0.0),
    ("reference", 0.0, None),
    ("reference", None, 0.0),
])
def test_tmd_local_normalization_rejects_wrong_or_missing_momenta(operation, target_momentum, local_momentum):
    attrs = {"observable": "tmdwf", "coord_unit": "fm", "bT_fm": 0.0}
    target = EnsembleData(None, "bootstrap", [np.ones(2), 2 * np.ones(2)], ["z"], {"z": [0.0, 0.06]},
                          attrs={**attrs, "momentum_gev": target_momentum})
    local = target.copy()
    local.array.attrs["momentum_gev"] = local_momentum
    with pytest.raises(ValueError, match="momentum"):
        tmd_ratio(target, local, operation=operation)


@pytest.mark.parametrize("role", ["target", "local"])
def test_tmd_local_normalization_requires_wavefunction_observable(role):
    attrs = {"observable": "tmdwf", "coord_unit": "fm", "bT_fm": 0.0, "momentum_gev": 3.44}
    target = EnsembleData(None, "bootstrap", [np.ones(2), 2 * np.ones(2)], ["z"], {"z": [0.0, 0.06]},
                          attrs=attrs)
    local = target.copy()
    (target if role == "target" else local).array.attrs["observable"] = "tmdpdf"
    with pytest.raises(ValueError, match="observable='tmdwf'"):
        tmd_ratio(target, local, operation="normalize_wf")


@pytest.mark.parametrize(
    "operation,role", [("normalize_wf", "local"), ("reference", "local"), ("apply", "denominator")]
)
def test_tmd_ratio_contract_selects_explicit_input_roles(operation, role):
    block = {"jobs": [{"id": "tmd", "strategy": "tmd_ratio", "operation": operation,
                       "normalization": False, "inputs": {"target": "target", role: "factor"}}]}
    assert evaluate_rules(block, JOB_RULES, complete=True) == []
    assert block["jobs"][0]["scheme"] == "ratio"
    assert block["jobs"][0]["inputs"] == {"target": "target", role: "factor"}
    assert block["jobs"][0]["projection"] == "preserve"


def test_tmd_real_projection_preserves_ratio_samples_and_lineage(tmp_path):
    samples = _sample("moving", 20)
    denominator = _sample("rest", 20)
    target = EnsembleData(
        None, "bootstrap", list(samples.values[:, None] * np.array([[2 + 3j, 4 - 1j]])),
        ["z"], {"z": [0., .06]},
        attrs={**samples.attrs, "coord_unit": "fm", "output_component": "both"},
    )
    # Project after division, not before: the denominator may be complex.
    denominator.array.data = denominator.values * (1 + .25j)
    unchanged = tmd_ratio(target, denominator, operation="apply")
    explicit = tmd_ratio(target, denominator, operation="apply", projection="preserve")
    projected = tmd_ratio(target, denominator, operation="apply", projection="real")
    np.testing.assert_array_equal(unchanged.values, explicit.values)
    np.testing.assert_array_equal(projected.values.real, unchanged.values.real)
    np.testing.assert_array_equal(projected.values.imag, 0.)
    np.testing.assert_array_equal(projected.real.sdev, unchanged.real.sdev)
    assert projected.attrs["resample_lineage"] == unchanged.attrs["resample_lineage"]
    assert set(json.loads(projected.attrs["resample_lineage"])) == {"moving", "rest"}
    assert target.attrs["output_component"] == "both"
    assert projected.attrs["output_component"] == "re"
    assert projected.attrs["renormalization_projection"] == "real"
    projected.to_netcdf(tmp_path / "projected.nc")
    restored = EnsembleData.from_netcdf(tmp_path / "projected.nc")
    np.testing.assert_array_equal(restored.values, projected.values)
    assert restored.attrs == projected.attrs
    with pytest.raises(ValueError, match="projection"):
        tmd_ratio(target, denominator, operation="apply", projection="invalid")


@pytest.mark.parametrize("projection", ["real", "preserve", "invalid"])
def test_tmd_ratio_contract_validates_projection(projection):
    block = {"jobs": [{"id": "tmd", "strategy": "tmd_ratio", "operation": "apply",
                       "normalization": False, "projection": projection,
                       "inputs": {"target": "target", "denominator": "factor"}}]}
    issues = evaluate_rules(block, JOB_RULES, complete=True)
    assert bool(issues) == (projection == "invalid")
