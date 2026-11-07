"""Regression checks for TMD replica identities and persisted Review evidence."""

import json

import numpy as np
import pytest

from lamet_agent.agent import ToolContext, _review_summary
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.stages._reporting import StageReportRecord
from lamet_agent.stages.correlator_analysis import _fit_ff, _fit_matrix, _publish
from lamet_agent.stages.matching.tmd import match
from lamet_agent.stages.renormalization.physics import physical_z_coordinates, tmd_ratio
from lamet_agent.stages.soft_analysis.physics import endpoint_fit, extract_cs, extract_intrinsic
from lamet_agent.stages.soft_analysis.reporting import write_stage_report
from lamet_agent.stages.soft_analysis.workflow import replica_distribution
from test_correlator_prior_integration import _context, _correlators, _settings
from test_ff_reference import _inputs as wall_inputs
from test_tmd_soft_matching import _attrs, _chain, _data


def _ratio_inputs():
    ensemble = EnsembleInfo("test", "test", 0.06, 0.06, 48, 64, 0.67)
    attrs = {
        "coord_unit": "lattice",
        "observable": "tmdpdf",
        "bT_fm": 0.24,
        "replica_frame_id": "frame",
        "resample_lineage": '{"shared":"plan"}',
    }
    target = EnsembleData(
        ensemble, "bootstrap", [np.array([value, 2 * value]) for value in [1.0, 2.0, 3.0]], ["z"], {"z": [0, 1]}, attrs
    )
    denominator = EnsembleData(ensemble, "bootstrap", [np.array(value) for value in [1.0, 2.0, 3.0]], [], {}, attrs)
    return target, denominator


def test_coordinate_conversion_preserves_replica_labels_and_ratio_pairing():
    target, denominator = _ratio_inputs()
    for data in (target, denominator):
        data.array = data.array.isel(resample=slice(None, None, -1))
    converted = physical_z_coordinates(target)
    np.testing.assert_array_equal(converted.array.coords["resample"], [2, 1, 0])
    np.testing.assert_array_equal(converted.values, target.values)
    np.testing.assert_allclose(converted.coords["z"], [0, 0.06])
    assert target.coords["z"] == [0, 1]
    assert target.attrs["coord_unit"] == "lattice"
    output = tmd_ratio(target, denominator, operation="apply")
    np.testing.assert_array_equal(output.array.coords["resample"], [2, 1, 0])
    np.testing.assert_allclose(output.values, [[1, 2]] * 3)


def test_coordinate_conversion_cannot_hide_misaligned_ratio_replicas():
    target, denominator = _ratio_inputs()
    target.array = target.array.isel(resample=slice(None, None, -1))
    with pytest.raises(ValueError, match="replica order"):
        tmd_ratio(target, denominator, operation="apply")


def test_extracted_soft_functions_expose_percentiles_and_numerical_checks_to_review(tmp_path):
    quasi, *_ = _chain()
    cs, _ = extract_cs(quasi, {})
    ff = _data(
        np.ones((cs.n_sample, 1)),
        ["bT"],
        {"bT": [0.24]},
        **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58},
    )
    soft, diagnostics = extract_intrinsic(quasi, [ff], cs, {})
    for result in (cs, soft):
        assert result.attrs["target_observable"] == "tmd"
    path = tmp_path / "soft.nc"
    soft.to_netcdf(path)
    soft = EnsembleData.from_netcdf(path)
    diagnostics["replica_distribution"] = replica_distribution(soft)
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "soft_analysis",
        "soft",
        {},
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(0),
        output=soft,
    )
    evidence = _review_summary(context, {"diagnostics": diagnostics})["tmd_diagnostics"]
    assert evidence["replica_distribution"] == diagnostics["replica_distribution"]
    assert evidence["minimum_endpoint_Q"] >= 0
    assert evidence["max_quadrature_relative_change"] < 0.005
    assert len(evidence["momentum_consistency"]) == 1


def test_soft_jackknife_diagnostics_use_jackknife_error_and_label_raw_quantiles():
    data = EnsembleData(
        None, "jackknife", [np.array([value]) for value in [1.0, 2.0, 3.0, 4.0]], ["bT"], {"bT": [0.24]}
    )
    diagnostics = replica_distribution(data)[0]
    assert diagnostics["sdev"] == pytest.approx(np.std([1, 2, 3, 4], ddof=1) * 3 / 2)
    assert diagnostics["sdev"] == pytest.approx(data.sdev[0])
    assert diagnostics["q16"] == pytest.approx(np.quantile([1, 2, 3, 4], 0.16))
    assert "not a confidence interval" in diagnostics["quantile_interpretation"]


@pytest.mark.parametrize("resample", ["bootstrap", "jackknife"])
def test_soft_report_distinguishes_bootstrap_intervals_from_jackknife_quantiles(tmp_path, resample):
    data = EnsembleData(None, resample, [np.array([value]) for value in [1.0, 2.0, 3.0, 4.0]], ["bT"], {"bT": [0.24]})
    record = StageReportRecord(
        "soft",
        {},
        {},
        data,
        {
            "diagnostics": {
                "replica_distribution": replica_distribution(data),
                "aggregation": "arithmetic",
                "bT_records": [],
            }
        },
        tmp_path,
    )
    write_stage_report(records=[record], artifact_directory=tmp_path)
    report = (tmp_path / "report.md").read_text()
    if resample == "bootstrap":
        assert "Bootstrap percentile intervals" in report
        assert "jackknife scaling" not in report
    else:
        assert "not a confidence interval" in report
        assert "jackknife scaling factor" in report
        assert "Bootstrap percentile intervals" not in report


@pytest.mark.parametrize("observable", ["tmdpdf", "form_factor"])
def test_manifest_tmd_identity_routes_legacy_artifact_fit_quality(tmp_path, observable):
    data = _data(np.ones((4, 1)), ["bT"], {"bT": [0.24]}, observable=observable)
    assert "target_observable" not in data.attrs
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance", "target_observable": "tmd"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "legacy",
        {},
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
        output=data,
    )
    fits = {"fits": [{"Q": 0.001}, {"Q": 0.8}], "n_failed_samples": 0}
    summary = {"diagnostics": fits if observable == "form_factor" else {"selected_application_fit": fits}}
    evidence = _review_summary(context, summary)["tmd_diagnostics"]
    assert evidence["center_fit_count"] == 2
    assert evidence["Q_below_0_05"] == 1


def test_pdf_publication_persists_manifest_tmd_identity(tmp_path):
    settings = _settings()
    records = {key: value for key, value in _correlators().items() if key != "qda"}
    records["c3"].array.attrs["observable"] = "tmdpdf"
    context = _context(tmp_path, settings, records)
    context.manifest["metadata"]["target_observable"] = "tmd"
    _fit_matrix.run(context, tune_z_values=[1])
    candidate = context.state["matrix_element_candidates"][0]
    _publish.run(context, candidate_id=candidate["id"])
    output = EnsembleData.from_netcdf(tmp_path / "output.nc")
    assert output.attrs["target_observable"] == "tmd"
    assert output.attrs["observable"] == "tmdpdf"
    evidence = _review_summary(context, context.summary)
    assert evidence["tmd_diagnostics"]["center_fit_count"] == 1


def test_ff_publication_persists_manifest_tmd_identity(tmp_path):
    settings = {
        "analysis_method": "lsqfit",
        "fit_scope": ["ff"],
        "component": "re",
        "fitting_form": "Breit",
        "ff_energy_gev": 2.67,
        "svdcut": 1e-12,
        "posterior_prior_error_scale": 10.0,
        "sample_prior_mode": "original",
        "pt3_windows": [{"tsep_ls": [8, 10], "tau_cut": 3}],
        "nstate": {"ff": [2]},
        "prior_width": [1.0],
        "q_min": 0.05,
    }
    context = _context(tmp_path, settings, wall_inputs("bootstrap"))
    context.manifest["metadata"]["target_observable"] = "tmd"
    _fit_ff.run(context)
    output = EnsembleData.from_netcdf(tmp_path / "output.nc")
    assert output.attrs["target_observable"] == "tmd"
    assert output.attrs["observable"] == "form_factor"
    evidence = _review_summary(context, context.summary)
    assert evidence["tmd_diagnostics"]["center_fit_count"] == 4
    stored = json.loads((tmp_path / "diagnostics/candidates.json").read_text())
    assert len(stored["fits"]) == 4


@pytest.mark.parametrize("resample", ["bootstrap", "jackknife"])
@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1e8, 1e12])
def test_endpoint_fit_is_invariant_to_overall_input_amplitude(resample, scale):
    rng = np.random.default_rng(1)
    x = np.linspace(0.3, 0.7, 41)
    values = (x * (1 - x)) ** 0.4 * (1 + 0.01 * rng.normal(size=(50, len(x))))
    windows = [[0.3, 0.45], [0.55, 0.7]]
    reference, before = endpoint_fit(x, values, windows, resample)
    scaled, after = endpoint_fit(x, scale * values, windows, resample)
    # The fitted shape, goodness of fit and covariance must transform in the
    # original parameter units, independently of the numerical amplitude.
    scaled[:, 0] /= scale
    np.testing.assert_allclose(scaled, reference, rtol=1e-6)
    assert after["n"] == pytest.approx(before["n"], rel=1e-6)
    assert after["c"] / scale == pytest.approx(before["c"], rel=1e-6)
    assert after["chi2"] == pytest.approx(before["chi2"], rel=1e-9)
    assert after["Q"] == pytest.approx(before["Q"], rel=1e-9)
    assert after["covariance"] == before["covariance"] == "correlated; eigenvalue floor 1e-6"
    restored = np.array(after["replica_parameters"]["covariance"]) / np.outer([scale, 1], [scale, 1])
    np.testing.assert_allclose(restored, before["replica_parameters"]["covariance"], rtol=1e-5)


def _physical_soft_inputs():
    quasi, *_ = _chain()
    ensemble = EnsembleInfo("heavy", "wf", 0.06, 0.06, 48, 64, 0.67)
    for data in quasi:
        data.ensemble = ensemble
    cs, _ = extract_cs(quasi, {})
    ff = _data(
        np.ones((cs.n_sample, 1)),
        ["bT"],
        {"bT": [0.24]},
        **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58},
    )
    ff.ensemble = ensemble
    soft = _data(
        np.ones((cs.n_sample, 1)),
        ["bT"],
        {"bT": [0.24]},
        observable="intrinsic_soft",
        mu=2.0,
        alpha_s=0.293,
        order="nlo",
    )
    soft.ensemble = ensemble
    return quasi, cs, ff, soft


@pytest.mark.parametrize("stage", ["cs", "intrinsic", "matching"])
@pytest.mark.parametrize("field,value", [("a_s", 0.12), ("a_t", 0.03), ("L_s", 24), ("L_t", 128)])
def test_tmd_stages_reject_incompatible_lattice_geometry(stage, field, value):
    quasi, cs, ff, soft = _physical_soft_inputs()
    changed = {"cs": quasi[1], "intrinsic": ff, "matching": cs}[stage]
    changed.ensemble = changed.ensemble._replace(**{field: value})
    with pytest.raises(ValueError, match=f"incompatible lattice geometry: {field}"):
        if stage == "cs":
            extract_cs(quasi, {})
        elif stage == "intrinsic":
            extract_intrinsic(quasi, [ff], cs, {})
        else:
            match(quasi[0], cs, soft, {"observable": "tmdwf", "lc_x_ls": [0.4, 0.5, 0.6]})


def test_tmd_matching_rejects_partially_missing_geometry():
    quasi, cs, _, soft = _physical_soft_inputs()
    soft.ensemble = None
    with pytest.raises(ValueError, match="must all declare lattice geometry"):
        match(quasi[0], cs, soft, {"observable": "tmdwf", "lc_x_ls": [0.4, 0.5, 0.6]})


@pytest.mark.parametrize("field,value", [("gfix", "GI"), ("parton", "gluon"), ("gfix", None), ("parton", None)])
def test_intrinsic_soft_rejects_incompatible_or_missing_form_factor_identity(field, value):
    quasi, cs, ff, _ = _physical_soft_inputs()
    ff.array.attrs[field] = value
    with pytest.raises(ValueError, match="form_factor must declare CG quark provenance"):
        extract_intrinsic(quasi, [ff], cs, {})


def test_universal_soft_inputs_allow_other_valence_masses_on_the_same_lattice():
    quasi, cs, ff, _ = _physical_soft_inputs()
    ff.ensemble = ff.ensemble._replace(series="other", id="ff", m_pi=0.4)
    soft, _ = extract_intrinsic(quasi, [ff], cs, {})
    light_pdf = quasi[0].copy()
    light_pdf.ensemble = light_pdf.ensemble._replace(id="pdf", m_pi=0.14)
    light_pdf.array.attrs["observable"] = "tmdpdf"
    output, _ = match(light_pdf, cs, soft, {"observable": "tmdpdf", "lc_x_ls": [0.4, 0.5, 0.6]})
    assert output.ensemble.m_pi == 0.14
    assert np.isfinite(output.values).all()
