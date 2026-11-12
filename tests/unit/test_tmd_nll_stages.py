"""NLL stage factorization, historical soft controls, and provenance boundaries."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.interpolate import interp1d

from lamet_agent.contract import evaluate_rules
from lamet_agent.data import EnsembleData, sample_covariance, sample_mean, sample_sdev
from lamet_agent.kernels.tmd.coefficients import hard_factor
from lamet_agent.stages.matching.contract import JOB_RULES as MATCH_RULES, check_tmd
from lamet_agent.stages.matching.tmd import match
from lamet_agent.stages.soft_analysis.contract import JOB_RULES as SOFT_RULES, check_settings
from lamet_agent.stages.soft_analysis import physics
from lamet_agent.stages.soft_analysis.physics import (
    endpoint_fit, evolution_gamma, extract_cs, extract_intrinsic, quadrature,
)
from test_tmd_soft_matching import _attrs, _data


NLL = {"order": "nll", "mu": 2., "alpha_s": .293, "infrared_policy": "freeze", "freeze_scale_gev": 1.}


def _chain():
    x = np.linspace(.05, .95, 181)
    amplitude = np.linspace(1.8, 2.2, 24)
    gamma = -.35 + .1 * (amplitude - 2)
    shape = amplitude[:, None] * (x * (1 - x)) ** .8
    quasi = []
    for index, p in ((8, 3.44), (9, 3.87), (10, 4.30)):
        values = shape * hard_factor(x, observable="tmdwf", momentum_gev=p, **NLL)
        values *= np.exp(gamma[:, None] * np.log(p / 2.58))
        data = _data(values, ["x"], {"x": x}, observable="tmdwf", momentum_index=index,
                     momentum_gev=p, bT_fm=.24)
        # A noncanonical order catches accidental independent resampling/reordering.
        data.array = data.array.assign_coords(resample=np.arange(24)[::-1])
        quasi.append(data)
    return quasi, amplitude, gamma


def _ff(quasi, values):
    ff = _data(np.asarray(values)[:, None], ["bT"], {"bT": [.24]},
               **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58})
    ff.array = ff.array.assign_coords(resample=quasi[0].array.coords["resample"])
    return ff


def test_nll_synthetic_cs_soft_wf_pdf_factorization_and_netcdf_lineage(tmp_path):
    quasi, amplitude, gamma = _chain()
    cs, cs_diagnostic = extract_cs(quasi, NLL)
    np.testing.assert_allclose(cs.values[:, 0], gamma, atol=2e-14)
    assert cs.attrs["order"] == cs.attrs["resummation"] == "nll"
    for data in (cs, cs_diagnostic["_reduced_samples"]):
        np.testing.assert_array_equal(data.array.coords["resample"], quasi[0].array.coords["resample"])
    nodes, matrix, _ = quadrature(.05, .3, .7, 96, 2.58, 2., .293, "nll", "freeze", 1.)
    beam = amplitude[:, None] * (nodes * (1 - nodes)) ** .8
    integral = np.einsum("si,ij,sj->s", beam, matrix, beam)
    intrinsic = np.linspace(.65, .75, len(amplitude))
    soft, _ = extract_intrinsic(quasi, [_ff(quasi, intrinsic * integral)], cs, NLL)
    np.testing.assert_allclose(soft.values[:, 0], intrinsic, rtol=6e-5)
    for name, data in (("cs", cs), ("soft", soft)):
        data.to_netcdf(tmp_path / f"{name}.nc")
    cs = EnsembleData.from_netcdf(tmp_path / "cs.nc")
    soft = EnsembleData.from_netcdf(tmp_path / "soft.nc")
    wf, diagnostic = match(quasi[0], cs, soft,
                            {**NLL, "observable": "tmdwf", "p_ref_gev": 2.58,
                             "lc_x_ls": {"start": .2, "stop": .8}})
    xx = np.asarray(wf.coords["x"])
    expected = np.sqrt(intrinsic[:, None]) * amplitude[:, None] * (xx * (1 - xx)) ** .8
    np.testing.assert_allclose(wf.values, expected, rtol=4e-5)
    assert wf.attrs["kernel_id"] == "tmd_cg_nll"
    assert diagnostic["coefficient_provenance"]["freeze_scale_gev"] == 1.
    assert set(json.loads(wf.attrs["resample_lineage"])) == {"wf", "ff"}
    x = np.asarray(quasi[0].coords["x"])
    p = 2.43
    target = np.exp(-x)[None, :] * np.linspace(.9, 1.1, len(gamma))[:, None]
    raw = target / 2 * hard_factor(x, observable="tmdpdf", momentum_gev=p, **NLL)
    raw *= np.exp(.5 * gamma[:, None] * np.log((2 * x * p) ** 2 / 4)) / np.sqrt(intrinsic[:, None])
    pdf = _data(raw, ["x"], {"x": x}, observable="tmdpdf", momentum_gev=p, bT_fm=.24)
    pdf.array = pdf.array.assign_coords(resample=quasi[0].array.coords["resample"])
    matched, _ = match(pdf, cs, soft,
                       {**NLL, "observable": "tmdpdf", "lc_x_ls": {"start": .1, "stop": .9}})
    selected = (x >= .1 - 1e-12) & (x <= .9 + 1e-12)
    np.testing.assert_allclose(matched.values, target[:, selected], rtol=4e-5)


def test_published_raw_cs_is_the_original_quasi_ratio_and_never_matches_as_physical(tmp_path):
    quasi, _, _ = _chain()
    cs, diagnostic = extract_cs(quasi, NLL)
    x = np.asarray(quasi[0].coords["x"])
    selected = (x >= .34 - 1e-12) & (x <= .66 + 1e-12)
    raw = np.concatenate([
        np.log(second.values[:, selected] / first.values[:, selected])
        / np.log(second.attrs["momentum_gev"] / first.attrs["momentum_gev"])
        for first, second in zip(quasi[:-1], quasi[1:])
    ], axis=1).mean(axis=1)
    np.testing.assert_allclose(evolution_gamma(cs, .24, "quasi_ratio"), raw, atol=2e-14)
    np.testing.assert_allclose(diagnostic["_reduced_samples"].values[:, 0], raw, atol=2e-14)
    cs.to_netcdf(tmp_path / "cs.nc")
    np.testing.assert_allclose(evolution_gamma(EnsembleData.from_netcdf(tmp_path / "cs.nc"), .24, "quasi_ratio"),
                               raw, atol=2e-14)
    bad = EnsembleData.from_netcdf(tmp_path / "cs.nc")
    bad.array.attrs["cs_convention"] = "quasi_ratio"
    with pytest.raises(ValueError, match="physical hard-corrected"):
        extract_intrinsic(quasi, [], bad, NLL)
    del bad.array.attrs["cs_hard_correction"]
    with pytest.raises(ValueError, match="cs_hard_correction"):
        evolution_gamma(bad, .24, "quasi_ratio")


def test_nll_reference_grid_and_working_covariance_keep_original_replicas(monkeypatch):
    quasi, _, _ = _chain()
    cs, _ = extract_cs(quasi, NLL)
    ff = _ff(quasi, np.linspace(.09, .11, 24))
    captured = []
    original = physics.endpoint_fit

    def capture(x, values, windows, resample, attrs=None, **kwargs):
        captured.append((x.copy(), values.copy(), kwargs["covariance"].copy()))
        return original(x, values, windows, resample, attrs, **kwargs)

    monkeypatch.setattr(physics, "endpoint_fit", capture)
    params = {**NLL, "evolution_cs": "quasi_ratio", "sudakov_momentum": "quasi",
              "endpoint_fit_points_per_window": 20, "endpoint_central_points": 100,
              "endpoint_covariance": "reference_cs_diagonal"}
    soft, diagnostic = extract_intrinsic(quasi, [ff], cs, params)
    raw_gamma = evolution_gamma(cs, .24, "quasi_ratio")
    for source, (fit_x, values, covariance), record in zip(
            quasi, captured, diagnostic["bT_records"][0]["momenta"]):
        assert len(fit_x) == 40
        assert record["endpoint_fit"]["fit_points"] == 40
        assert record["sudakov_momentum_gev"] == source.attrs["momentum_gev"]
        assert record["endpoint_fit"]["working_covariance"] == "reference_cs_diagonal"
        interpolated = interp1d(source.coords["x"], source.values, axis=1, kind="cubic")(fit_x)
        p = source.attrs["momentum_gev"]
        h = hard_factor(fit_x, observable="tmdwf", momentum_gev=p, **NLL)
        ratio_log = np.log(2.58 / p)
        expected = interpolated / h * np.exp(raw_gamma[:, None] * ratio_log)
        np.testing.assert_allclose(values, expected, rtol=1e-14)
        center_factor = np.exp(sample_mean(raw_gamma, "bootstrap") * ratio_log) / h
        independent_cs_error = (sample_mean(interpolated, "bootstrap") * center_factor * ratio_log
                                * sample_sdev(raw_gamma, "bootstrap"))
        expected_metric = sample_covariance(interpolated * center_factor, "bootstrap")
        expected_metric += np.diag(independent_cs_error**2)
        np.testing.assert_allclose(covariance, expected_metric, rtol=1e-12, atol=1e-16)
        # The historical likelihood metric is intentionally different from
        # the true covariance, which remains represented by these same rows.
        assert not np.allclose(covariance, sample_covariance(values, "bootstrap"), rtol=1e-5, atol=1e-12)
    assert soft.n_sample == quasi[0].n_sample
    assert soft.attrs["evolution_cs"] == "quasi_ratio"
    np.testing.assert_array_equal(soft.array.coords["resample"], quasi[0].array.coords["resample"])
    matched, diagnostic = match(quasi[0], cs, soft, {**NLL, "observable": "tmdwf", "lc_x_ls": [.3, .5, .7]})
    assert matched.attrs["cs_convention"] == "physical"
    assert diagnostic["soft_evolution_cs"] == "quasi_ratio"
    assert diagnostic["soft_sudakov_momentum"] == "quasi"


def test_coefficient_order_and_infrared_provenance_must_match_every_dag_branch():
    quasi, _, _ = _chain()
    cs, _ = extract_cs(quasi, NLL)
    for params, message in (({}, "order differs"), ({**NLL, "freeze_scale_gev": 1.1}, "infrared prescription")):
        with pytest.raises(ValueError, match=message):
            extract_intrinsic(quasi, [], cs, params)
    del cs.array.attrs["coefficient_provenance"]
    with pytest.raises(ValueError, match="requires explicit coefficient_provenance"):
        extract_intrinsic(quasi, [], cs, NLL)


def test_endpoint_working_covariance_has_physical_units_and_positive_semidefinite_validation():
    x = np.linspace(.3, .7, 41)
    values = np.linspace(1.9, 2.1, 24)[:, None] * (x * (1 - x)) ** .8
    covariance = np.diag(np.linspace(.001, .004, len(x)))
    windows = [[.3, .45], [.55, .7]]
    fit, _ = endpoint_fit(x, values, windows, "bootstrap", covariance=covariance)
    scaled, _ = endpoint_fit(x, values * 1e-6, windows, "bootstrap", covariance=covariance * 1e-12)
    np.testing.assert_allclose(scaled / [1e-6, 1], fit, rtol=1e-10)
    with pytest.raises(ValueError, match="positive semidefinite"):
        endpoint_fit(x, values, windows, "bootstrap", covariance=-covariance)


def test_nll_contracts_expose_reference_controls_and_reject_ignored_nlo_ir_settings():
    inputs = {"quasi": ["q"], "form_factor": "ff", "cs_kernel": "cs"}
    job = {"id": "soft", "type": "intrinsic_soft", "inputs": inputs, **NLL,
           "evolution_cs": "quasi_ratio", "sudakov_momentum": "quasi", "endpoint_fit_points_per_window": 20,
           "endpoint_central_points": 100, "endpoint_covariance": "reference_cs_diagonal"}
    assert evaluate_rules({"jobs": [job]}, SOFT_RULES) == []
    assert check_settings(SimpleNamespace(params=job, inputs=inputs)) is None
    invalid = {**job, "order": "nlo"}
    assert "only to NLL" in check_settings(SimpleNamespace(params=invalid, inputs=inputs)).message
    match_inputs = {"quasi": "q", "cs_kernel": "cs", "intrinsic_soft": "soft"}
    job = {"id": "wf", "observable": "tmdwf", "inputs": match_inputs, **NLL, "lc_x_ls": [.3, .5, .7]}
    assert evaluate_rules({"jobs": [job]}, MATCH_RULES) == []
    assert check_tmd(SimpleNamespace(params=job, inputs=match_inputs)) is None
    assert check_tmd(SimpleNamespace(params={"observable": "collinear", "order": "nll"}, inputs={})).path == "order"


def test_nll_jackknife_shared_cs_wf_nuisance_cancels_after_branch_merge(tmp_path):
    from lamet_agent.data import align_sample_lineage
    from test_tmd_jackknife_correlated_cs import _group

    moving = _group("wf", [1.7, 1.85, 1.93, 2., 2.05, 2.15, 2.3, 2.4])
    rest = _group("rest", [.7, .85, 1., 1.1, 1.35])
    ff = _group("ff", [.08, .09, .095, .105, .11, .12])
    moving, rest = align_sample_lineage(moving, rest)
    x = np.linspace(.05, .95, 181)
    gamma = -.3 + .4 * (moving.values - 2)
    amplitude = moving.values / rest.values
    quasi = []
    for index, p in ((8, 3.44), (9, 3.87), (10, 4.30)):
        values = amplitude[:, None] * (x * (1 - x)) ** .8
        values = values * hard_factor(x, observable="tmdwf", momentum_gev=p, **NLL)
        values *= np.exp(gamma[:, None] * np.log(p / 2.58))
        attrs = {**moving.attrs, "observable": "tmdwf", "parton": "quark", "gfix": "CG",
                 "distribution": "quasi", "renormalization_scheme": "ratio", "bT_fm": .24,
                 "momentum_gev": p, "momentum_index": index}
        quasi.append(EnsembleData(None, "jackknife", list(values), ["x"], {"x": x}, attrs))
    cs, _ = extract_cs(quasi, NLL)
    np.testing.assert_allclose(cs.values[:, 0], gamma, atol=1e-14)
    ff.array.attrs.update(observable="form_factor", parton="quark", gfix="CG", bT_fm=.24, momentum_gev=2.58)
    soft, _ = extract_intrinsic(quasi, [ff], cs, NLL)
    soft.to_netcdf(tmp_path / "soft.nc")
    soft = EnsembleData.from_netcdf(tmp_path / "soft.nc")
    matched, _ = match(quasi[0], cs, soft,
                       {**NLL, "observable": "tmdwf", "p_ref_gev": 2.58, "lc_x_ls": [.3, .5, .7]})
    assert json.loads(matched.attrs["jackknife_layout"]) == {"ff": 6, "rest": 5, "wf": 8}
    aligned_ff, matched = align_sample_lineage(ff, matched)
    normalized = matched.values / np.sqrt(aligned_ff.values[:, None])
    np.testing.assert_allclose(normalized, np.broadcast_to(normalized[-1], normalized.shape), atol=1e-10)
    independent_ff = np.sqrt(ff.values[:, None]) * normalized[-1]
    centered = independent_ff - independent_ff.mean(axis=0)
    expected_covariance = 5 / 6 * centered.T @ centered
    np.testing.assert_allclose(sample_covariance(matched.values, "jackknife", matched.attrs),
                               expected_covariance, rtol=1e-9, atol=1e-14)
