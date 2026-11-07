"""Analytic coefficient checks and a correlated synthetic TMD factorization chain."""

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from lamet_agent.agent import ToolContext
from lamet_agent.contract import evaluate_rules
from lamet_agent.data import EnsembleData
from lamet_agent.kernels.tmd.cg_nlo import (
    coefficient_correction, form_factor_kernel, hard_factor, sudakov_correction,
)
from lamet_agent.stages.matching.tmd import match
from lamet_agent.stages.soft_analysis.physics import (
    correlated_average, endpoint_fit, extract_cs, extract_intrinsic, quadrature,
)


def _attrs(group="wf"):
    return {"replica_frame_id": "synthetic-frame", "resample_lineage": json.dumps({group: group + "-plan"}),
            "resample_id": group + "-plan", "renormalization_scheme": "ratio",
            "gfix": "CG", "parton": "quark", "distribution": "quasi", "bT_unit": "fm"}


def _data(values, dims, coords, **attrs):
    return EnsembleData(None, "bootstrap", list(values), dims, coords, {**_attrs(), **attrs})


def test_hard_coefficients_are_strictly_nlo_and_have_correct_lo_limit():
    x = np.array([0.3, 0.5, 0.7])
    for observable in ("tmdwf", "tmdpdf"):
        h0 = hard_factor(x, observable=observable, momentum_gev=4, alpha_s=0)
        h1 = hard_factor(x, observable=observable, momentum_gev=4, alpha_s=0.1)
        h2 = hard_factor(x, observable=observable, momentum_gev=4, alpha_s=0.2)
        np.testing.assert_allclose(h0, 1)
        np.testing.assert_allclose(h2 - 1, 2 * (h1 - 1))
    delta = coefficient_correction(x, momentum_gev=4)
    np.testing.assert_allclose(hard_factor(x, observable="tmdpdf", momentum_gev=4), 1 + 2 * delta)
    np.testing.assert_allclose(hard_factor(x, observable="tmdwf", momentum_gev=4),
                               1 + delta + coefficient_correction(1 - x, momentum_gev=4))
    q = 2 * 2.58 * np.sqrt(x[:, None] * x[None, :])
    qbar = 2 * 2.58 * np.sqrt((1 - x[:, None]) * (1 - x[None, :]))
    np.testing.assert_allclose(form_factor_kernel(x[:, None], x[None, :], momentum_gev=2.58),
                               1 + sudakov_correction(q) + sudakov_correction(qbar))
    # The small-scale fixed-order endpoint is finite and involves no running coupling pole.
    assert np.isfinite(form_factor_kernel(.05, .05, momentum_gev=2.58))


def test_coefficients_at_natural_scale_and_the_one_loop_scale_evolution():
    # Independent benchmarks from paper v5 A1/A6 at L=0, and A2--A4
    # differentiated with respect to log(mu); constants and running signs
    # are checked separately from the composite-product truncation above.
    assert coefficient_correction(.5, momentum_gev=2) == pytest.approx(-.194075671815384, abs=1e-14)
    assert sudakov_correction(2) == pytest.approx(-.197567977321501, abs=1e-14)
    x, p, mu, h = .4, 3.5, 2., 1e-5
    derivative = (
        coefficient_correction(x, momentum_gev=p, mu=mu * np.exp(h))
        - coefficient_correction(x, momentum_gev=p, mu=mu * np.exp(-h))
    ) / (2 * h)
    a = .293 * (4 / 3) / (4 * np.pi)
    expected = a * (-2 * np.log(mu**2 / (2 * x * p)**2) - 6)
    assert derivative == pytest.approx(expected, abs=1e-10)


def _chain():
    rng = np.random.default_rng(27)
    count = 24
    x = np.linspace(.05, .95, 181)
    gamma = -.35 + rng.normal(0, .012, count)
    amplitudes = 2 + rng.normal(0, .08, count)
    exponents = .8 + rng.normal(0, .035, count)
    shape = amplitudes[:, None] * (x * (1 - x))**exponents[:, None]
    datasets = []
    for index, p in ((8, 3.44), (9, 3.87), (10, 4.30)):
        h = hard_factor(x, observable="tmdwf", momentum_gev=p)
        datasets.append(_data(shape * h * np.exp(gamma[:, None] * np.log(p / 2.58)), ["x"], {"x": x},
                              observable="tmdwf", momentum_index=index, momentum_gev=p, bT_fm=.24))
    return datasets, x, gamma, amplitudes, exponents


def test_cs_recovers_the_physical_kernel_sample_by_sample():
    quasi, _, gamma, _, _ = _chain()
    cs, diagnostics = extract_cs(quasi, {})
    np.testing.assert_allclose(cs.values[:, 0], gamma, atol=2e-14)
    assert diagnostics["systematics_included"] is False
    assert cs.attrs["observable"] == "cs_kernel"


def test_soft_and_matching_preserve_noncanonical_shared_replica_coordinates():
    quasi, _, _, _, _ = _chain()
    for data in quasi:
        data.array = data.array.isel(resample=slice(None, None, -1))
    cs, diagnostics = extract_cs(quasi, {})
    ff = _data(np.ones((cs.n_sample, 1)), ["bT"], {"bT": [.24]},
               **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58})
    ff.array = ff.array.assign_coords(resample=quasi[0].array.coords["resample"])
    soft, _ = extract_intrinsic(quasi, [ff], cs, {})
    matched, _ = match(quasi[0], cs, soft,
                       {"observable": "tmdwf", "lc_x_ls": {"start": .2, "stop": .8}})
    for data in (cs, diagnostics["_pair_samples"], soft, matched):
        np.testing.assert_array_equal(data.array.coords["resample"], quasi[0].array.coords["resample"])


def test_endpoint_fit_windows_include_authored_shoulders_outside_central_region():
    quasi, x, _, _, _ = _chain()
    cs, _ = extract_cs(quasi, {})
    ff = _data(np.ones((cs.n_sample, 1)), ["bT"], {"bT": [.24]},
               **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58})
    params = {"endpoint_fit_windows": [[.2, .4], [.6, .8]]}
    original, diagnostic = extract_intrinsic(quasi, [ff], cs, params)
    for record in diagnostic["bT_records"][0]["momenta"]:
        assert record["endpoint_fit"]["fit_points"] == 82
    # Change only the authored shoulders that used to be silently discarded.
    for data in quasi:
        data.array.values[:, (x < .285) | (x > .715)] *= 1.1
    changed, _ = extract_intrinsic(quasi, [ff], cs, params)
    assert not np.allclose(changed.values, original.values, rtol=1e-3)


def test_default_fourier_grid_keeps_all_closed_window_endpoints():
    x = np.linspace(-1, 2, 301)
    count = 8
    quasi = [_data(np.ones((count, len(x))), ["x"], {"x": x}, observable="tmdwf",
                   momentum_index=index, momentum_gev=p, bT_fm=.24)
             for index, p in ((8, 3.44), (9, 3.87), (10, 4.30))]
    cs, diagnostics = extract_cs(quasi, {})
    pair = diagnostics["bT_records"][0]["pairs"][0]
    assert len(pair["x"]) == 33
    np.testing.assert_allclose([pair["x"][0], pair["x"][-1]], [.34, .66], atol=1e-12)
    soft = _data(np.full((count, 1), .7), ["bT"], {"bT": [.24]}, observable="intrinsic_soft",
                 mu=2., alpha_s=.293, order="nlo")
    wf, _ = match(quasi[0], cs, soft, {"observable": "tmdwf", "lc_x_ls": {"start": .1, "stop": .9}})
    assert len(wf.coords["x"]) == 81
    np.testing.assert_allclose([wf.coords["x"][0], wf.coords["x"][-1]], [.1, .9], atol=1e-12)
    qpdf = _data(np.ones((count, len(x))), ["x"], {"x": x}, observable="tmdpdf",
                 momentum_gev=1.83, bT_fm=.24)
    pdf, _ = match(qpdf, cs, soft, {"observable": "tmdpdf", "lc_x_ls": {"start": .2, "stop": .9}})
    assert len(pdf.coords["x"]) == 71
    np.testing.assert_allclose([pdf.coords["x"][0], pdf.coords["x"][-1]], [.2, .9], atol=1e-12)
    physical = (x > 0) & (x < 1)
    xx = x[physical]
    values = np.linspace(1.9, 2.1, count)[:, None] * (xx * (1 - xx))**.8
    _, fit = endpoint_fit(xx, values, [[.3, .45], [.55, .7]], "bootstrap")
    assert fit["fit_points"] == 32


def test_reloaded_quasi_and_soft_artifacts_keep_numeric_metadata_and_lineage(tmp_path):
    quasi, _, gamma, _, _ = _chain()
    reloaded = []
    for i, data in enumerate(quasi):
        path = tmp_path / f"quasi_{i}.nc"
        data.to_netcdf(path)
        reloaded.append(EnsembleData.from_netcdf(path))
    cs, _ = extract_cs(reloaded, {})
    np.testing.assert_allclose(cs.values[:, 0], gamma, atol=2e-14)
    cs.to_netcdf(tmp_path / "cs.nc")
    cs = EnsembleData.from_netcdf(tmp_path / "cs.nc")
    soft = _data(np.full((len(gamma), 1), .7), ["bT"], {"bT": [.24]}, observable="intrinsic_soft",
                 mu=2., alpha_s=.293, order="nlo")
    soft.to_netcdf(tmp_path / "soft.nc")
    matched, _ = match(reloaded[0], cs, EnsembleData.from_netcdf(tmp_path / "soft.nc"),
                       {"observable": "tmdwf", "lc_x_ls": {"start": .2, "stop": .8}})
    assert matched.n_sample == len(gamma)
    assert json.loads(matched.attrs["resample_lineage"]) == {"wf": "wf-plan"}


def test_equal_weight_average_keeps_covariance_and_separates_the_gls_diagnostic():
    rng = np.random.default_rng(1)
    raw = rng.normal(size=(3, 500))
    estimates = 1 + np.array([[1., 0, 0], [.5, 1.5, 0], [.7, .2, 2.]]) @ raw
    result, metadata = correlated_average(estimates, "bootstrap")
    inverse = np.linalg.inv(np.cov(estimates))
    expected_weights = inverse @ np.ones(3)
    expected_weights /= expected_weights.sum()
    np.testing.assert_allclose(metadata["weights"], np.full(3, 1 / 3), atol=1e-13)
    np.testing.assert_allclose(result, estimates.mean(axis=0), atol=1e-13)
    np.testing.assert_allclose(np.var(result, ddof=1), np.cov(estimates).sum() / 9, atol=1e-13)
    np.testing.assert_allclose(metadata["momentum_consistency"]["fit_weights"], expected_weights, atol=1e-13)
    assert metadata["momentum_consistency"]["covariance_rank"] == 3
    shifted = estimates + np.array([0., 0., 20.])[:, None]
    shifted_result, shifted_metadata = correlated_average(shifted, "bootstrap")
    np.testing.assert_allclose(shifted_result, shifted.mean(axis=0), atol=1e-12)
    assert shifted_metadata["momentum_consistency"]["Q"] < .05
    assert shifted_metadata["momentum_consistency"]["warnings"]
    common = rng.normal(size=500)
    common_result, extrapolated = correlated_average([1 + common, 2 + 2 * common], "bootstrap")
    np.testing.assert_allclose(common_result, 1.5 + 1.5 * common, atol=1e-13)
    assert min(extrapolated["momentum_consistency"]["fit_weights"]) < 0
    assert extrapolated["momentum_consistency"]["fitted_mean"] < common_result.mean()


def test_soft_double_quadrature_matches_independent_integral_benchmarks():
    cut, p, mu, alpha = .05, 2.58, 2., .293
    nodes, matrix, _ = quadrature(cut, .3, .7, 4, p, mu, 0.)
    beam = nodes * (1 - nodes)
    def primitive(x):
        return x**2 / 2 - x**3 / 3

    expected_lo = (primitive(1 - cut) - primitive(cut))**2
    assert beam @ matrix @ beam == pytest.approx(expected_lo, abs=1e-15)

    # HF at NLO separates into products of 1D logarithmic moments of this
    # symmetric beam. Adaptive 1D integration is independent of the 2D
    # Gauss-Legendre implementation and tests its normalization and signs.
    moments = [quad(lambda x: (x * (1 - x))**.8 * np.log(x)**power,
                    cut, 1 - cut, epsabs=1e-12, epsrel=1e-12)[0] for power in range(3)]
    i0, i1, i2 = moments
    logarithm = np.log(mu**2 / (4 * p**2))
    correction = (-(logarithm**2 * i0**2 - 4 * logarithm * i0 * i1 + 2 * i0 * i2 + 2 * i1**2)
                  - 3 * (logarithm * i0**2 - 2 * i0 * i1) + (-8 + np.pi**2 / 6) * i0**2)
    expected_nlo = i0**2 + 2 * alpha * (4 / 3) / (4 * np.pi) * correction
    errors = []
    for order in (4, 16):
        nodes, matrix, _ = quadrature(cut, .3, .7, order, p, mu, alpha)
        beam = (nodes * (1 - nodes))**.8
        errors.append(abs(beam @ matrix @ beam - expected_nlo))
    assert errors[1] < 1e-12
    assert errors[1] < errors[0]


def test_soft_quadrature_must_be_small_relative_to_statistical_uncertainty():
    x = np.linspace(.05, .95, 181)
    amplitude = 2 + np.linspace(-1e-6, 1e-6, 24)
    quasi = []
    for index, p in ((8, 3.44), (9, 3.87), (10, 4.30)):
        values = amplitude[:, None] * (x * (1 - x))**.8 * hard_factor(x, observable="tmdwf", momentum_gev=p)
        quasi.append(_data(values, ["x"], {"x": x}, observable="tmdwf", momentum_index=index,
                           momentum_gev=p, bT_fm=.24))
    cs, _ = extract_cs(quasi, {})
    ff = _data(np.ones((len(amplitude), 1)), ["bT"], {"bT": [.24]},
               **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58})
    # A small relative quadrature shift still matters for sufficiently precise
    # replicas; reporting it as a statistical fluctuation would understate error.
    with pytest.raises(ValueError, match="statistical errors"):
        extract_intrinsic(quasi, [ff], cs, {"quadrature_order": 8})


def test_intrinsic_soft_and_both_matching_outputs_close_factorization():
    quasi, x, gamma, amplitudes, exponents = _chain()
    cs, _ = extract_cs(quasi, {})
    nodes, matrix, _ = quadrature(.05, .3, .7, 96, 2.58, 2, .293)
    beam = amplitudes[:, None] * (nodes * (1 - nodes))**exponents[:, None]
    integral = np.einsum("si,ij,sj->s", beam, matrix, beam)
    intrinsic = .7 + np.linspace(-.02, .02, len(gamma))
    ff = _data((intrinsic * integral)[:, None], ["bT"], {"bT": [.24]},
               **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58})
    soft, diagnostics = extract_intrinsic(quasi, [ff], cs, {})
    np.testing.assert_allclose(soft.values[:, 0], intrinsic, rtol=5e-5)
    assert set(json.loads(soft.attrs["resample_lineage"])) == {"wf", "ff"}
    assert diagnostics["bT_records"][0]["momenta"][0]["ff_momentum_gev"] == 2.58
    fitted = diagnostics["bT_records"][0]["momenta"][0]["endpoint_fit"]["replica_parameters"]
    np.testing.assert_allclose(fitted["mean"], [amplitudes.mean(), exponents.mean()], rtol=1e-5)
    np.testing.assert_allclose(fitted["covariance"], np.cov([amplitudes, exponents]), rtol=1e-4)

    wf, _ = match(quasi[0], cs, soft, {"observable": "tmdwf", "p_ref_gev": 2.58,
                                      "lc_x_ls": {"start": .2, "stop": .8}})
    xx = np.asarray(wf.coords["x"])
    expected = np.sqrt(intrinsic[:, None]) * amplitudes[:, None] * (xx * (1 - xx))**exponents[:, None]
    np.testing.assert_allclose(wf.values, expected, rtol=3e-5)

    p = 2.43
    lc = np.exp(-x)[None, :] * np.linspace(.9, 1.1, len(gamma))[:, None]
    h = hard_factor(x, observable="tmdpdf", momentum_gev=p)
    qpdf = _data(lc / 2 * h / np.sqrt(intrinsic[:, None]) *
                 np.exp(.5 * gamma[:, None] * np.log((2 * x * p)**2 / 4)), ["x"], {"x": x},
                 observable="tmdpdf", momentum_gev=p, bT_fm=.24)
    pdf, meta = match(qpdf, cs, soft, {"observable": "tmdpdf", "lc_x_ls": {"start": .2, "stop": .9}})
    selected = (x >= .2 - 1e-12) & (x <= .9 + 1e-12)
    np.testing.assert_allclose(pdf.values, lc[:, selected], rtol=3e-5)
    assert meta["flavor_factor"] == 2


def test_reused_wavefunction_amplitude_uncertainty_cancels_in_matching():
    # FF has its own independent fluctuations; the qWF normalization is a nuisance shared
    # between the direct matching input and the intrinsic-soft denominator.
    x = np.linspace(.05, .95, 181)
    amplitude = np.linspace(1.8, 2.2, 24)
    shape = x * (1 - x)
    quasi = []
    for index, p in ((8, 3.44), (9, 3.87), (10, 4.30)):
        values = amplitude[:, None] * shape * hard_factor(x, observable="tmdwf", momentum_gev=p)
        quasi.append(_data(values, ["x"], {"x": x}, observable="tmdwf", momentum_index=index,
                           momentum_gev=p, bT_fm=.24))
    cs, _ = extract_cs(quasi, {})
    rng = np.random.default_rng(12)
    ff_values = .1 * (1 + rng.normal(0, .02, len(amplitude)))
    ff = _data(ff_values[:, None], ["bT"], {"bT": [.24]},
               **{**_attrs("ff"), "observable": "form_factor", "momentum_gev": 2.58})
    soft, _ = extract_intrinsic(quasi, [ff], cs, {})
    matched, _ = match(quasi[0], cs, soft, {"observable": "tmdwf", "p_ref_gev": 2.58,
                                          "lc_x_ls": {"start": .2, "stop": .8}})
    # Only FF fluctuations survive: row-wise proportionality tests the branch cancellation.
    normalized = matched.values / np.sqrt(ff_values[:, None])
    np.testing.assert_allclose(normalized, np.broadcast_to(normalized.mean(axis=0), normalized.shape), rtol=1e-7)


def test_no_replica_dropping_and_no_silent_hard_kernel_clipping():
    quasi, _, _, _, _ = _chain()
    quasi[0].array.values[0, 90] = -1
    with pytest.raises(ValueError, match="never clipped or discarded"):
        extract_cs(quasi, {})
    quasi, _, gamma, _, _ = _chain()
    cs, _ = extract_cs(quasi, {})
    soft = _data(np.ones((len(gamma), 1)), ["bT"], {"bT": [.24]}, observable="intrinsic_soft",
                 mu=2., alpha_s=.293, order="nlo")
    qpdf = _data(np.ones((len(gamma), 3)), ["x"], {"x": [.1, .2, .3]}, observable="tmdpdf",
                 momentum_gev=1.83, bT_fm=.24)
    with pytest.raises(ValueError, match="NLO hard factor"):
        match(qpdf, cs, soft, {"observable": "tmdpdf", "lc_x_ls": [.1, .2, .3]})
    soft.array.values[0, 0] = -1
    with pytest.raises(ValueError, match="intrinsic soft function"):
        match(quasi[0], cs, soft, {"observable": "tmdwf", "lc_x_ls": {"start": .3, "stop": .7}})


def test_lineage_and_scale_mismatch_are_rejected():
    quasi, _, _, _, _ = _chain()
    quasi[1].array.attrs["replica_frame_id"] = "different"
    with pytest.raises(ValueError, match="replica frames"):
        extract_cs(quasi, {})
    quasi, _, _, _, _ = _chain()
    cs, _ = extract_cs(quasi, {})
    cs.array.attrs["mu"] = 3.
    with pytest.raises(ValueError, match="mu differs"):
        extract_intrinsic(quasi, [], cs, {})


def test_cg_quark_quasi_provenance_and_physical_b_units_are_required():
    for key, value, message in (("gfix", "GI", "CG quark"), ("parton", "gluon", "CG quark"),
                                 ("distribution", "light_cone", "distribution='quasi'")):
        quasi, _, _, _, _ = _chain()
        quasi[0].array.attrs[key] = value
        with pytest.raises(ValueError, match=message):
            extract_cs(quasi, {})
    quasi, _, _, _, _ = _chain()
    cs, _ = extract_cs(quasi, {})
    cs.array.attrs["bT_unit"] = "lattice"
    soft = _data(np.full((len(cs.values), 1), .7), ["bT"], {"bT": [.24]}, observable="intrinsic_soft",
                 mu=2., alpha_s=.293, order="nlo")
    with pytest.raises(ValueError, match="declare fm units"):
        match(quasi[0], cs, soft, {"observable": "tmdwf", "lc_x_ls": {"start": .3, "stop": .7}})


def test_soft_and_matching_contracts_expose_the_dag():
    from lamet_agent.stages.soft_analysis.contract import JOB_RULES as soft_rules
    from lamet_agent.stages.matching.contract import JOB_RULES as matching_rules
    assert evaluate_rules({"jobs": []}, soft_rules) == []
    job = {"id": "si", "type": "intrinsic_soft", "inputs": {"quasi": ["q1", "q2"]}}
    issues = evaluate_rules({"jobs": [job]}, soft_rules)
    assert {issue.path for issue in issues} >= {"jobs[0].inputs.cs_kernel", "jobs[0].inputs.form_factor"}
    job = {"id": "pdf", "observable": "tmdpdf", "order": "nlo", "mu": 2,
           "lc_x_ls": {"start": .2, "stop": .9},
           "inputs": {"quasi": "q", "cs_kernel": "cs", "intrinsic_soft": "si"}}
    assert evaluate_rules({"jobs": [job]}, matching_rules) == []
    assert job["flavor_factor"] == 2


def _context(path, stage, params, inputs):
    path.mkdir(parents=True)
    return ToolContext(manifest={"metadata": {"sample_error_mode": "covariance"}},
                       manifest_path=path / "manifest.json", stage_id=stage, job_id=path.name,
                       params=params, inputs=inputs, input_summaries={}, state={}, artifact_directory=path,
                       rng=np.random.default_rng(1))


def test_soft_and_tmd_workflow_publish_netcdf_diagnostics_and_stage_reports(tmp_path):
    from lamet_agent.stages.soft_analysis.workflow import run as soft_run
    from lamet_agent.stages.soft_analysis.reporting import write_stage_report
    from lamet_agent.stages._reporting import StageReportRecord
    quasi, _, gamma, _, _ = _chain()
    ctx = _context(tmp_path / "cs", "soft_analysis", {"type": "cs_kernel"}, {"quasi": quasi})
    soft_run(ctx)
    assert EnsembleData.from_netcdf(ctx.artifact_directory / "output.nc").attrs["observable"] == "cs_kernel"
    pairs = EnsembleData.from_netcdf(ctx.artifact_directory / "diagnostics" / "cs_pairs.nc")
    assert pairs.dims == ["bT", "pair", "x"]
    assert pairs.coords["pair"] == ["8_9", "9_10"]
    np.testing.assert_allclose(pairs.values, np.broadcast_to(gamma[:, None, None, None], pairs.values.shape),
                               atol=2e-14)
    assert pairs.attrs["resample_lineage"] == ctx.output.attrs["resample_lineage"]
    diagnostics = json.loads((ctx.artifact_directory / "diagnostics" / "soft.json").read_text())
    assert "_pair_samples" not in diagnostics
    np.testing.assert_allclose(diagnostics["bT_records"][0]["pairs"][0]["gamma_sdev"],
                               np.std(gamma, ddof=1), atol=2e-14)
    record = StageReportRecord(ctx.job_id, ctx.params, ctx.inputs, ctx.output, ctx.summary, ctx.artifact_directory)
    report = write_stage_report(records=(record,), artifact_directory=tmp_path)
    assert Path(report).is_file()
    from lamet_agent.stages.matching.workflow import run as matching_run
    from lamet_agent.stages.matching.reporting import write_stage_report as matching_report
    cs = ctx.output
    soft = _data(np.full((len(cs.values), 1), .7), ["bT"], {"bT": [.24]}, observable="intrinsic_soft",
                 mu=2., alpha_s=.293, order="nlo")
    ctx = _context(tmp_path / "matching" / "wf", "matching",
                   {"observable": "tmdwf", "lc_x_ls": {"start": .2, "stop": .8}},
                   {"quasi": quasi[0], "cs_kernel": cs, "intrinsic_soft": soft})
    matching_run(ctx, None)
    persisted = EnsembleData.from_netcdf(ctx.artifact_directory / "output.nc")
    assert persisted.attrs["distribution"] == "light_cone"
    assert persisted.attrs["p_ref_gev"] == 4.30
    record = StageReportRecord(ctx.job_id, ctx.params, ctx.inputs, ctx.output, ctx.summary, ctx.artifact_directory)
    report = matching_report(records=(record,), artifact_directory=tmp_path / "matching")
    assert "TMD Matching" in report.read_text()
