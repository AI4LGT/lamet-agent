"""Unequal independent jackknife groups across the complete numerical TMD DAG."""

import json

import gvar as gv
import numpy as np
import pytest

from lamet_agent.data import (
    EnsembleData, align_sample_lineage, sample_covariance, sample_mean, sample_sdev,
)
from lamet_agent.stages.correlator_analysis._input import resample_correlators
from lamet_agent.stages.fourier_transform._tmd import transform_tmd
from lamet_agent.stages.matching.tmd import match
from lamet_agent.stages.renormalization.physics import tmd_ratio
from lamet_agent.stages.soft_analysis.physics import (
    correlated_average, endpoint_fit, extract_cs, extract_intrinsic,
)
from lamet_agent.stages.soft_analysis.workflow import replica_distribution


def _group(name, configurations):
    raw = EnsembleData(None, "raw", list(np.asarray(configurations)), [], {})
    return resample_correlators(
        {"correlators": {"value": raw}, "configuration_ids": list(map(str, range(raw.n_sample)))},
        mode="jackknife", group=name, bin_size=1, n_boot=64, seed=1984, tracked=True,
    )["value"]


def _data(source, values, dims, coords, **attrs):
    data = EnsembleData(None, "jackknife", list(np.asarray(values)), dims, coords,
                        {**source.attrs, "observable": "tmdwf", "gfix": "CG", "parton": "quark",
                         "coord_unit": "fm", "phase_rotation_applied": "true", **attrs})
    data.array = data.array.assign_coords(resample=source.array.coords["resample"])
    return data


def _block_covariance(values, layout):
    """Independent oracle: sum the per-ensemble leave-one-out covariances."""
    matrix = np.asarray(values).reshape(len(values), -1)
    covariance = np.zeros((matrix.shape[1], matrix.shape[1]))
    start = 0
    for count in layout.values():
        block = matrix[start:start + count]
        delta = block - block.mean(axis=0)
        covariance += (count - 1) / count * delta.T @ delta
        start += count
    assert start + 1 == len(values)
    return covariance


def test_independent_jackknife_groups_close_the_full_tmd_branch_dag(tmp_path):
    # Unequal counts rule out accidental row-wise alignment of unrelated groups.
    wf = _group("moving", np.linspace(1.7, 2.3, 8))
    rest = _group("rest", [0.7, 0.85, 1., 1.1, 1.35])
    ff = _group("ff", [0.08, 0.09, 0.095, 0.105, 0.11, 0.12])
    b = .24
    reference_target = _data(rest, rest.values[:, None], ["z"], {"z": [0.]},
                             bT_fm=b, momentum_gev=0.)
    reference_local = _data(rest, np.ones((rest.n_sample, 1)), ["z"], {"z": [0.]},
                            bT_fm=0., momentum_gev=0.)
    reference = tmd_ratio(reference_target, reference_local, operation="reference")
    x = np.linspace(.05, .95, 181)
    quasi = []
    # Holding P dz fixed gives the same Fourier shape at each momentum. With
    # alpha_s=0 its known physical CS kernel is identically zero.
    for index, p in ((8, 3.44), (9, 3.87), (10, 4.30)):
        z = np.arange(21) * .18 / p
        shape = np.exp(-.5 * np.arange(21)**2)
        target = _data(wf, wf.values[:, None] * shape, ["z"], {"z": z},
                       bT_fm=b, momentum_gev=p, momentum_index=index)
        local = _data(wf, np.ones((wf.n_sample, 1)), ["z"], {"z": [0.]},
                      bT_fm=0., momentum_gev=p)
        normalized = tmd_ratio(target, local, operation="normalize_wf")
        renormalized = tmd_ratio(normalized, reference, operation="apply")
        transformed = transform_tmd(renormalized, x, observable="tmdwf")
        transformed.to_netcdf(tmp_path / f"quasi_{index}.nc")
        quasi.append(EnsembleData.from_netcdf(tmp_path / f"quasi_{index}.nc"))
    params = {"alpha_s": 0.}
    cs, diagnostics = extract_cs(quasi, params)
    np.testing.assert_allclose(cs.values, 0, atol=2e-14)
    np.testing.assert_allclose(diagnostics["bT_records"][0]["pairs"][0]["gamma_sdev"], 0, atol=2e-13)
    form_factor = _data(ff, ff.values, [], {}, observable="form_factor", bT_fm=b, momentum_gev=2.58)
    soft, diagnostic = extract_intrinsic(quasi, [form_factor], cs, params)
    soft.to_netcdf(tmp_path / "soft.nc")
    soft = EnsembleData.from_netcdf(tmp_path / "soft.nc")
    matched, _ = match(quasi[0], cs, soft,
                       {**params, "observable": "tmdwf", "p_ref_gev": 2.58,
                        "lc_x_ls": {"start": .2, "stop": .8}})
    layout = json.loads(matched.attrs["jackknife_layout"])
    assert layout == {"ff": 6, "moving": 8, "rest": 5}
    assert matched.n_sample == sum(layout.values()) + 1
    assert set(json.loads(matched.attrs["resample_lineage"])) == set(layout)
    aligned_ff, aligned_matched = align_sample_lineage(form_factor, matched)
    # Direct matching and the intrinsic-soft denominator reuse the same WF and
    # zero-momentum normalization. Their fluctuations cancel, including after
    # the FF block is added and the soft artifact makes a NetCDF round trip.
    normalized = aligned_matched.values / np.sqrt(aligned_ff.values[:, None])
    np.testing.assert_allclose(normalized, np.broadcast_to(normalized[-1], normalized.shape), rtol=3e-7)
    expected = normalized[-1][None, :] * np.sqrt(aligned_ff.values[:, None])
    expected_covariance = _block_covariance(expected, layout)
    np.testing.assert_allclose(gv.evalcov(matched.gvar), expected_covariance, rtol=2e-6, atol=1e-14)
    np.testing.assert_allclose(matched.mean, expected[-1], rtol=3e-7)
    # The same known-zero CS construction exercises the PDF branch, including
    # flavor normalization and union with the shared soft/FF blocks.
    quasi_pdf = quasi[0].copy()
    quasi_pdf.array.attrs["observable"] = "tmdpdf"
    matched_pdf, pdf_diagnostic = match(
        quasi_pdf, cs, soft,
        {**params, "observable": "tmdpdf", "zeta_gev2": 4.,
         "lc_x_ls": {"start": .2, "stop": .8}},
    )
    np.testing.assert_allclose(matched_pdf.values, 2 * matched.values, rtol=1e-13)
    np.testing.assert_allclose(matched_pdf.sdev, 2 * matched.sdev, rtol=1e-12)
    assert pdf_diagnostic["flavor_factor"] == 2.
    assert pdf_diagnostic["jackknife_groups"] == layout
    assert diagnostic["bT_records"][0]["momenta"][0]["quadrature_change_over_stat_error"] < .1
    rows = replica_distribution(soft)
    assert rows[0]["mean"] == pytest.approx(soft.values[-1, 0])
    assert rows[0]["sdev"] == pytest.approx(float(soft.sdev[0]))
    assert rows[0]["median"] == pytest.approx(float(np.median(soft.values[:-1, 0])))


def test_endpoint_and_momentum_diagnostics_use_each_jackknife_block_count():
    first = _group("amplitude", [1.7, 1.85, 2., 2.15, 2.3])
    second = _group("exponent", [.65, .75, .8, .85, .95, .8, .8])
    first, second = align_sample_lineage(first, second)
    x = np.linspace(.05, .95, 181)
    expected = np.stack([first.values, second.values], axis=1)
    shape = first.values[:, None] * (x * (1 - x))**second.values[:, None]
    parameters, diagnostic = endpoint_fit(x, shape, [[.3, .45], [.55, .7]], "jackknife", first.attrs)
    np.testing.assert_allclose(parameters, expected, rtol=1e-6)
    covariance = _block_covariance(expected, json.loads(first.attrs["jackknife_layout"]))
    np.testing.assert_allclose(diagnostic["replica_parameters"]["covariance"], covariance,
                               rtol=1e-5, atol=1e-12)
    np.testing.assert_allclose(diagnostic["replica_parameters"]["mean"], expected[-1], rtol=1e-6)
    np.testing.assert_allclose(diagnostic["replica_parameters"]["sdev"], np.sqrt(np.diag(covariance)), rtol=1e-5)

    estimates = np.array([first.values + second.values, 2 * first.values - second.values])
    averaged, metadata = correlated_average(estimates, "jackknife", first.attrs)
    estimate_covariance = _block_covariance(estimates.T, json.loads(first.attrs["jackknife_layout"]))
    inverse = np.linalg.inv(estimate_covariance)
    weights = inverse @ np.ones(2)
    weights /= weights.sum()
    np.testing.assert_allclose(metadata["momentum_consistency"]["fit_weights"], weights, rtol=1e-12)
    assert metadata["momentum_consistency"]["fitted_mean"] == pytest.approx(float(weights @ estimates[:, -1]))
    np.testing.assert_allclose(averaged, estimates.mean(axis=0))
    np.testing.assert_allclose(sample_covariance(estimates.T, "jackknife", first.attrs), estimate_covariance)
    np.testing.assert_allclose(sample_sdev(estimates.T, "jackknife", first.attrs),
                               np.sqrt(np.diag(estimate_covariance)))
    np.testing.assert_allclose(sample_mean(estimates.T, "jackknife", first.attrs), estimates[:, -1])
