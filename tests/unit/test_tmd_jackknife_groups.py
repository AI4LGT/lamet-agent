"""Independent-group jackknife propagates real leave-one-out blocks through DAGs."""

import json

import gvar as gv
import numpy as np
import pytest

from lamet_agent.data import (
    EnsembleData,
    align_sample_lineage,
    merge_sample_lineage,
    sample_covariance,
    sample_mean,
    sample_sdev,
)
from lamet_agent.stages.correlator_analysis._input import resample_correlators


def sampled(group, raw):
    values = np.asarray(raw, dtype=float)
    dims = [] if values.ndim == 1 else ["x"]
    coords = {} if not dims else {"x": list(range(values.shape[1]))}
    data = EnsembleData(None, "raw", list(values), dims, coords)
    source = {"correlators": {"a": data}, "configuration_ids": list(map(str, range(len(values))))}
    return resample_correlators(source, mode="jackknife", group=group, bin_size=1,
                                n_boot=None, seed=123, tracked=True)["a"]


def rebuild(template, values):
    return EnsembleData(None, "jackknife", list(values), template.dims, template.coords, template.attrs)


def test_native_sampling_is_exact_delete_one_and_seed_independent():
    raw = np.array([2., 4., 7., 10.])
    data = sampled("wf", raw)
    np.testing.assert_allclose(data.values, [(raw.sum() - x) / 3 for x in raw])
    assert data.n_sample == 4
    assert json.loads(data.attrs["jackknife_groups"]) == {"wf": 4}
    assert "jackknife_layout" not in data.attrs
    same = align_sample_lineage(data, data)
    assert same == (data, data)
    np.testing.assert_allclose(data.sdev, raw.std(ddof=1) / np.sqrt(4))


def test_independent_groups_covariances_add_without_artificial_cross_covariance():
    first = sampled("wf", [[1., 3.], [3., 4.], [5., 9.], [8., 10.]])
    second = sampled("rest", [[3., 1.], [6., 7.], [10., 4.]])
    a, b = align_sample_lineage(first, second)
    assert a.n_sample == 8  # 4 + 3 independent rows and one central row
    assert json.loads(a.attrs["jackknife_layout"]) == {"rest": 3, "wf": 4}
    result = a.add(b)
    expected_cov = gv.evalcov(first.gvar) + gv.evalcov(second.gvar)
    np.testing.assert_allclose(gv.evalcov(result.gvar), expected_cov, atol=1e-14)
    np.testing.assert_allclose(result.mean, first.mean + second.mean)
    # Concatenating independent observables must produce a zero covariance block.
    paired = EnsembleData.concat([first, second], "observable", ["wf", "rest"])
    np.testing.assert_allclose(gv.evalcov(paired.gvar)[0, :, 1, :], 0., atol=1e-14)


def test_shared_factor_cancels_after_non_linear_branch_merge():
    wf = sampled("wf", [2., 4., 7., 9.])
    rest = sampled("rest", [1., 2., 4.])
    ff = sampled("ff", [2., 3., 5., 8., 11.])
    reduced = wf.div(rest)
    soft_like = ff.div(reduced.mul(reduced))
    left, right = align_sample_lineage(reduced, soft_like)
    matched = left.mul(rebuild(right, np.sqrt(right.values)))
    expected, _ = align_sample_lineage(ff, matched)
    np.testing.assert_allclose(matched.values, np.sqrt(expected.values), rtol=2e-15)
    expected_cov = sample_covariance(np.sqrt(ff.values), "jackknife")
    np.testing.assert_allclose(gv.evalcov(matched.gvar), expected_cov, atol=1e-14)
    assert json.loads(matched.attrs["jackknife_groups"]) == {"ff": 5, "rest": 3, "wf": 4}


def test_existing_nonlinear_blocks_and_central_row_survive_new_group_alignment():
    a, b = align_sample_lineage(sampled("a", [1., 3., 8.]), sampled("b", [2., 4., 5., 7.]))
    ratio = a.div(b)
    expanded, _ = align_sample_lineage(ratio, sampled("c", [5., 8.]))
    np.testing.assert_array_equal(expanded.values[:7], ratio.values[:7])
    np.testing.assert_array_equal(expanded.values[7:], np.repeat(ratio.values[-1], 3))
    assert expanded.mean == ratio.mean
    assert expanded.sdev == ratio.sdev
    # Its central row is the propagated native center, not a pooled block mean.
    assert ratio.mean == a.mean / b.mean
    assert not np.isclose(ratio.mean, np.mean(ratio.values))


def test_block_covariance_centers_each_group_separately():
    attrs = {"jackknife_layout": '{"a":2,"b":3}'}
    values = np.array([10., 12., -3., -2., -1., 4.])
    expected = .5 * (1 + 1) + 2 / 3 * (1 + 0 + 1)
    np.testing.assert_allclose(sample_covariance(values, "jackknife", attrs), [[expected]])
    np.testing.assert_allclose(sample_sdev(values, "jackknife", attrs), np.sqrt(expected))
    assert sample_mean(values, "jackknife", attrs) == 4.


def test_same_count_independent_groups_are_not_naively_paired():
    a, b = sampled("a", [1., 3., 5.]), sampled("b", [2., 4., 6.])
    with pytest.raises(ValueError, match="aligned"):
        merge_sample_lineage(a, b)
    result = a.add(b)
    assert result.n_sample == 7
    np.testing.assert_allclose(result.sdev ** 2, a.sdev ** 2 + b.sdev ** 2)


def test_plan_conflicts_reordered_rows_and_malformed_layout_are_rejected():
    first = sampled("wf", [1., 3., 7.])
    other = sampled("wf", [1., 3., 7., 11.])
    with pytest.raises(ValueError, match="conflicting resampling plans"):
        align_sample_lineage(first, other)
    reordered = first.copy()
    reordered.array = reordered.array.isel(resample=[2, 1, 0])
    with pytest.raises(ValueError, match="replica order"):
        align_sample_lineage(first, reordered)
    bad = first.copy()
    bad.array.attrs["jackknife_layout"] = '{"wf":3}'
    with pytest.raises(ValueError, match="central row"):
        align_sample_lineage(bad, first)


def test_mixed_group_netcdf_preserves_blocks_center_and_uncertainty(tmp_path):
    first = sampled("wf", [[1., 2.], [3., 6.], [5., 4.]])
    second = sampled("rest", [[2., 3.], [4., 5.], [7., 9.], [8., 12.]])
    data = first.div(second)
    path = tmp_path / "group-jackknife.nc"
    data.to_netcdf(path)
    restored = EnsembleData.from_netcdf(path)
    np.testing.assert_array_equal(restored.values, data.values)
    np.testing.assert_array_equal(restored.mean, data.mean)
    np.testing.assert_array_equal(gv.evalcov(restored.gvar), gv.evalcov(data.gvar))
    assert merge_sample_lineage(restored, data) == merge_sample_lineage(data)
    np.testing.assert_allclose(gv.sdev(restored.gvar_mean), restored.sdev)
    with pytest.raises(ValueError, match="percentiles are undefined"):
        restored.average("one_sigma")
