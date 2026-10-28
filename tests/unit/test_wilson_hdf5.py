"""Wilson HDF5 packing preserves table coordinates and fixed-center fits."""

import h5py
import numpy as np
import pytest

from lamet_agent.stages.tmdwf_analysis.physics import wilson_factor
from lamet_agent.stages.tmdwf_analysis.wilson_io import GROUP, load_wilson, parse_text
from scripts.add_c32p23_wilson_h5 import add_wilson


def table(tmp_path):
    rng = np.random.default_rng(21)
    indices = np.indices((12, 3, 26, 11))
    length = indices[2]
    values = 1.2 * np.exp(-0.08 * length) * (1 + 0.2 * np.exp(-0.3 * length))
    values *= rng.normal(1, 0.02, values.shape)
    data = np.column_stack([100 + indices[0].ravel(), indices[1].ravel(), indices[2].ravel(),
                            indices[3].ravel(), values.ravel()])
    path = tmp_path / "wilson.txt"
    np.savetxt(path, data, fmt=["%d", "%d", "%d", "%d", "%.17g"])
    h5 = tmp_path / "combined.h5"
    with h5py.File(h5, "w") as handle:
        handle.create_dataset("2pt/data", data=np.arange(24).reshape(4, 6))
        handle.attrs["resample_mode"] = "bootstrap"
    return path, h5, values


def test_wilson_pack_preserves_text_coordinates_and_existing_correlators(tmp_path):
    path, h5, expected = table(tmp_path)
    result = add_wilson(path, h5)
    assert result["exact_numeric_match"]
    values, provenance = load_wilson(h5)
    np.testing.assert_array_equal(values, expected)
    assert provenance["source_sha256"] == load_wilson(path)[1]["source_sha256"]
    with h5py.File(h5, "r") as handle:
        np.testing.assert_array_equal(handle["2pt/data"][()], np.arange(24).reshape(4, 6))
        assert handle.attrs["resample_mode"] == "bootstrap"
        group = handle[GROUP]
        assert group["source_text"][()].tobytes() == path.read_bytes()
        assert group.attrs["sample_kind"] == "raw_configuration"
        np.testing.assert_array_equal(group["configuration"][()], np.arange(100, 112))
        np.testing.assert_array_equal(group["mean"][()], expected.mean(axis=1).mean(axis=0))
    assert add_wilson(path, h5) == result  # An identical repeated import is safe.


def test_wilson_hdf5_produces_identical_fixed_center_factors(tmp_path):
    path, h5, _ = table(tmp_path)
    add_wilson(path, h5)
    cfg = {"b": 1, "a_fm": 0.1053, "L": 6, "ZO": 0.92208, "fit_start": 5, "fit_stop": 20,
           "priors": {"c": [0, 10], "E": [0, 10], "c1": [10, 10], "delta_E": [1, 1]}, "svdcut": 1e-8}
    text, text_diag = wilson_factor(path, cfg, 20)
    stored, h5_diag = wilson_factor(h5, cfg, 20)
    np.testing.assert_array_equal(stored.values, text.values)
    np.testing.assert_array_equal(stored.values, np.broadcast_to(stored.values[0], stored.values.shape))
    assert stored.attrs["factor_uncertainty"] == "fixed_central_value_not_propagated"
    assert stored.attrs["wilson_error_estimator"] == "leave_one_out_jackknife"
    assert h5_diag["fit_mode"] == "single_mean_fit_with_diagonal_errors"
    assert h5_diag["parameters"] == text_diag["parameters"]
    assert h5_diag["Q"] == text_diag["Q"]


def test_wilson_rejects_bad_row_order_or_conflicting_import(tmp_path):
    path, h5, _ = table(tmp_path)
    add_wilson(path, h5)
    lines = path.read_text().splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    with pytest.raises(ValueError, match="axis order"):
        parse_text("\n".join(lines).encode())
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        add_wilson(path, h5)
    with h5py.File(h5, "r") as handle:
        assert handle[GROUP + "/source_text"][()].tobytes() == original
