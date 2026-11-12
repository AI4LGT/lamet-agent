"""Scalar ratio factors must report the same uncertainty as their one-point form."""

import gvar as gv
import numpy as np
import pytest

from lamet_agent.data import EnsembleData


@pytest.mark.parametrize("resample", ["raw", "bootstrap", "jackknife"])
def test_scalar_covariance_matches_sample_standard_error_and_vector(resample, tmp_path):
    values = np.array([.4, .8, 1.1, 1.7])
    data = EnsembleData(None, resample, list(values), [], {})
    vector = EnsembleData(None, resample, list(values[:, None]), ["bT"], {"bT": [.24]})
    expected = values.std(ddof=1)
    if resample == "raw":
        expected /= len(values)**.5
    elif resample == "jackknife":
        expected *= (len(values) - 1) / len(values)**.5
    covariance = data.average("covariance")
    assert gv.mean(covariance) == pytest.approx(values.mean())
    assert gv.sdev(covariance) == pytest.approx(expected)
    assert gv.sdev(covariance) == pytest.approx(gv.sdev(vector.gvar[0]))
    assert gv.sdev(covariance) == pytest.approx(gv.sdev(data.average("variance")))
    data.to_netcdf(tmp_path / "scalar.nc")
    restored = EnsembleData.from_netcdf(tmp_path / "scalar.nc")
    assert gv.sdev(restored.gvar) == pytest.approx(expected)
