import json

import numpy as np
import pytest

from lamet_agent.contract import evaluate_rules
from lamet_agent.data import EnsembleData, EnsembleInfo, merge_sample_lineage
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.stages.fourier_transform._tmd import transform_tmd
from lamet_agent.stages.fourier_transform.contract import PARAM_RULES


def _data(observable, direction=(1, 0, 0)):
    ensemble = EnsembleInfo("HISQ", "test", 0.06, 0.06, 48, 64, 0.67)
    z = np.arange(21) * 0.06 * np.linalg.norm(direction)
    values = np.exp(-3 * z**2)
    attrs = {
        "observable": observable,
        "phase_rotation_applied": "true",
        "coord_unit": "fm",
        "momentum_gev": 3.0,
        "bT_fm": 0.24,
        "longitudinal_direction": json.dumps(direction),
        "replica_frame_id": "frame",
        "resample_lineage": '{"raw":"plan"}',
    }
    return EnsembleData(ensemble, "bootstrap", [values, 1.1 * values, 0.9 * values], ["z"], {"z": list(z)}, attrs=attrs)


def test_centered_wave_function_and_replica_linearity():
    data = _data("tmdwf")
    x = np.linspace(0, 1, 101)
    result = transform_tmd(data, x, observable="tmdwf")
    np.testing.assert_allclose(result.values, result.values[:, ::-1], atol=1e-14)
    np.testing.assert_allclose(result.values[1], 1.1 * result.values[0], atol=1e-14)
    expected = (data.values[0, 0] + 2 * data.values[0, 1:].sum()) * 3 * 0.06 / (2 * np.pi * HBAR_C_GEV_FM)
    assert result.values[0, 50] == pytest.approx(expected)
    assert result.attrs["resample_lineage"] == data.attrs["resample_lineage"]


def test_fourier_preserves_replica_identity_and_rejects_misaligned_downstream_branch():
    original = _data("tmdwf")
    reversed_data = original.copy()
    reversed_data.array = reversed_data.array.isel(resample=slice(None, None, -1))
    x = np.array([.3, .5, .7])
    result = transform_tmd(reversed_data, x, observable="tmdwf")
    canonical = transform_tmd(original, x, observable="tmdwf")
    np.testing.assert_array_equal(result.array.coords["resample"], reversed_data.array.coords["resample"])
    np.testing.assert_allclose(result.values, canonical.values[::-1])
    with pytest.raises(ValueError, match="replica order"):
        merge_sample_lineage(result, canonical)


def test_off_axis_spacing_and_taper_are_not_lattice_spacing():
    data = _data("tmdpdf", (1, 1, 0))
    result = transform_tmd(data, np.array([0.0, 0.3, 0.5]), observable="tmdpdf")
    assert result.ensemble.a_s == 0.06
    assert result.attrs["z_step_fm"] == pytest.approx(np.sqrt(2) * 0.06)
    z = np.asarray(data.coords["z"])
    taper = np.clip((20 - np.arange(21)) / 7, 0, 1)
    weights = 2 * taper
    weights[0] = 1
    direct = np.sum(data.values[0] * weights * np.cos(0.3 * 3 * z / HBAR_C_GEV_FM))
    direct *= 3 * np.sqrt(2) * 0.06 / (2 * np.pi * HBAR_C_GEV_FM)
    assert result.values[0, 1] == pytest.approx(direct)


def test_tmd_contract_does_not_request_tail_parameters():
    params = {"strategy": "tmd", "observable": "tmdwf"}
    issues = evaluate_rules(
        params, PARAM_RULES, complete=True, root_document={"metadata": {"target_observable": "tmd"}}
    )
    assert not issues
    assert "zmin_fm" not in params
    assert params["quasi_y_ls"]["num"] == 301


def test_nonfinite_fourier_input_is_rejected():
    data = _data("tmdwf")
    data.array.values[1, 3] = np.nan
    with pytest.raises(ValueError, match="replica/z index"):
        transform_tmd(data, np.array([0.3, 0.5]), observable="tmdwf")


def test_wave_function_requires_centered_phase_provenance():
    data = _data("tmdwf")
    data.array.attrs.pop("phase_rotation_applied")
    with pytest.raises(ValueError, match="phase rotation"):
        transform_tmd(data, np.array([0.3, 0.5]), observable="tmdwf")


def test_scalar_normalization_can_be_reviewed():
    from lamet_agent.agent import _review_output_summary

    scalar = _data("tmdwf").at("z", 0.0)
    result = _review_output_summary(scalar, "covariance", max_points=60)
    assert result["dims"] == []
    assert result["sampling"]["sent_points"] == 1


def test_hermitian_pdf_matches_full_complex_lattice_sum_with_reference_sign():
    data = _data("tmdpdf", (1, 1, 0))
    z = np.asarray(data.coords["z"])
    phase = .4 * data.attrs["momentum_gev"] * z / HBAR_C_GEV_FM
    data.array = data.array.astype(complex)
    data.array.values[:] *= np.exp(1j * phase)[None, :]
    data.array.attrs["output_component"] = "both"
    data.array = data.array.isel(resample=slice(None, None, -1))
    x = np.array([-.4, 0., .4])
    result = transform_tmd(data, x, observable="tmdpdf", projection="hermitian")
    taper = np.clip((20 - np.arange(len(z))) / 7, 0, 1)
    positive = data.values * taper
    full_z = np.concatenate((-z[:0:-1], z))
    full_values = np.concatenate((positive[:, :0:-1].conj(), positive), axis=1)
    direct = full_values @ np.exp(-1j * np.outer(x, data.attrs["momentum_gev"] * full_z / HBAR_C_GEV_FM)).T
    direct *= data.attrs["momentum_gev"] * (z[1] - z[0]) / (2 * np.pi * HBAR_C_GEV_FM)
    np.testing.assert_allclose(direct.imag, 0, atol=1e-14)
    np.testing.assert_allclose(result.values, direct.real, atol=1e-14)
    assert np.all(result.values[:, -1] > result.values[:, 0])  # The complex phase shifts toward positive x.
    assert result.attrs["sector"] == "full"
    assert result.attrs["source_component"] == "both"
    assert result.attrs["output_component"] == "re"
    assert result.attrs["fourier_phase_sign"] == -1
    assert result.ensemble.a_s == .06
    np.testing.assert_array_equal(result.array.coords["resample"], data.array.coords["resample"])


def test_real_even_default_is_bitwise_identical_with_complex_input():
    data = _data("tmdpdf", (1, 1, 0))
    data.array = data.array.astype(complex)
    data.array.values[:] += 1j * np.arange(21)[None, :] / 10
    x = np.linspace(-1, 2, 301)
    default = transform_tmd(data, x, observable="tmdpdf")
    authored = transform_tmd(data, x, observable="tmdpdf", projection="real_even")
    z = np.asarray(data.coords["z"])
    step = float(z[1] - z[0])
    taper = np.clip((20 - z / step) / (20 - 13), 0., 1.)
    weights = taper * 2.
    weights[0] = taper[0]
    phase = x[:, None] * data.attrs["momentum_gev"] * z[None, :] / HBAR_C_GEV_FM
    matrix = np.cos(phase) * weights * (data.attrs["momentum_gev"] * step / (2 * np.pi * HBAR_C_GEV_FM))
    np.testing.assert_array_equal(default.values, data.values.real @ matrix.T)
    np.testing.assert_array_equal(default.values, authored.values)
    assert default.attrs["sector"] == "valence"
    assert default.attrs["fourier_phase_sign"] == 1


@pytest.mark.parametrize("provenance", [None, "re", "im"])
def test_hermitian_pdf_requires_both_fitted_channels(provenance):
    data = _data("tmdpdf")
    if provenance is not None:
        data.array.attrs["output_component"] = provenance
    with pytest.raises(ValueError, match="fit both real and imaginary"):
        transform_tmd(data, np.array([.2, .5]), observable="tmdpdf", projection="hermitian")


def test_hermitian_projection_rejects_wf_and_nonfinite_imaginary_channel():
    with pytest.raises(ValueError, match="only for tmdpdf"):
        transform_tmd(_data("tmdwf"), np.array([.2, .5]), observable="tmdwf", projection="hermitian")
    data = _data("tmdpdf")
    data.array = data.array.astype(complex)
    data.array.attrs["output_component"] = "both"
    data.array.values[1, 3] = complex(1, np.nan)
    with pytest.raises(ValueError, match="replica/z index"):
        transform_tmd(data, np.array([.2, .5]), observable="tmdpdf", projection="hermitian")


def test_hermitian_pdf_contract_is_explicit_and_wf_keeps_default():
    from types import SimpleNamespace
    from lamet_agent.stages.fourier_transform.contract import check_tmd
    metadata = {"metadata": {"target_observable": "tmd"}}
    params = {"strategy": "tmd", "observable": "tmdpdf", "projection": "hermitian"}
    assert evaluate_rules(params, PARAM_RULES, complete=True, root_document=metadata) == []
    assert check_tmd(SimpleNamespace(params=params, manifest=metadata)) is None
    params["observable"] = "tmdwf"
    assert check_tmd(SimpleNamespace(params=params, manifest=metadata)).path == "projection"
    params.pop("projection")
    assert evaluate_rules(params, PARAM_RULES, complete=True, root_document=metadata) == []
    assert params["projection"] == "real_even"
