"""Bootstrap preservation and numerical interface checks for the TMDWF extension."""

import copy
import json

import h5py
import numpy as np
import pytest

from lamet_agent.agent import create_session
from lamet_agent.contract import CheckContext
from lamet_agent.manifest import load_manifest
from lamet_agent.stages.tmdwf_analysis.contract import check
from lamet_agent.stages.tmdwf_analysis.physics import ensemble, fit_one_z, read_ratios, renormalize
from lamet_agent.ui import PlainUi


def pair():
    attrs = {"ensemble_id": "C32P23", "a_fm": 0.1, "b_perp_lattice": 1,
             "resample_id": "paired", "coord_unit": "fm"}
    target = ensemble(np.arange(18).reshape(6, 3) + 1j, {"z": [0, 0.1, 0.2]}, attrs, "bare")
    attrs = {k: v for k, v in attrs.items() if k != "resample_id"}
    attrs.update({"factor_uncertainty": "fixed_central_value_not_propagated", "ZO": 1.0, "L": 6})
    factor = ensemble(np.tile([2., 3., 4.], (6, 1)), target.coords, attrs, "denominator")
    return target, factor


def test_renormalization_preserves_samples_metadata_and_no_z0_normalization(tmp_path):
    target, factor = pair()
    result = renormalize(target, factor)
    np.testing.assert_array_equal(result.values, target.values / factor.values)
    assert result.attrs["resample_id"] == "paired"
    assert result.attrs["renormalization_scheme"] == "tmdwf_wilson_loop"
    assert result.attrs["normalized_at_origin"] == 0
    result.to_netcdf(tmp_path / "result.nc")
    restored = type(result).from_netcdf(tmp_path / "result.nc")
    np.testing.assert_array_equal(restored.values, result.values)
    assert restored.attrs["renormalization_scheme"] == "tmdwf_wilson_loop"


def test_mismatched_wilson_b_rejected():
    target, factor = pair()
    factor.array.attrs["b_perp_lattice"] = 2
    with pytest.raises(ValueError, match="b_perp"):
        renormalize(target, factor)


def test_one_state_matches_analytic_gaussian_posterior():
    rng = np.random.default_rng(12)
    samples = rng.normal(0.7, 0.1, (40, 15)) + 1j * rng.normal(-0.2, 0.08, (40, 15))
    fit, _ = fit_one_z((samples, [1, 3, 14, 4, 14], 1e-3))
    for observed, component, start, mean in ((fit.real, samples.real, 3, 1), (fit.imag, samples.imag, 4, -1)):
        data = component[:, start:14]
        variance = data.std(axis=0)**2
        expected = (np.sum(data / variance, axis=1) + mean / 4) / (np.sum(1 / variance) + 1 / 4)
        np.testing.assert_allclose(observed, expected, rtol=1e-9, atol=1e-10)


def test_bootstrap_loader_phase_and_source_pairing(tmp_path):
    cfg = {"pz": 5, "p1": 3, "p2": -2, "b": 1, "a_fm": 0.1053, "L_s": 32}
    path = tmp_path / "input.h5"
    expected_channels = []
    with h5py.File(path, "w") as handle:
        handle.attrs.update(resample_mode="bootstrap", bootstrap_samples=6, source_sha256="test")
        for direction in (0, 1):
            for channel, operator in enumerate(("gtg5", "gzg5")):
                prefix = f"{operator}/gamma_sink0/PX0PY0PZ5/p1_3_p2_-2"
                local = (np.arange(6) + 2)[None, :] * np.ones((15, 1))
                handle.create_dataset(f"{prefix}/bT0/zdir{direction}/bz0", data=local + 9j)
                parts = []
                for z in range(13):
                    raw = local * (1 + channel + 0.1 * direction + 0.01 * z + 0.2j)
                    handle.create_dataset(f"{prefix}/bT1/zdir{direction}/bz{z}", data=raw)
                    theta = 2 * np.pi * (3 - 5 / 2) * z / 32 * (-1)**direction
                    parts.append((raw / local).T * (np.cos(theta) + 1j * np.sin(theta)))
                expected_channels.append(np.stack(parts, axis=1))
    result, attrs = read_ratios(path, cfg, 6)
    np.testing.assert_allclose(result, np.mean(expected_channels, axis=0), rtol=1e-14)
    assert result.shape == (6, 13, 15)
    assert attrs["resample_id"] == "test"
    with pytest.raises(ValueError, match="bootstrap"):
        read_ratios(path, cfg, 7)


def test_contract_rejects_wrong_input_role_and_raw_sampling():
    context = CheckContext({"metadata": {"target_observable": "tmdwf", "resample_mode": "jackknife", "bin_size": 1}},
                           "tmdwf_analysis", "renormalize", {"operation": "renormalize", "settings": {}},
                           {"target": "earlier"})
    assert {issue.path for issue in check(context)} == {"inputs", "operation"}


def test_actual_agent_loop_offline_publishes_renormalization(tmp_path):
    target, factor = pair()
    target.to_netcdf(tmp_path / "bare.nc")
    factor.to_netcdf(tmp_path / "factor.nc")
    document = {"metadata": {"run_id": "test", "root_directory": ".", "artifacts_directory": "out",
                             "random_seed": 0, "workers": 1, "target_observable": "tmdwf",
                             "resample_mode": "bootstrap", "samples": 6, "bin_size": 1,
                             "sample_error_mode": "variance"},
                "stages": {"tmdwf_analysis": {"jobs": [
                    {"id": "rn", "operation": "renormalize", "settings": {},
                     "inputs": {"target": {"file": "bare.nc"}, "denominator": {"file": "factor.nc"}}}]}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))

    class Backend:
        identity = "no-llm"

        def complete(self, **kwargs):
            pytest.fail("Deterministic stage called the LLM")

    result = create_session(Backend(), ui=PlainUi()).run_manifest(load_manifest(path))
    assert result["outputs"]["rn"].n_sample == 6
    assert (tmp_path / "out/01_tmdwf_analysis/rn/diagnostics.json").is_file()
    assert (tmp_path / "out/01_tmdwf_analysis/report.md").is_file()
    invalid = copy.deepcopy(document)
    invalid["stages"]["tmdwf_analysis"]["jobs"][0]["settings"]["normalization"] = True
    path.write_text(json.dumps(invalid))
    assert load_manifest(path).validate()
