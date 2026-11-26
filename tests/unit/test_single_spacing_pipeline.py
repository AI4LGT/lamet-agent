"""Synthetic integration checks through the public manifest runner (no LLM)."""

import json

import numpy as np
import pytest

from lamet_agent.agent import create_session
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM as HC
from lamet_agent.manifest import load_manifest


class NoLlm:
    identity = "test:deterministic"

    def complete(self, **kwargs):
        raise AssertionError("A fully authored smoke test must not call an LLM")


@pytest.mark.parametrize("observable,coefficient", [("pdf", "nlo"), ("gpd", "nnlo")])
def test_single_spacing_manifest_to_matching(tmp_path, monkeypatch, observable, coefficient):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(321)
    z = np.arange(17) * 0.09
    count = 40
    total = 0.50 + rng.normal(0, 0.002, count)
    delta = 0.70 + rng.normal(0, 0.003, count)
    kernel_id = f"z_msbar_pdf_{coefficient}"
    c0 = np.ones_like(z)
    c0[1:] = load_renormalization_kernel(kernel_id)(z[1:], mu=2.0)
    ref = np.where(
        z[None, :] <= 0.27,
        c0[None, :] * np.exp(-total[:, None] * z / HC),
        0.8 * np.exp(-delta[:, None] * z / HC),
    )
    ref *= np.exp(rng.normal(0, 0.0001, ref.shape))
    info = EnsembleInfo("synthetic", "a09", 0.09, 0.09, 64, 128, 0.14)
    attrs = {
        "coord_unit": "fm",
        "resample_id": "common",
        "observable": "pdf",
        "gfix": "gi",
        "operator": "gt",
        "kernel_operator": "gt",
        "parton": "quark",
        "polarization": "unpolarized",
        "hadron": "proton",
        "momentum_gev": 0.0,
    }
    EnsembleData(info, "bootstrap", list(ref), ["z"], {"z": z.tolist()}, attrs=attrs).to_netcdf("ref.nc")
    local = (1.0 if observable == "pdf" else 0.73) + rng.normal(0, 0.002, count)
    decay = 0.25 + rng.normal(0, 0.001, count)
    signal = local[:, None] * np.exp(-decay[:, None] * z / HC)
    signal = signal.astype(complex) * np.exp(0.12j * np.minimum(z / 0.54, 1))
    signal *= np.exp(rng.normal(0, 0.001, signal.shape))
    # Invert the intended hybrid factor independently to generate bare inputs.
    denominator = ref[:, np.minimum(np.arange(len(z)), 3)]
    bare = signal * denominator * np.exp(-total[:, None] * np.maximum(z - 0.27, 0) / HC)
    target_attrs = {
        **attrs,
        "observable": observable,
        "momentum_gev": 8 * info.k_s,
        "source_momentum": json.dumps([-1, 0, 7]),
        "sink_momentum": json.dumps([1, 0, 9]),
    }
    EnsembleData(info, "bootstrap", list(bare), ["z"], {"z": z.tolist()}, attrs=target_attrs).to_netcdf("target.nc")
    partner_attrs = {
        **target_attrs,
        "source_momentum": target_attrs["sink_momentum"],
        "sink_momentum": target_attrs["source_momentum"],
    }
    EnsembleData(info, "bootstrap", list(bare), ["z"], {"z": z.tolist()}, attrs=partner_attrs).to_netcdf("partner.nc")
    apply_inputs = {"target": {"file": "target.nc"}, "denominator": {"file": "ref.nc"}, "masses": "mass"}
    jobs = [
        {
            "id": "mass",
            "type": "fit",
            "inputs": {"reference": {"file": "ref.nc"}},
            "kernel_id": kernel_id,
            "mu": 2.0,
            "long_distance_window_fm": [0.54, 1.44],
            "short_distance_points_fm": [0.09, 0.18, 0.27],
        },
        {"id": "renormalized", "type": "apply", "inputs": apply_inputs, "zs_fm": 0.27},
    ]
    ft_inputs = {"input": "renormalized"}
    if observable == "gpd":
        jobs.append(
            {
                "id": "partner",
                "type": "apply",
                "zs_fm": 0.27,
                "inputs": {**apply_inputs, "target": {"file": "partner.nc"}},
            }
        )
        ft_inputs["hermitian_partner"] = "partner"
    document = {
        "metadata": {
            "run_id": "single_spacing_smoke",
            "root_directory": ".",
            "artifacts_directory": "artifacts",
            "random_seed": 7,
            "workers": 1,
            "target_observable": observable,
            "parton": "quark",
            "resample_mode": "bootstrap",
            "samples": count,
            "bin_size": 1,
            "sample_error_mode": "variance",
        },
        "stages": {
            "renormalization": {
                "defaults": {
                    "strategy": "single_spacing_self_renormalization",
                    "scheme": "hybrid",
                    "normalization": False,
                },
                "jobs": jobs,
            },
            "fourier_transform": {
                "jobs": [
                    {
                        "id": "fourier",
                        "inputs": ft_inputs,
                        "quasi_y_ls": {"start": -2.0, "stop": 2.0, "num": 40},
                        "zmin_fm": [0.54],
                        "zmax_fm": [1.26],
                        "zmax_ext_fm": 1.44,
                        "smooth": "none",
                        "scheme_scan": {
                            "order": ["LA"],
                            "sector": "full",
                            "Lambda0_gev": 0.1,
                            "posterior_prior_error_scale": [3.0],
                            "model_average": False,
                            "q_min": 0.0,
                        },
                    }
                ]
            },
            "matching": {
                "jobs": [
                    {
                        "id": "matched",
                        "inputs": {"quasi": "fourier"},
                        "order": "nlo",
                        "resummation": "",
                        "mu": 2.0,
                        "lc_x_ls": {"start": -1.0, "stop": 1.0},
                    }
                ]
            },
        },
    }
    if observable == "gpd":
        document["stages"]["matching"]["jobs"][0]["kernel_parameters"] = {"skewness": 0.125}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    manifest = load_manifest(path)
    assert not manifest.validate()
    session = create_session(NoLlm(), progress_mode="none")
    try:
        result = session.run_manifest(manifest)
    finally:
        session.close()
    outputs = result["outputs"]
    np.testing.assert_allclose(outputs["mass"].mean[:3], [0.70, -0.20, 0.50], atol=0.003)
    np.testing.assert_allclose(outputs["renormalized"].values, signal, rtol=0.01, atol=1e-8)
    np.testing.assert_allclose(outputs["renormalized"].values[:, 0], signal[:, 0], rtol=1e-12)
    for name in ("mass", "renormalized", "fourier", "matched"):
        output = outputs[name]
        assert output.n_sample == count
        assert np.all(np.isfinite(output.values))
        assert np.any(output.real.sdev > 0)
    assert outputs["fourier"].attrs["zs_fm"] == 0.27
    assert outputs["fourier"].attrs["renormalization_scheme"] == "hybrid"
    assert outputs["matched"].dims == ["x"]
    assert set(result["stage_reports"]) == {"renormalization", "fourier_transform", "matching"}
    for job in manifest.jobs:
        saved = EnsembleData.from_netcdf(job.artifact_directory / "output.nc")
        np.testing.assert_allclose(saved.values, outputs[job.job_id].values)
