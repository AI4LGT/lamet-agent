"""Wilson strategy contracts and normal agent-loop publication without a separate stage."""

import copy
import json

import numpy as np
import pytest

from lamet_agent.agent import create_session
from lamet_agent.contract import CheckContext
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.manifest import load_manifest
from lamet_agent.stages.correlator_analysis.physics import average_tmdwf_channels
from lamet_agent.stages.fourier_transform.physics import prepare_tmdwf_projection
from lamet_agent.stages.renormalization.contract import check_wilson
from lamet_agent.stages.renormalization.physics import apply_wilson_factor
from lamet_agent.ui import PlainUi


def sources():
    info = EnsembleInfo("C32P23", "test", 0.1, 0.1, 32, 64, 0.23)
    attrs = {"bT": 1, "phase_convention": "midpoint", "resample_id": "paired", "coord_unit": "lattice",
             "momentum_gev": info.k_s * 4, "gfix": "GI", "parton": "quark", "hadron": "pion"}
    values = np.arange(18).reshape(6, 3) / 20 + 0.2j
    channels = [EnsembleData(info, "bootstrap", list(values + 0.1 * i), ["z"], {"z": [0, 1, 2]},
                             attrs={**attrs, "source_operator": op, "zdir": direction})
                for i, (op, direction) in enumerate((op, d) for op in ("gtg5", "gzg5") for d in (0, 1))]
    factor = EnsembleData(None, "bootstrap", list(np.tile([2., 3., 4.], (6, 1))), ["z"], {"z": [0, 0.1, 0.2]},
                          attrs={"ensemble_id": info.id, "a_fm": info.a_s, "b_perp_lattice": 1,
                                 "factor_uncertainty": "fixed_central_value_not_propagated", "ZO": 1., "L": 6})
    return channels, factor


def test_wilson_application_preserves_complex_bootstrap_ensemble_and_origin():
    channels, factor = sources()
    target = average_tmdwf_channels(channels)
    result = apply_wilson_factor(target, factor)
    np.testing.assert_array_equal(result.values, np.mean([v.values for v in channels], axis=0) / factor.values)
    assert result.ensemble == target.ensemble
    assert result.attrs["resample_id"] == "paired"
    assert result.attrs["normalized_at_origin"] == 0
    assert result.coords == factor.coords
    factor.array.attrs["b_perp_lattice"] = 2
    with pytest.raises(ValueError, match="b_perp"):
        apply_wilson_factor(target, factor)
    factor.array.attrs["b_perp_lattice"] = 1
    factor.array.data[1, 0] += 0.1
    with pytest.raises(ValueError, match="identical"):
        apply_wilson_factor(target, factor)


def test_projection_rejects_unrenormalized_data_and_double_phase_rotation():
    channels, factor = sources()
    target = average_tmdwf_channels(channels)
    with pytest.raises(ValueError, match="Wilson-renormalized"):
        prepare_tmdwf_projection(target, "real", "even_both_notebook")
    data = apply_wilson_factor(target, factor)
    projected = prepare_tmdwf_projection(data, "real", "even_both_notebook")
    with pytest.raises(ValueError, match="Wilson-renormalized"):
        prepare_tmdwf_projection(projected, "real", "even_both_notebook")


def test_wilson_contract_rejects_origin_normalization_and_missing_channels():
    context = CheckContext({"metadata": {"target_observable": "tmdwf", "resample_mode": "bootstrap", "bin_size": 1}},
                           "renormalization", "rn", {"strategy": "wilson_loop", "type": "apply", "normalization": True},
                           {"targets": ["one"], "denominator": "factor"})
    assert {i.path for i in check_wilson(context)} == {"normalization", "inputs.targets"}


def test_existing_agent_loop_publishes_wilson_application_and_report(tmp_path):
    channels, factor = sources()
    for i, data in enumerate(channels):
        data.to_netcdf(tmp_path / f"channel{i}.nc")
    factor.to_netcdf(tmp_path / "factor.nc")
    document = {"metadata": {"run_id": "test", "root_directory": ".", "artifacts_directory": "out",
                             "random_seed": 0, "workers": 1, "target_observable": "tmdwf", "parton": "quark",
                             "resample_mode": "bootstrap", "samples": 6, "bin_size": 1,
                             "sample_error_mode": "variance"},
                "stages": {"renormalization": {"jobs": [
                    {"id": "rn", "strategy": "wilson_loop", "type": "apply", "normalization": False,
                     "inputs": {"targets": [{"file": f"channel{i}.nc"} for i in range(4)],
                                "denominator": {"file": "factor.nc"}}}]}}}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))

    class Backend:
        identity = "no-llm"

        def complete(self, **kwargs):
            pytest.fail("Deterministic renormalization called the LLM")

    result = create_session(Backend(), ui=PlainUi()).run_manifest(load_manifest(path))
    output = result["outputs"]["rn"]
    assert output.n_sample == 6 and output.ensemble == channels[0].ensemble
    assert output.attrs["resample_id"] == "paired"
    report = (tmp_path / "out/01_renormalization/report.md").read_text(encoding="utf-8")
    assert "Wilson-loop prescription" in report
    assert "self-renormalization factor was extracted" not in report
    assert (tmp_path / "out/01_renormalization/rn/diagnostics.json").is_file()
    invalid = copy.deepcopy(document)
    invalid["stages"]["renormalization"]["jobs"][0]["normalization"] = True
    path.write_text(json.dumps(invalid))
    assert load_manifest(path).validate()
