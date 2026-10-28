"""Wilson factor fitting and application within the renormalization workflow."""

import json

import numpy as np

from lamet_agent.stages.correlator_analysis.physics import average_tmdwf_channels
from lamet_agent.stages.renormalization.physics import apply_wilson_factor, load_data, wilson_factor
from lamet_agent.ui import log


def run(context):
    """Fit one central denominator or apply it to the average of paired channel fits."""
    params = context.params
    samples = int(context.manifest["metadata"]["samples"])
    if params["normalization"]:
        raise ValueError("Wilson renormalization must preserve the non-unit b>0 origin")
    if params["type"] == "fit":
        log("Wilson loop: fit the raw mean once with jackknife errors; retain only the central factor")
        data, diagnostics = wilson_factor(context.inputs["raw"], params["settings"], samples)
    else:
        average = average_tmdwf_channels([load_data(value) for value in context.inputs["targets"]])
        data = apply_wilson_factor(average, load_data(context.inputs["denominator"]))
        diagnostics = {"factor_uncertainty": data.attrs["factor_uncertainty"],
                       "origin_normalization": False, "source_combination": data.attrs["source_combination"]}
    if data.n_sample != samples or not np.isfinite(data.values).all():
        raise ValueError("Wilson jobs must preserve the sample count and publish finite values")
    folder = context.artifact_directory
    data.to_netcdf(folder / "output.nc")
    (folder / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    (folder / "settings.json").write_text(json.dumps(params, indent=2), encoding="utf-8")
    context.finish(data, {
        "stage_id": context.stage_id, "job_id": context.job_id,
        "result": "wilson_factor" if params["type"] == "fit" else "renormalized_matrix_element",
        "decisions": {"strategy": "wilson_loop", "type": params["type"], "normalization": False,
                      "factor_uncertainty": "fixed_central_value_not_propagated"},
        "diagnostics": diagnostics, "artifacts": ["output.nc", "diagnostics.json", "settings.json"],
    })
