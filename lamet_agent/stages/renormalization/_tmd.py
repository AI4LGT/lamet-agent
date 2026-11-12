"""Publish the deterministic Coulomb-gauge TMD ratio operations."""

from __future__ import annotations

import json

from lamet_agent.agent import ToolContext
from lamet_agent.stages.renormalization._plotting import render_result
from lamet_agent.stages.renormalization.physics import load_data, tmd_ratio


def run(context: ToolContext) -> None:
    operation = str(context.params["operation"])
    projection = str(context.params.get("projection", "preserve"))
    role = "denominator" if operation == "apply" else "local"
    target = load_data(context.inputs["target"])
    denominator = load_data(context.inputs[role])
    result = tmd_ratio(target, denominator, operation=operation, projection=projection)
    result.array.attrs["sample_error_mode"] = str(context.manifest["metadata"]["sample_error_mode"])
    result.to_netcdf(context.artifact_directory / "output.nc")
    coordinates = result.coords.get("z", [])
    diagnostics = {
        "strategy": "tmd_ratio", "scheme": "ratio", "type": "apply", "operation": operation,
        "projection": projection,
        "target_samples": result.n_sample, "sample_count": result.n_sample,
        "bT_fm": result.attrs.get("bT_fm"), "momentum_gev": result.attrs.get("momentum_gev"),
        "dims": result.dims, "z_count": len(coordinates),
        "z_range_fm": [float(min(coordinates)), float(max(coordinates))] if coordinates else None,
        "replica_frame_id": result.attrs.get("replica_frame_id"),
        "resample_lineage": json.loads(result.attrs["resample_lineage"])
        if result.attrs.get("resample_lineage") else {},
        "minimum_denominator_absolute": result.attrs["minimum_denominator_absolute"],
        "local_normalization_component": result.attrs.get("local_normalization_component"),
    }
    if result.attrs.get("jackknife_layout"):
        diagnostics.update({
            "jackknife_groups": json.loads(result.attrs["jackknife_layout"]),
            "statistical_replica_count": result.n_sample - 1,
            "central_evaluation_row": result.n_sample - 1,
        })
    diagnostic_path = context.artifact_directory / "diagnostics" / "renormalization.json"
    diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    artifacts = ["output.nc", "diagnostics/renormalization.json"]
    if result.dims == ["z"]:
        render_result(result, directory=context.artifact_directory / "plots", stem="result", formats=("pdf", "svg"))
        artifacts.extend(["plots/result.pdf", "plots/result.svg"])
    context.finish(result, {
        "stage_id": context.stage_id, "job_id": context.job_id,
        "result": "tmd_reference" if operation == "reference" else "renormalized_matrix_element",
        "decisions": {"strategy": "tmd_ratio", "operation": operation, "scheme": "ratio",
                      "projection": projection},
        "diagnostics": diagnostics, "artifacts": artifacts,
    })
