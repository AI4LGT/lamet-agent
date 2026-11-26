"""Publish single-spacing mass fits and hybrid-ratio applications."""

import json

from lamet_agent.agent import ToolContext
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.stages.renormalization._single_spacing_physics import (
    apply_single_spacing,
    fit_single_spacing,
)


def run(context: ToolContext) -> dict[str, object]:
    params = context.params
    inputs = context.state["aligned_inputs"]
    if params["type"] == "fit":
        output, diagnostics = fit_single_spacing(
            inputs["reference"],
            long_distance_window_fm=params["long_distance_window_fm"],
            short_distance_points_fm=params["short_distance_points_fm"],
            kernel=load_renormalization_kernel(params["kernel_id"]),
            kernel_id=params["kernel_id"],
            mu=params["mu"],
            kernel_parameters=params.get("kernel_parameters", {}),
            svdcut=params.get("svdcut", 1e-10),
            scan_centers_fm=params.get("scan_centers_fm", []),
            scan_half_width_fm=params.get("scan_half_width_fm", 0.02),
        )
        filename = "self_renormalization.json"
        result = "single_spacing_masses"
    else:
        output = apply_single_spacing(
            inputs["target"],
            inputs["denominator"],
            inputs["masses"],
            zs_fm=params["zs_fm"],
            normalization=params["normalization"],
        )
        diagnostics = {
            "formula": "single_spacing_hybrid_ratio",
            "source_arxiv": "2312.10829",
            "target_samples": output.n_sample,
            "zs_fm": params["zs_fm"],
            "z_range_fm": [min(output.coords["z"]), max(output.coords["z"])],
            "z_count": len(output.coords["z"]),
            "scale_gev": output.attrs["scale_gev"],
            "kernel_id": output.attrs["renormalization_kernel_id"],
        }
        filename = "renormalization.json"
        result = "renormalized_matrix_element"
    directory = context.artifact_directory
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "diagnostics").mkdir(exist_ok=True)
    output.to_netcdf(directory / "output.nc")
    (directory / "diagnostics" / filename).write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    summary = {
        "stage_id": context.stage_id,
        "job_id": context.job_id,
        "result": result,
        "decisions": {"strategy": params["strategy"], "type": params["type"]},
        "diagnostics": diagnostics,
        "artifacts": ["output.nc", f"diagnostics/{filename}"],
    }
    context.finish(output, summary)
    return {"summary": result, "metrics": diagnostics, "state_keys": [], "artifacts": summary["artifacts"]}
