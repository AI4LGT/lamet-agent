"""Deterministic soft extraction and publication."""

import json

import numpy as np

from lamet_agent.data import sample_mean
from lamet_agent.plotting import configure_plot, errorline, save_figure, start_plot
from lamet_agent.stages.soft_analysis.physics import extract_cs, extract_intrinsic, load, load_many, replica_sdev


def replica_distribution(result):
    """Report raw replica quantiles and the resampling-aware standard error."""
    values = np.asarray(result.values)
    replicas = values[:-1] if result.attrs.get("jackknife_layout") else values
    quantiles = np.quantile(replicas, [0.16, 0.5, 0.84], axis=0)
    errors = replica_sdev(values, result.resample, result.attrs)
    means = sample_mean(values, result.resample, result.attrs)
    return [
        {
            "bT_fm": float(b), "mean": float(means[i]),
            "sdev": float(errors[i]),
            "quantile_interpretation": (
                "bootstrap percentile interval" if result.resample == "bootstrap"
                else ("raw leave-one-out replica quantiles across independent group blocks; "
                      "central evaluation excluded; not a confidence interval"
                      if result.attrs.get("jackknife_layout")
                      else "raw leave-one-out replica quantiles; not a confidence interval")
            ),
            "q16": float(quantiles[0, i]), "median": float(quantiles[1, i]),
            "q84": float(quantiles[2, i]), "minimum": float(replicas[:, i].min()),
            "maximum": float(replicas[:, i].max()),
        }
        for i, b in enumerate(result.coords["bT"])
    ]


def run(context, _session=None):
    quasi = load_many(context.inputs["quasi"])
    if context.params["type"] == "cs_kernel":
        result, diagnostics = extract_cs(quasi, context.params)
    elif context.params["type"] == "intrinsic_soft":
        result, diagnostics = extract_intrinsic(
            quasi, load_many(context.inputs["form_factor"]), load(context.inputs["cs_kernel"]), context.params
        )
    else:
        raise ValueError("soft_analysis.type must be 'cs_kernel' or 'intrinsic_soft'")
    root = context.artifact_directory
    root.mkdir(parents=True, exist_ok=True)
    result.array.attrs["sample_error_mode"] = context.manifest["metadata"]["sample_error_mode"]
    diagnostics["replica_distribution"] = replica_distribution(result)
    if result.attrs.get("jackknife_layout"):
        diagnostics.update({
            "jackknife_groups": json.loads(result.attrs["jackknife_layout"]),
            "statistical_replica_count": result.n_sample - 1,
            "central_evaluation_row": result.n_sample - 1,
        })
    result.to_netcdf(root / "output.nc")
    (root / "diagnostics").mkdir(exist_ok=True)
    artifacts = ["output.nc", "diagnostics/soft.json", "plots/result.pdf", "plots/result.svg"]
    pair_samples = diagnostics.pop("_pair_samples", None)
    if pair_samples is not None:
        pair_samples.array.attrs["sample_error_mode"] = context.manifest["metadata"]["sample_error_mode"]
        pair_samples.to_netcdf(root / "diagnostics" / "cs_pairs.nc")
        diagnostics["pair_samples_artifact"] = "diagnostics/cs_pairs.nc"
        artifacts.append("diagnostics/cs_pairs.nc")
    reduced_samples = diagnostics.pop("_reduced_samples", None)
    if reduced_samples is not None:
        reduced_samples.array.attrs["sample_error_mode"] = context.manifest["metadata"]["sample_error_mode"]
        reduced_samples.to_netcdf(root / "diagnostics" / "cs_quasi_ratio.nc")
        diagnostics["quasi_ratio_samples_artifact"] = "diagnostics/cs_quasi_ratio.nc"
        artifacts.append("diagnostics/cs_quasi_ratio.nc")
    (root / "diagnostics" / "soft.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    start_plot()
    errorline(result.coords["bT"], result.average(context.manifest["metadata"]["sample_error_mode"]),
              label=context.params["type"])
    configure_plot(xlabel=r"$b_T$ [fm]", ylabel=context.params["type"], legend=True)
    save_figure(root / "plots" / "result.pdf", root / "plots" / "result.svg")
    context.finish(result, {
        "stage_id": context.stage_id, "job_id": context.job_id, "result": context.params["type"],
        "decisions": {"type": context.params["type"], "order": result.attrs["order"],
                      "resummation": result.attrs["resummation"], "mu": result.attrs["mu"],
                      "alpha_s": result.attrs["alpha_s"], "scheme": "ratio",
                      "coefficient_provenance": json.loads(result.attrs["coefficient_provenance"]),
                      **{key: result.attrs[key] for key in ("evolution_cs", "sudakov_momentum")
                         if key in result.attrs}},
        "diagnostics": diagnostics,
        "artifacts": artifacts,
    })
