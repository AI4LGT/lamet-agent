"""TMDWF operations using the existing job loop and artifact format."""

import json
from pathlib import Path

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.ui import log, warning
from lamet_agent.stages.tmdwf_analysis import physics


def load(value):
    return value if isinstance(value, EnsembleData) else EnsembleData.from_netcdf(Path(value))


def quality(array):
    q = array[..., 2]
    return {"fits": int(q.size), "min_Q": float(np.min(q)), "median_Q": float(np.median(q)),
            "fraction_Q_below_0_05": float(np.mean(q < 0.05))}


def run(context, _session):
    cfg = context.params["settings"]
    operation = context.params["operation"]
    samples = int(context.manifest["metadata"]["samples"])
    folder = context.artifact_directory
    files = ["output.nc", "diagnostics.json", "plot.png", "settings.json"]
    if operation == "wilson_loop_fit":
        log(f"TMDWF Wilson loop: fit the raw-configuration mean; share one central factor across {samples} samples")
    else:
        log(f"TMDWF {operation}: {samples} existing bootstrap samples")
    if operation == "bare_fit":
        data, fits = physics.bare_fit(context.inputs["raw"], cfg, samples, context._parallel)
        diagnostics = quality(fits)
        np.save(folder / "fit_quality.npy", fits)
        files.append("fit_quality.npy")
    elif operation == "wilson_loop_fit":
        data, diagnostics = physics.wilson_factor(context.inputs["raw"], cfg, samples)
    elif operation == "renormalize":
        target, denominator = load(context.inputs["target"]), load(context.inputs["denominator"])
        data = physics.renormalize(target, denominator)
        diagnostics = {"factor_uncertainty": "fixed_central_value_not_propagated",
                       "origin_normalization": False}
    elif operation == "native_bridge":
        from lamet_agent.stages.tmdwf_analysis.native import prepare_projection

        data = prepare_projection([load(value) for value in context.inputs["targets"]],
                                  load(context.inputs["denominator"]), cfg["projection"])
        diagnostics = {"projection": cfg["projection"], "negative_z": cfg["negative_z"],
                       "factor_uncertainty": data.attrs["factor_uncertainty"],
                       "source_combination": data.attrs["source_combination"]}
    else:
        data, extended, fits, parameters = physics.transform(load(context.inputs["input"]), cfg,
                                                           context.workers, context._parallel)
        diagnostics = quality(fits)
        diagnostics["matching_applied"] = False
        extended.to_netcdf(folder / "extended.nc")
        np.save(folder / "fit_quality.npy", fits)
        (folder / "tail_parameters.json").write_text(json.dumps(parameters, indent=2), encoding="utf-8")
        files += ["extended.nc", "fit_quality.npy", "tail_parameters.json"]
    if data.n_sample != samples or not np.isfinite(data.values).all():
        raise ValueError("Output must preserve the declared sample count and contain only finite values")
    if diagnostics.get("fraction_Q_below_0_05", 0) > 0:
        warning(f"{operation}: {diagnostics['fraction_Q_below_0_05']:.1%} of fits have Q < 0.05; "
                "results are retained for notebook reproduction, not marked as quality-passing.")
    data.to_netcdf(folder / "output.nc")
    (folder / "settings.json").write_text(json.dumps(context.params, indent=2), encoding="utf-8")
    (folder / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    plot(data, folder / "plot.png")
    context.finish(data, {
        "stage_id": context.stage_id, "job_id": context.job_id, "result": operation,
        "decisions": {"operation": operation, "settings": cfg,
                      "resampling": "jackknife_errors_for_single_mean_fit" if operation == "wilson_loop_fit"
                      else "preserve_existing_bootstrap",
                      "validation_scope": "integration_test_not_physics_certification"},
        "diagnostics": diagnostics, "artifacts": files,
    })


def plot(data, path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    figure = Figure(figsize=(8, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.add_subplot()
    dim = data.dims[0]
    x = np.asarray(data.coords[dim])
    for label, values in (("real", data.values.real), ("imag", data.values.imag)):
        if label == "imag" and not np.iscomplexobj(data.values):
            continue
        center, error = values.mean(axis=0), values.std(axis=0)
        axis.plot(x, center, label=label)
        axis.fill_between(x, center - error, center + error, alpha=0.2)
    axis.set_xlabel("z [fm]" if dim == "z" else "x")
    axis.set_title(data.name)
    axis.legend()
    figure.savefig(path, dpi=150)
