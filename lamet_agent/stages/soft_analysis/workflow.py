"""Deterministic Soft Analysis workflow using the normal stage/job harness."""

from __future__ import annotations

import json

import gvar
import numpy as np

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.plotting import configure_plot, errorband, hline, save_figure, start_plot
from lamet_agent.stages.soft_analysis.physics import compute_pair, load_quasi


def run(context: ToolContext, _session: LlmSession) -> None:
    """Inspect paired inputs, compute CSK sample-wise, and publish standard artifacts."""
    first = load_quasi(context.inputs["quasi1"])
    second = load_quasi(context.inputs["quasi2"])
    if context.manifest["metadata"]["resample_mode"] != "bootstrap":
        raise ValueError("Soft Analysis requires metadata.resample_mode='bootstrap'")
    if any(data.n_sample != context.manifest["metadata"]["samples"] for data in (first, second)):
        raise ValueError("input bootstrap counts disagree with metadata.samples")
    mode = str(context.manifest["metadata"]["sample_error_mode"])
    computed = compute_pair(first, second, context.params, sample_error_mode=mode)
    output, raw = computed["data"], computed["raw"]
    output.to_netcdf(context.artifact_directory / "output.nc")
    raw.to_netcdf(context.artifact_directory / "raw_ratio.nc")
    x = np.asarray(output.coords["x"], dtype=float)
    diagnostics = computed["diagnostics"]
    (context.artifact_directory / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    columns = [x]
    labels = ["x"]
    for component in ("real", "imag"):
        for label, data in (("raw", raw), ("csk", output)):
            average = getattr(data, component).average(mode)
            columns.extend((gvar.mean(average), gvar.sdev(average)))
            labels.extend((f"{label}_{component}", f"{label}_{component}_error"))
        columns.append(getattr(computed["correction"], component))
        labels.append(f"delta_gamma_{component}")
    np.savetxt(
        context.artifact_directory / "csk.csv",
        np.column_stack(columns),
        delimiter=",",
        header=",".join(labels),
        comments="",
    )
    artifacts = ["output.nc", "raw_ratio.nc", "diagnostics.json", "csk.csv"]
    plot_components = (
        (("real", "Re"),) if context.manifest["metadata"]["run_id"] == "CS_kernel" else (("real", "Re"), ("imag", "Im"))
    )
    for component, label in plot_components:
        start_plot()
        hline(0.0, color="black")
        for source in (first, second):
            selected = getattr(source.at("x", list(x)), component)
            momentum = float(source.attrs["momentum_gev"])
            errorband(x, selected.average(mode), label=rf"$P_z={momentum:.3f}$ GeV")
        configure_plot(xlabel=r"$x$", ylabel=rf"${label}\,\widetilde\Psi$", legend=True)
        paths = [f"plots/quasi_{component}.{extension}" for extension in ("pdf", "svg")]
        save_figure(*(context.artifact_directory / path for path in paths))
        artifacts.extend(paths)
        start_plot()
        hline(0.0, color="black")
        errorband(x, getattr(raw, component).average(mode), label="quasi ratio")
        errorband(x, getattr(output, component).average(mode), label="ratio + uNLO")
        configure_plot(xlabel=r"$x$", ylabel=rf"${label}\,\gamma_q$", legend=True)
        paths = [f"plots/csk_{component}.{extension}" for extension in ("pdf", "svg")]
        save_figure(*(context.artifact_directory / path for path in paths))
        artifacts.extend(paths)
    context.finish(
        output,
        {
            "stage_id": context.stage_id,
            "job_id": context.job_id,
            "result": "finite_momentum_collins_soper_kernel",
            "decisions": {"kernel_id": "tmdwf_gi_unlo", **context.params},
            "diagnostics": diagnostics,
            "artifacts": artifacts,
        },
    )
