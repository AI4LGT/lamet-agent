"""Deterministic Soft Analysis workflow using the normal stage/job harness."""

from __future__ import annotations

import csv
import json
from itertools import combinations

import gvar
import numpy as np

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.data import EnsembleData
from lamet_agent.plotting import configure_plot, errorband, hline, save_figure, start_plot
from lamet_agent.stages._priors import validate_prior_overrides
from lamet_agent.stages.soft_analysis.physics import (
    compute_pair,
    fit_power,
    load_quasi,
    transverse_distance,
    valid_slope_prior,
)


def run(context: ToolContext, _session: LlmSession) -> None:
    """Inspect paired inputs, compute CSK sample-wise, and publish standard artifacts."""
    if context.params.get("type") == "cs_kernel":
        run_grouped(context)
        return
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


def group_quasi(sources, momenta, metadata):
    if not isinstance(sources, list) or not sources or metadata["resample_mode"] != "bootstrap":
        raise ValueError("Grouped CSK requires a list of paired bootstrap quasi distributions")
    datasets = [load_quasi(source) for source in sources]
    reference = datasets[0]
    groups = {}
    for data in datasets:
        if (data.ensemble is None or data.ensemble != reference.ensemble or data.n_sample != metadata["samples"]
                or data.resample != "bootstrap"
                or not np.array_equal(data.array.coords["resample"], reference.array.coords["resample"])):
            raise ValueError("All b and momentum inputs must share the ensemble and bootstrap replica coordinates")
        for key in ("resample_id", "replica_frame_id", "resample_lineage", "parton", "gfix", "hadron",
                    "phase_convention", "fourier_convention", "renormalization_scheme"):
            if data.attrs.get(key) != reference.attrs.get(key):
                raise ValueError(f"Grouped quasi inputs disagree on {key}")
        index_float = data.attrs["momentum_gev"] / data.ensemble.k_s
        index = round(index_float)
        if not np.isclose(index_float, index, atol=1e-8, rtol=0) or data.attrs.get("momentum_index", index) != index:
            raise ValueError("Physical momentum and lattice momentum index are inconsistent")
        b = transverse_distance(data)
        by_momentum = groups.setdefault(b, {})
        if index in by_momentum:
            raise ValueError(f"Repeated quasi input at b={b} fm and momentum index {index}")
        by_momentum[index] = data
    for b, items in groups.items():
        if items.keys() != momenta:
            raise ValueError(f"b={b} fm requires momenta {sorted(momenta)}, received {sorted(items)}")
    return groups


def _save_pair(result, folder, b, p, q, artifacts):
    path = f"pairs/b{b!r}fm_p{p}_p{q}.nc"
    result["data"].to_netcdf(folder / path)
    artifacts.append(path)
    return {"momentum_indices": [p, q], "curve": path, **result["diagnostics"]}


def run_grouped(context):
    params, folder = context.params, context.artifact_directory
    metadata = context.manifest["metadata"]
    mode = metadata["sample_error_mode"]
    pairs = params["momentum_pairs"]
    estimator = params.get("estimator", "equal_weight")
    if estimator not in {"equal_weight", "pca_power"}:
        raise ValueError(f"Unknown grouped CSK estimator {estimator!r}")
    power = estimator == "pca_power"
    fraction = params.get("pca_variance_fraction", 0.99)
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError("pca_variance_fraction must lie in (0, 1]")
    prior = params.get("slope_prior")
    if "slope_prior" in params and not valid_slope_prior(prior):
        raise ValueError("slope_prior requires finite mean_gev2 and positive finite sigma_gev2")
    if "priors" in params:
        if "slope_prior" in params:
            raise ValueError("Specify the slope prior only once, using priors or slope_prior")
        named = validate_prior_overrides(params["priors"])
        if named.keys() - {"A"}:
            raise ValueError("The CSK power fit accepts a named prior only for A")
        if "A" in named:
            prior = {"mean_gev2": named["A"]["mean"], "sigma_gev2": named["A"]["sdev"]}
    if not power:
        params = {k: v for k, v in params.items() if k not in {"priors", "slope_prior", "pca_variance_fraction"}}
    elif len({tuple(sorted(pair)) for pair in pairs}) < 2:
        raise ValueError("pca_power requires at least two distinct momentum pairs")
    prior_attrs = {}
    if power and prior is not None:
        prior_attrs = {
            "slope_prior": json.dumps(prior, sort_keys=True),
            "prior_structure": "independent_per_x_and_component",
            "uncertainty": "fixed_prior_paired_bootstrap",
        }
    groups = group_quasi(context.inputs["quasi"], {p for pair in pairs for p in pair}, metadata)
    distances = sorted(groups)
    artifacts = ["output.nc", "csk.csv", "diagnostics.json"]
    (folder / "pairs").mkdir(exist_ok=True)
    samples, diagnostics = [], []
    for b in distances:
        by_momentum = groups[b]
        curves, pair_info = [], []
        for p, q in pairs:
            result = compute_pair(by_momentum[p], by_momentum[q], params, sample_error_mode=mode)
            curve = result["data"]
            if curves and curve.coords != curves[0].coords:
                raise ValueError(f"Selected momentum pairs have incompatible x grids at b={b} fm")
            curves.append(curve)
            pair_info.append(_save_pair(result, folder, b, p, q, artifacts))
        entry = {"bT_fm": b, "pairs": pair_info}
        if power:
            intercepts, slopes, fit_info = fit_power(curves, fraction, slope_prior=prior)
            samples.append(intercepts)
            slope_attrs = {
                **_common_attrs(curves[0]),
                "observable": "power_coefficient",
                "symbol": "A",
                "estimator": estimator,
                "units": "GeV^2",
                "bT_fm": b,
                "predictor": "1/P1_z^2 - 1/P2_z^2 [GeV^-2]",
                "momentum_pairs": json.dumps(pairs),
                "x_window": json.dumps(params["x_window"]),
                "pca_variance_fraction": fraction,
                **prior_attrs,
            }
            slope = EnsembleData(
                curve.ensemble,
                "bootstrap",
                list(slopes),
                ["x"],
                curve.coords,
                attrs=slope_attrs,
                name="power_coefficient",
            )
            slope.array = slope.array.assign_coords(resample=curve.array.coords["resample"])
            path = f"slopes/b{b!r}fm.nc"
            slope.to_netcdf(folder / path)
            artifacts.append(path)
            entry.update(power_fit=fit_info, slope_file=path)
            # Compute remaining ratios directly from the quasi distributions for the x plot.
            # Only the configured pairs above contribute to the fitted intercept and slopes.
            selected = {tuple(sorted(pair)) for pair in pairs}
            entry["plot_only_pairs"] = []
            for p, q in combinations(sorted(by_momentum), 2):
                if (p, q) in selected:
                    continue
                try:
                    result = compute_pair(by_momentum[p], by_momentum[q], params, sample_error_mode=mode)
                except (ValueError, ArithmeticError) as exc:
                    entry.setdefault("skipped_plot_pairs", []).append({"momentum_indices": [p, q], "reason": str(exc)})
                    continue
                entry["plot_only_pairs"].append(_save_pair(result, folder, b, p, q, artifacts))
        else:
            samples.append(np.mean([curve.values for curve in curves], axis=0).mean(axis=1))
        diagnostics.append(entry)

    # Only common physical provenance belongs on a b-dependent estimate.
    attrs = _common_attrs(curves[0])
    attrs.update(
        {
            "observable": "cs_kernel",
            "target_observable": "csk",
            "bT_unit": "fm",
            "estimator": estimator if power else "equal_weight_x_and_momentum_pairs",
            "momentum_pairs": json.dumps(pairs),
            "x_window": json.dumps(params["x_window"]),
        }
    )
    if power:
        attrs.update(
            units="dimensionless",
            quantity="infinite_momentum_intercept",
            pca_variance_fraction=fraction,
            model="K(bT) + A(x,bT)*(1/P1_z^2 - 1/P2_z^2)",
            **prior_attrs,
        )
    output = EnsembleData(
        curve.ensemble,
        "bootstrap",
        list(np.stack(samples, axis=1)),
        ["bT"],
        {"bT": distances},
        attrs=attrs,
        name="cs_kernel",
    )
    output.array = output.array.assign_coords(resample=curve.array.coords["resample"])
    output.to_netcdf(folder / "output.nc")
    real, imag = output.real.average(mode), output.imag.average(mode)
    rows = [
        dict(
            bT_fm=b,
            real=float(gvar.mean(r)),
            real_error=float(gvar.sdev(r)),
            imag=float(gvar.mean(i)),
            imag_error=float(gvar.sdev(i)),
        )
        for b, r, i in zip(distances, real, imag)
    ]
    with (folder / "csk.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "samples": output.n_sample,
        "bT_records": diagnostics,
        "estimates": rows,
        "systematics_included": False,
        "estimator": estimator,
    }
    if power:
        report.update(pca_variance_fraction=fraction, intercept_unit="dimensionless", slope_unit="GeV^2")
        if prior is not None:
            report.update(prior_attrs, slope_prior=prior)
    (folder / "diagnostics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    context.finish(
        output,
        {
            "stage_id": context.stage_id,
            "job_id": context.job_id,
            "result": "infinite_momentum_cs_intercept" if power else "finite_momentum_collins_soper_kernel",
            "decisions": dict(params),
            "diagnostics": report,
            "artifacts": artifacts,
        },
    )


def _common_attrs(curve):
    attrs = curve.attrs
    keys = (
        "resample_id",
        "resample_lineage",
        "replica_frame_id",
        "ensemble_id",
        "hadron",
        "parton",
        "gfix",
        "renormalization_scheme",
        "factor_uncertainty",
        "kernel_id",
        "mu",
        "alpha_s",
        "wilson_line_sign",
        "hard_convention",
        "hard_logarithm",
        "order",
        "sample_error_mode",
        "matching_uncertainty",
    )
    return {key: attrs[key] for key in keys if key in attrs}
