"""Multiplicative CG TMD matching with shared soft-function replicas."""

from __future__ import annotations

import json

import numpy as np

from lamet_agent.data import EnsembleData, align_sample_lineage, merge_sample_lineage
from lamet_agent.kernels.tmd.cg_nlo import rapidity_factor
from lamet_agent.kernels.tmd.coefficients import hard_factor
from lamet_agent.plotting import configure_plot, errorband, save_figure, start_plot
from lamet_agent.stages._reporting import artifact_rows, describe_grid, figure_lines, format_value
from lamet_agent.stages.soft_analysis.physics import (
    at_b, inclusive_window, kernel_controls, load, numeric_attr, perturbative_attrs, positive,
    validate_lattice_geometry, validate_perturbative_input, validate_quasi, validate_settings,
)


def match(quasi: EnsembleData, cs: EnsembleData, soft: EnsembleData, params: dict):
    validate_settings(params)
    observable = params["observable"]
    quasi, cs, soft = align_sample_lineage(quasi, cs, soft)
    lineage = merge_sample_lineage(quasi, cs, soft)
    validate_lattice_geometry(quasi, cs, soft)
    x, values = validate_quasi(quasi, observable)
    validate_perturbative_input(cs, params, "cs_kernel")
    validate_perturbative_input(soft, params, "intrinsic_soft")
    output_grid = params["lc_x_ls"]
    if isinstance(output_grid, dict):
        selected = np.flatnonzero(inclusive_window(x, output_grid["start"], output_grid["stop"]))
    else:
        selected = []
        for value in output_grid:
            indices = np.flatnonzero(np.isclose(x, value, rtol=0, atol=1e-10))
            if len(indices) != 1:
                raise ValueError("TMD lc_x_ls must select existing quasi-grid points; interpolation is not implicit")
            selected.append(indices[0])
        selected = np.asarray(selected, dtype=int)
    if not len(selected):
        raise ValueError("TMD lc_x_ls selects no quasi-grid points")
    xx, yy = x[selected], values[:, selected]
    if np.any(np.diff(xx) <= 0):
        raise ValueError("TMD output x grid must be strictly increasing without duplicate points")
    b, p = numeric_attr(quasi, "bT_fm", nonnegative=True), numeric_attr(quasi, "momentum_gev")
    gamma, intrinsic = at_b(cs, b), at_b(soft, b)
    positive(f"intrinsic soft function at bT={b:g} fm", intrinsic)
    mu, alpha = params.get("mu", 2.0), params.get("alpha_s", 0.293)
    h = hard_factor(xx, observable=observable, momentum_gev=p, **kernel_controls(params))
    positive(f"{observable} {params.get('order', 'nlo').upper()} hard factor "
             f"(P={p:g} GeV, mu={mu:g}, x={xx[0]:g}..{xx[-1]:g})", h)
    if observable == "tmdwf":
        pref = float(params.get("p_ref_gev", 4.30))
        factor = rapidity_factor(gamma, momentum_gev=p, p_ref_gev=pref)[:, None]
        rapidity = {"p_ref_gev": pref}
    elif observable == "tmdpdf":
        zeta = float(params.get("zeta_gev2", 4.0))
        if not np.isfinite(zeta) or zeta <= 0:
            raise ValueError("TMDPDF zeta_gev2 must be finite and positive")
        factor = np.exp(0.5 * gamma[:, None] * np.log(zeta / (2 * xx * p)**2))
        rapidity = {"zeta_gev2": zeta}
    else:
        raise ValueError("TMD matching observable must be 'tmdwf' or 'tmdpdf'")
    flavor = float(params.get("flavor_factor", 2.0 if observable == "tmdpdf" else 1.0))
    if not np.isfinite(flavor) or flavor <= 0:
        raise ValueError("flavor_factor must be finite and positive")
    result = flavor * yy * np.sqrt(intrinsic[:, None]) / h * factor
    if not np.all(np.isfinite(result)):
        raise ValueError("TMD matching produced nonfinite replicas; inspect soft-input and rapidity-evolution errors")
    attrs = {**quasi.attrs, **lineage, **rapidity,
             "observable": observable, **perturbative_attrs(params),
             "soft_evolution_cs": soft.attrs.get("evolution_cs", "physical"),
             "soft_sudakov_momentum": soft.attrs.get("sudakov_momentum", "form_factor"),
             "cs_convention": "physical",
             "soft_endpoint_covariance": soft.attrs.get("endpoint_covariance", "full"),
             "mu": float(mu), "alpha_s": float(alpha), "flavor_factor": flavor,
             "distribution": "light_cone", "output_component": "re",
             "uncertainty": "statistical; systematic variations deferred"}
    if "infrared_diagnostics" in soft.attrs:
        attrs["soft_infrared_diagnostics"] = soft.attrs["infrared_diagnostics"]
    data = EnsembleData(quasi.ensemble, quasi.resample, list(result), ["x"], {"x": xx.tolist()}, attrs,
                        observable)
    data.array = data.array.assign_coords(resample=quasi.array.coords["resample"])
    diagnostics = {"observable": observable, "bT_fm": b, "momentum_gev": p, "mu": float(mu),
                   "alpha_s": float(alpha), **rapidity, "flavor_factor": flavor,
                   "order": data.attrs["order"], "resummation": data.attrs["resummation"],
                   "coefficient_provenance": json.loads(data.attrs["coefficient_provenance"]),
                   "soft_evolution_cs": data.attrs["soft_evolution_cs"],
                   "soft_sudakov_momentum": data.attrs["soft_sudakov_momentum"],
                   "cs_convention": "physical",
                   "soft_endpoint_covariance": data.attrs["soft_endpoint_covariance"],
                   "hard_factor_min": float(h.min()), "hard_factor_max": float(h.max()),
                   "x_count": len(xx), "x_min": float(xx[0]), "x_max": float(xx[-1]),
                   "systematics_included": False}
    if params.get("order", "nlo") == "nll":
        from lamet_agent.kernels.tmd.cg_nll import scale_diagnostics
        scales = 2 * p * (np.concatenate((xx, 1 - xx)) if observable == "tmdwf" else xx)
        diagnostics["hard_scales"] = scale_diagnostics(
            scales, **{key: value for key, value in kernel_controls(params).items() if key != "order"})
        data.array.attrs["hard_scale_diagnostics"] = json.dumps(diagnostics["hard_scales"], sort_keys=True)
    if "soft_infrared_diagnostics" in data.attrs:
        diagnostics["soft_infrared_diagnostics"] = json.loads(data.attrs["soft_infrared_diagnostics"])
    if data.attrs.get("jackknife_layout"):
        diagnostics.update({
            "jackknife_groups": json.loads(data.attrs["jackknife_layout"]),
            "statistical_replica_count": data.n_sample - 1,
            "central_evaluation_row": data.n_sample - 1,
        })
    return data, diagnostics


def run(context, _session=None):
    quasi = load(context.inputs["quasi"])
    result, diagnostics = match(quasi, load(context.inputs["cs_kernel"]),
                                load(context.inputs["intrinsic_soft"]), context.params)
    root = context.artifact_directory
    root.mkdir(parents=True, exist_ok=True)
    mode = context.manifest["metadata"]["sample_error_mode"]
    result.array.attrs["sample_error_mode"] = mode
    result.to_netcdf(root / "output.nc")
    (root / "diagnostics").mkdir(exist_ok=True)
    (root / "diagnostics" / "matching.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    start_plot()
    selected_quasi = quasi.at("x", result.coords["x"])
    errorband(result.coords["x"], selected_quasi.real.average(mode), label="quasi")
    errorband(result.coords["x"], result.average(mode), label=context.params["observable"])
    configure_plot(xlabel=r"$x$", ylabel=context.params["observable"], legend=True)
    save_figure(root / "plots" / "result.pdf", root / "plots" / "result.svg")
    context.finish(result, {
        "stage_id": context.stage_id, "job_id": context.job_id, "result": context.params["observable"],
        "decisions": {"kernel_id": result.attrs["kernel_id"], "order": result.attrs["order"],
                      "resummation": result.attrs["resummation"], "scheme": "ratio",
                      "mu": diagnostics["mu"], "alpha_s": diagnostics["alpha_s"],
                      "coefficient_provenance": diagnostics["coefficient_provenance"],
                      "soft_evolution_cs": diagnostics["soft_evolution_cs"],
                      "soft_sudakov_momentum": diagnostics["soft_sudakov_momentum"]},
        "diagnostics": diagnostics,
        "artifacts": ["output.nc", "diagnostics/matching.json", "plots/result.pdf", "plots/result.svg"],
    })


def report_lines(records, artifact_directory):
    lines = ["# TMD Matching Stage Report", "",
             "The quasi distribution, CS kernel and intrinsic soft function retain their shared replica lineage. "
             "Each job declares fixed-order NLO or NLL coefficients and its infrared prescription; "
             "uncertainty bands are statistical. Matching always uses the physical hard-corrected CS kernel.", "",
             r"$$\phi=\sqrt{S_I}\,\widetilde\phi/H_\phi\,e^{\gamma\log(P_{ref}/P)},\qquad"
             r"f=N_f\sqrt{S_I}\,\widetilde f/H_f\,e^{\gamma\log[\zeta/(2xP)^2]/2}.$" + "$", "",
             "Here Nf is the explicit flavor_factor, not the QCD number of flavors. "
             "The pion PDF example uses 2 to convert (u−d)/2 to u−d.", ""]
    for record in records:
        lines.extend([f"## `{record.job_id}`", "", "| setting | value |", "|---|---|"])
        for key, value in record.summary["diagnostics"].items():
            lines.append(f"| {key} | {format_value(value)} |")
        lines.extend(["", describe_grid(record.output.coords["x"]), "",
                      *figure_lines(record, artifact_directory), "", "| job | artifact |", "|---|---|",
                      *artifact_rows(record, artifact_directory), ""])
    return lines
