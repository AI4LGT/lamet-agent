"""Formula and diagnostics report for the complete soft-analysis DAG."""

from pathlib import Path

import gvar

from lamet_agent.stages._reporting import artifact_rows, figure_lines, format_value, write_report


def write_stage_report(*, records, artifact_directory: Path):
    lines = ["# Soft Analysis Stage Report", "", "The analysis uses the ratio scheme and aligned replicas. "
             "Each job declares fixed-order NLO or NLL and its infrared prescription. "
             "Errors are statistical; systematic sweeps are deferred.", "",
             r"$$\gamma=\frac{\log[H_\phi(P_1)\widetilde\phi(P_2)/"
             r"(H_\phi(P_2)\widetilde\phi(P_1))]}{\log(P_2/P_1)}.$" + "$", "",
             r"$$S_I=F/\int dx_1dx_2\,H_F(P_H)\Phi^*(x_1)\Phi(x_2),\qquad"
             r"\Phi=\widetilde\phi/H_\phi\,\exp[\gamma_{evol}\log(P_{FF}/P_{WF})].$$", "",
             "By default gamma_evol is the physical CS kernel and PH is the FF momentum. The explicit "
             "reference options use the uncorrected quasi momentum ratio for evolution and the source WF "
             "momentum in HF. These choices are recorded on the intrinsic-soft artifact.", "",
             "The endpoint model is c[x(1−x)]^n, fitted with the authored working covariance; aligned replicas "
             "propagate the output uncertainty. It replaces only the "
             "endpoint regions. Low fit Q is recorded and never used to discard replicas. The NLO kernel "
             "retains its sign. NLL infrared freezing or historical pole-crossing quadrature requires an "
             "explicit policy; historical quadrature diagnostics do not establish controlled perturbative precision. "
             "No external cached soft result or implicit bT extrapolation enters the calculation.", ""]
    for record in records:
        diagnostics = record.summary["diagnostics"]
        lines.extend([f"## `{record.job_id}`", "", "| setting | value |", "|---|---|"])
        for key, value in record.params.items():
            lines.append(f"| {key} | {format_value(value)} |")
        if diagnostics.get("coefficient_provenance"):
            lines.append(f"| coefficient provenance | {format_value(diagnostics['coefficient_provenance'])} |")
        if diagnostics.get("infrared_diagnostics"):
            lines.append(f"| infrared diagnostics | {format_value(diagnostics['infrared_diagnostics'])} |")
        if diagnostics.get("jackknife_groups"):
            lines.extend([
                f"| jackknife configuration groups | {format_value(diagnostics['jackknife_groups'])} |",
                f"| statistical replicas | {diagnostics['statistical_replica_count']} |",
                "| central evaluation | stored separately; excluded from covariance and quantiles |",
            ])
        lines.extend(["", "| bT [fm] | mean | statistical error |", "|---:|---:|---:|"])
        avg = record.output.average(record.output.attrs.get("sample_error_mode", "covariance"))
        for b, value in zip(record.output.coords["bT"], avg):
            lines.append(f"| {b:g} | {gvar.mean(value):.8g} | {gvar.sdev(value):.6g} |")
        if diagnostics.get("replica_distribution"):
            interpretation = (
                "Bootstrap percentile intervals below retain nonlinear skewness. "
                "Mean ± standard deviation need not represent a 68% interval."
                if record.output.resample == "bootstrap"
                else ("The quantiles below describe the raw leave-one-out replicas across independent group "
                      "blocks, not a confidence interval; they exclude the central evaluation. The statistical "
                      "error above sums each group's jackknife covariance using its own configuration count."
                      if record.output.attrs.get("jackknife_layout")
                      else "The quantiles below describe the raw leave-one-out replicas, not a confidence "
                           "interval. The statistical error above includes the jackknife scaling factor.")
            )
            lines.extend(["", interpretation, "",
                          "| bT [fm] | median | 16th percentile | 84th percentile | minimum | maximum |",
                          "|---:|---:|---:|---:|---:|---:|"])
            for row in diagnostics["replica_distribution"]:
                lines.append(f"| {row['bT_fm']:g} | {row['median']:.8g} | {row['q16']:.8g} | "
                             f"{row['q84']:.8g} | {row['minimum']:.8g} | {row['maximum']:.8g} |")
        lines.extend(["", "Aggregation: " + diagnostics["aggregation"], ""])
        if "pair_samples_artifact" in diagnostics:
            lines.extend(["Per-pair means and statistical errors are in diagnostics/soft.json. "
                          "The cs_pairs.nc artifact retains every bT/pair/x replica, so correlations "
                          "between the selected pairs and x points can be reconstructed.", ""])
        if "quasi_ratio_samples_artifact" in diagnostics:
            lines.extend(["The cs_quasi_ratio.nc artifact publishes the same replicas before the deterministic "
                          "hard correction. The main output remains the physical CS kernel for matching.", ""])
        for row in diagnostics["bT_records"]:
            if "correlated_average" in row:
                average = row["correlated_average"]
                consistency = average.get("momentum_consistency", {})
                lines.append(f"- bT={row['bT_fm']:g} fm: arithmetic averaging weights="
                             f"{format_value(average['weights'])}.")
                lines.append(f"- Separate common-constant GLS diagnostic: "
                             f"fitted mean={format_value(consistency.get('fitted_mean'))}, "
                             f"fit weights={format_value(consistency.get('fit_weights'))}, "
                             f"covariance rank={consistency.get('covariance_rank')}; "
                             f"chi2/dof={format_value(consistency.get('chi2'))}/"
                             f"{consistency.get('dof')}, Q={format_value(consistency.get('Q'))} "
                             "(approximate, regularized covariance; this fit does not set the output).")
                for warning in consistency.get("warnings", []):
                    lines.append(f"- bT={row['bT_fm']:g} fm diagnostic: {warning}")
            for momentum in row.get("momenta", []):
                fit = momentum["endpoint_fit"]
                parameters = fit["replica_parameters"]
                lines.append(
                    f"- bT={row['bT_fm']:g} fm, P={momentum['momentum_gev']:g} GeV: "
                    f"endpoint Q={format_value(fit['Q'])}, c={fit['c']:.6g}, n={fit['n']:.6g}; "
                    f"quadrature max relative change={momentum['quadrature_max_relative_change']:.3g}; "
                    f"negative HF node fraction={momentum['negative_HF_node_fraction']:.3g}."
                )
                lines.append(
                    f"- Endpoint replica c={parameters['mean'][0]:.6g} ± {parameters['sdev'][0]:.3g}, "
                    f"n={parameters['mean'][1]:.6g} ± {parameters['sdev'][1]:.3g}; "
                    f"covariance(c,n)={format_value(parameters['covariance'])}."
                )
                if fit.get("effective_priors"):
                    lines.append(
                        "- Endpoint Gaussian priors in physical units: "
                        f"{format_value(fit['effective_priors'])}; original priors reused for every replica. "
                        f"Data chi2={fit['data_chi2']:.6g}, prior chi2={fit['prior_chi2']:.6g}; "
                        f"{fit['Q_interpretation']}"
                    )
        lines.extend(["", *figure_lines(record, artifact_directory), "", "| job | artifact |", "|---|---|",
                      *artifact_rows(record, artifact_directory), ""])
    return write_report(artifact_directory, lines)
