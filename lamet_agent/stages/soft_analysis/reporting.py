"""Soft Analysis report and overlays of all authored momentum-pair jobs."""

import re
from pathlib import Path

from lamet_agent.data import EnsembleData
from lamet_agent.stages._reporting import (
    StageReportRecord,
    artifact_rows,
    figure_lines,
    format_value,
    stage_overlay_lines,
    write_report,
)


def _pair_plots(records, artifact_directory):
    """Draw the C32P23 momentum pairs on one real-CSK figure per transverse separation."""
    import matplotlib.pyplot as plt
    import numpy as np

    groups = {}
    for record in records:
        match = re.fullmatch(r"csk_p(\d+)_p(\d+)_b(\d+)", record.job_id)
        if match is None:
            return stage_overlay_lines(
                records,
                artifact_directory,
                coordinate="x",
                stem="csk_pairs",
                xlabel=r"$x$",
                ylabel=r"$\gamma_q$",
                band=True,
            )
        p, q, b = map(int, match.groups())
        groups.setdefault(b, []).append((p, q, record))

    colors = {(4, 5): "#0072B2", (4, 6): "#D55E00", (5, 6): "#009E73"}
    directory = artifact_directory / "plots"
    directory.mkdir(exist_ok=True)
    lines = []
    for b, pairs in groups.items():
        with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
            fig, axis = plt.subplots(figsize=(5.5, 3.6), layout="constrained")
            for p, q, record in pairs:
                samples = record.output.values.real
                mean, error = samples.mean(axis=0), samples.std(axis=0, ddof=1)
                x = record.output.coords["x"]
                color = colors.get(tuple(sorted((p, q))))
                (line,) = axis.plot(x, mean, color=color, label=f"{p}/{q}")
                axis.fill_between(x, mean - error, mean + error, color=line.get_color(), alpha=0.18)
            windows = [item.params["x_window"] for _, _, item in pairs]
            bounds = [(w["start"], w["stop"]) if isinstance(w, dict) else w for w in windows]
            limits = (min(lo for lo, _ in bounds), max(hi for _, hi in bounds))
            axis.axhline(0, color="gray", linewidth=0.7)
            axis.set(
                xlim=limits,
                xticks=np.linspace(*limits, 5),
                xlabel="x",
                ylabel=r"$\mathrm{Re}\,\gamma_q$",
                title=rf"$b_\perp/a={b}$  ({pairs[0][2].output.attrs['b_perp_fm']:.4f} fm)",
            )
            axis.legend(fontsize=8)
            stem = f"csk_real_b{b}"
            for extension in ("pdf", "png"):
                fig.savefig(directory / f"{stem}.{extension}", dpi=160)
            plt.close(fig)
        lines.extend([f"![Real CSK, b/a={b}](plots/{stem}.png)", "", f"[PDF](plots/{stem}.pdf)", ""])
    return lines


def _power_plots(record, stage_directory):
    """Plot finite-momentum curves at original x coordinates and the fitted real intercept."""
    import matplotlib.pyplot as plt
    import numpy as np

    directory = stage_directory / "plots"
    directory.mkdir(exist_ok=True)
    window = record.params["x_window"]
    lo, hi = (window["start"], window["stop"]) if isinstance(window, dict) else window
    styles = {(5, 6): ("#0072B2", "o"), (4, 6): ("#D55E00", "s"), (4, 5): ("#00885B", "^")}
    style_order = {indices: i for i, indices in enumerate(styles)}
    other_colors = plt.get_cmap("tab20").colors
    other_markers = ("D", "v", "P", "X", "<", ">", "h")
    ensemble = record.output.ensemble
    lines = [
        "", "### Real CS kernel versus x", "",
        "Markers sample the original x grid. Error bars show one pointwise paired-bootstrap standard deviation. "
        "The fit uses the full x grid of the fitted momentum pairs. The gray band spans the fit window and shows "
        "the real infinite-momentum intercept K ± one bootstrap standard deviation. "
        "Pairs labeled held out are shown for comparison.", "",
    ]
    def sort_key(pair):
        indices = tuple(sorted(pair["momentum_indices"]))
        return style_order.get(indices, len(styles)), indices

    for entry in record.summary["diagnostics"]["bT_records"]:
        b = entry["bT_fm"]
        all_pairs = [*entry["pairs"], *entry.get("plot_only_pairs", [])]
        held_out = {tuple(sorted(item["momentum_indices"])) for item in entry.get("plot_only_pairs", [])}
        closure_momenta = {tuple(item["momenta_gev"]) for item in entry["power_fit"]["closure_checks"]}
        b_index = b / ensemble.a_s
        intercept = record.output.array.sel(bT=b).values.real
        center, error = intercept.mean(), intercept.std(ddof=1)
        fig, axis = plt.subplots(figsize=(6.3, 4.2), layout="constrained")
        try:
            handles = []
            for i, item in enumerate(sorted(all_pairs, key=sort_key)):
                p, q = item["momentum_indices"]
                pair = tuple(sorted((p, q)))
                curve = EnsembleData.from_netcdf(record.artifact_directory / item["curve"])
                x = np.asarray(curve.coords["x"])
                n_show = min(10, len(x))
                fraction = (i + .5) / len(all_pairs)
                indices = ((np.arange(n_show) + fraction) * len(x) / n_show).astype(int)
                samples = curve.values.real[:, indices]
                mom = (curve.attrs["momentum1_gev"], curve.attrs["momentum2_gev"])
                label = f"{p}/{q}" + (" (held out)" if pair in held_out or mom in closure_momenta else "")
                fallback = (other_colors[i % len(other_colors)], other_markers[i % len(other_markers)])
                color, marker = styles.get(pair, fallback)
                points = axis.errorbar(
                    x[indices], samples.mean(axis=0), yerr=samples.std(axis=0, ddof=1), fmt=marker,
                    linestyle="none", markersize=5.5, markerfacecolor="white", markeredgewidth=1.1,
                    color=color, elinewidth=1.05, capsize=2.6, label=label, zorder=3,
                )
                handles.append(points)
            band = axis.fill_between([lo, hi], center - error, center + error, facecolor="0.78", alpha=.7,
                                     label=r"$\mathrm{Re}\,K(b_\perp)$, $P^z\to\infty$")
            axis.plot([lo, hi], [center, center], color="0.45", lw=.8)
            axis.axhline(0, color="0.35", ls="--", lw=.9, zorder=1)
            axis.set(xlim=(lo, hi), xlabel=r"$x$", ylabel=r"$\mathrm{Re}\,K(x,b_\perp;\,P_1^z,P_2^z)$",
                     title=rf"{ensemble.id}    $b_\perp/a={b_index:g}$ ({b:g} fm),    "
                           rf"$\mu={record.params['mu']:g}$ GeV")
            axis.tick_params(direction="in", top=True, right=True)
            axis.legend(handles=[*handles, band], title=r"$P_1^z/P_2^z$", ncols=min(4, len(handles) + 1),
                        fontsize=8.5, title_fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False)
            stem = f"{record.job_id}_real_b{b!r}fm"
            for extension in ("pdf", "png"):
                fig.savefig(directory / f"{stem}.{extension}", dpi=180)
        finally:
            plt.close(fig)
        lines += [f"![Real CS kernel and fitted K at b = {b:g} fm](plots/{stem}.png)", "",
                  f"[PDF at b = {b:g} fm](plots/{stem}.pdf)", ""]
    return lines


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    single = tuple(record for record in records if record.params.get("type") != "cs_kernel")
    lines = _pair_report_lines(single, artifact_directory) if single else ["# Soft Analysis"]
    for record in records:
        if record.params.get("type") != "cs_kernel":
            continue
        params = record.params
        diagnostics = record.summary["diagnostics"]
        power = params.get("estimator") == "pca_power"
        prior = diagnostics.get("slope_prior")
        lines += ["", f"## {record.job_id}", ""]
        if power:
            lines += [
                f"The selected ordered momentum pairs are {params['momentum_pairs']}. The fit to independent "
                f"pairs gives one dimensionless infinite-momentum intercept K per bT over x in {params['x_window']}.",
                "",
                r"$$Y_{ij}(x,b_T)=K(b_T)+A(x,b_T)\left[(P_i^z)^{-2}-(P_j^z)^{-2}\right].$$",
                "",
            ]
            if prior is None:
                lines.append("The fit removes the free slope A at each x before estimating the covariance of the "
                             "remaining bootstrap combinations. A generalized least-squares fit in the retained "
                             "covariance modes gives K and the fitted intercept determines the slopes.")
            else:
                lines += [
                    f"Each real and imaginary slope A(x,bT) has an independent Gaussian prior with mean "
                    f"{prior['mean_gev2']:.6g} GeV² and standard deviation {prior['sigma_gev2']:.6g} GeV². "
                    "For a slope shared by all x points, the joint prior has width σ/√Nₓ. The intercept K has "
                    "a flat prior. The fit uses the joint bootstrap covariance of the free K and A estimates "
                    "to propagate the slope constraint to the intercept.",
                    "",
                    "The statistical errors on K and A are fixed-prior paired-bootstrap standard deviations. "
                    "The data vary across replicas while the prior mean and width stay fixed. The diagnostics "
                    "also record the Gaussian posterior standard deviation of K as intercept_posterior_sd.",
                ]
            lines += [
                "The real and imaginary fits preserve the same bootstrap row order. A has units of GeV² "
                "and the inverse-momentum difference has units of GeV⁻².",
                "",
                f"The requested cumulative variance fraction is {diagnostics['pca_variance_fraction']:.6g}.",
            ]
        else:
            lines.append(f"Each bootstrap estimate is averaged equally over x in {params['x_window']} "
                         f"and over the selected momentum pairs {params['momentum_pairs']}.")
        label = "intercept K" if power else "CSK"
        lines += ["Wilson and matching parameters are held fixed.", "",
                  f"| bT [fm] | Real {label} | Statistical error | Imaginary {label} | Statistical error |",
                  "|---:|---:|---:|---:|---:|"]
        for row in diagnostics["estimates"]:
            lines.append("| " + " | ".join(f"{row[key]:.6g}" for key in
                         ("bT_fm", "real", "real_error", "imag", "imag_error")) + " |")
        if power:
            lines += _power_plots(record, artifact_directory)
            if prior is None:
                lines += ["", "| bT [fm] | Component | Retained modes | Retained variance | χ² | Fit dof |",
                          "|---:|---|---:|---:|---:|---:|"]
            else:
                lines += [
                    "", "Data χ² measures the residual in the data likelihood. Prior χ² is the sum of squared "
                    "deviations of the fitted slopes from the prior mean in units of the prior width. "
                    "Effective data dof and effective parameter count are trace-based fit diagnostics.",
                    "", "| bT [fm] | Component | Retained modes | Retained variance | Data χ² | Prior χ² | "
                    "Effective parameters | Effective data dof |",
                    "|---:|---|---:|---:|---:|---:|---:|---:|",
                ]
            for entry in diagnostics["bT_records"]:
                for component in ("real", "imag"):
                    fit = entry["power_fit"][component]
                    if prior is None:
                        fit_columns = f"{fit['chi2']:.6g} | {fit['fit_dof']}"
                    else:
                        fit_columns = " | ".join(f"{fit[key]:.6g}" for key in
                                                 ("chi2", "prior_chi2", "effective_parameters", "effective_data_dof"))
                    lines.append(f"| {entry['bT_fm']:.6g} | {component} | {fit['retained_modes']} | "
                                 f"{fit['explained_variance_fraction']:.6g} | {fit_columns} |")
            lines += ["", "| bT [fm] | Fitted ordered momentum pairs [GeV] | Closure checks among configured pairs "
                      "| Additional pairs plotted (lattice indices) |", "|---:|---|---:|---|"]
            for entry in diagnostics["bT_records"]:
                fit = entry["power_fit"]
                extra_pairs = [item["momentum_indices"] for item in entry.get("plot_only_pairs", [])]
                extra = ", ".join(f"{p}/{q}" for p, q in extra_pairs)
                lines.append(f"| {entry['bT_fm']:.6g} | {format_value(fit['fit_pairs_gev'])} | "
                             f"{len(fit['closure_checks'])} | {extra or '—'} |")
        lines += ["", "| Job | Artifact |", "|---|---|", *artifact_rows(record, artifact_directory)]
    return write_report(artifact_directory, lines)


def _pair_report_lines(records, artifact_directory):
    lines = [
        "# Soft Analysis",
        "",
        "Paired-bootstrap, finite-momentum Collins-Soper kernel estimates from quasi-TMDWF ratios.",
        "",
        r"$$\widehat\gamma_q(x,b_T;P_1,P_2)=\frac{\log[\widetilde\Psi(x,P_1)/\widetilde\Psi(x,P_2)]}"
        r"{\log(P_1/P_2)}+\delta\gamma^{uNLO}(x,b_T;P_1,P_2).$$",
        "",
        "The data logarithm is principal-valued separately for every bootstrap sample and x point. "
        "Its real part uses log magnitudes; the imaginary part is retained as a diagnostic. "
        "No x-plateau fit, averaging over pairs, or infinite-momentum extrapolation is performed.",
        "",
        "## Momentum pairs",
        "",
        "| job | P1 [GeV] | P2 [GeV] | bT [fm] | mu [GeV] | samples | input scheme | hard convention |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for record in records:
        attrs = record.output.attrs
        values = [attrs[key] for key in ("momentum1_gev", "momentum2_gev", "b_perp_fm", "mu")]
        lines.append(
            f"| {record.job_id} | "
            + " | ".join(map(format_value, values))
            + f" | {record.output.n_sample} | {attrs['renormalization_scheme']} | {attrs['hard_convention']} |"
        )
    lines.extend(
        [
            "",
            "## Pair comparison",
            "",
            *_pair_plots(records, artifact_directory),
            "",
            "## Interpretation and errors",
            "",
            "Bootstrap rows must use identical configuration draws and ordering across momenta. "
            "The stage checks ensemble, resample_id, sample coordinates, x grid, bT, and shared conventions; "
            "it does not invent correlations between independently resampled inputs.",
            "",
            "Error bands propagate the paired quasi samples. The perturbative correction is fixed, so matching "
            "scale/coupling variations and any upstream fixed Wilson-loop central-value uncertainty are not included.",
            "",
            "The ratio assumes remaining normalization/soft factors are momentum independent and cancel. "
            "Wilson-loop-divided input retains its original scheme label; this stage does not perform an MSbar "
            "conversion or prove that all finite-momentum artifacts cancel. Residual x and momentum-pair dependence "
            "and the imaginary part remain diagnostics.",
            "",
            "The default author convention uses sqrt(pi) in the two Meijer-G terms. "
            "The hard logarithm is expanded to one loop by default; "
            "principal is available to reproduce the supplied .m.",
        ]
    )
    for record in records:
        diagnostic = record.summary["diagnostics"]
        lines.extend(
            [
                "",
                f"## {record.job_id}",
                "",
                f"Parameters: {format_value(dict(record.params))}",
                "",
                f"Sample/x points near the data-log branch cut (|phase| > 0.9 pi): "
                f"{diagnostic['near_branch_cut_sample_points']}. No phase unwrapping or sample rejection was applied.",
                "",
                *figure_lines(record, artifact_directory),
                "",
                "| job | artifact |",
                "|---|---|",
                *artifact_rows(record, artifact_directory),
            ]
        )
    return lines
