"""Soft Analysis report and overlays of all authored momentum-pair jobs."""

import re
from pathlib import Path

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
            limits = (min(w["start"] for w in windows), max(w["stop"] for w in windows))
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


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
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
    return write_report(artifact_directory, lines)
