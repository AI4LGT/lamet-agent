"""Stage-level perturbative-matching reporting."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels import load_kernel_document
from lamet_agent.stages.perturbative_matching.physics import is_even_about_zero, load_data, select_output_component
from lamet_agent.stages._reporting import (
    StageReportRecord,
    artifact_rows,
    describe_grid,
    figure_lines,
    format_value,
    grid_segments,
    output_attrs,
    stage_overlay_lines,
    write_report,
)


def _kernel_document(kernel_id: str) -> str:
    return load_kernel_document(kernel_id)


def _record_kernel_id(record: StageReportRecord) -> str:
    value = output_attrs(record).get("kernel_id", record.params.get("kernel_id"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"matching record '{record.job_id}' has no derived kernel_id")
    return value


def _integral(data: EnsembleData, *, lo: float, hi: float) -> float | complex:
    if data.dims != ["x"]:
        raise ValueError("matching report requires one-dimensional x distributions")
    x = np.asarray(data.coords["x"], dtype=float)
    selected = select_output_component(data)
    values = (
        np.mean(np.asarray(selected.values), axis=0)
        if np.iscomplexobj(selected.values)
        else np.asarray(selected.mean)
    )
    if lo == hi:
        return 0.0
    if lo < x[0] or hi > x[-1]:
        raise ValueError("matching report integration window exceeds input coverage")
    interior = (x > lo) & (x < hi)
    nodes = np.r_[lo, x[interior], hi]
    integral = np.trapezoid(np.interp(nodes, x, values), nodes)
    return complex(integral) if np.iscomplexobj(values) else float(integral)


def _common_intervals(record: StageReportRecord) -> list[tuple[float, float]]:
    quasi = record.inputs.get("quasi")
    if not isinstance(quasi, EnsembleData) or not isinstance(record.output, EnsembleData):
        raise TypeError("matching report requires numerical quasi and matched distributions")
    x_in = np.asarray(quasi.coords["x"], dtype=float)
    x_out = np.asarray(record.output.coords["x"], dtype=float)
    intervals = []
    for output_indices in grid_segments(x_out):
        for input_indices in grid_segments(x_in):
            lo = float(max(x_out[output_indices[0]], x_in[input_indices[0]]))
            hi = float(min(x_out[output_indices[-1]], x_in[input_indices[-1]]))
            if hi > lo:
                intervals.append((lo, hi))
    return intervals


def _diagnostics(record: StageReportRecord) -> tuple[float | complex, float | complex, float]:
    intervals = _common_intervals(record)
    if not intervals:
        return float("nan"), float("nan"), float("nan")
    quasi = record.inputs["quasi"]
    quasi_integral = sum(_integral(quasi, lo=lo, hi=hi) for lo, hi in intervals)
    matched_integral = sum(_integral(record.output, lo=lo, hi=hi) for lo, hi in intervals)
    relative = abs(matched_integral - quasi_integral) / abs(quasi_integral) if quasi_integral else float("nan")
    return quasi_integral, matched_integral, relative


def _scheme_text(scheme: str) -> str:
    return {
        "ratio": "The ratio kernel uses the regular coefficient without an additional finite conversion.",
        "msbar": "The MSbar kernel includes the finite MSbar conversion at the declared scale.",
        "hybrid": (
            "The hybrid kernel adds the Wilson-line sine-integral correction and depends on the dimensionless "
            "product $z_sP_z$."
        ),
    }[scheme]


def _kernel_structure(kernel_id: str) -> dict[str, object]:
    tokens = kernel_id.split("_")
    gauge_index = next((index for index, token in enumerate(tokens) if token in {"gi", "cg"}), None)
    if gauge_index is None or len(tokens) < 5 or gauge_index + 1 >= len(tokens):
        raise ValueError(f"kernel id has no recognized gauge/operator structure: {kernel_id}")
    scheme_tokens = [token for token in tokens if token in {"ratio", "hybrid", "msbar"}]
    if len(scheme_tokens) != 1:
        raise ValueError(f"kernel id must contain exactly one scheme: {kernel_id}")
    distribution = (
        "DA"
        if "da" in tokens[:gauge_index]
        else "PDF"
        if "pdf" in tokens[:gauge_index]
        else "GPD"
        if "gpd" in tokens[:gauge_index]
        else None
    )
    if distribution is None:
        raise ValueError(f"kernel id has no PDF/DA/GPD target: {kernel_id}")
    return {
        "gauge": tokens[gauge_index].upper(),
        "operator": tokens[gauge_index + 1],
        "distribution": distribution,
        "scheme": scheme_tokens[0],
        "order": next((token.upper() for token in reversed(tokens) if token in {"lo", "nlo", "nnlo"}), "not encoded"),
        "component": next((token for token in tokens if token in {"re", "im"}), "full"),
        "resummation": "+".join(token.upper() for token in ("rgr", "lrr") if token in tokens) or "none",
    }


def _has_interior_gap(data: EnsembleData) -> bool:
    return len(grid_segments(data.coords["x"])) > 1


def _resolved_record(record: StageReportRecord) -> StageReportRecord:
    quasi = record.inputs.get("quasi")
    if isinstance(quasi, list):
        if len(quasi) != 1:
            raise ValueError("matching accepts one quasi source")
        quasi = quasi[0]
    return replace(record, inputs={**record.inputs, "quasi": load_data(quasi)})


def write_stage_report(*, records: tuple[StageReportRecord, ...], artifact_directory: Path) -> Path:
    records = tuple(_resolved_record(record) for record in records)
    kernel_ids = list(dict.fromkeys(_record_kernel_id(record) for record in records))
    lines = [
        "# Perturbative Matching Stage Report",
        "",
        "This stage applies the selected perturbative matching kernel sample by sample to convert quasi-distributions "
        "into light-cone distributions.",
        "",
        "## Job Summary",
        "",
        r"| job | kernel | scheme | momentum [GeV] | $\mu$ [GeV] | quasi partial integral | matched partial integral | "
        "relative change |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    cached_diagnostics: dict[str, tuple[float | complex, float | complex, float]] = {}
    for record in records:
        attrs = output_attrs(record)
        quasi_integral, matched_integral, relative = _diagnostics(record)
        cached_diagnostics[record.job_id] = (quasi_integral, matched_integral, relative)
        lines.append(
            f"| `{record.job_id}` | `{_record_kernel_id(record)}` | `{attrs['renormalization_scheme']}` | "
            f"{format_value(attrs.get('momentum_gev'))} | {format_value(record.params['mu'])} | "
            f"{format_value(quasi_integral)} | {format_value(matched_integral)} | {format_value(100.0 * relative)}% |"
        )
    lines.extend(
        [
            "",
            "The integrals use the intersection of supported quasi and light-cone intervals for both arrays. "
            "Coverage outside that intersection and detected gaps in either grid are excluded. "
            "Gaps are spacings greater than twice the median spacing on grids with at least four points. "
            "Values are linearly interpolated at interval boundaries; isolated points contribute zero area. "
            "These partial integrals are diagnostics, not a "
            "normalization verdict: the expected normalization is fixed upstream by the coordinate-space matrix "
            "element and its projection convention.",
            "",
            "## Kernel-id and Field Definitions",
            "",
            "| field | meaning |",
            "|---|---|",
            "| `order` | Explicit perturbative order: `nlo` or the supported `nnlo` channel. "
            "Encoded in the kernel filename. |",
            "| `kernel_id` | Runtime-derived public kernel filename stem built from upstream "
            "renormalization and operator provenance, "
            "`order`, and resummation options. |",
            "| `resummation` | Empty selects fixed order; GI gamma-t hybrid supports rgr_lrr at NLO/NNLO. "
            "Unpolarized nonsinglet `rgr` accepts `re`, `im`, or `both` without a suffix. "
            "Legacy polarized RGR derives its suffix from `source_component`; `lrr` has no suffix. |",
            "| `mu` | MSbar renormalization/matching scale in GeV. |",
            "| `zs_fm` | Wilson-line switching distance inherited from upstream hybrid renormalization. |",
            "| `kernel_parameters` | Kernel-signature parameters not supplied by the stage, such as `kappa` "
            "and `mu_min_gev`. |",
            "| matching matrix | Discretized convolution from the quasi input grid to the requested light-cone "
            "output grid. |",
            "",
            "## Stage Overview",
            "",
            *stage_overlay_lines(
                records,
                artifact_directory,
                coordinate="x",
                stem="matching_overview",
                xlabel=r"$x$",
                ylabel="matched distribution",
                band=True,
                split_gaps=True,
            ),
        ]
    )
    for kernel_id in kernel_ids:
        structure = _kernel_structure(kernel_id)
        lines.extend(
            [
                "",
                f"## Kernel `{kernel_id}`",
                "",
                "| property | value |",
                "|---|---|",
                *[f"| {name} | `{format_value(value)}` |" for name, value in structure.items()],
                "",
                "### Matching Formula and Literature Consistency Check",
                "",
                _kernel_document(kernel_id),
            ]
        )
    for record in records:
        attrs = output_attrs(record)
        diagnostics = record.summary.get("diagnostics", {})
        quasi_integral, matched_integral, relative = cached_diagnostics[record.job_id]
        quasi = record.inputs["quasi"]
        scale = float(attrs.get("output_scale", 1.0))
        mirrored = abs(scale - 1.0) > 1e-12 and is_even_about_zero(record.output)
        gap = _has_interior_gap(record.output)
        intervals = _common_intervals(record)
        lines.extend(
            [
                "",
                f"## `{record.job_id}`",
                "",
                "### Analysis Settings",
                "",
                "| quantity | value |",
                "|---|---|",
                f"| kernel | `{_record_kernel_id(record)}` |",
                f"| scheme | `{attrs['renormalization_scheme']}` |",
                f"| order | `{record.params.get('order', 'nlo')}` |",
                f"| momentum | {format_value(attrs.get('momentum_gev'))} GeV |",
                f"| renormalization scale | {format_value(record.params['mu'])} GeV |",
                f"| hybrid switch | {format_value(attrs.get('zs_fm'))} fm |",
                f"| quasi grid | {describe_grid(quasi.coords['x'], symbol='x')} |",
                f"| light-cone grid | {describe_grid(record.output.coords['x'], symbol='x')} |",
                f"| kernel parameters | {format_value(record.params['kernel_parameters'])} |",
                f"| matching matrix shape | {format_value(diagnostics.get('matrix_shape'))} |",
                f"| resampling | `{getattr(record.output, 'resample', 'n/a')}` with "
                f"{format_value(getattr(record.output, 'n_sample', None))} samples |",
                "",
                "### Partial Integral Diagnostic",
                "",
                "- Common integration intervals: " + (
                    ", ".join(f"[{format_value(lo)}, {format_value(hi)}]" for lo, hi in intervals)
                    if intervals else "unavailable: no common interval of positive width."
                ),
                f"- Quasi input: {format_value(quasi_integral)}",
                f"- Matched output: {format_value(matched_integral)}",
                f"- Relative change: {format_value(100.0 * relative)}%",
                f"- Fourier projection scale: {format_value(scale)}",
                *(
                    [
                        "- The stored matched distribution is symmetric about x=0; one-sided quasi/matched "
                        "integrals after removing the projection scale are "
                        f"{format_value(quasi_integral / scale)} / {format_value(matched_integral / scale)}."
                    ]
                    if mirrored
                    else []
                ),
                *(
                    [
                        "- Partial integrals: the matched grid contains an interior gap, which is excluded "
                        "from both quasi and matched integrals. These are not full-support normalization integrals."
                    ]
                    if gap
                    else []
                ),
                "- Compare these values with the normalization convention fixed upstream (`normalization=true` "
                "gives unity only for the corresponding operator/projection convention).",
                "",
                "### Matching Scheme",
                "",
                _scheme_text(str(attrs["renormalization_scheme"])),
                "",
                "The LO contribution is the identity. The shipped kernel document above is the source of truth "
                "for the implemented fixed-order coefficients, plus prescription, support regions, "
                "and any RGR or hybrid correction.",
                "",
                "### Figures",
                "",
                *figure_lines(record, artifact_directory),
                "",
                "### Artifacts",
                "",
                "| job | artifact |",
                "|---|---|",
                *artifact_rows(record, artifact_directory),
            ]
        )
    return write_report(artifact_directory, lines)
