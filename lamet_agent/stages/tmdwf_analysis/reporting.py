"""Report completed TMDWF jobs without repeating any numerical calculation."""


def write_stage_report(*, records, artifact_directory):
    path = artifact_directory / "report.md"
    lines = ["# C32P23 bootstrap TMDWF", "",
             "This stage applies the authored TMDWF conventions. Wilson-loop and ZO uncertainties "
             "are not propagated in fixed-central-value mode. Fourier output precedes hard matching. "
             "The legacy negative-z extension makes both real and imaginary parts even.", ""]
    if any(record.summary["result"] == "native_bridge" for record in records):
        lines += ["Native bridge inputs are four separately fitted source/direction channels. "
                  "They are averaged after fitting, divided by the Wilson factor, and split into "
                  "real/imaginary midpoint amplitudes. Each is multiplied by exp(-i*lambda/2) "
                  "for the existing Fourier stage's explicitly chosen pion-DA tail template.", ""]
    for record in records:
        lines += [f"## {record.job_id}", "", f"Operation: {record.summary['result']}", "",
                  f"[Diagnostics]({record.job_id}/diagnostics.json) | [Samples]({record.job_id}/output.nc)", "",
                  f"![Result]({record.job_id}/plot.png)", ""]
        diagnostics = record.summary["diagnostics"]
        if "median_Q" in diagnostics:
            lines += [f"Median Q: {diagnostics['median_Q']:.6g}. "
                      f"Fraction with Q < 0.05: {diagnostics['fraction_Q_below_0_05']:.1%}. "
                      "Low-Q results are retained for reproduction and require scientific review.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
