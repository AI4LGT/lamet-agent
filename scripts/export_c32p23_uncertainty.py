"""Show and export pointwise statistical errors from the saved bootstrap replicas."""

import argparse
import json
from pathlib import Path

import gvar as gv
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from lamet_agent.data import EnsembleData


def export(source):
    data = EnsembleData.from_netcdf(source)
    if data.resample != "bootstrap" or data.dims != ["x"] or data.n_sample < 2:
        raise ValueError("Expected bootstrap replicas on an x grid")
    if not np.isfinite(data.values).all():
        raise ValueError("Nonfinite replicas cannot be silently excluded from uncertainty estimates")
    x = np.asarray(data.coords["x"])
    arrays = {"x": x}
    for part in ("real", "imag"):
        values = getattr(data.values, part)
        center, error = values.mean(axis=0), values.std(axis=0, ddof=1)
        # Match the agent's sample_error_mode='variance'. Bootstrap spread is
        # already the statistical error; do not divide it by sqrt(n_sample).
        reference = getattr(data, part).average("variance")
        np.testing.assert_allclose(center, gv.mean(reference), rtol=1e-14, atol=1e-15)
        np.testing.assert_allclose(error, gv.sdev(reference), rtol=1e-14, atol=1e-15)
        arrays[f"mean_{part}"] = center
        arrays[f"sd_{part}"] = error
        arrays[f"q16_{part}"], arrays[f"q84_{part}"] = np.percentile(values, [16, 84], axis=0)
    arrays["cov_real_imag_at_same_x"] = np.sum(
        (data.values.real - arrays["mean_real"]) * (data.values.imag - arrays["mean_imag"]), axis=0
    ) / (data.n_sample - 1)
    prefix = source.parent / (source.stem + "_uncertainty")
    np.savez_compressed(str(prefix) + ".npz", **arrays)
    with np.load(str(prefix) + ".npz") as restored:
        for key, value in arrays.items():
            np.testing.assert_array_equal(restored[key], value)

    figure = Figure(figsize=(10, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2, sharex=True)
    for col, (part, color, label) in enumerate((("real", "#1665a8", "Real"), ("imag", "#c85116", "Imaginary"))):
        center, error = arrays[f"mean_{part}"], arrays[f"sd_{part}"]
        ax = axes[0, col]
        ax.fill_between(x, center-error, center+error, color=color, alpha=0.3, label=r"Mean $\pm 1\sigma$")
        ax.plot(x, center, color=color, linewidth=1.2)
        sample_points = np.arange(0, len(x), max(len(x)//20, 1))
        ax.errorbar(x[sample_points], center[sample_points], yerr=error[sample_points], fmt="none",
                    color=color, capsize=2, linewidth=0.7)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax.set_title(f"{label} part")
        ax.set_ylabel(r"$\widetilde{\Psi}(x,b_\perp)$")
        ax.legend(loc="best", frameon=False)
        axes[1, col].plot(x, error, color=color)
        axes[1, col].fill_between(x, 0, error, color=color, alpha=0.15)
        axes[1, col].set_ylim(bottom=0)
        axes[1, col].set_ylabel(r"Bootstrap statistical error $\sigma$")
        axes[1, col].set_xlabel("x")
        for row in (0, 1):
            axes[row, col].grid(alpha=0.15)
    figure.suptitle(f"C32P23 quasi-TMDWF: {data.n_sample} bootstrap replicas\n"
                   "Pointwise statistical errors; fixed Wilson/ZO and selected model/windows", fontsize=13)
    figure.savefig(str(prefix) + ".png", dpi=180)
    figure.savefig(str(prefix) + ".pdf")
    points = []
    for requested in (0., 0.25, 0.5, 0.75, 1.):
        i = int(np.argmin(np.abs(x-requested)))
        points.append({"requested_x": requested, "x": float(x[i]), **{
            key: float(arrays[key][i]) for key in ("mean_real", "sd_real", "mean_imag", "sd_imag")}})
    summary = {"source": str(source), "bootstrap_samples": data.n_sample, "ddof": 1,
               "error_definition": "std of bootstrap fitted outputs; no division by sqrt(N_bootstrap)",
               "interpretation": "pointwise propagated statistical error conditional on selected model and windows",
               "fixed_inputs": ["Wilson-loop central factor", "ZO"],
               "systematic_variations_included": False,
               "note": "The Fourier numerical agreement at about 3e-15 is a separate consistency check.",
               "error_ranges": {part: [float(arrays[f'sd_{part}'].min()), float(arrays[f'sd_{part}'].max())]
                                for part in ("real", "imag")}, "representative_points": points}
    Path(str(prefix) + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.source), indent=2))
