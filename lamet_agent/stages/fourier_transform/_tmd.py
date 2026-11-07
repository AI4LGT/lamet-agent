"""Measured-coordinate TMD transforms, with explicit off-axis geometry."""

from __future__ import annotations

import json

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.plotting import configure_plot, errorband, save_figure, start_plot
from lamet_agent.stages.fourier_transform.physics import load_data


def transform_tmd(
    data: EnsembleData,
    x: np.ndarray,
    *,
    observable: str,
    projection: str = "real_even",
    taper_start: int = 13,
    taper_stop: int = 20,
) -> EnsembleData:
    """Transform the centered WF or the authored PDF projection sample by sample.

    The ensemble spacing remains the underlying lattice spacing. The z grid
    already includes the norm of its displacement direction; momentum_gev is
    the projection onto that direction, so their product is P dot z.
    """
    if observable not in {"tmdwf", "tmdpdf"}:
        raise ValueError("TMD Fourier observable must be tmdwf or tmdpdf")
    if projection not in {"real_even", "hermitian"}:
        raise ValueError("TMD Fourier projection must be real_even or hermitian")
    if projection == "hermitian":
        if observable != "tmdpdf":
            raise ValueError("Hermitian TMD Fourier projection is supported only for tmdpdf")
        if data.attrs.get("output_component") != "both":
            raise ValueError("Hermitian TMDPDF Fourier requires input attrs.output_component='both'; "
                             "fit both real and imaginary channels before this transform")
    if data.dims != ["z"] or data.attrs.get("coord_unit") != "fm":
        raise ValueError("TMD Fourier requires one z dimension with physical fm coordinates")
    if data.attrs.get("observable") != observable:
        raise ValueError("TMD Fourier observable disagrees with the input provenance")
    if observable == "tmdwf" and str(data.attrs.get("phase_rotation_applied", "false")).lower() not in {"true", "1"}:
        raise ValueError("TMDWF Fourier requires provenance for the centered -i exp(-i P.z/2) phase rotation")
    if data.resample not in {"bootstrap", "jackknife"}:
        raise ValueError("TMD Fourier requires sample-bearing bootstrap or jackknife data")
    z = np.asarray(data.coords["z"], dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("TMD Fourier x grid must be finite and strictly increasing")
    if z.size < 2 or not np.isclose(z[0], 0) or np.any(np.diff(z) <= 0):
        raise ValueError("TMD Fourier input must start at z=0 and increase on the measured positive branch")
    step = float(z[1] - z[0])
    if not np.allclose(np.diff(z), step, rtol=1e-10, atol=1e-12):
        raise ValueError("TMD Fourier requires a uniform longitudinal displacement grid")
    momentum = float(data.attrs["momentum_gev"])
    if not np.isfinite(momentum) or momentum <= 0:
        raise ValueError("TMD Fourier requires a positive momentum along the displacement")
    raw_values = np.asarray(data.values)
    values = np.real(raw_values)
    checked_values = raw_values if projection == "hermitian" else values
    if not np.all(np.isfinite(checked_values)):
        bad = np.argwhere(~np.isfinite(checked_values))[0].tolist()
        raise ValueError(f"TMD Fourier nonfinite input at replica/z index {bad}")
    taper = np.ones(z.size)
    if observable == "tmdpdf":
        if not 0 <= taper_start < taper_stop:
            raise ValueError("PDF taper requires 0 <= start < stop")
        if z[-1] / step < taper_stop - 1e-8:
            raise ValueError("PDF measured z grid does not cover taper_stop")
        taper = np.clip((taper_stop - z / step) / (taper_stop - taper_start), 0.0, 1.0)
    shift = 0.5 if observable == "tmdwf" else 0.0
    phase = (x[:, None] - shift) * momentum * z[None, :] / HBAR_C_GEV_FM
    # Full lattice-sum weights: the origin appears once, reflected sites twice.
    weights = taper * 2.0
    weights[0] = taper[0]
    matrix = np.cos(phase) * weights * (momentum * step / (2 * np.pi * HBAR_C_GEV_FM))
    result = values @ matrix.T
    if projection == "hermitian":
        # h(-z)=h(z)* and exp(-i x Pz) give Re(h) cos + Im(h) sin.
        # The reference explicitly negates Im(h) before its positive-sign FT;
        # expressing that convention here avoids mutating upstream samples.
        sine_matrix = np.sin(phase) * weights * (momentum * step / (2 * np.pi * HBAR_C_GEV_FM))
        result += np.imag(raw_values) @ sine_matrix.T
    attrs = dict(data.attrs)
    attrs.update(
        {
            "observable": observable,
            "target_observable": "tmd",
            "distribution": "quasi",
            "source_component": "both" if projection == "hermitian" else "re",
            "output_component": "re",
            "sector": "full" if observable == "tmdwf" or projection == "hermitian" else "valence",
            "fourier_projection": projection,
            "fourier_strategy": "direct" if observable == "tmdwf" else "linear_taper",
            "fourier_x_shift": shift,
            "fourier_phase_sign": -1 if projection == "hermitian" else 1,
            "fourier_prefactor": "pz_over_2pi",
            "fourier_flavor_factor": 1.0,
            "z_step_fm": step,
            "z_max_fm": float(z[-1]),
            "coord_unit": "dimensionless",
            "units": json.dumps({"x": "dimensionless", "values": "dimensionless", "bT": "fm"}),
        }
    )
    if observable == "tmdpdf":
        attrs.update({"taper_start": taper_start, "taper_stop": taper_stop})
    output = EnsembleData(
        data.ensemble, data.resample, list(result), ["x"], {"x": x.tolist()}, attrs=attrs, name=f"quasi_{observable}"
    )
    output.array = output.array.assign_coords(resample=data.array.coords["resample"])
    return output


def run(context) -> None:
    data = load_data(context.inputs["input"])
    grid = context.params["quasi_y_ls"]
    x = (
        np.linspace(float(grid["start"]), float(grid["stop"]), int(grid["num"]))
        if isinstance(grid, dict)
        else np.asarray(grid, dtype=float)
    )
    output = transform_tmd(
        data,
        x,
        observable=context.params["observable"],
        projection=context.params.get("projection", "real_even"),
        taper_start=context.params["taper_start"],
        taper_stop=context.params["taper_stop"],
    )
    directory = context.artifact_directory
    output.to_netcdf(directory / "output.nc")
    diagnostics = {
        "input_z_count": len(data.coords["z"]),
        "output_x_count": len(x),
        "z_step_fm": output.attrs["z_step_fm"],
        "bT_fm": output.attrs["bT_fm"],
        "imaginary_channel_rms": float(np.sqrt(np.mean(np.imag(data.values) ** 2))),
        "projection": output.attrs["fourier_projection"],
        "projection_definition": (
            "Re(h) cos(xPz) + Im(h) sin(xPz); h(-z)=conjugate(h(z)); exp(-i xPz)"
            if output.attrs["fourier_projection"] == "hermitian"
            else "real centered WF" if context.params["observable"] == "tmdwf" else "real even valence PDF"
        ),
        "source_component": output.attrs["source_component"], "sector": output.attrs["sector"],
        "flavor_factor": 1.0,
    }
    (directory / "diagnostics").mkdir(exist_ok=True)
    (directory / "diagnostics" / "transform.json").write_text(json.dumps(diagnostics, indent=2))
    (directory / "plots").mkdir(exist_ok=True)
    start_plot()
    errorband(x, output.gvar)
    configure_plot(xlabel=r"$x$", ylabel=r"$\widetilde f(x,b_T)$")
    save_figure(directory / "plots" / "quasi.svg")
    context.finish(
        output,
        {
            "stage_id": context.stage_id,
            "job_id": context.job_id,
            "result": "quasi_distribution",
            "decisions": {
                "observable": context.params["observable"],
                "strategy": output.attrs["fourier_strategy"],
                "x_shift": output.attrs["fourier_x_shift"],
                "projection": output.attrs["fourier_projection"],
                "sector": output.attrs["sector"],
            },
            "diagnostics": diagnostics,
            "artifacts": ["output.nc", "diagnostics/transform.json", "plots/quasi.svg"],
        },
    )
