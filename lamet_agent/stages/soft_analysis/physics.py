"""Sample-wise quasi-TMDWF ratios and their additive uNLO CS correction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.tmdwf_gi_unlo import delta_gamma


def load_quasi(source: EnsembleData | Path) -> EnsembleData:
    if isinstance(source, EnsembleData):
        return source
    if isinstance(source, Path) and source.suffix.lower() == ".nc":
        return EnsembleData.from_netcdf(source)
    raise TypeError("Soft Analysis requires one quasi-TMDWF EnsembleData or .nc file per input")


def _positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def transverse_distance(data: EnsembleData) -> float:
    attrs = data.attrs
    candidates = []
    if "b_perp_fm" in attrs:
        candidates.append(_positive(attrs["b_perp_fm"], "b_perp_fm"))
    spacing = attrs.get("a_fm", data.ensemble.a_s if data.ensemble else None)
    if spacing is not None and data.ensemble is not None:
        if not np.isclose(_positive(spacing, "a_fm"), data.ensemble.a_s, rtol=1e-10, atol=1e-12):
            raise ValueError("a_fm disagrees with EnsembleInfo")
    for key in ("b_perp_lattice", "bT"):
        if key in attrs:
            candidates.append(_positive(attrs[key], key) * _positive(spacing, "a_fm"))
    if not candidates:
        raise ValueError("quasi input requires b_perp_fm or b_perp_lattice/bT with lattice spacing")
    if not np.allclose(candidates, candidates[0], rtol=1e-10, atol=1e-12):
        raise ValueError("inconsistent transverse-distance metadata")
    return candidates[0]


def inspect_pair(first: EnsembleData, second: EnsembleData, params: dict) -> dict:
    """Require compatible physics provenance and an identical paired sample axis."""
    for data in (first, second):
        if data.dims != ["x"] or data.resample != "bootstrap" or data.n_sample < 2:
            raise ValueError("quasi inputs must be one-dimensional x distributions with >=2 bootstrap samples")
        if data.attrs.get("target_observable") != "tmdwf" or data.attrs.get("parton") != "quark":
            raise ValueError("Soft Analysis requires quark TMDWF inputs")
        if str(data.attrs.get("gfix", "")).lower() != "gi":
            raise ValueError("uNLO hard matching requires gauge-invariant (GI) quasi-TMDWF inputs")
        if data.attrs.get("output_component") != "both":
            raise ValueError("complex Fourier output_component='both' is required; do not discard the phase")
        if data.attrs.get("normalized_at_origin") != 0:
            raise ValueError("require normalized_at_origin=0; momentum-dependent normalization changes the ratio")
        if data.attrs.get("renormalization_scheme") not in {"msbar", "tmdwf_wilson_loop"}:
            raise ValueError("unsupported quasi-TMDWF renormalization_scheme")
        if "kernel_id" in data.attrs or data.attrs.get("observable") == "collins_soper_kernel":
            raise ValueError("input must precede hard matching; refusing a possible double correction")
        if "wilson_line_sign" in data.attrs and data.attrs["wilson_line_sign"] != params["sign"]:
            raise ValueError("input wilson_line_sign disagrees with matching sign")
        for key in ("mu", "renormalization_scale_gev"):
            if key in data.attrs and not np.isclose(float(data.attrs[key]), params["mu"], rtol=1e-10, atol=1e-12):
                raise ValueError(f"input {key} does not match the requested mu")
        if not np.all(np.isfinite(data.values)):
            raise ValueError("quasi input contains nonfinite samples")
    if first.ensemble is None or first.ensemble != second.ensemble:
        raise ValueError("inputs must have the same explicit EnsembleInfo")
    resample_id = first.attrs.get("resample_id")
    if not isinstance(resample_id, str) or not resample_id or resample_id != second.attrs.get("resample_id"):
        raise ValueError("identical nonempty resample_id is required to establish bootstrap pairing")
    if first.n_sample != second.n_sample or not np.array_equal(
        first.array.coords["resample"], second.array.coords["resample"]
    ):
        raise ValueError("bootstrap sample counts and sample coordinates must agree")
    # Equality of the pairing id asserts the same configuration draws and row order.
    # A matching length alone is not evidence of correlation.
    required = ("renormalization_scheme", "phase_convention", "fourier_convention")
    optional = (
        "source_operator",
        "sink_operator",
        "source_combination",
        "kernel_operator",
        "hadron",
        "phase_sign",
        "x_shift",
        "prefactor",
        "output_scale",
        "component_combination",
        "negative_z",
        "signed_z_completion",
        "L",
        "ZO",
        "factor_uncertainty",
    )
    for key in (*required, *optional):
        left, right = first.attrs.get(key), second.attrs.get(key)
        if (key in required and (left is None or left == "")) or left != right:
            raise ValueError(f"incompatible or missing {key} metadata")
    b1, b2 = transverse_distance(first), transverse_distance(second)
    if not np.isclose(b1, b2, rtol=1e-10, atol=1e-12):
        raise ValueError("different b_perp values cannot form a CS ratio")
    p1 = _positive(first.attrs.get("momentum_gev"), "momentum1_gev")
    p2 = _positive(second.attrs.get("momentum_gev"), "momentum2_gev")
    log_ratio = float(np.log(p1) - np.log(p2))
    if abs(log_ratio) < 1e-10:
        raise ValueError("distinct well-resolved momenta are required")
    grids = []
    masks = []
    lo, hi = params["x_window"]["start"], params["x_window"]["stop"]
    if not 0 < lo < hi < 1:
        raise ValueError("x_window must be strictly inside (0,1)")
    for data in (first, second):
        grid = np.asarray(data.coords["x"], dtype=float)
        if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError("x coordinates must be finite and strictly increasing")
        mask = (grid >= lo) & (grid <= hi)
        grids.append(grid[mask])
        masks.append(mask)
    if grids[0].size < 2 or not np.array_equal(grids[0], grids[1]):
        raise ValueError("x_window must select >=2 identical x points in both inputs; no implicit interpolation")
    return {
        "momentum1_gev": p1,
        "momentum2_gev": p2,
        "log_momentum_ratio": log_ratio,
        "b_perp_fm": b1,
        "x": grids[0],
        "masks": masks,
    }


def compute_pair(first: EnsembleData, second: EnsembleData, params: dict, *, sample_error_mode: str) -> dict:
    info = inspect_pair(first, second, params)
    v1 = np.asarray(first.values, dtype=complex)[:, info["masks"][0]]
    v2 = np.asarray(second.values, dtype=complex)[:, info["masks"][1]]
    floor = _positive(params["min_magnitude"], "min_magnitude")
    if np.any(abs(v1) <= floor) or np.any(abs(v2) <= floor):
        raise ValueError("quasi sample magnitude is below min_magnitude; choose a nonsingular x window")
    phase = np.angle((v1 / abs(v1)) * np.conjugate(v2 / abs(v2)))
    raw_values = (np.log(abs(v1)) - np.log(abs(v2)) + 1j * phase) / info["log_momentum_ratio"]
    x = info["x"]
    correction = np.array(
        [
            delta_gamma(
                float(value),
                momentum1_gev=info["momentum1_gev"],
                momentum2_gev=info["momentum2_gev"],
                b_perp_fm=info["b_perp_fm"],
                scale_gev=params["mu"],
                alpha_s=params["alpha_s"],
                sign=params["sign"],
                convention=params["convention"],
                logarithm=params["logarithm"],
                dps=params["dps"],
            )
            for value in x
        ]
    )
    values = raw_values + correction
    if not np.all(np.isfinite(values)):
        raise ValueError("nonfinite CS kernel estimate")
    attrs = {
        key: value
        for key, value in first.attrs.items()
        if key not in {"momentum_gev", "source_momentum", "sink_momentum", "p1", "p2"}
    }
    attrs.update(
        {
            "observable": "collins_soper_kernel",
            "target_observable": "csk",
            "estimator": "finite_momentum_pair",
            "kernel_id": "tmdwf_gi_unlo",
            "order": "unlo",
            "momentum1_gev": info["momentum1_gev"],
            "momentum2_gev": info["momentum2_gev"],
            "b_perp_fm": info["b_perp_fm"],
            "mu": params["mu"],
            "alpha_s": params["alpha_s"],
            "wilson_line_sign": params["sign"],
            "hard_convention": params["convention"],
            "hard_logarithm": params["logarithm"],
            "data_logarithm": "principal_per_sample",
            "sample_error_mode": sample_error_mode,
            "output_component": "both",
            "soft_analysis_parameters": json.dumps(params, sort_keys=True),
            "matching_uncertainty": "fixed_parameters_not_propagated",
            "scheme_assumption": "momentum-independent residual factors cancel in the ratio",
        }
    )
    raw = EnsembleData(
        first.ensemble,
        "bootstrap",
        list(raw_values),
        ["x"],
        {"x": list(x)},
        attrs={**attrs, "observable": "unmatched_cs_ratio"},
        name="unmatched_cs_ratio",
    )
    output = EnsembleData(
        first.ensemble, "bootstrap", list(values), ["x"], {"x": list(x)}, attrs=attrs, name="collins_soper_kernel"
    )
    diagnostics = {
        "momentum1_gev": info["momentum1_gev"],
        "momentum2_gev": info["momentum2_gev"],
        "b_perp_fm": info["b_perp_fm"],
        "samples": first.n_sample,
        "x_points": len(x),
        "max_abs_ratio_phase": float(np.max(abs(phase))),
        "near_branch_cut_sample_points": int(np.count_nonzero(abs(phase) > 0.9 * np.pi)),
        "min_input_magnitude": float(min(np.min(abs(v1)), np.min(abs(v2)))),
        "input_renormalization_scheme": first.attrs["renormalization_scheme"],
    }
    return {"data": output, "raw": raw, "correction": correction, "diagnostics": diagnostics}
