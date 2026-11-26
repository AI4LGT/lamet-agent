"""Finite-momentum CS kernels from quasi-TMDWF ratios and their power extrapolation."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from sys import float_info

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
    window = params["x_window"]
    lo, hi = (window["start"], window["stop"]) if isinstance(window, dict) else window
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
        if key not in {"momentum_gev", "source_momentum", "sink_momentum", "p1", "p2",
                       "momentum_partitions", "partition_combination", "input_transform",
                       "partition_reference_local_dataset"}
    }
    for index, quasi in enumerate((first, second), start=1):
        parts = quasi.attrs.get("momentum_partitions")
        if parts is None and all(isinstance(quasi.attrs.get(k), (int, np.integer)) for k in ("p1", "p2")):
            parts = json.dumps([[int(quasi.attrs["p1"]), int(quasi.attrs["p2"])]])
        if parts is not None:
            attrs[f"momentum{index}_partitions"] = parts
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
    for curve in (raw, output):
        curve.array = curve.array.assign_coords(resample=first.array.coords["resample"])
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


def _independent_pairs(pairs):
    """Choose a shortest-edge spanning forest, and find each remaining edge's path."""
    graph, selected, closure = {}, [], []
    for index in sorted(range(len(pairs)), key=lambda i: (abs(pairs[i][0] - pairs[i][1]), sorted(pairs[i]))):
        p, q = pairs[index]
        parents = {p: None}
        todo = deque([p])
        while todo and q not in parents:
            node = todo.popleft()
            for neighbor, edge in graph.get(node, ()):
                if neighbor not in parents:
                    parents[neighbor] = (node, edge)
                    todo.append(neighbor)
        if q in parents:
            path = []
            node = q
            while node != p:
                prev, edge = parents[node]
                path.append((edge, np.log(prev / node) / np.log(p / q)))
                node = prev
            closure.append((index, path))
        else:
            selected.append(index)
            graph.setdefault(p, []).append((q, index))
            graph.setdefault(q, []).append((p, index))
    return selected, closure


def _fit_component(values, powers, basis, fraction):
    n, n_pairs, nx = values.shape
    # At each x, projection perpendicular to d eliminates the unrestricted A(x).
    # Only this subspace constrains the shared intercept.
    contrast = np.einsum("npx,pk->nkx", values, basis).reshape(n, -1)
    constant = np.repeat(basis.sum(axis=0), nx)
    norm_d = np.linalg.norm(powers)
    unit = powers / norm_d
    parallel = np.einsum("npx,p->nx", values, unit)
    # Subtracting one replica first keeps an exactly constant sample column at zero.
    centered = contrast - contrast[0]
    centered -= centered.mean(axis=0)
    scale = np.max(np.abs(values))
    roundoff = 64 * np.finfo(float).eps
    if np.max(np.abs(centered)) <= roundoff * scale:
        intercept = contrast[0] @ constant / (constant @ constant)
        if np.max(np.abs(contrast[0] - intercept * constant)) > roundoff * scale * n_pairs:
            raise ValueError("zero bootstrap contrast variance with a nonconstant intercept across x or pairs")
        k = np.full(n, intercept)
        slopes = (parallel - k[:, None] * unit.sum()) / norm_d
        return k, slopes, dict(retained_modes=0, numerical_rank=0, fit_dof=0, chi2=0.0,
                               explained_variance_fraction=0.0, mode_variance_fractions=[],
                               deterministic_contrast=True)

    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2 / (n - 1)
    rank = int(np.count_nonzero(singular > singular[0] * np.finfo(float).eps * max(centered.shape)))
    fractions = variance[:rank] / variance[:rank].sum()
    modes = min(rank, int(np.searchsorted(np.cumsum(fractions), fraction) + 1))
    vectors = axes[:modes].T
    projected_constant = constant @ vectors
    if np.linalg.norm(projected_constant) <= roundoff * np.linalg.norm(constant):
        raise ValueError("retained PCA modes cannot identify K(b); increase pca_variance_fraction")
    whitener = vectors / np.sqrt(variance[:modes])
    design = constant @ whitener
    scores = contrast @ whitener
    k = scores @ design / (design @ design)
    residual = scores - k[:, None] * design
    # Conditional GLS estimate for the discarded pair direction. This restores a
    # free slope on every original x point, including its covariance with K(b).
    cross_cov = (centered @ whitener).T @ (parallel - parallel.mean(axis=0)) / (n - 1)
    slopes = (parallel - k[:, None] * unit.sum() - residual @ cross_cov) / norm_d
    central_residual = residual.mean(axis=0)
    return k, slopes, dict(
        retained_modes=modes, numerical_rank=rank, fit_dof=modes - 1,
        chi2=float(central_residual @ central_residual),
        explained_variance_fraction=float(fractions[:modes].sum()),
        mode_variance_fractions=fractions[:modes].tolist(), deterministic_contrast=False,
    )


def valid_slope_prior(value):
    """Validate an independent Gaussian prior on each x point, in GeV²."""
    return (
        isinstance(value, dict) and set(value) == {"mean_gev2", "sigma_gev2"}
        and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
                and -float_info.max <= v <= float_info.max for v in value.values())
        and value["sigma_gev2"] > 0
    )


def _apply_slope_prior(k, slopes, diagnostics, prior):
    """Update the joint GLS estimates of (K, A) with fixed Gaussian prior centers.

    With V the paired covariance of the unconstrained GLS estimates, the update
    is theta -= V[:, A] @ inv(V[A, A] + sigma² I) @ (A - mean). The SVD handles
    exactly correlated x points without inverting their covariance.
    """
    mean, sigma = prior["mean_gev2"], prior["sigma_gev2"]
    n, nx = slopes.shape
    slope_deviations = slopes - slopes[0]
    slope_deviations -= slope_deviations.mean(axis=0)
    k_deviations = k - k[0]
    k_deviations -= k_deviations.mean()
    normalization = np.sqrt(n - 1)
    u, singular, axes = np.linalg.svd(slope_deviations / normalization, full_matrices=False)
    keep = singular > singular[0] * np.finfo(float).eps * max(n, nx)
    u, singular, axes = u[:, keep], singular[keep], axes[keep]
    total_sd = np.hypot(singular, sigma)
    shrink = singular / total_sd
    retain = sigma / total_sd
    k_deviations /= normalization
    k_modes = k_deviations @ u
    slope_modes = (slopes - mean) @ axes.T
    fitted_k = k - slope_modes @ (k_modes * shrink / total_sd)
    # A component with zero sampling variance is fixed by the data. Such a
    # component is identical across replicas even when its prior mean differs.
    constant_part = np.zeros(nx)
    if singular.size < nx:
        constant_part = (slopes - mean).mean(axis=0) - slope_modes.mean(axis=0) @ axes
    fitted_slopes = mean + constant_part + (slope_modes * retain**2) @ axes
    extra_data_chi2 = np.sum((slope_modes.mean(axis=0) * shrink / total_sd)**2)
    data_chi2 = float(diagnostics["chi2"] + extra_data_chi2)
    prior_chi2 = float(np.sum(((fitted_slopes.mean(axis=0) - mean) / sigma)**2))
    shrink_trace = float(shrink @ shrink)
    posterior_k_sd = np.hypot(np.linalg.norm(k_deviations - u @ k_modes), np.linalg.norm(k_modes * retain))
    return fitted_k, fitted_slopes, dict(
        diagnostics, unconstrained_chi2=diagnostics["chi2"], chi2=data_chi2, data_chi2=data_chi2,
        prior_chi2=prior_chi2, augmented_chi2=data_chi2 + prior_chi2,
        effective_parameters=1 + nx - shrink_trace, effective_data_dof=diagnostics["fit_dof"] + shrink_trace,
        intercept_posterior_sd=float(posterior_k_sd),
    )


def fit_power(curves, variance_fraction, slope_prior=None):
    """Return paired K(b) samples, paired A(x,b) samples (GeV²), and fit diagnostics.

    The pooled bootstrap covariance and PCA basis are fixed across replicas.
    Real and imaginary components each use their own covariance and common x intercept.
    An extra pair closing a momentum cycle is checked against its path through the
    independent ratios and held out of the fit. When supplied, slope_prior sets
    the Gaussian mean and standard deviation for each real/imaginary A(x,b).
    The prior center is held fixed across bootstrap draws.
    """
    if (isinstance(variance_fraction, (bool, np.bool_)) or not np.isscalar(variance_fraction)
            or not isinstance(variance_fraction, (int, float, np.integer, np.floating))
            or not 0 < variance_fraction <= 1):
        raise ValueError("pca_variance_fraction must lie in (0, 1]")
    if len(curves) < 2:
        raise ValueError("power extrapolation requires at least two distinct momentum pairs")
    if slope_prior is not None:
        if not valid_slope_prior(slope_prior):
            raise ValueError("slope_prior requires finite mean_gev2 and positive finite sigma_gev2")
        slope_prior = {key: float(value) for key, value in slope_prior.items()}
    first = curves[0]
    pairs = []
    for curve in curves:
        if (curve.dims != ["x"] or curve.resample != "bootstrap" or curve.n_sample < 2
                or curve.ensemble != first.ensemble or curve.coords != first.coords
                or not first.attrs.get("resample_id") or curve.attrs.get("resample_id") != first.attrs["resample_id"]
                or not np.array_equal(curve.array.coords["resample"], first.array.coords["resample"])
                or curve.attrs.get("b_perp_fm") != first.attrs.get("b_perp_fm")
                or not np.isfinite(curve.values).all()):
            raise ValueError("power fit requires matched finite bootstrap curves with shared b, x and replica frame")
        pair = (float(curve.attrs["momentum1_gev"]), float(curve.attrs["momentum2_gev"]))
        if not np.isfinite(pair).all() or min(pair) <= 0 or pair[0] == pair[1]:
            raise ValueError("power fit requires distinct, positive physical momenta in GeV")
        pairs.append(pair)
    if len({tuple(sorted(pair)) for pair in pairs}) != len(pairs):
        raise ValueError("repeated or reversed copy of a momentum pair in the power fit")
    selected, held_out = _independent_pairs(pairs)
    momenta = np.asarray([pairs[index] for index in selected])
    powers = 1 / momenta[:, 0]**2 - 1 / momenta[:, 1]**2
    if len(powers) < 2 or np.ptp(powers) <= 1e-12 * np.max(np.abs(powers)):
        raise ValueError("independent pairs need distinct inverse-square momentum differences to identify K(b)")
    basis = np.linalg.qr(powers[:, None], mode="complete")[0][:, 1:]
    values = np.stack([curves[index].values for index in selected], axis=1)
    real_k, real_a, real_diag = _fit_component(values.real, powers, basis, variance_fraction)
    imag_k, imag_a, imag_diag = _fit_component(values.imag, powers, basis, variance_fraction)
    if slope_prior is not None:
        real_k, real_a, real_diag = _apply_slope_prior(real_k, real_a, real_diag, slope_prior)
        imag_k, imag_a, imag_diag = _apply_slope_prior(imag_k, imag_a, imag_diag, slope_prior)
    intercepts = real_k + 1j * imag_k
    slopes = real_a + 1j * imag_a
    closure_checks = []
    for index, path in held_out:
        p, q = pairs[index]
        observed = curves[index].values
        path_curve = sum(weight * curves[edge].values for edge, weight in path)
        model_residual = (observed - intercepts[:, None] - (p**-2 - q**-2) * slopes).mean(axis=1)
        closure_checks.append({
            "momenta_gev": [p, q], "max_abs_log_closure": float(np.max(np.abs(observed - path_curve))),
            "x_mean_model_residual": {name: [float(part.mean()), float(part.std(ddof=1))]
                                      for name, part in (("real", model_residual.real), ("imag", model_residual.imag))},
        })
    diagnostics = dict(
        fit_pairs_gev=momenta.tolist(), delta_inv_p2_gev_minus2=powers.tolist(),
        pca_variance_fraction=float(variance_fraction), closure_checks=closure_checks,
        real=real_diag, imag=imag_diag,
    )
    if slope_prior is not None:
        diagnostics.update(slope_prior=dict(slope_prior), prior_structure="independent_per_x_and_component",
                           uncertainty="fixed_prior_paired_bootstrap")
    return intercepts, slopes, diagnostics
