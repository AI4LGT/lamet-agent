"""Sample-preserving extraction of the CG Collins–Soper and intrinsic-soft functions."""

from __future__ import annotations

import json
from functools import lru_cache
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from scipy.stats import chi2

from lamet_agent.data import (
    EnsembleData, align_sample_lineage, merge_sample_lineage, sample_covariance, sample_mean, sample_sdev,
)
from lamet_agent.kernels.tmd.cg_nlo import rapidity_factor
from lamet_agent.kernels.tmd.coefficients import coefficient_metadata, form_factor_kernel, hard_factor
from lamet_agent.stages._priors import validate_prior_overrides


def load(value) -> EnsembleData:
    if isinstance(value, EnsembleData):
        return value
    if isinstance(value, (str, Path)) and Path(value).suffix == ".nc":
        return EnsembleData.from_netcdf(value)
    raise TypeError("TMD input must be an EnsembleData artifact or a NetCDF path")


def load_many(value) -> list[EnsembleData]:
    values = value if isinstance(value, list) else [value]
    if not values:
        raise ValueError("TMD input source list must not be empty")
    return [load(item) for item in values]


def samples(data: EnsembleData) -> np.ndarray:
    if data.resample not in {"bootstrap", "jackknife"}:
        raise ValueError("TMD stages require bootstrap or jackknife replicas with explicit sample lineage")
    if not data.attrs.get("resample_lineage") and not data.attrs.get("resample_id"):
        raise ValueError("TMD input requires explicit resample_lineage or a shared legacy resample_id")
    if data.n_sample < 2:
        raise ValueError("TMD statistical analysis requires at least two replicas")
    values = np.asarray(data.values)
    if np.iscomplexobj(values):
        if np.any(np.abs(values.imag) > 1e-12 * np.maximum(1.0, np.abs(values.real))):
            raise ValueError("TMD extraction expects the real pion channel; project the Fourier output explicitly")
        values = values.real
    if not np.all(np.isfinite(values)):
        raise ValueError("TMD input contains nonfinite replicas; fix the upstream fit rather than dropping samples")
    return np.asarray(values, dtype=float)


def positive(name: str, values) -> None:
    values = np.asarray(values)
    bad = ~np.isfinite(values) | (values <= 0)
    if np.any(bad):
        indices = np.argwhere(bad)
        raise ValueError(
            f"{name} must be finite and positive: {len(indices)} invalid entries; "
            f"first indices={indices[:5].tolist()}. Revise the physical x/b range or upstream fits; "
            "replicas are never clipped or discarded."
        )


def numeric_attr(data: EnsembleData, key: str, *, nonnegative: bool = False) -> float:
    value = data.attrs.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"TMD input requires finite numeric attrs.{key}")
    if value < 0 or (value == 0 and not nonnegative):
        raise ValueError(f"TMD attrs.{key} must be {'nonnegative' if nonnegative else 'positive'}")
    return float(value)


def validate_lattice_geometry(*datasets: EnsembleData) -> None:
    """Require common lattice geometry, while allowing different valence masses.

    Entirely ensemble-free physical inputs remain usable for analytic calculations.
    Once any input declares a lattice, every input must declare compatible geometry.
    """
    ensembles = [data.ensemble for data in datasets]
    if not any(ensemble is not None for ensemble in ensembles):
        return
    if any(ensemble is None for ensemble in ensembles):
        raise ValueError("TMD inputs must all declare lattice geometry when any input declares an ensemble")
    first = ensembles[0]
    for field in ("a_s", "a_t", "L_s", "L_t"):
        expected = getattr(first, field)
        for ensemble in ensembles:
            value = getattr(ensemble, field)
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"TMD lattice geometry requires finite positive {field}")
            equal = value == expected if field.startswith("L_") else np.isclose(value, expected, rtol=1e-12, atol=0)
            if not equal:
                raise ValueError(f"TMD inputs have incompatible lattice geometry: {field}={value} versus {expected}")


def validate_quasi(data: EnsembleData, observable: str = "tmdwf") -> tuple[np.ndarray, np.ndarray]:
    if data.dims != ["x"] or data.attrs.get("observable") != observable:
        raise ValueError(f"expected a per-b {observable} quasi artifact with dims=['x'] and attrs.observable")
    if data.attrs.get("renormalization_scheme") != "ratio":
        raise ValueError("TMD quasi input must declare renormalization_scheme='ratio'")
    if data.attrs.get("gfix") != "CG" or data.attrs.get("parton") != "quark":
        raise ValueError("CG quark TMD coefficients require attrs.gfix='CG' and attrs.parton='quark'")
    if data.attrs.get("distribution") != "quasi":
        raise ValueError("TMD input must declare attrs.distribution='quasi'; a matched light-cone input is invalid")
    numeric_attr(data, "bT_fm", nonnegative=True)
    numeric_attr(data, "momentum_gev")
    x = np.asarray(data.coords["x"], dtype=float)
    if len(x) < 3 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("TMD quasi x grid must have at least three finite, strictly increasing points")
    return x, samples(data)


def at_b(data: EnsembleData, b: float) -> np.ndarray:
    if data.dims != ["bT"]:
        raise ValueError("soft-function artifacts must have dims=['bT'] with coordinates in fm")
    declared = data.attrs.get("bT_unit")
    units = data.attrs.get("units", {})
    if isinstance(units, str):
        try:
            units = json.loads(units)
        except json.JSONDecodeError as exc:
            raise ValueError("bT artifact attrs.units must be a JSON mapping declaring bT in fm") from exc
    mapped = units.get("bT") if isinstance(units, dict) else None
    if ((declared is not None and declared != "fm") or (mapped is not None and mapped != "fm")
            or (declared != "fm" and mapped != "fm")):
        raise ValueError("bT artifact coordinates must explicitly declare fm units; conversion is not implicit")
    grid = np.asarray(data.coords["bT"], dtype=float)
    selected = np.flatnonzero(np.isclose(grid, b, rtol=0, atol=1e-10))
    if len(selected) != 1:
        raise ValueError(f"soft-function artifact must contain exactly one bT={b:g} fm; extrapolation is not implicit")
    return samples(data)[:, selected[0]]


def validate_perturbative_input(data: EnsembleData, params: dict, observable: str) -> None:
    if data.attrs.get("observable") != observable:
        raise ValueError(f"expected attrs.observable={observable!r}")
    if data.attrs.get("gfix") != "CG" or data.attrs.get("parton") != "quark":
        raise ValueError(f"{observable} must declare CG quark provenance")
    for name, expected in (("mu", params.get("mu", 2.0)), ("alpha_s", params.get("alpha_s", 0.293))):
        if not np.isclose(numeric_attr(data, name, nonnegative=name == "alpha_s"), expected, rtol=1e-12, atol=0):
            raise ValueError(f"{observable} {name} differs from this job; use consistently matched soft inputs")
    expected_order = params.get("order", "nlo")
    expected_resummation = "nll" if expected_order == "nll" else ""
    if data.attrs.get("order") != expected_order or data.attrs.get("resummation", "") != expected_resummation:
        raise ValueError(f"{observable} perturbative order differs from this job ({expected_order})")
    provenance = data.attrs.get("coefficient_provenance")
    if provenance is None and expected_order == "nll":
        raise ValueError(f"{observable} NLL input requires explicit coefficient_provenance")
    if provenance is not None:
        try:
            saved = json.loads(provenance)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{observable} coefficient_provenance must be a JSON object") from exc
        if saved != coefficient_metadata(**kernel_controls(params)):
            raise ValueError(f"{observable} coefficient/infrared prescription differs from this job")
    if observable == "cs_kernel" and data.attrs.get("cs_convention", "physical") != "physical":
        raise ValueError("cs_kernel input must be the physical hard-corrected CS kernel")
    if data.attrs.get("renormalization_scheme") != "ratio":
        raise ValueError(f"{observable} must declare the same ratio scheme as the quasi distribution")


def kernel_controls(params: dict) -> dict:
    """The same coefficient and infrared controls apply throughout the TMD DAG."""
    return {"order": params.get("order", "nlo"), "mu": params.get("mu", 2.0),
            "alpha_s": params.get("alpha_s", 0.293),
            "infrared_policy": params.get("infrared_policy", "error"),
            "freeze_scale_gev": params.get("freeze_scale_gev")}


def perturbative_attrs(params: dict) -> dict:
    order = params.get("order", "nlo")
    return {"order": order, "resummation": "nll" if order == "nll" else "",
            "kernel_id": f"tmd_cg_{order}",
            "coefficient_provenance": json.dumps(coefficient_metadata(**kernel_controls(params)), sort_keys=True)}


def validate_settings(params: dict) -> None:
    if params.get("resummation", ""):
        raise ValueError("TMD resummation is selected with order='nll'; leave resummation empty")
    coefficient_metadata(**kernel_controls(params))


def evolution_gamma(cs: EnsembleData, b: float, convention: str) -> np.ndarray:
    """Recover the uncorrected quasi ratio without changing any statistical replica."""
    gamma = at_b(cs, b)
    if convention == "physical":
        return gamma
    if convention != "quasi_ratio":
        raise ValueError("evolution_cs must be 'physical' or 'quasi_ratio'")
    try:
        correction = json.loads(cs.attrs["cs_hard_correction"])
        coords, values = np.asarray(correction["bT_fm"]), np.asarray(correction["values"])
        selected = np.flatnonzero(np.isclose(coords, b, rtol=0, atol=1e-10))
        if coords.ndim != 1 or values.shape != coords.shape or len(selected) != 1:
            raise ValueError("incompatible bT coordinates")
        delta = float(values[selected[0]])
        if not np.isfinite(delta):
            raise ValueError("nonfinite correction")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"quasi_ratio evolution needs the CS job's cs_hard_correction at bT={b:g}") from exc
    return gamma - delta


def replica_sdev(values, resample: str, attrs=None):
    """Statistical errors, including separate group factors for combined jackknives."""
    return sample_sdev(np.asarray(values), resample, attrs)


def inclusive_window(x, lower: float, upper: float):
    """Keep closed physical interval endpoints despite floating-point grid roundoff."""
    x = np.asarray(x)
    return (x >= lower - 1e-12) & (x <= upper + 1e-12)


def grouped_quasi(datasets: list[EnsembleData]) -> dict[float, dict[int, EnsembleData]]:
    grouped: dict[float, dict[int, EnsembleData]] = {}
    for data in datasets:
        validate_quasi(data)
        b = numeric_attr(data, "bT_fm", nonnegative=True)
        index = data.attrs.get("momentum_index")
        if isinstance(index, bool) or not isinstance(index, Integral) or index <= 0:
            raise ValueError("qTMDWF input requires a positive integer attrs.momentum_index")
        index = int(index)
        if index in grouped.setdefault(b, {}):
            raise ValueError(f"duplicate quasi input at bT={b:g}, momentum_index={index}")
        grouped[b][index] = data
    return grouped


def _output(datasets, values, b_values, observable, params) -> EnsembleData:
    attrs = {
        **merge_sample_lineage(*datasets),
        "observable": observable,
        "target_observable": "tmd",
        "gfix": "CG",
        "parton": "quark",
        **perturbative_attrs(params),
        "mu": float(params.get("mu", 2.0)),
        "alpha_s": float(params.get("alpha_s", 0.293)),
        "renormalization_scheme": "ratio",
        "units": '{"values":"dimensionless","bT":"fm"}',
        "bT_unit": "fm",
        "uncertainty": "statistical; systematic variations deferred",
    }
    data = EnsembleData(datasets[0].ensemble, datasets[0].resample, list(values), ["bT"],
                        {"bT": b_values}, attrs, observable)
    data.array = data.array.assign_coords(resample=datasets[0].array.coords["resample"])
    return data


def extract_cs(datasets: list[EnsembleData], params: dict) -> tuple[EnsembleData, dict]:
    validate_settings(params)
    datasets = list(align_sample_lineage(*datasets))
    merge_sample_lineage(*datasets)
    validate_lattice_geometry(*datasets)
    groups = grouped_quasi(datasets)
    pairs = params.get("momentum_pairs", [[8, 9], [9, 10]])
    low, high = params.get("x_window", [0.34, 0.66])
    if not (0 < low < high < 1):
        raise ValueError("CS x_window must lie strictly inside (0,1)")
    output, records, pair_samples, hard_corrections = [], [], [], []
    b_values = sorted(groups)
    for b in b_values:
        results, pair_records, selected_grids, corrections = [], [], [], []
        for first, second in pairs:
            if first not in groups[b] or second not in groups[b]:
                raise ValueError(f"CS bT={b:g} is missing requested momentum pair {first},{second}")
            d1, d2 = groups[b][first], groups[b][second]
            x, y1 = validate_quasi(d1)
            x2, y2 = validate_quasi(d2)
            if not np.array_equal(x, x2):
                raise ValueError("CS momentum pairs require the same physical x grid")
            selected = inclusive_window(x, low, high)
            if np.count_nonzero(selected) < 2:
                raise ValueError(f"CS x_window {low,high} selects fewer than two quasi-grid points")
            p1, p2 = numeric_attr(d1, "momentum_gev"), numeric_attr(d2, "momentum_gev")
            if p2 <= p1:
                raise ValueError("each CS momentum pair must have P2 > P1 > 0")
            kwargs = {"observable": "tmdwf", **kernel_controls(params)}
            h1 = hard_factor(x[selected], momentum_gev=p1, **kwargs)
            h2 = hard_factor(x[selected], momentum_gev=p2, **kwargs)
            positive(f"CS Hphi(P={p1:g})", h1)
            positive(f"CS Hphi(P={p2:g})", h2)
            for momentum, values in ((p1, y1[:, selected]), (p2, y2[:, selected])):
                bad_x = x[selected][np.any(~np.isfinite(values) | (values <= 0), axis=0)]
                positive(f"CS quasi(P={momentum:g},b={b:g}, first invalid x={bad_x[:5].tolist()})", values)
            gamma = np.log(y2[:, selected] * h1 / (y1[:, selected] * h2)) / np.log(p2 / p1)
            results.append(gamma)
            correction = np.log(h1 / h2) / np.log(p2 / p1)
            corrections.append(correction)
            selected_grids.append(x[selected])
            pair_records.append({"momentum_indices": [first, second], "momenta_gev": [p1, p2],
                                 "x": x[selected].tolist(),
                                 "hard_correction": correction.tolist(),
                                 "quasi_ratio_mean": sample_mean(gamma - correction, d1.resample, d1.attrs).tolist(),
                                 "gamma_mean": sample_mean(gamma, d1.resample, d1.attrs).tolist(),
                                 "gamma_sdev": replica_sdev(gamma, d1.resample, d1.attrs).tolist()})
        combined = np.concatenate(results, axis=1)
        pair_samples.append(list(zip(selected_grids, results)))
        output.append(combined.mean(axis=1))
        hard_corrections.append(float(np.concatenate(corrections).mean()))
        records.append({"bT_fm": b, "pairs": pair_records,
                        "mean_x_pair_spread": float(sample_mean(combined.std(axis=1),
                                                               datasets[0].resample, datasets[0].attrs))})
    data = _output(datasets, np.stack(output, axis=1), b_values, "cs_kernel", params)
    data.array.attrs.update({"cs_convention": "physical", "cs_hard_correction": json.dumps({
        "bT_fm": b_values, "values": hard_corrections,
        "definition": "physical gamma = quasi log-ratio gamma + mean(log(H1/H2)/log(P2/P1))"})})
    reduced = _output(datasets, np.asarray(data.values) - np.asarray(hard_corrections)[None, :],
                      b_values, "cs_kernel_quasi_ratio", params)
    reduced.array.attrs.update({"cs_convention": "quasi_ratio", "cs_hard_correction": data.attrs["cs_hard_correction"]})
    diagnostic_x = np.unique(np.concatenate([x for group in pair_samples for x, _ in group]))
    diagnostic_values = np.full((data.n_sample, len(b_values), len(pairs), len(diagnostic_x)), np.nan)
    for b_index, group in enumerate(pair_samples):
        for pair_index, (x, gamma) in enumerate(group):
            diagnostic_values[:, b_index, pair_index, np.searchsorted(diagnostic_x, x)] = gamma
    pair_data = EnsembleData(data.ensemble, data.resample, list(diagnostic_values), ["bT", "pair", "x"],
                            {"bT": b_values, "pair": [f"{a}_{b}" for a, b in pairs], "x": diagnostic_x},
                            {**data.attrs, "observable": "cs_kernel_pairs",
                             "units": '{"values":"dimensionless","bT":"fm","x":"dimensionless"}',
                             "missing_cells": "NaN denotes an x coordinate absent from a source pair grid"},
                            "cs_kernel_pairs")
    pair_data.array = pair_data.array.assign_coords(resample=data.array.coords["resample"])
    return data, {"x_window": [low, high], "momentum_pairs": pairs, "bT_records": records,
                  "aggregation": "arithmetic mean over selected x and momentum pairs within each replica",
                  "systematics_included": False, "_pair_samples": pair_data, "_reduced_samples": reduced,
                  "order": params.get("order", "nlo"),
                  "coefficient_provenance": coefficient_metadata(**kernel_controls(params)),
                  "cs_convention": "physical", "hard_correction": hard_corrections}


def endpoint_fit(x, values, windows, resample: str, attrs=None, *, priors=None, covariance=None):
    """Fit positive symmetric endpoint powers with a fixed correlated replica covariance."""
    authored_priors = validate_prior_overrides(priors)
    priors = {key: value for key, value in authored_priors.items() if key in {"c", "log(n)"}}
    selected = np.zeros(len(x), dtype=bool)
    for low, high in windows:
        if not 0 < low < high < 1:
            raise ValueError("endpoint fit windows must lie inside (0,1)")
        selected |= inclusive_window(x, low, high)
    if np.count_nonzero(selected) < 3:
        raise ValueError("endpoint_fit_windows select fewer than three quasi-grid points")
    xx, yy = x[selected], values[:, selected]
    # Fit dimensionless O(1) amplitudes so covariance classification, solver
    # tolerances and parameter bounds cannot depend on an overall input scale.
    amplitude_scale = float(np.max(np.abs(yy))) or 1.0
    yy = yy / amplitude_scale
    if covariance is None:
        cov = sample_covariance(yy, resample, attrs)
    else:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (len(x), len(x)) or not np.all(np.isfinite(covariance)):
            raise ValueError("endpoint covariance override must be a finite square matrix on the fit grid")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0):
            raise ValueError("endpoint covariance override must be symmetric")
        cov = covariance[np.ix_(selected, selected)] / amplitude_scale**2
    eigenvalues, vectors = np.linalg.eigh(cov)
    if covariance is not None and eigenvalues[0] < -1e-10 * max(eigenvalues[-1], np.finfo(float).tiny):
        raise ValueError("endpoint covariance override must be positive semidefinite")
    if eigenvalues[-1] <= np.finfo(float).eps**2 * len(xx):
        whitening = np.eye(len(xx))
        covariance_mode = "unweighted: numerically deterministic input"
    else:
        floor = eigenvalues[-1] * 1e-6
        whitening = (vectors / np.sqrt(np.maximum(eigenvalues, floor))).T
        covariance_mode = "correlated; eigenvalue floor 1e-6"
    logbasis = np.log(xx * (1 - xx))

    def prior_residual(p):
        physical = {"c": np.exp(p[0]) * amplitude_scale, "log(n)": p[1]}
        return np.asarray([(physical[name] - prior["mean"]) / prior["sdev"]
                           for name, prior in priors.items()])

    def fit(y, start):
        def residual(p):
            data_residual = whitening @ (np.exp(p[0] + np.exp(p[1]) * logbasis) - y)
            # The Gaussian for c is in physical linear-amplitude units. The
            # logarithmic optimizer coordinate only enforces the existing
            # positive endpoint model; it does not turn this into a log prior.
            return np.concatenate((data_residual, prior_residual(p))) if priors else data_residual
        result = least_squares(residual, start, bounds=([-30, -10], [30, 8]), max_nfev=1000)
        if not result.success or not np.all(np.isfinite(result.x)):
            raise ValueError(f"endpoint power fit failed: {result.message}; inspect reduced-WF fit windows")
        return result

    central_values = sample_mean(yy, resample, attrs)
    center = fit(central_values, [np.log(max(float(central_values.mean()), 1e-8)) + np.log(4), 0])
    fitted = [fit(y, center.x) for y in yy]
    parameters = np.exp(np.array([r.x for r in fitted]))
    parameters[:, 0] *= amplitude_scale
    parameter_covariance = sample_covariance(parameters, resample, attrs)
    dof = max(whitening.shape[0] + len(priors) - 2, 0)
    record = {"c": float(np.exp(center.x[0]) * amplitude_scale), "n": float(np.exp(center.x[1])),
              "chi2": float(np.sum(center.fun**2)), "dof": dof,
              "Q": float(chi2.sf(np.sum(center.fun**2), dof)) if dof else None,
              "covariance": covariance_mode, "fit_points": int(np.count_nonzero(selected)),
              "covariance_override": covariance is not None,
              "replica_parameters": {"names": ["c", "n"],
                                     "mean": sample_mean(parameters, resample, attrs).tolist(),
                                     "sdev": replica_sdev(parameters, resample, attrs).tolist(),
                                     "covariance": parameter_covariance.tolist()}}
    if priors:
        prior_chi2 = float(np.sum(prior_residual(center.x)**2))
        data_chi2 = float(np.sum(center.fun[:len(xx)]**2))
        data_dof = max(len(xx) - 2, 0)
        record.update({
            "data_chi2": data_chi2, "data_dof": data_dof,
            "data_Q": float(chi2.sf(data_chi2, data_dof)) if data_dof else None,
            "prior_chi2": prior_chi2,
            "Q_interpretation": "Bayesian augmented chi-square; dof=N_data+N_priors-2. "
                                "Approximate with regularized covariance and nonlinear positive parameters.",
            "effective_priors": priors,
            "conditioned_priors": {
                name: {key: value / amplitude_scale if name == "c" else value
                       for key, value in prior.items()}
                for name, prior in priors.items()
            },
            "amplitude_conditioning_scale": amplitude_scale,
            "prior_application": "same original physical priors on the center and every replica",
        })
    if authored_priors:
        record["inactive_prior_overrides"] = sorted(set(authored_priors) - set(priors))
    return parameters, record


@lru_cache(maxsize=64)
def quadrature(cut: float, lower: float, upper: float, order: int, p: float, mu: float, alpha: float,
               perturbative_order: str = "nlo", infrared_policy: str = "error", freeze_scale_gev: float | None = None):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    xx, ww = [], []
    for a, b in zip((cut, lower, upper), (lower, upper, 1 - cut)):
        xx.extend((a + b) / 2 + (b - a) / 2 * nodes)
        ww.extend((b - a) / 2 * weights)
    x, w = np.asarray(xx), np.asarray(ww)
    h = form_factor_kernel(x[:, None], x[None, :], momentum_gev=p, mu=mu, alpha_s=alpha,
                           order=perturbative_order, infrared_policy=infrared_policy, freeze_scale_gev=freeze_scale_gev)
    return x, h * np.outer(w, w), float(np.mean(h < 0))


def correlated_average(estimates, resample: str, attrs=None):
    """Average aligned replicas equally; diagnose momentum consistency with a separate GLS fit."""
    values = np.asarray(estimates)
    weights = np.full(len(values), 1 / len(values))
    result = weights @ values
    if len(values) == 1:
        return result, {"weights": [1.0], "momentum_consistency": {
            "chi2": None, "dof": 0, "Q": None, "warnings": [], "fit_weights": [1.0],
            "fitted_mean": float(sample_mean(result, resample, attrs)),
            "covariance_rank": 1, "covariance_rcond": 1e-6}}
    cov = sample_covariance(values.T, resample, attrs)
    eigenvalues, vectors = np.linalg.eigh(cov)
    if eigenvalues[-1] <= np.finfo(float).eps**2 * max(1.0, np.max(np.abs(values)))**2:
        fit_weights = weights
        rank = 0
        inverse = None
    else:
        floor = eigenvalues[-1] * 1e-6
        rank = int(np.count_nonzero(eigenvalues > floor))
        inverse = (vectors / np.maximum(eigenvalues, floor)) @ vectors.T
        fit_weights = inverse @ np.ones(len(values))
        fit_weights /= fit_weights.sum()
    centers = sample_mean(values.T, resample, attrs)
    fitted_center = float(fit_weights @ centers)
    residual = centers - fitted_center
    statistic = float(residual @ inverse @ residual) if inverse is not None else None
    dof = len(values) - 1
    probability = float(chi2.sf(statistic, dof)) if statistic is not None else None
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(centers))))
    warnings = []
    if probability is not None and probability < 0.05:
        warnings.append("Momentum estimates are inconsistent with a common constant at Q < 0.05. "
                        "Momentum-dependent power corrections are not included in the statistical error.")
    if inverse is None and np.ptp(centers) > tolerance:
        warnings.append("Numerically zero replica covariance prevents a momentum-consistency statistic.")
    consistency = {"chi2": statistic, "dof": dof, "Q": probability,
                   "warnings": warnings, "fit_weights": fit_weights.tolist(), "fitted_mean": fitted_center,
                   "covariance_rank": rank, "covariance_rcond": 1e-6,
                   "method": "diagnostic common-constant GLS fit; Q is approximate; does not set the output"}
    return result, {"weights": weights.tolist(), "momentum_consistency": consistency}


def extract_intrinsic(quasi: list[EnsembleData], form_factors: list[EnsembleData],
                      cs: EnsembleData, params: dict) -> tuple[EnsembleData, dict]:
    validate_settings(params)
    all_data = list(align_sample_lineage(*quasi, *form_factors, cs))
    quasi, form_factors, cs = all_data[:len(quasi)], all_data[len(quasi):-1], all_data[-1]
    merge_sample_lineage(*all_data)
    validate_lattice_geometry(*all_data)
    validate_perturbative_input(cs, params, "cs_kernel")
    groups = grouped_quasi(quasi)
    ff_by_b = {}
    for ff in form_factors:
        if ff.attrs.get("observable") != "form_factor":
            raise ValueError("intrinsic-soft input requires normalized attrs.observable='form_factor'")
        if ff.attrs.get("gfix") != "CG" or ff.attrs.get("parton") != "quark":
            raise ValueError("form_factor must declare CG quark provenance")
        if ff.dims == []:
            b = numeric_attr(ff, "bT_fm", nonnegative=True)
            if b in ff_by_b:
                raise ValueError(f"duplicate form-factor input at bT={b:g}")
            ff_by_b[b] = (ff, samples(ff))
        elif ff.dims == ["bT"]:
            for b in ff.coords["bT"]:
                if b in ff_by_b:
                    raise ValueError(f"duplicate form-factor input at bT={b:g}")
                ff_by_b[float(b)] = (ff, at_b(ff, float(b)))
        else:
            raise ValueError("form_factor inputs must be scalar per-b or carry a bT grid in fm")
    if set(groups) != set(ff_by_b):
        raise ValueError("quasi and form_factor inputs must cover exactly the same bT grid")
    cut = float(params.get("integration_cut", 0.05))
    lower, upper = params.get("endpoint_region", [0.3, 0.7])
    if not (0 < cut < lower < upper < 1 - cut < 1):
        raise ValueError("require 0 < integration_cut < endpoint_region < 1-integration_cut < 1")
    windows = params.get("endpoint_fit_windows", [[0.3, 0.45], [0.55, 0.7]])
    order = params.get("quadrature_order", 24)
    if isinstance(order, bool) or not isinstance(order, int) or order < 8:
        raise ValueError("quadrature_order must be an integer >= 8")
    mu, alpha = float(params.get("mu", 2.0)), float(params.get("alpha_s", 0.293))
    tolerance = float(params.get("quadrature_rtol", 0.005))
    evolution = params.get("evolution_cs", "physical")
    sudakov_momentum = params.get("sudakov_momentum", "form_factor")
    if evolution not in {"physical", "quasi_ratio"}:
        raise ValueError("evolution_cs must be 'physical' or 'quasi_ratio'")
    if sudakov_momentum not in {"form_factor", "quasi"}:
        raise ValueError("sudakov_momentum must be 'form_factor' or 'quasi'")
    fit_points = params.get("endpoint_fit_points_per_window")
    central_points = params.get("endpoint_central_points")
    covariance_mode = params.get("endpoint_covariance", "full")
    for name, count in (("endpoint_fit_points_per_window", fit_points), ("endpoint_central_points", central_points)):
        if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 4):
            raise ValueError(f"{name} must be an integer >= 4")
    if covariance_mode not in {"full", "reference_cs_diagonal"}:
        raise ValueError("endpoint_covariance must be 'full' or 'reference_cs_diagonal'")
    output, records = [], []
    for b in sorted(groups):
        ff, numerator = ff_by_b[b]
        pff = numeric_attr(ff, "momentum_gev")
        gamma = evolution_gamma(cs, b, evolution)
        estimates, contributions = [], []
        for index, data in sorted(groups[b].items()):
            x, values = validate_quasi(data)
            center_mask = inclusive_window(x, lower, upper)
            # Include adjacent knots so interpolation covers the entire central interval.
            indices = np.flatnonzero(center_mask)
            if len(indices) < 3 or x[0] > lower or x[-1] < upper:
                raise ValueError("quasi x grid must cover endpoint_region with at least three points")
            start, stop = max(indices[0] - 1, 0), min(indices[-1] + 2, len(x))
            required = np.zeros(len(x), dtype=bool)
            required[start:stop] = True
            for fit_low, fit_high in windows:
                required |= inclusive_window(x, fit_low, fit_high)
            xx, yy = x[required], values[:, required]
            p = numeric_attr(data, "momentum_gev")
            h = hard_factor(xx, observable="tmdwf", momentum_gev=p, **kernel_controls(params))
            positive(f"intrinsic-soft Hphi(P={p:g},b={b:g})", h)
            evolution_factor = rapidity_factor(gamma, momentum_gev=p, p_ref_gev=pff)
            reduced = yy / h * evolution_factor[:, None]

            def interpolated_beam(grid):
                # Match the reference sequence: interpolate the raw quasi data,
                # then evaluate the coefficient and evolution at the new x.
                raw = interp1d(x, values, kind="cubic", axis=1, bounds_error=True)(grid)
                coefficient = hard_factor(grid, observable="tmdwf", momentum_gev=p, **kernel_controls(params))
                positive(f"interpolated intrinsic-soft Hphi(P={p:g},b={b:g})", coefficient)
                return raw, coefficient, raw / coefficient * evolution_factor[:, None]

            fit_x, fit_quasi, fit_h, fit_reduced = xx, yy, h, reduced
            if fit_points is not None:
                fit_x = np.unique(np.concatenate([np.linspace(a, z, fit_points) for a, z in windows]))
                fit_quasi, fit_h, fit_reduced = interpolated_beam(fit_x)
            working_covariance = None
            if covariance_mode == "reference_cs_diagonal":
                # This is an explicit historical likelihood metric only. Fit
                # every true aligned replica, retaining all output covariance.
                logarithm = np.log(pff / p)
                gamma_center = float(sample_mean(gamma, data.resample, data.attrs))
                diagonal = np.exp(gamma_center * logarithm) / fit_h
                working_covariance = sample_covariance(fit_quasi, data.resample, data.attrs)
                working_covariance *= np.outer(diagonal, diagonal)
                beam_center = sample_mean(fit_quasi, data.resample, data.attrs) * diagonal
                gamma_error = float(replica_sdev(gamma, data.resample, data.attrs))
                working_covariance += np.diag((beam_center * logarithm * gamma_error)**2)
            powers, fit_record = endpoint_fit(fit_x, fit_reduced, windows, data.resample, data.attrs,
                                              priors=params.get("priors"), covariance=working_covariance)
            fit_record["working_covariance"] = covariance_mode
            fit_record["replica_propagation"] = "all original aligned replicas; no independent CS redraws"
            center_x, center_reduced = xx, reduced
            if central_points is not None:
                center_x = np.linspace(lower, upper, central_points)
                _, _, center_reduced = interpolated_beam(center_x)

            kernel_p = pff if sudakov_momentum == "form_factor" else p

            def integral(n):
                grid, matrix, negative_fraction = quadrature(
                    cut, lower, upper, n, kernel_p, mu, alpha, params.get("order", "nlo"),
                    params.get("infrared_policy", "error"), params.get("freeze_scale_gev"))
                tails = (grid < lower) | (grid > upper)
                if central_points is None:
                    beam = np.array([np.interp(grid, center_x, y) for y in center_reduced])
                else:
                    beam = np.empty((len(center_reduced), len(grid)))
                    beam[:, ~tails] = interp1d(center_x, center_reduced, kind="cubic", axis=1,
                                               bounds_error=True)(grid[~tails])
                beam[:, tails] = powers[:, 0, None] * (grid[tails] * (1 - grid[tails]))**powers[:, 1, None]
                return np.einsum("si,ij,sj->s", beam, matrix, beam, optimize=True), negative_fraction

            coarse, _ = integral(order)
            denominator, negative_fraction = integral(2 * order)
            positive(f"intrinsic-soft denominator(P={p:g},b={b:g})", denominator)
            relative = np.abs(denominator - coarse) / np.maximum(np.abs(denominator), 1e-12)
            if np.max(relative) > tolerance:
                raise ValueError(f"soft quadrature failed convergence at bT={b:g}, P={p:g}: "
                                 f"max relative change {np.max(relative):g} > {tolerance:g}; "
                                 "increase quadrature_order")
            stat = float(replica_sdev(denominator, data.resample, data.attrs))
            absolute_change = float(np.max(np.abs(denominator - coarse)))
            sigma_ratio = absolute_change / stat if stat > 1e-12 * np.max(np.abs(denominator)) else None
            if sigma_ratio is not None and sigma_ratio > 0.1:
                raise ValueError(f"soft quadrature change at bT={b:g}, P={p:g} is {sigma_ratio:g} statistical "
                                 "errors (>0.1); increase quadrature_order")
            scale_record = {}
            if params.get("order", "nlo") == "nll":
                from lamet_agent.kernels.tmd.cg_nll import scale_diagnostics
                grid, _, _ = quadrature(cut, lower, upper, 2 * order, kernel_p, mu, alpha, "nll",
                                         params.get("infrared_policy", "error"), params.get("freeze_scale_gev"))
                scales = 2 * kernel_p * np.sqrt(grid[:, None] * grid[None, :])
                scale_record = scale_diagnostics(scales, **{key: value for key, value in kernel_controls(params).items()
                                                           if key != "order"})
                scale_record["interpretation"] = "one Sudakov factor; antiquark factor has the reflected node grid"
            estimates.append(numerator / denominator)
            contributions.append({"momentum_index": index, "momentum_gev": p, "ff_momentum_gev": pff,
                                  "sudakov_momentum_gev": kernel_p, "evolution_cs": evolution,
                                  "sudakov_scales": scale_record,
                                  "endpoint_fit": fit_record,
                                  "denominator_mean": float(sample_mean(denominator, data.resample, data.attrs)),
                                  "soft_mean": float(sample_mean(estimates[-1], data.resample, data.attrs)),
                                  "quadrature_max_relative_change": float(relative.max()),
                                  "quadrature_change_over_stat_error": sigma_ratio,
                                  "negative_HF_node_fraction": negative_fraction})
        result, average_record = correlated_average(estimates, quasi[0].resample, quasi[0].attrs)
        if not np.all(np.isfinite(result)):
            raise ValueError(f"intrinsic soft function has nonfinite samples at bT={b:g}")
        output.append(result)
        records.append({"bT_fm": b, "momenta": contributions, "correlated_average": average_record,
                        "nonpositive_soft_replicas": int(np.count_nonzero(result <= 0))})
    data = _output(all_data, np.stack(output, axis=1), sorted(groups), "intrinsic_soft", params)
    data.array.attrs.update({"evolution_cs": evolution, "sudakov_momentum": sudakov_momentum,
                             "endpoint_covariance": covariance_mode})
    for name in ("endpoint_fit_points_per_window", "endpoint_central_points"):
        if params.get(name) is not None:
            data.array.attrs[name] = params[name]
    infrared = infrared_diagnostics(params)
    if infrared:
        data.array.attrs["infrared_diagnostics"] = json.dumps(infrared, sort_keys=True)
    return data, {"integration_cut": cut, "endpoint_region": [lower, upper],
                  "endpoint_fit_windows": windows, "quadrature_order": 2 * order,
                  "order": params.get("order", "nlo"), "evolution_cs": evolution,
                  "sudakov_momentum": sudakov_momentum, "endpoint_covariance": covariance_mode,
                  "endpoint_fit_points_per_window": fit_points, "endpoint_central_points": central_points,
                  "infrared_diagnostics": infrared,
                  "coefficient_provenance": coefficient_metadata(**kernel_controls(params)),
                  "bT_records": records,
                  "aggregation": "equal-weight arithmetic mean over momenta within every aligned replica; "
                                 "the full momentum covariance is retained in the output statistical error. "
                                 "The common-constant GLS fit is a separate consistency diagnostic",
                  "systematics_included": False}


def infrared_diagnostics(params: dict) -> dict:
    """Summarize the historical pole-crossing quadrature without hiding its warnings."""
    if params.get("order", "nlo") != "nll" or params.get("infrared_policy", "error") != "reference_quad":
        return {}
    from lamet_agent.kernels.tmd.cg_nll import reference_quad_diagnostics
    diagnostics = reference_quad_diagnostics()
    records = diagnostics.pop("evaluations", [])
    diagnostics["scope"] = "cumulative coefficient evaluations and cache entries in this process"
    diagnostics["first_problematic_evaluations"] = [
        record for record in records if record.get("crosses_coupling_pole") or record.get("warnings")][:5]
    errors = [record.get("quad_estimated_absolute_error", 0) for record in records]
    diagnostics["maximum_quad_estimated_absolute_error"] = max(errors, default=0)
    return diagnostics
