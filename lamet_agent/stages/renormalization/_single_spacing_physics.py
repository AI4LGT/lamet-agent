"""Single-spacing mass subtraction of arXiv:2312.10829, Sec. II.

The fitted long-distance mass is a subtraction convention, not an independent
measurement of the UV divergence. Only delta_m + m0 enters the hybrid ratio.
All lengths are fm and all fitted masses are GeV.
"""

from __future__ import annotations

import json
import hashlib
from collections.abc import Callable, Sequence

import gvar as gv
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2

from lamet_agent.data import EnsembleData, align_sample_lineage, merge_sample_lineage
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM as HC
from lamet_agent.stages.renormalization.physics import physical_z_coordinates

STRATEGY = "single_spacing_self_renormalization"
PARAMETERS = ["delta_m_gev", "m0_gev", "total_mass_gev", "B", "c"]


def _reference_digest(data: EnsembleData) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(data.coords["z"], dtype="<f8").tobytes())
    digest.update(np.asarray(np.real(data.values), dtype="<f8").tobytes())
    return digest.hexdigest()


def _spacing(data: EnsembleData) -> float:
    a = float(data.ensemble.a_s)
    if not np.isfinite(a) or a <= 0:
        raise ValueError("single-spacing inputs need a finite positive ensemble.a_s")
    return a


def _channel(data: EnsembleData, *, reference: bool = False) -> None:
    attrs = data.attrs
    if attrs.get("observable") not in ({"pdf"} if reference else {"pdf", "gpd"}):
        raise ValueError("reference must be a collinear PDF; target must be a collinear PDF or GPD")
    if attrs.get("gfix") != "gi" or attrs.get("operator") != "gt":
        raise ValueError("single-spacing subtraction supports gauge-invariant gamma-t quark operators")
    if attrs.get("parton", "quark") != "quark":
        raise ValueError("single-spacing subtraction supports quark operators only")
    if reference:
        p = attrs.get("momentum_gev")
        if isinstance(p, bool) or not isinstance(p, (int, float)) or not np.isfinite(p) or abs(p) > 1e-12:
            raise ValueError("reference must explicitly have momentum_gev=0")
        if data.dims != ["z"]:
            raise ValueError("reference must have only the z dimension at one lattice spacing")
    _spacing(data)


def _replicas(data: EnsembleData) -> None:
    if data.resample not in {"bootstrap", "jackknife"} or data.n_sample < 3:
        raise ValueError("single-spacing fits require at least three bootstrap or jackknife replicas")
    if not np.all(np.isfinite(data.values)):
        raise ValueError("single-spacing inputs must be finite")


def _whitener(values: np.ndarray, source: EnsembleData, svdcut: float):
    data = EnsembleData(
        source.ensemble,
        source.resample,
        list(values),
        ["point"],
        {"point": list(range(values.shape[1]))},
        attrs=source.attrs,
    )
    average = data.average("covariance")
    cov = np.asarray(gv.evalcov(average), dtype=float)
    eig, vec = np.linalg.eigh(cov)
    if not 0 < svdcut < 1 or not np.isfinite(svdcut):
        raise ValueError("svdcut must be between zero and one")
    if eig[-1] <= 0:
        raise ValueError("fit covariance has no statistical variance")
    floor = svdcut * eig[-1]
    rank = int(np.count_nonzero(eig > floor))
    if rank < 2:
        raise ValueError("fit covariance needs at least two independent statistical directions")
    return (vec / np.sqrt(np.maximum(eig, floor))).T, np.asarray(gv.mean(average)), rank


def _quality(residual, n_points: int, rank: int) -> dict:
    value = float(np.dot(residual, residual))
    dof = n_points - 2
    return {
        "chi2": value,
        "dof": dof,
        "chi2_dof": value / dof if dof else None,
        "Q": float(chi2.sf(value, dof)) if dof else None,
        "covariance_rank": rank,
        "n_points": n_points,
        "svd_regularized": rank < n_points,
    }


def _interpolate(z: np.ndarray, values: np.ndarray, points: Sequence[float]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if (
        points.ndim != 1
        or not np.all(np.isfinite(points))
        or np.any(np.diff(points) <= 0)
        or points[0] < z[0] - 1e-12
        or points[-1] > z[-1] + 1e-12
    ):
        raise ValueError("fit points must be increasing and inside the reference grid; no extrapolation")
    return np.asarray([np.interp(points, z, sample) for sample in values])


def fit_single_spacing(
    reference: EnsembleData,
    *,
    long_distance_window_fm: Sequence[float],
    short_distance_points_fm: Sequence[float],
    kernel: Callable,
    kernel_id: str,
    mu: float,
    kernel_parameters: dict | None = None,
    svdcut: float = 1e-10,
    scan_centers_fm: Sequence[float] = (),
    scan_half_width_fm: float = 0.02,
) -> tuple[EnsembleData, dict]:
    """Fit B exp(-delta_m z), then log(C0/h)-delta_m z = m0 z+c.

    Correlated fits use a fixed covariance across replicas. The short-distance
    linear estimator is applied to every replica, preserving cancellation of
    the long-distance subtraction convention and its statistical covariance.
    A scan is diagnostic; the authored fit points select the result explicitly.
    """
    reference = physical_z_coordinates(reference)
    _channel(reference, reference=True)
    _replicas(reference)
    z = np.asarray(reference.coords["z"], dtype=float)
    if not np.all(np.isfinite(z)) or np.any(np.diff(z) <= 0) or np.any(z < 0):
        raise ValueError("reference z grid must be finite, nonnegative and strictly increasing")
    values = np.real(np.asarray(reference.values))
    window = np.asarray(long_distance_window_fm, dtype=float)
    if window.shape != (2,) or not np.all(np.isfinite(window)) or not 0 < window[0] < window[1]:
        raise ValueError("long_distance_window_fm must be [positive minimum, maximum]")
    if window[0] < z[0] - 1e-12 or window[1] > z[-1] + 1e-12:
        raise ValueError("long-distance window must be covered by reference data")
    mask = (z >= window[0] - 1e-12) & (z <= window[1] + 1e-12)
    if mask.sum() < 3 or np.any(values[:, mask] <= 0):
        raise ValueError("long-distance fit needs at least three positive real reference points in every replica")
    x = z[mask] / HC
    y = values[:, mask]
    whitening, center, rank = _whitener(y, reference, svdcut)
    initial = np.polyfit(x, np.log(center), 1)

    def solve(sample, start):
        fit = least_squares(
            lambda p: whitening @ (np.exp(p[0] - p[1] * x) - sample),
            start,
            max_nfev=2000,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
        if not fit.success or not np.all(np.isfinite(fit.x)):
            raise ValueError("long-distance exponential fit did not converge")
        return fit

    central = solve(center, [initial[1], -initial[0]])
    long_pars = np.asarray([solve(sample, central.x).x for sample in y])
    delta = long_pars[:, 1]
    options = dict(kernel_parameters or {})
    if "mu" in options or "z_fm" in options:
        raise ValueError("kernel_parameters must not override the scale or coordinates")
    if not np.isfinite(mu) or mu <= 0:
        raise ValueError("mu must be finite and positive")

    def short_fit(points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 1 or len(points) < 3 or np.any(points <= 0):
            raise ValueError("short_distance_points_fm needs at least three positive coordinates")
        h = _interpolate(z, values, points)
        if np.any(h <= 0):
            raise ValueError("short-distance reference real parts must be positive in every replica")
        c0 = np.broadcast_to(np.asarray(kernel(points, mu=mu, **options), dtype=float), points.shape)
        if np.any(~np.isfinite(c0)) or np.any(c0 <= 0):
            raise ValueError("Wilson coefficient must be finite and positive")
        # Determine the covariance from log(C0/h), before the arbitrary linear
        # mass shift. This makes the total mass strictly convention-independent.
        log_ratio = np.log(c0 / h)
        w, mean, covariance_rank = _whitener(log_ratio, reference, svdcut)
        design = np.column_stack((points / HC, np.ones(len(points))))
        weighted = w @ design
        if np.linalg.matrix_rank(weighted) < 2:
            raise ValueError("short-distance slope and intercept are not identifiable")
        estimator = np.linalg.pinv(weighted) @ w
        total_and_c = log_ratio @ estimator.T
        total, intercept = total_and_c.T
        m0 = total - delta
        mean_pars = estimator @ mean
        quality = _quality(w @ (design @ mean_pars - mean), len(points), covariance_rank)
        return m0, total, intercept, quality

    m0, total, intercept, short_quality = short_fit(short_distance_points_fm)
    masses = np.column_stack((delta, m0, total, np.exp(long_pars[:, 0]), intercept))
    attrs = {
        **reference.attrs,
        "strategy": STRATEGY,
        "source_arxiv": "2312.10829",
        "formula": "single_spacing_hybrid_ratio",
        "kernel_id": kernel_id,
        "scale_gev": float(mu),
        "kernel_parameters": json.dumps(options, sort_keys=True),
        "mass_unit": "GeV",
        "reference_projection": "real",
        "interpolation": "piecewise_linear_in_h",
        "reference_ensemble_id": reference.ensemble.id,
        "reference_digest": _reference_digest(reference),
        "long_distance_window_fm": json.dumps(list(long_distance_window_fm)),
        "short_distance_points_fm": json.dumps(list(short_distance_points_fm)),
    }
    output = EnsembleData(
        reference.ensemble,
        reference.resample,
        list(masses),
        ["parameter"],
        {"parameter": PARAMETERS},
        attrs=attrs,
        name="single_spacing_masses",
    )
    estimates = {name: {"mean": float(m), "sdev": float(s)} for name, m, s in zip(PARAMETERS, output.mean, output.sdev)}
    scan = []
    if not np.isfinite(scan_half_width_fm) or scan_half_width_fm <= 0:
        raise ValueError("scan_half_width_fm must be finite and positive")
    for coordinate in scan_centers_fm:
        points = [float(coordinate) - scan_half_width_fm, float(coordinate), float(coordinate) + scan_half_width_fm]
        sm0, stotal, _, quality = short_fit(points)
        samples = EnsembleData(
            reference.ensemble,
            reference.resample,
            list(np.column_stack((sm0, stotal))),
            ["parameter"],
            {"parameter": ["m0", "total"]},
            attrs=reference.attrs,
        )
        scan.append(
            {
                "center_fm": float(coordinate),
                "points_fm": points,
                "m0_gev": float(samples.mean[0]),
                "m0_sdev_gev": float(samples.sdev[0]),
                "total_mass_gev": float(samples.mean[1]),
                "total_mass_sdev_gev": float(samples.sdev[1]),
                "fit_quality": quality,
            }
        )
    return output, {
        "formula": attrs["formula"],
        "source_arxiv": "2312.10829",
        "masses": estimates,
        "fit_quality": {"reference": _quality(central.fun, len(x), rank), "m0_matching": short_quality},
        "reference_sample_count": reference.n_sample,
        "kernel_id": kernel_id,
        "kernel_parameters": options,
        "scale_gev": mu,
        "long_distance_window_fm": list(long_distance_window_fm),
        "short_distance_points_fm": list(short_distance_points_fm),
        "lattice_spacing_range_fm": [_spacing(reference)] * 2,
        "scan": scan,
        "plateau_selection": "authored short_distance_points_fm; no automatic selection",
    }


def apply_single_spacing(
    target: EnsembleData, reference: EnsembleData, masses: EnsembleData, *, zs_fm: float, normalization: bool = False
) -> EnsembleData:
    """Apply Eq. (3), using |z| and preserving nonforward local form factors."""
    target, reference = physical_z_coordinates(target), physical_z_coordinates(reference)
    _channel(target)
    _channel(reference, reference=True)
    for data in (target, reference, masses):
        _replicas(data)
    if masses.attrs.get("strategy") != STRATEGY or masses.dims != ["parameter"]:
        raise ValueError("masses must come from a single-spacing fit")
    if list(masses.coords["parameter"]) != PARAMETERS:
        raise ValueError("invalid single-spacing mass parameter coordinates")
    if any(not np.isclose(_spacing(data), _spacing(reference), rtol=0, atol=1e-12) for data in (target, masses)):
        raise ValueError("target, reference and masses must share one lattice spacing")
    if masses.attrs.get("reference_ensemble_id") != reference.ensemble.id:
        raise ValueError("mass fit and denominator must use the same reference ensemble")
    if masses.attrs.get("reference_digest") != _reference_digest(reference):
        raise ValueError("mass fit and denominator must use the same reference samples and z grid")
    if target.ensemble.id != reference.ensemble.id:
        raise ValueError("target and reference must use the same lattice ensemble and operator regularization")
    if "z" not in target.dims or any(dim not in {"z", "t", "xi", "Q2"} for dim in target.dims):
        raise ValueError("target dimensions must be z with optional t, xi or Q2")
    if normalization and target.attrs["observable"] == "gpd":
        raise ValueError("GPD normalization must be false to preserve the local form factor")
    if not np.isfinite(zs_fm) or zs_fm <= 0:
        raise ValueError("zs_fm must be finite and positive")
    # Check the original lineage before expanding independent jackknife groups.
    merge_sample_lineage(reference, masses)
    target, reference, masses = align_sample_lineage(target, reference, masses)
    lineage = merge_sample_lineage(target, reference, masses)
    zref = np.asarray(reference.coords["z"], dtype=float)
    z = np.asarray(target.coords["z"], dtype=float)
    if np.any(np.diff(zref) <= 0) or np.any(zref < 0) or not np.all(np.isfinite(zref)):
        raise ValueError("reference grid must be finite, nonnegative and strictly increasing")
    if not np.all(np.isfinite(z)) or len(np.unique(z)) != len(z):
        raise ValueError("target z coordinates must be finite and unique")
    switch = np.flatnonzero(np.isclose(zref, zs_fm, rtol=0, atol=1e-12))
    if len(switch) != 1:
        raise ValueError("zs_fm must lie on the reference grid")
    # No interpolation is used in applying the short-distance ratio.
    requested = np.minimum(np.abs(z), zs_fm)
    indices = []
    for point in requested:
        found = np.flatnonzero(np.isclose(zref, point, rtol=0, atol=1e-12))
        if len(found) != 1:
            raise ValueError("reference must cover each short-distance target |z| exactly")
        indices.append(int(found[0]))
    denominator = np.real(reference.values)[:, indices]
    if np.any(denominator <= 0):
        raise ValueError("hybrid denominator must have positive real parts in every replica")
    total = np.asarray(masses.values)[:, PARAMETERS.index("total_mass_gev")]
    factor = np.exp(total[:, None] * np.maximum(np.abs(z) - zs_fm, 0) / HC) / denominator
    shape = [target.n_sample] + [1] * len(target.dims)
    shape[1 + target.dims.index("z")] = len(z)
    values = np.asarray(target.values) * factor.reshape(shape)
    if normalization:
        origins = np.flatnonzero(np.isclose(z, 0, atol=1e-12, rtol=0))
        if len(origins) != 1:
            raise ValueError("PDF normalization requires exactly one origin")
        origin = np.take(values, [origins[0]], axis=1 + target.dims.index("z"))
        if np.any(np.abs(origin) == 0):
            raise ValueError("cannot normalize a zero local matrix element")
        values = values / origin
    if not np.all(np.isfinite(values)):
        raise ValueError("hybrid mass subtraction produced nonfinite values")
    attrs = {
        **target.attrs,
        **lineage,
        "strategy": STRATEGY,
        "renormalization_scheme": "hybrid",
        "hybrid_switch_coord_fm": zs_fm,
        "zs_fm": zs_fm,
        "coord_unit": "fm",
        "source_arxiv": "2312.10829",
        "renormalization_kernel_id": masses.attrs["kernel_id"],
        "renormalization_kernel_parameters": masses.attrs["kernel_parameters"],
        "scale_gev": masses.attrs["scale_gev"],
        "mass_unit": "GeV",
    }
    return EnsembleData(
        target.ensemble,
        target.resample,
        list(values),
        target.dims,
        target.coords,
        attrs=attrs,
        name="renormalized_matrix_element",
    )
