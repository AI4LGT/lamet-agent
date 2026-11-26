"""Single-spacing residual Z_O extraction after Wilson-loop subtraction."""

import json

import numpy as np
from scipy.stats import chi2

from lamet_agent.data import (
    EnsembleData,
    align_sample_lineage,
    merge_sample_lineage,
    sample_covariance,
    sample_mean,
    sample_sdev,
)
from lamet_agent.kernels.tmd.zo import coefficient
from lamet_agent.stages.renormalization.physics import load_data


def _check(data, *, reference=False):
    if data.resample not in ("bootstrap", "jackknife") or data.n_sample < 3:
        raise ValueError("Z_O requires at least three bootstrap/jackknife samples")
    if str(data.attrs.get("gfix", "")).lower() != "gi":
        raise ValueError("Z_O short-distance conversion is for gauge-invariant (GI) operators")
    if data.attrs.get("observable") != "tmdpdf":
        raise ValueError("this Z_O prescription currently supports tmdpdf only")
    if data.attrs.get("wilson_loop_subtracted") != "true":
        raise ValueError("input must declare wilson_loop_subtracted='true'; raw staples are not accepted")
    if data.attrs.get("renormalization_scheme") == "msbar":
        raise ValueError("input is already MSbar renormalized")
    if data.attrs.get("reference_normalization") != "local_vector_charge":
        raise ValueError("reference and target must share reference_normalization='local_vector_charge'")
    if data.ensemble is None or not np.isfinite(data.ensemble.a_s) or data.ensemble.a_s <= 0:
        raise ValueError("input requires a lattice ensemble with positive a_s")
    for key in ("operator", "renormalization_family"):
        if not isinstance(data.attrs.get(key), str) or not data.attrs[key]:
            raise ValueError(f"input requires explicit {key}")
    if data.attrs.get("coord_unit") != "fm":
        raise ValueError("Z_O inputs require longitudinal coordinates in fm")
    if set(data.dims) not in ({"z"}, {"z", "bT"}):
        raise ValueError("Z_O matrix elements require z and optional bT axes")
    if "bT" in data.dims:
        if data.attrs.get("bT_unit") != "fm":
            raise ValueError("bT coordinates must have bT_unit='fm'")
        b = np.asarray(data.coords["bT"], float)
    else:
        b = np.asarray([data.attrs.get("bT_fm", np.nan)], float)
    if np.any(~np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("transverse separations must be positive and finite")
    for dim in data.dims:
        grid = np.asarray(data.coords[dim], float)
        if np.any(~np.isfinite(grid)) or len(np.unique(grid)) != len(grid):
            raise ValueError("coordinates must be finite and unique")
    if not np.all(np.isfinite(data.values)):
        raise ValueError("matrix elements must be finite")
    if reference:
        if data.attrs.get("momentum_gev") != 0 or data.attrs["operator"].lower() != "gt":
            raise ValueError("reference must be the zero-momentum gamma-t unpolarized TMDPDF")


def _new(source, values, dims, coords, attrs, name):
    result = EnsembleData(source.ensemble, source.resample, list(values), dims, coords, attrs, name=name)
    result.array = result.array.assign_coords(resample=source.array.coords["resample"])
    return result


def extract(reference, *, points_fm, mu=2.0, alpha_s=0.293, n_f=3, order="nlo_rgr", min_scale_gev=0.7):
    """Equal-weight average of authored (z,b) points, not a covariance-weighted fit."""
    _check(reference, reference=True)
    points = np.asarray(points_fm, float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
        raise ValueError("points_fm requires at least three [z_fm,b_fm] pairs")
    if not np.all(np.isfinite(points)) or np.any(points[:, 1] <= reference.ensemble.a_s):
        raise ValueError("reference points must be finite with b > a")
    if len(np.unique(points, axis=0)) != len(points):
        raise ValueError("reference points must not be repeated")
    if len(np.unique(points[:, 0])) < 2 or len(np.unique(points[:, 1])) < 2:
        raise ValueError("reference window must vary both z and b")
    columns = []
    for z, b in points:
        selectors = {}
        for dim, value in (("z", z), ("bT", b)):
            grid = reference.coords[dim] if dim in reference.dims else [reference.attrs["bT_fm"]]
            indices = np.flatnonzero(np.isclose(grid, value, rtol=0, atol=1e-10))
            if len(indices) != 1:
                raise ValueError(f"reference point {z, b} must exist exactly; no interpolation")
            if dim in reference.dims:
                selectors[dim] = int(indices[0])
        columns.append(reference.array.isel(**selectors).values)
    selected = np.stack(columns, axis=1)
    if np.any(np.abs(selected.imag) > 1e-12):
        raise ValueError("reference must be explicitly projected to real before extraction")
    selected = selected.real
    if np.any(selected <= 0):
        raise ValueError("reference replicas must be positive in the extraction window")
    scales = [0.8, 1.0, 1.2] if order.endswith("_rgr") else [1.0]
    c = np.stack(
        [
            coefficient(
                points[:, 0],
                points[:, 1],
                mu=mu,
                alpha_s=alpha_s,
                n_f=n_f,
                order=order,
                scale_factor=s,
                min_scale_gev=min_scale_gev,
            )
            for s in scales
        ]
    )
    effective = selected[:, None, :] / c[None, :, :]
    averages = effective.mean(axis=2)
    settings = dict(mu=mu, alpha_s=alpha_s, n_f=n_f, order=order, min_scale_gev=min_scale_gev)
    if order.startswith("nnlo"):
        settings.update(
            boundary_source="TMD(2).nb v2 / TMD_NNLO.wl; massless pure-DR nonsinglet gamma-t",
            rgr_accuracy="three-loop beta and endpoint anomalous dimension; NNLL" if order.endswith("_rgr") else "none",
            quadrature="48/64/12 refined independently and jointly to 72/96/14; absolute S tolerance 1e-5",
        )
    attrs = {
        **reference.attrs,
        "strategy": "tmd_zo",
        "zo_artifact": "residual_factor",
        "zo_settings": json.dumps(settings),
        "zo_points_fm": json.dumps(points.tolist()),
        "scale_gev": float(mu),
        "source_arxiv": "2412.19988",
        "zo_average": "equal_weight",
    }
    # Factor is independent of momentum and geometry; retain only provenance, not misleading axes.
    for key in ("bT_fm", "momentum_gev", "renormalization_scheme"):
        attrs.pop(key, None)
    factor = _new(reference, averages, ["scale_factor"], {"scale_factor": scales}, attrs, "Z_O")
    central = scales.index(1.0)
    values = effective[:, central, :]
    means = sample_mean(values, reference.resample, reference.attrs)
    covariance = sample_covariance(values, reference.resample, reference.attrs)
    projector = np.eye(len(points)) - np.ones((len(points), len(points))) / len(points)
    residual_cov = projector @ covariance @ projector.T
    rank = int(np.linalg.matrix_rank(residual_cov, tol=max(np.linalg.norm(covariance, 2) * 1e-10, 1e-30)))
    residual = projector @ means
    statistic = float(residual @ np.linalg.pinv(residual_cov, rcond=1e-10) @ residual)
    central_values = sample_mean(averages, reference.resample, reference.attrs)
    diagnostics = {
        "source_arxiv": "2412.19988",
        "settings": settings,
        "points_fm": points.tolist(),
        "scale_factors": scales,
        "Z_O": central_values.tolist(),
        "statistical_errors": sample_sdev(averages, reference.resample, reference.attrs).tolist(),
        "scale_envelope": float(np.max(np.abs(central_values - central_values[central]))),
        "effective_Z_O": sample_mean(effective, reference.resample, reference.attrs).tolist(),
        "effective_statistical_errors": sample_sdev(effective, reference.resample, reference.attrs).tolist(),
        "reference_covariance": covariance.tolist(),
        "sample_count": reference.n_sample,
        "window_diagnostic": {
            "chi2": statistic,
            "covariance_rank": rank,
            "Q": float(chi2.sf(statistic, rank)) if rank else None,
            "automatic_plateau_validation": False,
        },
        "notes": (
            "Authored window; diagnostic Q is not a plateau certification. "
            "Scale envelope is separate from statistics."
        ),
    }
    return factor, diagnostics


def apply(target, factor, *, mu, scale_factor=1.0):
    _check(target)
    if factor.dims != ["scale_factor"] or factor.attrs.get("zo_artifact") != "residual_factor":
        raise ValueError("factor must be a tmd_zo extraction artifact")
    if target.ensemble != factor.ensemble:
        raise ValueError("target and Z_O must use the identical lattice ensemble")
    if target.attrs["renormalization_family"] != factor.attrs.get("renormalization_family"):
        raise ValueError("target and Z_O require identical operator/regularization family")
    if not np.isclose(mu, factor.attrs["scale_gev"], rtol=0, atol=1e-12):
        raise ValueError("application mu must equal the factor's extraction scale")
    target, factor = align_sample_lineage(target, factor)
    lineage = merge_sample_lineage(target, factor)
    indices = np.flatnonzero(np.isclose(factor.coords["scale_factor"], scale_factor, rtol=0, atol=1e-12))
    if len(indices) != 1:
        raise ValueError("requested scale_factor is not present in the factor artifact")
    denominator = np.asarray(factor.values)[:, indices[0]]
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        raise ValueError("Z_O replicas must be finite and positive")
    values = target.values / denominator.reshape((-1,) + (1,) * len(target.dims))
    attrs = {
        **target.attrs,
        **lineage,
        "renormalization_scheme": "msbar",
        "strategy": "tmd_zo",
        "scale_gev": float(mu),
        "zo_scale_factor": float(scale_factor),
        "zo_settings": factor.attrs["zo_settings"],
        "zo_points_fm": factor.attrs["zo_points_fm"],
    }
    return _new(target, values, target.dims, target.coords, attrs, "renormalized_matrix_element")


def run(context):
    params = context.params
    if params.get("normalization"):
        raise ValueError("tmd_zo requires normalization=false")
    if params["type"] == "fit":
        result, diagnostics = extract(
            load_data(context.inputs["reference"]),
            **{key: params[key] for key in ("points_fm", "mu", "alpha_s", "n_f", "order", "min_scale_gev")},
        )
    else:
        result = apply(
            load_data(context.inputs["target"]),
            load_data(context.inputs["zO"]),
            mu=params["mu"],
            scale_factor=params["scale_factor"],
        )
        diagnostics = {
            "sample_count": result.n_sample,
            "scale_factor": params["scale_factor"],
            "mu": params["mu"],
            "scheme": "msbar",
        }
    directory = context.artifact_directory
    result.array.attrs["sample_error_mode"] = context.manifest["metadata"]["sample_error_mode"]
    (directory / "diagnostics").mkdir(parents=True, exist_ok=True)
    result.to_netcdf(directory / "output.nc")
    (directory / "diagnostics" / "zo.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    context.finish(
        result,
        {
            "stage_id": context.stage_id,
            "job_id": context.job_id,
            "result": "residual_Z_O" if params["type"] == "fit" else "renormalized_matrix_element",
            "decisions": {"strategy": "tmd_zo", "type": params["type"]},
            "diagnostics": diagnostics,
            "artifacts": ["output.nc", "diagnostics/zo.json"],
        },
    )
