"""C32P23 fixed-central Wilson-loop prescription for paired GI pion wavefunctions."""

import gvar as gv
import lsqfit
import numpy as np

from lamet_agent.data import EnsembleData, merge_sample_lineage
from lamet_agent.stages._priors import validate_prior_overrides
from lamet_agent.stages.renormalization._wilson_io import load_wilson
from lamet_agent.stages.renormalization.physics import physical_z_coordinates


def average_tmdwf_channels(targets):
    """Average paired gtg5/gzg5 x two-direction fits without refitting or resampling."""
    if len(targets) != 4 or {
        (item.attrs.get("source_operator"), item.attrs.get("zdir")) for item in targets
    } != {(operator, direction) for operator in ("gtg5", "gzg5") for direction in (0, 1)}:
        raise ValueError("requires exactly gtg5/gzg5 times zdir0/zdir1")
    first = targets[0]
    if first.ensemble is None or first.resample != "bootstrap" or not first.attrs.get("resample_id"):
        raise ValueError("native fits require ensemble metadata and paired bootstrap samples")
    if first.attrs.get("gfix") != "GI":
        raise ValueError("Wilson-loop subtraction requires GI data")
    keys = ("resample_id", "bT", "sink_momentum", "source_momentum", "momentum_gev",
            "p1", "p2", "momentum_partitions", "partition_combination", "phase_convention", "gfix", "parton", "hadron")
    for item in targets:
        if (item.ensemble != first.ensemble or item.resample != first.resample
                or item.n_sample != first.n_sample or item.dims != ["z"] or item.coords != first.coords
                or not np.array_equal(item.array.coords["resample"], first.array.coords["resample"])
                or any(item.attrs.get(key) != first.attrs.get(key) for key in keys)):
            raise ValueError("native source fits have incompatible grids, sampling, or physical metadata")
        if item.attrs.get("phase_convention") != "midpoint" or not np.isfinite(item.values).all():
            raise ValueError("native fits must be finite midpoint amplitudes")
    attrs = {**first.attrs, **merge_sample_lineage(*targets)}
    attrs.update({"ensemble_id": first.ensemble.id, "a_fm": first.ensemble.a_s,
                  "b_perp_lattice": first.attrs["bT"], "target_observable": "tmdwf",
                  "source_combination": "equal average AFTER four independent native qda_ratio fits"})
    for key in ("source_operator", "zdir", "correlator_id", "partition_reference_local_dataset"):
        attrs.pop(key, None)
    result = EnsembleData(first.ensemble, "bootstrap", list(np.mean([d.values for d in targets], axis=0)),
                          ["z"], first.coords, attrs=attrs, name="native_bare_tmdwf")
    result.array = result.array.assign_coords(resample=first.array.coords["resample"])
    return result


def wilson_model(x, p):
    return p["c"] * np.exp(-x * p["E"]) * (1 + p["c1"] * np.exp(-x * p["delta_E"]))


def parse_wilson_priors(priors):
    """Read named gvar strings or the original [mean, width] Wilson inputs."""
    if not isinstance(priors, dict) or set(priors) != {"c", "E", "c1", "delta_E"}:
        raise ValueError("requires named priors for c, E, c1 and delta_E")
    named = {}
    for name, value in priors.items():
        if isinstance(value, list):
            if len(value) != 2 or any(type(v) not in (int, float) for v in value) or value[1] <= 0:
                raise ValueError(f"{name} requires a numeric [mean, positive width] or a gvar string")
            value = f"{value[0]} +/- {value[1]}"
        named[name] = value
    parsed = validate_prior_overrides(named)
    return {name: gv.gvar(prior["mean"], prior["sdev"]) for name, prior in parsed.items()}


def wilson_factor(path, cfg, samples):
    """Fit the raw Wilson mean once; return identical central denominators for paired samples."""
    raw, provenance = load_wilson(path)
    values = (raw[:, 0, :, cfg["b"]] + raw[:, 1, :, cfg["b"]] + raw[:, 2, :, cfg["b"]]) / 3
    n = values.shape[0]
    # Match the notebook's leave-one-out, diagonal-error mean fit.
    jackknife = (np.sum(values, axis=0) - values) / (n - 1)
    x = np.arange(cfg["fit_start"], cfg["fit_stop"])
    y = gv.gvar(jackknife.mean(axis=0)[x], jackknife.std(axis=0)[x] * np.sqrt(n - 1))
    fit = lsqfit.nonlinear_fit(data=(x, y), prior=parse_wilson_priors(cfg["priors"]),
                              fcn=wilson_model, svdcut=cfg["svdcut"], maxit=500000)
    if fit.error is not None:
        raise RuntimeError(f"Wilson fit failed: {fit.error}")
    central = wilson_model(np.arange(13) + 2 * cfg["L"], gv.mean(fit.p))
    if not np.all(np.isfinite(central)) or np.any(central <= 0):
        raise ValueError("Fitted Wilson loop must be positive and finite on the application grid")
    denominator = cfg["ZO"] * np.sqrt(central)
    attrs = {"target_observable": "tmdwf", "ensemble_id": "C32P23", "a_fm": cfg["a_fm"],
             "coord_unit": "fm", "b_perp_lattice": cfg["b"], "ZO": cfg["ZO"], "L": cfg["L"],
             "factor_uncertainty": "fixed_central_value_not_propagated",
             "factor_samples": "identical copies of fixed denominator, not resampled Wilson data",
             "wilson_error_estimator": "leave_one_out_jackknife",
             "wilson_fit_mode": "single_mean_fit_with_diagonal_errors",
             "wilson_configuration_count": n, "wilson_source_sha256": provenance["source_sha256"]}
    data = EnsembleData(None, "bootstrap", list(np.broadcast_to(denominator, (samples, 13)).copy()),
                        ["z"], {"z": (np.arange(13) * cfg["a_fm"]).tolist()},
                        attrs=attrs, name="wilson_denominator")
    diagnostics = {"chi2": float(fit.chi2), "dof": int(fit.dof), "Q": float(fit.Q),
                   "parameters": {k: [float(gv.mean(v)), float(gv.sdev(v))] for k, v in fit.p.items()},
                   "configuration_count": n, "factor_uncertainty": attrs["factor_uncertainty"],
                   "error_estimator": attrs["wilson_error_estimator"], "fit_mode": attrs["wilson_fit_mode"],
                   "input_provenance": provenance}
    return data, diagnostics


def apply_wilson_factor(target, denominator):
    """Apply a fixed positive Wilson denominator, preserving the target ensemble and replicas."""
    target = physical_z_coordinates(target)
    for key in ("ensemble_id", "a_fm", "b_perp_lattice"):
        if target.attrs.get(key) != denominator.attrs.get(key):
            raise ValueError(f"Target and Wilson factor mismatch: {key}")
    if target.coords != denominator.coords:
        raise ValueError("Target and Wilson factor z grids differ")
    if not np.isfinite(denominator.values).all() or np.any(denominator.values <= 0):
        raise ValueError("Wilson denominator must be positive and finite")
    if denominator.attrs.get("factor_uncertainty") != "fixed_central_value_not_propagated":
        raise ValueError("Wilson application requires the explicit fixed-central policy")
    if not np.array_equal(denominator.values, np.broadcast_to(denominator.values[0], denominator.values.shape)):
        raise ValueError("Wilson fixed-central denominator must be identical across samples")
    if target.resample != denominator.resample or target.n_sample != denominator.n_sample:
        raise ValueError("Wilson factor and target must have matching replica counts and modes")
    attrs = target.attrs
    attrs.update({"renormalization_scheme": "tmdwf_wilson_loop",
                         "factor_uncertainty": denominator.attrs["factor_uncertainty"],
                         "ZO": denominator.attrs["ZO"], "L": denominator.attrs["L"],
                         "normalized_at_origin": 0})
    result = EnsembleData(target.ensemble, target.resample, list(target.values / denominator.values[0]), target.dims,
                          target.coords, attrs=attrs, name="renormalized_tmdwf")
    result.array = result.array.assign_coords(resample=target.array.coords["resample"])
    return result
