"""C32P23 notebook equations with explicit inputs and no resampling or LLM calls."""

from pathlib import Path

import gvar as gv
import h5py
import lsqfit
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.parallel.fourier import fourier_transform
from lamet_agent.stages.renormalization.physics import ratio
from lamet_agent.stages.tmdwf_analysis.wilson_io import load_wilson


def ensemble(values, coords, attrs, name):
    return EnsembleData(None, "bootstrap", list(values), list(coords), coords, attrs=dict(attrs), name=name)


def read_ratios(path: Path, cfg: dict, samples: int):
    """Return (sample,z,t), matching cal_bare_tmdwf_single in the source notebook."""
    channels = []
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("resample_mode") != "bootstrap" or handle.attrs.get("bootstrap_samples") != samples:
            raise ValueError("Input must contain the declared number of existing bootstrap samples")
        for direction in (0, 1):
            for operator in ("gtg5", "gzg5"):
                prefix = (f"{operator}/gamma_sink0/PX0PY0PZ{cfg['pz']}/"
                          f"p1_{cfg['p1']}_p2_{cfg['p2']}")
                local = handle[f"{prefix}/bT0/zdir{direction}/bz0"][()].real.T
                if np.any(local == 0):
                    raise ValueError("Local real denominator contains zero")
                values = np.stack([handle[f"{prefix}/bT{cfg['b']}/zdir{direction}/bz{z}"][()].T
                                   for z in range(13)], axis=1)
                if values.shape != (samples, 13, 15):
                    raise ValueError(f"Unexpected input shape {values.shape}")
                theta = 2 * np.pi * (cfg["p1"] - cfg["pz"] / 2) * np.arange(13) / cfg["L_s"]
                channels.append(values / local[:, None, :] * np.exp(1j * theta * (-1)**direction)[None, :, None])
        source_hash = str(handle.attrs["source_sha256"])
    # Match the two consecutive source averages and then direction average.
    values = ((channels[0] + channels[1]) / 2 + (channels[2] + channels[3]) / 2) / 2
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite preprocessed correlators")
    attrs = {
        "target_observable": "tmdwf", "ensemble_id": "C32P23", "a_fm": cfg["a_fm"],
        "L_s": cfg["L_s"], "b_perp_lattice": cfg["b"], "pz_lattice": cfg["pz"],
        "p1": cfg["p1"], "p2": cfg["p2"], "coord_unit": "fm",
        "momentum_gev": 2 * np.pi * HBAR_C_GEV_FM * cfg["pz"] / (cfg["L_s"] * cfg["a_fm"]),
        "resample_id": source_hash, "input_source_sha256": source_hash,
        "source_combination": "equal gtg5/gzg5 and zdir0/zdir1 averages after ratio and phase",
        "fit_time_convention": "legacy zero-based array index; stored physical times are index+1",
    }
    return values, attrs


def fit_one_z(task):
    """Fit every replica using the fixed full-bootstrap standard deviation."""
    values, window, svdcut = task
    nstate, r0, r1, i0, i1 = window
    x = {"re": np.arange(r0, r1), "im": np.arange(i0, i1)}
    real, imag = values.real[:, r0:r1], values.imag[:, i0:i1]
    errors = {"re": np.std(real, axis=0), "im": np.std(imag, axis=0)}
    if any(np.any(v <= 0) for v in errors.values()):
        raise ValueError("Zero bootstrap error in a fitted time window")
    prior = {"psi_re": gv.gvar(1, 2 if nstate == 1 else 3),
             "psi_im": gv.gvar(-1, 2 if nstate == 1 else 3)}
    if nstate == 2:
        prior["delta_E"] = gv.gvar(0, 15)

    def model(x, p):
        return {part: p[f"psi_{part}"] * (np.ones_like(x[part]) if nstate == 1
                else 1 + np.exp(-x[part] * p["delta_E"])) for part in ("re", "im")}

    results, diagnostics = [], []
    for re, im in zip(real, imag):
        fit = lsqfit.nonlinear_fit(data=(x, {"re": gv.gvar(re, errors["re"]),
                                           "im": gv.gvar(im, errors["im"])}),
                                  prior=prior, fcn=model, svdcut=svdcut)
        if fit.error is not None:
            raise RuntimeError(f"Bare fit failed: {fit.error}")
        results.append(complex(gv.mean(fit.p["psi_re"]), gv.mean(fit.p["psi_im"])))
        diagnostics.append([float(fit.chi2), int(fit.dof), float(fit.Q)])
    return np.asarray(results), np.asarray(diagnostics)


def bare_fit(path, cfg, samples, pool):
    values, attrs = read_ratios(path, cfg, samples)
    results = pool.map(fit_one_z, [(values[:, z, :], cfg["windows"][z], cfg["svdcut"]) for z in range(13)])
    fitted = np.stack([row[0] for row in results], axis=1)
    diag = np.stack([row[1] for row in results], axis=1)
    return ensemble(fitted, {"z": (np.arange(13) * cfg["a_fm"]).tolist()}, attrs, "bare_tmdwf"), diag


def wilson_model(x, p):
    return p["c"] * np.exp(-x * p["E"]) * (1 + p["c1"] * np.exp(-x * p["delta_E"]))


def wilson_factor(path, cfg, samples):
    raw, provenance = load_wilson(path)
    values = (raw[:, 0, :, cfg["b"]] + raw[:, 1, :, cfg["b"]] + raw[:, 2, :, cfg["b"]]) / 3
    n = values.shape[0]
    # Match the notebook's leave-one-out, diagonal-error mean fit.
    jackknife = (np.sum(values, axis=0) - values) / (n - 1)
    x = np.arange(cfg["fit_start"], cfg["fit_stop"])
    y = gv.gvar(jackknife.mean(axis=0)[x], jackknife.std(axis=0)[x] * np.sqrt(n - 1))
    fit = lsqfit.nonlinear_fit(data=(x, y), prior={k: gv.gvar(*v) for k, v in cfg["priors"].items()},
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
    data = ensemble(np.broadcast_to(denominator, (samples, 13)).copy(),
                    {"z": (np.arange(13) * cfg["a_fm"]).tolist()}, attrs, "wilson_denominator")
    diagnostics = {"chi2": float(fit.chi2), "dof": int(fit.dof), "Q": float(fit.Q),
                   "parameters": {k: [float(gv.mean(v)), float(gv.sdev(v))] for k, v in fit.p.items()},
                   "configuration_count": n, "factor_uncertainty": attrs["factor_uncertainty"],
                   "error_estimator": attrs["wilson_error_estimator"], "fit_mode": attrs["wilson_fit_mode"],
                   "input_provenance": provenance}
    return data, diagnostics


def renormalize(target, denominator):
    for key in ("ensemble_id", "a_fm", "b_perp_lattice"):
        if target.attrs.get(key) != denominator.attrs.get(key):
            raise ValueError(f"Target and Wilson factor mismatch: {key}")
    if target.coords != denominator.coords:
        raise ValueError("Target and Wilson factor z grids differ")
    if not np.isfinite(denominator.values).all() or np.any(denominator.values <= 0):
        raise ValueError("Wilson denominator must be positive and finite")
    output = ratio(target, denominator)
    attrs = output.attrs
    attrs.update({"renormalization_scheme": "tmdwf_wilson_loop",
                         "factor_uncertainty": denominator.attrs["factor_uncertainty"],
                         "ZO": denominator.attrs["ZO"], "L": denominator.attrs["L"],
                         "normalized_at_origin": 0})
    return ensemble(output.values, output.coords, attrs, "renormalized_tmdwf")


def tail_model(x, p):
    output = {}
    for part in ("real", "imag"):
        v = x[part]
        angle = v / 2 - np.pi * p["d"] / 2
        common = v**(-p["d"]) * np.exp(-v / p["lambda0"])
        output[part] = common * (
            (p["c1"] + p["c2"]) * p["f1"] * np.cos(angle)
            + (p["c2"] - p["c1"]) * p["f2"] * np.sin(angle)
            if part == "real" else
            (p["c1"] + p["c2"]) * p["f2"] * np.cos(angle)
            - (p["c2"] - p["c1"]) * p["f1"] * np.sin(angle))
    return output


def fit_tail_batch(task):
    x, values, cov_re, cov_im, priors, svdcut = task
    prior = {k: gv.gvar(*v) for k, v in priors.items()}
    results, diagnostics = [], []
    for sample in values:
        fit = lsqfit.nonlinear_fit(data=({"real": x, "imag": x},
                                  {"real": gv.gvar(sample.real, cov_re), "imag": gv.gvar(sample.imag, cov_im)}),
                                  prior=prior, fcn=tail_model, svdcut=svdcut)
        if fit.error is not None:
            raise RuntimeError(f"Tail fit failed: {fit.error}")
        results.append({k: float(gv.mean(v)) for k, v in fit.p.items()})
        diagnostics.append([float(fit.chi2), int(fit.dof), float(fit.Q)])
    return results, diagnostics


def transform(data, cfg, workers, pool):
    if data.attrs.get("renormalization_scheme") != "tmdwf_wilson_loop":
        raise ValueError("TMDWF Fourier input must be Wilson-loop renormalized")
    if data.dims != ["z"] or len(data.coords["z"]) != 13 or data.attrs.get("coord_unit") != "fm":
        raise ValueError("Expected the 13-point renormalized physical-z grid")
    momentum = float(data.attrs["momentum_gev"])
    z = np.asarray(data.coords["z"])
    lam = z * momentum / HBAR_C_GEV_FM
    start, stop = cfg["tail_start"], cfg["tail_stop"]
    values = np.asarray(data.values)[:, start:stop]
    cov_re, cov_im = np.cov(values.real, rowvar=False), np.cov(values.imag, rowvar=False)
    tasks = [(lam[start:stop], chunk, cov_re, cov_im, cfg["priors"], cfg["svdcut"])
             for chunk in np.array_split(values, min(workers, data.n_sample))]
    batches = pool.map(fit_tail_batch, tasks)
    parameters = [p for batch in batches for p in batch[0]]
    diagnostics = np.asarray([d for batch in batches for d in batch[1]])
    positive_lam = np.arange(0, cfg["extent_factor"] * max(lam), lam[1] - lam[0])
    stitch = cfg["stitch_index"]
    if len(positive_lam) <= stitch + 1:
        raise ValueError("Extension does not reach beyond the stitching point")
    extended = np.empty((data.n_sample, len(positive_lam)), dtype=complex)
    extended[:, :stitch + 1] = data.values[:, :stitch + 1]
    for i, p in enumerate(parameters):
        if p["lambda0"] <= 0:
            raise ValueError(f"Nondecaying fitted tail for bootstrap sample {i}")
        prediction = tail_model({"real": positive_lam[stitch + 1:], "imag": positive_lam[stitch + 1:]}, p)
        extended[i, stitch + 1:] = prediction["real"] + 1j * prediction["imag"]
    # Explicit reproduction: both components are even in the old bootstrap notebook.
    signed = np.concatenate((extended[:, ::-1], extended[:, 1:]), axis=1)
    signed_lam = np.concatenate((-positive_lam[::-1], positive_lam[1:]))
    attrs = dict(data.attrs)
    attrs.update({"negative_z": cfg["negative_z"], "tail_start": start, "tail_stop": stop,
                  "stitch_index": stitch, "tail_uncertainty": "per_bootstrap_fits"})
    ext = ensemble(signed, {"z": (signed_lam * HBAR_C_GEV_FM / momentum).tolist()}, attrs, "extended_tmdwf")
    result = fourier_transform(ext, np.linspace(cfg["x_start"], cfg["x_stop"], cfg["x_points"]).tolist(),
                               momentum_gev=momentum, phase_sign=cfg["phase_sign"], x_shift=cfg["x_shift"],
                               prefactor="pz_over_2pi", workers=workers, _parallel=pool)
    attrs = result.attrs
    attrs["observable"] = "quasi_tmdwf_before_hard_matching"
    result = ensemble(result.values, result.coords, attrs, "quasi_tmdwf")
    return result, ext, diagnostics, parameters
