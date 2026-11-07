"""Axial TMD wave-function fits with explicit physical amplitude normalization.

SS and SP correlators have different overlap powers. Conditioning an SS block
by s and its SP block by sqrt(s) preserves the physical axial matrix element.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import gvar as gv
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.parallel import nonlinear_fit
from lamet_agent.parallel._pool import _ParallelPool

from lamet_agent.stages._priors import prior_diagnostics
from ._priors import build_priors, sample_prior_scale
from ._qda_windows import qda_time_masks, resolve_qda_windows


def tmdwf_fcn(x: Mapping[str, Any], p: Mapping[str, Any]) -> np.ndarray:
    """Periodic SS and axial SP blocks; phi0 is O00/E0."""
    energies = [p["E0"]]
    for state in range(1, x["n_states"]):
        energies.append(energies[-1] + p[f"dE{state}"])
    qtime = np.asarray(x["qda_t"])
    qdecay = [np.exp(-e * qtime) + np.exp(-e * (x["extent"] - qtime)) for e in energies]
    blocks = []
    if x["output"] == "tmdwf":
        ptime = np.asarray(x["pt2_t"])
        blocks.append(
            sum(
                p[f"overlap{j}"] ** 2 / (2 * e) * (np.exp(-e * ptime) + np.exp(-e * (x["extent"] - ptime)))
                for j, e in enumerate(energies)
            )
        )
        for component in x["components"]:
            blocks.append(
                sum(p[f"overlap{j}"] * p[f"phi{j}_{component}"] / 2 * qdecay[j] for j in range(len(energies)))
            )
    else:
        blocks.extend(p[f"amplitude_{c}"] * qdecay[0] for c in x["components"])
    return np.concatenate(blocks)


def _scale(values: np.ndarray, target: float) -> float:
    finite = np.abs(values[np.isfinite(values) & (values != 0)])
    if not finite.size:
        raise ValueError("TMD correlator fit window has no finite nonzero signal")
    return float(10.0 ** np.floor(np.log10(target / np.median(finite))))


def fit_tmdwf_samples(
    correlators: Mapping[str, EnsembleData],
    *,
    qda_output: str,
    qda_window: dict[str, int] | None,
    tmd_phase_rotation: bool,
    fit_scope: list[str] | tuple[str, ...],
    components: str,
    tmin: int,
    tmax: int,
    n_states: Mapping[str, int] | int,
    prior_width: float,
    svdcut: float,
    posterior_prior_error_scale: float,
    qda_windows_by_z: Mapping[str, Any] | None = None,
    sample_error_mode: str = "covariance",
    priors: Mapping[str, Any] | None = None,
    sample_prior_mode: str = "center_posterior",
    workers: int = 1,
    tune_z: int | float | None = None,
    fit_samples: bool = True,
    show_progress: bool = False,
    _parallel: _ParallelPool | None = None,
) -> tuple[EnsembleData | None, dict[str, Any]]:
    """Fit a joint axial WF or a standalone one-state p0 reference."""
    from ._scope import parse_fit_scope, resolve_atom_n_states
    from .physics import _sample_diagnostic_records

    replica_prior_scale = sample_prior_scale(sample_prior_mode, posterior_prior_error_scale)
    scope = parse_fit_scope(fit_scope)
    counts = resolve_atom_n_states(n_states, scope)
    sources = [d for d in correlators.values() if d.attrs.get("correlator_type") == "qda"]
    if len(sources) != 1 or sources[0].dims != ["t", "z"]:
        raise ValueError("TMDWF fitting requires one qda input with t,z dimensions")
    source = sources[0]
    if source.ensemble is None:
        raise ValueError("TMDWF fitting requires ensemble metadata")
    nstate = counts["qda"]
    if qda_output == "tmdwf" and (
        set(scope.atoms) != {"2pt", "qda"} or len(scope.stages) != 1 or counts["2pt"] != nstate
    ):
        raise ValueError("tmdwf requires a joint 2pt+qda fit with equal state counts")
    if qda_output == "amplitude" and (scope.atoms != ("qda",) or nstate != 1):
        raise ValueError("amplitude requires a single-state qda-only fit")
    if qda_output not in {"tmdwf", "amplitude"}:
        raise ValueError("unsupported axial output convention")
    selected_components = {"real": ("re",), "imag": ("im",), "both": ("re", "im")}[components]
    times = np.asarray(source.coords["t"], dtype=float)
    z = np.asarray(source.coords["z"], dtype=float)
    window = qda_window or {"tmin": tmin, "tmax": tmax}
    windows = resolve_qda_windows(z, window, qda_windows_by_z)
    time_masks = qda_time_masks(times, z, window, qda_windows_by_z)
    for zi, mask in enumerate(time_masks.T):
        if np.count_nonzero(mask) < 2 * nstate:
            raise ValueError(f"qDA window at z={z[zi]:g} must contain at least 2*n_states times")
    source_values = np.asarray(source.values, dtype=complex).copy()
    if tmd_phase_rotation:
        if str(source.attrs.get("phase_rotation_applied", "false")).lower() in {"true", "1"}:
            raise ValueError("the input qTMDWF phase is already rotated")
        if source.attrs.get("kernel_operator") not in {"g14", "gzg5"}:
            raise ValueError("native axial phase rotation requires the longitudinal g14 channel")
        momentum = json.loads(source.attrs["sink_momentum"])
        direction = source.attrs.get("longitudinal_direction", source.attrs.get("displacement_direction", [1, 0, 0]))
        direction = json.loads(direction) if isinstance(direction, str) else direction
        phase = -1j * np.exp(-1j * np.dot(momentum, direction) * 2 * np.pi / source.ensemble.L_s * z / 2)
        source_values *= phase[None, None, :]
    ptvalues = None
    ptime = np.array([])
    s2 = 1.0
    if qda_output == "tmdwf":
        points = [d for d in correlators.values() if d.attrs.get("correlator_type") == "two_point"]
        if len(points) != 1:
            raise ValueError("axial TMDWF requires one SS two-point input")
        pt = points[0]
        if (
            pt.n_sample != source.n_sample
            or pt.resample != source.resample
            or pt.attrs.get("resample_id") != source.attrs.get("resample_id")
            or not np.array_equal(pt.array.coords["resample"], source.array.coords["resample"])
            or pt.attrs.get("sink_momentum") != source.attrs.get("sink_momentum")
        ):
            raise ValueError("TMDWF and SS inputs must share momentum and resampling")
        all_ptime = np.asarray(pt.coords["t"], dtype=float)
        pmask = (all_ptime >= tmin) & (all_ptime < tmax)
        ptime = all_ptime[pmask]
        if len(ptime) < 2 * nstate:
            raise ValueError("two-point window must contain at least 2*n_states times")
        ptvalues = np.asarray(pt.values).real[:, pmask]
        s2 = _scale(ptvalues, 1e-4)
        sq = np.sqrt(s2)
    else:
        sq = _scale(source_values[:, time_masks].real, 1e-4)
    zindices = list(range(len(z)))
    if tune_z is not None:
        matches = np.flatnonzero(np.isclose(z, tune_z, rtol=0, atol=1e-12))
        if len(matches) != 1 or fit_samples:
            raise ValueError("tune_z must identify one available center-fit separation")
        zindices = [int(matches[0])]
    elif not fit_samples:
        raise ValueError("center TMDWF tuning requires tune_z")
    output = np.zeros((source.n_sample, len(z)), dtype=complex) if fit_samples else None
    records, failures = [], []
    parallel = _parallel or _ParallelPool(min(workers, source.n_sample))
    try:
        for zi in zindices:
            mask = time_masks[:, zi]
            x = {
                "n_states": nstate,
                "extent": source.ensemble.L_t,
                "output": qda_output,
                "qda_t": times[mask],
                "pt2_t": ptime,
                "components": selected_components,
            }
            pieces = [ptvalues * s2] if ptvalues is not None else []
            for c in selected_components:
                vals = source_values[:, mask, zi].real if c == "re" else source_values[:, mask, zi].imag
                pieces.append(vals * sq)
            observations = np.concatenate(pieces, axis=1)
            if not np.all(np.isfinite(observations)):
                raise ValueError("nonfinite TMDWF values remain in the selected fit domain")
            defaults = gv.BufferDict()
            conditioning = {}
            # Positive energies and overlaps remove the spectral sign ambiguity.
            energy_guess = (
                1.0 if qda_output == "tmdwf" else max(source.ensemble.m_pi * source.ensemble.a_t / 0.1973269804, 0.1)
            )
            defaults["log(E0)"] = gv.log(gv.gvar(energy_guess, 0.5 * energy_guess))
            for j in range(1, nstate):
                defaults[f"log(dE{j})"] = gv.gvar(-1.0, 0.5)
            if qda_output == "tmdwf":
                for j in range(nstate):
                    name = f"log(overlap{j})"
                    # Existing heuristic defaults are expressed in conditioned
                    # units; explicit manifest priors are supplied in raw units.
                    defaults[name] = gv.gvar(-1.0, 3.0)
                    if name in (priors or {}):
                        conditioning[name] = sq
                    for c in selected_components:
                        defaults[f"phi{j}_{c}"] = gv.gvar(0.0, 1.0)
            else:
                for c in selected_components:
                    name = f"amplitude_{c}"
                    defaults[name] = gv.gvar(0.0, 1.0)
                    if name in (priors or {}):
                        conditioning[name] = sq
            prior = build_priors(
                defaults, priors, prior_width=prior_width, conditioning=conditioning
            )
            fit_data = EnsembleData(
                source.ensemble,
                source.resample,
                list(observations),
                ["observation"],
                {"observation": list(range(observations.shape[1]))},
            )
            result = nonlinear_fit(
                (x, fit_data),
                tmdwf_fcn,
                prior,
                workers=workers,
                sample_prior_scale=replica_prior_scale,
                sample_error_mode=sample_error_mode,
                mode="resamples" if fit_samples else "center",
                tolerate_sample_failures=fit_samples,
                capture_sample_posteriors=(0,) if fit_samples else (),
                svdcut=svdcut,
                maxit=10000,
                _parallel=parallel if fit_samples else None,
            )
            energy_samples = [float(p["E0"]) if p is not None else None for p in result.samples]
            metric = {
                "z": float(z[zi]),
                "qda_window": windows[zi],
                "chi2": result.chi2,
                "dof": result.dof,
                "chi2_dof": result.chi2 / result.dof,
                "Q": result.Q,
                "logGBF": result.logGBF,
                "n_data": observations.shape[1],
                "n_params": len(prior),
                "E0": float(result.pmean["E0"]),
                "E0_sdev": float(gv.sdev(result.p["E0"])),
                "E0_samples": energy_samples,
                "sample_diagnostics": _sample_diagnostic_records(result),
                "sample0_plot": None,
                **prior_diagnostics(prior, priors),
            }
            if fit_samples:
                for i, p in enumerate(result.samples):
                    if p is None:
                        output[i, zi] = np.nan + 1j * np.nan
                        failures.append({"z": float(z[zi]), "sample": i})
                    else:

                        def value(c):
                            return float(p[f"phi0_{c}"]) if qda_output == "tmdwf" else float(p[f"amplitude_{c}"]) / sq

                        output[i, zi] = (value("re") if "re" in selected_components else 0.0) + 1j * (
                            value("im") if "im" in selected_components else 0.0
                        )
                if result.sample_posteriors and result.sample_posteriors[0] is not None:
                    posterior = result.sample_posteriors[0]
                    curve = tmdwf_fcn(x, posterior)
                    offset = len(ptime) if ptvalues is not None else 0
                    err = gv.sdev(fit_data.average(sample_error_mode))
                    plots = []
                    for ci, c in enumerate(selected_components):
                        lo = offset + ci * int(mask.sum())
                        sl = slice(lo, lo + int(mask.sum()))
                        # Remove the leading ground-state decay to display an amplitude plateau.
                        e0 = float(gv.mean(posterior["E0"]))
                        decay = np.exp(-e0 * times[mask]) + np.exp(-e0 * (source.ensemble.L_t - times[mask]))
                        factor = (2 / float(gv.mean(posterior["overlap0"])) if ptvalues is not None else 1 / sq) / decay
                        fitcurve = np.asarray(curve[sl], dtype=object) * factor
                        plateau = posterior[f"phi0_{c}"] if ptvalues is not None else posterior[f"amplitude_{c}"] / sq
                        plots.append(
                            {
                                "kind": "tmdwf_amplitude",
                                "component": c,
                                "series": [
                                    {
                                        "label": "axial correlator / ground-state decay",
                                        "x": times[mask].tolist(),
                                        "y": (observations[0, sl] * factor).tolist(),
                                        "yerr": (err[sl] * factor).tolist(),
                                        "fit_x": times[mask].tolist(),
                                        "fit_mean": gv.mean(fitcurve).tolist(),
                                        "fit_sdev": gv.sdev(fitcurve).tolist(),
                                    }
                                ],
                                "plateau_mean": float(gv.mean(plateau)),
                                "plateau_sdev": float(gv.sdev(plateau)),
                            }
                        )
                    metric["sample0_plot"] = {"z": float(z[zi]), "plots": plots}
            records.append(metric)
    finally:
        if _parallel is None:
            parallel.close()
    diagnostics = {
        "fit_scope": scope.as_list(),
        "qda_output": qda_output,
        "qda_window": window,
        "qda_windows_by_z": dict(qda_windows_by_z or {}),
        "pt2_conditioning": s2,
        "qda_conditioning": sq,
        "sample_prior_mode": sample_prior_mode,
        "fits": records,
        "min_Q": min(r["Q"] for r in records),
        "max_chi2_dof": max(r["chi2_dof"] for r in records),
        "n_data": records[0]["n_data"],
        "n_params": records[0]["n_params"],
        "n_failed_samples": len(failures),
        "sample_failures": failures,
    }
    if tune_z is not None:
        diagnostics.update({k: records[0][k] for k in ("chi2", "dof", "chi2_dof", "Q", "logGBF")})
        diagnostics["tune_z"] = float(tune_z)
    if output is None:
        return None, diagnostics
    attrs = {
        **source.attrs,
        "method": "lsqfit",
        "observable": "tmdwf",
        "target_observable": "tmd",
        "qda_output": qda_output,
        "sample_prior_mode": sample_prior_mode,
        "amplitude_convention": "O00/E0" if qda_output == "tmdwf" else "periodic_amplitude",
        "phase_rotation_applied": "true" if tmd_phase_rotation else "false",
        "fit_scope": json.dumps(scope.as_list()),
        "tmin": tmin,
        "tmax": tmax,
        "qda_tmin": window["tmin"],
        "qda_tmax": window["tmax"],
    }
    if qda_windows_by_z:
        attrs["qda_windows_by_z"] = json.dumps(qda_windows_by_z, sort_keys=True)
    values = output.real if components == "real" else output.imag if components == "imag" else output
    data = EnsembleData(
        source.ensemble,
        source.resample,
        list(values),
        ["z"],
        {"z": z.tolist()},
        attrs=attrs,
        name="bare_matrix_element",
    )
    data.array = data.array.assign_coords(resample=source.array.coords["resample"])
    return data, diagnostics
