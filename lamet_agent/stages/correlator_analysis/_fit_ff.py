"""Wall-source two-current form factor used by the intrinsic soft function."""

from __future__ import annotations

import json
from itertools import product
from typing import Any, Mapping

import gvar as gv
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.parallel import FitNumericalError, nonlinear_fit
from lamet_agent.plotting import configure_plot, errorband, errorline, save_figure, start_plot

from lamet_agent.stages._priors import prior_diagnostics
from ._priors import build_priors, sample_prior_scale

CURRENTS = ("two_current_X_X", "two_current_Y_Y", "two_current_X5_X5", "two_current_Y5_Y5")


def ff_fcn(x: Mapping[str, Any], p: Mapping[str, Any]) -> np.ndarray:
    """Ground-state wall ratio with the reference two-state contamination."""
    t, tau = np.asarray(x["t"]), np.asarray(x["tau"])
    if x["n_states"] == 1:
        return p["ff"] * np.ones_like(t, dtype=float)
    return (
        p["ff"]
        * (1 + p["c1"] * (np.exp(-p["dE"] * tau) + np.exp(-p["dE"] * (t - tau))))
        / (1 + p["c2"] * np.exp(-p["dE"] * t / 2))
    )


def _index(coords: list, value: float, name: str) -> int:
    found = np.flatnonzero(np.isclose(coords, value, rtol=0, atol=1e-12))
    if len(found) != 1:
        raise ValueError(f"form-factor input requires exactly one {name}={value}")
    return int(found[0])


def wall_ratios(correlators: Mapping[str, EnsembleData], tsep_values: list[int], tau_cut: int):
    """Divide each current by Re[C_Z5(t/2)+C_X5(t/2)] squared per sample."""
    by_operator = {}
    for data in correlators.values():
        operator = data.attrs.get("kernel_operator")
        if operator in by_operator:
            raise ValueError("form factor requires one input per wall-current operator")
        by_operator[operator] = data
    if set(by_operator) != {*CURRENTS, "gZ5", "gX5"}:
        raise ValueError("ff requires four X/X,Y/Y,X5/X5,Y5/Y5 currents and local gZ5,gX5 denominators")
    source = by_operator[CURRENTS[0]]
    if source.ensemble is None:
        raise ValueError("form-factor normalization requires ensemble metadata")
    if "bT_fm" not in source.attrs:
        raise ValueError("form-factor currents require physical bT_fm metadata")
    for data in by_operator.values():
        if (
            data.ensemble != source.ensemble
            or data.n_sample != source.n_sample
            or data.resample != source.resample
            or data.attrs.get("resample_id") != source.attrs.get("resample_id")
            or not np.array_equal(data.array.coords["resample"], source.array.coords["resample"])
        ):
            raise ValueError("all form-factor inputs must share an ensemble and resampling plan")
    for operator in CURRENTS:
        data = by_operator[operator]
        if data.dims != ["tsep", "tau"] or data.attrs.get("bT_fm") != source.attrs.get("bT_fm"):
            raise ValueError("four form-factor currents require tsep,tau dimensions and the same bT_fm")
        if any(data.attrs.get(key) != source.attrs.get(key) for key in ("source_momentum", "sink_momentum")):
            raise ValueError("four form-factor currents must share source and sink momenta")
    for operator in ("gZ5", "gX5"):
        data = by_operator[operator]
        if data.dims != ["t"] or float(data.attrs.get("bT_fm", -1)) != 0:
            raise ValueError("wall denominators must be local bT=0 inputs with t dimension")
        if data.attrs.get("source_momentum") != source.attrs.get("source_momentum"):
            raise ValueError("wall denominators must share the source momentum of the currents")
    if not tsep_values or len(set(tsep_values)) != len(tsep_values) or tau_cut < 0:
        raise ValueError("invalid form-factor insertion window")
    t, tau, ratios = [], [], {c: [] for c in CURRENTS}
    for tsep in tsep_values:
        if tsep % 2 or 2 * tau_cut > tsep:
            raise ValueError("ff requires even tsep with nonempty insertion windows")
        den = sum(
            np.asarray(by_operator[c].values).real[:, _index(by_operator[c].coords["t"], tsep / 2, "t")]
            for c in ("gZ5", "gX5")
        )
        if np.any(~np.isfinite(den)) or np.any(den == 0):
            raise ValueError("wall form-factor denominator is zero or nonfinite")
        for insertion in range(tau_cut, tsep - tau_cut + 1):
            t.append(tsep)
            tau.append(insertion)
            for operator in CURRENTS:
                data = by_operator[operator]
                values = np.asarray(data.values).real[
                    :, _index(data.coords["tsep"], tsep, "tsep"), _index(data.coords["tau"], insertion, "tau")
                ]
                ratios[operator].append(values / den**2)
    values = {c: np.stack(ratios[c], axis=1) for c in CURRENTS}
    if any(not np.all(np.isfinite(v)) for v in values.values()):
        raise ValueError("nonfinite form-factor values remain in the selected fitting domain")
    return source, np.asarray(t), np.asarray(tau), values


def fit_ff_samples(
    correlators: Mapping[str, EnsembleData],
    *,
    tsep_values: list[int],
    tau_cut: int,
    n_states: int,
    prior_width: float,
    energy_gev: float,
    svdcut: float,
    posterior_prior_error_scale: float,
    sample_error_mode: str = "covariance",
    priors: Mapping[str, Any] | None = None,
    sample_prior_mode: str = "center_posterior",
    workers: int = 1,
    fit_samples: bool = True,
    _parallel=None,
) -> tuple[EnsembleData | None, dict[str, Any]]:
    """Fit the four currents separately and combine their aligned fitted samples.

    ``original`` reuses the physical priors from the reference FF fit on each
    replica. ``center_posterior`` retains the earlier center-posterior replica
    regularization for controlled comparisons. Both preserve the input samples,
    the authored covariance mode, and cross-current pairing.
    """
    from .physics import _sample_diagnostic_records

    if n_states not in (1, 2) or energy_gev <= 0:
        raise ValueError("ff requires one or two states and positive energy_gev")
    replica_prior_scale = sample_prior_scale(sample_prior_mode, posterior_prior_error_scale)
    source, t, tau, ratios = wall_ratios(correlators, tsep_values, tau_cut)
    momentum = np.asarray(json.loads(source.attrs["source_momentum"]), dtype=float)
    p_gev = float(np.linalg.norm(momentum) * source.ensemble.k_s)
    if p_gev <= 0:
        raise ValueError("wall-source form factor requires nonzero momentum")
    normalization = -(source.ensemble.L_s**3) * 6 / ((p_gev**2 + energy_gev**2) / (2 * p_gev**2))
    records, fitted, failures = [], [], []
    for current, observations in ratios.items():
        typical = float(np.median(np.abs(observations)))
        if typical == 0:
            raise ValueError("form-factor ratio has no nonzero signal")
        scale = 10.0 ** np.floor(-np.log10(typical))
        scaled = observations * scale
        fit_data = EnsembleData(
            source.ensemble, source.resample, list(scaled), ["observation"], {"observation": list(range(len(t)))}
        )
        defaults = gv.BufferDict(ff=gv.gvar(-3e-6, 1e-2))
        if n_states == 2:
            defaults["log(dE)"] = gv.gvar(-1.0, 0.5)
            defaults["c1"] = gv.gvar(-0.5, 0.5)
            defaults["c2"] = gv.gvar(0.5, 0.5)
        prior = build_priors(
            defaults, priors, prior_width=prior_width, conditioning={"ff": scale}
        )
        x = {"t": t, "tau": tau, "n_states": n_states}
        initial = {key: float(gv.mean(value)) for key, value in prior.items()}
        initial["ff"] = float(np.median(scaled))
        result = nonlinear_fit(
            (x, fit_data),
            ff_fcn,
            prior,
            mode="resamples" if fit_samples else "center",
            workers=workers,
            sample_error_mode=sample_error_mode,
            sample_prior_scale=replica_prior_scale,
            svdcut=svdcut,
            maxit=10000,
            tolerate_sample_failures=fit_samples,
            capture_sample_posteriors=(0,) if fit_samples else (),
            _parallel=_parallel,
            p0=initial,
        )
        record = {
            "current": current,
            "Q": result.Q,
            "chi2": result.chi2,
            "dof": result.dof,
            "chi2_dof": result.chi2 / result.dof,
            "logGBF": result.logGBF,
            "ff": float(gv.mean(result.p["ff"])) / scale,
            "ff_sdev": float(gv.sdev(result.p["ff"])) / scale,
            "conditioning": float(scale),
            "ff_prior_mean": float(gv.mean(prior["ff"])) / scale,
            "ff_prior_sdev": float(gv.sdev(prior["ff"])) / scale,
            **prior_diagnostics(prior, priors),
            "sample_diagnostics": _sample_diagnostic_records(result),
        }
        if fit_samples:
            fitted.append([float(p["ff"]) / scale if p is not None else np.nan for p in result.samples])
            record["ff_sample_mean"] = float(np.mean(fitted[-1]))
            for i, p in enumerate(result.samples):
                if p is None:
                    failures.append({"current": current, "sample": i})
            posterior = result.sample_posteriors[0] if result.sample_posteriors else None
            if posterior is not None:
                predicted = ff_fcn(x, posterior) / scale
                record["sample0_plot"] = {
                    "t": t.tolist(),
                    "tau": tau.tolist(),
                    "y": observations[0].tolist(),
                    "yerr": (gv.sdev(fit_data.average(sample_error_mode)) / scale).tolist(),
                    "fit_mean": gv.mean(predicted).tolist(),
                    "fit_sdev": gv.sdev(predicted).tolist(),
                }
        records.append(record)
    diagnostics = {
        "fit_scope": ["ff"],
        "fits": records,
        "min_Q": min(r["Q"] for r in records),
        "worst_chi2_dof": max(r["chi2_dof"] for r in records),
        "normalization": normalization,
        "energy_gev": energy_gev,
        "momentum_gev": p_gev,
        "n_failed_samples": len(failures),
        "sample_failures": failures,
        "denominator": "Re(C_gZ5(tsep/2)+C_gX5(tsep/2))^2",
        "sample_prior_mode": sample_prior_mode,
        "sample_prior": "initial" if sample_prior_mode == "original" else "broadened_center_posterior",
        "fit_covariance": "full" if sample_error_mode == "covariance" else "diagonal",
        "sample_error_mode": sample_error_mode,
        "current_covariance": "paired_resamples",
    }
    if not fit_samples:
        return None, diagnostics
    values = np.mean(np.asarray(fitted), axis=0)[:, None] * normalization
    attrs = {
        **source.attrs,
        "observable": "form_factor",
        "method": "lsqfit",
        "fit_scope": '["ff"]',
        "energy_gev": energy_gev,
        "momentum_gev": p_gev,
        "normalization": normalization,
        "current_combination": "(X_X+Y_Y+X5_X5+Y5_Y5)/4",
        "denominator": diagnostics["denominator"],
        **{
            key: diagnostics[key]
            for key in (
                "sample_prior_mode",
                "sample_prior",
                "fit_covariance",
                "sample_error_mode",
                "current_covariance",
            )
        },
        "units": '{"values":"dimensionless","bT":"fm"}',
    }
    data = EnsembleData(
        source.ensemble,
        source.resample,
        list(values),
        ["bT"],
        {"bT": [float(source.attrs["bT_fm"])]},
        attrs=attrs,
        name="form_factor",
    )
    data.array = data.array.assign_coords(resample=source.array.coords["resample"])
    return data, diagnostics


def run(context) -> None:
    """Choose an authored wall-ratio candidate and publish all resampled currents."""
    params = context.params
    common = dict(
        correlators=context.state["correlators"],
        energy_gev=float(params["ff_energy_gev"]),
        svdcut=float(params["svdcut"]),
        posterior_prior_error_scale=float(params["posterior_prior_error_scale"]),
        sample_error_mode=context.manifest["metadata"]["sample_error_mode"],
        priors=params.get("priors"),
        sample_prior_mode=params.get("sample_prior_mode", "center_posterior"),
        workers=context.workers,
        _parallel=context._parallel,
    )
    candidates = []
    for window, nstate, width in product(params["pt3_windows"], params["nstate"]["ff"], params["prior_width"]):
        options = dict(tsep_values=window["tsep_ls"], tau_cut=window["tau_cut"], n_states=nstate, prior_width=width)
        _, diagnostic = fit_ff_samples(**common, **options, fit_samples=False)
        candidates.append(
            {
                "candidate_id": f"ff_{len(candidates) + 1:03d}",
                "method": "lsqfit",
                "fit_scope": ["ff"],
                "window": {"tau_min": window["tau_cut"]},
                "tsep_values": window["tsep_ls"],
                "nstate": {"ff": nstate},
                "prior_width": width,
                "quality_passed": diagnostic["min_Q"] >= params["q_min"],
                "min_Q": diagnostic["min_Q"],
                "worst_chi2_dof": diagnostic["worst_chi2_dof"],
                "options": options,
                "numerical_failure": False,
            }
        )
    selected = max(candidates, key=lambda c: c["min_Q"])
    data, diagnostic = fit_ff_samples(**common, **selected["options"])
    if data is None or not np.all(np.isfinite(data.values)):
        raise FitNumericalError("form-factor resamples did not all converge; refusing to publish incomplete samples")
    if context.manifest["metadata"].get("target_observable") == "tmd":
        data.array.attrs["target_observable"] = "tmd"
    data.to_netcdf(context.artifact_directory / "output.nc")
    artifacts = ["output.nc", "diagnostics/candidates.json"]
    for record in diagnostic["fits"]:
        plot = record.pop("sample0_plot", None)
        if plot is not None:
            start_plot()
            times = np.asarray(plot["t"])
            for i, tsep in enumerate(np.unique(times)):
                mask = times == tsep
                x = np.asarray(plot["tau"])[mask] - tsep / 2
                errorline(x, gv.gvar(np.asarray(plot["y"])[mask], np.asarray(plot["yerr"])[mask]), label=f"t={tsep:g}")
                errorband(x, gv.gvar(np.asarray(plot["fit_mean"])[mask], np.asarray(plot["fit_sdev"])[mask]))
            configure_plot(
                xlabel=r"$\tau-t_{\rm sep}/2$",
                ylabel="Wall-source ratio",
                legend=True,
                title=f"{context.job_id}: sample 0, {record['current']}",
            )
            path = f"plots/{record['current']}.pdf"
            save_figure(context.artifact_directory / path)
            artifacts.append(path)
    sample_records = [sample for fit in diagnostic["fits"] for sample in fit["sample_diagnostics"]]
    diagnostics = {
        **diagnostic,
        "Q": diagnostic["min_Q"],
        "chi2_dof": diagnostic["worst_chi2_dof"],
        "candidates": candidates,
        "fallback_no_q_passing": not selected["quality_passed"],
        "sample_fit_quality": {"chi2_dof": [sample["chi2_dof"] for sample in sample_records]},
    }
    directory = context.artifact_directory / "diagnostics"
    directory.mkdir(exist_ok=True)
    (directory / "candidates.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    context.state["correlator_result"] = data
    context.finish(
        data,
        {
            "stage_id": context.stage_id,
            "job_id": context.job_id,
            "result": "form_factor",
            "decisions": {
                "candidate_id": selected["candidate_id"],
                "method": "lsqfit",
                "fit_scope": ["ff"],
                "fallback_no_q_passing": not selected["quality_passed"],
            },
            "diagnostics": diagnostics,
            "artifacts": artifacts,
        },
    )
