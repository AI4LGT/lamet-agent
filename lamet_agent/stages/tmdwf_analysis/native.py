"""Data adapters for the native correlator and Fourier workflows; no fit models."""

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.stages.renormalization.physics import physical_z_coordinates
from lamet_agent.stages.tmdwf_analysis.physics import renormalize


def prepare_projection(targets, denominator, projection):
    """Average four fitted channels, renormalize, and rotate one midpoint part."""
    if projection not in {"real", "imag"}:
        raise ValueError("projection must be real or imag")
    if len(targets) != 4 or {
        (item.attrs.get("source_operator"), item.attrs.get("zdir")) for item in targets
    } != {(operator, direction) for operator in ("gtg5", "gzg5") for direction in (0, 1)}:
        raise ValueError("requires exactly gtg5/gzg5 times zdir0/zdir1")
    first = targets[0]
    if first.ensemble is None or first.resample != "bootstrap" or not first.attrs.get("resample_id"):
        raise ValueError("native fits require ensemble metadata and paired bootstrap samples")
    keys = ("resample_id", "bT", "sink_momentum", "source_momentum", "momentum_gev",
            "p1", "p2", "phase_convention", "gfix", "parton", "hadron")
    for item in targets:
        if (item.ensemble != first.ensemble or item.resample != first.resample
                or item.n_sample != first.n_sample or item.dims != ["z"] or item.coords != first.coords
                or any(item.attrs.get(key) != first.attrs.get(key) for key in keys)):
            raise ValueError("native source fits have incompatible grids, sampling, or physical metadata")
        if item.attrs.get("phase_convention") != "midpoint" or not np.isfinite(item.values).all():
            raise ValueError("native fits must be finite midpoint amplitudes")
    physical = physical_z_coordinates(first)
    attrs = physical.attrs
    attrs.update({"ensemble_id": first.ensemble.id, "a_fm": first.ensemble.a_s,
                  "b_perp_lattice": first.attrs["bT"], "target_observable": "tmdwf",
                  "source_combination": "equal average AFTER four independent native qda_ratio fits"})
    for key in ("source_operator", "zdir", "correlator_id"):
        attrs.pop(key, None)
    average = EnsembleData(first.ensemble, "bootstrap", list(np.mean([d.values for d in targets], axis=0)),
                           ["z"], physical.coords, attrs=attrs, name="native_bare_tmdwf")
    renormalized = renormalize(average, denominator)
    values = getattr(renormalized.values, projection)
    lam = np.asarray(physical.coords["z"]) * attrs["momentum_gev"] / HBAR_C_GEV_FM
    attrs = renormalized.attrs
    attrs.update({"tmdwf_projection": projection, "phase_convention": "endpoint_from_even_midpoint",
                  "negative_z": "even_both_notebook", "native_tail_profile": "pion_da_components"})
    # F[exp(-i lambda/2) Re(Psi)] + i F[exp(-i lambda/2) Im(Psi)]
    # equals the midpoint transform with exp(+i (x-1/2) lambda).
    return EnsembleData(first.ensemble, "bootstrap", list(values * np.exp(-0.5j * lam)),
                        ["z"], physical.coords, attrs=attrs, name=f"tmdwf_{projection}_endpoint")


def combine_transforms(real, imag):
    """Combine paired native transforms without dropping imaginary TMDWF values."""
    if (real.ensemble != imag.ensemble or real.resample != imag.resample or real.n_sample != imag.n_sample
            or real.dims != ["x"] or imag.dims != ["x"] or real.coords != imag.coords):
        raise ValueError("native component transforms must have identical sample and x grids")
    keys = ("resample_id", "b_perp_lattice", "momentum_gev", "phase_convention", "negative_z")
    if not real.attrs.get("resample_id") or any(real.attrs.get(k) != imag.attrs.get(k) for k in keys):
        raise ValueError("native component transforms must share physical and sampling provenance")
    if real.attrs.get("tmdwf_projection") != "real" or imag.attrs.get("tmdwf_projection") != "imag":
        raise ValueError("expected the real and imaginary midpoint projections in order")
    if (not np.isfinite(real.values).all() or not np.isfinite(imag.values).all()
            or np.any(np.asarray(real.values).imag != 0) or np.any(np.asarray(imag.values).imag != 0)):
        raise ValueError("component Fourier outputs must be finite and real")
    attrs = real.attrs
    attrs.pop("tmdwf_projection", None)
    for key in ("source_operator", "zdir", "correlator_id"):
        attrs.pop(key, None)
    attrs.update({"observable": "quasi_tmdwf_before_hard_matching", "output_component": "both",
                  "component_combination": "native_FT(real_midpoint) + i*native_FT(imag_midpoint)",
                  "fourier_convention": "exp(+1i*(x-0.5)*P*z) on even complex midpoint amplitude",
                  "x_shift": 0.5, "phase_convention": "midpoint",
                  "component_input_phase_convention": "endpoint_from_even_midpoint",
                  "tail_model_note": "independent native light-light DA ansatz per midpoint component"})
    return EnsembleData(real.ensemble, real.resample, list(real.values + 1j * imag.values),
                        real.dims, real.coords, attrs=attrs, name="quasi_tmdwf")
