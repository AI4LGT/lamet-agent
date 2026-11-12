"""Correlator-specific prior scaling and model controls."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any

import gvar as gv
import numpy as np

from lamet_agent.stages._priors import merge_priors


def build_priors(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    *,
    prior_width: float = 1.0,
    conditioning: Mapping[str, float] | None = None,
) -> gv.BufferDict:
    """Override raw-unit priors, scale widths once, and condition fit variables.

    ``conditioning`` uses the same parameter keys as ``defaults``. Linear
    variables multiply mean and width by the factor. For ``log(name)`` keys,
    the mean shifts by log(factor) and the width stays in logarithmic units.
    Extra overrides are ignored by the shared merge operation.
    """
    if isinstance(prior_width, bool) or not isinstance(prior_width, Real) or not np.isfinite(prior_width):
        raise ValueError("prior_width must be finite and positive")
    if prior_width <= 0:
        raise ValueError("prior_width must be finite and positive")
    merged = merge_priors(defaults, overrides)
    factors = conditioning or {}
    unknown_factors = set(factors) - set(defaults)
    if unknown_factors:
        raise ValueError(f"conditioning contains unknown prior parameters: {sorted(unknown_factors)}")
    prior = gv.BufferDict()
    for name, default in merged.items():
        mean, sdev = gv.mean(default), gv.sdev(default)
        if np.ndim(mean) or np.ndim(sdev):
            raise ValueError(f"prior {name!r} must be scalar; use the model's individual parameter names")
        if not np.isfinite(mean) or not np.isfinite(sdev) or sdev <= 0:
            raise ValueError(f"prior {name!r} must have a finite mean and positive finite sdev")
        factor = factors.get(name, 1.0)
        if isinstance(factor, bool) or not isinstance(factor, Real) or not np.isfinite(factor) or factor <= 0:
            raise ValueError(f"conditioning factor for {name!r} must be finite and positive")
        sdev *= prior_width
        if name.startswith("log(") and name.endswith(")"):
            mean += np.log(factor)
        else:
            mean *= factor
            sdev *= factor
        prior[name] = gv.gvar(float(mean), float(sdev))
    return prior


def sample_prior_scale(mode: str, posterior_prior_error_scale: float) -> float | None:
    """Select original priors or broadened center posteriors, independently of width scans."""
    if mode not in {"original", "center_posterior"}:
        raise ValueError("sample_prior_mode must be 'original' or 'center_posterior'")
    if mode == "original":
        return None
    if (
        isinstance(posterior_prior_error_scale, bool)
        or not isinstance(posterior_prior_error_scale, Real)
        or not np.isfinite(posterior_prior_error_scale)
        or posterior_prior_error_scale <= 0
    ):
        raise ValueError("posterior_prior_error_scale must be finite and positive")
    return float(posterior_prior_error_scale)


def declared_prior_names(params: Mapping[str, Any]) -> set[str]:
    """Names in the authored state-count union, before data-dependent fitting."""
    from ._scope import parse_fit_scope

    scope = parse_fit_scope(params["fit_scope"])
    counts = {atom: max(values) for atom, values in params["nstate"].items()}
    if scope.is_spectrum:
        return {f"{prefix}{state}" for prefix in ("E", "z") for state in range(counts["2pt"])}
    if scope.is_form_factor:
        return {"ff"} | ({"log(dE)", "c1", "c2"} if counts["ff"] > 1 else set())
    output = params.get("qda_output", "ratio")
    if output in {"tmdwf", "amplitude"}:
        names = {"log(E0)"} | {f"log(dE{state})" for state in range(1, counts["qda"])}
        components = ("re", "im") if params["component"] == "both" else (params["component"],)
        if output == "tmdwf":
            names |= {f"log(overlap{state})" for state in range(counts["qda"])}
            names |= {f"phi{state}_{component}" for state in range(counts["qda"]) for component in components}
        else:
            names |= {f"amplitude_{component}" for component in components}
        return names
    from .physics import matrix_element_prior

    names = set()
    for denominator in ("external_2pt", "qda_z0") if scope.is_qda else ("external_2pt",):
        names.update(
            matrix_element_prior(
                counts,
                form="Breit" if scope.is_qda else params["fitting_form"],
                scope=scope.atoms,
                components=("re", "im"),
                width_scale=1.0,
                denominator_kind=denominator,
            )
        )
    return names
