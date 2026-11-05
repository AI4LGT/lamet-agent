"""Shared scalar Gaussian prior controls for all correlator fit models."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
from typing import Any

import gvar as gv
import numpy as np


def validate_prior_overrides(
    overrides: Mapping[str, Any] | None, *, allowed_names: Iterable[str] | None = None
) -> dict[str, dict[str, float]]:
    """Validate explicit scalar means/widths; model names are never guessed."""
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError("priors must map parameter names to {mean, sdev}")
    allowed = set(allowed_names) if allowed_names is not None else None
    result = {}
    for name, value in overrides.items():
        if not isinstance(name, str) or not name:
            raise ValueError("prior parameter names must be nonempty strings")
        if allowed is not None and name not in allowed:
            raise ValueError(f"unknown prior parameter {name!r}; allowed names: {sorted(allowed)}")
        if not isinstance(value, Mapping) or set(value) != {"mean", "sdev"}:
            raise ValueError(f"prior {name!r} requires exactly scalar mean and sdev")
        for field in ("mean", "sdev"):
            number = value[field]
            if isinstance(number, bool) or not isinstance(number, Real) or not np.isfinite(number):
                raise ValueError(f"prior {name!r} {field} must be a finite scalar")
        if value["sdev"] <= 0:
            raise ValueError(f"prior {name!r} sdev must be positive")
        result[name] = {field: float(value[field]) for field in ("mean", "sdev")}
    return result


def build_priors(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None = None,
    *,
    prior_width: float = 1.0,
    conditioning: Mapping[str, float] | None = None,
    allowed_names: Iterable[str] | None = None,
) -> gv.BufferDict:
    """Override raw-unit priors, scale widths once, and condition fit variables.

    ``conditioning`` uses the same parameter keys as ``defaults``. Linear
    variables multiply mean and width by the factor. For ``log(name)`` keys,
    the mean shifts by log(factor) and the width stays in logarithmic units.
    A declared model-union ``allowed_names`` permits inactive higher-state
    overrides; callers report these through :func:`prior_diagnostics`.
    """
    if isinstance(prior_width, bool) or not isinstance(prior_width, Real) or not np.isfinite(prior_width):
        raise ValueError("prior_width must be finite and positive")
    if prior_width <= 0:
        raise ValueError("prior_width must be finite and positive")
    authored = validate_prior_overrides(overrides, allowed_names=defaults if allowed_names is None else allowed_names)
    factors = conditioning or {}
    unknown_factors = set(factors) - set(defaults)
    if unknown_factors:
        raise ValueError(f"conditioning contains unknown prior parameters: {sorted(unknown_factors)}")
    prior = gv.BufferDict()
    for name, default in defaults.items():
        mean, sdev = gv.mean(default), gv.sdev(default)
        if np.ndim(mean) or np.ndim(sdev):
            raise ValueError(f"prior {name!r} must be scalar; use the model's individual parameter names")
        if name in authored:
            mean, sdev = authored[name]["mean"], authored[name]["sdev"]
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


def prior_diagnostics(prior: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Record the effective fit priors and every declared inactive override."""
    return {
        "effective_priors": {
            name: {"mean": float(gv.mean(value)), "sdev": float(gv.sdev(value))} for name, value in prior.items()
        },
        "inactive_prior_overrides": sorted(set(overrides or {}) - set(prior)),
    }


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
