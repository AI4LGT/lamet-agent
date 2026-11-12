"""Named Gaussian prior overrides shared by numerical fit stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gvar as gv
import numpy as np


def validate_prior_overrides(overrides: Mapping[str, Any] | None) -> dict[str, dict[str, float]]:
    """Parse gvar strings independently of a model's names."""
    if overrides is None:
        return {}
    if not isinstance(overrides, Mapping):
        raise ValueError('priors must map parameter names to gvar strings such as "1.0(0.3)"')
    result = {}
    for name, value in overrides.items():
        if not isinstance(name, str) or not name:
            raise ValueError("prior parameter names must be nonempty strings")
        if not isinstance(value, str):
            raise ValueError(f'prior {name!r} must be a gvar string such as "1.0(0.3)"')
        try:
            gaussian = gv.gvar(value)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"prior {name!r} is not a valid scalar gvar string: {value!r}") from exc
        mean, sdev = float(gv.mean(gaussian)), float(gv.sdev(gaussian))
        if not np.isfinite(mean) or not np.isfinite(sdev) or sdev <= 0:
            raise ValueError(f"prior {name!r} requires a finite mean and positive finite sdev")
        result[name] = {"mean": mean, "sdev": sdev}
    return result


def valid_prior_overrides(value: Mapping[str, Any]) -> bool:
    """Predicate for stage contract Value rules."""
    try:
        validate_prior_overrides(value)
    except ValueError:
        return False
    return True


def merge_priors(defaults: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> gv.BufferDict:
    """Override model parameters, preserving missing defaults and ignoring extras.

    An override of an array parameter applies the same scalar mean and width
    independently to every element. Unmodified defaults retain their covariance.
    Model-specific scaling and coordinate transformations belong to callers.
    """
    authored = validate_prior_overrides(overrides)
    prior = gv.BufferDict()
    for name, default in defaults.items():
        if name not in authored:
            prior[name] = default
            continue
        mean, sdev = authored[name]["mean"], authored[name]["sdev"]
        shape = np.shape(gv.mean(default))
        prior[name] = gv.gvar(np.full(shape, mean), np.full(shape, sdev)) if shape else gv.gvar(mean, sdev)
    return prior


def prior_diagnostics(prior: Mapping[str, Any], overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return JSON-compatible effective priors and unused input parameter names."""
    return {
        "effective_priors": {
            name: {"mean": np.asarray(gv.mean(value)).tolist(), "sdev": np.asarray(gv.sdev(value)).tolist()}
            for name, value in prior.items()
        },
        "inactive_prior_overrides": sorted(set(overrides or {}) - set(prior)),
    }
