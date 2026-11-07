"""Authored coordinate-dependent time windows for native axial fits."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def validate_qda_windows(value: Any) -> dict[float, dict[str, int]]:
    """Validate JSON coordinate keys without silently aliasing 2 and 2.0."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("qda_windows_by_z must map numeric coordinate strings to time windows")
    parsed: dict[float, dict[str, int]] = {}
    for key, window in value.items():
        try:
            coordinate = float(key)
        except (TypeError, ValueError) as exc:
            raise ValueError("qda_windows_by_z keys must be finite numeric coordinate strings") from exc
        if not isinstance(key, str) or not np.isfinite(coordinate):
            raise ValueError("qda_windows_by_z keys must be finite numeric coordinate strings")
        if any(abs(coordinate - other) <= 1e-12 for other in parsed):
            raise ValueError("qda_windows_by_z contains duplicate numeric coordinates")
        if (
            not isinstance(window, Mapping)
            or set(window) != {"tmin", "tmax"}
            or any(isinstance(v, bool) or not isinstance(v, int) for v in window.values())
            or not 0 <= window["tmin"] < window["tmax"]
        ):
            raise ValueError(f"qda_windows_by_z[{key!r}] requires integer 0 <= tmin < tmax")
        parsed[coordinate] = dict(window)
    return parsed


def resolve_qda_windows(
    z: Sequence[float], default: Mapping[str, int], overrides: Mapping[str, Any] | None,
) -> list[dict[str, int]]:
    """Resolve every coordinate once; unknown coordinates are authoring errors."""
    coordinates = np.asarray(z, dtype=float)
    windows = [dict(default) for _ in coordinates]
    for coordinate, window in validate_qda_windows(overrides).items():
        matches = np.flatnonzero(np.isclose(coordinates, coordinate, rtol=0, atol=1e-12))
        if len(matches) != 1:
            raise ValueError(f"qda_windows_by_z coordinate {coordinate:g} must name one available z")
        windows[int(matches[0])] = window
    return windows


def qda_time_masks(
    times: Sequence[float], z: Sequence[float], default: Mapping[str, int], overrides: Mapping[str, Any] | None,
) -> np.ndarray:
    """Return the exact (t,z) domain, shared by fitting and finite preflight."""
    time_values = np.asarray(times, dtype=float)
    return np.column_stack([
        (time_values >= window["tmin"]) & (time_values < window["tmax"])
        for window in resolve_qda_windows(z, default, overrides)
    ])
