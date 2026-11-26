"""Manifest rules for the residual gauge-invariant TMD factor."""

from typing import Literal

import numpy as np

from lamet_agent.contract import Depends, Issue, Provides, Recommends, Source, Value
from lamet_agent.kernels.tmd.zo import validate_settings

PHYSICS = "arXiv:2412.19988 (5)-(8): average short-distance ratios after Wilson-loop division."

PARAM_RULES = (
    Provides("", "tmd_zo", "strategy", physics=PHYSICS),
    Depends("tmd_zo", "type", physics=PHYSICS),
    Value("tmd_zo.type", Literal["fit", "apply"], physics=PHYSICS),
    Provides("tmd_zo", "fit", "tmd_zo.type", physics=PHYSICS),
    Provides("tmd_zo", "apply", "tmd_zo.type", physics=PHYSICS),
    Recommends("tmd_zo", "scheme", default="msbar", physics=PHYSICS),
    Value("tmd_zo.scheme", Literal["msbar"], physics=PHYSICS),
    Depends("tmd_zo", "mu", physics=PHYSICS),
    Value("tmd_zo.mu", (int, float), validator=lambda x: np.isfinite(x) and x > 0, physics=PHYSICS),
    Depends("tmd_zo.fit", "points_fm", physics=PHYSICS),
    Value("tmd_zo.fit.points_fm", list, physics=PHYSICS),
    Recommends("tmd_zo.fit", "order", default="nlo_rgr", physics=PHYSICS),
    Value("tmd_zo.fit.order", Literal["nlo", "nlo_rgr"], physics=PHYSICS),
    Depends("tmd_zo.fit", "alpha_s", physics="Supply alpha_s(mu) in the fixed-flavor scheme."),
    Value("tmd_zo.fit.alpha_s", (int, float), physics=PHYSICS),
    Depends("tmd_zo.fit", "n_f", physics="Fixed active-flavor count, shared by running and anomalous dimension."),
    Value("tmd_zo.fit.n_f", Literal[3, 4, 5], physics=PHYSICS),
    Recommends("tmd_zo.fit", "min_scale_gev", default=0.7, physics="Reject low scales; do not freeze the coupling."),
    Value("tmd_zo.fit.min_scale_gev", (int, float), physics=PHYSICS),
    Recommends("tmd_zo.apply", "scale_factor", default=1.0, physics="Select one stored scale variant coherently."),
    Value("tmd_zo.apply.scale_factor", Literal[0.8, 1.0, 1.2], physics=PHYSICS),
)

INPUT_RULES = (
    Depends("tmd_zo.fit", "inputs", physics=PHYSICS),
    Depends("tmd_zo.fit.inputs", "reference", physics=PHYSICS),
    Source("tmd_zo.fit.inputs.reference", physics=PHYSICS),
    Depends("tmd_zo.apply", "inputs", physics=PHYSICS),
    Depends("tmd_zo.apply.inputs", "target", physics=PHYSICS),
    Depends("tmd_zo.apply.inputs", "zO", physics=PHYSICS),
    Source("tmd_zo.apply.inputs.target", physics=PHYSICS),
    Source("tmd_zo.apply.inputs.zO", physics=PHYSICS),
)


def check(context):
    p = context.params
    if p.get("strategy") != "tmd_zo":
        return None
    if p.get("normalization") is not False:
        return Issue("normalization", "must be false for residual Z_O extraction/application", PHYSICS)
    expected = {"reference"} if p.get("type") == "fit" else {"target", "zO"}
    if set(context.inputs) != expected:
        return Issue("inputs", f"must provide exactly {sorted(expected)}", PHYSICS)
    if any(
        not isinstance(s, str) and not (isinstance(s, dict) and set(s) == {"file"} and isinstance(s["file"], str))
        for s in context.inputs.values()
    ):
        return Issue("inputs", "each input must be one job or NetCDF file", PHYSICS)
    if p.get("type") == "fit":
        try:
            validate_settings(**{k: p[k] for k in ("mu", "alpha_s", "n_f", "order", "min_scale_gev")})
            points = np.asarray(p["points_fm"], float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.all(np.isfinite(points)):
                raise ValueError("points_fm requires at least three finite [z,b] pairs")
            if (
                np.any(points[:, 1] <= 0)
                or len(np.unique(points, axis=0)) != len(points)
                or any(len(np.unique(points[:, i])) < 2 for i in (0, 1))
            ):
                raise ValueError("points must be unique, b > 0, and span both coordinates")
        except KeyError:
            return None  # Required-field rules supply the specific missing-field issue.
        except (TypeError, ValueError) as exc:
            return Issue("points_fm", str(exc), PHYSICS)
    return None
