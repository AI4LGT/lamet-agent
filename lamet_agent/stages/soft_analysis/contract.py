"""Manifest contract for one momentum pair per Soft Analysis job."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import Depends, Recommends, Source, Value, stage_job_rules


def _positive(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value > 0


def _coupling(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _window(value: dict) -> bool:
    return (
        set(value) == {"start", "stop"}
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value.values())
        and 0 < value["start"] < value["stop"] < 1
    )


PARAM_RULES = (
    Depends("", "mu", physics="Matching scale in GeV."),
    Value("mu", (int, float), physics="Positive matching scale.", validator=_positive),
    Depends("", "alpha_s", physics="Explicit coupling at mu; no automatic running."),
    Value("alpha_s", (int, float), physics="Nonnegative finite coupling.", validator=_coupling),
    Depends("", "sign", physics="Wilson-line orientation, independent of the source operator."),
    Value("sign", Literal[-1, 1], physics="Select the plus or minus hard coefficient."),
    Depends("", "x_window", physics="Keep a common interior x grid without interpolation."),
    Value("x_window", dict, physics="Require 0 < start < stop < 1.", validator=_window),
    Recommends("", "convention", default="author", physics="Use the author-confirmed sqrt(pi) Meijer-G factor."),
    Value("convention", Literal["author", "printed"], physics="Printed is for formula comparison only."),
    Recommends("", "logarithm", default="expanded", physics="Expand the hard logarithm to one loop."),
    Value("logarithm", Literal["expanded", "principal"], physics="Principal reproduces the supplied Mathematica logs."),
    Recommends("", "dps", default=30, physics="Local mpmath precision for the hard coefficient."),
    Value("dps", int, physics="At least 20 decimal digits.", validator=lambda value: value >= 20),
    Recommends(
        "", "min_magnitude", default=1e-14, physics="Reject singular or numerically vanishing sample amplitudes."
    ),
    Value(
        "min_magnitude", (int, float), physics="Positive amplitude floor; no samples are dropped.", validator=_positive
    ),
)

INPUT_RULES = (
    Depends("", "quasi1", physics="Fourier output at the numerator momentum."),
    Source("quasi1", physics="One prior job id or external NetCDF file."),
    Depends("", "quasi2", physics="Fourier output at the denominator momentum."),
    Source("quasi2", physics="One prior job id or external NetCDF file with paired bootstrap samples."),
)

JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES)
CHECKS = ()
