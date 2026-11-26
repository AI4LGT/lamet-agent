"""Manifest contract for pair curves and grouped Collins-Soper kernel estimates."""

from __future__ import annotations

import math
from typing import Literal

from lamet_agent.contract import CheckContext, Depends, Issue, Provides, Recommends, Source, Value, stage_job_rules
from lamet_agent.stages._priors import valid_prior_overrides
from lamet_agent.stages.soft_analysis.physics import valid_slope_prior


def _positive(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value > 0


def _coupling(value: float) -> bool:
    return not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _window(value: dict | list) -> bool:
    if isinstance(value, list):
        if len(value) != 2:
            return False
        value = dict(zip(("start", "stop"), value))
    return (
        set(value) == {"start", "stop"}
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in value.values())
        and 0 < value["start"] < value["stop"] < 1
    )


def _pairs(value: list) -> bool:
    return bool(value) and all(
        isinstance(pair, list) and len(pair) == 2 and all(type(p) is int and p > 0 for p in pair)
        and pair[0] != pair[1] for pair in value
    ) and len({tuple(sorted(pair)) for pair in value}) == len(value)


PARAM_RULES = (
    Recommends("", "type", default="pair", physics="A pair curve or a b-dependent average over specified pairs."),
    Value("type", Literal["pair", "cs_kernel"], physics="Select the finite-momentum CS estimator."),
    Provides("", "pair", "type", physics="One momentum ratio at fixed transverse separation."),
    Provides("", "cs_kernel", "type", physics="Estimate the kernel from matched momentum-pair curves at each b."),
    Depends("cs_kernel", "momentum_pairs", physics="Specify the ordered lattice momentum indices used in each ratio."),
    Value("cs_kernel.momentum_pairs", list, validator=_pairs, physics="Unique pairs of distinct positive indices."),
    Recommends("", "estimator", default=None, physics="Select how a grouped job reduces its finite-momentum curves."),
    Recommends("cs_kernel", "estimator", default="equal_weight",
               physics="Reduce the paired x-dependent curves by an arithmetic mean or a correlated power fit."),
    Value("cs_kernel.estimator", Literal["equal_weight", "pca_power"],
          physics="The power fit estimates the common infinite-momentum intercept at each b."),
    Recommends("", "pca_variance_fraction", default=None,
               physics="Optional PCA threshold used when a grouped job selects the power estimator."),
    Provides("cs_kernel", "pca_power", "cs_kernel.estimator", physics="Leading finite-momentum power correction."),
    Recommends("cs_kernel.pca_power", "pca_variance_fraction", default=0.99,
               physics="Retain principal covariance modes up to the specified cumulative variance fraction."),
    Value("pca_variance_fraction", (int, float),
          validator=lambda value: not isinstance(value, bool) and 0 < value <= 1,
          physics="The cumulative variance fraction must lie in (0, 1]."),
    Recommends("", "priors", default=None,
               physics="Optional named Gaussian prior for the power correction slope A(x,bT), in GeV²."),
    Value("priors", dict, validator=lambda value: value.keys() <= {"A"} and valid_prior_overrides(value),
          physics='Use a named Gaussian in the fit-stage format, e.g. {"A": "0.0 +- 1.0"}.'),
    Recommends("", "slope_prior", default=None,
               physics="Optional fixed Gaussian prior on A(x,bT) for the grouped PCA power fit."),
    Value("slope_prior", dict, validator=valid_slope_prior,
          physics="Set finite mean_gev2 and positive finite sigma_gev2, common to real and imaginary slopes."),
    Recommends("", "kernel_id", default="tmdwf_gi_unlo", physics="Use the GI uNLO hard matching coefficient."),
    Value("kernel_id", Literal["tmdwf_gi_unlo"], physics="GI quasi-TMDWF hard coefficient."),
    Depends("", "mu", physics="Matching scale in GeV."),
    Value("mu", (int, float), physics="Positive matching scale.", validator=_positive),
    Depends("", "alpha_s", physics="Explicit coupling at mu; no automatic running."),
    Value("alpha_s", (int, float), physics="Nonnegative finite coupling.", validator=_coupling),
    Depends("", "sign", physics="Wilson-line orientation, independent of the source operator."),
    Value("sign", Literal[-1, 1], physics="Select the plus or minus hard coefficient."),
    Depends("", "x_window", physics="Keep a common interior x grid without interpolation."),
    Value("x_window", (dict, list), physics="Require 0 < start < stop < 1.", validator=_window),
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

JOB_INPUTS = (
    Depends("pair", "inputs", physics="Use explicit numerator and denominator sources."),
    Depends("pair.inputs", "quasi1", physics="Fourier output at the numerator momentum."),
    Source("pair.inputs.quasi1", physics="One prior job id or external NetCDF file."),
    Depends("pair.inputs", "quasi2", physics="Fourier output at the denominator momentum."),
    Source("pair.inputs.quasi2", physics="One prior job id or external NetCDF file with paired bootstrap samples."),
    Depends("cs_kernel", "inputs", physics="Group quasi distributions by b and lattice momentum index."),
    Depends("cs_kernel.inputs", "quasi", physics="All selected GI quasi distributions."),
    Source("cs_kernel.inputs.quasi", allow_list=True, physics="Prior jobs or NetCDF files for the selected momenta."),
)

INPUT_RULES = ()
JOB_RULES = stage_job_rules(PARAM_RULES, INPUT_RULES, JOB_INPUTS, allow_empty=True)


def check_power_pairs(context: CheckContext) -> Issue | None:
    params = context.params
    if params.get("type", "pair") != "cs_kernel":
        for key in ("pca_variance_fraction", "priors", "slope_prior", "estimator"):
            if key in params:
                return Issue(key, "requires a grouped cs_kernel job", "A single pair gives a finite-momentum curve.")
    if "priors" in params and "slope_prior" in params:
        return Issue("priors", "specify the slope prior only once", "Both fields configure the same A(x,bT).")
    if (
        params.get("estimator") == "pca_power"
        and isinstance(params.get("momentum_pairs"), list)
        and len(params["momentum_pairs"]) < 2
    ):
        return Issue(
            "momentum_pairs",
            "pca_power requires at least two distinct momentum pairs",
            "Separating an intercept from a power correction requires different pair kinematics.",
        )
    return None


CHECKS = (check_power_pairs,)
