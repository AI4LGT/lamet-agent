"""Manifest contract for the two dependent soft-analysis operations."""

from dataclasses import replace
from typing import Literal

import numpy as np

from lamet_agent.contract import (
    CheckContext, Depends, Issue, List, Provides, Recommends, Source, Value, stage_job_rules,
)
from lamet_agent.stages._priors import validate_prior_overrides
from lamet_agent.kernels.tmd.coefficients import validate_settings as validate_coefficient_settings


def _positive(value):
    return np.isfinite(value) and value > 0


def _window(value):
    return (len(value) == 2 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                  and np.isfinite(v) for v in value) and 0 < value[0] < value[1] < 1)


PARAM_RULES = (
    Depends("", "type", physics="Soft analysis first extracts the CS kernel and then the intrinsic soft function."),
    Value("type", Literal["cs_kernel", "intrinsic_soft"], physics="Each operation publishes a bT-dependent function."),
    Recommends("", "order", default="nlo", physics="Fixed-order NLO or NLL with an infrared prescription."),
    Value("order", Literal["nlo", "nll"], physics="NLL resums the hard and Sudakov coefficients."),
    Recommends("", "infrared_policy", default="error", physics="NLL rejects pole crossings by default."),
    Value("infrared_policy", Literal["error", "freeze", "reference_quad"],
          physics="freeze floors the running scale; reference_quad records historical pole-crossing warnings."),
    Recommends("", "freeze_scale_gev", default=None, physics="An explicit scale is needed only for freezing."),
    Value("freeze_scale_gev", (int, float), physics="Positive scale required only for freeze.", validator=_positive),
    Recommends("", "mu", default=2.0, physics="Renormalization scale in GeV."),
    Value("mu", (int, float), physics="The perturbative scale is finite and positive.", validator=_positive),
    Recommends("", "alpha_s", default=0.293, physics="alpha_s(mu), fixed for NLO and anchoring NLL running."),
    Value("alpha_s", (int, float), physics="Fixed coupling is finite and positive.", validator=_positive),
    Provides("", "cs_kernel", "type", physics="Momentum ratios cancel the intrinsic soft function."),
    Recommends("cs_kernel", "momentum_pairs", default=[[8, 9], [9, 10]],
               physics="Pairs identify upstream momentum_index attrs in increasing momentum order."),
    Value("cs_kernel.momentum_pairs", list, physics="Explicit pairs of positive integer momentum indices."),
    Recommends("cs_kernel", "x_window", default=[0.34, 0.66],
               physics="Average central x points where power corrections are smaller."),
    Value("cs_kernel.x_window", list, physics="The selected CS interval lies inside (0,1).", validator=_window),
    Provides("", "intrinsic_soft", "type", physics="Divide the form factor by a reduced-WF double integral."),
    Recommends("intrinsic_soft", "evolution_cs", default="physical",
               physics="physical uses the hard-corrected CS kernel; quasi_ratio uses the uncorrected reference ratio."),
    Value("intrinsic_soft.evolution_cs", Literal["physical", "quasi_ratio"],
          physics="The uncorrected ratio is reconstructed from the same CS replicas and saved hard correction."),
    Recommends("intrinsic_soft", "sudakov_momentum", default="form_factor",
               physics="Use the FF momentum in HF, or the source WF momentum for reference reproduction."),
    Value("intrinsic_soft.sudakov_momentum", Literal["form_factor", "quasi"],
          physics="This scale is independent of the FF target momentum used by rapidity evolution."),
    Recommends("intrinsic_soft", "integration_cut", default=0.05,
               physics="Both momentum fractions are integrated between cut and 1-cut."),
    Value("intrinsic_soft.integration_cut", (int, float), physics="Cut excludes endpoint singularities.",
          validator=lambda v: np.isfinite(v) and 0 < v < 0.5),
    Recommends("intrinsic_soft", "endpoint_region", default=[0.3, 0.7],
               physics="Measured reduced WF is retained inside; fitted powers supply endpoints."),
    Value("intrinsic_soft.endpoint_region", list, physics="Central interval is inside (0,1).", validator=_window),
    Recommends("intrinsic_soft", "endpoint_fit_windows", default=[[0.3, 0.45], [0.55, 0.7]],
               physics="Fit c[x(1-x)]^n to the two shoulders of the reduced wave function."),
    Value("intrinsic_soft.endpoint_fit_windows", list, physics="Nonempty list of intervals inside (0,1).",
          validator=lambda v: bool(v) and all(isinstance(w, list) and _window(w) for w in v)),
    Recommends("intrinsic_soft", "endpoint_fit_points_per_window", default=None,
               physics="Omission preserves the native quasi grid for endpoint fits."),
    Value("intrinsic_soft.endpoint_fit_points_per_window", int,
          physics="Cubic interpolation of raw quasi replicas to this many shoulder points before reduction.",
          validator=lambda v: v >= 4),
    Recommends("intrinsic_soft", "endpoint_central_points", default=None,
               physics="Omission preserves the native central interpolation grid."),
    Value("intrinsic_soft.endpoint_central_points", int,
          physics="Optional cubic central interpolation grid evaluated before hard-factor division and evolution.",
          validator=lambda v: v >= 4),
    Recommends("intrinsic_soft", "endpoint_covariance", default="full",
               physics="reference_cs_diagonal changes only the fit metric; full uses the replica covariance."),
    Value("intrinsic_soft.endpoint_covariance", Literal["full", "reference_cs_diagonal"],
          physics="The reference metric retains quasi covariance but treats CS uncertainty as independent between x; "
                  "true shared replicas still determine the propagated result."),
    Recommends("intrinsic_soft", "priors", default={},
               physics="Optional scalar Gaussian priors use gvar strings, as in the other fit stages. "
                       "An empty mapping preserves the unregularized endpoint fit."),
    Value("intrinsic_soft.priors", dict,
          physics="Named c and log(n) priors for c[x(1-x)]^n, in physical amplitude and natural-log exponent "
                  "units before numerical conditioning; the same priors apply to every replica."),
    Recommends("intrinsic_soft", "quadrature_order", default=24,
               physics="Gauss-Legendre nodes per interval, doubled for a convergence check."),
    Value("intrinsic_soft.quadrature_order", int, physics="At least eight nodes per interval.",
          validator=lambda v: v >= 8),
    Recommends("intrinsic_soft", "quadrature_rtol", default=0.005,
               physics="Maximum relative change for every replica when quadrature order doubles."),
    Value("intrinsic_soft.quadrature_rtol", (int, float), physics="Finite positive convergence tolerance.",
          validator=_positive),
)

INPUT_RULES = (
    Depends("", "quasi", physics="Per-b quasi-TMDWF artifacts at the selected momenta."),
    Source("quasi", allow_list=True, physics="List of preceding Fourier jobs or NetCDF artifacts."),
)

JOB_CONDITIONAL_RULES = (
    Depends("intrinsic_soft", "inputs", physics="Intrinsic-soft extraction also consumes FF and CS kernel."),
    Depends("intrinsic_soft.inputs", "form_factor", physics="Normalized form factors at each bT."),
    Source("intrinsic_soft.inputs.form_factor", allow_list=True,
           physics="FF correlator outputs carrying momentum and bT coordinates."),
    Depends("intrinsic_soft.inputs", "cs_kernel", physics="Rapidity evolution aligns qWF and FF momenta."),
    Source("intrinsic_soft.inputs.cs_kernel", physics="Prior CS-kernel job or compatible NetCDF artifact."),
)


def check_settings(context: CheckContext):
    params = context.params
    try:
        validate_coefficient_settings(**{name: params[name] for name in (
            "order", "mu", "alpha_s", "infrared_policy", "freeze_scale_gev") if name in params})
    except ValueError as exc:
        return Issue("infrared_policy", str(exc), "Coefficients require a valid infrared prescription.")
    try:
        validate_prior_overrides(params.get("priors"))
    except ValueError as exc:
        return Issue("priors", str(exc), "Gaussian endpoint priors must name the physical model parameters.")
    required = {"quasi"} if params.get("type") == "cs_kernel" else {"quasi", "form_factor", "cs_kernel"}
    if set(context.inputs) != required:
        return Issue("inputs", f"must provide exactly {sorted(required)}", "Inputs define the physical DAG.")
    if params.get("type") == "cs_kernel":
        pairs = params.get("momentum_pairs", [])
        if (not pairs or any(not isinstance(p, list) or len(p) != 2 or
                             any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in p) or
                             p[0] >= p[1] for p in pairs)):
            return Issue("momentum_pairs", "must contain increasing pairs of positive integer indices",
                         "Each CS ratio needs distinct momenta.")
        if len({tuple(p) for p in pairs}) != len(pairs):
            return Issue("momentum_pairs", "must not repeat a pair", "Duplicate pairs change the average.")
    elif "endpoint_region" in params and "integration_cut" in params:
        low, high = params["endpoint_region"]
        cut = params["integration_cut"]
        if not cut < low < high < 1 - cut:
            return Issue("endpoint_region", "must lie inside the integration bounds",
                         "The integral has two endpoint branches and one measured central branch.")
    return None


JOB_RULES = tuple(
    replace(rule, validator=None) if isinstance(rule, List) and rule.path == "jobs" else rule
    for rule in stage_job_rules(PARAM_RULES, INPUT_RULES, JOB_CONDITIONAL_RULES)
)
CHECKS = (check_settings,)
