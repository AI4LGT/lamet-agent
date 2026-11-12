"""Tree-boundary NLL coefficients of arXiv:2504.04625v5, Appendix A.

The running coupling uses the explicitly identified reference NNLO analytic
approximation. Its infrared pole is never silently crossed by the controlled
policies. ``reference_quad`` is a diagnostic reproduction of the historical
QUADPACK calculation, not a prescription for a convergent infrared integral.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq

from .cg_nlo import _controls

CF = 4.0 / 3.0
BETA0, BETA1, BETA2 = 9.0, 64.0, 643.8333333333334
GAMMA0 = 2 * CF
GAMMA1 = 2 * CF * ((67 / 9 - np.pi**2 / 3) * 3 - 20 / 9 * 0.5 * 3)
NONCUSP0 = -6 * CF
_REFERENCE_DIAGNOSTICS: dict[tuple, dict] = {}


def _denominator(scale, mu, alpha_s):
    a = alpha_s / (4 * np.pi)
    t = 1 + a * BETA0 * np.log((np.asarray(scale) / mu)**2)
    return t + a * BETA1 / BETA0 * np.log(t) + a**2 * (
        BETA2 / BETA0 * (1 - 1 / t)
        + BETA1**2 / BETA0**2 * (np.log(t) / t + 1 / t - 1)
    )


@lru_cache(maxsize=64)
def coupling_pole_gev(*, mu: float = 2.0, alpha_s: float = 0.293) -> float:
    """First denominator zero reached by running down from ``mu``."""
    _controls(1.0, mu, alpha_s)
    if alpha_s == 0:
        return 0.0
    a = alpha_s / (4 * np.pi)
    if 1 / (2 * a * BETA0) > 2 * abs(np.log(np.finfo(float).tiny)):
        return 0.0  # The pole scale is smaller than representable positive floats.
    # Solve in t rather than scale, so arbitrarily weak couplings do not
    # underflow the logarithm's lower endpoint during root finding.
    def denominator_t(t):
        return t + a * BETA1 / BETA0 * np.log(t) + a*a * (
            BETA2 / BETA0 * (1 - 1/t)
            + BETA1**2 / BETA0**2 * (np.log(t)/t + 1/t - 1)
        )
    root = brentq(denominator_t, 1e-100, 1.0, xtol=1e-14)
    return float(mu * np.exp((root - 1) / (2 * a * BETA0)))


def validate_controls(*, mu=2.0, alpha_s=0.293, infrared_policy="error", freeze_scale_gev=None):
    _controls(1.0, mu, alpha_s)
    if infrared_policy not in {"error", "freeze", "reference_quad"}:
        raise ValueError("infrared_policy must be 'error', 'freeze', or 'reference_quad'")
    if infrared_policy == "freeze":
        if (isinstance(freeze_scale_gev, bool) or not isinstance(freeze_scale_gev, (int, float))
                or not np.isfinite(freeze_scale_gev) or freeze_scale_gev <= 0):
            raise ValueError("infrared_policy='freeze' requires finite positive freeze_scale_gev")
        pole = coupling_pole_gev(mu=mu, alpha_s=alpha_s)
        if freeze_scale_gev <= pole or freeze_scale_gev > mu:
            raise ValueError(f"freeze_scale_gev must lie above pole={pole:.12g} GeV and at or below mu={mu:g}")
    elif freeze_scale_gev is not None:
        raise ValueError("freeze_scale_gev is only meaningful with infrared_policy='freeze'")


def running_coupling(scale_gev, *, mu=2.0, alpha_s=0.293,
                     infrared_policy="error", freeze_scale_gev=None):
    """Reference NNLO running anchored at the explicitly supplied alpha_s(mu).

    Even ``reference_quad`` cannot request a negative coupling here; its
    deliberately historical continuation is confined to the Sudakov integrator.
    """
    validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    scale = np.asarray(scale_gev, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("running coupling scale must be finite and positive")
    if alpha_s == 0:
        return np.zeros_like(scale)
    if infrared_policy == "freeze":
        scale = np.maximum(scale, freeze_scale_gev)
    pole = coupling_pole_gev(mu=mu, alpha_s=alpha_s)
    if np.any(scale <= pole):
        raise ValueError(
            f"running coupling scale {float(np.min(scale)):.12g} GeV reaches pole={pole:.12g} GeV; "
            "select an explicit freeze prescription or a narrower physical x interval"
        )
    value = alpha_s / _denominator(scale, mu, alpha_s)
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("running coupling is nonfinite or nonpositive at the requested scales")
    return value


def _hard_exponent(a0, alpha_s):
    r = alpha_s / a0
    log_r = np.log(r)
    d = r - 1
    small = np.abs(d) < 1e-3
    # Both combinations start at d**2. Expanding only close to r=1
    # avoids catastrophic cancellation in the weak-coupling/boundary limit.
    first = np.where(small, sum((-1)**(n+1) * (n-1)/n * d**n for n in range(2, 9)),
                     1 - 1/r - log_r)
    second = np.where(small, sum((-1)**(n+1) / n * d**n for n in range(2, 9)),
                      1 - r + log_r)
    term1 = 4 * np.pi / a0 * first
    term2 = (GAMMA1/GAMMA0 - BETA1/BETA0) * second
    term3 = BETA1 / (2 * BETA0) * log_r**2
    k_cusp = -GAMMA0 / (4 * BETA0**2) * (term1 + term2 + term3)
    k_gamma = -NONCUSP0 / (2 * BETA0) * log_r
    return -2 * k_cusp + k_gamma


def tmd_coefficient(x, *, momentum_gev, mu=2.0, alpha_s=0.293,
                    infrared_policy="error", freeze_scale_gev=None):
    """Single-leg C_TMD; H_phi=C(x)C(1-x), H_f=C(x)^2.

    ``freeze`` explicitly floors the natural/evolution scale, including its
    logarithms and tree boundary. This model-dependent infrared completion
    plateaus below the floor; physical hadron momenta are never changed.
    """
    _controls(momentum_gev, mu, alpha_s)
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("TMD coefficient requires finite x > 0")
    q = 2 * x * momentum_gev
    a0 = running_coupling(q, mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                          freeze_scale_gev=freeze_scale_gev)
    if alpha_s == 0:
        return np.ones_like(x)
    exponent = _hard_exponent(a0, alpha_s)
    result = np.exp(exponent)
    if np.any(~np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("NLL TMD coefficient is nonfinite or nonpositive at the requested coordinates")
    return result


def hard_factor(x, *, observable, **kwargs):
    x = np.asarray(x, dtype=float)
    coefficient = tmd_coefficient(x, **kwargs)
    if observable == "tmdpdf":
        return coefficient**2
    if observable == "tmdwf":
        if np.any(x >= 1):
            raise ValueError("TMDWF hard factor requires 0 < x < 1")
        return coefficient * tmd_coefficient(1 - x, **kwargs)
    raise ValueError("observable must be 'tmdwf' or 'tmdpdf'")


@lru_cache(maxsize=65536)
def _sudakov_scalar(q, mu, alpha_s, infrared_policy, freeze_scale_gev):
    if infrared_policy == "freeze":
        q = max(q, freeze_scale_gev)
    if alpha_s == 0 or q == mu:
        return 1.0
    if infrared_policy == "reference_quad":
        # Deliberately exactly the original argument, integration variable and
        # QUADPACK tolerance. Do not regularize a pole without saying so.
        if 1 + alpha_s/(4*np.pi) * BETA0 * np.log((q/mu)**2) <= 0:
            raise ValueError(f"reference_quad Q={q:g} GeV lies below the coupling logarithm branch")
        def integrand(u):
            coupling = alpha_s / _denominator(u, mu, alpha_s)
            return -CF * coupling / (4 * np.pi) * (4 * np.log(u**2/q**2) + 6) / u
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            value, error = quad(integrand, q, mu, epsrel=1e-3)
        pole = coupling_pole_gev(mu=mu, alpha_s=alpha_s)
        _REFERENCE_DIAGNOSTICS[(q, mu, alpha_s)] = {
            "q_gev": q, "mu_gev": mu, "alpha_s": alpha_s,
            "log_coefficient": float(value), "quad_estimated_absolute_error": float(error),
            "crosses_coupling_pole": min(q, mu) <= pole <= max(q, mu),
            "warnings": [str(item.message) for item in caught],
            "controlled_integral": min(q, mu) > pole and not caught,
        }
    else:
        # Preflight the full interval; quad must never decide whether it happens
        # to sample the infrared pole. The log-scale integral is well conditioned.
        running_coupling(min(q, mu), mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                         freeze_scale_gev=freeze_scale_gev)
        def integrand_log(t):
            coupling = running_coupling(
                np.exp(t), mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                freeze_scale_gev=freeze_scale_gev,
            )
            return -CF * coupling / (4*np.pi) * (8 * (t - np.log(q)) + 6)
        lower, upper = np.log(q), np.log(mu)
        value, error = quad(integrand_log, lower, upper, epsabs=1e-10, epsrel=1e-10)
        if error > 1e-8 * max(1.0, abs(value)):
            raise ValueError(f"Sudakov quadrature failed at Q={q:g} GeV: estimated error={error:g}")
    result = float(np.exp(value))
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"Sudakov coefficient nonfinite/nonpositive at Q={q:g} GeV, log coefficient={value:g}")
    return result


def sudakov_factor(q_gev, *, mu=2.0, alpha_s=0.293, infrared_policy="error", freeze_scale_gev=None):
    validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    q = np.asarray(q_gev, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q <= 0):
        raise ValueError("Sudakov scale Q must be finite and positive")
    unique, inverse = np.unique(q, return_inverse=True)
    values = [_sudakov_scalar(float(v), float(mu), float(alpha_s), infrared_policy, freeze_scale_gev)
              for v in unique]
    return np.asarray(values)[inverse].reshape(q.shape)


def form_factor_kernel(x1, x2, *, momentum_gev, **kwargs):
    _controls(momentum_gev, kwargs.get("mu", 2.0), kwargs.get("alpha_s", 0.293))
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    if any(np.any(~np.isfinite(x)) or np.any((x <= 0) | (x >= 1)) for x in (x1, x2)):
        raise ValueError("form-factor kernel requires 0 < x1,x2 < 1")
    q = 2 * momentum_gev * np.sqrt(x1*x2)
    qbar = 2 * momentum_gev * np.sqrt((1-x1)*(1-x2))
    return sudakov_factor(q, **kwargs) * sudakov_factor(qbar, **kwargs)


def reset_reference_quad_diagnostics():
    """Start an auditable reproduction calculation, including cached evaluations."""
    _REFERENCE_DIAGNOSTICS.clear()
    _sudakov_scalar.cache_clear()


def reference_quad_diagnostics():
    """Return all unique historical quadrature evaluations, including warnings."""
    rows = [dict(row) for _, row in sorted(_REFERENCE_DIAGNOSTICS.items())]
    return {
        "policy": "reference_quad", "evaluations": rows, "evaluation_count": len(rows),
        "pole_crossing_count": sum(row["crosses_coupling_pole"] for row in rows),
        "warning_count": sum(bool(row["warnings"]) for row in rows),
        "controlled_integral": all(row["controlled_integral"] for row in rows),
        "disclaimer": "Pole-crossing historical QUADPACK values do not define a controlled convergent integral.",
    }


def scale_diagnostics(scale_gev, *, mu=2.0, alpha_s=0.293,
                      infrared_policy="error", freeze_scale_gev=None):
    """Describe actual node scales without concealing a floor or pole crossing."""
    validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    scale = np.asarray(scale_gev, dtype=float)
    if not scale.size or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scale diagnostics require nonempty finite positive scales")
    effective = np.maximum(scale, freeze_scale_gev) if infrared_policy == "freeze" else scale
    return {
        "node_count": int(scale.size), "minimum_scale_gev": float(scale.min()),
        "maximum_scale_gev": float(scale.max()),
        "minimum_effective_scale_gev": float(effective.min()),
        "floored_node_count": int(np.count_nonzero(effective > scale)),
        "below_pole_node_count": int(np.count_nonzero(scale <= coupling_pole_gev(mu=mu, alpha_s=alpha_s))),
        "infrared_policy": infrared_policy,
    }
