from __future__ import annotations

import numpy as np


def _controls(momentum_gev: float, mu: float, alpha_s: float) -> None:
    for name, value in (("momentum_gev", momentum_gev), ("mu", mu), ("alpha_s", alpha_s)):
        if isinstance(value, bool) or not np.isfinite(value) or value < 0 or (value == 0 and name != "alpha_s"):
            raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'alpha_s' else 'positive'}")


def coefficient_correction(x, *, momentum_gev: float, mu: float = 2.0, alpha_s: float = 0.293):
    _controls(momentum_gev, mu, alpha_s)
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any(x <= 0):
        raise ValueError("TMD coefficient requires finite x > 0")
    logarithm = np.log(mu**2 / (2 * x * momentum_gev)**2)
    return alpha_s * (4 / 3) / (4 * np.pi) * (
        -0.5 * logarithm**2 - 3 * logarithm - 12 + 7 * np.pi**2 / 12
    )


def hard_factor(x, *, observable: str, momentum_gev: float, mu: float = 2.0, alpha_s: float = 0.293):
    x = np.asarray(x, dtype=float)
    correction = coefficient_correction(x, momentum_gev=momentum_gev, mu=mu, alpha_s=alpha_s)
    if observable == "tmdpdf":
        return 1 + 2 * correction
    if observable == "tmdwf":
        if np.any(x >= 1):
            raise ValueError("TMDWF hard factor requires 0 < x < 1")
        return 1 + correction + coefficient_correction(
            1 - x, momentum_gev=momentum_gev, mu=mu, alpha_s=alpha_s
        )
    raise ValueError("observable must be 'tmdwf' or 'tmdpdf'")


def sudakov_correction(q_gev, *, mu: float = 2.0, alpha_s: float = 0.293):
    _controls(1.0, mu, alpha_s)
    q = np.asarray(q_gev, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q <= 0):
        raise ValueError("Sudakov scale Q must be finite and positive")
    logarithm = np.log(mu**2 / q**2)
    return alpha_s * (4 / 3) / (4 * np.pi) * (
        -logarithm**2 - 3 * logarithm - 8 + np.pi**2 / 6
    )


def form_factor_kernel(x1, x2, *, momentum_gev: float, mu: float = 2.0, alpha_s: float = 0.293):
    _controls(momentum_gev, mu, alpha_s)
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    if any(np.any(~np.isfinite(x)) or np.any((x <= 0) | (x >= 1)) for x in (x1, x2)):
        raise ValueError("form-factor kernel requires 0 < x1,x2 < 1")
    q = 2 * momentum_gev * np.sqrt(x1 * x2)
    qbar = 2 * momentum_gev * np.sqrt((1 - x1) * (1 - x2))
    return 1 + sudakov_correction(q, mu=mu, alpha_s=alpha_s) + sudakov_correction(
        qbar, mu=mu, alpha_s=alpha_s
    )


def rapidity_factor(gamma, *, momentum_gev: float, p_ref_gev: float):
    _controls(momentum_gev, p_ref_gev, 1.0)
    return np.exp(np.log(p_ref_gev / momentum_gev) * np.asarray(gamma))
