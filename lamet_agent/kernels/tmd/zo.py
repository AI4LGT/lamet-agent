"""Short-distance GI quasi-TMD coefficient, arXiv:2412.19988 (6),(7).

The evolution anomalous dimension includes BOTH endpoints: d ln C/d ln mu.
Running uses fixed nf and an explicitly supplied alpha_s(mu), without freezing.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp

from lamet_agent.kernels.implementation import HBAR_C_GEV_FM

CF, CA, TF = 4 / 3, 3.0, 0.5


def validate_settings(mu=2.0, alpha_s=0.293, n_f=3, order="nlo_rgr", min_scale_gev=0.7):
    for name, value in (("mu", mu), ("alpha_s", alpha_s), ("min_scale_gev", min_scale_gev)):
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if alpha_s >= 1 or mu < min_scale_gev:
        raise ValueError("reference coupling must be < 1 and mu >= min_scale_gev")
    if isinstance(n_f, bool) or n_f not in (3, 4, 5):
        raise ValueError("n_f must be 3, 4, or 5 (fixed-flavor prescription)")
    if order not in ("nlo", "nlo_rgr"):
        raise ValueError("order must be nlo or nlo_rgr; NNLO input is not validated")


def anomalous_dimension(alpha_s, n_f):
    """-2 times the HQET current gamma; positive 6 CF a_s at one loop."""
    a = alpha_s / (4 * np.pi)
    gamma1 = CF * (CF * (-5 + 16 * np.pi**2 / 3) + CA * (49 / 3 - 4 * np.pi**2 / 3) - 20 / 3 * TF * n_f)
    return 6 * CF * a + gamma1 * a**2


@lru_cache(maxsize=4096)
def _evolve(mu, alpha_s, mu0, n_f):
    beta0, beta1 = 11 - 2 * n_f / 3, 102 - 38 * n_f / 3

    def rhs(t, y):
        alpha = y[0]
        a = alpha / (4 * np.pi)
        return [-2 * alpha * (beta0 * a + beta1 * a**2), anomalous_dimension(alpha, n_f)]

    def strong_coupling(t, y):
        return 1 - y[0]

    strong_coupling.terminal = True
    solution = solve_ivp(rhs, (np.log(mu), np.log(mu0)), [alpha_s, 0.0], events=strong_coupling, rtol=2e-10, atol=2e-12)
    if not solution.success or len(solution.t_events[0]) or not np.all(np.isfinite(solution.y)):
        raise ValueError("RGR evolution leaves the perturbative domain (alpha_s >= 1)")
    # Integration was mu -> mu0; reverse its gamma integral for C(mu0) -> C(mu).
    return solution.y[0, -1], np.exp(-solution.y[1, -1])


def coefficient(z_fm, b_fm, *, mu=2.0, alpha_s=0.293, n_f=3, order="nlo_rgr", scale_factor=1.0, min_scale_gev=0.7):
    """Return a real coefficient for broadcastable coordinates in fm, b > 0."""
    validate_settings(mu, alpha_s, n_f, order, min_scale_gev)
    if not np.isfinite(scale_factor) or scale_factor <= 0 or isinstance(scale_factor, bool):
        raise ValueError("scale_factor must be finite and positive")
    if order == "nlo" and scale_factor != 1:
        raise ValueError("initial-scale variation is defined only for nlo_rgr")
    z, b = np.broadcast_arrays(np.asarray(z_fm, float), np.asarray(b_fm, float))
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("coordinates must be finite, with b_fm > 0")
    radius = np.hypot(z, b) / HBAR_C_GEV_FM
    natural = 2 * np.exp(-np.euler_gamma) / radius
    boundary = np.full(z.shape, mu) if order == "nlo" else natural * scale_factor
    if np.any(boundary < min_scale_gev):
        raise ValueError("natural boundary scale is below min_scale_gev; no freezing is applied")
    alphas, evolution = np.full(z.shape, alpha_s), np.ones(z.shape)
    if order == "nlo_rgr":
        for index in np.ndindex(z.shape):
            alphas[index], evolution[index] = _evolve(mu, alpha_s, float(boundary[index]), n_f)
    # Keep the boundary log, including 2 ln(scale_factor) away from the natural scale.
    bracket = 3 * np.log(boundary / natural) - 2 * z / b * np.arctan(z / b) + 0.5
    result = (1 + alphas * CF / (2 * np.pi) * bracket) * evolution
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("nonpositive/nonfinite perturbative coefficient; revise reference window")
    return result
