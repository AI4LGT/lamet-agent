"""Short-distance GI quasi-TMD coefficient, arXiv:2412.19988 (6),(7).

The evolution anomalous dimension includes BOTH endpoints: d ln C/d ln mu.
Running uses fixed nf and an explicitly supplied alpha_s(mu), without freezing.
"""

from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import zeta

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
    if order not in ("nlo", "nlo_rgr", "nnlo", "nnlo_rgr"):
        raise ValueError("order must be nlo, nlo_rgr, nnlo, or nnlo_rgr")


def anomalous_dimension(alpha_s, n_f, loops=2):
    """-2 gamma_j(HQET), arXiv:2311.09894v3 Eq. (3.3), both endpoints."""
    if loops not in (2, 3):
        raise ValueError("RGR loops must be 2 or 3")
    a = alpha_s / (4 * np.pi)
    gamma1 = CF * (CF * (-5 + 16 * np.pi**2 / 3) + CA * (49 / 3 - 4 * np.pi**2 / 3) - 20 / 3 * TF * n_f)
    result = 6 * CF * a + gamma1 * a**2
    if loops == 3:
        # Coefficient of a_s^3 in the single HQET current anomalous dimension.
        gamma_j2 = CF * (
            -CF**2 * (36*zeta(3) + 8*np.pi**4/9 - 32*np.pi**2/3 + 37/2)
            + CF*CA/3 * (142*zeta(3) - 8*np.pi**4/15 - 592*np.pi**2/9 - 655/12)
            - CA**2/3 * (22*zeta(3) + 4*np.pi**4/5 - 130*np.pi**2/9 - 1451/36)
            - 2*CF*TF*n_f/3 * (88*zeta(3) - 112*np.pi**2/9 - 235/3)
            + 8*CA*TF*n_f/3 * (19*zeta(3) - 7*np.pi**2/9 - 64/9)
            + 140/27 * (TF*n_f)**2
        )
        result -= 2 * gamma_j2 * a**3
    return result


@lru_cache(maxsize=4096)
def _evolve(mu, alpha_s, mu0, n_f, loops=2):
    if loops not in (2, 3):
        raise ValueError("RGR loops must be 2 or 3")
    beta0, beta1 = 11 - 2 * n_f / 3, 102 - 38 * n_f / 3
    beta2 = 2857/2 - 5033*n_f/18 + 325*n_f**2/54 if loops == 3 else 0.0

    def rhs(t, y):
        alpha = y[0]
        a = alpha / (4 * np.pi)
        return [-2 * alpha * (beta0*a + beta1*a**2 + beta2*a**3), anomalous_dimension(alpha, n_f, loops)]

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
    if order in ("nlo", "nnlo") and scale_factor != 1:
        raise ValueError("initial-scale variation requires an RGR order")
    z, b = np.broadcast_arrays(np.asarray(z_fm, float), np.asarray(b_fm, float))
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("coordinates must be finite, with b_fm > 0")
    radius = np.hypot(z, b) / HBAR_C_GEV_FM
    natural = 2 * np.exp(-np.euler_gamma) / radius
    boundary = np.full(z.shape, mu) if order in ("nlo", "nnlo") else natural * scale_factor
    if np.any(boundary < min_scale_gev):
        raise ValueError("natural boundary scale is below min_scale_gev; no freezing is applied")
    alphas, evolution = np.full(z.shape, alpha_s), np.ones(z.shape)
    if order.endswith("_rgr"):
        for index in np.ndindex(z.shape):
            alphas[index], evolution[index] = _evolve(
                mu, alpha_s, float(boundary[index]), n_f, loops=3 if order == "nnlo_rgr" else 2,
            )
    # Keep the boundary log, including 2 ln(scale_factor) away from the natural scale.
    bracket = 3 * np.log(boundary / natural) - 2 * z / b * np.arctan(z / b) + 0.5
    fixed = 1 + alphas * CF / (2 * np.pi) * bracket
    if order.startswith("nnlo"):
        from lamet_agent.kernels.tmd.zo_nnlo import checked_same_side_sum, color_coefficients

        ratios = z / b
        same = np.array([checked_same_side_sum(float(r)) for r in ratios.flat]).reshape(z.shape)
        hff, hfa, hfferm = color_coefficients(ratios, 2 * np.log(boundary / natural), same)
        fixed += (alphas / (4 * np.pi))**2 * (CF**2 * hff + CF * CA * hfa + CF * TF * n_f * hfferm)
    result = fixed * evolution
    if not np.all(np.isfinite(result)) or np.any(result <= 0):
        raise ValueError("nonpositive/nonfinite perturbative coefficient; revise reference window")
    return result
