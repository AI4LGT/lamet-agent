"""NNLL extension of the NLL Sudakov kernel, with the same H_F interface.

Source: J. Collins and T. C. Rogers, arXiv:1705.07167, Sec. VI.2,
Eqs. (58)-(60): https://arxiv.org/html/1705.07167#S6.SS2.
Our C_Sud is their spacelike H^Sud (not their collinear C^Sud):
Gamma_cusp = gamma_K/2, gamma_V = -gamma_j, a = alpha_s/(4*pi).
Use three-loop cusp, two-loop non-cusp and the one-loop boundary at mu=Q.
The H_F product and scales follow arXiv:2504.04625v5, Eq. (18), App. A.2.
The nf=3 analytic three-loop coupling and scale-floor convention are shared
with cg_nll; see also arXiv:1002.2213, Table 1 and Eq. (155).
This module extends only the form-factor Sudakov kernel, not H_phi or H_f.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad
from scipy.special import zeta

from . import cg_nll, cg_nlo

CF, CA, TF, NF = 4.0 / 3.0, 3.0, 0.5, 3
ZETA3 = float(zeta(3, 1))
# Eq. (59)/2: the Sudakov cusp is twice the C_TMD cusp in cg_nll.
GAMMA0 = 4 * CF
GAMMA1 = 4 * CF * ((67 / 9 - np.pi**2 / 3) * CA - 20 / 9 * TF * NF)
GAMMA2 = 4 * CF * (
    CA**2 * (245 / 6 - 134 * np.pi**2 / 27 + 11 * np.pi**4 / 45 + 22 * ZETA3 / 3)
    + CA * TF * NF * (-418 / 27 + 40 * np.pi**2 / 27 - 56 * ZETA3 / 3)
    + CF * TF * NF * (-55 / 3 + 16 * ZETA3)
    - 16 / 27 * TF**2 * NF**2
)
# Minus Eq. (58), through two loops.
NONCUSP0 = -6 * CF
NONCUSP1 = (
    CF**2 * (-3 + 4 * np.pi**2 - 48 * ZETA3)
    + CF * CA * (-961 / 27 - 11 * np.pi**2 / 3 + 52 * ZETA3)
    + CF * TF * NF * (260 / 27 + 4 * np.pi**2 / 3)
)


def validate_controls(*, mu=2.0, alpha_s=0.293, infrared_policy="error", freeze_scale_gev=None):
    if infrared_policy not in {"error", "freeze"}:
        raise ValueError("NNLL infrared_policy must be 'error' or 'freeze'; reference_quad is NLL-only")
    cg_nll.validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                             freeze_scale_gev=freeze_scale_gev)


@lru_cache(maxsize=65536)
def _sudakov_scalar(q, mu, alpha_s, infrared_policy, freeze_scale_gev):
    if infrared_policy == "freeze":
        q = max(q, freeze_scale_gev)
    if alpha_s == 0:
        return 1.0
    alpha_q = cg_nll.running_coupling(q, mu=mu, alpha_s=alpha_s,
                                    infrared_policy=infrared_policy, freeze_scale_gev=freeze_scale_gev)
    # Eq. (60) at t=log(Q^2/mu^2)=0. Keep the finite matching factor unexponentiated.
    boundary = float(1 + cg_nlo.sudakov_correction(q, mu=q, alpha_s=float(alpha_q)))
    if q == mu:
        return boundary
    log_q = np.log(q)

    def integrand_log(t):
        a = cg_nll.running_coupling(
            np.exp(t), mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
            freeze_scale_gev=freeze_scale_gev,
        ) / (4 * np.pi)
        cusp = a * (GAMMA0 + a * (GAMMA1 + a * GAMMA2))
        noncusp = a * (NONCUSP0 + a * NONCUSP1)
        return -2 * cusp * (t - log_q) + noncusp

    value, error = quad(integrand_log, log_q, np.log(mu), epsabs=1e-10, epsrel=1e-10)
    if error > 1e-8 * max(1.0, abs(value)):
        raise ValueError(f"NNLL Sudakov quadrature failed at Q={q:g} GeV: estimated error={error:g}")
    result = float(boundary * np.exp(value))
    if not np.isfinite(result):
        raise ValueError(f"NNLL Sudakov coefficient nonfinite at Q={q:g} GeV, log evolution={value:g}")
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
    cg_nlo._controls(momentum_gev, kwargs.get("mu", 2.0), kwargs.get("alpha_s", 0.293))
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    if any(np.any(~np.isfinite(x)) or np.any((x <= 0) | (x >= 1)) for x in (x1, x2)):
        raise ValueError("form-factor kernel requires 0 < x1,x2 < 1")
    q = 2 * momentum_gev * np.sqrt(x1*x2)
    qbar = 2 * momentum_gev * np.sqrt((1-x1)*(1-x2))
    return sudakov_factor(q, **kwargs) * sudakov_factor(qbar, **kwargs)
