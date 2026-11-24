"""N3LL spacelike Sudakov coefficient for H_F (fixed nf=3).

Conventions: a=alpha_s/(4*pi), dC/dln(mu)=[Gamma*ln(Q^2/mu^2)+gamma_V]*C.
Matching and gamma_V: hep-ph/0607228 (50), (51), (96).
Exact four-loop cusp, including quartic invariants: 2002.04617 (6).
Four-loop MSbar beta function: hep-ph/9701390 (11).
Only H_F is extended, not the quasi-TMDWF/PDF matching coefficients.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import zeta

from . import cg_nlo, cg_nnll

CF, CA, TF, NF = cg_nnll.CF, cg_nnll.CA, cg_nnll.TF, cg_nnll.NF
Z2, Z3, Z5 = (float(zeta(n, 1)) for n in (2, 3, 5))
GAMMA0, GAMMA1, GAMMA2 = cg_nnll.GAMMA0, cg_nnll.GAMMA1, cg_nnll.GAMMA2
NONCUSP0, NONCUSP1 = cg_nnll.NONCUSP0, cg_nnll.NONCUSP1
# SU(3): d_F^{abcd} d_A^{abcd}/N_F=5/2, d_F^{abcd} d_F^{abcd}/N_F=5/36.
GAMMA3 = (
    NF**3 * CF * (64*Z3/27 - 32/81)
    + NF**2 * CA * CF * (-224*Z2**2/15 + 2240*Z3/27 - 608*Z2/81 + 923/81)
    + NF**2 * CF**2 * (64*Z2**2/5 - 640*Z3/9 + 2392/81)
    + NF * CA**2 * CF * (2096*Z5/9 + 448*Z3*Z2/3 - 352*Z2**2/15
                         - 23104*Z3/27 + 20320*Z2/81 - 24137/81)
    + NF * CA * CF**2 * (160*Z5 - 128*Z3*Z2 - 352*Z2**2/5 + 3712*Z3/9 + 440*Z2/3 - 34066/81)
    + NF * CF**3 * (-320*Z5 + 592*Z3/3 + 572/9)
    + NF * (5/36) * (-1280*Z5/3 - 256*Z3/3 + 256*Z2)
    + (5/2) * (-384*Z3**2 - 7936*Z2**3/35 + 3520*Z5/3 + 128*Z3/3 - 128*Z2)
    + CA**3 * CF * (-16*Z3**2 - 20032*Z2**3/105 - 3608*Z5/9 - 352*Z3*Z2/3
                    + 3608*Z2**2/5 + 20944*Z3/27 - 88400*Z2/81 + 84278/81)
)
# hep-ph/0607228 Eq. (96), with T_F explicit.
NONCUSP2 = (
    CF**3 * (-29 - 6*np.pi**2 - 16*np.pi**4/5 - 136*Z3 + 32*np.pi**2*Z3/3 + 480*Z5)
    + CF**2*CA * (-151/2 + 410*np.pi**2/9 + 494*np.pi**4/135
                  - 1688*Z3/3 - 16*np.pi**2*Z3/3 - 240*Z5)
    + CF*CA**2 * (-139345/1458 - 7163*np.pi**2/243 - 83*np.pi**4/45
                  + 7052*Z3/9 - 88*np.pi**2*Z3/9 - 272*Z5)
    + CF**2*TF*NF * (5906/27 - 52*np.pi**2/9 - 56*np.pi**4/27 + 1024*Z3/9)
    + CF*CA*TF*NF * (-34636/729 + 5188*np.pi**2/243 + 44*np.pi**4/45 - 3856*Z3/27)
    + CF*TF**2*NF**2 * (19336/729 - 80*np.pi**2/27 - 64*Z3/27)
)
BETA0, BETA1, BETA2 = 9., 64., 3863/6
BETA3 = (149753/6 + 3564*Z3 - (1078361/162 + 6508*Z3/27)*NF
         + (50065/162 + 6472*Z3/81)*NF**2 + 1093*NF**3/729)
BOUNDARY1 = CF * (-8 + np.pi**2/6)
BOUNDARY2 = (
    CF**2 * (255/8 + 7*np.pi**2/2 - 83*np.pi**4/360 - 30*Z3)
    + CF*CA * (-51157/648 - 337*np.pi**2/108 + 11*np.pi**4/45 + 313*Z3/9)
    + CF*TF*NF * (4085/162 + 23*np.pi**2/27 + 4*Z3/9)
)


@lru_cache(maxsize=64)
def coupling_pole_gev(*, mu=2.0, alpha_s=0.293):
    """Pole of the numerical four-loop beta function, not the NNLL approximation."""
    cg_nlo._controls(1., mu, alpha_s)
    if alpha_s == 0:
        return 0.
    a = alpha_s / (4*np.pi)
    # Integrate in inverse coupling, mapped to [0,1]. No singular endpoint.
    distance = quad(lambda v: v**3 / (2*a*(BETA0*v**3 + BETA1*a*v**2
                                         + BETA2*a*a*v + BETA3*a**3)),
                    0., 1., epsabs=1e-11, epsrel=1e-11)[0]
    return float(np.exp(np.log(mu) - distance))


def validate_controls(*, mu=2.0, alpha_s=0.293, infrared_policy="error", freeze_scale_gev=None):
    cg_nlo._controls(1., mu, alpha_s)
    if infrared_policy not in {"error", "freeze"}:
        raise ValueError("NNNLL infrared_policy must be 'error' or 'freeze'; reference_quad is NLL-only")
    if infrared_policy == "freeze":
        if (isinstance(freeze_scale_gev, bool) or not isinstance(freeze_scale_gev, (int, float))
                or not np.isfinite(freeze_scale_gev) or freeze_scale_gev <= 0):
            raise ValueError("infrared_policy='freeze' requires finite positive freeze_scale_gev")
        pole = coupling_pole_gev(mu=mu, alpha_s=alpha_s)
        if freeze_scale_gev <= pole or freeze_scale_gev > mu:
            raise ValueError(f"freeze_scale_gev must lie above pole={pole:.12g} GeV and at or below mu={mu:g}")
    elif freeze_scale_gev is not None:
        raise ValueError("freeze_scale_gev is only meaningful with infrared_policy='freeze'")


@lru_cache(maxsize=65536)
def _evolve(q, mu, alpha_s):
    """Return alpha_s(Q), log[C(Q,mu)/C(Q,Q)] by a coupled RG solve."""
    if alpha_s == 0:
        return 0., 0.
    pole = coupling_pole_gev(mu=mu, alpha_s=alpha_s)
    if q <= pole:
        raise ValueError(f"NNNLL scale Q={q:g} GeV reaches coupling pole={pole:.12g} GeV")
    if q == mu:
        return alpha_s, 0.
    a_ref = alpha_s / (4*np.pi)
    log_ratio = np.log(q) - np.log(mu)

    def rhs(t, y):
        a = a_ref*y[0]
        beta = -2*a*y[0]*(BETA0 + a*(BETA1 + a*(BETA2 + a*BETA3)))
        cusp = a*(GAMMA0 + a*(GAMMA1 + a*(GAMMA2 + a*GAMMA3)))
        gamma = a*(NONCUSP0 + a*(NONCUSP1 + a*NONCUSP2))
        return beta, 2*cusp*(log_ratio-t) + gamma

    # Integrate from mu to Q and reverse the exponent for evolution Q -> mu.
    solution = solve_ivp(rhs, (0., log_ratio), (1., 0.), method="DOP853", rtol=2e-12, atol=2e-14)
    if not solution.success or not np.all(np.isfinite(solution.y[:, -1])):
        raise ValueError(f"NNNLL RG integration failed at Q={q:g} GeV: {solution.message}")
    return float(alpha_s*solution.y[0, -1]), float(-solution.y[1, -1])


def _scales(q_gev, mu, alpha_s, infrared_policy, freeze_scale_gev):
    validate_controls(mu=mu, alpha_s=alpha_s, infrared_policy=infrared_policy,
                      freeze_scale_gev=freeze_scale_gev)
    q = np.asarray(q_gev, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q <= 0):
        raise ValueError("Sudakov scale Q must be finite and positive")
    return np.maximum(q, freeze_scale_gev) if infrared_policy == "freeze" else q


def running_coupling(scale_gev, *, mu=2.0, alpha_s=0.293, infrared_policy="error", freeze_scale_gev=None):
    """Four-loop MSbar running, with alpha_s specified at mu; fixed nf=3."""
    q = _scales(scale_gev, mu, alpha_s, infrared_policy, freeze_scale_gev)
    unique, inverse = np.unique(q, return_inverse=True)
    values = [_evolve(float(v), float(mu), float(alpha_s))[0] for v in unique]
    return np.asarray(values)[inverse].reshape(q.shape)


def _evolution_orders(a0, a1, beta, cusp, noncusp):
    """Return E0,E1,E2 for 2S(Q,mu)-a_gammaV(Q,mu).

    Expand the inverse beta function before integration, as in
    hep-ph/0607228 (93),(94). E0 contains the 1/a and constant terms;
    E1 and E2 are the O(a) and O(a^2) contributions, not cumulative sums.
    Arrays are arguments so the historical nf=5 benchmark can use this
    same implementation without changing the production nf=3 constants.
    """
    if a0 == a1:
        return np.zeros(3)
    b = np.asarray(beta)/beta[0]
    d = np.array([1., -b[1], b[1]**2-b[2], -b[1]**3+2*b[1]*b[2]-b[3]])
    c = np.convolve(cusp, d)[:4]
    v = np.convolve(noncusp, d)[:3]
    r = a1/a0
    result = np.zeros(3)
    for i in range(4):
        for j in range(4-i):
            def integrand(t):
                inner = np.log(t) if j == 1 else (t**(j-1)-1)/(j-1)
                return t**(i-1)*inner
            result[max(0, i+j-1)] -= (
                c[i]*d[j]*a0**(i+j-1)
                * quad(integrand, 1., r, epsabs=1e-12, epsrel=1e-12)[0]/(2*beta[0]**2)
            )
    result -= v*np.array([np.log(r), a1-a0, (a1*a1-a0*a0)/2])/(2*beta[0])
    return result


@lru_cache(maxsize=65536)
def _parts(q, mu, alpha_s):
    if alpha_s == 0:
        return 0., 0., 0.
    alpha_q, _ = _evolve(q, mu, alpha_s)
    a = alpha_q/(4*np.pi)
    e0, e1, e2 = _evolution_orders(
        a, alpha_s/(4*np.pi), (BETA0, BETA1, BETA2, BETA3),
        (GAMMA0, GAMMA1, GAMMA2, GAMMA3), (NONCUSP0, NONCUSP1, NONCUSP2),
    )
    return e0, e1+a*BOUNDARY1, e2+e1*e1/2+a*a*BOUNDARY2+e1*a*BOUNDARY1


def _array_parts(q_gev, *, mu=2.0, alpha_s=0.293, infrared_policy="error", freeze_scale_gev=None):
    q = _scales(q_gev, mu, alpha_s, infrared_policy, freeze_scale_gev)
    unique, inverse = np.unique(q, return_inverse=True)
    values = np.asarray([_parts(float(v), float(mu), float(alpha_s)) for v in unique]).reshape(-1, 3)
    return tuple(values[inverse, k].reshape(q.shape) for k in range(3))


def _assemble(e0, p1, p2):
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.exp(e0)*(1+p1+p2)
    if np.any(~np.isfinite(value)):
        raise ValueError("NNNLL Sudakov coefficient nonfinite")
    return value


def sudakov_factor(q_gev, *, mu=2.0, alpha_s=0.293, infrared_policy="error", freeze_scale_gev=None):
    """Amplitude C_V, consistently expanded through NNLO in RG counting."""
    return _assemble(*_array_parts(q_gev, mu=mu, alpha_s=alpha_s,
                     infrared_policy=infrared_policy, freeze_scale_gev=freeze_scale_gev))


def form_factor_kernel(x1, x2, *, momentum_gev, **kwargs):
    cg_nlo._controls(momentum_gev, kwargs.get("mu", 2.), kwargs.get("alpha_s", .293))
    x1, x2 = np.asarray(x1, dtype=float), np.asarray(x2, dtype=float)
    if any(np.any(~np.isfinite(x)) or np.any((x <= 0) | (x >= 1)) for x in (x1, x2)):
        raise ValueError("form-factor kernel requires 0 < x1,x2 < 1")
    q = 2*momentum_gev*np.sqrt(x1*x2)
    qbar = 2*momentum_gev*np.sqrt((1-x1)*(1-x2))
    e0, p1, p2 = _array_parts(q, **kwargs)
    f0, r1, r2 = _array_parts(qbar, **kwargs)
    # Truncate the product as well: p1*r1 is NNLO; p1*r2 and p2*r2 are not.
    return _assemble(e0+f0, p1+r1, p2+r2+p1*r1)
