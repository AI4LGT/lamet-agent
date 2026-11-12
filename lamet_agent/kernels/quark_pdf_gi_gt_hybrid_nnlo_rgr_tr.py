from __future__ import annotations

import math
import numpy as np
from scipy.integrate import quad
from scipy.special import zeta

from lamet_agent.kernels.implementation import CF, CA, TF, GEV_FM, alpha_s, _rgr_nonsinglet
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo import _prepare_nnlo_rgr_matching
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_rgr_tr import _geometry, _star_matrix


def _anomalous_dimensions(nf: int = 3):
    b0 = (11 * CA - 2 * nf) / (6 * np.pi)
    b1 = (17 * CA**2 - nf * TF * (10 * CA + 6 * CF)) / (12 * np.pi**2)
    b2 = (2857 - 5033 * nf / 9 + 325 * nf**2 / 27) / (64 * np.pi**3)
    b3 = 2 * (29243 - 6946.3 * nf + 405.089 * nf**2 + 1.49931 * nf**3) / (4 * np.pi)**4
    g1 = CF / np.pi
    g2 = CF * CA * (-1 / 12 + 67 / (36 * np.pi**2)) - 5 * CF * nf / (18 * np.pi**2)
    g3 = 4 * CF / (4 * np.pi)**3 * (
        -16 * nf**2 * TF**2 / 27
        + CA * nf * TF * (-418 / 27 + 40 * np.pi**2 / 27 - 56 * zeta(3) / 3)
        + CA**2 * (245 / 6 - 134 * np.pi**2 / 27 + 11 * np.pi**4 / 45 + 22 * zeta(3) / 3)
        + CF * nf * TF * (-55 / 3 + 16 * zeta(3))
    )
    g4 = .8301952532517868 - .20740149085932388 * nf + .00784293896438143 * nf**2
    g4 += .000131225944768728 * nf**3
    j1 = -CF / np.pi
    j2 = (23 * CA * CF / 144 - 349 * CA * CF / (108 * np.pi**2)
          - CF * nf * TF / 36 + 29 * CF * nf * TF / (27 * np.pi**2)
          + 5 * CA * CF * zeta(3) / (4 * np.pi**2))
    j3 = -.07788903112582936 + .05491061519528693 * nf + .0006023279289568775 * nf**2
    return np.array([b0, b1, b2, b3]), np.array([g1, g2, g3, g4]), np.array([j1, j2, j3])


def _evolution_integrals(a_i: float, a_h: float, nf: int = 3) -> tuple[float, float, float]:
    if a_i == a_h:
        return 0., 0., 0.
    beta, cusp, soft = _anomalous_dimensions(nf)
    b0 = beta[0]
    u = beta[1:] / b0
    inverse = np.array([1., -u[0], u[0]**2 - u[1], -u[0]**3 + 2 * u[0] * u[1] - u[2]])
    g = np.convolve(cusp, inverse)[:4]
    j = np.convolve(soft, inverse)[:3]
    r = a_h / a_i
    lr = np.log1p(r - 1)
    def integral(coefficients):
        return -(coefficients[0] * lr + sum(
            coefficients[k] * a_i**k * np.expm1(k * lr) / k
            for k in range(1, len(coefficients))
        )) / b0
    def sudakov_integrand(t):
        inner = np.array([1 / t - 1, -inverse[1] * np.log(t),
                          -inverse[2] * (t - 1), -inverse[3] * (t*t - 1) / 2]) / b0
        return sum(-g[k] * t**(k - 1) / b0 * inner[ell] * a_i**(k + ell - 1)
                   for k in range(4) for ell in range(4 - k))
    sud = quad(sudakov_integrand, 1., r, epsabs=2e-13, epsrel=2e-13)[0]
    return float(sud), float(integral(g)), float(integral(j))


def _soft_coefficients(coupling: float, nf: int = 3) -> np.ndarray:
    j1 = np.array([2 + np.pi**2 / 12, 1., .5, 0., 0.]) * CF / (2 * np.pi)
    j2 = np.array([
        1.777071001087165 - .12262625669319276 * nf,
        .6822951380677521 - .058791304088763895 * nf,
        .2841803732865709 - .015010545724790781 * nf,
        -.0018763182155988477 * (-28.5 + nf), .005628954646796543,
    ])
    return np.array([1., 0., 0., 0., 0.]) - coupling * j1 + coupling**2 * (np.convolve(j1, j1)[:5] - j2)


def _exp_coefficients(logarithm: np.ndarray) -> list[np.ndarray]:
    c2, c3, c4 = np.pi**2 / 24, zeta(3) / 3, 7 * np.pi**4 / 2880
    return [np.ones_like(logarithm), logarithm, logarithm**2 / 2 + c2,
            logarithm**3 / 6 + c2 * logarithm + c3,
            logarithm**4 / 24 + c2 * logarithm**2 / 2 + c3 * logarithm + c4 + c2**2 / 2]


def _soft_nnlo(y: np.ndarray, momentum: float, mu: float, coupling: float) -> np.ndarray:
    distance, weight, edges = _geometry(y)
    off_coeff = _exp_coefficients(np.log(2 * momentum * distance / mu))
    mass_coeff = _exp_coefficients(np.log(2 * momentum * edges / mu))
    coefficients = _soft_coefficients(coupling)
    off = np.zeros_like(distance)
    mass = np.zeros(len(y))
    for n, coefficient in enumerate(coefficients):
        multiplier = coefficient * (-2)**n * math.factorial(n) / 2
        if n:
            off += multiplier * weight * off_coeff[n - 1]
        mass += multiplier * mass_coeff[n].sum(axis=0)
    return off + np.diag(mass - off.sum(axis=0))


def _hard_factors(y: np.ndarray, momentum: float, mu: float, zs_fm: float, coupling: float):
    nf = 3
    b, g, _ = _anomalous_dimensions(nf)
    b0, g2 = b[0], g[1]
    p2 = np.pi**2
    ell = np.log(mu**2 / (4 * (y * momentum)**2))
    lh = np.log((zs_fm * mu / GEV_FM)**2 * np.exp(2 * np.euler_gamma) / 4)
    a = coupling
    gh2 = (CF * nf * (TF / 12 + 20 * TF / (27 * p2))
           - CA * CF * (1108 + 99 * p2 - 1188 * zeta(3)) / (432 * p2)
           - CF**2 * (6 - 7 * p2 + 36 * zeta(3)) / (12 * p2))
    ch = .0725 * CF**2 - .0840 * CF * CA + .1453 * CF * nf * TF
    h1 = a * CF / (2 * np.pi) * (-ell*ell / 2 - ell - 2 + p2 / 12)
    h2 = a*a * (CF**2 / (32 * p2) * ell**4
                + (CF**2 / (8 * p2) - CF * b0 / (24 * np.pi)) * ell**3
                + (-CF**2 / 96 + 3 * CF**2 / (8 * p2) - CF * b0 / (8 * np.pi) - g2 / 4) * ell*ell
                + (-CF**2 / 48 + CF**2 / (2 * p2) - CF * b0 / (2 * np.pi)
                   + CF * np.pi * b0 / 48 + gh2 / 2) * ell
                + ch - CF**2 / 24 + CF**2 / (2 * p2) + CF**2 * p2 / 1152)
    conversion1 = a * CF / (2 * np.pi) * (1.5 * lh + 2.5)
    conversion2 = a*a / 2 * (
        lh*lh * (11 * CA * CF / (16 * p2) + 9 * CF**2 / (16 * p2) - CF * nf * TF / (4 * p2))
        + lh * (CA * CF * (53 / 16 - p2 / 12) / p2 + CF**2 * (25 / 16 + p2 / 3) / p2
                - 5 * CF * nf * TF / (4 * p2))
        + CF**2 * (223 / 192 + p2 / 9 - 4 * zeta(3)) / p2
        + CA * CF * (4877 / 576 - 5 * p2 / 24 + zeta(3)) / p2 - 469 * CF * nf * TF / (144 * p2)
    )
    modulus = 1 - h1 - h2 + h1*h1 + conversion1 - conversion1*h1 + conversion2
    ca = (CF**2 * (-3 * zeta(3) / p2 + 7 / 12 - 1 / (2 * p2))
          + CF * CA * (11 * zeta(3) / (4 * p2) - 11 / 24 - 475 / (108 * p2))
          + CF * nf / 2 * (1 / 6 + 38 / (27 * p2)))
    phase = a * CF / 2 * (1 + ell) - a*a * np.pi / 2 * (-g2*ell - CF*b0/(2*np.pi)*(ell*ell/2+ell) + ca)
    return modulus, phase


def _hard_nnlo(y: np.ndarray, momentum: float, mu: float, zs_fm: float, coupling: float) -> np.ndarray:
    modulus, phase = _hard_factors(y, momentum, mu, zs_fm, coupling)
    den = 1 - y[:, None] / y[None, :]
    np.fill_diagonal(den, 1.)
    off = (y[1] - y[0]) / np.abs(y)[None, :] / den
    np.fill_diagonal(off, 0.)
    return off * (np.sin(phase) * modulus / np.pi)[None, :] + np.diag(np.cos(phase) * modulus)


def _cut_weights(y: np.ndarray, threshold_cut: float, quadrature: str) -> np.ndarray:
    if quadrature == "point":
        return (np.abs(1 - y[:, None] / y[None, :]) < threshold_cut - 32 * np.finfo(float).eps).astype(float)
    if quadrature != "fractional":
        raise ValueError("cut_quadrature must be 'fractional' or 'point'.")
    dy = y[1] - y[0]
    a, b = y / (1 + threshold_cut), y / (1 - threshold_cut)
    lo, hi = np.minimum(a, b)[:, None], np.maximum(a, b)[:, None]
    result = np.clip((np.minimum(y[None, :] + dy / 2, hi) - np.maximum(y[None, :] - dy / 2, lo)) / dy, 0., 1.)
    np.fill_diagonal(result, 1.)
    return result


def _threshold_correction(
    y: np.ndarray, momentum: float, mu_h: float, mu_i: float, zs_fm: float,
    threshold_cut: float, cut_quadrature: str = "fractional",
) -> np.ndarray:
    if mu_i >= mu_h:
        return np.zeros((len(y), len(y)))
    ai, ah = alpha_s(3, mu_i, "running"), alpha_s(3, mu_h, "running")
    sud, ag, aj = _evolution_integrals(ai, ah)
    mask = _cut_weights(y, threshold_cut, cut_quadrature)
    hard = mask * _hard_nnlo(y, momentum, mu_h, zs_fm, ah)
    ji = mask * _soft_nnlo(y, momentum, mu_i, ai)
    jh = mask * _soft_nnlo(y, momentum, mu_h, ah)
    evolution = mask * (np.exp(-2 * sud + aj) * _star_matrix(y, momentum, mu_i, 2 * ag))
    result = (evolution @ ji - jh) @ hard
    if not np.isfinite(result).all():
        raise ValueError("NNLO TR produced nonfinite coefficients; restrict the matching scales.")
    return result


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float,
    kappa: float = 1., spectator_kappa: float = 1., threshold_cut: float = .4,
    cut_quadrature: str = "fractional", mu_min_gev: float = .6,
    evolution_points: int = 100, evolution_steps: int = 40, y_max: float = 100.,
    eps: float = 1e-12, quadrature_order: int = 64,
) -> np.ndarray:
    for name, value in (("zs_fm", zs_fm), ("spectator_kappa", spectator_kappa), ("threshold_cut", threshold_cut)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if threshold_cut >= 1:
        raise ValueError("threshold_cut must be less than 1.")
    if cut_quadrature not in {"point", "fractional"}:
        raise ValueError("cut_quadrature must be 'fractional' or 'point'.")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, int) or not 16 <= quadrature_order <= 256:
        raise ValueError("quadrature_order must be an integer between 16 and 256.")

    def prepare(x, y):
        radius = np.abs(np.asarray(x_out, dtype=float))
        hard = 2 * kappa * momentum_gev * radius
        soft = 2 * spectator_kappa * momentum_gev * (1 - radius)
        if np.any((hard >= mu_min_gev) & (soft < hard) & (soft < mu_min_gev)):
            raise ValueError("TR spectator scale is below mu_min_gev; restrict x_out away from |x|=1.")
        fixed = _prepare_nnlo_rgr_matching(x, y, momentum_gev=momentum_gev, zs_fm=zs_fm,
                                           eps=eps, quadrature_order=quadrature_order)
        def at_scale(mu):
            radius = mu / (2 * kappa * momentum_gev)
            mu_i = 2 * spectator_kappa * momentum_gev * (1 - radius)
            base = fixed(mu)
            if mu_i >= mu:
                return base
            return base + _threshold_correction(y, momentum_gev, mu, mu_i, zs_fm, threshold_cut, cut_quadrature)
        return at_scale

    return _rgr_nonsinglet(
        x_out, x_in, prepare_matching=prepare, order="nnlo",
        momentum_gev=momentum_gev, scale_gev=scale_gev, kappa=kappa,
        mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps,
    )
