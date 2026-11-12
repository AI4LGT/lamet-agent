from __future__ import annotations

import numpy as np
from scipy.special import gamma, zeta

from lamet_agent.kernels.implementation import (
    CF, CA, TF, GEV_FM, alpha_s, _gi_gt_hybrid_nlo_tables, _rgr_nonsinglet,
)


def _evolution_integrals(a_i: float, a_h: float, nf: int = 3) -> tuple[float, float, float]:
    b0 = (11 * CA - 2 * nf) / (6 * np.pi)
    b1 = (17 * CA**2 - nf * TF * (10 * CA + 6 * CF)) / (12 * np.pi**2)
    b2 = (2857 - 5033 * nf / 9 + 325 * nf**2 / 27) / (64 * np.pi**3)
    g1 = CF / np.pi
    g2 = CF * CA * (-1 / 12 + 67 / (36 * np.pi**2)) - 5 * CF * nf / (18 * np.pi**2)
    g3 = 4 * CF / (4 * np.pi)**3 * (
        -16 * nf**2 * TF**2 / 27
        + CA * nf * TF * (-418 / 27 + 40 * np.pi**2 / 27 - 56 * zeta(3) / 3)
        + CA**2 * (245 / 6 - 134 * np.pi**2 / 27 + 11 * np.pi**4 / 45 + 22 * zeta(3) / 3)
        + CF * nf * TF * (-55 / 3 + 16 * zeta(3))
    )
    j1 = -CF / np.pi
    j2 = (23 * CA * CF / 144 - 349 * CA * CF / (108 * np.pi**2)
          - CF * nf * TF / 36 + 29 * CF * nf * TF / (27 * np.pi**2)
          + 5 * CA * CF * zeta(3) / (4 * np.pi**2))
    r = a_h / a_i
    lr = np.log1p(r - 1)
    ag = -(g1 * lr / b0 + (a_h - a_i) * (b0 * g2 - b1 * g1) / b0**2
           + (a_h**2 - a_i**2) * (b1**2 * g1 - b0 * b2 * g1 - b0 * b1 * g2 + b0**2 * g3)
           / (2 * b0**3))
    sud = -(g1 * (-1 + r - r * lr) / (r * b0**2 * a_i)
            + (2 * (r - 1) * (b1 * g1 - b0 * g2)
               + lr * (-2 * b1 * g1 + 2 * b0 * g2 + b1 * g1 * lr)) / (2 * b0**3)
            - a_i * ((r - 1) * ((r - 1) * b1**2 * g1 - (r - 3) * b0 * b1 * g2
                                    + b0 * (-(1 + r) * b2 * g1 + (r - 1) * b0 * g3))
                     + 2 * ((r - 1) * b1**2 * g1 + b0 * b2 * g1 - r * b0 * b1 * g2) * lr)
            / (2 * b0**4))
    aj = -(j1 / b0 * np.log(r * (b0 + b1 * a_i) / (b0 + b1 * a_h))
           + j2 / b1 * np.log((b0 + b1 * a_h) / (b0 + b1 * a_i)))
    return float(sud), float(ag), float(aj)


def _geometry(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dy = y[1] - y[0]
    distance = np.abs(y[:, None] - y[None, :])
    np.fill_diagonal(distance, 1.0)
    weight = dy / distance
    np.fill_diagonal(weight, 0.0)
    edges = np.stack((y - (y[0] - dy / 2), (y[-1] + dy / 2) - y))
    return distance, weight, edges


def _star_matrix(y: np.ndarray, momentum: float, mu: float, eta: float) -> np.ndarray:
    if not np.isfinite(eta) or not -1 < eta < 1:
        raise ValueError("TR requires -1 < eta < 1; increase the spectator scale or restrict x_out.")
    if eta == 0:
        return np.eye(len(y))
    distance, weight, edges = _geometry(y)
    prefactor = np.sinc(eta / 2) * gamma(1 - eta) * np.exp(-np.euler_gamma * eta) / 2
    off = eta * prefactor * weight * np.exp(eta * np.log(2 * momentum * distance / mu))
    total = prefactor * np.sum(np.exp(eta * np.log(2 * momentum * edges / mu)), axis=0)
    return off + np.diag(total - off.sum(axis=0))


def _soft_nlo(y: np.ndarray, momentum: float, mu: float, coupling: float) -> np.ndarray:
    distance, weight, edges = _geometry(y)
    a = coupling * CF / (2 * np.pi)
    off = a * weight * (1 - 2 * np.log(2 * momentum * distance / mu))
    logs = np.log(2 * momentum * edges / mu)
    total = 1 - a * (2 + np.pi**2 / 4 - logs.sum(axis=0) + (logs**2).sum(axis=0))
    return off + np.diag(total - off.sum(axis=0))


def _hard_nlo(y: np.ndarray, momentum: float, mu: float, zs_fm: float, coupling: float) -> np.ndarray:
    logarithm = np.log(mu**2 / (4 * (y * momentum)**2))
    a = coupling * CF / (2 * np.pi)
    hybrid = 1.5 * np.log((zs_fm * mu / GEV_FM)**2 * np.exp(2 * np.euler_gamma) / 4) + 2.5
    modulus = 1 + a * (logarithm**2 / 2 + logarithm + 2 - np.pi**2 / 12 + hybrid)
    phase = a * np.pi * (1 + logarithm)
    denominator = 1 - y[:, None] / y[None, :]
    np.fill_diagonal(denominator, 1.0)
    off = (y[1] - y[0]) / np.abs(y)[None, :] / denominator
    np.fill_diagonal(off, 0.0)
    return off * (np.sin(phase) * modulus / np.pi)[None, :] + np.diag(np.cos(phase) * modulus)


def _threshold_correction(
    y: np.ndarray, momentum: float, mu_h: float, mu_i: float, zs_fm: float, threshold_cut: float,
) -> np.ndarray:
    if mu_i >= mu_h:
        return np.zeros((len(y), len(y)))
    a_i, a_h = alpha_s(2, mu_i, "running"), alpha_s(2, mu_h, "running")
    sud, ag, aj = _evolution_integrals(a_i, a_h)
    distance = np.abs(1 - y[:, None] / y[None, :])
    mask = distance < threshold_cut - 32 * np.finfo(float).eps
    hard = mask * _hard_nlo(y, momentum, mu_h, zs_fm, a_h)
    soft_i = mask * _soft_nlo(y, momentum, mu_i, a_i)
    soft_h = mask * _soft_nlo(y, momentum, mu_h, a_h)
    evolution = mask * (np.exp(-2 * sud + aj) * _star_matrix(y, momentum, mu_i, 2 * ag))
    correction = (evolution @ soft_i - soft_h) @ hard
    if not np.isfinite(correction).all():
        raise ValueError("TR produced nonfinite coefficients; restrict the matching scales.")
    return correction


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *,
    momentum_gev: float, scale_gev: float, zs_fm: float,
    kappa: float = 1.0, spectator_kappa: float = 1.0,
    threshold_cut: float = 0.4, mu_min_gev: float = 0.6,
    evolution_points: int = 100, evolution_steps: int = 40,
    y_max: float = 100.0, eps: float = 1e-12,
) -> np.ndarray:
    for name, value in (("zs_fm", zs_fm), ("spectator_kappa", spectator_kappa), ("threshold_cut", threshold_cut)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if threshold_cut >= 1:
        raise ValueError("threshold_cut must be less than 1.")

    def prepare(x, y):
        radius = np.abs(np.asarray(x_out, dtype=float))
        hard_scales = 2 * kappa * momentum_gev * radius
        soft_scales = 2 * spectator_kappa * momentum_gev * (1 - radius)
        if np.any((hard_scales >= mu_min_gev) & (soft_scales < hard_scales) & (soft_scales < mu_min_gev)):
            raise ValueError("TR spectator scale is below mu_min_gev; restrict x_out away from |x|=1.")
        identity = np.eye(len(y))
        constant, slope = _gi_gt_hybrid_nlo_tables(y, zs_fm * momentum_gev / GEV_FM, eps)

        def at_scale(mu):
            a = alpha_s(2, mu, "running") / (2 * np.pi)
            fixed = identity - a * (constant + 2 * np.log(mu / momentum_gev) * slope)
            radius = mu / (2 * kappa * momentum_gev)
            mu_i = 2 * spectator_kappa * momentum_gev * (1 - radius)
            if mu_i >= mu:
                return fixed
            return fixed + _threshold_correction(y, momentum_gev, mu, mu_i, zs_fm, threshold_cut)
        return at_scale

    return _rgr_nonsinglet(
        x_out, x_in, prepare_matching=prepare, order="nlo",
        momentum_gev=momentum_gev, scale_gev=scale_gev, kappa=kappa,
        mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps,
    )
