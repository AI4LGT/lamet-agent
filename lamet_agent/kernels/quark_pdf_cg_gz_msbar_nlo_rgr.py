from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    _rgr_nonsinglet,
    _zpsi_msbar_ratio,
    C_msbar_gz,
    C_msbar_gz_plus,
    build_matching_matrix_row_plus,
)


def _fixed_order(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, eps: float = 1e-12
) -> np.ndarray:
    lc_x_ls = np.asarray(x_out, dtype=float)
    quasi_y_ls = np.asarray(x_in, dtype=float)
    mu = scale_gev
    return build_matching_matrix_row_plus(
        lc_x_ls,
        momentum_gev,
        mu,
        quasi_y_ls,
        eps,
        coeff=lambda ksi, log_scale, y: C_msbar_gz(ksi, log_scale, eps),
        plus_coeff=lambda ksi, log_scale, y: C_msbar_gz_plus(ksi, log_scale, eps),
        diagonal_extra=lambda log_scale: 0.5 * (1.0 + log_scale) + 1.0,
    )


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *,
    momentum_gev: float, scale_gev: float,
    kappa: float = 1.0,
    mu_min_gev: float = 0.6,
    evolution_points: int = 100,
    evolution_steps: int = 40,
    y_max: float = 100.0,
    eps: float = 1e-12,
) -> np.ndarray:
    def prepare(x, y):
        def at_scale(mu):
            matrix = _fixed_order(
                x, y, momentum_gev=momentum_gev, scale_gev=mu, eps=eps,
            )
            return matrix * _zpsi_msbar_ratio(mu)
        return at_scale

    return _rgr_nonsinglet(
        x_out, x_in, prepare_matching=prepare, order="nlo",
        momentum_gev=momentum_gev, scale_gev=scale_gev, kappa=kappa,
        mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps,
    )
