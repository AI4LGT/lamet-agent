from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    _rgr_nonsinglet,
    C_hybrid_gz,
    GEV_FM,
    _pdf_density,
    build_matching_matrix_column_plus,
)


def _fixed_order(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float, eps: float = 1e-12
) -> np.ndarray:
    lc_x_ls = np.asarray(x_out, dtype=float)
    quasi_y_ls = np.asarray(x_in, dtype=float)
    mu = scale_gev
    zspz = zs_fm * momentum_gev / GEV_FM
    if zspz is None:
        raise ValueError("`zspz` is required for the hybrid matching kernel.")
    z = float(zspz)
    return build_matching_matrix_column_plus(
        lc_x_ls,
        mu,
        quasi_y_ls,
        eps,
        density=_pdf_density(lambda ksi, log_scale, y: C_hybrid_gz(ksi, log_scale, y, z, eps), momentum_gev, mu),
    )


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *,
    momentum_gev: float, scale_gev: float,
    zs_fm: float,
    kappa: float = 1.0,
    mu_min_gev: float = 0.6,
    evolution_points: int = 100,
    evolution_steps: int = 40,
    y_max: float = 100.0,
    eps: float = 1e-12,
) -> np.ndarray:
    if isinstance(zs_fm, bool) or not np.isfinite(zs_fm) or zs_fm <= 0:
        raise ValueError("zs_fm must be finite and positive.")
    def prepare(x, y):
        def at_scale(mu):
            matrix = _fixed_order(
                x, y, momentum_gev=momentum_gev, scale_gev=mu, eps=eps,
                zs_fm=zs_fm,
            )
            return matrix
        return at_scale

    return _rgr_nonsinglet(
        x_out, x_in, prepare_matching=prepare, order="nlo",
        momentum_gev=momentum_gev, scale_gev=scale_gev, kappa=kappa,
        mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps,
    )
