from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    GEV_FM,
    G_qq_cg_z,
    _da_cg_matrix,
)


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float, eps: float = 1e-12
) -> np.ndarray:
    lc_x_ls = np.asarray(x_out, dtype=float)
    quasi_y_ls = np.asarray(x_in, dtype=float)
    mu = scale_gev
    zspz = zs_fm * momentum_gev / GEV_FM
    return _da_cg_matrix(lc_x_ls, momentum_gev, mu, quasi_y_ls, eps, coefficient=G_qq_cg_z, scheme="hybrid", zspz=zspz)
