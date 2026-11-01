from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import (
    GEV_FM, _gi_gt_hybrid_nlo_tables, _lrr_resum, _lrr_shape, _rgr_nonsinglet, alpha_s,
)


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *,
    momentum_gev: float, scale_gev: float, zs_fm: float,
    kappa: float = 1.0, mu_min_gev: float = 0.6,
    evolution_points: int = 100, evolution_steps: int = 40,
    y_max: float = 100.0, eps: float = 1e-12, eps_m: float = 0.005,
) -> np.ndarray:
    for name, value in (("zs_fm", zs_fm), ("eps_m", eps_m)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")

    def prepare(x, y):
        identity = np.eye(len(y))
        shape = _lrr_shape(y, momentum_gev, zs_fm * momentum_gev / GEV_FM, eps, eps_m)
        constant, slope = _gi_gt_hybrid_nlo_tables(y, zs_fm * momentum_gev / GEV_FM, eps)

        def at_scale(mu):
            a = alpha_s(2, mu, "running") / (2 * np.pi)
            forward_one = a * (constant + 2 * np.log(mu / momentum_gev) * slope)
            return _lrr_resum(identity - forward_one, forward_one, shape, mu, order="nlo")

        return at_scale

    return _rgr_nonsinglet(
        x_out, x_in, prepare_matching=prepare, order="nlo", momentum_gev=momentum_gev,
        scale_gev=scale_gev, kappa=kappa, mu_min_gev=mu_min_gev,
        evolution_points=evolution_points, evolution_steps=evolution_steps, y_max=y_max, eps=eps,
    )
