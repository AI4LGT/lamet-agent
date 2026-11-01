from __future__ import annotations

import numpy as np
from lamet_agent.kernels.implementation import _rgr_nonsinglet
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo import _prepare_nnlo_rgr_matching


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *,
    momentum_gev: float, scale_gev: float, zs_fm: float,
    kappa: float = 1.0,
    mu_min_gev: float = 0.6,
    evolution_points: int = 100,
    evolution_steps: int = 40,
    y_max: float = 100.0,
    eps: float = 1e-12,
    quadrature_order: int = 64,
) -> np.ndarray:
    if isinstance(zs_fm, bool) or not np.isfinite(zs_fm) or zs_fm <= 0:
        raise ValueError("zs_fm must be finite and positive.")
    if isinstance(quadrature_order, bool) or not isinstance(quadrature_order, int) or not 16 <= quadrature_order <= 256:
        raise ValueError("quadrature_order must be an integer between 16 and 256.")

    def prepare(x, y):
        return _prepare_nnlo_rgr_matching(
            x, y, momentum_gev=momentum_gev, zs_fm=zs_fm,
            eps=eps, quadrature_order=quadrature_order,
        )

    return _rgr_nonsinglet(
        x_out, x_in, prepare_matching=prepare, order="nnlo",
        momentum_gev=momentum_gev, scale_gev=scale_gev, kappa=kappa,
        mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps,
    )
