from __future__ import annotations

import numpy as np

from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_rgr_lrr_tr import _kernel


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float,
    kappa: float = 1., spectator_kappa: float = 1., threshold_cut: float = .4,
    cut_quadrature: str = "fractional", mu_min_gev: float = .6,
    evolution_points: int = 100, evolution_steps: int = 40, y_max: float = 100.,
    eps: float = 1e-12, eps_m: float = .005, soft_eps_m: float = .01, soft_zs_gev_inv: float = .01,
    quadrature_order: int = 64,
) -> np.ndarray:
    return _kernel(
        x_out, x_in, order="nnlo", momentum_gev=momentum_gev, scale_gev=scale_gev, zs_fm=zs_fm,
        kappa=kappa, spectator_kappa=spectator_kappa, threshold_cut=threshold_cut,
        cut_quadrature=cut_quadrature, mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps, eps_m=eps_m,
        soft_eps_m=soft_eps_m, soft_zs_gev_inv=soft_zs_gev_inv, quadrature_order=quadrature_order,
    )
