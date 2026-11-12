from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from lamet_agent.kernels.implementation import (
    CF, GEV_FM, _gi_gt_hybrid_nlo_tables, _lrr_resum, _lrr_shape,
    _rgr_nonsinglet, alpha_s, dPVasym, rnasym,
)
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nlo_rgr_tr import (
    _evolution_integrals as _nlo_integrals, _soft_nlo, _star_matrix,
)
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo_rgr_tr import (
    _cut_weights, _evolution_integrals as _nnlo_integrals, _hard_factors, _soft_nnlo,
)
from lamet_agent.kernels.quark_pdf_gi_gt_hybrid_nnlo import _prepare_nnlo_rgr_matching


def _hard_lrr(y, momentum, mu, zs_fm, coupling, order):
    ell = np.log(mu**2 / (4 * (y * momentum)**2))
    zs = zs_fm / GEV_FM
    hybrid = 1.5 * np.log(zs**2 * mu**2 * np.exp(2 * np.euler_gamma) / 4) + 2.5
    first = coupling * CF / (2 * np.pi) * (ell**2 / 2 + ell + 2 - np.pi**2 / 12 + hybrid)
    r0 = rnasym(0, 1., mu, 3) * coupling
    pv = dPVasym(1., mu, 3, coupling)
    if order == "nnlo":
        modulus, phase = _hard_factors(y, momentum, mu, zs_fm, coupling)
        r1 = rnasym(1, 1., mu, 3) * coupling**2
        modulus = modulus - zs*(r0+r1) - zs*r0*first + .5*(zs*r0)**2
        phase = phase + 2*(pv-r0-r1)/(np.abs(y)*momentum)
    else:
        modulus = 1 + first - zs*r0
        phase = coupling*CF/2*(1+ell) + 2*(pv-r0)/(np.abs(y)*momentum)
    modulus = modulus * np.exp(zs*pv)
    den = 1-y[:, None]/y[None, :]
    np.fill_diagonal(den, 1.)
    off = (y[1]-y[0])/np.abs(y)[None, :]/den
    np.fill_diagonal(off, 0.)
    return off*(np.sin(phase)*modulus/np.pi)[None, :] + np.diag(np.cos(phase)*modulus)


def _soft_lrr(y, momentum, mu, coupling, shape, order):
    first = _soft_nlo(y, momentum, mu, coupling)
    r0 = rnasym(0, 1., mu, 3)*coupling
    pv = dPVasym(1., mu, 3, coupling)
    if order == "nnlo":
        r1 = rnasym(1, 1., mu, 3)*coupling**2
        fixed = _soft_nnlo(y, momentum, mu, coupling)
        prefactor = fixed + (r0+r1)*shape + r0*((first-np.eye(len(y)))@shape) + .5*r0**2*(shape@shape)
    else:
        prefactor = first + r0*shape
    return prefactor @ expm(-pv*shape)


def _threshold_correction(y, momentum, mu_h, mu_i, zs_fm, threshold_cut, cut_quadrature, shape, order):
    if mu_i >= mu_h:
        return np.zeros((len(y), len(y)))
    loops = 3 if order == "nnlo" else 2
    ai, ah = alpha_s(loops, mu_i, "running"), alpha_s(loops, mu_h, "running")
    sud, ag, aj = (_nnlo_integrals if order == "nnlo" else _nlo_integrals)(ai, ah)
    mask = _cut_weights(y, threshold_cut, cut_quadrature)
    hard = mask*_hard_lrr(y, momentum, mu_h, zs_fm, ah, order)
    ji = mask*_soft_lrr(y, momentum, mu_i, ai, shape, order)
    jh = mask*_soft_lrr(y, momentum, mu_h, ah, shape, order)
    evolution = mask*(np.exp(-2*sud+aj)*_star_matrix(y, momentum, mu_i, 2*ag))
    result = (evolution@ji-jh)@hard
    if not np.isfinite(result).all():
        raise ValueError("LRR+TR produced nonfinite coefficients; restrict the matching scales.")
    return result


def _kernel(
    x_out, x_in, *, order, momentum_gev, scale_gev, zs_fm, kappa, spectator_kappa,
    threshold_cut, cut_quadrature, mu_min_gev, evolution_points, evolution_steps,
    y_max, eps, eps_m, soft_eps_m, soft_zs_gev_inv, quadrature_order,
):
    for name, value in (("zs_fm", zs_fm), ("spectator_kappa", spectator_kappa),
                        ("threshold_cut", threshold_cut), ("eps_m", eps_m),
                        ("soft_eps_m", soft_eps_m), ("soft_zs_gev_inv", soft_zs_gev_inv)):
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
        hard = 2*kappa*momentum_gev*radius
        soft = 2*spectator_kappa*momentum_gev*(1-radius)
        if np.any((hard >= mu_min_gev) & (soft < hard) & (soft < mu_min_gev)):
            raise ValueError("TR spectator scale is below mu_min_gev; restrict x_out away from |x|=1.")
        shape_soft = _lrr_shape(y, momentum_gev, soft_zs_gev_inv*momentum_gev, eps, soft_eps_m)
        if order == "nnlo":
            fixed = _prepare_nnlo_rgr_matching(x, y, momentum_gev=momentum_gev, zs_fm=zs_fm,
                                               eps=eps, quadrature_order=quadrature_order, lrr=True, eps_m=eps_m)
        else:
            identity = np.eye(len(y))
            shape = _lrr_shape(y, momentum_gev, zs_fm*momentum_gev/GEV_FM, eps, eps_m)
            constant, slope = _gi_gt_hybrid_nlo_tables(y, zs_fm*momentum_gev/GEV_FM, eps)

            def fixed(mu):
                a = alpha_s(2, mu, "running")/(2*np.pi)
                forward = a*(constant+2*np.log(mu/momentum_gev)*slope)
                return _lrr_resum(identity-forward, forward, shape, mu, order="nlo")

        def at_scale(mu):
            radius = mu/(2*kappa*momentum_gev)
            mu_i = 2*spectator_kappa*momentum_gev*(1-radius)
            base = fixed(mu)
            if mu_i >= mu:
                return base
            return base + _threshold_correction(y, momentum_gev, mu, mu_i, zs_fm,
                                                 threshold_cut, cut_quadrature, shape_soft, order)
        return at_scale

    return _rgr_nonsinglet(
        x_out, x_in, prepare_matching=prepare, order=order, momentum_gev=momentum_gev,
        scale_gev=scale_gev, kappa=kappa, mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps,
    )


def kernel(
    x_out: np.ndarray, x_in: np.ndarray, *, momentum_gev: float, scale_gev: float, zs_fm: float,
    kappa: float = 1., spectator_kappa: float = 1., threshold_cut: float = .4,
    cut_quadrature: str = "fractional", mu_min_gev: float = .6,
    evolution_points: int = 100, evolution_steps: int = 40, y_max: float = 100.,
    eps: float = 1e-12, eps_m: float = .005, soft_eps_m: float = .01, soft_zs_gev_inv: float = .01,
) -> np.ndarray:
    return _kernel(
        x_out, x_in, order="nlo", momentum_gev=momentum_gev, scale_gev=scale_gev, zs_fm=zs_fm,
        kappa=kappa, spectator_kappa=spectator_kappa, threshold_cut=threshold_cut,
        cut_quadrature=cut_quadrature, mu_min_gev=mu_min_gev, evolution_points=evolution_points,
        evolution_steps=evolution_steps, y_max=y_max, eps=eps, eps_m=eps_m,
        soft_eps_m=soft_eps_m, soft_zs_gev_inv=soft_zs_gev_inv, quadrature_order=64,
    )
