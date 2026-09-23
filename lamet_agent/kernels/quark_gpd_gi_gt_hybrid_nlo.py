"""Current-contract wrapper for the GI gamma^t quark GPD hybrid-NLO kernel.

The coefficient is the two-variable ``GI_gt_quark_GPD_hybrid_NLO`` kernel from
the project GPD matching implementation.  The public ``kernel`` callable uses
the lamet-agent contract and returns a matrix acting as ``lightcone = matrix @
quasi``.
"""

from __future__ import annotations

from typing import Final

import numpy as np


GEV_FM: Final[float] = 0.1973269631
CF: Final[float] = 4.0 / 3.0


def _beta(order: int, nf: int = 3) -> float:
    if order == 0:
        return 11.0 - 2.0 * nf / 3.0
    if order == 1:
        return 102.0 - 38.0 * nf / 3.0
    raise NotImplementedError(f"alpha_s order={order} is not implemented")


def _alphas_nloop(mu: float, order: int = 1, nf: int = 3) -> float:
    """Use the coupling convention of the original GPD implementation."""
    a_s_ref = 0.293 / (4.0 * np.pi)
    b0 = _beta(0, nf)
    temp = 1.0 + a_s_ref * b0 * np.log((mu / 2.0) ** 2)
    if order == 0:
        return a_s_ref * 4.0 * np.pi / temp
    if order == 1:
        b1 = _beta(1, nf)
        return a_s_ref * 4.0 * np.pi / (temp + a_s_ref * b1 / b0 * np.log(temp))
    raise NotImplementedError(f"alpha_s order={order} is not implemented")


def _sine_integral(value: float) -> float:
    try:
        from scipy.special import sici

        return float(sici(value)[0])
    except ModuleNotFoundError:
        pass
    if np.isclose(value, 0.0, atol=1e-14, rtol=0.0):
        return 0.0
    sign = 1.0 if value > 0.0 else -1.0
    upper = abs(value)
    n_steps = max(256, int(128 * upper))
    if n_steps % 2:
        n_steps += 1
    grid = np.linspace(0.0, upper, n_steps + 1)
    integrand = np.ones_like(grid)
    integrand[1:] = np.sin(grid[1:]) / grid[1:]
    h = upper / n_steps
    return sign * float(
        h
        / 3.0
        * (
            integrand[0]
            + integrand[-1]
            + 4.0 * np.sum(integrand[1:-1:2])
            + 2.0 * np.sum(integrand[2:-2:2])
        )
    )


def _gpd_log(value: float, momentum_gev: float, mu: float, eps: float) -> float:
    value_sq = max(float(value) ** 2, eps**2)
    return float(np.log(4.0 * momentum_gev**2 * value_sq / mu**2))


def _gpd_abs_log_piece(value: float, momentum_gev: float, mu: float, eps: float) -> float:
    magnitude = abs(float(value))
    if magnitude <= eps:
        return 0.0
    return float(magnitude * (_gpd_log(value, momentum_gev, mu, eps) - 1.0))


def _c_gpd_gt_hybrid(
    x: float,
    y: float,
    skewness: float,
    momentum_gev: float,
    mu: float,
    zspz: float,
    eps: float,
) -> float:
    """Regular off-diagonal coefficient from ``GI_gt_quark_GPD_hybrid_NLO``."""
    delta = x - y
    x1 = skewness + x
    x2 = skewness - x
    y1 = skewness + y
    y2 = skewness - y
    log_delta = _gpd_log(delta, momentum_gev, mu, eps)
    x1_piece = _gpd_abs_log_piece(x1, momentum_gev, mu, eps)
    x2_piece = _gpd_abs_log_piece(x2, momentum_gev, mu, eps)
    delta_piece = _gpd_abs_log_piece(delta, momentum_gev, mu, eps)

    term1 = x1_piece / (y1 * (y1 + y2))
    term2 = x2_piece / (y2 * (y1 + y2))
    term3 = -delta_piece / (y1 * y2)
    term_transversity = (
        x1_piece / (y1 * (y1 - x1))
        + x2_piece / (y2 * (y2 - x2))
        + (x1 / y1 + x2 / y2) * (log_delta - 1.0) / abs(delta)
    )
    term_hybrid = 6.0 * _sine_integral(delta * zspz) / (np.pi * delta)
    alpha_s = _alphas_nloop(mu, order=1, nf=3)
    return float(alpha_s * CF / (4.0 * np.pi) * (term1 + term2 + term3 + term_transversity + term_hybrid))


def _lo_interp_matrix(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(y_grid)
    ys = y_grid[order]
    lo_sorted = np.column_stack(
        [np.interp(x_grid, ys, unit, left=0.0, right=0.0) for unit in np.eye(len(y_grid))]
    )
    lo = np.empty_like(lo_sorted)
    lo[:, order] = lo_sorted
    return lo


def _build_matrix(
    x_out: np.ndarray,
    x_in: np.ndarray,
    momentum_gev: float,
    mu: float,
    zspz: float,
    skewness: float,
    eps: float,
) -> np.ndarray:
    x_grid = np.asarray(x_out, dtype=float)
    y_grid = np.asarray(x_in, dtype=float)
    if x_grid.ndim != 1 or y_grid.ndim != 1 or y_grid.size < 2:
        raise ValueError("x_out and x_in must be one-dimensional grids with at least two input points")
    if abs(skewness) <= eps:
        raise ValueError("skewness must be non-zero for this GPD kernel")
    if np.any(np.isclose(y_grid, skewness, atol=eps, rtol=0.0)) or np.any(
        np.isclose(y_grid, -skewness, atol=eps, rtol=0.0)
    ):
        raise ValueError("x_in must avoid y = +/-skewness")
    steps = np.diff(y_grid)
    dy = float(abs(steps[0]))
    if dy <= eps or not np.allclose(steps, steps[0], rtol=0.0, atol=eps):
        raise ValueError("x_in must be uniformly spaced")

    nlo = np.zeros((len(x_grid), len(y_grid)), dtype=float)
    identity = _lo_interp_matrix(x_grid, y_grid)
    diagonal_rows = np.abs(x_grid[:, None] - y_grid[None, :]).argmin(axis=0)
    for row, x_value in enumerate(x_grid):
        for column, y_value in enumerate(y_grid):
            if abs(x_value - y_value) <= eps * max(1.0, abs(y_value)):
                continue
            nlo[row, column] = _c_gpd_gt_hybrid(
                x_value, y_value, skewness, momentum_gev, mu, zspz, eps
            )
    for column, diagonal_row in enumerate(diagonal_rows):
        nlo[int(diagonal_row), column] -= np.sum(nlo[:, column])
    return identity - nlo * dy


def kernel(
    x_out: np.ndarray,
    x_in: np.ndarray,
    *,
    momentum_gev: float,
    scale_gev: float,
    zs_fm: float,
    skewness: float,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return the inverse-matching matrix on the supplied x grids."""
    zspz = float(zs_fm) * float(momentum_gev) / GEV_FM
    return _build_matrix(
        x_out,
        x_in,
        float(momentum_gev),
        float(scale_gev),
        zspz,
        float(skewness),
        float(eps),
    )


kernel.matching_structure = {
    "factorization": (
        r"F(x,\xi,t,\mu)=\int dy\,C^{-1}\!\left(x,y,\xi,\frac{\mu}{P_z},z_sP_z\right)"
        r"\widetilde F(y,\xi,t,P_z)+O(\alpha_s^2)."
    ),
    "result_noun": "light-cone GPD",
    "source_noun": "quasi-GPD",
    "notation": (
        "This is a genuine two-variable coefficient at fixed skewness; the "
        "off-diagonal coefficient is plus-prescribed column by column, and "
        "the returned matrix acts as lightcone = matrix @ quasi."
    ),
}
