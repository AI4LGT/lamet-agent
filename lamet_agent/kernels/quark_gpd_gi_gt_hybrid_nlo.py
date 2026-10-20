"""GI gamma^t hybrid-NLO inverse matching on nonuniform quasi-GPD grids.

Real and virtual terms share the exclusion |y-x| < epsilon2, following the
ordered-limit prescription in arXiv:2609.20545, Eqs. (9)-(10). The existing
hybrid coefficient and coupling convention are retained. The returned matrix
acts as ``lightcone = matrix @ quasi``.
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
        h / 3.0 * (integrand[0] + integrand[-1] + 4.0 * np.sum(integrand[1:-1:2]) + 2.0 * np.sum(integrand[2:-2:2]))
    )


def _gpd_log(value: float, momentum_gev: float, mu: float, eps: float) -> float:
    value_sq = max(float(value) ** 2, eps**2)
    return float(np.log(4.0 * momentum_gev**2 * value_sq / mu**2))


def _gpd_abs_log_piece(value: float, momentum_gev: float, mu: float, eps: float) -> float:
    magnitude = abs(float(value))
    if magnitude <= eps:
        return 0.0
    return float(magnitude * (_gpd_log(value, momentum_gev, mu, eps) - 1.0))


def _abs_log_divided_difference(
    a: float,
    b: float,
    momentum_gev: float,
    mu: float,
    eps: float,
) -> float:
    """Evaluate [A(a)-A(b)]/(a-b), including its removable same-sign limit."""
    if a == b:
        if abs(a) <= eps:
            raise ValueError("divided difference at the logarithmic cusp is undefined")
        return float(np.sign(a) * (_gpd_log(a, momentum_gev, mu, eps) + 1.0))
    if a * b > 0.0 and min(abs(a), abs(b)) > eps and abs(a - b) < 0.01 * abs(b):
        ratio = (a - b) / b
        return float(
            np.sign(b) * (_gpd_log(b, momentum_gev, mu, eps) - 1.0 + 2.0 * (1.0 + ratio) * np.log1p(ratio) / ratio)
        )
    return (_gpd_abs_log_piece(a, momentum_gev, mu, eps) - _gpd_abs_log_piece(b, momentum_gev, mu, eps)) / (a - b)


def _c_gpd_gt_hybrid(
    x: float,
    y: float,
    skewness: float,
    momentum_gev: float,
    mu: float,
    zspz: float,
    eps: float,
) -> float:
    """Original off-diagonal coefficient, with removable y=+/-xi poles combined.

    Writing A(v)=|v|[log(4 Pz^2 v^2/mu^2)-1], set
    D1=[A(x+xi)-A(x-y)]/(y+xi),
    D2=[A(xi-x)-A(x-y)]/(xi-y).
    Then the original first three terms are (D1+D2)/(2 xi), and the
    transversity-named piece is (D2-D1)/(x-y)+2 A(x-y)/(x-y)^2.
    """
    delta = x - y
    if abs(delta) <= eps:
        raise ValueError("off-diagonal coefficient requires |x-y| > eps")
    d1 = _abs_log_divided_difference(x + skewness, delta, momentum_gev, mu, eps)
    d2 = _abs_log_divided_difference(skewness - x, -delta, momentum_gev, mu, eps)
    scalar = (d1 + d2) / (2.0 * skewness)
    transversity = (d2 - d1) / delta + 2.0 * _gpd_abs_log_piece(delta, momentum_gev, mu, eps) / delta**2
    hybrid = 6.0 * _sine_integral(delta * zspz) / (np.pi * delta)
    alpha_s = _alphas_nloop(mu, order=1, nf=3)
    return float(alpha_s * CF / (4.0 * np.pi) * (scalar + transversity + hybrid))


def _lo_interp_matrix(x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    order = np.argsort(y_grid)
    ys = y_grid[order]
    lo_sorted = np.column_stack([np.interp(x_grid, ys, unit, left=0.0, right=0.0) for unit in np.eye(len(y_grid))])
    lo = np.empty_like(lo_sorted)
    lo[:, order] = lo_sorted
    return lo


def _basis_weights(value: float, grid: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation weights on a strictly increasing grid."""
    result = np.zeros(grid.size)
    if value < grid[0] or value > grid[-1]:
        return result
    right = min(max(int(np.searchsorted(grid, value, side="right")), 1), grid.size - 1)
    left = right - 1
    fraction = (value - grid[left]) / (grid[right] - grid[left])
    result[left], result[right] = 1.0 - fraction, fraction
    return result


def _build_matrix(
    x_out: np.ndarray,
    x_in: np.ndarray,
    momentum_gev: float,
    mu: float,
    zspz: float,
    skewness: float,
    eps: float,
    epsilon2: float,
    hierarchy_ratio: float,
    quad_epsabs: float,
    quad_epsrel: float,
) -> np.ndarray:
    from scipy.integrate import quad_vec

    x_grid, y_grid = np.asarray(x_out, dtype=float), np.asarray(x_in, dtype=float)
    if x_grid.ndim != 1 or y_grid.ndim != 1 or x_grid.size == 0 or y_grid.size < 2:
        raise ValueError("nonempty one-dimensional grids and at least two input points are required")
    if not np.all(np.isfinite(x_grid)) or not np.all(np.isfinite(y_grid)):
        raise ValueError("grids must contain finite values")
    if np.any(np.diff(y_grid) <= 0.0):
        raise ValueError("x_in must be strictly increasing; nonuniform spacing is supported")
    if abs(skewness) <= eps:
        raise ValueError("skewness must be non-zero for this GPD kernel")
    if epsilon2 <= 100.0 * eps:
        raise ValueError("epsilon2 must exceed 100*eps; eps is a numerical tolerance, not the cutoff")
    if not 0.0 < hierarchy_ratio <= 0.01:
        raise ValueError("hierarchy_ratio must lie in (0, 0.01]")
    distances = np.minimum(abs(x_grid - skewness), abs(x_grid + skewness))
    if np.any(distances <= eps):
        raise ValueError("x_out must avoid +/-skewness; use one-sided limits after cutoff convergence")
    if epsilon2 > hierarchy_ratio * float(np.min(distances)):
        raise ValueError("epsilon2 must be <= hierarchy_ratio * min_x |x +/- skewness|")
    # Both sides of the symmetric diagonal exclusion must lie in the data window.
    if np.any(x_grid - epsilon2 <= y_grid[0]) or np.any(x_grid + epsilon2 >= y_grid[-1]):
        raise ValueError("x_out +/- epsilon2 must lie strictly inside the x_in range")

    matrix = _lo_interp_matrix(x_grid, y_grid)
    for row, x in enumerate(x_grid):
        interpolation = matrix[row].copy()

        def integrand(y, x=x, interpolation=interpolation):
            # The subtraction is integrated at the same y and with the same cutoff.
            real = _c_gpd_gt_hybrid(x, y, skewness, momentum_gev, mu, zspz, eps)
            virtual = _c_gpd_gt_hybrid(y, x, skewness, momentum_gev, mu, zspz, eps)
            return real * _basis_weights(y, y_grid) - virtual * interpolation

        for lower, upper in ((y_grid[0], x - epsilon2), (x + epsilon2, y_grid[-1])):
            # Split at interpolation knots and at the coefficient's logarithmic cusps.
            points = np.unique(np.concatenate((y_grid, [-skewness, skewness])))
            points = points[(points > lower) & (points < upper)]
            correction, _, info = quad_vec(
                integrand,
                lower,
                upper,
                points=points,
                norm="max",
                epsabs=quad_epsabs / 2.0,
                epsrel=quad_epsrel,
                limit=max(10000, 4 * (len(points) + 1)),
                full_output=True,
            )
            if not info.success:
                raise RuntimeError(f"matching quadrature failed at x={x}: {info.message}")
            matrix[row] -= correction
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError("matching produced nonfinite coefficients")
    return matrix


def kernel(
    x_out: np.ndarray,
    x_in: np.ndarray,
    *,
    momentum_gev: float,
    scale_gev: float,
    zs_fm: float,
    skewness: float,
    eps: float = 1e-12,
    epsilon2: float = 1e-5,
    hierarchy_ratio: float = 0.01,
    quad_epsabs: float = 1e-8,
    quad_epsrel: float = 1e-6,
) -> np.ndarray:
    """Return a finite-window inverse-matching matrix on nonuniform input grids.

    x_in must be strictly increasing; x_out may be uniform or arbitrary.
    The input values are interpolated linearly inside [x_in[0], x_in[-1]].
    Both real and virtual integrals use this finite window and exclude
    |y-x| < epsilon2. Widen the input window to check tail truncation.

    Require epsilon2 <= hierarchy_ratio * min_i |x_out[i] +/- skewness|.
    The default ratio 0.01 is a guard, not a convergence guarantee. First
    reduce epsilon2 at fixed output points and input data, then approach
    +/-skewness. No crossover extrapolation or forced continuity is applied.

    Input refinement (e.g. h/10 near +/-skewness) belongs to the caller;
    adaptive quadrature resolves the cutoff independently of input spacing.
    eps controls only numerical near-zero handling. Quadrature tolerances
    bound integration error estimates, not input-interpolation or tail errors.
    """
    controls = {
        "momentum_gev": momentum_gev,
        "scale_gev": scale_gev,
        "zs_fm": zs_fm,
        "eps": eps,
        "epsilon2": epsilon2,
        "quad_epsabs": quad_epsabs,
        "quad_epsrel": quad_epsrel,
    }
    for name, value in controls.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(skewness) or not np.isfinite(hierarchy_ratio):
        raise ValueError("skewness and hierarchy_ratio must be finite")
    coupling = _alphas_nloop(float(scale_gev))
    if not np.isfinite(coupling) or coupling <= 0.0:
        raise ValueError("scale_gev is outside the supported perturbative coupling domain")
    zspz = float(zs_fm) * float(momentum_gev) / GEV_FM
    return _build_matrix(
        x_out,
        x_in,
        float(momentum_gev),
        float(scale_gev),
        zspz,
        float(skewness),
        float(eps),
        float(epsilon2),
        float(hierarchy_ratio),
        float(quad_epsabs),
        float(quad_epsrel),
    )


kernel.matching_structure = {
    "factorization": (
        r"F(x,\xi,t,\mu)=\int dy\,C^{-1}(x,y,\xi,\mu/P_z,z_sP_z)"
        r"\widetilde F(y,\xi,t,P_z)+O(\alpha_s^2)."
    ),
    "result_noun": "light-cone GPD",
    "source_noun": "quasi-GPD",
    "notation": (
        "Piecewise-linear quasi-GPD interpolation on a nonuniform input grid; "
        "real and virtual terms share |y-x|>=epsilon2 and the finite input window. "
        "The cutoff must be much smaller than the distance to +/-skewness. "
        "lightcone = matrix @ quasi."
    ),
}

