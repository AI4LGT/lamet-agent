"""Wilson-loop-divided zero-momentum coefficient from TMD_V3.nb.

Massless pure-DR, nonsinglet gamma-t prescription. The finite same-side
remainder is analytic; this is not a finite-external-momentum limit.
"""

from functools import lru_cache

import numpy as np
from scipy.special import spence, zeta
from mpmath import mp


@lru_cache(maxsize=256)
def analytic_same_side_sum(r):
    """Closed Sigma(r) from TMD_V3.nb, qTMDNNLOClosed-2.

    Principal polylogarithms, finite real r=z/b. Extra working precision
    protects the cancellations at large |r|. Numerical quadrature below is
    retained only as an independent diagnostic, not used in extraction.
    """
    r = abs(float(r))
    if not np.isfinite(r):
        raise ValueError("finite real r is required")
    ctx = mp.clone()
    ctx.dps = 40 + 4 * max(0, int(np.ceil(np.log10(max(1, r)))))
    theta = ctx.atan(r)
    w = -ctx.exp(2j * theta)
    result = (3*ctx.pi**2/2 - 4*ctx.zeta(3) - 2*theta**2
              - 2*(ctx.pi**2-4*theta**2)*ctx.log(2*ctx.cos(theta))
              + 8*theta*ctx.im(ctx.polylog(2, w))
              + 4*ctx.re(ctx.polylog(3, w)))
    return float(result)


@lru_cache(maxsize=16)
def _rule(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) / 2, w / 2


def _kernel(xs, ys, n):
    """Vectorized ordered source integral, final axis is the two-vector."""
    xs, ys = np.broadcast_arrays(np.asarray(xs, float), np.asarray(ys, float))
    aa = np.sum(ys**2, axis=-1)[..., None]
    bb = np.sum(xs**2, axis=-1)[..., None]
    dd = np.sum((ys - xs)**2, axis=-1)[..., None]
    ratio = aa / bb
    lr = np.log(ratio)
    ends = [np.minimum(lr, 0) - 32, np.minimum(lr, 0), np.maximum(lr, 0), np.maximum(lr, 0) + 32]
    nodes, weights = _rule(n)
    total = np.zeros(xs.shape[:-1])
    for low, high in zip(ends[:-1], ends[1:]):
        x = np.exp(low + (high - low) * nodes)
        q, c = 1 + x, aa + x * bb
        rho = np.maximum(x * dd / (q * c), 1e-300)
        delta = rho - 1
        near = np.abs(delta) < 0.05
        # Evaluate the rational form away from its removable singularity.
        safe = np.where(near, 0.5, rho)
        l1 = (1 - safe + safe * np.log(safe)) / (1 - safe)**2
        l20 = (1 - safe**2 + 2 * safe * np.log(safe)) / (1 - safe)**3
        l21 = -((1 + safe) * np.log(safe) + 2 * (1 - safe)) / (1 - safe)**3
        d = delta[near]
        l1[near] = sum((-d)**k / ((k + 1) * (k + 2)) for k in range(12))
        l20[near] = sum(2 * (-d)**k / ((k + 2) * (k + 3)) for k in range(12))
        l21[near] = sum((k + 1) * (-d)**k / ((k + 2) * (k + 3)) for k in range(12))
        y = x / (ratio + x)
        s = 1 - y
        near = s < 0.01
        safe = np.where(near, 0.5, y)
        ed = -(1 - safe + np.log(safe)) / (1 - safe)**2
        ed[near] = sum(s[near]**k / (k + 2) for k in range(14))
        ux = ys[..., 0, None] + x * xs[..., 0, None]
        uy = ys[..., 1, None] + x * xs[..., 1, None]
        t0 = uy * xs[..., 1, None] - ux * xs[..., 0, None]
        t1 = uy * (xs - ys)[..., 1, None] - ux * (xs - ys)[..., 0, None]
        term = c / q * l1 + t0 / q * l20 + t1 / q**2 * l21 - ys[..., 1, None] * xs[..., 1, None] * ed
        total += np.sum((high - low) * weights * ratio * x / (ratio + x)**2 * term, axis=-1)
    return -4 * total / (aa * bb)[..., 0]


def _same(r, radial_order, kernel_order, log_range):
    beta = 1 / np.hypot(1, r)
    delta = r * beta
    qn, qw = _rule(radial_order)
    logv = np.concatenate((log_range * (qn - 1), log_range * qn))
    wlog = np.tile(qw, 2) * log_range
    den = qn**2 + (1 - qn)**2
    qq = qn**2 / den
    wratio = qw * 2 * qn * (1 - qn) / den**2
    hr = -(1 - qq) / qq * np.log1p(-qq) - (2 - qq) / (1 - qq) * np.log(qq)
    total = 0.0
    for v, weight in zip(np.exp(logv), wlog):
        u = v * qq
        xu = np.stack((np.full_like(u, beta), delta + u), axis=-1)
        un = np.stack((np.zeros_like(u), u), axis=-1)
        xv, vn = np.array([beta, delta + v]), np.array([0, v])
        bracket = _kernel(xu, xv, kernel_order) - _kernel(xu, vn, kernel_order)
        bracket -= _kernel(xv, un, kernel_order)
        if v > 1:
            bracket -= 4 * hr / v**2
        total += weight * np.sum(wratio * v**2 * bracket) / 2
    return float(total)


@lru_cache(maxsize=256)
def same_side_sum(r, radial_order=48, kernel_order=64, log_range=12.0):
    """S_A(r)+S_A(-r); refine all controls for a convergence assessment."""
    if not np.isfinite(r) or not np.isfinite(log_range) or not 0 < log_range <= 20:
        raise ValueError("finite r and 0 < log_range <= 20 are required")
    for n in (radial_order, kernel_order):
        if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 256:
            raise ValueError("quadrature orders must be integers between 2 and 256")
    r = abs(float(r))
    one = _same(r, radial_order, kernel_order, log_range)
    return one + (_same(-r, radial_order, kernel_order, log_range) if r else one)


def color_coefficients(r, ell, same):
    """Return HFF, HFA, HFf in the notebook's alpha_s/(4*pi) convention."""
    r, ell, same = np.broadcast_arrays(r, ell, same)
    theta = np.arctan(r)
    a = r * theta
    d = 2 * theta * np.log(2 / np.hypot(1, r)) - spence(1 - np.exp(1j * (np.pi - 2 * theta))).imag
    m = 4/3 + 4*np.pi**2/9 + 4*a - ((1+r*r)*a+r*r)*(np.pi**2-4*theta**2)/3 + 8*r*r*theta**2/3
    e = -4*same - 11*np.pi**2/3 - 4*theta**2 + 4*r*theta*(np.pi**2-4*theta**2)/3
    hff = (9*ell**2/2 + (0.5+8*np.pi**2/3-12*a)*ell + (1-4*a)**2/2
           - 32*zeta(3) - 77/24 - np.pi**2/9 + e + 4*theta**2 + m)
    comb = (53*ell**2/8 + (287/24-53*a/3)*ell + 1883/96 - 17*np.pi**2/9
            + 13*np.pi**2*r*r/12 + (np.pi**2*(r*r-3)/12-214/9)*a
            + ((1-r*r/3)*a-6-3*r*r)*theta**2 - 44*r*d/3)
    hfferm = -2*ell**2 + (16*a/3-14/3)*ell - 13/2 + 80*a/9 + 16*r*d/3
    return hff, -hff/4 + comb, hfferm


@lru_cache(maxsize=256)
def checked_same_side_sum(r):
    """Independent refinement checks, not a certified quadrature error bound."""
    r = abs(float(r))
    base = same_side_sum(r)
    controls = [(72, 64, 12.0), (48, 96, 12.0), (48, 64, 14.0), (72, 96, 14.0)]
    refined = [same_side_sum(r, *control) for control in controls]
    if not np.all(np.isfinite([base, *refined])) or max(abs(x - base) for x in refined) > 1e-5:
        raise ValueError(f"NNLO same-side quadrature failed refinement check at z/b={r}; revise the window")
    return refined[-1]
