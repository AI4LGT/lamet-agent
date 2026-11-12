from __future__ import annotations

from functools import lru_cache
from typing import Literal

import mpmath
import numpy as np

GEV_FM = 0.1973269631
Convention = Literal["author", "printed"]
Prescription = Literal["exponentiated", "strict_nlo"]


def _positive(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _coupling(alpha_s: float) -> float:
    if isinstance(alpha_s, (bool, np.bool_)) or not np.isfinite(alpha_s) or alpha_s < 0:
        raise ValueError("alpha_s must be finite and nonnegative")
    return float(alpha_s) / (4 * np.pi)


def _x(value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or not 0 < value < 1:
        raise ValueError("x must lie strictly inside (0, 1); endpoints are not regulated")
    return float(value)


@lru_cache(maxsize=8192)
def _coefficient(
    b_perp_fm: float, scale_gev: float, parton_momentum_gev: float, sign: int, convention: str, dps: int
) -> complex:

    mp = mpmath.mp.clone()
    mp.dps = dps
    b, mu, p, hc = map(mp.mpf, map(str, (b_perp_fm, scale_gev, parton_momentum_gev, GEV_FM)))
    r = p * b / hc
    z = r**2 / 4
    half = mp.mpf(1) / 2
    g1 = mp.meijerg([[3 * half, 3 * half], []], [[3 * half, 3 * half], [0, half]], z)
    g2 = mp.meijerg([[half, half, half], []], [[half, half], [-half, -half, 0]], z)
    prefactor = mp.sqrt(mp.pi) if convention == "author" else 1 / mp.sqrt(mp.pi)
    lb = 2 * mp.log(b * mu / (hc * 2 * mp.exp(-mp.euler)))
    lz = 2 * mp.log(2 * p / mu)
    real = prefactor * (g1 / r - r * g2 / 2) + lb**2 / 2 - lb * (1 - lz) + mp.pi**2 / 12
    imag = sign * mp.pi * (2 * mp.ei(-r) + mp.expm1(-r) / r - lz + 1)
    result = complex(mp.mpf(4) / 3 * (real + mp.j * imag))
    if not np.isfinite(result):
        raise ValueError("C_phi coefficient is nonfinite at the requested kinematics")
    return result


def c_phi_one_loop(
    parton_momentum_gev: float,
    *,
    b_perp_fm: float,
    scale_gev: float,
    sign: Literal[-1, 1],
    convention: Convention = "author",
    dps: int = 30,
) -> complex:
    b = _positive("b_perp_fm", b_perp_fm)
    mu = _positive("scale_gev", scale_gev)
    p = _positive("parton_momentum_gev", parton_momentum_gev)
    if isinstance(sign, (bool, np.bool_)) or sign not in (-1, 1):
        raise ValueError("sign must be -1 or +1 (Wilson-line orientation)")
    if convention not in ("author", "printed"):
        raise ValueError("convention must be 'author' or 'printed'")
    if isinstance(dps, bool) or not isinstance(dps, int) or dps < 20:
        raise ValueError("dps must be an integer >= 20")
    return _coefficient(b, mu, p, sign, convention, dps)


def _factor(correction: complex, prescription: str, *, inverse: bool = False) -> complex:
    if prescription not in ("exponentiated", "strict_nlo"):
        raise ValueError("prescription must be 'exponentiated' or 'strict_nlo'")
    exponent = -correction if inverse else correction
    result = complex(np.exp(exponent) if prescription == "exponentiated" else 1 + exponent)
    if not np.isfinite(result):
        raise ValueError("hard factor overflow; check kinematics and coupling")
    return result


def c_phi(
    parton_momentum_gev: float,
    *,
    b_perp_fm: float,
    scale_gev: float,
    alpha_s: float,
    sign: Literal[-1, 1],
    prescription: Prescription = "exponentiated",
    convention: Convention = "author",
    dps: int = 30,
) -> complex:
    correction = _coupling(alpha_s) * c_phi_one_loop(
        parton_momentum_gev,
        b_perp_fm=b_perp_fm,
        scale_gev=scale_gev,
        sign=sign,
        convention=convention,
        dps=dps,
    )
    return _factor(correction, prescription)


def _hard_correction(x: float, momentum_gev: float, alpha_s: float, **kwargs) -> complex:
    x = _x(x)
    p = _positive("momentum_gev", momentum_gev)
    return _coupling(alpha_s) * (c_phi_one_loop(x * p, **kwargs) + c_phi_one_loop((1 - x) * p, **kwargs))


def hard_factor(
    x: float,
    *,
    momentum_gev: float,
    scale_gev: float,
    b_perp_fm: float,
    alpha_s: float,
    sign: Literal[-1, 1],
    prescription: Prescription = "exponentiated",
    convention: Convention = "author",
    dps: int = 30,
) -> complex:
    correction = _hard_correction(
        x,
        momentum_gev,
        alpha_s,
        b_perp_fm=b_perp_fm,
        scale_gev=scale_gev,
        sign=sign,
        convention=convention,
        dps=dps,
    )
    return _factor(correction, prescription)


def delta_gamma(
    x: float,
    *,
    momentum1_gev: float,
    momentum2_gev: float,
    scale_gev: float,
    b_perp_fm: float,
    alpha_s: float,
    sign: Literal[-1, 1],
    logarithm: Literal["expanded", "principal"] = "expanded",
    convention: Convention = "author",
    dps: int = 30,
) -> complex:
    x = _x(x)
    p1 = _positive("momentum1_gev", momentum1_gev)
    p2 = _positive("momentum2_gev", momentum2_gev)
    if p1 == p2:
        raise ValueError("the two momenta must differ")
    if logarithm not in ("expanded", "principal"):
        raise ValueError("logarithm must be 'expanded' or 'principal'")
    coupling = _coupling(alpha_s)
    kwargs = dict(b_perp_fm=b_perp_fm, scale_gev=scale_gev, sign=sign, convention=convention, dps=dps)
    total = 0j
    for fraction in (x, 1 - x):
        difference = coupling * (c_phi_one_loop(fraction * p1, **kwargs) - c_phi_one_loop(fraction * p2, **kwargs))
        if logarithm == "principal":
            difference = complex(difference.real, np.angle(np.exp(1j * difference.imag)))
        total += difference
    denominator = np.log1p((p1 - p2) / p2) if abs(p1 - p2) < p2 / 2 else np.log(p1) - np.log(p2)
    return -total / denominator


def kernel(
    x_out: np.ndarray,
    x_in: np.ndarray,
    *,
    momentum_gev: float,
    scale_gev: float,
    b_perp_fm: float,
    alpha_s: float,
    sign: Literal[-1, 1],
    prescription: Prescription = "exponentiated",
    convention: Convention = "author",
    dps: int = 30,
) -> np.ndarray:
    incoming, outgoing = np.asarray(x_in, dtype=float), np.asarray(x_out, dtype=float)
    for name, grid in (("x_in", incoming), ("x_out", outgoing)):
        if grid.ndim != 1 or not grid.size or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0):
            raise ValueError(f"{name} must be a nonempty finite strictly increasing grid")
    if np.any((outgoing <= 0) | (outgoing >= 1)):
        raise ValueError("x_out must lie strictly inside (0, 1)")
    indices = np.searchsorted(incoming, outgoing)
    if np.any(indices >= len(incoming)) or not np.array_equal(incoming[indices], outgoing):
        raise ValueError("x_out must be an exact subset of x_in; no implicit interpolation")
    matrix = np.zeros((len(outgoing), len(incoming)), dtype=complex)
    for row, (x, column) in enumerate(zip(outgoing, indices)):
        correction = _hard_correction(
            float(x),
            momentum_gev,
            alpha_s,
            b_perp_fm=b_perp_fm,
            scale_gev=scale_gev,
            sign=sign,
            convention=convention,
            dps=dps,
        )
        matrix[row, column] = _factor(correction, prescription, inverse=True)
    return matrix
