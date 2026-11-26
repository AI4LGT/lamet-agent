"""Unpolarized gamma-t C0 at NNLO, arXiv:2312.10829 Eq. (5)."""

import numpy as np

from .implementation import CF, HBAR_C_GEV_FM, alpha_s


def kernel(z_fm: np.ndarray | float, mu: float = 2.0, n_f: int = 3) -> np.ndarray:
    if isinstance(n_f, bool) or n_f not in (3, 4, 5):
        raise ValueError("n_f must be 3, 4 or 5")
    z = np.abs(np.asarray(z_fm, dtype=float)) / HBAR_C_GEV_FM
    if np.any(~np.isfinite(z)) or np.any(z <= 0) or not np.isfinite(mu) or mu <= 0:
        raise ValueError("C0 requires finite nonzero z and positive mu")
    # Retain the project's threshold-matched coupling convention; n_f controls
    # the explicit fermion-loop coefficients, as in the matching kernels.
    coupling = alpha_s(3, float(mu), "running") / (2 * np.pi)
    log = np.log(z**2 * mu**2 * np.exp(2 * np.euler_gamma) / 4)
    return (
        1
        + coupling * CF * (1.5 * log + 2.5)
        + coupling**2 * ((7.5 - n_f / 3) * log**2 + (37.1731 - 5 * n_f / 3) * log + 51.836 - 4.34259 * n_f)
    )
