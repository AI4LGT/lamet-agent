"""Regress same-flow reflection without changing the existing endpoint labels."""

import numpy as np
import pytest

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.parallel import fourier_transform
from lamet_agent.stages.fourier_transform._inspection import _gpd_signed_pair


@pytest.mark.parametrize("label", ["barpsi_at_0", "mid_at_0", "psi_at_0"])
@pytest.mark.parametrize("delta", [-0.4, 0.0, 0.4])
@pytest.mark.parametrize("include_origin", [False, True])
def test_gpd_completion_preserves_positive_branch_and_real_fourier(label, delta, include_origin):
    z = np.round(np.arange(13) * 0.1, 12)
    if not include_origin:
        z = z[1:]
    primary = np.exp(-(z / 0.7) ** 2) * np.exp(-0.55j * z / HBAR_C_GEV_FM)
    # A compatible exchanged flow; both flows transform to real Gaussians.
    partner = np.exp(1j * delta * z / HBAR_C_GEV_FM) * primary
    primary = np.array([primary, 1.1 * primary])
    partner = np.array([partner, 1.1 * partner])
    if include_origin:
        # Independent imaginary origin noise must not survive odd completion.
        primary[:, 0] += 0.2j
        partner[:, 0] -= 0.4j
    def data(values):
        return EnsembleData(None, "bootstrap", list(values), ["z"], {"z": z.tolist()})

    result = _gpd_signed_pair(
        data(primary), data(partner), phase_transfer_gpd=label, delta_momentum_gev=delta,
    )
    values = np.asarray(result.values)
    signed_z = np.asarray(result.coords["z"])
    # Independent expectations for the ORIGINAL positive-z operations.
    if label == "mid_at_0":
        expected_positive = primary * np.exp(0.5j * delta * z / HBAR_C_GEV_FM)
    elif label == "psi_at_0":
        expected_positive = partner.conjugate()
    else:
        expected_positive = primary
    np.testing.assert_allclose(values[:, signed_z > 0], expected_positive[:, z > 0])
    np.testing.assert_allclose(values[:, signed_z < 0], expected_positive[:, z > 0][:, ::-1].conjugate())
    np.testing.assert_allclose(values[:, ::-1], values.conjugate())
    if include_origin:
        expected_origin = np.real(0.5 * (primary[:, 0] + partner[:, 0].conjugate()))
        np.testing.assert_allclose(values[:, signed_z == 0].ravel(), expected_origin)
    transformed = fourier_transform(
        result, [-0.7, -0.2, 0.0, 0.15, 0.6], momentum_gev=1.0, prefactor="pz_over_2pi",
    )
    np.testing.assert_allclose(np.imag(transformed.values), 0.0, atol=1e-14)
    assert result.attrs["signed_z_completion"] == "gpd_same_flow_conjugation"
    assert result.attrs["phase_transfer_gpd"] == label
    assert result.attrs["gpd_completion_mode"] == "paired_flow"
