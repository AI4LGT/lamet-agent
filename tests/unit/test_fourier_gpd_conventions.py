"""Check endpoint translations and real-distribution completion independently."""

from types import SimpleNamespace

import numpy as np
import pytest

from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.parallel import fourier_transform
from lamet_agent.stages.fourier_transform._inspection import prepare
from lamet_agent.stages.fourier_transform.physics import (
    complete_signed_z,
    extend_tail,
    tail_model_values,
)


LABELS = ["psi_at_0", "mid_at_0", "barpsi_at_0"]


def _context(label, *, observable="gpd", partner=False, signed=False, forward=False):
    ensemble = EnsembleInfo("test", "conventions", 0.1, 0.1, 64, 128, 0.14)
    z = np.array([0.0, 0.1, 0.2, 0.3])
    values = np.array([1.0 + 0.3j, 0.8 + 0.2j, 0.5 - 0.3j, 0.2 + 0.1j])
    if signed:
        z = np.concatenate([-z[1:][::-1], z])
        values = np.concatenate([np.full(3, 70.0 + 40.0j), values])
    attrs = {
        "coord_unit": "fm", "momentum_gev": 1.0, "parton": "quark",
        "gfix": "GI", "polarization": "unpolarized", "hadron": "pion",
        "source_momentum": [0, 0, 2], "sink_momentum": [0, 0, 2 if forward else 3],
    }
    data = EnsembleData(
        ensemble, "bootstrap", [values, 1.1 * values], ["z"], {"z": z.tolist()}, attrs=attrs,
    )
    inputs = {"input": data}
    if partner:
        inputs["hermitian_partner"] = EnsembleData(
            ensemble, "bootstrap", [9.0 * values, -7.0 * values], ["z"], {"z": z.tolist()},
            attrs={**attrs, "source_momentum": attrs["sink_momentum"],
                   "sink_momentum": attrs["source_momentum"]},
        )
    return SimpleNamespace(
        inputs=inputs, state={}, params={"phase_transfer_gpd": label, "scheme_scan": {"sector": "full"}},
        manifest={"metadata": {"target_observable": observable, "parton": "quark"}},
    )


@pytest.mark.parametrize("observable", ["pdf", "da", "gpd"])
@pytest.mark.parametrize("signed", [False, True])
def test_all_observables_complete_from_the_same_nonnegative_branch(observable, signed):
    context = _context("mid_at_0", observable=observable, signed=signed)
    prepared, _ = prepare(context)
    z = np.asarray(prepared.coords["z"])
    values = np.asarray(prepared.values)
    np.testing.assert_allclose(values[:, z < 0], values[:, z > 0][:, ::-1].conjugate())
    np.testing.assert_array_equal(np.imag(values[:, z == 0]), 0.0)
    assert np.max(np.abs(values)) < 2.0


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("forward", [False, True])
def test_endpoint_preparation_uses_paper_input_without_a_partner(label, forward):
    context = _context(label, forward=forward)
    primary = context.inputs["input"]
    z = np.asarray(primary.coords["z"])
    paper = np.asarray(primary.values).copy()
    delta = primary.ensemble.k_s * (0 if forward else 1)
    if label == "mid_at_0":
        expected = np.exp(0.5j * delta * z / HBAR_C_GEV_FM) * paper
    elif label == "barpsi_at_0":
        expected = np.exp(-1j * delta * z / HBAR_C_GEV_FM) * paper.conjugate()
    else:
        expected = paper
    expected[:, 0] = expected[:, 0].real
    prepared, _ = prepare(context)
    output_z = np.asarray(prepared.coords["z"])
    np.testing.assert_allclose(np.asarray(prepared.values)[:, output_z >= 0], expected)
    assert prepared.attrs["gpd_completion_mode"] == "single_flow"


@pytest.mark.parametrize("label", LABELS)
def test_optional_partner_values_do_not_change_completion(label):
    without, _ = prepare(_context(label))
    with_partner, _ = prepare(_context(label, partner=True))
    np.testing.assert_array_equal(without.values, with_partner.values)
    assert with_partner.attrs["hermitian_partner_used_for_completion"] == "false"


@pytest.mark.parametrize("label", LABELS)
@pytest.mark.parametrize("order", ["LA", "NLA"])
@pytest.mark.parametrize("model", ["gi_nla", "cg_nla"])
@pytest.mark.parametrize("hadron,sector", [("pion", "full"), ("pion", "sea"), ("proton", "full")])
def test_tail_matches_paper_and_literal_endpoint_translations(label, order, model, hadron, sector):
    z = np.array([-0.7, -0.3, 0.3, 0.7])
    initial, final = 0.9, 1.5
    delta = final - initial
    terms = ["1", "3", "2", "t2"] if (hadron, sector) == ("pion", "full") else ["2", "t2"]
    frequencies = {"1": -final, "3": initial, "2": 0.0, "t2": -delta}
    parameters = {"Lambda": 0.6}
    paper = np.zeros(z.size, dtype=complex)
    for index, term in enumerate(terms):
        amplitude, phase = 0.2 + 0.1 * index, -0.15 + 0.2 * index
        parameters.update({f"A{term}": amplitude, f"phi{term}": phase})
        paper += amplitude * np.exp(1j * (np.sign(z) * phase + frequencies[term] * z / HBAR_C_GEV_FM))
        if order == "NLA":
            prime_amplitude, prime_phase = -0.04 + 0.01 * index, 0.35 - 0.1 * index
            parameters.update({f"A{term}p": prime_amplitude, f"phi{term}p": prime_phase})
            paper += prime_amplitude * np.exp(
                1j * (np.sign(z) * prime_phase + frequencies[term] * z / HBAR_C_GEV_FM)
            ) / np.abs(z)
    paper *= np.exp(-parameters["Lambda"] * np.abs(z) / HBAR_C_GEV_FM)
    if model == "cg_nla":
        parameters["n"] = 1.3
        paper /= np.abs(z) ** parameters["n"]
    if label == "mid_at_0":
        expected = np.exp(0.5j * delta * z / HBAR_C_GEV_FM) * paper
    elif label == "barpsi_at_0":
        expected = np.exp(-1j * delta * z / HBAR_C_GEV_FM) * paper[::-1]
    else:
        expected = paper
    actual = tail_model_values(
        z, model, parameters, order=order, observable="GPD", hadron=hadron, sector=sector,
        initial_momentum_gev=initial, final_momentum_gev=final, delta_momentum_gev=delta,
        phase_transfer_gpd=label,
    )
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    np.testing.assert_allclose(actual[::-1], actual.conjugate(), atol=1e-14)


@pytest.mark.parametrize("label", LABELS)
def test_preparation_tail_extension_and_fourier_preserve_reality(label):
    prepared, _ = prepare(_context(label))
    parameters = {"A1": 0.3, "phi1": 0.2, "A3": 0.4, "phi3": -0.1,
                  "A2": 0.5, "phi2": 0.3, "At2": -0.2, "phit2": -0.4, "Lambda": 0.6}
    extended = extend_tail(
        prepared, z_max_fm=0.8, z_min_fm=0.1, smoothing_method="linear", smoothing_width_fm=0.2,
        model_id="gi_nla", tail_parameters=[parameters, parameters], order="LA", observable="GPD",
        hadron="pion",
    )
    np.testing.assert_allclose(np.asarray(extended.values)[:, ::-1], np.asarray(extended.values).conjugate())
    # Evaluate the package's actual transform on a nonuniform momentum grid.
    result = fourier_transform(
        extended, x_grid=[-0.7, -0.2, 0.0, 0.15, 0.6], momentum_gev=1.0,
    )
    np.testing.assert_allclose(np.imag(result.values), 0.0, atol=1e-14)


def test_explicit_low_level_completion_remains_available():
    data = _context("mid_at_0", signed=True).inputs["input"]
    explicit = complete_signed_z(data, {"real": "explicit", "imag": "explicit"})
    np.testing.assert_array_equal(explicit.values, data.values)
