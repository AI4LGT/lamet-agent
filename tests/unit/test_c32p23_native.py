"""Preserve resample identities and complex amplitudes across native adapters."""

import json

import h5py
import numpy as np
import pytest

from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.parallel.fourier import fourier_transform
from lamet_agent.stages.correlator_analysis._input import load_descriptor, resample_correlators
from lamet_agent.stages.correlator_analysis.physics import fit_qda_samples
from lamet_agent.stages.fourier_transform._inspection import derive_conventions
from lamet_agent.stages.fourier_transform.physics import complete_signed_z
from lamet_agent.stages.tmdwf_analysis.native import combine_transforms, prepare_projection
from lamet_agent.stages.tmdwf_analysis.physics import ensemble


INFO = EnsembleInfo("test", "test", 0.1, 0.1, 32, 64, 0.23)


def descriptor(tmp_path, precomputed=True):
    rng = np.random.default_rng(31)
    local = np.exp(-0.3 * np.arange(1, 9))[None, :] * rng.normal(1, 0.05, (40, 8))
    ratio = np.array([0.6 + 0.2j, 0.4 + 0.1j, 0.2 - 0.1j])[None, None, :]
    numerator = local[:, :, None] * (ratio + rng.normal(0, 0.01, (40, 8, 3))
                                    + 1j * rng.normal(0, 0.01, (40, 8, 3)))
    sample_dim = "bootstrap" if precomputed else "configuration"
    records = []
    with h5py.File(tmp_path / "source.h5", "w") as f:
        for name, values in (("nonlocal", numerator), ("local", local)):
            f.create_dataset(name, data=values)
            dims = [sample_dim, "t", "z"] if name == "nonlocal" else [sample_dim, "t"]
            record = {"id": name, "ensemble": INFO._asdict(), "count": 40, "format": "hdf5",
                      "path": "source.h5", "dataset": name, "dataset_dims": dims, "dims": dims,
                      "coords": {"t": list(range(1, 9)), **({"z": [0, 1, 2]} if name == "nonlocal" else {})},
                      "selectors": {"gfix": "GI", "source_operator": "gtg5", "bT": 1, "zdir": 0},
                      "correlator_type": "qda" if name == "nonlocal" else "two_point",
                      "hadron": {"name": "pion"}, "source_momentum": [0, 0, 4], "sink_momentum": [0, 0, 4],
                      "current": {"kernel_operator": "tmdwf", "parton": "quark", "renormalization_scheme": "bare"}
                      if name == "nonlocal" else None, "source_sink_separation": None}
            if precomputed:
                record["sampling"] = {"mode": "bootstrap", "resample_id": "paired"}
            records.append(record)
    path = tmp_path / "descriptor.json"
    path.write_text(json.dumps({"correlators": records}))
    return load_descriptor(path), numerator, local


def test_descriptor_does_not_rebootstrap_or_reorder(tmp_path, monkeypatch):
    loaded, numerator, local = descriptor(tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("precomputed input was bootstrapped a second time")

    monkeypatch.setattr(EnsembleData, "bootstrap", forbidden)
    output = resample_correlators(loaded, mode="bootstrap", group="g", bin_size=1, n_boot=40, seed=987)
    np.testing.assert_array_equal(output["nonlocal"].values, numerator)
    np.testing.assert_array_equal(output["local"].values, local)
    assert output["nonlocal"].attrs["resample_id"] == "paired"
    assert output["nonlocal"].attrs["zdir"] == 0
    for settings in ({"bin_size": 2}, {"n_boot": 39}, {"mode": "jackknife"}):
        kwargs = dict(mode="bootstrap", group="g", bin_size=1, n_boot=40, seed=987)
        kwargs.update(settings)
        with pytest.raises(ValueError, match="precomputed bootstrap"):
            resample_correlators(loaded, **kwargs)
    output["local"].array.attrs["resample_id"] = "unpaired"
    with pytest.raises(ValueError, match="resample_id"):
        resample_correlators(loaded, mode="bootstrap", group="g", bin_size=1, n_boot=40, seed=987)


def test_raw_descriptors_still_resample(tmp_path):
    loaded, _, _ = descriptor(tmp_path, precomputed=False)
    output = resample_correlators(loaded, mode="bootstrap", group="g", bin_size=1, n_boot=23, seed=987)
    assert {d.n_sample for d in output.values()} == {23}
    assert len({d.attrs["resample_id"] for d in output.values()}) == 1


def test_native_ratio_uses_explicit_local_denominator(tmp_path):
    loaded, _, _ = descriptor(tmp_path)
    result, _ = fit_qda_samples(loaded["correlators"], fit_scope=["qda_ratio"], components="both",
                               tmin=2, tmax=8, n_states={"qda_ratio": 1}, prior_width=1.0, svdcut=1e-3,
                               posterior_prior_error_scale=10.0, sample_error_mode="variance", workers=1)
    # In particular z=0 at b>0 must remain 0.6+0.2j rather than being set to unity.
    np.testing.assert_allclose(result.values.mean(axis=0), [0.6+0.2j, 0.4+0.1j, 0.2-0.1j], atol=0.01)
    assert result.n_sample == 40 and result.attrs["resample_id"] == "paired"


def native_sources():
    z = np.arange(5)
    values = (np.arange(6)[:, None] / 20 + 1) * np.exp(-z[None, :] / 5) * (0.7 + 0.3j)
    attrs = {"resample_id": "paired", "bT": 1, "coord_unit": "lattice", "phase_convention": "midpoint",
             "momentum_gev": INFO.k_s * 4, "parton": "quark", "hadron": "pion", "gfix": "GI"}
    inputs = [EnsembleData(INFO, "bootstrap", list(values), ["z"], {"z": z.tolist()},
                          attrs={**attrs, "source_operator": op, "zdir": direction})
              for op in ("gtg5", "gzg5") for direction in (0, 1)]
    factor = ensemble(np.full((6, 5), 2.), {"z": (z * INFO.a_s).tolist()},
                      {"ensemble_id": INFO.id, "a_fm": INFO.a_s, "b_perp_lattice": 1,
                       "factor_uncertainty": "fixed_central_value_not_propagated", "ZO": 1., "L": 6}, "factor")
    return inputs, factor


def test_two_projection_transform_preserves_complex_midpoint_amplitude():
    inputs, factor = native_sources()
    parts = [prepare_projection(inputs, factor, part) for part in ("real", "imag")]
    momentum = parts[0].attrs["momentum_gev"]
    z = np.asarray(parts[0].coords["z"])
    lam = z * momentum / HBAR_C_GEV_FM
    np.testing.assert_allclose((parts[0].values + 1j * parts[1].values) * np.exp(0.5j * lam),
                               inputs[0].values / 2, atol=1e-15)
    x = np.linspace(-0.5, 1.5, 31).tolist()
    transforms = []
    for part in parts:
        conventions = derive_conventions(part.attrs, target_observable="tmdwf", sector="full")
        signed = complete_signed_z(part, conventions["symmetry"])
        transformed = fourier_transform(signed, x, momentum_gev=momentum, prefactor="pz_over_2pi")
        np.testing.assert_allclose(transformed.values.imag, 0, atol=1e-15)
        transforms.append(transformed.real)
    combined = combine_transforms(*transforms)
    signed_z = np.concatenate((-z[:0:-1], z))
    values = inputs[0].values / 2
    signed_values = np.concatenate((values[:, :0:-1], values), axis=1)
    midpoint = EnsembleData(INFO, "bootstrap", list(signed_values), ["z"], {"z": signed_z.tolist()})
    direct = fourier_transform(midpoint, x, momentum_gev=momentum, x_shift=0.5, prefactor="pz_over_2pi")
    np.testing.assert_allclose(combined.values, direct.values, atol=1e-15)
    transforms[1].array.attrs["resample_id"] = "mismatch"
    with pytest.raises(ValueError, match="provenance"):
        combine_transforms(*transforms)


def test_projection_rejects_missing_channel_or_sample_pairing():
    inputs, factor = native_sources()
    with pytest.raises(ValueError, match="exactly"):
        prepare_projection(inputs[:3], factor, "real")
    inputs[1].array.attrs["resample_id"] = "mismatch"
    with pytest.raises(ValueError, match="incompatible"):
        prepare_projection(inputs, factor, "real")
