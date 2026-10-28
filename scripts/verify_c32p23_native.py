"""Verify native C32P23 input preservation and the two-component Fourier identity."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM
from lamet_agent.stages.correlator_analysis._input import load_descriptor, resample_correlators


def verify(root, artifacts, descriptor):
    loaded = load_descriptor(descriptor)
    first = next(iter(loaded["correlators"].values()))
    samples = first.n_sample
    preserved = resample_correlators(loaded, mode="bootstrap", group="verify", bin_size=1,
                                    n_boot=samples, seed=9981)
    checked = 0
    with h5py.File(root / "data_C32P23/correlators/C32P23/2pt/C32P23_2pt_bootstrap.h5", "r") as original:
        for name, data in preserved.items():
            np.testing.assert_array_equal(data.values, loaded["correlators"][name].values)
            attrs = data.attrs
            pz = json.loads(attrs["sink_momentum"])[2]
            prefix = (f"{attrs['source_operator']}/gamma_sink0/PX0PY0PZ{pz}/"
                      f"p1_{attrs['p1']}_p2_{attrs['p2']}")
            if attrs["correlator_type"] == "qda":
                expected = np.stack([original[f"{prefix}/bT{attrs['bT']}/zdir{attrs['zdir']}/bz{z}"][()].T
                                     for z in range(13)], axis=-1)
                theta = (2 * np.pi * (attrs["p1"] - pz / 2) * np.arange(13)
                         / data.ensemble.L_s * (-1)**attrs["zdir"])
                expected = expected * np.exp(1j * theta)[None, None, :]
            else:
                expected = original[f"{prefix}/bT0/zdir{attrs['zdir']}/bz0"][()].real.T
            np.testing.assert_array_equal(data.values, expected)
            checked += data.values.size
    final = EnsembleData.from_netcdf(artifacts / "quasi_tmdwf.nc")
    assert final.n_sample == samples and np.isfinite(final.values).all()
    assert final.attrs["resample_id"] == first.attrs["resample_id"]
    factors = list((artifacts / "02_renormalization").glob("wilson_*/output.nc"))
    assert len(factors) == 1
    factor = EnsembleData.from_netcdf(factors[0])
    assert factor.n_sample == samples and factor.attrs["factor_uncertainty"] == "fixed_central_value_not_propagated"
    np.testing.assert_array_equal(factor.values, np.broadcast_to(factor.values[0], factor.values.shape))
    factor_diag = json.loads((factors[0].parent / "diagnostics.json").read_text(encoding="utf-8"))
    ft = artifacts / "03_fourier_transform"
    real = EnsembleData.from_netcdf(ft / "fourier_real/extended.nc")
    imag = EnsembleData.from_netcdf(ft / "fourier_imag/extended.nc")
    assert real.coords == imag.coords and real.n_sample == imag.n_sample == samples
    z = np.asarray(real.coords["z"])
    momentum = real.attrs["momentum_gev"]
    lam = z * momentum / HBAR_C_GEV_FM
    # Undo the endpoint phase and integrate directly in midpoint convention.
    # Use all samples, but chunks of x to bound memory.
    midpoint = (real.values + 1j * imag.values) * np.exp(0.5j * lam)[None, :]
    x = np.asarray(final.coords["x"])
    weights = np.full(len(z), np.diff(z)[0])
    np.testing.assert_allclose(np.diff(z), weights[0], atol=1e-12)
    maximum_error = 0.0
    for start in range(0, len(x), 100):
        stop = min(start + 100, len(x))
        kernel = np.exp(1j * (x[start:stop, None] - 0.5) * lam[None, :])
        expected = midpoint @ (kernel * weights).T * momentum / (2 * np.pi * HBAR_C_GEV_FM)
        observed = final.values[:, start:stop]
        np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-12)
        maximum_error = max(maximum_error, float(np.max(np.abs(expected - observed))))
    quality = {}
    for path in artifacts.glob("*/*/summary.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        diag = summary["diagnostics"]
        quality[path.parent.name] = {key: diag[key] for key in (
            "Q", "selected_Q", "tail_family", "fallback_no_q_passing") if key in diag}
    result = {"native_data_exactly_checked_values": checked, "bootstrap_samples": samples,
              "wilson_policy": "one fixed central factor shared by every bootstrap sample",
              "wilson_input": factor_diag.get("input_provenance", {"format": "legacy_text"}),
              "bootstrap_resample_id_preserved": True, "finite_complex_output_shape": list(final.values.shape),
              "direct_fourier_checked_values": final.values.size, "direct_fourier_max_abs_error": maximum_error,
              "quality": quality, "scope": "numerical integration; not physical model certification"}
    (artifacts / "native_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--descriptor", type=Path, default=Path("data_C32P23/native/p4_p1_2_b1/correlators.json"))
    args = parser.parse_args()
    print(json.dumps(verify(Path(__file__).resolve().parents[1], args.artifacts, args.descriptor), indent=2))
