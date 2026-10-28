"""Compare the full Pz=4,b=1 run against existing notebook-produced arrays."""

import argparse
import json
from pathlib import Path

import numpy as np

from lamet_agent.data import EnsembleData
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--legacy-directory", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifacts / "01_tmdwf_analysis"
    data = {job: EnsembleData.from_netcdf(root / job / "output.nc") for job in
            ("bare_p4_b1", "wilson_b1", "renormalized_p4_b1", "fourier_p4_b1")}
    metrics = {}

    def compare(label, actual, reference):
        np.testing.assert_allclose(actual, reference, rtol=1e-7, atol=1e-9, equal_nan=False)
        metrics[label] = {"max_absolute_difference": float(np.max(np.abs(actual - reference))),
                          "compared_scalars": int(actual.size), "passed": True}

    for name, value in data.items():
        if value.n_sample != 1000 or not np.isfinite(value.values).all():
            raise AssertionError(f"Invalid sample output: {name}")
    reference = np.load(args.legacy_directory / "fit_npy/bare_wf_b1.npy", allow_pickle=False)[0]
    compare("bare_vs_notebook", data["bare_p4_b1"].values, reference[0] + 1j * reference[1])
    compare("renormalization_samplewise", data["renormalized_p4_b1"].values,
            data["bare_p4_b1"].values / data["wilson_b1"].values)
    output = data["fourier_p4_b1"]
    for label, actual in (("Re", output.values.real), ("Im", output.values.imag)):
        reference = np.load(args.legacy_directory / f"wf_npy/{label}_wf_p4_b1.npy", allow_pickle=False)
        compare(f"fourier_{label}_vs_notebook", actual, reference)
    extended = EnsembleData.from_netcdf(root / "fourier_p4_b1/extended.nc")
    lam = np.asarray(extended.coords["z"]) * extended.attrs["momentum_gev"] / HBAR_C_GEV_FM
    x = np.asarray(output.coords["x"])
    angle = (x[:, None] - 0.5) * lam[None, :]
    for sample in (0, 123, 999):
        value = extended.values[sample]
        real = np.sum(np.cos(angle) * value.real - np.sin(angle) * value.imag, axis=1)
        imag = np.sum(np.sin(angle) * value.real + np.cos(angle) * value.imag, axis=1)
        direct = (real + 1j * imag) * (lam[1] - lam[0]) / (2 * np.pi)
        compare(f"direct_fourier_sample_{sample}", output.values[sample], direct)
    metrics["fit_quality"] = {job: json.loads((root / job / "diagnostics.json").read_text())
                              for job in ("bare_p4_b1", "fourier_p4_b1")}
    metrics["scope"] = {"samples": 1000, "pz": 4, "b": 1, "hard_matching": False,
                        "wilson_uncertainty_propagated": False, "legacy_directory": str(args.legacy_directory)}
    (args.artifacts / "verification.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
