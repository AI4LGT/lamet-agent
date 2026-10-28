"""Prepare one C32P23 momentum/b for the native qda_ratio and Fourier stages."""

import argparse
import copy
import json
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def prepare(root, pz, p1, p2, b):
    if (pz, p1, p2) not in {(4, 2, -2), (5, 2, -3), (5, 3, -2), (6, 3, -3)} or not 1 <= b <= 10:
        raise ValueError("unsupported momentum combination or b (1..10)")
    label = f"p{pz}_p1_{p1}_b{b}"
    folder = root / "data_C32P23" / "native" / label
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "correlators.h5"
    descriptor = folder / "correlators.json"
    manifest_path = root / "examples" / f"c32p23_native_{label}.json"
    if any(p.exists() for p in (path, descriptor, manifest_path)):
        raise FileExistsError("prepared files already exist; use the existing manifest")
    source = root / "data_C32P23/correlators/C32P23/2pt/C32P23_2pt_bootstrap.h5"
    ensemble = {"series": "C32P23", "id": "C32P23", "a_s": 0.1053, "a_t": 0.1053,
                "L_s": 32, "L_t": 64, "m_pi": 0.23}
    records, jobs = [], []
    with h5py.File(source, "r") as original, h5py.File(path, "x") as output:
        if original.attrs["resample_mode"] != "bootstrap":
            raise ValueError("requires original precomputed bootstrap input")
        count = int(original.attrs["bootstrap_samples"])
        source_hash = str(original.attrs["source_sha256"])
        output.attrs.update(source_sha256=source_hash, resample_mode="bootstrap", bootstrap_samples=count,
                            preparation="numerator phase rotation; real b0z0 denominator; no division or resampling")
        if "wilson_loops" not in original:
            raise ValueError("Add the Wilson-loop group to the source HDF5 before preparing native inputs")
        original.copy("wilson_loops", output)
        for operator in ("gtg5", "gzg5"):
            for direction in (0, 1):
                channel = f"{operator}_d{direction}"
                prefix = f"{operator}/gamma_sink0/PX0PY0PZ{pz}/p1_{p1}_p2_{p2}"
                values = np.stack([original[f"{prefix}/bT{b}/zdir{direction}/bz{z}"][()].T
                                   for z in range(13)], axis=-1)
                local = original[f"{prefix}/bT0/zdir{direction}/bz0"][()].real.T
                theta = 2 * np.pi * (p1 - pz / 2) * np.arange(13) / ensemble["L_s"] * (-1)**direction
                numerator = values * np.exp(1j * theta)[None, None, :]
                if not np.isfinite(numerator).all() or not np.isfinite(local).all() or np.any(local == 0):
                    raise ValueError("nonfinite data or zero local denominator")
                output.create_dataset(f"{channel}/nonlocal", data=numerator, compression="gzip")
                output.create_dataset(f"{channel}/local", data=local, compression="gzip")
                identifiers = []
                for kind in ("nonlocal", "local"):
                    identifier = f"{channel}_{kind}"
                    identifiers.append(identifier)
                    dims = ["bootstrap", "t", "z"] if kind == "nonlocal" else ["bootstrap", "t"]
                    coords = {"t": list(range(1, 16))}
                    if kind == "nonlocal":
                        coords["z"] = list(range(13))
                    records.append({
                        "id": identifier, "ensemble": ensemble, "count": count, "format": "hdf5",
                        "path": path.name, "dataset": f"{channel}/{kind}", "dataset_dims": dims, "dims": dims,
                        "coords": coords, "sampling": {"mode": "bootstrap", "resample_id": source_hash},
                        "selectors": {"gfix": "GI", "source_operator": operator, "sink_operator": "gamma_sink0",
                                      "zdir": direction, "bT": b if kind == "nonlocal" else 0,
                                      "p1": p1, "p2": p2, "phase_convention": "midpoint"},
                        "correlator_type": "qda" if kind == "nonlocal" else "two_point",
                        "hadron": {"name": "pion"}, "source_momentum": [0, 0, pz], "sink_momentum": [0, 0, pz],
                        "current": {"kernel_operator": "tmdwf", "parton": "quark", "renormalization_scheme": "bare"}
                        if kind == "nonlocal" else None, "source_sink_separation": None,
                    })
                jobs.append({"id": f"fit_{channel}", "inputs": {"correlators": [
                    {"json": descriptor.relative_to(root).as_posix(), "id": ident} for ident in identifiers]}})
    descriptor.write_text(json.dumps({"correlators": records}, indent=2), encoding="utf-8")
    baseline = json.loads((root / "examples/c32p23_tmdwf_manifest.json").read_text(encoding="utf-8"))
    wilson = copy.deepcopy(baseline["stages"]["tmdwf_analysis"]["jobs"][1])
    wilson["id"] = f"wilson_b{b}"
    wilson["settings"]["b"] = b
    wilson["inputs"]["raw"] = {"file": path.relative_to(root).as_posix()}
    bridges = [{"id": f"bridge_{part}", "operation": "native_bridge",
                "settings": {"projection": part, "negative_z": "even_both_notebook"},
                "inputs": {"targets": [job["id"] for job in jobs], "denominator": wilson["id"]}}
               for part in ("real", "imag")]
    metadata = dict(baseline["metadata"])
    metadata.update(run_id=f"c32p23_native_{label}", artifacts_directory=f"runs/c32p23_native_{label}/artifacts",
                    parameter_recommendation_retries=1, samples=count)
    manifest = {"metadata": metadata, "stages": {
        "correlator_analysis": {"defaults": {
            "analysis_method": "lsqfit", "component": "both", "fit_scope": ["qda_ratio"],
            "nstate": {"qda_ratio": [1]}, "fitting_form": "Breit", "model_average": False,
            "pt2_windows": [{"tmin": t, "tmax": 15} for t in (4, 6, 8)],
            "pt3_windows": [], "svdcut": 1e-3, "prior_width": [1.0],
            "posterior_prior_error_scale": 10.0, "q_min": 0.05}, "jobs": jobs},
        "tmdwf_analysis": {"jobs": [wilson, *bridges]},
        "fourier_transform": {"defaults": {
            "tmdwf_profile": "pion_da_components", "quasi_y_ls": {"start": -0.5, "stop": 1.5, "num": 1000},
            "zmin_fm": [round(t * ensemble["a_s"], 12) for t in (6, 8, 9)],
            "zmax_fm": [round(12 * ensemble["a_s"], 12)], "zmax_ext_fm": 48 * ensemble["a_s"],
            "smooth": "linear", "scheme_scan": {"order": ["LA", "NLA"], "sector": "full", "Lambda0_gev": 0.0,
                "posterior_prior_error_scale": [10.0], "model_average": False, "q_min": 0.05, "max_schemes": 20}},
            "jobs": [{"id": f"fourier_{part}", "inputs": {"input": f"bridge_{part}"}} for part in ("real", "imag")]}
    }}
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pz", type=int, default=4)
    parser.add_argument("--p1", type=int, default=2)
    parser.add_argument("--p2", type=int, default=-2)
    parser.add_argument("--b", type=int, default=1)
    args = parser.parse_args()
    print(prepare(ROOT, args.pz, args.p1, args.p2, args.b))
