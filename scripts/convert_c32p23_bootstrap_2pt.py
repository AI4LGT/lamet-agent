"""Losslessly pack C32P23 bootstrap correlators into example-style HDF5 leaves.

Source operators are user-confirmed: index 0 is gtg5 and index 1 is gzg5.
The sink retains its original index. This does not modify the agent input loader.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np


SHAPE = (1000, 4, 11, 2, 13, 2, 15, 2)
AXES = ["bootstrap", "momentum_combination", "b_perp", "zdir", "z",
        "gamma_source", "t", "component"]
MOMENTA = [(4, 2, -2), (5, 2, -3), (5, 3, -2), (6, 3, -3)]
SOURCE_OPERATORS = ("gtg5", "gzg5")


def entries():
    for ip, (pz, p1, p2) in enumerate(MOMENTA):
        for b in range(SHAPE[2]):
            for direction in range(SHAPE[3]):
                for z in range(SHAPE[4]):
                    for gamma in range(SHAPE[5]):
                        name = (f"{SOURCE_OPERATORS[gamma]}/gamma_sink0/PX0PY0PZ{pz}/"
                                f"p1_{p1}_p2_{p2}/bT{b}/zdir{direction}/bz{z}")
                        yield name, (ip, b, direction, z, gamma)


def source_leaf(array, index):
    ip, b, direction, z, gamma = index
    return array[:, ip, b, direction, z, gamma, :, :]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, destination: Path):
    source = source.resolve(strict=True)
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    print(f"Loading {source}", flush=True)
    data = np.load(source, allow_pickle=False)
    if data.shape != SHAPE or data.dtype != np.dtype("float64"):
        raise ValueError(f"Expected float64 {SHAPE}; found {data.dtype} {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("Source contains nonfinite values; inspect before conversion")
    source_hash = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with h5py.File(destination, "x") as handle:
        handle.attrs.update({
            "layout": "agent-style complex (t, bootstrap) leaves; extended channel hierarchy",
            "ensemble_id": "C32P23",
            "resample_mode": "bootstrap",
            "sample_kind": "bootstrap",
            "bootstrap_samples": SHAPE[0],
            "already_resampled": True,
            "source_file": str(source),
            "source_sha256": source_hash,
            "source_shape": json.dumps(list(data.shape)),
            "source_dtype": data.dtype.str,
            "source_axes": json.dumps(AXES),
            "momentum_combinations_pz_p1_p2": json.dumps(MOMENTA),
            "component_order": json.dumps(["real", "imag"]),
            "conversion": "Pack components as complex128 and transpose time/sample only",
            "operator_labels": "Source mapping confirmed by user; sink retains original index",
            "source_operator_mapping": json.dumps(dict(enumerate(SOURCE_OPERATORS))),
            "axis_evidence": "fit_bare_tmdwf_P23_npy.ipynb: save_gvar_data and cal_Psi_matrix",
            "loader_note": "Current agent raw-configuration loader must be adapted to skip resampling",
            "verification_status": "pending",
        })
        for name, index in entries():
            leaf = source_leaf(data, index)
            values = np.empty((SHAPE[6], SHAPE[0]), dtype=np.complex128)
            values.real = leaf[:, :, 0].T
            values.imag = leaf[:, :, 1].T
            dataset = handle.create_dataset(name, data=values, compression="gzip",
                                            compression_opts=1, shuffle=True)
            ip, b, direction, z, gamma = index
            pz, p1, p2 = MOMENTA[ip]
            dataset.dims[0].label = "t"
            dataset.dims[1].label = "bootstrap"
            dataset.attrs.update({
                "dataset_dims": json.dumps(["t", "bootstrap"]),
                "resample_mode": "bootstrap",
                "source_indices_ip_b_zdir_z_gamma": json.dumps(list(index)),
                "momentum_index": ip, "source_momentum": f"PX0PY0PZ{pz}",
                "p1": p1, "p2": p2, "b_perp": b, "zdir": direction,
                "z": z, "gamma_source_index": gamma, "gamma_sink_index": 0,
                "source_operator": SOURCE_OPERATORS[gamma],
                "t_coordinates": np.arange(1, SHAPE[6] + 1),
                "t_coordinate_evidence": "Notebook folds slices 1:16,17:32,33:48,49:64",
            })
            count += 1
            if count % 400 == 0:
                print(f"Written {count} datasets", flush=True)
    print("Verifying every real and imaginary value bit for bit...", flush=True)
    with h5py.File(destination, "r+") as handle:
        for name, index in entries():
            expected = source_leaf(data, index)
            stored = handle[name][()]
            for component, actual in enumerate((stored.real.T, stored.imag.T)):
                actual_bytes = np.ascontiguousarray(actual).tobytes()
                expected_bytes = np.ascontiguousarray(expected[:, :, component]).tobytes()
                if actual_bytes != expected_bytes:
                    raise AssertionError(f"Bitwise mismatch: {name}, component {component}")
        handle.attrs["verification_status"] = "passed: all source float64 values bitwise identical"
        handle.attrs["verified_real_scalar_count"] = data.size
        handle.attrs["dataset_count"] = count
    result = {"output": str(destination), "datasets": count,
              "source_shape": list(data.shape), "bootstrap_samples": SHAPE[0],
              "leaf_shape": [SHAPE[6], SHAPE[0]], "leaf_dtype": "complex128",
              "source_sha256": source_hash, "verified_real_scalar_count": data.size,
              "bitwise_equal": True, "bytes": destination.stat().st_size}
    print(json.dumps(result, indent=2), flush=True)


def relabel_existing(path: Path):
    """Rename source groups in place and verify every dataset's byte content."""
    with h5py.File(path, "r+") as handle:
        hashes = {}
        for name, index in entries():
            gamma = index[-1]
            old_name = f"gamma_source{gamma}/" + name.split("/", 1)[1]
            present = old_name if old_name in handle else name
            if present not in handle:
                raise ValueError(f"Missing expected dataset: {name}")
            hashes[name] = hashlib.sha256(handle[present][()].tobytes()).hexdigest()
        for gamma, operator in enumerate(SOURCE_OPERATORS):
            old_group = f"gamma_source{gamma}"
            if old_group in handle:
                if operator in handle:
                    raise ValueError(f"Both {old_group} and {operator} exist")
                handle.move(old_group, operator)
        for name, index in entries():
            dataset = handle[name]
            dataset.attrs["source_operator"] = SOURCE_OPERATORS[index[-1]]
            if hashlib.sha256(dataset[()].tobytes()).hexdigest() != hashes[name]:
                raise AssertionError(f"Dataset changed while relabeling: {name}")
        handle.attrs["source_operator_mapping"] = json.dumps(dict(enumerate(SOURCE_OPERATORS)))
        handle.attrs["operator_labels"] = "Source mapping confirmed by user; sink retains original index"
        handle.attrs["source_relabel_verification"] = "All 2288 dataset byte hashes unchanged"
    print(f"Updated source labels; verified {len(hashes)} unchanged datasets: {path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    convert(arguments.source, arguments.output)
