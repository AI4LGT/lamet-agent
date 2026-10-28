"""Append validated Wilson measurements to a C32P23 HDF5 without changing its 2pt data."""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from lamet_agent.stages.tmdwf_analysis.wilson_io import AXES, GROUP, load_wilson, parse_text


def add_wilson(source, destination):
    payload = source.read_bytes()
    values, configurations = parse_text(payload)
    digest = hashlib.sha256(payload).hexdigest()
    # r+ never creates an accidentally misspelled destination or replaces existing datasets.
    with h5py.File(destination, "r+") as handle:
        if GROUP in handle:
            group = handle[GROUP]
            if (group.attrs.get("source_sha256") != digest
                    or not np.array_equal(group["raw"][()], values)
                    or group["source_text"][()].tobytes() != payload):
                raise ValueError("Existing Wilson group differs; refusing to overwrite")
        else:
            parent = handle.require_group("wilson_loops")
            pending = parent.create_group("_pending_C32P23")
            pending.attrs.update(source_file=str(source.resolve()), source_sha256=digest,
                                 sample_kind="raw_configuration", use_in_analysis="fixed_central_value",
                                 axes=json.dumps(AXES), configuration_count=len(configurations),
                                 source_columns="configuration,direction,length_lattice,b_perp_lattice,value")
            dataset = pending.create_dataset("raw", data=values, compression="gzip", shuffle=True)
            for index, axis in enumerate(AXES):
                dataset.dims[index].label = axis
                pending.create_dataset(axis, data=configurations if index == 0 else np.arange(values.shape[index]))
            central = pending.create_dataset("mean", data=values.mean(axis=1).mean(axis=0))
            central.attrs["axes"] = json.dumps(AXES[2:])
            central.attrs["definition"] = "mean over directions and configurations; unfitted Wilson loop"
            pending.create_dataset("source_text", data=np.frombuffer(payload, dtype=np.uint8), compression="gzip")
            parent.move("_pending_C32P23", "C32P23")
    restored, provenance = load_wilson(destination)
    np.testing.assert_array_equal(restored, values)
    return {"hdf5": str(destination), "dataset": GROUP + "/raw", "shape": list(values.shape),
            "source_sha256": provenance["source_sha256"], "exact_numeric_match": True,
            "source_text_bytes": len(payload), "use_in_analysis": "fixed_central_value"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(json.dumps(add_wilson(args.source, args.destination), indent=2))
