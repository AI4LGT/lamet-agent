"""Explicit Wilson-loop coordinates shared by text and HDF5 inputs."""

import hashlib
import io
from pathlib import Path

import h5py
import numpy as np


GROUP = "wilson_loops/C32P23"
AXES = ("configuration", "direction", "length_lattice", "b_perp_lattice")


def parse_text(payload):
    """Validate configuration/direction/length/b ordering before reshaping."""
    table = np.loadtxt(io.BytesIO(payload))
    if table.ndim != 2 or table.shape[1] != 5 or not np.isfinite(table).all():
        raise ValueError("Wilson text requires five finite columns: configuration, direction, length, b, value")
    if len(table) % (3 * 26 * 11):
        raise ValueError("Wilson table must contain configuration x 3 directions x 26 lengths x 11 b values")
    blocks = table.reshape(-1, 3, 26, 11, 5)
    configurations = blocks[:, 0, 0, 0, 0]
    if len(configurations) < 2 or len(set(configurations)) != len(configurations):
        raise ValueError("Wilson configuration IDs must be unique and contain at least two configurations")
    expected = (configurations[:, None, None, None], np.arange(3)[None, :, None, None],
                np.arange(26)[None, None, :, None], np.arange(11)[None, None, None, :])
    if np.any(configurations != configurations.astype(np.int64)) or any(
        not np.array_equal(blocks[..., index], np.broadcast_to(coords, blocks.shape[:-1]))
        for index, coords in enumerate(expected)
    ):
        raise ValueError("Wilson table coordinates do not match the declared axis order")
    return blocks[..., 4].copy(), configurations.astype(np.int64)


def load_wilson(path):
    """Load raw Wilson measurements; the caller fits only their central factor."""
    path = Path(path)
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        payload = path.read_bytes()
        values, configurations = parse_text(payload)
        return values, {"source_sha256": hashlib.sha256(payload).hexdigest(), "format": "text"}
    with h5py.File(path, "r") as handle:
        group = handle[GROUP]
        if group.attrs.get("sample_kind") != "raw_configuration":
            raise ValueError("Wilson HDF5 input must contain raw configurations, separate from 2pt bootstrap")
        values = group["raw"][()]
        configurations = group["configuration"][()]
        if values.shape != (len(configurations), 3, 26, 11) or len(configurations) < 2:
            raise ValueError("Wilson HDF5 data must have axes configuration,direction,length_lattice,b_perp_lattice")
        if len(np.unique(configurations)) != len(configurations) or not np.isfinite(values).all():
            raise ValueError("Wilson HDF5 input contains duplicate configurations or nonfinite values")
        for axis, size in zip(AXES[1:], values.shape[1:]):
            if not np.array_equal(group[axis][()], np.arange(size)):
                raise ValueError(f"Wilson HDF5 coordinate mismatch: {axis}")
        return values, {"source_sha256": str(group.attrs["source_sha256"]), "format": "hdf5", "dataset": GROUP + "/raw"}
