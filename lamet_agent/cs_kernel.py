"""C32P23 CS_kernel workflow from bootstrap two-point functions through paired CSK."""

from __future__ import annotations

import copy
import json
from datetime import datetime
from itertools import combinations

import h5py
import numpy as np


def prepare_channels(source, folder, pz, p1, p2, b, *, descriptor_root=None):
    """Write the four channel descriptors and sample-preserving inputs for native ratio fits."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "correlators.h5"
    descriptor = folder / "correlators.json"
    descriptor_ref = descriptor.relative_to(descriptor_root) if descriptor_root is not None else descriptor.resolve()
    ensemble = {"series": "C32P23", "id": "C32P23", "a_s": 0.1053, "a_t": 0.1053, "L_s": 32, "L_t": 64, "m_pi": 0.23}
    records, jobs = [], []
    with h5py.File(source, "r") as original, h5py.File(path, "x") as output:
        if original.attrs["resample_mode"] != "bootstrap":
            raise ValueError("requires original precomputed bootstrap input")
        count = int(original.attrs["bootstrap_samples"])
        source_hash = str(original.attrs["source_sha256"])
        output.attrs.update(
            source_sha256=source_hash,
            resample_mode="bootstrap",
            bootstrap_samples=count,
            preparation="numerator phase rotation; real b0z0 denominator; no division or resampling",
        )
        if "wilson_loops" not in original:
            raise ValueError("Add the Wilson-loop group to the source HDF5 before preparing native inputs")
        original.copy("wilson_loops", output)
        for operator in ("gtg5", "gzg5"):
            for direction in (0, 1):
                channel = f"{operator}_d{direction}"
                prefix = f"{operator}/gamma_sink0/PX0PY0PZ{pz}/p1_{p1}_p2_{p2}"
                values = np.stack([original[f"{prefix}/bT{b}/zdir{direction}/bz{z}"][()].T for z in range(13)], axis=-1)
                local = original[f"{prefix}/bT0/zdir{direction}/bz0"][()].real.T
                theta = 2 * np.pi * (p1 - pz / 2) * np.arange(13) / ensemble["L_s"] * (-1) ** direction
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
                    records.append(
                        {
                            "id": identifier,
                            "ensemble": ensemble,
                            "count": count,
                            "format": "hdf5",
                            "path": path.name,
                            "dataset": f"{channel}/{kind}",
                            "dataset_dims": dims,
                            "dims": dims,
                            "coords": coords,
                            "sampling": {"mode": "bootstrap", "resample_id": source_hash},
                            "selectors": {
                                "gfix": "GI",
                                "source_operator": operator,
                                "sink_operator": "gamma_sink0",
                                "zdir": direction,
                                "bT": b if kind == "nonlocal" else 0,
                                "p1": p1,
                                "p2": p2,
                                "phase_convention": "midpoint",
                            },
                            "correlator_type": "qda" if kind == "nonlocal" else "two_point",
                            "hadron": {"name": "pion"},
                            "source_momentum": [0, 0, pz],
                            "sink_momentum": [0, 0, pz],
                            "current": {"kernel_operator": "tmdwf", "parton": "quark", "renormalization_scheme": "bare"}
                            if kind == "nonlocal"
                            else None,
                            "source_sink_separation": None,
                        }
                    )
                jobs.append(
                    {
                        "id": f"fit_{channel}",
                        "inputs": {
                            "correlators": [{"json": descriptor_ref.as_posix(), "id": ident} for ident in identifiers]
                        },
                    }
                )
    descriptor.write_text(json.dumps({"correlators": records}, indent=2), encoding="utf-8")
    return jobs, count


def read_settings(recipe, path):
    """Resolve one CS_kernel recipe and check the requested channels in its raw HDF5."""
    unknown = set(recipe) - {
        "analysis",
        "two_point_h5",
        "native_template",
        "output_directory",
        "b_lattice",
        "pairs",
        "matching",
    }
    if unknown:
        raise ValueError(f"Unknown CS_kernel settings: {sorted(unknown)}")
    if not isinstance(recipe.get("matching"), dict):
        raise ValueError("CS_kernel requires matching parameters")
    source = (path.parent / recipe["two_point_h5"]).resolve(strict=True)
    template = (path.parent / recipe["native_template"]).resolve(strict=True)
    output = (path.parent / recipe["output_directory"]).resolve()
    distances = recipe["b_lattice"]
    if (
        not isinstance(distances, list)
        or not distances
        or any(type(b) is not int or not 1 <= b <= 10 for b in distances)
    ):
        raise ValueError("b_lattice must be a nonempty list of integer lattice separations within 1..10")
    if len(set(distances)) != len(distances):
        raise ValueError("b_lattice contains duplicate separations")
    momenta = {4: (2, -2), 5: (3, -2), 6: (3, -3)}
    requested = recipe.get("pairs", "all")
    pairs = list(combinations(momenta, 2)) if requested == "all" else requested
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("pairs must be 'all' or a nonempty list such as [[4,6]]")
    for pair in pairs:
        if (
            not isinstance(pair, (list, tuple))
            or len(pair) != 2
            or any(type(p) is not int or p not in momenta for p in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError("Each pair needs two distinct momenta from 4, 5, 6")
    if len({tuple(sorted(pair)) for pair in pairs}) != len(pairs):
        raise ValueError("Repeated momentum pair")
    needed = sorted({p for pair in pairs for p in pair})
    with h5py.File(source, "r") as raw:
        if raw.attrs.get("resample_mode") != "bootstrap" or "wilson_loops/C32P23/raw" not in raw:
            raise ValueError("Expected C32P23 bootstrap 2pt HDF5 with the Wilson-loop group")
        count = int(raw.attrs["bootstrap_samples"])
        for b in distances:
            for p in needed:
                p1, p2 = momenta[p]
                for op in ("gtg5", "gzg5"):
                    for direction in (0, 1):
                        parent = f"{op}/gamma_sink0/PX0PY0PZ{p}/p1_{p1}_p2_{p2}"
                        for bt, z in [(0, 0), *((b, z) for z in range(13))]:
                            leaf = f"{parent}/bT{bt}/zdir{direction}/bz{z}"
                            if leaf not in raw or raw[leaf].shape != (15, count):
                                raise ValueError(f"Missing or incompatible C32P23 dataset: {leaf}")
    return source, template, output, distances, pairs, {p: momenta[p] for p in needed}, count


def build_manifest(recipe, path, *, work_directory=None):
    """Compile an ordered native four-stage graph; each (Pz,b) is fitted only once."""
    from lamet_agent.manifest import load_manifest

    source, template, output, distances, pairs, momenta, count = read_settings(recipe, path)
    native = json.loads(template.read_text(encoding="utf-8"))
    root = (work_directory or output) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True, exist_ok=False)
    doc = copy.deepcopy(native)
    doc["metadata"].update(root_directory=".", artifacts_directory="artifacts", run_id="CS_kernel", samples=count)
    stages = doc["stages"]
    for block in stages.values():
        block["jobs"] = []
    stages["soft_analysis"] = {"defaults": recipe["matching"], "jobs": []}
    for b in distances:
        wilson, apply_template = native["stages"]["renormalization"]["jobs"]
        wilson = copy.deepcopy(wilson)
        wilson.update(id=f"wilson_b{b}", inputs={"raw": {"file": source.as_posix()}})
        wilson["settings"]["b"] = b
        stages["renormalization"]["jobs"].append(wilson)
        for p, (p1, p2) in momenta.items():
            tag = f"p{p}_b{b}"
            fits, _ = prepare_channels(source, root / "inputs" / tag, p, p1, p2, b, descriptor_root=root)
            for fit in fits:
                fit["id"] += "_" + tag
            stages["correlator_analysis"]["jobs"].extend(fits)
            ren = copy.deepcopy(apply_template)
            ren.update(id=f"ren_{tag}", inputs={"targets": [j["id"] for j in fits], "denominator": wilson["id"]})
            stages["renormalization"]["jobs"].append(ren)
            stages["fourier_transform"]["jobs"].append(
                {
                    "id": f"real_{tag}",
                    "tmdwf_projection": "real",
                    "inputs": {"input": ren["id"]},
                }
            )
            quasi = {
                "id": f"quasi_{tag}",
                "tmdwf_projection": "imag",
                "inputs": {"input": ren["id"], "real_projection": f"real_{tag}"},
            }
            if (p, b) == (6, 3):
                quasi.update(zmin_fm=[0.5265], zmax_fm=[1.053])
            stages["fourier_transform"]["jobs"].append(quasi)
        for p, q in pairs:
            stages["soft_analysis"]["jobs"].append(
                {"id": f"csk_p{p}_p{q}_b{b}", "inputs": {"quasi1": f"quasi_p{p}_b{b}", "quasi2": f"quasi_p{q}_b{b}"}}
            )
    full = root / "pipeline.json"
    full.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return load_manifest(full)
