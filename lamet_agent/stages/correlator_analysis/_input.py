"""Strict correlator descriptor validation and HDF5 loading.

A dataset path may contain coordinate placeholders such as ``{tsep}`` and
``{z}``.  ``dataset_dims`` states the leaf-array axis order explicitly; the
loader never guesses axes from shapes.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from lamet_agent.data import EnsembleData, EnsembleInfo


def load_descriptor(path: Path, *, selected_ids: set[str] | None = None) -> dict[str, Any]:
    """Load the selected ensemble from one project correlator descriptor."""
    if path.suffix.lower() != ".json" or not path.is_file():
        raise ValueError(f"correlator descriptor must be an existing .json file: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or set(document) != {"correlators"}:
        raise ValueError("project correlator descriptor requires exactly one correlators list")
    records = document["correlators"]
    if not isinstance(records, list) or not records or any(not isinstance(record, dict) for record in records):
        raise ValueError("project correlator descriptor requires a nonempty correlators list")
    record_ids = [record.get("id") for record in records]
    if any(not isinstance(value, str) or not value for value in record_ids) or len(set(record_ids)) != len(record_ids):
        raise ValueError("project correlator ids must be nonempty and unique")
    requested = set(record_ids) if selected_ids is None else selected_ids
    selected = [record for record in records if record["id"] in requested]
    if {record["id"] for record in selected} != requested:
        raise ValueError("selected record ids are not present in the project descriptor")
    ensembles = [record.get("ensemble") for record in selected]
    counts = [record.get("count") for record in selected]
    if any(not isinstance(value, dict) for value in ensembles) or any(value != ensembles[0] for value in ensembles[1:]):
        raise ValueError("selected correlators must share exactly one ensemble")
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 2 for value in counts):
        raise ValueError("correlator count must be an integer of at least two")
    if any(value != counts[0] for value in counts[1:]):
        raise ValueError("selected correlators must share one configuration count")
    descriptor = {
        "ensemble": ensembles[0],
        "configuration_count": counts[0],
        "correlators": [
            {key: value for key, value in record.items() if key not in {"ensemble", "count"}} for record in selected
        ],
    }
    try:
        ensemble = EnsembleInfo(**descriptor["ensemble"])
    except (TypeError, ValueError) as exc:
        raise ValueError("ensemble must contain exactly the EnsembleInfo fields") from exc
    configuration_count = descriptor["configuration_count"]
    if not isinstance(configuration_count, int) or isinstance(configuration_count, bool) or configuration_count < 2:
        raise ValueError("configuration_count must be an integer of at least two")
    configuration_ids = [str(index) for index in range(configuration_count)]
    output: dict[str, Any] = {"descriptor": descriptor, "configuration_ids": list(configuration_ids), "correlators": {}}
    for record in descriptor["correlators"]:
        if not isinstance(record, dict):
            raise ValueError("each correlator record must be an object")
        required = {
            "id",
            "format",
            "path",
            "dataset",
            "dataset_dims",
            "dims",
            "coords",
            "selectors",
            "correlator_type",
            "hadron",
            "source_momentum",
            "sink_momentum",
            "current",
            "source_sink_separation",
        }
        if set(record) != required:
            raise ValueError(f"correlator record must contain exactly: {sorted(required)}")
        if (
            not isinstance(record["id"], str)
            or not record["id"]
            or not isinstance(record["path"], str)
            or not isinstance(record["dataset"], str)
        ):
            raise ValueError("correlator id, path, and dataset must be strings")
        if record["id"] in output["correlators"]:
            raise ValueError(f"correlator id is repeated: {record['id']}")
        if (
            record["format"] != "hdf5"
            or not isinstance(record["dims"], list)
            or not record["dims"]
            or record["dims"][0] != "configuration"
        ):
            raise ValueError("only HDF5 descriptors with configuration as the first dimension are supported")
        if any(not isinstance(dim, str) or not dim or dim == "resample" for dim in record["dims"]) or len(
            set(record["dims"])
        ) != len(record["dims"]):
            raise ValueError("descriptor dimensions must be unique nonempty names")
        if (
            not isinstance(record["coords"], dict)
            or not isinstance(record["dataset_dims"], list)
            or not isinstance(record["selectors"], dict)
        ):
            raise ValueError("coords, dataset_dims, and selectors must be explicit objects/lists")
        dataset_dims = record["dataset_dims"]
        if any(not isinstance(dim, str) or dim not in record["dims"] for dim in dataset_dims) or len(
            set(dataset_dims)
        ) != len(dataset_dims):
            raise ValueError("dataset_dims must be unique names from dims")
        if record["correlator_type"] not in {"two_point", "three_point", "qda"}:
            raise ValueError("correlator_type must be two_point, three_point, or qda")
        momenta = [record["source_momentum"], record["sink_momentum"]]
        if any(not isinstance(momentum, list) or len(momentum) != 3 for momentum in momenta) or any(
            not isinstance(value, int) or isinstance(value, bool) for momentum in momenta for value in momentum
        ):
            raise ValueError("source and sink momenta must be integer triples")
        if record["correlator_type"] == "two_point" and record["current"] is not None:
            raise ValueError("two-point correlators must have current=null")
        if record["correlator_type"] != "two_point" and not isinstance(record["current"], dict):
            raise ValueError("three-point and qDA correlators require a current object")
        if isinstance(record["current"], dict) and set(record["current"]) != {
            "kernel_operator",
            "parton",
            "renormalization_scheme",
        }:
            raise ValueError("current must contain exactly kernel_operator, parton, and renormalization_scheme")
        if record["source_sink_separation"] is not None and (
            not isinstance(record["source_sink_separation"], int)
            or isinstance(record["source_sink_separation"], bool)
            or record["source_sink_separation"] < 0
        ):
            raise ValueError("source_sink_separation must be a nonnegative integer or null")
        if (
            record["source_sink_separation"] is None
            and "tsep" not in record["dims"]
            and record["correlator_type"] == "three_point"
        ):
            raise ValueError("a null source_sink_separation requires an explicit tsep dimension")
        if not isinstance(record["hadron"], dict):
            raise ValueError("hadron must be an object")
        hdf5_path = Path(record["path"])
        if not hdf5_path.is_absolute():
            hdf5_path = path.parent / hdf5_path
        if not hdf5_path.is_file():
            raise ValueError(f"correlator HDF5 input does not exist: {hdf5_path.resolve()}")
        import h5py

        placeholders = re.findall(r"{([A-Za-z][A-Za-z0-9_]*)}", record["dataset"])
        if len(placeholders) != len(set(placeholders)) or any(
            dim not in record["dims"] or dim not in record["coords"] or dim in dataset_dims for dim in placeholders
        ):
            raise ValueError("dataset placeholders must be unique coordinate dimensions outside dataset_dims")
        remaining_dims = [dim for dim in record["dims"] if dim not in placeholders]
        if set(remaining_dims) != set(dataset_dims):
            raise ValueError("dataset_dims plus dataset placeholders must cover dims exactly")
        shape = [
            configuration_count if dim == "configuration" else len(record["coords"].get(dim, []))
            for dim in record["dims"]
        ]
        if any(size == 0 for size in shape):
            raise ValueError("every non-configuration dimension needs nonempty coordinates")
        values = None
        coordinate_products = (
            itertools.product(*(record["coords"][dim] for dim in placeholders)) if placeholders else [()]
        )
        with h5py.File(hdf5_path, "r") as handle:
            for coordinate_values in coordinate_products:
                selected = dict(zip(placeholders, coordinate_values))
                dataset_name = record["dataset"].format(**selected)
                if dataset_name not in handle:
                    raise ValueError(f"HDF5 dataset does not exist: {dataset_name}")
                leaf = np.asarray(handle[dataset_name])
                if leaf.ndim != len(dataset_dims):
                    raise ValueError(f"dataset rank does not match dataset_dims: {dataset_name}")
                sizes = dict(zip(dataset_dims, leaf.shape))
                if sizes.get("configuration") != configuration_count:
                    raise ValueError(f"configuration count does not match descriptor: {dataset_name}")
                for dim, size in sizes.items():
                    if dim != "configuration" and size > len(record["coords"][dim]):
                        raise ValueError(f"dataset is longer than declared coordinate {dim}: {dataset_name}")
                if values is None:
                    values = np.zeros(shape, dtype=leaf.dtype)
                elif values.dtype != leaf.dtype:
                    raise ValueError("templated HDF5 datasets must share one dtype")
                target_dims = [dim for dim in record["dims"] if dim not in placeholders]
                leaf = np.transpose(leaf, [dataset_dims.index(dim) for dim in target_dims])
                selection = []
                for dim in record["dims"]:
                    if dim in selected:
                        selection.append(record["coords"][dim].index(selected[dim]))
                    elif dim == "configuration":
                        selection.append(slice(None))
                    else:
                        selection.append(slice(0, sizes[dim]))
                values[tuple(selection)] = leaf
        assert values is not None
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("HDF5 correlator datasets must be real or complex numeric arrays")
        dims = list(record["dims"])
        coords = record["coords"]
        if values.ndim != len(dims) or values.shape[0] != len(configuration_ids):
            raise ValueError(f"dataset shape does not match descriptor for {record['id']}")
        for dim, size in zip(dims[1:], values.shape[1:]):
            if dim not in coords or not isinstance(coords[dim], list) or len(coords[dim]) != size:
                raise ValueError(f"coordinate length does not match dimension {dim}")
        current = record.get("current") if isinstance(record.get("current"), dict) else {}
        attrs = {
            "correlator_id": record["id"],
            "correlator_type": record["correlator_type"],
            "hadron": record.get("hadron", {}).get("name")
            if isinstance(record.get("hadron"), dict)
            else record.get("hadron"),
            "source_momentum": json.dumps(record["source_momentum"]),
            "sink_momentum": json.dumps(record["sink_momentum"]),
            "momentum_gev": float(np.linalg.norm(record["sink_momentum"]) * ensemble.k_s),
            "units": json.dumps({"values": "dimensionless", **{dim: "lattice" for dim in dims[1:]}}),
            "coord_unit": "lattice",
        }
        for key in (
            "gfix",
            "volume",
            "source_operator",
            "sink_operator",
            "current_operator",
            "polarization",
            "bz_direction",
            "bT",
            "momentum",
            "resample_group",
            "observable",
            "bT_fm",
            "momentum_index",
        ):
            if key in record["selectors"]:
                attrs[key] = record["selectors"][key]
        direction = record["selectors"].get("longitudinal_direction")
        if direction is not None:
            direction = np.asarray(direction, dtype=float)
            if direction.shape != (3,) or not np.all(np.isfinite(direction)) or np.linalg.norm(direction) == 0:
                raise ValueError("longitudinal_direction must be a finite nonzero three-vector")
            step = float(ensemble.a_s * np.linalg.norm(direction))
            authored_step = record["selectors"].get("bz_step_fm", step)
            if not np.isclose(float(authored_step), step, rtol=0.0, atol=1e-12):
                raise ValueError("bz_step_fm conflicts with lattice spacing and longitudinal_direction")
            attrs["longitudinal_direction"] = json.dumps(direction.tolist())
            attrs["bz_step_fm"] = step
            attrs["momentum_gev"] = float(
                ensemble.k_s * abs(np.dot(record["sink_momentum"], direction / np.linalg.norm(direction)))
            )
        if record["source_sink_separation"] is not None:
            attrs["source_sink_separation"] = int(record["source_sink_separation"])
        for key in ("kernel_operator", "parton", "renormalization_scheme"):
            value = record.get(key, current.get(key))
            if value is not None:
                attrs[key] = value
        raw = EnsembleData(
            ensemble,
            "raw",
            [values[index] for index in range(values.shape[0])],
            dims[1:],
            {dim: coords[dim] for dim in dims[1:]},
            attrs=attrs,
            name=record["id"],
        )
        output["correlators"][record["id"]] = raw
    return output


def resample_correlators(
    raw: dict[str, Any], *, mode: str, group: str, bin_size: int, n_boot: int | None, seed: int,
    tracked: bool = False,
) -> dict[str, EnsembleData]:
    """Apply one explicit plan to every configuration-aligned correlator."""
    data = raw["correlators"]
    first = next(iter(data.values()))
    n_configurations = first.n_sample
    if any(item.n_sample != n_configurations for item in data.values()):
        raise ValueError("configuration counts differ inside one resample group")
    if bin_size < 1 or n_configurations // bin_size < 2:
        raise ValueError("bin_size leaves fewer than two configurations")
    plan_seed: int | None = None
    if mode == "bootstrap":
        plan_seed = int(seed)
        if tracked:
            digest = hashlib.sha256(f"lamet-bootstrap-v1:{seed}:{group}".encode()).digest()
            plan_seed = int.from_bytes(digest[:8], "little")
    elif mode != "jackknife":
        raise ValueError("resampling must be bootstrap or jackknife")
    plan_header = json.dumps(
        {
            "group": group,
            "configurations": raw.get("configuration_ids", []),
            "bin_size": bin_size,
            "mode": mode,
            "n_boot": n_boot if mode == "bootstrap" else None,
            "seed": plan_seed,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    resample_id = hashlib.sha256(plan_header).hexdigest()
    result: dict[str, EnsembleData] = {}
    for correlator_id, item in data.items():
        prepared = item.bin(bin_size) if bin_size > 1 else item
        if mode == "bootstrap":
            sampled = prepared.bootstrap(int(n_boot), seed=plan_seed)
        else:
            sampled = prepared.jackknife()
        attrs = sampled.attrs
        attrs.update({"resample_id": resample_id, "resample_group": group})
        if tracked:
            frame_header = json.dumps(
                {"algorithm": "group-bootstrap-v1", "seed": int(seed), "samples": int(n_boot)}
                if mode == "bootstrap" else {"algorithm": "group-jackknife-v1"}, sort_keys=True
            )
            attrs.update({
                "replica_frame_id": hashlib.sha256(frame_header.encode()).hexdigest(),
                "resample_lineage": json.dumps({group: resample_id}, sort_keys=True, separators=(",", ":")),
                "configuration_ids": json.dumps(raw.get("configuration_ids", [])),
            })
            if mode == "jackknife":
                attrs["jackknife_groups"] = json.dumps(
                    {group: sampled.n_sample}, sort_keys=True, separators=(",", ":")
                )
                attrs["central_estimator"] = "mean_native_jackknife"
        result[correlator_id] = EnsembleData(
            sampled.ensemble,
            sampled.resample,
            [sample for sample in sampled.values],
            sampled.dims,
            sampled.coords,
            attrs=attrs,
            name=sampled.name,
        )
    return result


def _finite_fit_configurations(data: EnsembleData, params: dict[str, Any]) -> np.ndarray:
    """Check only the authored likelihood domain, including all published z."""
    physical_shape = np.asarray(data.values).shape[1:]
    domain = np.ones(physical_shape, dtype=bool)
    scopes = set("+".join(params.get("fit_scope", [])).split("+"))
    if "t" in data.dims:
        times = np.asarray(data.coords["t"], dtype=float)
        is_qda = data.attrs.get("correlator_type") == "qda"
        if "ff" in scopes and is_qda and "z" not in data.dims:
            chosen = [0.5 * float(tsep) for window in params.get("pt3_windows", []) for tsep in window["tsep_ls"]]
            keep = np.isin(times, chosen) if chosen else np.ones(times.shape, dtype=bool)
        else:
            windows = [params["qda_window"]] if is_qda and params.get("qda_window") else params.get("pt2_windows", [])
            keep = np.zeros(times.shape, dtype=bool) if windows else np.ones(times.shape, dtype=bool)
            for window in windows:
                keep |= (times >= window["tmin"]) & (times < window["tmax"])
            if not is_qda and scopes & {"3pt_over_2pt", "FH"}:
                # The ratio likelihood also consumes measured two-point
                # values outside the spectral fit window.
                for window in params.get("pt3_windows", []):
                    keep |= np.isin(times, window["tsep_ls"])
                    if params.get("fitting_form") == "NonBreit":
                        for tsep in window["tsep_ls"]:
                            keep |= (times >= window["tau_cut"]) & (times <= tsep - window["tau_cut"])
        shape = [1] * len(data.dims)
        shape[data.dims.index("t")] = len(times)
        if is_qda and "z" in data.dims and params.get("qda_windows_by_z"):
            from ._qda_windows import qda_time_masks

            windows = [params["qda_window"]] if params.get("qda_window") else params.get("pt2_windows", [])
            if not windows:
                raise ValueError("qda_windows_by_z requires an explicit default time window")
            keep_by_z = np.zeros((len(times), len(data.coords["z"])), dtype=bool)
            for window in windows:
                keep_by_z |= qda_time_masks(times, data.coords["z"], window, params["qda_windows_by_z"])
            shape[data.dims.index("z")] = len(data.coords["z"])
            if data.dims.index("t") > data.dims.index("z"):
                keep_by_z = keep_by_z.T
            domain &= keep_by_z.reshape(shape)
        else:
            domain &= keep.reshape(shape)
    if "tsep" in data.dims and "tau" in data.dims:
        tseps = np.asarray(data.coords["tsep"], dtype=float)
        taus = np.asarray(data.coords["tau"], dtype=float)
        windows = params.get("pt3_windows", [])
        keep = np.full((len(tseps), len(taus)), not bool(windows), dtype=bool)
        for window in windows:
            cut = int(window["tau_cut"])
            keep |= (
                np.isin(tseps, window["tsep_ls"])[:, None]
                & (taus[None, :] >= cut)
                & (taus[None, :] <= tseps[:, None] - cut)
            )
        shape = [1] * len(data.dims)
        shape[data.dims.index("tsep")] = len(tseps)
        shape[data.dims.index("tau")] = len(taus)
        if data.dims.index("tsep") > data.dims.index("tau"):
            keep = keep.T
        domain &= keep.reshape(shape)
    if not np.any(domain):
        raise ValueError("authored fit windows select no raw correlator data")
    values = np.asarray(data.values)[:, domain]
    # Ratios divide complex correlators before selecting a fitted component.
    # A nonfinite imaginary operand can therefore contaminate the real ratio.
    complex_arithmetic = params.get("tmd_phase_rotation") or bool(
        scopes & {"3pt_over_2pt", "3pt_over_3pt", "FH", "wf_over_2pt"}
    )
    if not complex_arithmetic:
        if params.get("component") == "re":
            values = values.real
        elif params.get("component") == "im":
            values = values.imag
    return np.all(np.isfinite(values), axis=1)


def prepare_resampling_groups(document: dict[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    """Freeze one common configuration mask before any correlated job is fitted.

    The union is over every authored job's fit domain, so later branch choices
    cannot silently change the sampling plan of an already fitted observable.
    """
    block = document.get("stages", {}).get("correlator_analysis", {})
    groups: dict[str, dict[str, Any]] = {}
    masks: dict[str, np.ndarray] = {}
    exclusion_reasons: dict[str, dict[str, list[dict[str, str]]]] = {}
    for job in block.get("jobs", []):
        params = {**block.get("defaults", {}), **job}
        for source in job.get("inputs", {}).get("correlators", []):
            path = Path(source["json"])
            if not path.is_absolute():
                path = root / path
            # Inspect selectors before loading large legacy datasets.
            records = json.loads(path.read_text(encoding="utf-8"))["correlators"]
            record = next(record for record in records if record["id"] == source["id"])
            group = record.get("selectors", {}).get("resample_group")
            if group is None:
                continue
            if not isinstance(group, str) or not group:
                raise ValueError("resample_group must be a nonempty string")
            loaded = load_descriptor(path, selected_ids={source["id"]})
            data = loaded["correlators"][source["id"]]
            count = data.n_sample
            identities = record.get("selectors", {}).get("configuration_ids", list(range(count)))
            if not isinstance(identities, list) or len(identities) != count or len(set(map(str, identities))) != count:
                raise ValueError("configuration_ids must uniquely identify every raw configuration in order")
            identities = list(map(str, identities))
            geometry = tuple(getattr(data.ensemble, name) for name in ("a_s", "a_t", "L_s", "L_t"))
            if group in groups:
                if groups[group]["all_configuration_ids"] != identities or groups[group]["geometry"] != geometry:
                    raise ValueError(f"resample_group '{group}' has inconsistent configuration ordering or lattice")
            else:
                groups[group] = {"all_configuration_ids": identities, "geometry": geometry}
                masks[group] = np.ones(count, dtype=bool)
                exclusion_reasons[group] = {}
            finite = _finite_fit_configurations(data, params)
            masks[group] &= finite
            for index in np.flatnonzero(~finite):
                exclusion_reasons[group].setdefault(str(index), []).append({
                    "configuration_id": identities[index],
                    "job_id": job["id"],
                    "correlator_id": source["id"],
                    "reason": "nonfinite_in_authored_fit_domain",
                })
    for group, entry in groups.items():
        mask = masks[group]
        retained = np.flatnonzero(mask).tolist()
        if len(retained) // int(document["metadata"]["bin_size"]) < 2:
            raise ValueError(f"resample_group '{group}' has fewer than two valid configuration bins")
        entry.update({
            "retained_indices": retained,
            "excluded_indices": np.flatnonzero(~mask).tolist(),
            "configuration_ids": [entry["all_configuration_ids"][index] for index in retained],
            "exclusion_reasons": {
                index: sorted(reasons, key=lambda reason: (reason["job_id"], reason["correlator_id"]))
                for index, reasons in sorted(exclusion_reasons[group].items(), key=lambda item: int(item[0]))
            },
        })
    return groups


def ensure_raw_correlators(
    context: Any,
    selected_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load the selected configuration-level correlators once for a job."""
    existing = context.state.get("raw_correlators")
    if isinstance(existing, dict):
        if selected_ids is not None and set(selected_ids) != set(existing):
            raise ValueError("correlators were already prepared with a different selection")
        return existing

    source = context.inputs["correlators"]
    if not isinstance(source, list):
        raise TypeError("correlators input must be a list of descriptor record selections")
    selections = source
    requested = set(selected_ids if selected_ids is not None else [item["id"] for item in selections])
    if not selections or not requested:
        raise ValueError("at least one correlator must be selected")

    by_path: dict[Path, list[str]] = {}
    for selection in selections:
        if not isinstance(selection, dict) or set(selection) != {"json", "id"}:
            raise TypeError("each correlator input must contain exactly json and id")
        path = selection["json"]
        identifier = selection["id"]
        if not isinstance(path, Path) or not isinstance(identifier, str):
            raise TypeError("correlator json must resolve to a Path and id must be a string")
        if identifier in requested:
            by_path.setdefault(path, []).append(identifier)
    selected: dict[str, Any] = {}
    records: dict[str, Any] = {}
    loaded_paths: list[Path] = []
    configuration_ids: list[str] | None = None
    ensemble: dict[str, Any] | None = None
    configuration_count: int | None = None
    for path, ids in by_path.items():
        loaded = load_descriptor(path, selected_ids=set(ids))
        loaded_paths.append(path)
        descriptor = loaded["descriptor"]
        current_ensemble = descriptor["ensemble"]
        current_count = descriptor["configuration_count"]
        if ensemble is None:
            ensemble = current_ensemble
            configuration_count = current_count
            configuration_ids = list(loaded["configuration_ids"])
        elif current_ensemble != ensemble or current_count != configuration_count:
            raise ValueError("selected correlators must share one ensemble and configuration count")
        selected.update(loaded["correlators"])
        records.update({record["id"]: record for record in descriptor["correlators"]})
    if set(selected) != requested:
        raise ValueError(
            f"selected correlator ids are not present in the project descriptors: {sorted(requested - set(selected))}"
        )
    context.state["correlator_descriptor_path"] = loaded_paths[0] if len(loaded_paths) == 1 else loaded_paths
    explicit_groups = {item.attrs.get("resample_group") for item in selected.values()}
    tracked = any(value is not None for value in explicit_groups)
    if tracked and (len(explicit_groups) != 1 or None in explicit_groups):
        raise ValueError("one correlator fit must use exactly one explicit resample_group")
    group = str(next(iter(explicit_groups))) if tracked else str((ensemble or {}).get("id", context.job_id))
    if tracked:
        shared = context.state.setdefault("stage_shared", {})
        groups = shared.get("resampling_groups", context.state.get("resampling_groups"))
        if groups is None:
            root = Path(context.manifest["metadata"]["root_directory"])
            if not root.is_absolute():
                root = context.manifest_path.parent / root
            groups = prepare_resampling_groups(context.manifest, root)
        shared["resampling_groups"] = groups
        context.state["resampling_groups"] = groups
        if group not in groups:
            raise ValueError(f"resampling group '{group}' was not preflighted")
        plan = groups[group]
        indices = np.asarray(plan["retained_indices"], dtype=int)
        configuration_ids = list(plan["configuration_ids"])
        selected = {
            key: EnsembleData(
                item.ensemble, "raw", list(np.asarray(item.values)[indices]), item.dims, item.coords,
                attrs={
                    **item.attrs,
                    "excluded_configurations": json.dumps(plan["excluded_indices"]),
                    "excluded_configuration_reasons": json.dumps(plan["exclusion_reasons"], sort_keys=True),
                }, name=item.name,
            )
            for key, item in selected.items()
        }
    context.state["correlator_resample_group"] = group
    context.state["correlator_tracked_resampling"] = tracked
    context.state["correlator_records"] = records
    context.state["correlator_configuration_ids"] = configuration_ids or []
    context.state["raw_correlators"] = selected
    return selected


def ensure_correlators(
    context: Any,
    selected_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Load and resample the selected correlators once for a job."""
    existing = context.state.get("correlators")
    if isinstance(existing, dict):
        if selected_ids is not None and set(selected_ids) != set(existing):
            raise ValueError("correlators were already prepared with a different selection")
        return existing

    raw = ensure_raw_correlators(context, selected_ids)
    resampled = resample_correlators(
        {
            "correlators": raw,
            "configuration_ids": context.state.get("correlator_configuration_ids", []),
        },
        mode=context.manifest["metadata"]["resample_mode"],
        group=str(context.state.get("correlator_resample_group", context.job_id)),
        bin_size=context.manifest["metadata"]["bin_size"],
        n_boot=context.manifest["metadata"].get("samples"),
        seed=int(context.manifest["metadata"]["random_seed"]),
        tracked=bool(context.state.get("correlator_tracked_resampling", False)),
    )
    context.state["correlators"] = resampled
    return resampled


__all__ = ["ensure_correlators", "ensure_raw_correlators", "load_descriptor", "resample_correlators",
           "prepare_resampling_groups"]
