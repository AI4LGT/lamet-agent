"""The shared sample-bearing numerical exchange type for the pipeline.

This module preserves the approved ``EnsembleData`` wrapper and serialization
behavior.  Stage code adds stricter physical checks at its own boundary rather
than changing this numerical base.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, NamedTuple, Optional, Sequence, Union, get_args

import gvar
import numpy
from numpy.typing import NDArray
import xarray

from .kernels import implementation as _kernel_implementation
from .ui import warning

_DimType = str
_DimsType = Sequence[_DimType]
_IndexType = Union[int, float, str]
_CoordType = Sequence[_IndexType]
_CoordsType = Dict[_DimType, _CoordType]
_ResampleType = Literal["raw", "jackknife", "bootstrap", "gvar"]
_RESAMPLE_TYPE_VALUES = get_args(_ResampleType)
_RESAMPLE_DIM = "resample"


def format_gvar(values: Any) -> str:
    """Format every gvar element compactly without NumPy ellipsis truncation."""
    return numpy.array2string(numpy.asarray(values, dtype=object), threshold=numpy.inf, separator=" ")


def _jackknife_counts(attrs: dict[str, Any], key: str) -> dict[str, int]:
    """Read a canonical, explicitly counted group layout; never infer groups."""
    try:
        counts = json.loads(attrs[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{key} must encode a group-to-count mapping") from exc
    if not isinstance(counts, dict) or not counts or any(
        not isinstance(group, str) or not group or not isinstance(count, int)
        or isinstance(count, bool) or count < 2 for group, count in counts.items()
    ):
        raise ValueError(f"{key} requires nonempty groups with at least two jackknife samples")
    return dict(sorted(counts.items()))


def _jackknife_blocks(values: Any, attrs: dict[str, Any]):
    """Yield independent delete-one blocks; the final row is the central value."""
    counts = _jackknife_counts(attrs, "jackknife_layout")
    if len(values) != sum(counts.values()) + 1:
        raise ValueError("jackknife_layout does not match sample count including its central row")
    if "jackknife_groups" in attrs and _jackknife_counts(attrs, "jackknife_groups") != counts:
        raise ValueError("jackknife_groups and jackknife_layout disagree")
    offset = 0
    for count in counts.values():
        yield values[offset:offset + count]
        offset += count


def sample_mean(values: Any, resample: str, attrs: dict[str, Any] | None = None):
    """Central estimate for native replicas or a combined independent-group jackknife.

    Native jackknife uses its sample mean, as the reference analysis does. At a
    group merge this estimate becomes an explicit last row, evaluated through
    every subsequent nonlinear operation. No jackknife bias correction is made.
    """
    values = numpy.asarray(values)
    if resample == "jackknife" and attrs and attrs.get("jackknife_layout"):
        # Exhaust validation even though the block values are not needed here.
        tuple(_jackknife_blocks(values, attrs))
        return values[-1]
    return numpy.mean(values, axis=0)


def sample_covariance(values: Any, resample: str, attrs: dict[str, Any] | None = None):
    """Covariance with flattened physical axes and the correct group weights.

    For superjackknife, each independent group contributes (N_g-1)/N_g
    times the sum of outer products about its own delete-one block mean.
    The central row is never counted as a statistical replica.
    """
    values = numpy.asarray(values)
    if numpy.iscomplexobj(values):
        raise TypeError("covariance requires real data; select .real or .imag first.")
    flat = values.reshape(len(values), -1)
    if resample == "jackknife" and attrs and attrs.get("jackknife_layout"):
        covariance = numpy.zeros((flat.shape[1], flat.shape[1]), dtype=float)
        for block in _jackknife_blocks(flat, attrs):
            delta = block - block.mean(axis=0)
            covariance += (len(block) - 1) / len(block) * (delta.T @ delta)
        return covariance
    if len(values) == 1:
        return numpy.zeros((flat.shape[1], flat.shape[1]), dtype=float)
    covariance = numpy.atleast_2d(numpy.cov(flat, rowvar=False, ddof=1))
    if resample == "raw":
        covariance /= len(values)
    elif resample == "jackknife":
        covariance *= (len(values) - 1) ** 2 / len(values)
    elif resample != "bootstrap":
        raise ValueError(f"Unknown resampling method '{resample}'.")
    return covariance


def sample_sdev(values: Any, resample: str, attrs: dict[str, Any] | None = None):
    """Diagonal uncertainty, without allocating a full covariance matrix."""
    values = numpy.asarray(values)
    if resample == "jackknife" and attrs and attrs.get("jackknife_layout"):
        variance = numpy.zeros(values.shape[1:], dtype=float)
        for block in _jackknife_blocks(values, attrs):
            delta = block - block.mean(axis=0)
            variance += (len(block) - 1) / len(block) * numpy.sum(numpy.abs(delta) ** 2, axis=0)
        return numpy.sqrt(variance)
    if len(values) == 1:
        return numpy.zeros(values.shape[1:], dtype=float)
    std = numpy.std(values, axis=0, ddof=1)
    if resample == "raw":
        std /= len(values) ** 0.5
    elif resample == "jackknife":
        std *= (len(values) - 1) / len(values) ** 0.5
    elif resample != "bootstrap":
        raise ValueError(f"Unknown resampling method '{resample}'.")
    return std


def _combined_lineage(datasets: Sequence["EnsembleData"]) -> dict[str, str]:
    lineage: dict[str, str] = {}
    for data in datasets:
        try:
            entries = json.loads(data.attrs["resample_lineage"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("resample_lineage must encode a group-to-plan mapping") from exc
        if not isinstance(entries, dict) or not entries:
            raise ValueError("resample_lineage must be a nonempty group-to-plan mapping")
        for group, plan in entries.items():
            if not isinstance(group, str) or not group or not isinstance(plan, str) or not plan:
                raise ValueError("sample lineage groups and plans must be nonempty strings")
            if group in lineage and lineage[group] != plan:
                raise ValueError(f"conflicting resampling plans for shared group '{group}'")
            lineage[group] = plan
    return lineage


def _lineage_metadata(lineage: dict[str, str], frame: str) -> dict[str, str]:
    encoded = json.dumps(lineage, sort_keys=True, separators=(",", ":"))
    identifier = (
        next(iter(lineage.values())) if len(lineage) == 1
        else hashlib.sha256((frame + encoded).encode()).hexdigest()
    )
    return {
        "replica_frame_id": frame,
        "resample_lineage": encoded,
        "resample_id": identifier,
        "resample_group": next(iter(lineage)) if len(lineage) == 1 else "composite",
    }


def _validate_native_jackknife(data: "EnsembleData") -> dict[str, int]:
    counts = _jackknife_counts(data.attrs, "jackknife_groups")
    lineage = _combined_lineage((data,))
    if set(counts) != set(lineage):
        raise ValueError("jackknife_groups must identify exactly the tracked resampling sources")
    if data.attrs.get("jackknife_layout"):
        tuple(_jackknife_blocks(data.values, data.attrs))
    elif len(counts) != 1 or next(iter(counts.values())) != data.n_sample:
        raise ValueError("native tracked jackknife requires one group matching its sample count")
    if not numpy.array_equal(data.array.coords["resample"], numpy.arange(data.n_sample)):
        raise ValueError("jackknife inputs must preserve the original replica order")
    return counts


def merge_sample_lineage(*datasets: "EnsembleData") -> dict[str, str]:
    """Validate aligned replica order and merge tracked raw sampling plans.

    Use :func:`align_sample_lineage` before numerical operations on independent
    jackknife groups. A bootstrap row instead represents a simultaneous draw
    from all independent groups and needs no expansion.
    """
    if not datasets:
        raise ValueError("sample lineage requires at least one dataset")
    first = datasets[0]
    if any(data.resample != first.resample or data.n_sample != first.n_sample for data in datasets):
        raise ValueError("inputs must share resampling mode and replica count; "
                         "align independent jackknife groups first")
    if any(not numpy.array_equal(data.array.coords["resample"], first.array.coords["resample"]) for data in datasets):
        raise ValueError("inputs must preserve the same replica order")
    tracked = [bool(data.attrs.get("resample_lineage")) for data in datasets]
    if not any(tracked):
        identifiers = {data.attrs.get("resample_id") for data in datasets}
        if len(identifiers) > 1:
            raise ValueError("legacy inputs must share resample_id; independent groups require sample lineage")
        return {key: first.attrs[key] for key in ("resample_id", "resample_group") if key in first.attrs}
    if not all(tracked) or first.resample not in {"bootstrap", "jackknife"}:
        raise ValueError("tracked combinations require bootstrap or jackknife lineage on every input")
    frames = {data.attrs.get("replica_frame_id") for data in datasets}
    if len(frames) != 1 or not next(iter(frames)):
        raise ValueError("inputs use different bootstrap replica frames" if first.resample == "bootstrap"
                         else "inputs use different jackknife replica frames")
    lineage = _combined_lineage(datasets)
    metadata = _lineage_metadata(lineage, str(next(iter(frames))))
    if first.resample == "jackknife":
        groups = [_validate_native_jackknife(data) for data in datasets]
        layouts = [data.attrs.get("jackknife_layout") for data in datasets]
        if any(group != groups[0] for group in groups[1:]) or any(layout != layouts[0] for layout in layouts[1:]):
            raise ValueError("independent jackknife groups must be aligned before combining samples")
        metadata["jackknife_groups"] = json.dumps(groups[0], sort_keys=True, separators=(",", ":"))
        if layouts[0]:
            metadata["jackknife_layout"] = layouts[0]
            metadata["central_estimator"] = "mean_native_jackknife_then_propagated"
    return metadata


def align_sample_lineage(*datasets: "EnsembleData") -> tuple["EnsembleData", ...]:
    """Align independent delete-one groups without drawing or inflating samples.

    Each group's native rows occupy their own canonical block; all other groups
    are held at their central estimates in that block. Existing composite blocks
    are copied unchanged, so shared factors cancel exactly across DAG branches.
    Native same-group data retain their original rows and statistical convention.
    """
    if not datasets:
        raise ValueError("sample lineage requires at least one dataset")
    first = datasets[0]
    if first.resample != "jackknife" or not first.attrs.get("resample_lineage"):
        merge_sample_lineage(*datasets)
        return tuple(datasets)
    if any(data.resample != "jackknife" or not data.attrs.get("resample_lineage") for data in datasets):
        raise ValueError("tracked combinations require jackknife lineage on every input")
    lineage = _combined_lineage(datasets)
    counts: dict[str, int] = {}
    source_counts = []
    for data in datasets:
        groups = _validate_native_jackknife(data)
        source_counts.append(groups)
        for group, count in groups.items():
            if group in counts and counts[group] != count:
                raise ValueError(f"conflicting jackknife counts for shared group '{group}'")
            counts[group] = count
    frames = {data.attrs.get("replica_frame_id") for data in datasets}
    if len(frames) != 1 or not next(iter(frames)):
        raise ValueError("inputs use different jackknife replica frames")
    if len(counts) == 1 and all(not data.attrs.get("jackknife_layout") for data in datasets):
        merge_sample_lineage(*datasets)
        return tuple(datasets)
    counts = dict(sorted(counts.items()))
    layout = json.dumps(counts, sort_keys=True, separators=(",", ":"))
    metadata = _lineage_metadata(lineage, str(next(iter(frames))))
    metadata.update({"jackknife_groups": layout, "jackknife_layout": layout,
                     "central_estimator": "mean_native_jackknife_then_propagated"})
    aligned = []
    for data, groups in zip(datasets, source_counts):
        central = sample_mean(data.values, data.resample, data.attrs)
        blocks = {}
        if data.attrs.get("jackknife_layout"):
            offset = 0
            for group, count in groups.items():
                blocks[group] = data.values[offset:offset + count]
                offset += count
        else:
            blocks[next(iter(groups))] = data.values
        values = numpy.concatenate([
            blocks[group] if group in blocks else numpy.broadcast_to(central, (count, *numpy.shape(central)))
            for group, count in counts.items()
        ] + [numpy.expand_dims(central, axis=0)], axis=0)
        aligned.append(EnsembleData(data.ensemble, data.resample, list(values), data.dims, data.coords,
                                    {**data.attrs, **metadata}, data.name))
    return tuple(aligned)


class EnsembleInfo(NamedTuple):
    """Lattice ensemble metadata used for momentum conversion."""

    series: str
    id: str
    a_s: float
    a_t: float
    L_s: int
    L_t: int
    m_pi: float

    @property
    def k_s(self) -> float:
        return 2 * numpy.pi / self.L_s * _kernel_implementation.HBAR_C_GEV_FM / self.a_s

    @property
    def k_t(self) -> float:
        return 2 * numpy.pi / self.L_t * _kernel_implementation.HBAR_C_GEV_FM / self.a_t


def _is_gvar_values(values: object) -> bool:
    if isinstance(values, gvar.GVar):
        return True
    array = numpy.asarray(values)
    if array.ndim == 0:
        return isinstance(array.item(), gvar.GVar)
    if array.dtype != object:
        return False
    return all(isinstance(value, gvar.GVar) for value in array.flat)


class EnsembleData:
    """An xarray array with a mandatory leading resampling dimension."""

    def __init__(
        self,
        ensemble: Optional[EnsembleInfo],
        resample: _ResampleType,
        values: Union[List[Union[int, float, complex, NDArray]], gvar.GVar, NDArray[gvar.GVar]],
        dims: _DimsType,
        coords: _CoordsType,
        attrs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
    ) -> None:
        if resample not in _RESAMPLE_TYPE_VALUES:
            raise ValueError(f"Unknown resampling method '{resample}'.")
        self.resample = resample
        self.ensemble = ensemble
        self.array = self._build_xarray(resample, values, dims, coords, attrs, name)

    @staticmethod
    def _build_xarray(resample, values, dims, coords, attrs=None, name=None) -> xarray.DataArray:
        if _RESAMPLE_DIM in dims:
            raise ValueError(f"Physical dimensions should not include resampling dimension '{_RESAMPLE_DIM}'.")
        if isinstance(values, list):
            if resample == "gvar":
                raise TypeError("'gvar' does not support list of samples")
            if len(values) == 0:
                raise ValueError("Resampled values cannot be empty.")
            resample_values = numpy.stack(values, axis=0)
        else:
            if resample != "gvar":
                raise TypeError("raw/jackknife/bootstrap data must be initialized from a list of samples.")
            if not _is_gvar_values(values):
                raise TypeError("resample='gvar' requires a gvar.GVar or an array of gvar.GVar.")
            resample_values = numpy.expand_dims(numpy.asarray(values, dtype=object), axis=0)
        if resample_values.ndim != len(dims) + 1:
            raise ValueError("Resampled data must have one leading sample axis")
        resample_coords: dict[str, Any] = {_RESAMPLE_DIM: list(range(resample_values.shape[0]))}
        for dim, size in zip(dims, resample_values.shape[1:]):
            if dim not in coords:
                raise ValueError(f"Missing dimension coordinate '{dim}'.")
            if len(coords[dim]) != size:
                raise ValueError(f"Unmatched length of coordinates for dimension '{dim}'")
            resample_coords[dim] = list(coords[dim])
        return xarray.DataArray(
            resample_values, coords=resample_coords, dims=(_RESAMPLE_DIM, *dims), name=name, attrs=attrs
        )

    @classmethod
    def _from_xarray(
        cls, ensemble: Optional[EnsembleInfo], resample: _ResampleType, array: xarray.DataArray
    ) -> "EnsembleData":
        if resample not in _RESAMPLE_TYPE_VALUES:
            raise ValueError(f"Unknown resampling method '{resample}'.")
        if len(array.dims) == 0 or array.dims[0] != _RESAMPLE_DIM:
            raise ValueError(f"The first xarray dimension must be '{_RESAMPLE_DIM}'.")
        if resample == "gvar":
            if array.sizes[_RESAMPLE_DIM] != 1:
                raise ValueError("resample='gvar' requires a length-1 dimension.")
            if not _is_gvar_values(array.values):
                raise TypeError("resample='gvar' requires gvar values.")
        obj = cls.__new__(cls)
        obj.ensemble = ensemble
        obj.resample = resample
        obj.array = array.copy(deep=False)
        return obj

    def to_netcdf(self, path: Union[str, Path]) -> None:
        """Persist the current sample representation as NetCDF4."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        array = self.array.copy(deep=False)
        array.attrs["ensemble"] = json.dumps(None if self.ensemble is None else self.ensemble._asdict())
        array.attrs["resample"] = self.resample
        if self.resample == "gvar":
            mean = numpy.asarray(gvar.mean(self.array.values), dtype=float)
            sdev = numpy.asarray(gvar.sdev(self.array.values), dtype=float)
            mean_da = xarray.DataArray(
                mean, coords=self.array.coords, dims=self.array.dims, name=self.array.name, attrs=dict(array.attrs)
            )
            mean_da.attrs["gvar_encoding"] = "mean_sdev"
            dataset = mean_da.to_dataset(name=mean_da.name or "data")
            dataset["sdev"] = (self.array.dims, sdev)
            dataset.to_netcdf(output, format="NETCDF4")
            return
        array.to_netcdf(output, format="NETCDF4", auto_complex=True)

    @classmethod
    def from_netcdf(cls, path: Union[str, Path]) -> "EnsembleData":
        try:
            dataset = xarray.open_dataset(path)
            if "gvar_encoding" in dataset.attrs or (
                len(dataset.data_vars) and "gvar_encoding" in next(iter(dataset.data_vars.values())).attrs
            ):
                data_name = next(name for name in dataset.data_vars if name != "sdev")
                mean_da = dataset[data_name]
                sdev = numpy.asarray(dataset["sdev"].values, dtype=float)
                gvar_values = gvar.gvar(numpy.asarray(mean_da.values, dtype=float), sdev)
                attrs = dict(mean_da.attrs)
                attrs.pop("gvar_encoding", None)
                ensemble_payload = json.loads(attrs.pop("ensemble"))
                ensemble = None if ensemble_payload is None else EnsembleInfo(**ensemble_payload)
                resample = attrs.pop("resample")
                rebuilt = xarray.DataArray(
                    gvar_values,
                    coords=mean_da.coords,
                    dims=mean_da.dims,
                    name=None if mean_da.name == "data" else mean_da.name,
                    attrs=attrs,
                )
                dataset.close()
                return cls._from_xarray(ensemble, resample, rebuilt)
            dataset.close()
        except (OSError, ValueError, KeyError, StopIteration):
            try:
                dataset.close()
            except UnboundLocalError:
                pass
        array = xarray.load_dataarray(path, auto_complex=True)
        ensemble_payload = json.loads(array.attrs.pop("ensemble"))
        ensemble = None if ensemble_payload is None else EnsembleInfo(**ensemble_payload)
        resample = array.attrs.pop("resample")
        return cls._from_xarray(ensemble, resample, array)

    def __repr__(self) -> str:
        return repr(self.array)

    @property
    def values(self):
        return self.array.values

    @property
    def dims(self) -> list[str]:
        return list(self.array.dims[1:])

    @property
    def coords(self) -> _CoordsType:
        return {dim: self.array.coords[dim].values.tolist() for dim in self.array.dims[1:]}

    @property
    def attrs(self) -> Dict[str, Any]:
        return dict(self.array.attrs)

    @property
    def name(self) -> Optional[str]:
        return self.array.name

    @property
    def real(self) -> "EnsembleData":
        return self._from_xarray(self.ensemble, self.resample, self.array.real)

    @property
    def imag(self) -> "EnsembleData":
        return self._from_xarray(self.ensemble, self.resample, self.array.imag)

    def copy(self, deep: bool = True) -> "EnsembleData":
        return self._from_xarray(self.ensemble, self.resample, self.array.copy(deep=deep))

    @property
    def n_sample(self) -> int:
        return self.array.sizes[_RESAMPLE_DIM]

    def bin(self, bin_size: int) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled.")
        if bin_size <= 0 or bin_size >= self.n_sample:
            raise ValueError("bin_size must be positive and smaller than the sample count.")
        if self.n_sample % bin_size:
            warning("The final incomplete bin is dropped.")
        n_bins = self.n_sample // bin_size
        values = [self.array.values[index * bin_size : (index + 1) * bin_size].mean(axis=0) for index in range(n_bins)]
        return EnsembleData(self.ensemble, "raw", values, self.dims, self.coords, self.attrs, self.name)

    def jackknife(self) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled.")
        total = self.array.values.sum(axis=0)
        values = [(total - self.array.values[index]) / (self.n_sample - 1) for index in range(self.n_sample)]
        return EnsembleData(self.ensemble, "jackknife", values, self.dims, self.coords, self.attrs, self.name)

    def bootstrap(self, n_resample: int, *, seed: int | None = None) -> "EnsembleData":
        if self.resample != "raw":
            raise ValueError("Only resample='raw' can be resampled.")
        if n_resample < 1:
            raise ValueError("n_resample must be positive.")
        indices = numpy.random.default_rng(seed).integers(0, self.n_sample, (n_resample, self.n_sample))
        values = [self.array.values[indices[index]].mean(axis=0) for index in range(n_resample)]
        return EnsembleData(self.ensemble, "bootstrap", values, self.dims, self.coords, self.attrs, self.name)

    @classmethod
    def concat(
        cls, data_list: Sequence["EnsembleData"], dim: _DimType, coord: Optional[_CoordType] = None
    ) -> "EnsembleData":
        if not data_list:
            raise ValueError("Cannot concatenate an empty list of EnsembleData.")
        if dim == _RESAMPLE_DIM:
            raise ValueError(f"Cannot concatenate along '{_RESAMPLE_DIM}'.")
        first = data_list[0]
        for data in data_list[1:]:
            if data.ensemble != first.ensemble or data.resample != first.resample or data.dims != first.dims:
                raise ValueError("EnsembleData metadata must match for concatenation.")
            for other_dim in first.dims:
                if other_dim != dim and data.coords[other_dim] != first.coords[other_dim]:
                    raise ValueError(f"Coordinates differ for dimension '{other_dim}'.")
        lineage = {}
        if any(data.attrs.get("resample_lineage") for data in data_list):
            data_list = align_sample_lineage(*data_list)
            first = data_list[0]
            lineage = merge_sample_lineage(*data_list)
        array = xarray.concat([data.array for data in data_list], dim)
        if dim in first.dims:
            if coord is not None:
                raise ValueError("Coordinates cannot be supplied for an existing dimension.")
            dims_out = [_RESAMPLE_DIM, *first.dims]
        else:
            if coord is None:
                raise ValueError("Coordinates are required for a new dimension.")
            array = array.assign_coords({dim: coord})
            dims_out = [_RESAMPLE_DIM, dim, *first.dims]
        array.attrs = {**first.attrs, **lineage}
        return cls._from_xarray(first.ensemble, first.resample, array.transpose(*dims_out).sortby(dim))

    def at(self, dim: _DimType, coord: Union[_IndexType, _CoordType]) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return self._from_xarray(self.ensemble, self.resample, self.array.sel({dim: coord}, drop=True))

    def near(self, dim: _DimType, coord: Union[_IndexType, _CoordType], tolerance: float = 1e-8) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return self._from_xarray(
            self.ensemble, self.resample, self.array.sel({dim: coord}, method="nearest", tolerance=tolerance, drop=True)
        )

    @property
    def gvar(self):
        if self.resample == "gvar":
            return self.array.values[0]
        else:
            if numpy.iscomplexobj(self.array.values):
                raise TypeError("gvar conversion requires real data; select .real or .imag first.")
            shape = self.array.values.shape[1:]
            mean = numpy.asarray(sample_mean(self.values, self.resample, self.attrs)).reshape(-1)
            cov = sample_covariance(self.values, self.resample, self.attrs)
            # A scalar gvar's second argument is a standard deviation, whereas
            # an array gvar accepts a covariance matrix. Scalar ratio factors
            # must therefore take the square root before crossing that API.
            if not shape:
                return gvar.gvar(mean[0], numpy.sqrt(cov[0, 0]))
            return gvar.gvar(mean.reshape(shape), cov.reshape(shape + shape))

    @property
    def gvar_mean(self):
        if self.resample == "gvar":
            return gvar.gvar(gvar.mean(self.array.values[0]), gvar.sdev(self.array.values[0]))
        else:
            if numpy.iscomplexobj(self.array.values):
                raise TypeError("gvar conversion requires real data; select .real or .imag first.")
            mean = sample_mean(self.values, self.resample, self.attrs)
            std = sample_sdev(self.values, self.resample, self.attrs)
            return gvar.gvar(mean, std)

    @property
    def gvar_median(self):
        if self.resample == "gvar":
            return gvar.gvar(gvar.mean(self.array.values[0]), gvar.sdev(self.array.values[0]))
        else:
            if numpy.iscomplexobj(self.array.values):
                raise TypeError("gvar conversion requires real data; select .real or .imag first.")
            if self.resample == "jackknife" and self.attrs.get("jackknife_layout"):
                raise ValueError("one_sigma percentiles are undefined for independent-group jackknife blocks; "
                                 "use covariance or variance")
            n_sample = self.array.values.shape[0]
            values = self.array.values
            meanm, mean, meanp = numpy.percentile(values, q=[50 - 34.1344746, 50, 50 + 34.1344746], axis=0)
            std = numpy.maximum(meanp - mean, mean - meanm)
            if self.resample == "raw":
                std /= n_sample**0.5
            elif self.resample == "jackknife":
                std *= (n_sample - 1) / n_sample**0.5
            elif self.resample == "bootstrap":
                pass
            else:
                raise ValueError(f"Unknown resampling method '{self.resample}'.")
            return gvar.gvar(mean, std)

    def average(self, mode: Literal["covariance", "variance", "one_sigma"] = "covariance"):
        """Return the selected center and uncertainty representation as gvars."""
        if mode == "covariance":
            return self.gvar
        if mode == "one_sigma":
            return self.gvar_median
        if mode == "variance":
            return self.gvar_mean
        raise ValueError("average mode must be 'covariance', 'variance', or 'one_sigma'")

    @property
    def mean(self):
        return gvar.mean(self.array.values[0]) if self.resample == "gvar" else gvar.mean(self.gvar)

    @property
    def sdev(self):
        return gvar.sdev(self.array.values[0]) if self.resample == "gvar" else gvar.sdev(self.gvar)

    def avg_data(self) -> "EnsembleData":
        return EnsembleData(self.ensemble, "gvar", self.gvar, self.dims, self.coords, self.attrs, self.name)

    def update_dim(self, dim: _DimType, dim_out: _DimType, coord_out: Optional[_CoordType] = None) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")
        if dim_out != dim and (dim_out == _RESAMPLE_DIM or dim_out in self.dims):
            raise ValueError(f"Output dimension '{dim_out}' already exists.")
        array = self.array.rename({dim: dim_out})
        if coord_out is not None:
            array = array.assign_coords({dim_out: coord_out})
        return self._from_xarray(self.ensemble, self.resample, array)

    def sort_dim(self, dim: _DimType, ascending: bool = True) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Dimension '{dim}' not found in data dimensions.")
        return self._from_xarray(self.ensemble, self.resample, self.array.sortby(dim, ascending=ascending))

    def aligned_ref_array(self, ref: "EnsembleData") -> xarray.DataArray:
        if not isinstance(ref, EnsembleData):
            raise TypeError("Reference data must be EnsembleData.")
        if self.resample == "gvar":
            ref_array = xarray.DataArray(ref.gvar, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        elif self.resample == "raw":
            ref_array = xarray.DataArray(ref.mean, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        elif ref.resample in {"gvar", "raw"}:
            ref_array = xarray.DataArray(ref.mean, dims=ref.dims, coords=ref.coords, attrs=ref.attrs, name=ref.name)
        elif ref.resample == self.resample:
            ref_array = ref.array
        else:
            raise ValueError("Target and reference resampling methods are incompatible.")
        for dim in ref_array.dims:
            if dim not in self.array.dims:
                raise ValueError(f"Reference dimension '{dim}' not found in target data.")
            try:
                ref_array = ref_array.sel({dim: self.array.coords[dim]})
            except KeyError as exc:
                raise ValueError(f"Reference coordinates for '{dim}' do not cover target data.") from exc
        return ref_array.broadcast_like(self.array).transpose(*self.array.dims)

    def apply_renormalization(
        self, renorm_scheme: "EnsembleData", operator: Callable[[xarray.DataArray, xarray.DataArray], xarray.DataArray]
    ) -> "EnsembleData":
        target, reference = self, renorm_scheme
        lineage = {}
        if self.attrs.get("resample_lineage") or renorm_scheme.attrs.get("resample_lineage"):
            target, reference = align_sample_lineage(self, renorm_scheme)
            lineage = merge_sample_lineage(target, reference)
        values = operator(target.array, target.aligned_ref_array(reference))
        values.attrs = {**target.attrs, **lineage}
        values.name = self.name
        return self._from_xarray(self.ensemble, self.resample, values)

    def mul(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value * other)

    def div(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value / other)

    def add(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value + other)

    def sub(self, rhs: "EnsembleData") -> "EnsembleData":
        return self.apply_renormalization(rhs, lambda value, other: value - other)

    def transform_dim(
        self,
        dim: _DimType,
        dim_out: _DimType,
        coord_out: _CoordType,
        function: Callable[[NDArray, _CoordType, _CoordType, Dict[_DimType, _IndexType]], NDArray],
        dims_dispatch: Union[_DimType, _DimsType, None] = None,
    ) -> "EnsembleData":
        if dim not in self.dims:
            raise ValueError(f"Input dimension '{dim}' not found in data dimensions.")
        if dim_out != dim and (dim_out == _RESAMPLE_DIM or dim_out in self.dims):
            raise ValueError(f"Output dimension '{dim_out}' already exists.")
        if dims_dispatch is None:
            dispatch: list[str] = []
        elif isinstance(dims_dispatch, str):
            dispatch = [dims_dispatch]
        else:
            dispatch = list(dims_dispatch)
        for dispatch_dim in dispatch:
            if dispatch_dim in {dim, dim_out} or dispatch_dim not in self.dims:
                raise ValueError(f"Invalid dispatch dimension '{dispatch_dim}'.")
        dims_core = [dim] + [
            candidate for candidate in self.array.dims if candidate != dim and candidate not in dispatch
        ]
        dims_out_core = [dim_out] + [candidate for candidate in dims_core if candidate != dim]
        dims_out = [dim_out if candidate == dim else candidate for candidate in self.array.dims]

        def apply_function(value: NDArray, *index_list: _IndexType) -> NDArray:
            return function(
                value, self.coords[dim], coord_out, {name: index for name, index in zip(dispatch, index_list)}
            )

        array = xarray.apply_ufunc(
            apply_function,
            self.array,
            *[self.array.coords[name] for name in dispatch],
            input_core_dims=[dims_core] + [[]] * len(dispatch),
            output_core_dims=[dims_out_core],
            exclude_dims={dim},
            vectorize=bool(dispatch),
            output_sizes={dim_out: len(coord_out)},
        )
        array = array.assign_coords({dim_out: coord_out}).transpose(*dims_out)
        array.attrs = self.attrs
        array.name = self.name
        return self._from_xarray(self.ensemble, self.resample, array)


__all__ = ["EnsembleInfo", "EnsembleData"]
