"""Physical and workflow checks for arXiv:2312.10829 single-spacing subtraction."""

import json
from pathlib import Path

import numpy as np
import pytest

from lamet_agent.agent import ToolContext
from lamet_agent.contract import CheckContext, evaluate_rules
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.kernels import load_renormalization_kernel
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM as HC
from lamet_agent.stages.renormalization import contract, workflow
from lamet_agent.stages.renormalization._single_spacing_physics import (
    STRATEGY,
    apply_single_spacing,
    fit_single_spacing,
)


def c0(z_fm, mu=2.0):
    return np.exp(0.2 * np.asarray(z_fm) ** 2)


def reference(mode="bootstrap"):
    z = np.arange(17) * 0.09
    rng = np.random.default_rng(123)
    values = np.empty((80, len(z)))
    for index in range(len(values)):
        total = 0.5 + rng.normal(0, 0.005)
        delta = 0.7 + rng.normal(0, 0.01)
        values[index] = np.where(z < 0.4, c0(z) * np.exp(-total * z / HC - 0.13), 0.7 * np.exp(-delta * z / HC))
    values *= np.exp(rng.normal(0, 0.0002, values.shape))
    info = EnsembleInfo("test", "a09", 0.09, 0.09, 64, 128, 0.14)
    return EnsembleData(
        info,
        mode,
        list(values),
        ["z"],
        {"z": z.tolist()},
        attrs={
            "coord_unit": "fm",
            "resample_id": "common",
            "observable": "pdf",
            "operator": "gt",
            "gfix": "gi",
            "momentum_gev": 0.0,
        },
    )


def fit(ref, **kwargs):
    options = dict(
        long_distance_window_fm=[0.54, 1.44],
        short_distance_points_fm=[0.09, 0.18, 0.27],
        kernel=c0,
        kernel_id="toy_c0",
        mu=2.0,
    )
    options.update(kwargs)
    return fit_single_spacing(ref, **options)


@pytest.mark.parametrize("mode", ["bootstrap", "jackknife"])
def test_recovers_two_masses_and_keeps_covariance(mode):
    masses, diagnostics = fit(reference(mode))
    assert masses.n_sample == 80
    np.testing.assert_allclose(masses.mean[:3], [0.7, -0.2, 0.5], atol=0.004)
    np.testing.assert_allclose(masses.values[:, 0] + masses.values[:, 1], masses.values[:, 2], atol=1e-15)
    assert np.all(masses.sdev[:3] > 0)
    assert diagnostics["fit_quality"]["reference"]["dof"] == 9


def test_total_mass_independent_of_long_distance_convention():
    ref = reference()
    first, _ = fit(ref)
    altered = ref.copy()
    z = np.asarray(ref.coords["z"])
    altered.array.data[:, z >= 0.54] *= np.exp(-0.19 * z[z >= 0.54] / HC)
    second, _ = fit(altered)
    np.testing.assert_allclose(first.values[:, 2], second.values[:, 2], atol=1e-12)
    np.testing.assert_allclose(second.mean[0] - first.mean[0], 0.19, atol=5e-5)
    np.testing.assert_allclose(second.mean[1] - first.mean[1], -0.19, atol=5e-5)


def test_interpolated_scan_and_netcdf_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tmp_path = Path(".")
    ref = reference()
    masses, diag = fit(ref, short_distance_points_fm=[0.16, 0.18, 0.20], scan_centers_fm=[0.16, 0.18])
    assert [p["center_fm"] for p in diag["scan"]] == [0.16, 0.18]
    masses.to_netcdf(tmp_path / "masses.nc")
    restored = EnsembleData.from_netcdf(tmp_path / "masses.nc")
    np.testing.assert_array_equal(restored.values, masses.values)
    assert restored.attrs["mass_unit"] == "GeV"
    assert restored.attrs["reference_digest"] == masses.attrs["reference_digest"]
    json.dumps(diag, allow_nan=False)


def target_from(ref, dims=("z",), observable="pdf"):
    z = np.array([-0.72, -0.27, -0.18, 0.0, 0.18, 0.27, 0.72])
    shape = [ref.n_sample] + [len(z) if d == "z" else 2 for d in dims]
    values = np.full(shape, 0.4 + 0.2j)
    coords = {d: z.tolist() if d == "z" else [0.0, 0.2] for d in dims}
    return EnsembleData(
        ref.ensemble,
        ref.resample,
        list(values),
        list(dims),
        coords,
        attrs={**ref.attrs, "observable": observable, "momentum_gev": 2.0},
    )


@pytest.mark.parametrize("dims", [("z",), ("t", "z", "xi"), ("z", "Q2")])
def test_signed_gpd_factor_broadcasting_and_switch(dims):
    ref = reference()
    masses, _ = fit(ref)
    target = target_from(ref, dims, "gpd")
    result = apply_single_spacing(target, ref, masses, zs_fm=0.27)
    z = np.asarray(target.coords["z"])
    for j, coordinate in enumerate(z):
        distance = abs(coordinate)
        index = round(min(distance, 0.27) / 0.09)
        expected = (0.4 + 0.2j) / ref.values[:, index] * np.exp(masses.values[:, 2] * max(distance - 0.27, 0) / HC)
        actual = np.take(result.values, j, axis=1 + list(dims).index("z"))
        np.testing.assert_allclose(
            actual, np.broadcast_to(expected.reshape([-1] + [1] * (actual.ndim - 1)), actual.shape)
        )
    assert result.attrs["renormalization_scheme"] == "hybrid"
    assert result.attrs["scale_gev"] == 2.0
    # Nonforward local value is not forced to unity.
    assert not np.allclose(np.take(result.values, 3, axis=1 + list(dims).index("z")), 1)


def test_pdf_normalization_and_gpd_guard():
    ref = reference()
    masses, _ = fit(ref)
    target = target_from(ref)
    result = apply_single_spacing(target, ref, masses, zs_fm=0.27, normalization=True)
    np.testing.assert_allclose(result.values[:, 3], 1)
    target.array.attrs["observable"] = "gpd"
    with pytest.raises(ValueError, match="form factor"):
        apply_single_spacing(target, ref, masses, zs_fm=0.27, normalization=True)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"momentum_gev": 0.2}, "momentum_gev=0"),
        ({"gfix": "cg"}, "gauge-invariant"),
        ({"observable": "da"}, "collinear"),
        ({"operator": "gz"}, "gamma-t"),
    ],
)
def test_reference_channel_rejections(change, match):
    ref = reference()
    ref.array.attrs.update(change)
    with pytest.raises(ValueError, match=match):
        fit(ref)


def test_wrong_reference_or_missing_switch_rejected():
    ref = reference()
    masses, _ = fit(ref)
    target = target_from(ref)
    with pytest.raises(ValueError, match="reference grid"):
        apply_single_spacing(target, ref, masses, zs_fm=0.25)
    modified = ref.copy()
    modified.array.data *= 1.01
    with pytest.raises(ValueError, match="same reference samples"):
        apply_single_spacing(target, modified, masses, zs_fm=0.27)
    wrong = target.copy()
    wrong.array.attrs["resample_id"] = "unrelated"
    with pytest.raises(ValueError, match="resample_id"):
        apply_single_spacing(wrong, ref, masses, zs_fm=0.27)


def test_fit_window_and_negative_data_rejections():
    ref = reference()
    with pytest.raises(ValueError, match="positive coordinates"):
        fit(ref, short_distance_points_fm=[0, 0.09, 0.18])
    with pytest.raises(ValueError, match="no extrapolation"):
        fit(ref, short_distance_points_fm=[0.09, 0.18, 2.0])
    ref.array.data[:, 1] *= -1
    with pytest.raises(ValueError, match="positive"):
        fit(ref)


def params():
    return {
        "strategy": STRATEGY,
        "type": "fit",
        "normalization": False,
        "scheme": "hybrid",
        "kernel_id": "z_msbar_pdf_nlo",
        "kernel_parameters": {},
        "mu": 2.0,
        "svdcut": 1e-10,
        "long_distance_window_fm": [0.54, 1.44],
        "short_distance_points_fm": [0.09, 0.18, 0.27],
        "scan_centers_fm": [0.16, 0.18],
        "scan_half_width_fm": 0.02,
    }


def test_contract_and_checks_fit_apply():
    for p, inputs in [
        (params(), {"reference": "ref"}),
        (
            {"strategy": STRATEGY, "type": "apply", "scheme": "hybrid", "normalization": False, "zs_fm": 0.27},
            {"target": "target", "denominator": "ref", "masses": "mass_fit"},
        ),
    ]:
        assert not evaluate_rules(p, contract.PARAM_RULES)
        context = CheckContext({}, "renormalization", "test", p, inputs)
        for check in contract.CHECKS:
            assert not check(context), check.__name__
    invalid = {**params(), "short_distance_points_fm": [0, 0.1, 0.2]}
    assert contract.check_single_spacing(CheckContext({}, "renormalization", "bad", invalid, {}))
    invalid = {**params(), "kernel_id": "z_msbar_da_nlo"}
    assert contract.check_single_spacing(CheckContext({}, "renormalization", "bad", invalid, {}))


def test_workflow_fit_and_apply_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tmp_path = Path(".")
    ref = reference()
    fit_context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "renormalization",
        "fit",
        params(),
        {"reference": ref},
        {},
        {},
        tmp_path / "fit",
        np.random.default_rng(1),
    )
    workflow.run(fit_context, None)
    masses = EnsembleData.from_netcdf(tmp_path / "fit" / "output.nc")
    apply_params = {"strategy": STRATEGY, "type": "apply", "scheme": "hybrid", "normalization": False, "zs_fm": 0.27}
    apply_context = ToolContext(
        {"metadata": {}},
        tmp_path / "manifest.json",
        "renormalization",
        "apply",
        apply_params,
        {"target": target_from(ref), "denominator": ref, "masses": masses},
        {},
        {},
        tmp_path / "apply",
        np.random.default_rng(2),
    )
    workflow.run(apply_context, None)
    output = EnsembleData.from_netcdf(tmp_path / "apply" / "output.nc")
    assert output.n_sample == ref.n_sample
    assert output.attrs["renormalization_scheme"] == "hybrid"
    assert (tmp_path / "fit" / "diagnostics" / "self_renormalization.json").exists()
    from lamet_agent.stages._reporting import StageReportRecord
    from lamet_agent.stages.renormalization.reporting import write_stage_report

    records = tuple(
        StageReportRecord(c.job_id, c.params, {}, c.output, c.summary, c.artifact_directory)
        for c in (fit_context, apply_context)
    )
    report = write_stage_report(records=records, artifact_directory=tmp_path)
    text = report.read_text(encoding="utf-8")
    assert "2312.10829" in text and "Single-spacing" in text and "total_mass_gev" in text


def test_manifest_public_jobs_validate_and_preserve_old_strategy(tmp_path):
    from lamet_agent.manifest import Manifest

    for name in ("reference.nc", "target.nc"):
        (tmp_path / name).touch()
    p = params()
    p["inputs"] = {"reference": {"file": "reference.nc"}}
    p["id"] = "mass_fit"
    document = {
        "metadata": {
            "run_id": "single_a",
            "root_directory": str(tmp_path),
            "artifacts_directory": "artifacts",
            "random_seed": 7,
            "workers": 1,
            "target_observable": "gpd",
            "parton": "quark",
            "resample_mode": "bootstrap",
            "samples": 80,
            "bin_size": 1,
            "sample_error_mode": "covariance",
        },
        "stages": {
            "renormalization": {
                "jobs": [
                    p,
                    {
                        "id": "apply",
                        "strategy": STRATEGY,
                        "type": "apply",
                        "normalization": False,
                        "scheme": "hybrid",
                        "zs_fm": 0.27,
                        "inputs": {
                            "target": {"file": "target.nc"},
                            "denominator": {"file": "reference.nc"},
                            "masses": "mass_fit",
                        },
                    },
                ]
            }
        },
    }
    manifest = Manifest(tmp_path / "manifest.json", document)
    assert not manifest.validate()


def test_coordinate_coefficient_not_advertised_as_lightcone_matching():
    from lamet_agent.kernels import list_kernel_ids

    assert "z_msbar_pdf_nnlo" not in list_kernel_ids()
    assert callable(load_renormalization_kernel("z_msbar_pdf_nnlo"))


def test_nnlo_wilson_coefficient_against_eq5(monkeypatch):
    import lamet_agent.kernels.z_msbar_pdf_nnlo as coefficient

    monkeypatch.setattr(coefficient, "alpha_s", lambda *args: 0.25)
    z = 2 * np.exp(-np.euler_gamma) / 2.0 * HC
    # At the natural distance all logs vanish; the explicit nf constant remains.
    expected = 1 + 0.25 / (2 * np.pi) * (4 / 3) * 2.5 + (0.25 / (2 * np.pi)) ** 2 * (51.836 - 3 * 4.34259)
    assert coefficient.kernel(z, mu=2.0) == pytest.approx(expected)
    assert np.isfinite(load_renormalization_kernel("z_msbar_pdf_nnlo")(0.18))
