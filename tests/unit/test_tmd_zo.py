"""Analytic RG checks and correlated extraction/application through public jobs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from lamet_agent.contract import evaluate_rules, evaluate_checks, CheckContext
from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.kernels.implementation import HBAR_C_GEV_FM as HC
from lamet_agent.kernels.tmd.zo import coefficient, anomalous_dimension, _evolve
from lamet_agent.stages.renormalization._zo import extract, apply
from lamet_agent.stages.renormalization.contract import JOB_RULES, CHECKS

POINTS = [[0.0, 0.16], [0.08, 0.16], [0.0, 0.24], [0.08, 0.24]]


def data(resample="bootstrap"):
    rng = np.random.default_rng(881)
    factor = np.linspace(1.15, 1.25, 20)
    z, b = np.array([0.0, 0.08]), np.array([0.16, 0.24])
    c = coefficient(z[:, None], b[None, :])
    values = factor[:, None, None] * c
    # Zero-mean coordinate noise, preserving the exact mean factor in every replica.
    noise = rng.normal(0, 1e-4, values.shape)
    noise -= noise.mean(axis=(1, 2), keepdims=True)
    values += c * noise
    attrs = dict(
        gfix="GI",
        observable="tmdpdf",
        operator="gt",
        coord_unit="fm",
        bT_unit="fm",
        momentum_gev=0.0,
        wilson_loop_subtracted="true",
        reference_normalization="local_vector_charge",
        renormalization_family="test-action-smearing-endpoints",
        resample_id="shared",
    )
    info = EnsembleInfo("synthetic", "a08", 0.08, 0.08, 48, 96, 0.35)
    ref = EnsembleData(info, resample, list(values), ["z", "bT"], {"z": z, "bT": b}, attrs)
    target = ref.copy()
    target.array.data = factor[:, None, None] * np.ones_like(values) * (2 + 0.3j)
    target.array.attrs.update(momentum_gev=1.5, operator="gtgperpg5")
    for item in (ref, target):
        item.array = item.array.assign_coords(resample=np.arange(20)[::-1])
    return ref, target, factor


def test_nlo_formula_sign_symmetry_and_endpoint_gamma():
    z, b, mu, alpha = 0.08, 0.16, 2.0, 0.22
    log = np.log((z * z + b * b) / HC**2 * mu * mu * np.exp(2 * np.euler_gamma) / 4)
    expected = 1 + alpha * (4 / 3) / (2 * np.pi) * (1.5 * log - 2 * z / b * np.arctan(z / b) + 0.5)
    assert coefficient(z, b, mu=mu, alpha_s=alpha, order="nlo") == pytest.approx(expected)
    assert coefficient(-z, b) == coefficient(z, b)
    h = 1e-5
    derivative = (
        coefficient(z, b, mu=mu * np.exp(h), alpha_s=alpha, order="nlo")
        - coefficient(z, b, mu=mu * np.exp(-h), alpha_s=alpha, order="nlo")
    ) / (2 * h)
    assert derivative == pytest.approx(3 * (4 / 3) * alpha / (2 * np.pi), rel=1e-9)


def test_rgr_matches_independent_coupling_quadrature_and_varied_boundary():
    # Solve implicit two-loop running by integrating d alpha / beta, independently of ODE.
    mu, alpha, z, b, kappa = 2.0, 0.293, 0.08, 0.16, 0.8
    mu0 = kappa * 2 * np.exp(-np.euler_gamma) * HC / np.hypot(z, b)

    def beta(a):
        return -9 * a * a / (2 * np.pi) - 64 * a**3 / (8 * np.pi**2)

    a0 = brentq(lambda a: quad(lambda q: 1 / beta(q), alpha, a)[0] - np.log(mu0 / mu), 0.294, 0.8)

    # gamma_h = -2 gamma_HQET, with CF=4/3, CA=3, TF=1/2, nf=3.
    def gamma(a):
        return (
            8 * a / (4 * np.pi)
            + (4 / 3)
            * ((4 / 3) * (-5 + 16 * np.pi**2 / 3) + 3 * (49 / 3 - 4 * np.pi**2 / 3) - 10)
            * (a / (4 * np.pi)) ** 2
        )

    u = np.exp(quad(lambda q: gamma(q) / beta(q), a0, alpha)[0])
    expected = (1 + a0 * (4 / 3) / (2 * np.pi) * (3 * np.log(kappa) - 2 * z / b * np.arctan(z / b) + 0.5)) * u
    assert coefficient(z, b, scale_factor=kappa) == pytest.approx(expected, rel=2e-9)
    # Evolution group property checks integration orientation and both endpoints.
    am, u1 = _evolve(2.0, alpha, 1.5, 3)
    _, u2 = _evolve(1.5, am, 1.2, 3)
    _, u = _evolve(2.0, alpha, 1.2, 3)
    assert u1 * u2 == pytest.approx(u, rel=2e-9)
    assert anomalous_dimension(alpha, 3) == pytest.approx(gamma(alpha))


@pytest.mark.parametrize("resample", ["bootstrap", "jackknife"])
def test_correlated_average_and_complex_application(resample):
    ref, target, truth = data(resample)
    factor, diagnostic = extract(ref, points_fm=POINTS)
    np.testing.assert_allclose(factor.values[:, 1], truth, atol=1e-14)
    result = apply(target, factor, mu=2.0)
    np.testing.assert_allclose(result.values, 2 + 0.3j, atol=1e-14)
    np.testing.assert_array_equal(result.array.coords["resample"], target.array.coords["resample"])
    assert diagnostic["scale_envelope"] > 0
    assert diagnostic["window_diagnostic"]["automatic_plateau_validation"] is False
    varied = apply(target, factor, mu=2.0, scale_factor=0.8)
    np.testing.assert_allclose(varied.values, target.values / factor.values[:, 0, None, None])
    assert varied.attrs["renormalization_scheme"] == "msbar"


def test_axis_permutation():
    ref, target, _ = data()
    ref.array = ref.array.transpose("resample", "bT", "z")
    f, _ = extract(ref, points_fm=POINTS)
    np.testing.assert_allclose(apply(target, f, mu=2.0).values, 2 + 0.3j, atol=1e-14)


def test_independent_jackknife_groups_preserve_central_row_and_covariance():
    from lamet_agent.data import sample_mean, sample_covariance

    ref, target, _ = data("jackknife")
    for obj, group in ((ref, "ref"), (target, "target")):
        obj.array.attrs.update(
            replica_frame_id="common-frame",
            resample_lineage=json.dumps({group: group + "-plan"}),
            jackknife_groups=json.dumps({group: obj.n_sample}),
        )
        obj.array = obj.array.assign_coords(resample=np.arange(obj.n_sample))
    f, _ = extract(ref, points_fm=POINTS)
    result = apply(target, f, mu=2.0)
    assert result.n_sample == 41
    assert json.loads(result.attrs["jackknife_layout"]) == {"ref": 20, "target": 20}
    np.testing.assert_allclose(result.values[-1], target.values.mean(axis=0) / f.values[:, 1].mean())
    np.testing.assert_allclose(sample_mean(result.values, "jackknife", result.attrs), result.values[-1])
    covariance = sample_covariance(result.values.real, "jackknife", result.attrs)
    flat = result.values.real.reshape(41, -1)
    expected = np.zeros_like(covariance)
    for block in (flat[:20], flat[20:40]):
        delta = block - block.mean(axis=0)
        expected += 19 / 20 * delta.T @ delta
    np.testing.assert_allclose(covariance, expected)


def test_fixed_order_extraction_and_fixed_b_target():
    ref, target, _ = data()
    f, d = extract(ref, points_fm=POINTS, order="nlo")
    assert f.coords["scale_factor"] == [1.0]
    assert d["scale_envelope"] == 0
    target.array = target.array.isel(bT=0, drop=True)
    target.array.attrs["bT_fm"] = 0.16
    result = apply(target, f, mu=2.0)
    assert result.dims == ["z"]
    np.testing.assert_allclose(result.values, target.values / f.values[:, 0, None])
    with pytest.raises(ValueError, match="not present"):
        apply(target, f, mu=2.0, scale_factor=0.8)


@pytest.mark.parametrize(
    "field,value",
    [
        ("gfix", "CG"),
        ("momentum_gev", 1.0),
        ("operator", "tensor"),
        ("wilson_loop_subtracted", "false"),
        ("reference_normalization", "unknown"),
        ("coord_unit", "lattice"),
    ],
)
def test_bad_reference_metadata(field, value):
    ref, _, _ = data()
    ref.array.attrs[field] = value
    with pytest.raises(ValueError):
        extract(ref, points_fm=POINTS)


@pytest.mark.parametrize(
    "points",
    [
        [[0, 0.16]],
        [[0, 0.16], [0, 0.16], [0.08, 0.24]],
        [[0, 0.16], [0.04, 0.16], [0, 0.24]],
        [[0, 0.08], [0.08, 0.08], [0, 0.24]],
        [[0, 0.16], [0, 0.24], [0, 0.32]],
    ],
)
def test_bad_reference_window(points):
    with pytest.raises(ValueError):
        extract(data()[0], points_fm=points)


def test_invalid_application_and_domain():
    ref, target, _ = data()
    f, _ = extract(ref, points_fm=POINTS)
    with pytest.raises(ValueError, match="scale"):
        apply(target, f, mu=3.0)
    target.array.attrs["resample_id"] = "different"
    with pytest.raises(ValueError, match="resample_id"):
        apply(target, f, mu=2.0)
    with pytest.raises(ValueError, match="scale"):
        coefficient(0, 1.0)
    with pytest.raises(ValueError, match="b_fm"):
        coefficient(0, 0)
    with pytest.raises(ValueError, match="NNLO"):
        coefficient(0, 0.2, order="nnlo")


def test_contract_defaults_and_checks():
    job = dict(
        id="zo",
        strategy="tmd_zo",
        type="fit",
        normalization=False,
        mu=2.0,
        alpha_s=0.293,
        n_f=3,
        points_fm=POINTS,
        inputs={"reference": {"file": "ref.nc"}},
    )
    assert not evaluate_rules({"jobs": [job]}, JOB_RULES)
    context = CheckContext({}, "renormalization", "zo", job, job["inputs"])
    assert not evaluate_checks(CHECKS, context)
    job["normalization"] = True
    assert evaluate_checks(CHECKS, context)


def test_manifest_netcdf_and_report(tmp_path, monkeypatch):
    from lamet_agent.agent import create_session
    from lamet_agent.manifest import load_manifest

    monkeypatch.chdir(tmp_path)
    ref, target, _ = data()
    ref.to_netcdf("ref.nc")
    target.to_netcdf("target.nc")
    jobs = [
        dict(id="zo", type="fit", inputs={"reference": {"file": "ref.nc"}}, points_fm=POINTS, alpha_s=0.293, n_f=3),
        dict(id="ren", type="apply", inputs={"target": {"file": "target.nc"}, "zO": "zo"}),
        dict(id="ren_low", type="apply", scale_factor=0.8, inputs={"target": {"file": "target.nc"}, "zO": "zo"}),
    ]
    document = {
        "metadata": dict(
            run_id="zo_smoke",
            root_directory=".",
            artifacts_directory="artifacts",
            random_seed=7,
            workers=1,
            target_observable="tmd",
            parton="quark",
            resample_mode="bootstrap",
            samples=20,
            bin_size=1,
            sample_error_mode="variance",
        ),
        "stages": {"renormalization": {"defaults": dict(strategy="tmd_zo", normalization=False, mu=2.0), "jobs": jobs}},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document))
    manifest = load_manifest(path)
    assert not manifest.validate()

    def no_llm(**kwargs):
        raise AssertionError("No LLM is needed for authored extraction")

    session = create_session(SimpleNamespace(identity="test", complete=no_llm), progress_mode="none")
    try:
        result = session.run_manifest(manifest)
    finally:
        session.close()
    np.testing.assert_allclose(result["outputs"]["ren"].values, 2 + 0.3j, atol=1e-14)
    for job in manifest.jobs:
        saved = EnsembleData.from_netcdf(job.artifact_directory / "output.nc")
        np.testing.assert_allclose(saved.values, result["outputs"][job.job_id].values)
    assert "renormalization" in result["stage_reports"]
