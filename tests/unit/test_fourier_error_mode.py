import json
from pathlib import Path

import gvar as gv
import numpy as np
import pytest

from lamet_agent.data import EnsembleData
from lamet_agent.stages.fourier_transform import physics


def data(mode=None):
    z = np.arange(5, 16) / 10
    rng = np.random.default_rng(52)
    samples = (
        np.exp(-0.5 * z / physics.HBAR_C_GEV_FM)[None, :]
        + 0.05 * rng.normal(size=(80, 1))
        + 0.01 * rng.normal(size=(80, len(z)))
    )
    attrs = {} if mode is None else {"sample_error_mode": mode}
    return EnsembleData(None, "bootstrap", list(samples), ["z"], {"z": z.tolist()}, attrs=attrs)


def fit(d):
    means, widths = physics._scan_tail_priors(model_id="gi_nla", order="LA", lambda0_gev=0, hadron="proton")
    return physics.fit_tail_parameters(
        d,
        model_id="gi_nla",
        order="LA",
        hadron="proton",
        source_component="re",
        z_min_fm=0.5,
        z_max_fm=1.5,
        prior_means=means,
        prior_widths=widths,
        mode="center",
    )


@pytest.mark.parametrize("construction", ["gi", "cg"])
def test_pdf_example_mode_controls_actual_fit_weights(monkeypatch, construction):
    manifest = json.loads(
        (Path(__file__).resolve().parents[2] / f"examples/pion_pdf_{construction}_manifest.json").read_text()
    )
    assert manifest["metadata"]["sample_error_mode"] == "variance"
    d = physics.bind_sample_error_mode(data(), manifest["metadata"])
    original = physics.nonlinear_fit
    seen = []

    def capture(inputs, fcn, prior, **kwargs):
        covariance = np.asarray(kwargs["covariance"])
        seen.append(kwargs["sample_error_mode"])
        np.testing.assert_allclose(covariance, np.diag(np.diag(covariance)))
        expected = np.var(d.values, axis=0, ddof=1)
        np.testing.assert_allclose(np.diag(covariance), expected)
        return original(inputs, fcn, prior, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    _, di = fit(d)
    assert seen == ["variance"]
    assert di["sample_error_mode"] == "variance"


def test_explicit_covariance_keeps_correlations(monkeypatch):
    d = data("covariance")
    original = physics.nonlinear_fit

    def capture(inputs, fcn, prior, **kwargs):
        np.testing.assert_allclose(kwargs["covariance"], gv.evalcov(d.average("covariance")))
        assert np.any(np.abs(np.triu(kwargs["covariance"], 1)) > 1e-5)
        return original(inputs, fcn, prior, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    assert fit(d)[1]["sample_error_mode"] == "covariance"


@pytest.mark.parametrize("mode", [None, "unknown"])
def test_missing_or_invalid_fit_weighting_fails(mode):
    with pytest.raises(ValueError, match="explicit attrs.sample_error_mode"):
        fit(data(mode))


def test_manifest_mismatch_fails_without_mutating_input():
    d = data("covariance")
    with pytest.raises(ValueError, match="mismatch"):
        physics.bind_sample_error_mode(d, {"sample_error_mode": "variance"})
    assert d.attrs["sample_error_mode"] == "covariance"
    missing = data()
    bound = physics.bind_sample_error_mode(missing, {"sample_error_mode": "variance"})
    assert "sample_error_mode" not in missing.attrs
    assert bound.attrs["sample_error_mode"] == "variance"
