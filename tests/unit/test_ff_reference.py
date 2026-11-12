"""Compare the wall-current extraction with independently fitted reference ratios."""

import gvar as gv
import lsqfit
import numpy as np
import pytest

from lamet_agent.data import EnsembleData, EnsembleInfo
from lamet_agent.stages.correlator_analysis._fit_ff import CURRENTS, fit_ff_samples, wall_ratios


def _inputs(resample):
    rng = np.random.default_rng(814)
    n = 24
    t, tsep, tau = np.arange(11), [8, 10], np.arange(11)
    ensemble = EnsembleInfo("test", "wall", 0.06, 0.06, 48, 64, 0.67)
    attrs = {
        "resample_id": "same-wall-configurations",
        "bT_fm": 1.08,
        "source_momentum": "[0,0,6]",
        "sink_momentum": "[0,0,-6]",
    }
    denominator = np.exp(-0.5 * t)[None, :] * (1 + 0.02 * rng.normal(size=(n, len(t))))
    data = {}
    for name, fraction in [("gZ5", 0.6), ("gX5", 0.4)]:
        data[name] = EnsembleData(
            ensemble,
            resample,
            list(denominator * fraction),
            ["t"],
            {"t": list(t)},
            attrs={**attrs, "kernel_operator": name, "bT_fm": 0.0},
        )
    common = rng.normal(size=(n, 1, 1))
    for i, name in enumerate(CURRENTS):
        amplitude = -1.3e-8 * (1 + i * 0.1)
        ratio = np.array(
            [
                amplitude
                * (1 - 0.5 * (np.exp(-0.35 * tau) + np.exp(-0.35 * (sep - tau))))
                / (1 + 0.5 * np.exp(-0.35 * sep / 2))
                for sep in tsep
            ]
        )
        spread = 0.06 / (np.sqrt(n - 1) if resample == "jackknife" else 1)
        ratios = ratio[None, :, :] * (1 + spread * (common + rng.normal(size=(n, len(tsep), len(tau)))))
        values = ratios * denominator[:, np.array(tsep) // 2, None] ** 2
        data[name] = EnsembleData(
            ensemble,
            resample,
            list(values),
            ["tsep", "tau"],
            {"tsep": tsep, "tau": list(tau)},
            attrs={**attrs, "kernel_operator": name},
        )
    return data


def _options():
    return dict(
        tsep_values=[8, 10],
        tau_cut=3,
        n_states=2,
        prior_width=1.0,
        energy_gev=2.67,
        svdcut=1e-12,
        posterior_prior_error_scale=10.0,
        sample_prior_mode="original",
    )


def _original_model(x, p):
    t, tau = x
    return (
        p["ff"]
        * (1 + p["c1"] * (np.exp(-p["dE"] * tau) + np.exp(-p["dE"] * (t - tau))))
        / (1 + p["c2"] * np.exp(-p["dE"] * t / 2))
    )


@pytest.mark.parametrize("resample", ["bootstrap", "jackknife"])
def test_original_priors_match_independent_fits_with_full_covariance(resample):
    data = _inputs(resample)
    _, t, tau, ratios = wall_ratios(data, [8, 10], 3)
    expected = []
    for current in CURRENTS:
        values = ratios[current]
        # Retain correlations between all insertions and tseps using the
        # framework's bootstrap/standard jackknife covariance convention.
        covariance = np.cov(values, rowvar=False)
        if resample == "jackknife":
            covariance *= (len(values) - 1) ** 2 / len(values)
        samples = []
        for sample in values:
            observations = gv.gvar(sample, covariance)
            prior = gv.BufferDict(
                {
                    "E0": gv.gvar(1, 10),
                    "log(dE)": gv.gvar(-1, 0.5),
                    "ff": gv.gvar(-3e-6, 1e-2),
                    "A0": gv.gvar(1, 10),
                    "c1": gv.gvar(-0.5, 0.5),
                    "c2": gv.gvar(0.5, 0.5),
                }
            )
            fit = lsqfit.nonlinear_fit(
                data=((t, tau), observations),
                prior=prior,
                fcn=_original_model,
                maxit=10000,
            )
            samples.append(float(gv.mean(fit.p["ff"])))
        expected.append(samples)
    output, diagnostic = fit_ff_samples(data, **_options())
    np.testing.assert_allclose(
        np.asarray(output.values)[:, 0],
        np.mean(expected, axis=0) * diagnostic["normalization"],
        rtol=2e-5,
    )
    assert output.n_sample == 24
    assert output.attrs["resample_id"] == "same-wall-configurations"
    assert output.attrs["sample_prior_mode"] == "original"
    assert diagnostic["sample_prior"] == "initial"
    assert diagnostic["fit_covariance"] == "full"
    assert diagnostic["current_covariance"] == "paired_resamples"
    assert all(record["ff_prior_mean"] == pytest.approx(-3e-6) for record in diagnostic["fits"])
    assert all(record["ff_prior_sdev"] == pytest.approx(1e-2) for record in diagnostic["fits"])


def test_reference_priors_do_not_depend_on_center_posterior_width():
    data = _inputs("bootstrap")
    reference, _ = fit_ff_samples(data, **_options())
    options = {**_options(), "posterior_prior_error_scale": 100.0}
    other, _ = fit_ff_samples(data, **options)
    np.testing.assert_array_equal(reference.values, other.values)
    legacy, diagnostic = fit_ff_samples(data, **{**_options(), "sample_prior_mode": "center_posterior"})
    assert diagnostic["sample_prior"] == "broadened_center_posterior"
    assert diagnostic["fit_covariance"] == "full"
    assert not np.allclose(reference.values, legacy.values, rtol=1e-4)


def test_rejects_unknown_wall_ff_prior_mode():
    with pytest.raises(ValueError, match="sample_prior_mode"):
        fit_ff_samples(_inputs("bootstrap"), **{**_options(), "sample_prior_mode": "unknown"})
