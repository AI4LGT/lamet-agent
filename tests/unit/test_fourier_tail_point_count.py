"""Tail windows must constrain more measured observations than parameters."""

import numpy as np
import pytest

from lamet_agent.data import EnsembleData
from lamet_agent.stages.fourier_transform import physics


def _data(n_positive):
    z = np.arange(-n_positive, n_positive + 1) / 10
    rng = np.random.default_rng(14)
    center = np.exp(-np.abs(z)) * np.exp(0.2j * np.sign(z))
    samples = center + 0.01 * (
        rng.normal(size=(20, z.size)) + 1j * rng.normal(size=(20, z.size))
    )
    return EnsembleData(
        None, "bootstrap", list(samples), ["z"], {"z": z.tolist()},
        attrs={
            "coord_unit": "fm", "momentum_gev": 2.0, "hadron": "proton",
            "sample_error_mode": "variance", "symmetry": '{"real":"even","imag":"odd"}',
        },
    )


@pytest.mark.parametrize("mode", ["center", "resamples"])
@pytest.mark.parametrize("component", ["re", "im", "both"])
@pytest.mark.parametrize("order,n_parameters", [("LA", 4), ("NLA", 6)])
def test_exactly_determined_tail_is_rejected_before_fitting(monkeypatch, mode, component, order, n_parameters):
    n_positive = n_parameters // (2 if component == "both" else 1)

    def unexpected_fit(*args, **kwargs):
        pytest.fail("an exactly determined tail reached the numerical fitter")

    monkeypatch.setattr(physics, "nonlinear_fit", unexpected_fit)
    means, widths = physics._scan_tail_priors(model_id="cg_nla", order=order, lambda0_gev=0, hadron="proton")
    with pytest.raises(ValueError, match="require N_observations > N_parameters"):
        physics.fit_tail_parameters(
            _data(n_positive), model_id="cg_nla", order=order, hadron="proton",
            z_min_fm=0.0, z_max_fm=n_positive / 10,
            source_component=component, prior_means=means, prior_widths=widths, mode=mode,
        )


@pytest.mark.parametrize("component,n_positive", [("re", 4), ("im", 4), ("both", 2)])
def test_smallest_strict_surplus_reaches_fitter(monkeypatch, component, n_positive):
    # GI LA proton has three parameters; two complex points already suffice.
    class ReachedFitter(Exception):
        pass

    def stop_at_fit(inputs, *args, **kwargs):
        fit_x, fit_data = inputs
        assert len(fit_x["parameter_names"]) == 3
        assert fit_data.values.shape[1] == 4
        assert np.all(fit_x["z"] > 0)
        raise ReachedFitter

    monkeypatch.setattr(physics, "nonlinear_fit", stop_at_fit)
    means, widths = physics._scan_tail_priors(model_id="gi_nla", order="LA", lambda0_gev=0, hadron="proton")
    with pytest.raises(ReachedFitter):
        physics.fit_tail_parameters(
            _data(n_positive), model_id="gi_nla", order="LA", hadron="proton",
            z_min_fm=0.0, z_max_fm=n_positive / 10, source_component=component,
            prior_means=means, prior_widths=widths, mode="center",
        )


@pytest.mark.parametrize("zmax,orders", [(1.0, ["LA"]), (1.1, ["LA", "NLA"])])
def test_scan_checks_strict_surplus_again_for_each_order(zmax, orders):
    result = physics.scan_fourier_transform(
        _data(12), [-0.5, 0.0, 0.5],
        transform={"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        tail={
            "models": ["cg_nla"], "z_min_fm": [0.5], "z_max_fm": [0.8, zmax],
            "extent_fm": 1.5, "smoothing_method": "linear",
        },
        scan={
            "orders": ["LA", "NLA"], "sector": "full", "lambda0_gev": 0.0,
            "prior_widths": [1.0], "model_average": False, "max_schemes": 10,
            "source_component": "re", "output_scale": 1.0, "q_min": 0.0,
        },
        workers=1,
    )
    # The four-point LA window is excluded. Six points support LA but not NLA.
    assert len(result["range_candidates"]) == 1
    assert result["selected_range"]["z_max_fm"] == zmax
    assert [candidate["order"] for candidate in result["model_candidates"]] == orders
    for candidate in [*result["range_candidates"], *result["model_candidates"]]:
        assert candidate["n_observations"] > candidate["n_parameters"]
        assert candidate["data_dof"] == candidate["n_observations"] - candidate["n_parameters"]
