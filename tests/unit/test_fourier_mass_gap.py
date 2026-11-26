import gvar as gv
import numpy as np
import pytest

from lamet_agent.data import EnsembleData
from lamet_agent.stages.fourier_transform import physics


def _data(gap):
    z = np.arange(5, 21) / 10
    # A negative synthetic gap deliberately pushes the fit against its boundary.
    center = np.exp(-gap * z / physics.HBAR_C_GEV_FM) * np.exp(0.2j)
    rng = np.random.default_rng(82)
    samples = center + 0.001 * (rng.normal(size=(12, z.size)) + 1j * rng.normal(size=(12, z.size)))
    return EnsembleData(
        None, "bootstrap", list(samples), ["z"], {"z": z.tolist()},
        attrs={"momentum_gev": 2.0, "sample_error_mode": "variance"},
    )


@pytest.mark.parametrize("offset", [0.0, 0.1, 0.8])
@pytest.mark.parametrize("mode", ["center", "resamples"])
def test_physical_gaussian_prior_and_strict_bound_survive_refinement(monkeypatch, offset, mode):
    means, widths = physics._scan_tail_priors(
        model_id="gi_nla", order="LA", lambda0_gev=offset, hadron="proton"
    )
    assert means["Lambda"] + offset == pytest.approx(0.5)
    assert widths["Lambda"] == 0.3
    original_fit = physics.nonlinear_fit
    priors = []

    def capture(inputs, fcn, prior, **kwargs):
        priors.append((gv.mean(prior["u2"]), gv.sdev(prior["u2"])))
        assert kwargs["bounds"][0]["u2"] > 0.0
        # The optimizer coordinate is the physical gap, even at nonzero offset.
        params = physics._physical_tail_parameters(
            {"u0": 0.0, "u1": 0.0, "u2": 0.5}, ["A2", "phi2", "Lambda"],
            physics._tail_bounds(["A2", "phi2", "Lambda"], offset), offset,
        )
        assert params["Lambda"] + offset == pytest.approx(0.5)
        return original_fit(inputs, fcn, prior, **kwargs)

    monkeypatch.setattr(physics, "nonlinear_fit", capture)
    records, diagnostics = physics.fit_tail_parameters(
        _data(-0.03), model_id="gi_nla", order="LA", hadron="proton",
        z_min_fm=0.5, z_max_fm=2.0, prior_means=means, prior_widths=widths,
        lambda0_gev=offset, mode=mode, posterior_prior_scale=10.0,
    )
    assert priors == [(0.5, 0.3), (0.5, 0.3)]
    assert all(0.0 < record["Lambda"] < 0.001 for record in records)
    assert diagnostics["mass_gap_prior_gev"] == {"mean": 0.5, "sdev": 0.3}
    assert diagnostics["mass_gap_lower_bound_gev"] == 0.0


@pytest.mark.parametrize("mean", [-0.01, 0.0])
def test_mass_gap_prior_mean_at_or_below_bound_is_rejected(mean):
    means, widths = physics._scan_tail_priors(
        model_id="gi_nla", order="LA", lambda0_gev=0.0, hadron="proton"
    )
    with pytest.raises(ValueError, match="mass-gap prior mean"):
        physics.fit_tail_parameters(
            _data(0.5), model_id="gi_nla", order="LA", hadron="proton",
            z_min_fm=0.5, z_max_fm=2.0, prior_means=means, prior_widths=widths,
            priors={"Lambda": f"{mean}(0.3)"}, mode="center",
        )
