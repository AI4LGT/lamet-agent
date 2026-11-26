from types import SimpleNamespace

import numpy as np
import pytest

from lamet_agent.stages.fourier_transform._inspection import effective_zmin_fm


@pytest.mark.parametrize(
    "target, lower, offset, expected",
    [
        ("gpd", [0.5, 0.6], 0, [0.5, 0.6]),
        ("gpd", [0.4], 1, [0.5]),
        ("gpd", [0.6], -1, [0.5]),
        ("pdf", [0.3], 0, [0.3]),
        ("da", [0.3], 0, [0.3]),
        ("tmdwf", [0.3], 0, [0.3]),
    ],
)
def test_effective_tail_minimum(target, lower, offset, expected):
    context = SimpleNamespace(
        params={"zmin_fm": lower, "tail_window_step_offset": offset},
        manifest={"metadata": {"target_observable": target}},
    )
    assert effective_zmin_fm(context, z_grid_step_fm=0.1) == expected


@pytest.mark.parametrize("lower, offset", [([0.4], 0), ([0.5], -1), ([0.6, 0.4], 0)])
def test_gpd_rejects_short_distance_tail_candidates(lower, offset):
    context = SimpleNamespace(
        params={"zmin_fm": lower, "tail_window_step_offset": offset},
        manifest={"metadata": {"target_observable": "gpd"}},
    )
    with pytest.raises(ValueError, match="effective zmin_fm >= 0.5 fm"):
        effective_zmin_fm(context, z_grid_step_fm=0.1)


def test_explicit_pion_gpd_minimum_is_applied_after_the_grid_offset():
    context = SimpleNamespace(
        params={"zmin_fm": [0.12], "tail_window_step_offset": -1, "gpd_min_zmin_fm": 0.08},
        manifest={"metadata": {"target_observable": "gpd"}},
    )
    assert effective_zmin_fm(context, z_grid_step_fm=0.04) == [0.08]
    context.params["tail_window_step_offset"] = -2
    with pytest.raises(ValueError, match="0.08 fm"):
        effective_zmin_fm(context, z_grid_step_fm=0.04)


def test_zero_skewness_pion_valence_tail_with_five_measured_points():
    from lamet_agent.data import EnsembleData
    from lamet_agent.stages.fourier_transform.physics import scan_fourier_transform

    z = np.round(0.04 * np.arange(-10, 11), 12)
    rng = np.random.default_rng(1984)
    center = np.exp(-2 * np.abs(z)) * (1 + 0.3 * np.cos(9 * z))
    samples = center + rng.normal(0, 0.01, (16, len(z)))
    data = EnsembleData(None, "bootstrap", list(samples), ["z"], {"z": z.tolist()}, attrs={
        "coord_unit": "fm", "momentum_gev": 1.94, "hadron": "pion", "xi": 0.0,
        "t_gev2": -1.69, "polarization": "unpolarized", "sample_error_mode": "covariance",
        "symmetry": '{"real":"explicit","imag":"explicit"}', "gpd_completion_mode": "measured_signed_z",
    })
    result = scan_fourier_transform(
        data, [-0.5, 0, 0.5], observable="GPD",
        transform={"phase_sign": 1, "x_shift": 0.0, "prefactor": "pz_over_2pi"},
        tail={"models": ["gi_nla"], "z_min_fm": [0.08], "z_max_fm": [0.24],
              "extent_fm": 1.4, "smoothing_method": "linear"},
        scan={"orders": ["LA", "NLA"], "sector": "valence", "lambda0_gev": 0.4,
              "prior_widths": [3.0], "model_average": True, "q_min": 0, "max_schemes": 1,
              "source_component": "re", "output_scale": 1.0},
    )
    assert result["data"].attrs["tail_family"] == "pion_gpd_zero_skewness_valence"
    assert np.isfinite(result["data"].values).all()
    assert [c["order"] for c in result["model_candidates"]] == ["LA"]
    assert result["weights"] == [1.0]
