from types import SimpleNamespace

import pytest

from lamet_agent.stages.fourier_transform._inspection import effective_zmin_fm


@pytest.mark.parametrize("target", ["pdf", "da", "gpd", "tmd", "tmdwf", "tmdpdf"])
@pytest.mark.parametrize(
    "lower, offset, expected",
    [
        ([0.5, 0.6], 0, [0.5, 0.6]),
        ([0.4], 1, [0.5]),
        ([0.6], -1, [0.5]),
    ],
)
def test_effective_tail_minimum(target, lower, offset, expected):
    context = SimpleNamespace(
        params={"zmin_fm": lower, "tail_window_step_offset": offset},
        manifest={"metadata": {"target_observable": target}},
    )
    assert effective_zmin_fm(context, z_grid_step_fm=0.1) == expected


@pytest.mark.parametrize("target", ["pdf", "da", "gpd", "tmd", "tmdwf", "tmdpdf"])
@pytest.mark.parametrize("lower, offset", [([0.4], 0), ([0.5], -1), ([0.6, 0.4], 0), ([-0.5], 0)])
def test_rejects_short_distance_tail_candidates(target, lower, offset):
    context = SimpleNamespace(
        params={"zmin_fm": lower, "tail_window_step_offset": offset},
        manifest={"metadata": {"target_observable": target}},
    )
    with pytest.raises(ValueError, match="effective zmin_fm >= 0.5 fm"):
        effective_zmin_fm(context, z_grid_step_fm=0.1)


@pytest.mark.parametrize("target", ["pdf", "da", "gpd", "tmd"])
def test_recommendation_fails_before_llm_when_no_shifted_grid_point_is_eligible(target):
    from lamet_agent.stages.fourier_transform.ask.ask_for_tail_ranges import recommend

    context = SimpleNamespace(
        params={"tail_window_step_offset": -1, "scheme_scan": {}, "zmax_ext_fm": 1.0},
        manifest={"metadata": {"target_observable": target}},
        state={
            "fourier_input": SimpleNamespace(coords={"z": [-0.5, 0.0, 0.5]}),
            "tail_inspection": {"z_grid_step_fm": 0.5},
        },
    )
    # No session is needed: an impossible range must fail before consulting the LLM.
    with pytest.raises(ValueError, match="No measured Fourier zmin_fm"):
        recommend(context, None, requested_fields={"zmin_fm"})
