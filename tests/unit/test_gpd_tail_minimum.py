from types import SimpleNamespace

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
    data = SimpleNamespace(ensemble=SimpleNamespace(a_s=0.1))
    assert effective_zmin_fm(context, data) == expected


@pytest.mark.parametrize("lower, offset", [([0.4], 0), ([0.5], -1), ([0.6, 0.4], 0)])
def test_gpd_rejects_short_distance_tail_candidates(lower, offset):
    context = SimpleNamespace(
        params={"zmin_fm": lower, "tail_window_step_offset": offset},
        manifest={"metadata": {"target_observable": "gpd"}},
    )
    data = SimpleNamespace(ensemble=SimpleNamespace(a_s=0.1))
    with pytest.raises(ValueError, match="effective zmin_fm >= 0.5 fm"):
        effective_zmin_fm(context, data)
