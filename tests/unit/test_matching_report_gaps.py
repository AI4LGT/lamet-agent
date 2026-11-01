from pathlib import Path

import numpy as np
import pytest

from lamet_agent.data import EnsembleData
from lamet_agent.stages._reporting import StageReportRecord, grid_segments, stage_overlay_lines
from lamet_agent.stages.perturbative_matching.reporting import _diagnostics, write_stage_report


def data(x, values):
    return EnsembleData(None, "bootstrap", [values, values], ["x"], {"x": x}, attrs={
        "source_component": "both", "output_component": "both",
        "sample_error_mode": "variance", "momentum_gev": 2.,
        "kernel_id": "quark_pdf_gi_gt_ratio_nlo_rgr", "renormalization_scheme": "ratio",
    })


def record(tmp_path, x, *, complex_values=False):
    # Unequal nodes require interpolation at the common integration boundaries.
    y = np.linspace(-1, 1, 8)
    factor = 1 + 2j if complex_values else 1
    quasi = data(y, factor * (2 + y))
    matched = data(x, factor * (2 + np.asarray(x)))
    return StageReportRecord("match", {"mu": 2., "order": "nlo", "kernel_parameters": {}},
                             {"quasi": quasi}, matched, {"artifacts": []}, tmp_path)


@pytest.mark.parametrize("complex_values", [False, True])
def test_partial_integrals_exclude_gap_and_share_exact_boundaries(tmp_path, complex_values):
    x = [-.9, -.8, -.7, .3, .4, .5]
    result = _diagnostics(record(tmp_path, x, complex_values=complex_values))
    # Integral of 2+x over [-.9,-.7] U [.3,.5], without the missing interval.
    expected = .72 * (1 + 2j if complex_values else 1)
    np.testing.assert_allclose(result[:2], [expected, expected], atol=1e-14)
    assert result[2] < 1e-14


def test_contiguous_grid_keeps_full_interval_and_isolated_points_add_no_area(tmp_path):
    np.testing.assert_allclose(_diagnostics(record(tmp_path, np.linspace(-.8, .8, 9))), [3.2, 3.2, 0], atol=1e-14)
    x = [-.9, -.8, -.7, .3]
    np.testing.assert_allclose(_diagnostics(record(tmp_path, x)), [.24, .24, 0], atol=1e-14)


def test_segments_do_not_impose_a_fixed_small_x_cutoff():
    assert len(grid_segments(np.linspace(-1, 1, 21))) == 1
    segments = grid_segments([-.9, -.8, -.7, .3, .4, .5])
    assert [part.tolist() for part in segments] == [[0, 1, 2], [3, 4, 5]]


def test_matching_overlay_paths_do_not_cross_gap(tmp_path, monkeypatch):
    from lamet_agent import plotting

    captured = []
    original = plotting.save_figure

    def inspect_and_save(*paths):
        axis = plotting._axis()
        for line in axis.lines:
            x = line.get_xdata()
            assert np.max(x) <= -.7 or np.min(x) >= .3
        for collection in axis.collections:
            for path in collection.get_paths():
                x = path.vertices[:, 0]
                assert np.max(x) <= -.7 or np.min(x) >= .3
        captured.append(len(axis.get_legend_handles_labels()[1]))
        original(*paths)

    monkeypatch.setattr(plotting, "save_figure", inspect_and_save)
    stage_overlay_lines((record(tmp_path, [-.9, -.8, -.7, .3, .4, .5]),), tmp_path,
                        coordinate="x", stem="gap", ylabel="PDF", band=True, split_gaps=True)
    assert captured == [1]
    assert (tmp_path / "plots/gap.pdf").exists()


def test_matching_report_labels_partial_integrals(tmp_path: Path):
    path = write_stage_report(records=(record(tmp_path, [-.9, -.8, -.7, .3, .4, .5]),),
                              artifact_directory=tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "quasi partial integral" in text
    assert "excluded from both quasi and matched integrals" in text
    assert "bridges that" not in text
