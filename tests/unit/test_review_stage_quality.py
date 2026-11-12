"""The model receives all-job fit quality independently of selected curves."""

import json
from types import SimpleNamespace

from lamet_agent.stages.review._inspect_results import _fit_quality_aggregate, run


def test_review_reports_include_unselected_bad_application_fits(tmp_path):
    directory = tmp_path / "artifacts" / "02_review" / "review"
    directory.mkdir(parents=True)
    stage = directory.parent.parent / "01_correlator_analysis"
    stage.mkdir()
    (stage / "report.md").write_text("A full report is retained as an artifact, not injected into model context.")
    jobs = ["selected", "unselected_bad", "unselected_generic", "missing"]
    summaries = {
        "selected": {
            "fit_quality": {"Q": 0.8},
            "tmd_diagnostics": {"center_fit_count": 3, "Q_below_0_05": 0, "minimum_Q": 0.3, "n_failed_samples": 0},
        },
        "unselected_bad": {
            "fit_quality": {"Q": 0.9},
            "tmd_diagnostics": {"center_fit_count": 10, "Q_below_0_05": 4, "minimum_Q": 1e-8, "n_failed_samples": 2},
        },
        "unselected_generic": {"fit_quality": {"status": "available", "Q": 0.01}},
    }
    manifest = {
        "metadata": {"run_id": "test"},
        "stages": {
            "correlator_analysis": {"jobs": [{"id": name, "inputs": {}} for name in jobs]},
            "review": {"jobs": [{"id": "review", "inputs": {"results": ["selected"]}}]},
        },
    }
    context = SimpleNamespace(
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        stage_id="review",
        job_id="review",
        inputs={"results": ["one representative curve"]},
        artifact_directory=directory,
        state={},
        runtime_records={name: {"review_summary": summary} for name, summary in summaries.items()},
    )
    observed = run(context)
    assert observed["results"] == [summaries["selected"]]
    report = observed["reports"][0]
    assert "text" not in report
    quality = report["fit_quality"]
    assert quality["job_count"] == 4
    assert quality["jobs_with_review_summary"] == 3
    assert quality["application_fit_job_count"] == 2
    assert quality["selected_fit_only_job_count"] == 1
    assert quality["center_fit_count"] == 14
    assert quality["Q_below_0_05"] == 5
    assert quality["minimum_Q"] == 1e-8
    assert quality["minimum_Q_job_id"] == "unselected_bad"
    assert quality["poor_Q_job_ids"] == ["unselected_bad", "unselected_generic"]
    assert quality["poor_Q_job_count"] == 2
    assert quality["n_failed_samples"] == 2
    assert quality["sample_failure_job_ids"] == ["unselected_bad"]
    assert quality["jobs_missing_failure_count"] == 1
    assert quality["jobs_without_center_fit_Q"] == 1
    stored = json.loads((directory / "review_bundle.json").read_text())
    assert stored["stage_reports"][0]["fit_quality"] == quality


def test_fit_quality_bounds_job_lists_and_preserves_total_counts():
    summaries = {
        f"job_{i:03d}": {"fit_quality": {"Q": 0.01 + i * 1e-5}, "diagnostics": {"n_failed_samples": 1}}
        for i in range(35)
    }
    result = _fit_quality_aggregate(list(summaries), summaries)
    assert result["center_fit_count"] == result["Q_below_0_05"] == 35
    assert result["poor_Q_job_count"] == result["sample_failure_job_count"] == 35
    assert len(result["poor_Q_job_ids"]) == len(result["sample_failure_job_ids"]) == 20
    assert result["poor_Q_job_ids_omitted"] == result["sample_failure_job_ids_omitted"] == 15
    assert result["n_failed_samples"] == 35
    assert result["minimum_Q_job_id"] == "job_000"


def test_absent_fit_errors_are_not_reported_as_zero_and_missing_q_keeps_failures():
    summaries = {
        "generic": {"fit_quality": {"Q": 0.8}},
        "nonfit": {"fit_quality": {"status": "not_applicable"}},
    }
    result = _fit_quality_aggregate(list(summaries), summaries)
    assert result["center_fit_count"] == 1
    assert result["n_failed_samples"] is None
    assert result["jobs_missing_failure_count"] == 1
    assert result["jobs_without_center_fit_Q"] == 1
    summaries["failed"] = {"fit_quality": {"status": "not_available"}, "diagnostics": {"n_failed_samples": 3}}
    result = _fit_quality_aggregate(list(summaries), summaries)
    assert result["center_fit_count"] == 1
    assert result["jobs_without_center_fit_Q"] == 2
    assert result["n_failed_samples"] == 3
    assert result["sample_failure_job_ids"] == ["failed"]
