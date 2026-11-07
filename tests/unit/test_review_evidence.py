"""Focused tests for bounded and on-demand Review numerical evidence."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lamet_agent.agent import ToolContext, _discover_tools, _review_summary
from lamet_agent.data import EnsembleData
from lamet_agent.stages.review.tools.read_full_resolution import run as read_full_resolution


def _distribution(points: int = 600) -> EnsembleData:
    x = np.linspace(0.0, 1.0, points)
    values = np.exp(1j * x)
    return EnsembleData(
        None,
        "bootstrap",
        [values, values * (1.0 + 0.01j)],
        ["x"],
        {"x": x.tolist()},
        attrs={"sample_error_mode": "covariance"},
        name="distribution",
    )


def test_tmd_review_retains_bad_fit_counts_and_nongaussian_soft_evidence(tmp_path: Path) -> None:
    from lamet_agent.stages.soft_analysis.workflow import replica_distribution

    data = EnsembleData(
        None, "bootstrap", [np.array([v]) for v in [1., 1., 1., 1., 20.]],
        ["bT"], {"bT": [1.08]}, attrs={"target_observable": "tmd", "observable": "intrinsic_soft"},
    )
    distribution = replica_distribution(data)
    assert distribution[0]["median"] == 1
    assert distribution[0]["mean"] == 4.8
    assert distribution[0]["q16"] == 1
    assert distribution[0]["q84"] < distribution[0]["mean"] + distribution[0]["sdev"]
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}}, tmp_path / "manifest.json",
        "soft_analysis", "intrinsic_soft", {}, {}, {}, {}, tmp_path, np.random.default_rng(1), output=data,
    )
    summary = {"diagnostics": {"replica_distribution": distribution}}
    evidence = _review_summary(context, summary)
    assert evidence["tmd_diagnostics"]["replica_distribution"] == distribution
    summary["diagnostics"] = {
        "selected_application_fit": {"fits": [{"Q": .01}, {"Q": .7}], "n_failed_samples": 0}
    }
    evidence = _review_summary(context, summary)["tmd_diagnostics"]
    assert evidence == {"center_fit_count": 2, "Q_below_0_05": 1, "minimum_Q": .01, "n_failed_samples": 0}


def test_review_summary_sends_sixty_plot_points_and_selected_fit_quality(tmp_path: Path) -> None:
    context = ToolContext(
        {"metadata": {"sample_error_mode": "covariance"}},
        tmp_path / "manifest.json",
        "correlator_analysis",
        "ca",
        {},
        {},
        {},
        {},
        tmp_path,
        np.random.default_rng(1),
        output=_distribution(),
    )
    summary = {
        "stage_id": "correlator_analysis",
        "job_id": "ca",
        "result": "matrix_element",
        "decisions": {"candidate_id": "candidate"},
        "diagnostics": {
            "candidates": [
                {
                    "candidate_id": "candidate",
                    "Q": 0.8,
                    "chi2": 9.0,
                    "dof": 10,
                    "chi2_dof": 0.9,
                    "logGBF": 3.5,
                }
            ]
        },
        "artifacts": [],
    }

    evidence = _review_summary(context, summary)

    assert len(evidence["output"]["coords"]["x"]) == 60
    assert evidence["output"]["coords"]["x"][0] == 0.0
    assert evidence["output"]["coords"]["x"][-1] == 1.0
    assert evidence["output"]["sampling"] == {
        "dimension": "x",
        "method": "uniform_plot_indices",
        "original_points": 600,
        "sent_points": 60,
        "full_resolution_available": True,
    }
    assert evidence["fit_quality"] == {
        "status": "available",
        "Q": 0.8,
        "chi2": 9.0,
        "dof": 10,
        "chi2_dof": 0.9,
        "logGBF": 3.5,
    }


def test_review_full_resolution_tool_returns_one_selected_job(tmp_path: Path) -> None:
    data = _distribution()
    manifest = {
        "metadata": {"sample_error_mode": "covariance"},
        "stages": {"review": {"jobs": [{"id": "review", "inputs": {"results": ["matched"]}}]}},
    }
    context = ToolContext(
        manifest,
        tmp_path / "manifest.json",
        "review",
        "review",
        {},
        {"results": [data]},
        {},
        {"review_bundle": {"jobs": {"matched": {"stage_id": "matching"}}}},
        tmp_path,
        np.random.default_rng(1),
    )

    observation = read_full_resolution(context, job_id="matched")

    assert observation["job_id"] == "matched"
    assert len(observation["output"]["coords"]["x"]) == 600
    assert observation["output"]["sampling"]["method"] == "full_resolution"
    assert context.state["full_resolution_reads"] == ["matched"]
    assert "read_full_resolution" in {tool.name for tool in _discover_tools("review")}
