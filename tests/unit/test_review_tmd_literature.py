"""TMD review finds the workflow paper under the catalog's kinematic tags."""

from types import SimpleNamespace

from lamet_agent.stages.review._list_literature import run


def _context(target):
    return SimpleNamespace(
        manifest={
            "metadata": {"target_observable": target, "parton": "quark"},
            "stages": {name: {"jobs": []} for name in ("soft_analysis", "matching", "review")},
        },
        params={"catalog": "builtin", "max_papers": 3},
        state={"consistency": {}, "result_summary": [{"attrs": {"hadron": "pion", "gfix": "CG"}}]},
    )


def test_tmd_review_selects_paper_from_builtin_catalog():
    context = _context("tmd")
    run(context)
    candidates = context.state["literature_candidates"]
    paper = next(paper for paper in candidates if paper["id"] == "2504.04625")
    assert "observables" in paper["exact_metadata_matches"]


def test_collinear_pdf_review_still_excludes_tmd_lattice_paper():
    context = _context("pdf")
    run(context)
    assert "2504.04625" not in {paper["id"] for paper in context.state["literature_candidates"]}
