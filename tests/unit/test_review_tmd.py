"""Review audits the complete TMD DAG while numerical evidence stays bounded."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from lamet_agent.stages.review._check_consistency import run


def _context(tmp_path):
    jobs, summaries = {}, {}
    lattice = {"series": "HISQ", "a_s": 0.06, "a_t": 0.06, "L_s": 48, "L_t": 64}
    wf, rest, ff = {"wf": "wf-plan"}, {"rest": "rest-plan"}, {"ff": "ff-plan"}
    normalized, combined = {**wf, **rest}, {**wf, **rest, **ff}

    def add(name, stage, inputs, observable, operator, mass, p, lineage, unit, scheme, dims, coords):
        jobs[name] = {"stage_id": stage, "inputs": inputs, "params": {"observable": observable}}
        attrs = {
            "observable": observable,
            "kernel_operator": operator,
            "hadron": "pion",
            "parton": "quark",
            "gfix": "CG",
            "renormalization_scheme": scheme,
            "target_observable": "tmdwf" if stage in {"correlator_analysis", "renormalization"} else "tmd",
            "momentum_gev": p,
            "coord_unit": unit,
            "resample_lineage": json.dumps(lineage),
            "replica_frame_id": "frame",
            "resample_id": "-".join(sorted(lineage)),
            "units": json.dumps({"bT": "fm", "values": "dimensionless"}),
        }
        summaries[name] = {
            "job_id": name,
            "stage_id": stage,
            "dims": dims,
            "coords": coords,
            "ensemble": {**lattice, "id": f"m{mass}", "m_pi": mass},
            "attrs": attrs,
            "resample": "bootstrap",
            "n_sample": 20,
            "replica_order_digest": "order",
        }

    z = {"z": [0.0, 0.06]}
    x = {"x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}
    b = {"bT": [0.24]}
    add("cw", "correlator_analysis", {}, "tmdwf", "g14", 0.67, 3.44, wf, "lattice", "bare", ["z"], z)
    add("cl", "correlator_analysis", {}, "tmdwf", "g7", 0.67, 0.0, rest, "lattice", "bare", ["z"], z)
    add("cp", "correlator_analysis", {}, "tmdpdf", "g8", 0.3, 2.43, wf, "lattice", "bare", ["z"], z)
    add("cf", "correlator_analysis", {}, "form_factor", "two_current_X_X", 0.67, 2.58, ff, "lattice", "bare", ["bT"], b)
    add(
        "reference",
        "renormalization",
        {"target": "cl", "local": "cl"},
        "tmdwf",
        "g7",
        0.67,
        0.0,
        rest,
        "fm",
        "ratio",
        [],
        {},
    )
    for name, source, obs, op, mass, p in (
        ("rw", "cw", "tmdwf", "g14", 0.67, 3.44),
        ("rp", "cp", "tmdpdf", "g8", 0.3, 2.43),
    ):
        add(
            name,
            "renormalization",
            {"target": source, "denominator": "reference"},
            obs,
            op,
            mass,
            p,
            normalized,
            "fm",
            "ratio",
            ["z"],
            z,
        )
    for name, source, obs, op, mass, p in (
        ("fw", "rw", "tmdwf", "g14", 0.67, 3.44),
        ("fp", "rp", "tmdpdf", "g8", 0.3, 2.43),
    ):
        add(
            name,
            "fourier_transform",
            {"input": source},
            obs,
            op,
            mass,
            p,
            normalized,
            "dimensionless",
            "ratio",
            ["x"],
            x,
        )
    add("cs", "soft_analysis", {"quasi": ["fw"]}, "cs_kernel", "cs", 0.67, None, normalized, None, "ratio", ["bT"], b)
    add(
        "si",
        "soft_analysis",
        {"quasi": ["fw"], "form_factor": ["cf"], "cs_kernel": "cs"},
        "intrinsic_soft",
        "soft",
        0.67,
        None,
        combined,
        None,
        "ratio",
        ["bT"],
        b,
    )
    add(
        "mp",
        "matching",
        {"quasi": "fp", "cs_kernel": "cs", "intrinsic_soft": "si"},
        "tmdpdf",
        "g8",
        0.3,
        2.43,
        combined,
        "dimensionless",
        "ratio",
        ["x"],
        {"x": [0.2, 0.4, 0.6, 0.8]},
    )
    return SimpleNamespace(
        state={
            "result_summary": [copy.deepcopy(summaries["mp"])],
            "review_bundle": {"jobs": jobs, "provenance_summaries": summaries},
        },
        params={"checks": ["identity", "units", "kinematics", "schemes", "resampling", "grids"]},
        artifact_directory=tmp_path,
    )


def _findings(context):
    run(context)
    return context.state["consistency"]["findings"]


def test_tmd_review_accepts_operator_joins_valence_masses_and_merged_lineage(tmp_path):
    context = _context(tmp_path)
    assert _findings(context) == []
    assert len(context.state["result_summary"]) == 1


@pytest.mark.parametrize("field,value", [("gfix", "GI"), ("parton", "gluon"), ("kernel_operator", "wrong")])
def test_tmd_review_audits_unselected_upstream_distribution_identity(tmp_path, field, value):
    context = _context(tmp_path)
    context.state["review_bundle"]["provenance_summaries"]["rw"]["attrs"][field] = value
    findings = _findings(context)
    assert any(
        f["status"] == "error" and f["field"] == field and f["source_job"] == "cw" and f["consumer_job"] == "rw"
        for f in findings
    )


@pytest.mark.parametrize("field,value", [("a_s", 0.09), ("a_t", 0.09), ("L_s", 32), ("L_t", 96)])
def test_tmd_review_rejects_changed_lattice_on_universal_soft_edges(tmp_path, field, value):
    context = _context(tmp_path)
    context.state["review_bundle"]["provenance_summaries"]["si"]["ensemble"][field] = value
    assert any(
        f["status"] == "error" and f["field"] == "ensemble" and f["source_job"] == "si" and f["consumer_job"] == "mp"
        for f in _findings(context)
    )


@pytest.mark.parametrize("lineage", [{"wf": "different", "rest": "rest-plan"}, {"rest": "rest-plan"}])
def test_tmd_review_detects_conflicting_or_dropped_upstream_sampling_group(tmp_path, lineage):
    context = _context(tmp_path)
    context.state["review_bundle"]["provenance_summaries"]["rw"]["attrs"]["resample_lineage"] = json.dumps(lineage)
    assert any(
        f["status"] == "error"
        and f["field"] == "resample_lineage"
        and f["source_job"] == "cw"
        and f["consumer_job"] == "rw"
        for f in _findings(context)
    )


@pytest.mark.parametrize("field", ["replica_frame_id", "replica_order_digest"])
def test_tmd_review_detects_unselected_upstream_replica_misalignment(tmp_path, field):
    context = _context(tmp_path)
    data = context.state["review_bundle"]["provenance_summaries"]["rw"]
    (data if field == "replica_order_digest" else data["attrs"])[field] = "different"
    assert any(
        f["status"] == "error" and f["field"] == field and f["source_job"] == "cw" and f["consumer_job"] == "rw"
        for f in _findings(context)
    )


def test_tmd_review_checks_matching_quasi_momentum_and_invalid_output_grid(tmp_path):
    context = _context(tmp_path)
    selected = context.state["result_summary"][0]
    selected["attrs"]["momentum_gev"] = 9.0
    selected["coords"]["x"] = [0.2, 0.35, 0.6]
    findings = _findings(context)
    for field in ("momentum_gev", "dims/coords"):
        assert any(
            f["field"] == field
            and f["status"] in {"error", "warning"}
            and f["source_job"] == "fp"
            and f["consumer_job"] == "mp"
            for f in findings
        )


@pytest.mark.parametrize("field,value", [("gfix", "GI"), ("parton", "gluon")])
def test_tmd_auxiliary_edges_still_enforce_gauge_and_parton(tmp_path, field, value):
    context = _context(tmp_path)
    context.state["review_bundle"]["provenance_summaries"]["si"]["attrs"][field] = value
    assert any(
        f["status"] == "error" and f["field"] == field and f["source_job"] == "si" and f["consumer_job"] == "mp"
        for f in _findings(context)
    )


def _jackknife_context(tmp_path):
    context = _context(tmp_path)
    counts = {"wf": 11, "rest": 5, "ff": 7}
    records = list(context.state["review_bundle"]["provenance_summaries"].values())
    records += context.state["result_summary"]
    for record in records:
        groups = {name: counts[name] for name in json.loads(record["attrs"]["resample_lineage"])}
        record["resample"] = "jackknife"
        record["attrs"]["jackknife_groups"] = json.dumps(groups, sort_keys=True)
        record["n_sample"] = sum(groups.values()) + (len(groups) > 1)
        if len(groups) > 1:
            record["attrs"]["jackknife_layout"] = json.dumps(groups, sort_keys=True)
        record["replica_order_digest"] = hashlib.sha256(
            json.dumps(list(range(record["n_sample"]))).encode()
        ).hexdigest()
    return context


def test_tmd_review_accepts_independent_jackknife_block_expansion(tmp_path):
    assert _findings(_jackknife_context(tmp_path)) == []


@pytest.mark.parametrize("corruption", ["count", "layout", "order", "group_count"])
def test_tmd_review_rejects_corrupt_jackknife_block_metadata(tmp_path, corruption):
    context = _jackknife_context(tmp_path)
    record = context.state["review_bundle"]["provenance_summaries"]["si"]
    if corruption == "count":
        record["n_sample"] -= 1
    elif corruption == "layout":
        record["attrs"]["jackknife_layout"] = json.dumps({"wf": 11, "ff": 12})
    elif corruption == "order":
        record["replica_order_digest"] = "permuted"
    else:
        groups = {"wf": 12, "rest": 5, "ff": 6}
        record["attrs"]["jackknife_groups"] = json.dumps(groups)
        record["attrs"]["jackknife_layout"] = json.dumps(groups)
    assert any(
        finding["status"] == "error" and finding["field"] == "jackknife_layout"
        for finding in _findings(context)
    )
