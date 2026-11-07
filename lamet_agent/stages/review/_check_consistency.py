"""Audit provenance along the manifest job graph."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from lamet_agent.agent import ToolContext


_TMD_OBSERVABLES = {"tmdwf", "tmdpdf", "cs_kernel", "intrinsic_soft", "form_factor"}


def _tmd_job(job, summary):
    params = job.get("params", {})
    return (
        params.get("strategy") in {"tmd", "tmd_ratio"}
        or params.get("observable") in _TMD_OBSERVABLES
        or summary.get("attrs", {}).get("observable") in _TMD_OBSERVABLES
        or job["stage_id"] == "soft_analysis"
    )


def _lineage(value):
    entries = json.loads(value) if isinstance(value, str) else value
    if (
        not isinstance(entries, dict)
        or not entries
        or any(not isinstance(k, str) or not k or not isinstance(v, str) or not v for k, v in entries.items())
    ):
        raise ValueError("lineage must be a nonempty group-to-plan mapping")
    return entries


def _jackknife_groups(attrs):
    """Validate native or block jackknife provenance without loading samples."""
    groups = attrs.get("jackknife_groups")
    groups = json.loads(groups) if isinstance(groups, str) else groups
    if not isinstance(groups, dict) or not groups or any(
        not isinstance(key, str) or not key or isinstance(count, bool)
        or not isinstance(count, int) or count < 2 for key, count in groups.items()
    ):
        raise ValueError("jackknife_groups must give each group a count of at least two")
    if set(groups) != set(_lineage(attrs.get("resample_lineage"))):
        raise ValueError("jackknife groups and sampling lineage disagree")
    layout = attrs.get("jackknife_layout")
    if layout is not None:
        layout = json.loads(layout) if isinstance(layout, str) else layout
        if layout != groups:
            raise ValueError("jackknife block layout must contain exactly the recorded groups")
        expected = sum(groups.values()) + 1
    else:
        if len(groups) != 1:
            raise ValueError("multiple jackknife groups require an explicit block layout")
        expected = next(iter(groups.values()))
    if attrs.get("n_sample") != expected:
        raise ValueError("jackknife row count disagrees with the group/block layout")
    digest = hashlib.sha256(json.dumps(list(range(expected))).encode()).hexdigest()
    if attrs.get("replica_order_digest") != digest:
        raise ValueError("jackknife replica order disagrees with the canonical block ordering")
    return groups


def _matching_subset(source, consumer):
    if source.get("dims") != ["x"] or consumer.get("dims") != ["x"]:
        return False
    try:
        before = np.asarray(source["coords"]["x"], dtype=float)
        after = np.asarray(consumer["coords"]["x"], dtype=float)
    except (KeyError, TypeError, ValueError):
        return False
    return bool(
        before.ndim == after.ndim == 1
        and before.size
        and after.size
        and np.all(np.isfinite(before))
        and np.all(np.isfinite(after))
        and np.all(np.diff(before) > 0)
        and np.all(np.diff(after) > 0)
        and all(np.count_nonzero(np.isclose(before, x, rtol=0, atol=1e-10)) == 1 for x in after)
    )


def run(context: ToolContext) -> dict[str, object]:
    """Compare only metadata that adjacent stages are expected to preserve."""
    summaries = context.state["result_summary"]
    bundle = context.state["review_bundle"]
    checks = set(context.params["checks"])
    by_job = dict(bundle.get("provenance_summaries", {}))
    by_job.update({item["job_id"]: item for item in summaries if item["job_id"] is not None})
    jobs = bundle["jobs"]
    findings: list[dict[str, object]] = []

    for consumer_id, consumer in by_job.items():
        stack = list(jobs[consumer_id]["inputs"].items())
        sources = []
        while stack:
            source_role, value = stack.pop(0)
            if isinstance(value, str):
                if value in jobs:
                    sources.append((source_role, value))
            elif isinstance(value, list):
                stack.extend((source_role, item) for item in value)
            elif isinstance(value, dict):
                stack.extend((source_role, item) for item in value.values())

        for source_role, source_id in sources:
            source_stage = jobs[source_id]["stage_id"]
            consumer_stage = jobs[consumer_id]["stage_id"]
            source = by_job.get(source_id)
            if source is None:
                findings.append(
                    {
                        "status": "not_checkable",
                        "group": "provenance",
                        "source_job": source_id,
                        "consumer_job": consumer_id,
                        "field": "output",
                        "message": "source output metadata is unavailable for Review inspection",
                    }
                )
                continue
            left = dict(source.get("attrs", {}))
            right = dict(consumer.get("attrs", {}))
            tmd = _tmd_job(jobs[consumer_id], consumer)
            auxiliary = tmd and (
                (consumer_stage == "renormalization" and source_role != "target")
                or consumer_stage == "soft_analysis"
                or (consumer_stage == "matching" and source_role != "quasi")
            )
            for field in ("replica_order_digest",):
                if field in source:
                    left[field] = source[field]
                if field in consumer:
                    right[field] = consumer[field]

            def finding(status, group, field, message, source_value=None, consumer_value=None):
                findings.append(
                    {
                        "status": status,
                        "group": group,
                        "source_job": source_id,
                        "consumer_job": consumer_id,
                        "field": field,
                        "source": source_value,
                        "consumer": consumer_value,
                        "message": message,
                    }
                )

            if source.get("ensemble") is not None:
                left["ensemble"] = source["ensemble"]
            if consumer.get("ensemble") is not None:
                right["ensemble"] = consumer["ensemble"]
            if source.get("resample") is not None:
                left["resample"] = source["resample"]
            if consumer.get("resample") is not None:
                right["resample"] = consumer["resample"]
            if source.get("n_sample") is not None:
                left["n_sample"] = source["n_sample"]
            if consumer.get("n_sample") is not None:
                right["n_sample"] = consumer["n_sample"]
            groups = {
                "identity": ("hadron", "kernel_operator", "parton", "target_observable", "polarization", "gfix"),
                "units": ("coord_unit",),
                "kinematics": ("momentum_gev", "initial_momentum_gev", "final_momentum_gev", "t_gev2", "ensemble"),
                "schemes": (
                    ("renormalization_scheme", "kernel_id")
                    if source_stage == "matching"
                    else ("renormalization_scheme",)
                ),
                "resampling": ("resample_id", "resample", "n_sample"),
            }
            for group, fields in groups.items():
                if group not in checks:
                    continue
                if group in {"kinematics", "resampling"} and consumer_stage == "extrapolation":
                    continue
                if group == "identity" and auxiliary:
                    # A soft factor/local-current denominator is a different
                    # operator and may come from another valence hadron.
                    fields = ("parton", "gfix")
                if group == "identity" and tmd and consumer_stage == "fourier_transform":
                    if right.get("target_observable") == "tmd" and left.get("target_observable") in {"tmdwf", "tmdpdf"}:
                        fields = tuple(field for field in fields if field != "target_observable")
                if group == "kinematics" and auxiliary:
                    if "ensemble" in left and "ensemble" in right:
                        before = {key: left["ensemble"].get(key) for key in ("a_s", "a_t", "L_s", "L_t")}
                        after = {key: right["ensemble"].get(key) for key in before}
                        if before != after:
                            finding(
                                "error",
                                group,
                                "ensemble",
                                "lattice geometry changed across a TMD input edge",
                                before,
                                after,
                            )
                    else:
                        finding("not_checkable", group, "ensemble", "TMD input lattice geometry is unavailable")
                    continue
                if group == "resampling" and (left.get("resample_lineage") or right.get("resample_lineage")):
                    jackknife = left.get("resample") == right.get("resample") == "jackknife"
                    tracked_fields = ("resample", "replica_frame_id") if jackknife else (
                        "resample", "n_sample", "replica_frame_id", "replica_order_digest"
                    )
                    for field in tracked_fields:
                        if field not in left or field not in right:
                            finding("not_checkable", group, field, f"tracked replica metadata {field} is missing")
                        elif left[field] != right[field]:
                            finding(
                                "error",
                                group,
                                field,
                                f"tracked replica {field} changed across the input edge",
                                left[field],
                                right[field],
                            )
                    if jackknife:
                        try:
                            source_groups = _jackknife_groups(left)
                            consumer_groups = _jackknife_groups(right)
                            if any(consumer_groups.get(name) != count for name, count in source_groups.items()):
                                raise ValueError("a source jackknife group was dropped or its bin count changed")
                        except (TypeError, ValueError) as exc:
                            finding("error", group, "jackknife_layout", str(exc))
                    try:
                        source_lineage = _lineage(left.get("resample_lineage"))
                        consumer_lineage = _lineage(right.get("resample_lineage"))
                    except (TypeError, ValueError) as exc:
                        finding("error", group, "resample_lineage", f"invalid or missing tracked lineage: {exc}")
                    else:
                        for name, plan in source_lineage.items():
                            if consumer_lineage.get(name) != plan:
                                finding(
                                    "error",
                                    group,
                                    "resample_lineage",
                                    f"source resampling group {name!r} was dropped or its plan changed",
                                    {name: plan},
                                    {name: consumer_lineage.get(name)},
                                )
                    continue
                if group == "units" and tmd:
                    expected = (
                        "fm"
                        if consumer_stage == "renormalization"
                        else "dimensionless"
                        if consumer_stage in {"fourier_transform", "matching"}
                        else None
                    )
                    if expected is not None and right.get("coord_unit") != expected:
                        finding(
                            "error",
                            group,
                            "coord_unit",
                            f"TMD {consumer_stage} output must use {expected}",
                            left.get("coord_unit"),
                            right.get("coord_unit"),
                        )
                    if consumer_stage == "soft_analysis":
                        units = right.get("units", {})
                        try:
                            units = json.loads(units) if isinstance(units, str) else units
                        except ValueError:
                            units = {}
                        if not isinstance(units, dict) or units.get("bT") != "fm":
                            finding("error", group, "units.bT", "TMD soft coordinates must declare fm units")
                    continue
                if group == "schemes" and tmd and consumer_stage == "renormalization":
                    if right.get("renormalization_scheme") != "ratio":
                        finding(
                            "error",
                            group,
                            "renormalization_scheme",
                            "TMD renormalization must output ratio scheme",
                            left.get("renormalization_scheme"),
                            right.get("renormalization_scheme"),
                        )
                    continue
                if group == "schemes" and tmd and consumer_stage == "soft_analysis" and source_role == "form_factor":
                    # The physical FF has no quasi-WF renormalization scheme.
                    continue
                if group == "kinematics" and consumer_stage == "renormalization" and source_role != "target":
                    if left.get("ensemble") != right.get("ensemble") and "ensemble" in left and "ensemble" in right:
                        findings.append(
                            {
                                "status": "warning",
                                "group": group,
                                "source_job": source_id,
                                "consumer_job": consumer_id,
                                "field": "ensemble",
                                "source": left["ensemble"],
                                "consumer": right["ensemble"],
                                "message": "ensemble changed across the renormalization denominator edge",
                            }
                        )
                    continue
                if group == "units" and source_stage == "correlator_analysis" and consumer_stage == "renormalization":
                    if right.get("coord_unit") != "fm":
                        findings.append(
                            {
                                "status": "error",
                                "group": group,
                                "source_job": source_id,
                                "consumer_job": consumer_id,
                                "field": "coord_unit",
                                "source": left.get("coord_unit"),
                                "consumer": right.get("coord_unit"),
                                "message": "renormalization output must use fm coordinates",
                            }
                        )
                    continue
                compared = 0
                for field in fields:
                    if field not in left or field not in right:
                        continue
                    compared += 1
                    if left[field] != right[field]:
                        findings.append(
                            {
                                "status": "error" if group == "identity" else "warning",
                                "group": group,
                                "source_job": source_id,
                                "consumer_job": consumer_id,
                                "field": field,
                                "source": left[field],
                                "consumer": right[field],
                                "message": f"{field} changed across {source_stage} -> {consumer_stage}",
                            }
                        )
                if compared == 0:
                    findings.append(
                        {
                            "status": "not_checkable",
                            "group": group,
                            "source_job": source_id,
                            "consumer_job": consumer_id,
                            "field": None,
                            "message": f"no shared {group} metadata across {source_stage} -> {consumer_stage}",
                        }
                    )
            if "grids" in checks and (source_stage, consumer_stage) in {
                ("fourier_transform", "matching"),
                ("matching", "extrapolation"),
            }:
                if source.get("dims") is None or consumer.get("dims") is None:
                    findings.append(
                        {
                            "status": "not_checkable",
                            "group": "grids",
                            "source_job": source_id,
                            "consumer_job": consumer_id,
                            "field": "dims",
                            "message": "one output has no inspectable grid",
                        }
                    )
                elif (
                    tmd
                    and consumer_stage == "matching"
                    and source_role == "quasi"
                    and _matching_subset(source, consumer)
                ):
                    pass
                elif source.get("dims") != consumer.get("dims") or source.get("coords") != consumer.get("coords"):
                    findings.append(
                        {
                            "status": "warning",
                            "group": "grids",
                            "source_job": source_id,
                            "consumer_job": consumer_id,
                            "field": "dims/coords",
                            "source": {"dims": source.get("dims"), "coords": source.get("coords")},
                            "consumer": {"dims": consumer.get("dims"), "coords": consumer.get("coords")},
                            "message": f"distribution grid changed across {source_stage} -> {consumer_stage}",
                        }
                    )

    if "extrapolation" in checks:
        extrapolated = [item for item in summaries if item.get("stage_id") == "extrapolation"]
        for item in extrapolated:
            attrs = item.get("attrs", {})
            if "physical_pion_mass_gev" not in attrs and "physical_point" not in attrs:
                findings.append(
                    {
                        "status": "not_checkable",
                        "group": "extrapolation",
                        "source_job": None,
                        "consumer_job": item["job_id"],
                        "field": "physical_point",
                        "message": "extrapolation output does not identify its physical point",
                    }
                )

    document = {
        "checks": list(context.params["checks"]),
        "findings": findings,
        "counts": {
            status: sum(finding["status"] == status for finding in findings)
            for status in ("error", "warning", "info", "not_checkable")
        },
    }
    (context.artifact_directory / "consistency.json").write_text(
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8"
    )
    context.state["consistency"] = document
    return {
        "summary": f"recorded {len(findings)} stage-aware consistency findings",
        "metrics": document["counts"],
        "findings": findings,
        "state_keys": ["consistency"],
        "artifacts": ["consistency.json"],
    }
