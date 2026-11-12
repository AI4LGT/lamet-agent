"""Deterministic matching job workflow."""

from __future__ import annotations

from lamet_agent.agent import LlmSession, ToolContext
from lamet_agent.stages.matching._apply import run as apply
from lamet_agent.stages.matching._inspection import run as inspect


def run(context: ToolContext, _session: LlmSession) -> None:
    """Inspect the selected kernel and publish its deterministic application."""
    if context.params.get("observable") in {"tmdwf", "tmdpdf"}:
        from lamet_agent.stages.matching.tmd import run as match_tmd

        match_tmd(context, _session)
        return
    inspect(context)
    apply(context)


__all__ = ["run"]
