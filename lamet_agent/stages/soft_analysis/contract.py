"""Accept an empty Soft Analysis stage while rejecting executable jobs."""

from dataclasses import replace

from lamet_agent.contract import CheckContext, Depends, Issue, List, stage_job_rules


PARAM_RULES = ()
INPUT_RULES = ()

# Keep the shared job envelope, allowing an empty list only for this placeholder.
JOB_RULES = tuple(
    replace(rule, validator=None)
    if isinstance(rule, List) and rule.path == "jobs"
    else replace(rule, physics="Soft Analysis may declare an empty job list as a placeholder.")
    if isinstance(rule, Depends) and rule.path == "jobs"
    else rule
    for rule in stage_job_rules(PARAM_RULES, INPUT_RULES)
)


def _not_implemented(_context: CheckContext) -> list[Issue]:
    return [
        Issue(
            "",
            "Soft Analysis is not implemented yet; use an empty jobs list or omit the stage.",
            "Soft Analysis currently reserves a stage name without defining a numerical workflow.",
        )
    ]


CHECKS = (_not_implemented,)
