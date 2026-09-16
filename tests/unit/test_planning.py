"""Tests for the rewritten validator-grounded Plan mode."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import lamet_agent.plan.tools as planning_tools_package
from lamet_agent.agent import create_session
from lamet_agent.llm import _AssistantResponse, _ToolCall
from lamet_agent.plan.state import PlanState
from lamet_agent.plan.tools import (
    PLANNING_TOOLS,
    planning_controller_prompt,
    planning_tool_schemas,
    run_planning_tool,
)
from lamet_agent.ui import PlainUi


class _ScriptedBackend:
    identity = "scripted:plan"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def complete(self, *, messages, tools, prompt_digest, response_schema=None):
        self.calls.append((list(messages), tools, prompt_digest, response_schema))
        return self.responses.pop(0)


class _FakeTui(PlainUi):
    def __init__(self, *, answers=(), confirmations=(), reviews=()):
        self.answers = list(answers)
        self.confirmations = list(confirmations)
        self.reviews = list(reviews)
        self.messages = []
        self.patch_events = []

    def write(self, message):
        self.messages.append(str(message))

    def log(self, message="", **_kwargs):
        self.write(message)

    def ask(self, question, state=None, *, placeholder=""):
        self.messages.append(str(question))
        return self.answers.pop(0)

    def confirm(self, question):
        self.messages.append(str(question))
        return self.confirmations.pop(0)

    def review_plan(self, question, _state, *, accept_label="Accept and save"):
        self.messages.append(str(question))
        self.messages.append(accept_label)
        return self.reviews.pop(0) if self.reviews else self.confirmations.pop(0)

    def show_patch(self, edits, state):
        self.patch_events.append((copy.deepcopy(edits), len(state.issues)))

    def close(self):
        pass


def _review_manifest(tmp_path: Path, *, include_seed: bool) -> tuple[Path, dict]:
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    metadata = {
        "run_id": "plan",
        "root_directory": str(tmp_path),
        "artifacts_directory": "runs",
        "workers": 1,
        "target_observable": "pdf",
        "parton": "quark",
        "resample_mode": "jackknife",
        "sample_error_mode": "covariance",
        "bin_size": 1,
    }
    if include_seed:
        metadata["random_seed"] = 7
    document = {
        "metadata": metadata,
        "stages": {
            "review": {
                "defaults": {
                    "catalog": "builtin",
                    "max_papers": 1,
                    "report_language": "en",
                    "checks": ["identity"],
                },
                "jobs": [{"id": "review", "inputs": {"results": [{"file": str(source)}]}}],
            }
        },
    }
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, document


def test_plan_conversation_parses_natural_answer_and_requires_confirmation(tmp_path: Path) -> None:
    path, original = _review_manifest(tmp_path, include_seed=False)
    backend = _ScriptedBackend(
        [
            _AssistantResponse("Which reproducible random seed should this run use?"),
            _AssistantResponse(
                "",
                _ToolCall(
                    "patch",
                    "apply_manifest_patch",
                    {"patches": [{"op": "add", "path": "/metadata/random_seed", "value": 7}]},
                ),
            ),
            _AssistantResponse(
                "",
                _ToolCall(
                    "finish",
                    "finish_plan",
                    {
                        "summary": "The review-only workflow is complete and reproducible.",
                        "changes": ["Set the reproducible random seed to 7."],
                    },
                ),
            ),
        ]
    )
    tui = _FakeTui(answers=["draft.planned.json", "Use seed seven for reproducibility."], confirmations=[True])

    output = create_session(backend).plan_manifest(path, tui=tui)

    assert output == tmp_path / "draft.planned.json"
    planned = json.loads(output.read_text(encoding="utf-8"))
    assert planned["metadata"]["random_seed"] == 7
    assert "random_seed" not in original["metadata"]
    assert len(tui.patch_events) == 1
    assert any("Set the reproducible random seed to 7." in message for message in tui.messages)
    assert "Accept and save" in tui.messages
    output_question = next(i for i, message in enumerate(tui.messages) if "Output filename" in message)
    seed_question = next(i for i, message in enumerate(tui.messages) if "Which reproducible" in message)
    assert output_question < seed_question


def test_natural_control_tools_read_and_undo_without_a_diff(tmp_path: Path) -> None:
    path, document = _review_manifest(tmp_path, include_seed=False)
    state = PlanState(path, tmp_path / "draft.planned.json", copy.deepcopy(document), copy.deepcopy(document))
    state.refresh()
    applied = run_planning_tool(
        state,
        "apply_manifest_patch",
        {"patches": [{"op": "add", "path": "/metadata/random_seed", "value": 11}]},
    )

    shown = run_planning_tool(state, "read_manifest", {"path": "/metadata/random_seed"})
    undone = run_planning_tool(state, "undo_manifest_change", {})

    assert applied["ok"] is True
    assert shown == {"tool": "read_manifest", "ok": True, "path": "/metadata/random_seed", "value": 11}
    assert undone["undone"] is True
    assert state.manifest_view("/metadata/random_seed")["ok"] is False
    assert {tool.name for tool in PLANNING_TOOLS} >= {
        "read_manifest",
        "undo_manifest_change",
        "save_draft",
        "cancel_plan",
    }
    prompt = planning_controller_prompt()
    assert "undo_manifest_change" in prompt
    assert "Natural-language control requests" in prompt
    assert "explicit user request may intentionally extend" in prompt
    assert "inspect_manifest_contract" in prompt
    assert "Do not mechanically dump the complete Issue list" in prompt
    assert "several independent Issues" in prompt
    assert "Multi-turn completion is available but is not mandatory" in prompt
    assert "long or burdensome" in prompt
    assert "After every successful patch" in prompt
    assert "physicist, not a manifest-schema author" in prompt
    assert "unexplained keys or bare option lists" in prompt
    assert "enough plain-language meaning" in prompt
    assert "informational question does not authorize" in prompt
    assert "never skip a pending user question" in prompt
    finish_schema = next(
        schema for schema in planning_tool_schemas() if schema["function"]["name"] == "finish_plan"
    )
    assert finish_schema["function"]["parameters"]["required"] == ["summary", "changes"]


def test_every_planning_tool_owns_its_implementation_prompt_and_schema() -> None:
    tools_root = Path(planning_tools_package.__file__).parent
    controller_prompt = planning_controller_prompt()
    schemas = {schema["function"]["name"]: schema["function"] for schema in planning_tool_schemas()}

    for tool in PLANNING_TOOLS:
        directory = tools_root / tool.name
        prompt_path = directory / "prompts.md"
        assert (directory / "__init__.py").is_file()
        assert prompt_path.is_file()
        assert "Call schema:" in tool.prompt
        assert tool.prompt == prompt_path.read_text(encoding="utf-8").strip()
        assert tool.prompt in controller_prompt
        assert schemas[tool.name]["parameters"] == tool.parameters


def test_valid_manifest_requires_output_name_before_confirmation_without_llm(tmp_path: Path) -> None:
    path, _document = _review_manifest(tmp_path, include_seed=True)
    backend = _ScriptedBackend([])
    tui = _FakeTui(answers=["chosen.json"], confirmations=[True])

    output = create_session(backend).plan_manifest(path, tui=tui)

    assert output == tmp_path / "chosen.json"
    assert output.is_file()
    assert backend.calls == []
    assert "Accept and save" in tui.messages
    assert sum("Output filename" in message for message in tui.messages) == 1
    assert not (tmp_path / "draft.planned.json").exists()
    panel = next(message for message in tui.messages if message.startswith("● Manifest validated"))
    assert "Save as: chosen.json" in panel
    assert str(tmp_path) not in panel


@pytest.mark.parametrize("in_place", [False, True])
@pytest.mark.parametrize("run_after", [False, True])
def test_explicit_output_skips_filename_question_and_labels_acceptance(tmp_path, in_place, run_after) -> None:
    path, _document = _review_manifest(tmp_path, include_seed=True)
    target = path if in_place else tmp_path / "existing.json"
    if not in_place:
        target.write_text("{}", encoding="utf-8")
    tui = _FakeTui(confirmations=[True])
    output = create_session(_ScriptedBackend([])).plan_manifest(
        path, output_path=None if in_place else target, in_place=in_place, run_after=run_after, tui=tui
    )

    assert output == target
    assert not any("Output filename" in message for message in tui.messages)
    assert ("Save and run" if run_after else "Accept and save") in tui.messages
    expected = "overwrites source" if in_place else "overwrites existing file"
    assert any(expected in message for message in tui.messages)


def test_prompted_output_must_remain_beside_source(tmp_path) -> None:
    path, _document = _review_manifest(tmp_path, include_seed=True)
    tui = _FakeTui(answers=["../outside.json"])
    with pytest.raises(ValueError, match="remain beside"):
        create_session(_ScriptedBackend([])).plan_manifest(path, tui=tui)


def test_suggested_output_still_requires_final_acceptance(tmp_path) -> None:
    path, _document = _review_manifest(tmp_path, include_seed=True)
    original = path.read_text(encoding="utf-8")
    tui = _FakeTui(answers=[""], confirmations=[False])

    assert create_session(_ScriptedBackend([])).plan_manifest(path, tui=tui) is None
    assert path.read_text(encoding="utf-8") == original
    assert any("Save as: draft.planned.json" in message for message in tui.messages)
    assert not (tmp_path / "draft.planned.json").exists()


def test_output_and_in_place_remain_mutually_exclusive(tmp_path) -> None:
    path, _document = _review_manifest(tmp_path, include_seed=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        create_session(_ScriptedBackend([])).plan_manifest(
            path, output_path=tmp_path / "chosen.json", in_place=True, tui=_FakeTui()
        )


def test_rejected_plan_never_writes_or_enters_run_mode(tmp_path: Path) -> None:
    path, _document = _review_manifest(tmp_path, include_seed=True)
    tui = _FakeTui(answers=["draft.planned.json"], confirmations=[False])

    output = create_session(_ScriptedBackend([])).plan_manifest(path, tui=tui)

    assert output is None
    assert not (tmp_path / "draft.planned.json").exists()


def test_plan_returns_invalid_read_path_to_model_and_continues(tmp_path: Path) -> None:
    path, document = _review_manifest(tmp_path, include_seed=True)
    backend = _ScriptedBackend([
        _AssistantResponse("", _ToolCall("bad-read", "read_manifest", {"path": "metadata.random_seed"})),
        _AssistantResponse("", _ToolCall("fixed-read", "read_manifest", {"path": "/metadata/random_seed"})),
        _AssistantResponse("", _ToolCall("finish", "finish_plan", {"summary": "Reviewed.", "changes": []})),
    ])
    tui = _FakeTui(reviews=["Inspect this manifest.", True])

    assert create_session(backend).plan_manifest(path, in_place=True, tui=tui) == path
    observations = {
        message.tool_call_id: json.loads(message.content)
        for message in backend.calls[-1][0]
        if message.role == "tool"
    }
    assert observations["bad-read"]["ok"] is False
    assert "JSON Pointer" in observations["bad-read"]["error"]
    assert observations["fixed-read"]["ok"] is True
    assert observations["fixed-read"]["value"] == 7
    assert json.loads(path.read_text()) == document


def test_plan_can_continue_beyond_sixty_calls_and_tool_steps(tmp_path: Path) -> None:
    path, _document = _review_manifest(tmp_path, include_seed=True)
    backend = _ScriptedBackend([
        *[
            _AssistantResponse("", _ToolCall(f"read-{index}", "read_manifest", {"path": "/metadata"}))
            for index in range(65)
        ],
        _AssistantResponse("", _ToolCall("finish", "finish_plan", {"summary": "Reviewed.", "changes": []})),
    ])
    tui = _FakeTui(reviews=["Inspect this manifest.", True])

    assert create_session(backend).plan_manifest(path, in_place=True, tui=tui) == path
    assert len(backend.calls) == 66


def test_valid_plan_accepts_explicit_revision_before_final_confirmation(tmp_path: Path, monkeypatch) -> None:
    path, document = _review_manifest(tmp_path, include_seed=True)
    second_job = copy.deepcopy(document["stages"]["review"]["jobs"][0])
    second_job["id"] = "review_extra"
    backend = _ScriptedBackend(
        [
            _AssistantResponse(
                "",
                _ToolCall(
                    "inspect",
                    "inspect_manifest_contract",
                    {"path": "stages.review.jobs"},
                ),
            ),
            _AssistantResponse("What identifier should the additional review job use?"),
            _AssistantResponse(
                "",
                _ToolCall(
                    "patch",
                    "apply_manifest_patch",
                    {"patches": [{"op": "add", "path": "/stages/review/jobs/-", "value": second_job}]},
                ),
            ),
            _AssistantResponse(
                "",
                _ToolCall(
                    "finish",
                    "finish_plan",
                    {
                        "summary": "The valid review workflow now contains an additional requested review job.",
                        "changes": ["Added review job review_extra with the same selected input."],
                    },
                ),
            ),
        ]
    )
    answers = iter(
        ["chosen.json", "?", "Add another review job using the same input.",
         "Use review_extra and the same selected result.", "y"]
    )
    monkeypatch.setattr("builtins.input", lambda *_args: next(answers))
    tui = PlainUi()

    output = create_session(backend).plan_manifest(path, tui=tui)

    planned = json.loads(output.read_text(encoding="utf-8"))
    assert [job["id"] for job in planned["stages"]["review"]["jobs"]] == ["review", "review_extra"]
    inspect_observation = json.loads(backend.calls[1][0][-1].content)
    assert inspect_observation["tool"] == "inspect_manifest_contract"
    assert inspect_observation["ok"] is True
    assert backend.calls[0][0][-1].content == "Add another review job using the same input."


def test_cli_plan_writes_manifest_without_entering_run(tmp_path: Path, monkeypatch, capsys) -> None:
    import lamet_agent.__main__ as cli

    path, _document = _review_manifest(tmp_path, include_seed=True)

    class Backend:
        identity = "fake"

        def close(self):
            pass

    class Session:
        ui = PlainUi()

        def close(self):
            pass

        def validate_manifest(self, manifest, **_kwargs):
            return manifest.validate()

        def plan_manifest(self, *_args, **_kwargs):
            return path

        def run_manifest(self, _manifest):
            raise AssertionError("plan command must not enter run mode")

    backend = Backend()
    monkeypatch.setattr(cli, "create_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(cli, "create_session", lambda selected, **_kwargs: Session() if selected is backend else None)

    status = cli.main(["plan", str(path), "--provider", "codex", "--model", "gpt-test"])

    assert status == 0
    assert f"manifest written: {path}" in capsys.readouterr().out


def test_cli_run_valid_manifest_skips_plan(tmp_path: Path, monkeypatch, capsys) -> None:
    import lamet_agent.__main__ as cli

    path, _document = _review_manifest(tmp_path, include_seed=True)
    events = []

    class Backend:
        identity = "fake"

        def close(self):
            pass

    class Session:
        ui = PlainUi()

        def close(self):
            pass

        def validate_manifest(self, manifest, **_kwargs):
            return manifest.validate()

        def plan_manifest(self, *_args, **_kwargs):
            raise AssertionError("valid run manifest must not enter Plan")

        def run_manifest(self, manifest):
            events.append(("run", manifest.path))
            return {"summaries": {"review": {"result": "review"}}}

    backend = Backend()
    monkeypatch.setattr(cli, "create_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(cli, "create_session", lambda selected, **_kwargs: Session() if selected is backend else None)

    status = cli.main(["run", str(path), "--provider", "codex", "--model", "gpt-test"])

    assert status == 0
    assert events == [("run", path.resolve())]
    assert '"status": "completed"' in capsys.readouterr().out


def test_cli_run_invalid_manifest_plans_then_runs_accepted_manifest(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import lamet_agent.__main__ as cli

    draft, document = _review_manifest(tmp_path, include_seed=False)
    document["metadata"]["random_seed"] = 19
    accepted = tmp_path / "draft.planned.json"
    accepted.write_text(json.dumps(document), encoding="utf-8")
    events = []

    class Backend:
        identity = "fake"

        def close(self):
            pass

    class Session:
        ui = PlainUi()

        def close(self):
            pass

        def validate_manifest(self, manifest, **_kwargs):
            return manifest.validate()

        def plan_manifest(self, manifest_path, **_kwargs):
            assert _kwargs["run_after"] is True
            events.append(("plan", Path(manifest_path).resolve()))
            return accepted

        def run_manifest(self, manifest):
            events.append(("run", manifest.path))
            return {"summaries": {"review": {"result": "review"}}}

    backend = Backend()
    monkeypatch.setattr(cli, "create_backend", lambda *_args, **_kwargs: backend)
    monkeypatch.setattr(cli, "create_session", lambda selected, **_kwargs: Session() if selected is backend else None)

    status = cli.main(["run", str(draft), "--provider", "codex", "--model", "gpt-test"])

    assert status == 0
    assert events == [("plan", draft.resolve()), ("run", accepted.resolve())]
    assert '"status": "completed"' in capsys.readouterr().out


def test_cli_validate_never_constructs_an_llm_backend(tmp_path: Path, monkeypatch) -> None:
    import lamet_agent.__main__ as cli

    path, _document = _review_manifest(tmp_path, include_seed=False)
    monkeypatch.setattr(
        cli,
        "create_backend",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("validate must not construct an LLM")),
    )

    assert cli.main(["validate", str(path)]) == 1


def test_plan_uses_unique_shared_transcripts_for_response_repair(tmp_path):
    from lamet_agent.llm import InvalidResponseError

    source, _ = _review_manifest(tmp_path, include_seed=True)
    for _ in range(2):
        backend = _ScriptedBackend([])
        outputs = iter([
            InvalidResponseError('broken response', '{broken'),
            _AssistantResponse('', _ToolCall('finish', 'finish_plan', {'summary': 'Ready.', 'changes': []})),
        ])

        def complete(**kwargs):
            value = next(outputs)
            if isinstance(value, Exception):
                raise value
            return value

        backend.complete = complete
        tui = _FakeTui(reviews=['Inspect this manifest.', True])
        create_session(backend).plan_manifest(source, in_place=True, tui=tui)
    paths = list((tmp_path / 'artifacts' / 'plan').glob('*/llm_transcript.md'))
    assert len(paths) == 2
    for path in paths:
        text = path.read_text()
        assert 'Inspect this manifest.' in text
        assert '{broken' in text
        assert 'InvalidResponseError' in text
        assert 'elapsed_seconds' in text
        assert 'received from LLM' in text
