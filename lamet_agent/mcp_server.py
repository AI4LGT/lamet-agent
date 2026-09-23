"""Serve lamet-agent over the Model Context Protocol on stdio.

Purpose: let an MCP client (for example the DeepSeek Harness) validate, plan,
and run manifests as ordinary tool calls.

Design constraints this module exists to satisfy, each established by
inspection rather than assumption:

* **stdout is the protocol channel.** Nothing else may be written to it. The
  CLI's own UI prints a banner to stdout, so every in-process CLI call runs
  inside a stdout redirect and the captured text becomes the tool result.
* **stdin is the protocol channel too.** ``PlainUi.ask`` reads ``input()``,
  which on an MCP stdio server would consume JSON-RPC bytes and hang the
  connection. ``_McpUi`` therefore refuses to prompt: an interactive path
  fails loudly instead of deadlocking.
* **No new dependency.** The wire protocol is newline-delimited JSON-RPC 2.0,
  which the standard library covers. Pulling in the official ``mcp`` SDK would
  add uvicorn, starlette, sse-starlette, pyjwt and opentelemetry for a pipe.
* **Startup must be fast and quiet.** The client fails the whole connection if
  this process does not answer promptly, and it retries with backoff before
  giving up, so import cost and stderr discipline matter.
"""

from __future__ import annotations

import io
import json
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterator

# The revision every contemporary MCP server speaks. The client sends a newer
# one and accepts this as a downgrade.
PROTOCOL_VERSION = "2025-06-18"

SERVER_INFO = {"name": "lamet-agent", "version": "0.1.0"}

_MAX_TEXT_CHARS = 60_000


class _PromptRequired(RuntimeError):
    """Raised when an in-process CLI path tries to read from the terminal."""


def _log(message: str) -> None:
    """Write a diagnostic to stderr, which the client forwards to its terminal.

    Diagnostics must never touch stdout: the client treats any non-protocol
    line there as a protocol violation.
    """
    print(f"[lamet-agent-mcp] {message}", file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _non_interactive_ui_class() -> type:
    """Build a PlainUi subclass whose interactive methods refuse to prompt.

    Every prompting path in ``PlainUi`` must be overridden, not just ``ask``:
    ``confirm`` (ui.py:300-304) and ``_plan_choice`` (ui.py:330-333) also call
    ``input()``, and an MCP stdio server keeps stdin open as the JSON-RPC
    channel. Reading it would consume protocol bytes and block forever, which
    is strictly worse than failing, because a client sees only a hang.

    Deferred to first call so that importing this module does not import the
    numerical stack: a server slow to import risks failing the client's
    startup handshake. Cached because the class is otherwise rebuilt on every
    tool call.
    """
    from .ui import PlainUi

    def _refuse(what: str) -> _PromptRequired:
        return _PromptRequired(
            f"this lamet-agent path needs an interactive {what}, which an MCP tool call cannot "
            "provide. Supply every value explicitly (for example --provider, --model and "
            "--output) so the CLI does not have to ask."
        )

    class _NonInteractiveUi(PlainUi):
        def ask(self, question: str, _state: Any | None = None, *, placeholder: str = "") -> str:
            raise _refuse("answer")

        def confirm(self, question: str) -> bool:
            raise _refuse("confirmation")

        def _plan_choice(self, question: str, accept_label: str) -> str:
            raise _refuse("acceptance choice")

        def select_model(self, provider: str, models: list[str], requested: str | None = None) -> str:
            available = ", ".join(models) if models else "(none reported)"
            raise _PromptRequired(
                f"model selection needs an interactive answer. Pass an explicit model argument. "
                f"Models available for provider {provider!r}: {available}"
            )

    return _NonInteractiveUi


def _run_cli(argv: list[str]) -> tuple[int, str]:
    """Run the CLI in-process with a non-interactive UI, capturing its output.

    Returns ``(exit_code, captured_text)``. The CLI is invoked as a function
    rather than a subprocess so a tool call costs no interpreter startup and
    works even where the ``lamet-agent`` console script is not on ``PATH``.

    The UI is replaced by patching ``lamet_agent.ui.PlainUi`` itself, because
    ``create_ui`` resolves that name from its own module namespace at call time.
    Patching ``ui.create_ui`` instead does nothing: ``__main__`` does
    ``from .ui import create_ui``, so it holds its own reference. Getting this
    wrong is not a crash -- it silently restores the interactive UI, which both
    re-enables the ``input()`` hang and leaks the ASCII banner into the tool
    result.
    """
    from . import ui as ui_module

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    interactive_class = _non_interactive_ui_class()
    original_plain_ui = ui_module.PlainUi

    def _quiet_plain_ui(*args: Any, **kwargs: Any) -> Any:
        instance = interactive_class()
        # Decorative only, and it would corrupt both the result and the channel.
        instance.show_banner = lambda: None  # type: ignore[method-assign]
        return instance

    # Imported after the patch target is chosen so dispatch sees it in place.
    from .__main__ import main as cli_main

    ui_module.PlainUi = _quiet_plain_ui  # type: ignore[assignment,misc]
    try:
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            code = cli_main(argv)
    except _PromptRequired as exc:
        return 2, f"interactive input required: {exc}"
    except SystemExit as exc:  # argparse and friends
        code = int(exc.code or 0)
    except Exception:
        return 2, "lamet-agent raised an unhandled error:\n" + traceback.format_exc()
    finally:
        ui_module.PlainUi = original_plain_ui  # type: ignore[assignment]

    text = stdout_buffer.getvalue()
    errors = stderr_buffer.getvalue()
    if errors.strip():
        text = f"{text}\n{errors}".strip()
    return int(code), text


def _text_result(text: str, *, is_error: bool = False) -> dict[str, Any]:
    if len(text) > _MAX_TEXT_CHARS:
        text = f"{text[:_MAX_TEXT_CHARS]}\n... (truncated at {_MAX_TEXT_CHARS} characters)"
    result: dict[str, Any] = {"content": [{"type": "text", "text": text or "(no output)"}]}
    if is_error:
        result["isError"] = True
    return result


def _require_string(arguments: dict[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name!r} is required and must be a non-empty string")
    return value


def _optional_string(arguments: dict[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name!r} must be a non-empty string when provided")
    return value


def _build_argv(arguments: dict[str, Any], command: str, *, needs_provider: bool) -> list[str]:
    """Translate tool arguments into CLI arguments.

    Values are forwarded explicitly so the CLI never reaches an interactive
    prompt during a tool call: an MCP call has no terminal to answer one.
    """
    manifest = _require_string(arguments, "manifest")
    argv = [command, manifest]
    if not needs_provider:
        return argv

    provider = _optional_string(arguments, "provider")
    model = _optional_string(arguments, "model")
    if provider is None or model is None:
        raise ValueError(
            f"{command} requires explicit 'provider' and 'model' arguments. lamet-agent never "
            "selects a model automatically, and an MCP tool call cannot answer a selection prompt."
        )
    argv += ["--provider", provider, "--model", model]
    api_key_file = _optional_string(arguments, "api_key_file")
    if api_key_file is not None:
        argv += ["--api-key-file", api_key_file]
    progress = _optional_string(arguments, "progress")
    if progress is not None:
        argv += ["--progress", progress]
    return argv


def _tool_validate(arguments: dict[str, Any]) -> dict[str, Any]:
    argv = _build_argv(arguments, "validate", needs_provider=False)
    code, text = _run_cli(argv)
    if code == 0:
        return _text_result(f"VALID\n{text}".strip())
    if code == 1:
        return _text_result(f"INVALID (validation issues)\n{text}".strip(), is_error=True)
    return _text_result(f"ERROR (exit {code})\n{text}".strip(), is_error=True)


def _tool_plan(arguments: dict[str, Any]) -> dict[str, Any]:
    argv = _build_argv(arguments, "plan", needs_provider=True)
    output = _optional_string(arguments, "output")
    if output is None:
        raise ValueError(
            "'output' is required for plan: without it the CLI prompts for a filename, which an "
            "MCP tool call cannot answer."
        )
    argv += ["--output", output]
    code, text = _run_cli(argv)
    if code == 0:
        return _text_result(f"PLAN ACCEPTED\n{text}".strip())
    return _text_result(f"PLAN FAILED (exit {code})\n{text}".strip(), is_error=True)


def _tool_run(arguments: dict[str, Any]) -> dict[str, Any]:
    argv = _build_argv(arguments, "run", needs_provider=True)
    code, text = _run_cli(argv)
    if code == 0:
        return _text_result(f"RUN COMPLETED\n{text}".strip())
    return _text_result(f"RUN FAILED (exit {code})\n{text}".strip(), is_error=True)


def _tool_read_run(arguments: dict[str, Any]) -> dict[str, Any]:
    """Report what a completed run produced, without parsing the numerics."""
    directory = Path(_require_string(arguments, "directory")).expanduser()
    if not directory.is_dir():
        return _text_result(f"no such directory: {directory}", is_error=True)

    lines: list[str] = [f"run directory: {directory}"]
    summary_path = directory / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            lines.append(f"summary.json unreadable: {exc}")
        else:
            for key in ("stage_id", "job_id", "result"):
                if key in summary:
                    lines.append(f"{key}: {json.dumps(summary[key], ensure_ascii=False)}")
            for key in ("decisions", "diagnostics"):
                if key in summary:
                    rendered = json.dumps(summary[key], ensure_ascii=False, indent=2)
                    if len(rendered) > 4000:
                        rendered = rendered[:4000] + "\n... (truncated)"
                    lines.append(f"{key}:\n{rendered}")
            artifacts = summary.get("artifacts")
            if artifacts:
                lines.append("artifacts: " + ", ".join(map(str, artifacts)))
    else:
        lines.append("summary.json: absent (the job did not finish)")
    for name in ("resolved_manifest.json", "report.md", "review.md", "review_bundle.json"):
        if (directory / name).is_file():
            lines.append(f"present: {name}")
    return _text_result("\n".join(lines))


def _tool_descriptor(arguments: dict[str, Any]) -> dict[str, Any]:
    """List the correlator records a manifest can bind to.

    Authoring a manifest is mostly binding to values that exist only inside a
    descriptor: record ids, momenta, the available tsep and z ranges. This
    exposes exactly that surface so a client can write correct job inputs.
    """
    path = Path(_require_string(arguments, "path")).expanduser()
    if not path.is_file():
        return _text_result(f"no such descriptor: {path}", is_error=True)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return _text_result(f"descriptor unreadable: {exc}", is_error=True)

    records = document.get("correlators")
    if not isinstance(records, list):
        return _text_result("descriptor has no 'correlators' list", is_error=True)

    lines = [f"descriptor: {path}", f"records: {len(records)}"]
    tseps: list[Any] = []
    zs: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        coords = record.get("coords") or {}
        for value in coords.get("tsep") or []:
            if value not in tseps:
                tseps.append(value)
        for value in coords.get("z") or []:
            if value not in zs:
                zs.append(value)
        current = record.get("current") or {}
        lines.append(
            f"  {record.get('id')}  type={record.get('correlator_type')} "
            f"P={record.get('source_momentum')} gfix={(record.get('selectors') or {}).get('gfix')} "
            f"scheme={current.get('renormalization_scheme')} "
            f"hadron={(record.get('hadron') or {}).get('name')} count={record.get('count')}"
        )
    if tseps:
        lines.append(f"tsep values available: {sorted(tseps, key=str)}")
    if zs:
        lines.append(f"z values available: {sorted(zs, key=str)}")
    return _text_result("\n".join(lines))


TOOLS: tuple[dict[str, Any], ...] = (
    {
        "name": "validate_manifest",
        "description": (
            "Validate a lamet-agent manifest against its stage contracts. Read-only, offline, "
            "and never prompts. Returns the CLI's own verdict: valid, or the list of issues."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "manifest": {"type": "string", "description": "Path to the manifest (JSON or JSONC)."},
            },
            "required": ["manifest"],
            "additionalProperties": False,
        },
    },
    {
        "name": "plan_manifest",
        "description": (
            "Complete or repair an incomplete manifest through lamet-agent's Plan conversation and "
            "write the result to 'output'. NOTE: Plan ends with an interactive acceptance review "
            "that cannot be answered over MCP, so this call reports an error explaining that instead "
            "of accepting the result. Prefer run_manifest, which repairs and executes in one step, "
            "or run 'lamet-agent plan' in a terminal."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "manifest": {"type": "string", "description": "Path to the manifest to repair."},
                "output": {"type": "string", "description": "Where to write the planned manifest."},
                "provider": {"type": "string", "description": "Registered provider, e.g. 'codex'."},
                "model": {"type": "string", "description": "Explicit model id; there is no auto-selection."},
                "api_key_file": {"type": "string", "description": "Optional API key file for API providers."},
            },
            "required": ["manifest", "output", "provider", "model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_manifest",
        "description": (
            "Validate and then execute a manifest, writing the full artifact tree. Long-running: "
            "raise the client's per-call timeout for real analyses. Requires provider and model."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "manifest": {"type": "string", "description": "Path to the manifest to execute."},
                "provider": {"type": "string", "description": "Registered provider, e.g. 'codex'."},
                "model": {"type": "string", "description": "Explicit model id; there is no auto-selection."},
                "api_key_file": {"type": "string", "description": "Optional API key file for API providers."},
                "progress": {
                    "type": "string",
                    "enum": ["auto", "stage", "job", "none"],
                    "description": "Progress reporting mode; 'none' is quietest for a tool call.",
                },
            },
            "required": ["manifest", "provider", "model"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_run",
        "description": (
            "Summarise what a run directory contains: the job summary, its decisions and "
            "diagnostics, and which reports and manifests are present."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {"type": "string", "description": "A job or run artifact directory."},
            },
            "required": ["directory"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_correlators",
        "description": (
            "List the correlator records inside a descriptor so a manifest can bind to real ids, "
            "momenta, and available tsep and z values instead of guessing them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a correlator descriptor JSON."},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
)

_HANDLERS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "validate_manifest": _tool_validate,
    "plan_manifest": _tool_plan,
    "run_manifest": _tool_run,
    "read_run": _tool_read_run,
    "list_correlators": _tool_descriptor,
}


def _response(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """Dispatch one JSON-RPC message. Returns None for notifications."""
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params") or {}

    if method == "initialize":
        return _response(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
                "instructions": (
                    "lamet-agent runs reproducible LaMET / lattice-QCD analysis from a JSONC "
                    "manifest. Use list_correlators to bind a manifest to real correlator records, "
                    "validate_manifest before running, run_manifest to execute, and read_run to "
                    "inspect the result. Every verdict comes from the CLI; this server does not "
                    "re-implement validation."
                ),
            },
        )

    if method in ("notifications/initialized", "notifications/cancelled"):
        return None

    if method == "ping":
        return _response(request_id, {})

    if method == "tools/list":
        return _response(request_id, {"tools": list(TOOLS)})

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        handler = _HANDLERS.get(str(name))
        if handler is None:
            return _error(request_id, -32602, f"unknown tool: {name!r}")
        if not isinstance(arguments, dict):
            return _error(request_id, -32602, "arguments must be an object")
        try:
            return _response(request_id, handler(arguments))
        except ValueError as exc:
            # Treat as a tool-level failure so the model can read and correct it.
            return _response(request_id, _text_result(str(exc), is_error=True))
        except Exception:
            _log("tool call failed:\n" + traceback.format_exc())
            return _response(
                request_id,
                _text_result("internal error; see the server's stderr", is_error=True),
            )

    if request_id is None:
        return None
    return _error(request_id, -32601, f"method not found: {method!r}")


def _iter_messages(stream: Any) -> Iterator[dict[str, Any]]:
    for line in stream:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            _log(f"ignoring non-JSON input line: {stripped[:120]!r}")
            continue
        if isinstance(message, dict):
            yield message


def serve(stdin: Any = None, stdout: Any = None) -> int:
    """Serve MCP over the given streams until they close."""
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    _log(f"ready on stdio (protocol {PROTOCOL_VERSION})")
    for message in _iter_messages(stdin):
        try:
            reply = _handle(message)
        except Exception:
            _log("dispatch failed:\n" + traceback.format_exc())
            if "id" in message:
                reply = _error(message.get("id"), -32603, "internal error")
            else:
                reply = None
        if reply is None:
            continue
        stdout.write(json.dumps(reply, ensure_ascii=False) + "\n")
        stdout.flush()
    _log("input closed; exiting")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``lamet-agent mcp``."""
    return serve()
