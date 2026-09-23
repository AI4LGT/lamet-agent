"""Run the LaMET Agent CLI.

Purpose: validate, plan, or execute one JSON manifest.
Inputs: a manifest path plus command-specific output/provider options.
Outputs: deterministic issue text, a completed manifest, or a JSON run summary.
Example: ``python -m lamet_agent validate examples/pion_pdf_gi_manifest.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .agent import create_session
from .contract import Issue
from .llm import create_backend
from .manifest import Manifest, load_manifest
from .ui import PlainUi, create_ui, use_ui


def _render_issues(issues: Sequence[Issue]) -> str:
    """Render validation issues for CLI output."""
    return "\n".join(f"{issue.path}: {issue.message} ({issue.physics})" for issue in issues)


def _validate(path: Path) -> tuple[Manifest, list[Issue]]:
    manifest = load_manifest(path)
    return manifest, manifest.validate()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lamet-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate one JSON manifest")
    validate.add_argument("manifest", type=Path)
    plan = subparsers.add_parser("plan", help="complete an incomplete manifest through an interactive LLM TUI")
    plan.add_argument("manifest", type=Path)
    plan.add_argument("--provider", required=True, help="registered provider or OpenAI-compatible API URL")
    plan.add_argument("--model", help="model ID; prompts for selection if omitted or unavailable")
    plan.add_argument("--api-key-file", type=Path, help="API key file for API providers")
    plan.add_argument(
        "--output", type=Path, help="output path; prompts for a filename if neither output nor in-place is set"
    )
    plan.add_argument("--in-place", action="store_true", help="overwrite the input manifest after explicit acceptance")
    run = subparsers.add_parser("run", help="execute one validated manifest")
    run.add_argument("manifest", type=Path)
    run.add_argument(
        "--provider", required=True, help="registered agent CLI/API provider, or an OpenAI-compatible API URL"
    )
    run.add_argument("--model", help="model ID; prompts for selection if omitted or unavailable")
    run.add_argument(
        "--api-key-file", type=Path, help="API key file; required for a custom URL, optional for registered APIs"
    )
    run.add_argument(
        "--progress",
        choices=("auto", "stage", "job", "none"),
        default="auto",
        help="progress granularity; auto uses stage progress when systematics are declared, otherwise job progress",
    )
    subparsers.add_parser("mcp", help="serve this tool over the Model Context Protocol on stdio")
    from .mcp_install import add_arguments as _add_install_arguments

    install = subparsers.add_parser(
        "install-mcp", help="register this tool's MCP server with local agent harnesses"
    )
    _add_install_arguments(install)
    for command in (plan, run):
        command.add_argument(
            "--plan-log-dir", type=Path,
            help="save Plan LLM transcripts in this directory (disabled by default)",
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch one CLI command and return its process status."""
    arguments = list(sys.argv[1:] if argv is None else argv)

    # Commands that own their own argument surface are dispatched before the
    # top-level parser sees their options. The parser still declares them so
    # they appear in `--help`, but letting it parse them would make its own
    # `-h` shadow the subcommand's richer help screen.
    if arguments and arguments[0] in ("mcp", "install-mcp"):
        if arguments[0] == "mcp":
            # The MCP server owns stdout as its protocol channel, so it must not
            # run inside the UI context below (whose banner would corrupt it).
            from .mcp_server import main as mcp_main

            return mcp_main()
        from .mcp_install import main as install_main

        return install_main(arguments[1:])

    args = _build_parser().parse_args(arguments)
    cli_ui = create_ui()
    with use_ui(cli_ui):
        try:
            cli_ui.start()
            return _dispatch(args, cli_ui)
        finally:
            cli_ui.close()


def _dispatch(args: argparse.Namespace, cli_ui: PlainUi) -> int:
    if args.command == "validate":
        cli_ui.set_phase("validate")
        try:
            _, issues = _validate(args.manifest)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            cli_ui.log(str(exc), level="error")
            return 2
        if issues:
            cli_ui.log(_render_issues(issues))
            return 1
        cli_ui.log("manifest is valid")
        return 0
    if args.command == "plan":
        backend = None
        session = None
        try:
            backend = create_backend(args.provider, args.model, args.api_key_file, select_model=cli_ui.select_model)
            session = create_session(backend, ui=cli_ui)
            planned_path = session.plan_manifest(
                args.manifest,
                output_path=args.output,
                in_place=args.in_place,
                plan_log_dir=args.plan_log_dir,
            )
            if planned_path is None:
                return 1
            manifest = load_manifest(planned_path)
            issues = manifest.validate()
            if issues:
                raise ValueError("accepted plan is not valid:\n" + _render_issues(issues))
            session.ui.log(f"manifest written: {planned_path}")
            return 0
        except (EOFError, OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            (session.ui if session is not None else cli_ui).log(str(exc), level="error")
            return 2
        finally:
            if session is not None:
                session.close()
            else:
                close_backend = getattr(backend, "close", None)
                if callable(close_backend):
                    close_backend()
    backend = None
    session = None
    try:
        manifest = load_manifest(args.manifest)
        backend = create_backend(args.provider, args.model, args.api_key_file, select_model=cli_ui.select_model)
        session = create_session(backend, ui=cli_ui, progress_mode=args.progress)
        issues = session.validate_manifest(manifest)
        if issues:
            planned_path = session.plan_manifest(args.manifest, run_after=True, plan_log_dir=args.plan_log_dir)
            if planned_path is None:
                return 1
            manifest = load_manifest(planned_path)
            planned_issues = manifest.validate()
            if planned_issues:
                raise ValueError("accepted plan is not valid:\n" + _render_issues(planned_issues))
        result = session.run_manifest(manifest)
        session.ui.log(json.dumps({"status": "completed", "jobs": sorted(result["summaries"])}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        (session.ui if session is not None else cli_ui).log(str(exc), level="error")
        return 1
    finally:
        if session is not None:
            session.close()
        else:
            close_backend = getattr(backend, "close", None)
            if callable(close_backend):
                close_backend()


if __name__ == "__main__":
    raise SystemExit(main())
