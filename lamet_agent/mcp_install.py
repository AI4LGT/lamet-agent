"""Register this package's built-in MCP server with local agent harnesses.

Purpose: ``lamet-agent mcp`` serves the CLI over the Model Context Protocol, but
no harness discovers it on its own -- each one reads a server list from its own
config file. This module writes that one entry, for Codex, Claude Code, and the
DeepSeek Harness, so the command is a single ``lamet-agent install-mcp``.

Design constraints:

* **Additive and idempotent.** Running twice must not duplicate an entry, and an
  unavailable harness is reported rather than treated as a failure.
* **Never destroy user configuration.** Codex's config is free-form TOML and the
  DSH patch may carry hand-written comments, so an existing entry this tool did
  not write is reported instead of rewritten.
* **Absolute launch path.** Harnesses spawn the server with a scrubbed
  environment, so a bare ``lamet-agent`` resolved through ``PATH`` is not
  reliable. The interpreter running this command is the authority.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Sole anchor for the DeepSeek Harness patch block, which is always replaced to EOF.
BEGIN_MARKER = "# >>> lamet-agent mcp (managed by `lamet-agent install-mcp`) >>>"

DEFAULT_SERVER_NAME = "lamet"

# A numerical analysis comfortably exceeds the 60s default used by MCP bridges.
DEFAULT_TOOL_CALL_TIMEOUT_MS = 3_600_000


@dataclass(frozen=True)
class LaunchCommand:
    """The argv a harness should use to start this package's MCP server."""

    command: str
    args: tuple[str, ...]

    def as_list(self) -> list[str]:
        return [self.command, *self.args]


@dataclass
class TargetPlan:
    """What the installer intends to do for one harness."""

    name: str
    supported: bool
    path: Path | None = None
    action: Literal["install", "update", "present", "skip", "conflict"] = "skip"
    detail: str = ""


@dataclass
class TargetResult:
    name: str
    status: Literal["installed", "updated", "present", "skipped", "failed", "conflict"]
    detail: str = ""


def resolve_launch_command() -> LaunchCommand:
    """Prefer the console script beside the running interpreter.

    Harnesses spawn the server without inheriting a useful ``PATH``, so the
    absolute console script is used when it exists and ``python -m`` otherwise.
    Both are absolute, which is what matters.
    """
    executable = Path(sys.executable)
    script = executable.with_name("lamet-agent")
    if script.is_file():
        return LaunchCommand(str(script), ("mcp",))
    return LaunchCommand(str(executable), ("-m", "lamet_agent", "mcp"))


# --------------------------------------------------------------------------
# Codex: ~/.codex/config.toml
# --------------------------------------------------------------------------


def codex_config_path() -> Path:
    return Path.home() / ".codex" / "config.toml"



def _toml_table_span(text: str, table: str) -> tuple[int, int] | None:
    """Locate a TOML table's extent: (header start, start of the next table).

    Codex rewrites ``config.toml`` itself, which drops comments. Anchoring on a
    marker comment therefore cannot work; the table header is the stable anchor,
    and its extent ends at the next top-level ``[...]`` header.
    """
    header = re.compile(rf"^\[{re.escape(table)}\]\s*$", re.MULTILINE)
    match = header.search(text)
    if match is None:
        return None
    following = re.compile(r"^\[", re.MULTILINE)
    nxt = following.search(text, match.end())
    return match.start(), (nxt.start() if nxt else len(text))


def codex_block(name: str, launch: LaunchCommand) -> str:
    args = ", ".join(json.dumps(arg) for arg in launch.args)
    return (
        f"[mcp_servers.{name}]\n"
        f"command = {json.dumps(launch.command)}\n"
        f"args = [{args}]\n"
        f"startup_timeout_sec = 60\n"
    )


def plan_codex(name: str, launch: LaunchCommand) -> TargetPlan:
    path = codex_config_path()
    if not path.is_file():
        return TargetPlan("codex", supported=False, path=path, detail=f"no Codex config at {path}")
    text = path.read_text(encoding="utf-8")
    server_table = f"mcp_servers.{name}"
    if _toml_table_span(text, server_table) is not None:
        # Ours or the user's -- either way the entry exists and the apply step
        # replaces exactly this table, leaving every other table untouched.
        return TargetPlan("codex", True, path, "update", f"refresh [{server_table}] in {path}")
    return TargetPlan("codex", True, path, "install", f"append [{server_table}] to {path}")


MCP_FEATURE_FLAG = "mcp_2026_07_28"


def _codex_feature_state(executable: str, flag: str) -> bool | None:
    """Return whether a Codex feature flag is on, or None if it cannot be read."""
    try:
        completed = subprocess.run(
            [executable, "features", "list"], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError):
        return None
    for line in (completed.stdout or "").splitlines():
        parts = line.split()
        if parts and parts[0] == flag:
            return parts[-1].lower() == "true"
    return None


def _enable_codex_feature(executable: str, flag: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            [executable, "features", "enable", flag], capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"could not run `codex features enable {flag}`: {exc}"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip().splitlines()
        return False, detail[-1] if detail else f"exit {completed.returncode}"
    return True, f"enabled Codex feature {flag!r}"


def apply_codex(plan: TargetPlan, name: str, launch: LaunchCommand) -> TargetResult:
    if plan.action == "conflict":
        return TargetResult("codex", "conflict", plan.detail)
    assert plan.path is not None
    text = plan.path.read_text(encoding="utf-8")
    span = _toml_table_span(text, f"mcp_servers.{name}")
    if span is None:
        text = text.rstrip("\n") + "\n\n" + codex_block(name, launch)
        action = "installed"
    else:
        start, end = span
        text = text[:start] + codex_block(name, launch) + text[end:]
        action = "updated"
    # Refuse to leave a config Codex cannot parse.
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        return TargetResult("codex", "failed", f"refusing to write; result would be invalid TOML: {exc}")
    servers = parsed.get("mcp_servers")
    if not isinstance(servers, dict) or name not in servers:
        return TargetResult("codex", "failed", f"refusing to write; [{name}] did not land under [mcp_servers]")
    plan.path.write_text(text, encoding="utf-8")

    # Registering the server is not enough: Codex 0.154 ships MCP tool exposure
    # behind an off-by-default feature flag, and with it off the server connects
    # and answers tools/list while the model still sees no tools at all. That
    # silent state is exactly the reported "I have no lamet tools".
    executable = shutil.which("codex")
    detail = str(plan.path)
    if executable is not None:
        state = _codex_feature_state(executable, MCP_FEATURE_FLAG)
        if state is True:
            detail = f"{plan.path} (MCP tool exposure already enabled)"
        elif state is False:
            enabled, message = _enable_codex_feature(executable, MCP_FEATURE_FLAG)
            detail = f"{plan.path}; {message}" if enabled else f"{plan.path}; WARNING: {message}"
        elif state is None:
            detail = (
                f"{plan.path}; WARNING: could not read Codex feature flags, so MCP tools may stay hidden. "
                f"Check with: codex features list | grep {MCP_FEATURE_FLAG}"
            )
    return TargetResult("codex", action, detail)


# --------------------------------------------------------------------------
# Claude Code: driven through its own CLI
# --------------------------------------------------------------------------


def plan_claude(name: str, launch: LaunchCommand) -> TargetPlan:
    executable = shutil.which("claude")
    if executable is None:
        return TargetPlan("claude", supported=False, detail="no `claude` executable on PATH")
    return TargetPlan(
        "claude",
        True,
        Path(executable),
        "install",
        f"claude mcp add -s user {name} -- {' '.join(launch.as_list())}",
    )


def _claude_listing(executable: str) -> str:
    try:
        completed = subprocess.run(
            [executable, "mcp", "list"], capture_output=True, text=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return completed.stdout or ""


def apply_claude(plan: TargetPlan, name: str, launch: LaunchCommand) -> TargetResult:
    executable = str(plan.path)
    listing = _claude_listing(executable)
    if f" {name}:" in listing or f"\n{name}:" in listing:
        # Already registered; re-adding from the same launch path is harmless, so
        # treat it as done rather than churning the config.
        return TargetResult("claude", "present", f"{name} already registered with Claude Code")
    try:
        subprocess.run(
            [executable, "mcp", "remove", "-s", "user", name],
            capture_output=True, text=True, timeout=60,
        )
        completed = subprocess.run(
            [executable, "mcp", "add", "-s", "user", name, "--", *launch.as_list()],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return TargetResult("claude", "failed", f"running the claude CLI failed: {exc}")
    output = (completed.stdout or completed.stderr or "").strip().splitlines()
    summary = output[-1] if output else f"exit {completed.returncode}"
    if completed.returncode != 0:
        return TargetResult("claude", "failed", summary)
    return TargetResult("claude", "installed", summary)


# --------------------------------------------------------------------------
# DeepSeek Harness: a cordis loader patch row
# --------------------------------------------------------------------------


def dsh_patch_candidates() -> list[Path]:
    """Every profile patch file that exists, most likely first."""
    home = Path.home()
    profiles = home / ".dsh" / "profiles"
    if not profiles.is_dir():
        return []
    return sorted(profiles.glob("*/cordis.patch.yml"))


def dsh_patch_path() -> Path:
    candidates = dsh_patch_candidates()
    web = Path.home() / ".dsh" / "profiles" / "web" / "cordis.patch.yml"
    if web in candidates:
        return web
    return candidates[0] if candidates else web


def _dsh_yaml() -> tuple[Any, Any]:
    """Return (module, loader) for reading a DeepSeek Harness patch.

    DSH patches routinely carry ``!!js`` expressions, which the stock safe
    loader rejects outright, so a profile that uses one could not be registered
    at all. The tag is resolved here as the scalar text it wraps: this command
    only needs the document's *structure* (to find an existing row), and the
    file itself is rewritten by text edit, so the expression is never evaluated
    and never reserialised.
    """
    import yaml

    class _PatchLoader(yaml.SafeLoader):
        pass

    def _js_scalar(loader: Any, node: Any) -> Any:
        if isinstance(node, yaml.ScalarNode):
            return loader.construct_scalar(node)
        if isinstance(node, yaml.SequenceNode):
            return loader.construct_sequence(node)
        return loader.construct_mapping(node)

    _PatchLoader.add_constructor("tag:yaml.org,2002:js", _js_scalar)
    _PatchLoader.add_constructor("!js", _js_scalar)
    return yaml, _PatchLoader


def _iter_dsh_rows(document: list[Any]) -> list[dict[str, Any]]:
    """Yield every loader row, including rows nested under an `insert:` key.

    Rows this tool writes live inside ``insert``, so a scan that only looks at
    top-level ``id`` keys finds nothing and appends a duplicate on every run.
    """
    rows: list[dict[str, Any]] = []
    for entry in document:
        if not isinstance(entry, dict):
            continue
        if isinstance(entry.get("insert"), list):
            rows.extend(item for item in entry["insert"] if isinstance(item, dict))
        elif "id" in entry:
            rows.append(entry)
    return rows


def dsh_block(name: str, launch: LaunchCommand) -> str:
    args = "\n".join(f"          - {json.dumps(arg)}" for arg in launch.args)
    return (
        f"{BEGIN_MARKER}\n"
        f"- insert:\n"
        f"    - id: mcp-{name}\n"
        f"      name: '@deepseek-ai/dsh-mcp-client'\n"
        f"      config:\n"
        f"        serverName: {name}\n"
        f"        transport: stdio\n"
        f"        command: {launch.command}\n"
        f"        args:\n{args}\n"
        f"        toolCallTimeoutMs: {DEFAULT_TOOL_CALL_TIMEOUT_MS}\n"
        f"        failOnStartupError: false\n"
    )


DSH_HEADER = (
    "# Managed by `lamet-agent install-mcp`.\n"
    "# The `insert:` wrapper is required: a bare row is treated as an override of an\n"
    "# existing entry, and @deepseek-ai/dsh-mcp-client is mounted by no bundle.\n"
)


def plan_dsh(name: str, launch: LaunchCommand) -> TargetPlan:
    candidates = dsh_patch_candidates()
    if not candidates:
        return TargetPlan(
            "dsh",
            supported=False,
            detail="no DeepSeek Harness profile found under ~/.dsh/profiles",
        )
    path = dsh_patch_path()
    try:
        yaml, loader = _dsh_yaml()
    except ModuleNotFoundError:
        return TargetPlan(
            "dsh", supported=False, path=path, detail="PyYAML is unavailable, so the patch cannot be edited safely"
        )
    if not path.is_file():
        return TargetPlan("dsh", True, path, "install", f"create {path}")
    text = path.read_text(encoding="utf-8")
    try:
        document = yaml.load(text, Loader=loader)
    except yaml.YAMLError as exc:
        return TargetPlan("dsh", True, path, "conflict", f"{path} is not parseable YAML: {exc}")
    if document is None:
        document = []
    if not isinstance(document, list):
        return TargetPlan("dsh", True, path, "conflict", f"{path} is not a YAML list of patch entries")
    entry_id = f"mcp-{name}"
    if any(row.get("id") == entry_id for row in _iter_dsh_rows(document)):
        if BEGIN_MARKER in text:
            return TargetPlan("dsh", True, path, "update", f"refresh the managed block in {path}")
        return TargetPlan(
            "dsh",
            True,
            path,
            "present",
            f"{entry_id} is already present in {path} and was not written by this command",
        )
    return TargetPlan("dsh", True, path, "install", f"insert {entry_id} into {path}")


def apply_dsh(plan: TargetPlan, name: str, launch: LaunchCommand) -> TargetResult:
    """Edit the patch by managed text block, never by reserialising the file.

    A YAML round-trip through ``safe_dump`` would preserve other rows' values
    but discard their comments and layout. The managed block is always the last
    thing in the file, so the single opening marker is enough to find and
    replace it: no closing marker means one less line for a host tool to strip.
    """
    yaml, loader = _dsh_yaml()

    assert plan.path is not None
    text = plan.path.read_text(encoding="utf-8") if plan.path.is_file() else ""

    if BEGIN_MARKER in text:
        text = text[: text.index(BEGIN_MARKER)] + dsh_block(name, launch)
        action = "updated"
    else:
        # plan_dsh already parsed the file and found no row for this server, so
        # no second validation is needed here.
        #
        # A stock profile patch is the empty document `[]`. Appending a block
        # after it would yield two YAML documents in one stream and fail to
        # parse, so the empty-list placeholder is removed first. Comments are
        # kept: they are the user's.
        prefix_lines = [
            line
            for line in text.splitlines(keepends=True)
            if line.strip() and not line.lstrip().startswith("#") and line.strip() != "[]"
        ]
        prefix = "".join(prefix_lines) if prefix_lines else DSH_HEADER
        if not prefix.endswith("\n"):
            prefix += "\n"
        text = prefix + dsh_block(name, launch)
        action = "installed"

    # Never leave a duplicate behind: verify the file still holds exactly one row.
    try:
        reparsed = yaml.load(text, Loader=loader)
        ids = [row.get("id") for row in _iter_dsh_rows(reparsed if isinstance(reparsed, list) else [])]
    except yaml.YAMLError as exc:
        return TargetResult("dsh", "failed", f"edited file is not valid YAML: {exc}")
    if ids.count(f"mcp-{name}") != 1:
        return TargetResult("dsh", "failed", f"refusing to write: found {ids.count(f'mcp-{name}')} rows for mcp-{name}")

    plan.path.parent.mkdir(parents=True, exist_ok=True)
    plan.path.write_text(text, encoding="utf-8")
    return TargetResult("dsh", action, str(plan.path))


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

_PLANNERS = {
    "codex": plan_codex,
    "claude": plan_claude,
    "dsh": plan_dsh,
}
_APPLIERS = {
    "codex": apply_codex,
    "claude": apply_claude,
    "dsh": apply_dsh,
}
TARGETS = ("codex", "claude", "dsh")


def install_mcp(
    *,
    targets: tuple[str, ...] = TARGETS,
    server_name: str = DEFAULT_SERVER_NAME,
    dry_run: bool = False,
) -> tuple[list[TargetPlan], list[TargetResult]]:
    launch = resolve_launch_command()
    plans: list[TargetPlan] = []
    results: list[TargetResult] = []
    for target in targets:
        planner = _PLANNERS[target]
        plan = planner(server_name, launch)
        plans.append(plan)
        if dry_run or not plan.supported:
            continue
        if plan.action in ("present", "conflict"):
            results.append(
                TargetResult(target, "present" if plan.action == "present" else "conflict", plan.detail)
            )
            continue
        try:
            results.append(_APPLIERS[target](plan, server_name, launch))
        except Exception as exc:  # reporting beats crashing mid-install
            results.append(TargetResult(target, "failed", f"{type(exc).__name__}: {exc}"))
    return plans, results


def add_arguments(parser: Any) -> None:
    """Declare this command's options on a parser.

    Shared with ``__main__`` so the top-level subparser advertises the same
    surface: ``lamet-agent install-mcp --help`` is answered by that parser, and
    a subparser declaring no options would print an empty help screen.
    """
    parser.add_argument(
        "--target",
        action="append",
        choices=(*TARGETS, "all"),
        default=[],
        help="harness to configure (repeatable); default is every detected harness",
    )
    parser.add_argument("--server-name", default=DEFAULT_SERVER_NAME, help="server name to register")
    parser.add_argument("--dry-run", action="store_true", help="report what would change and exit")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``lamet-agent install-mcp``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="lamet-agent install-mcp",
        description="Register this package's built-in MCP server with local agent harnesses.",
    )
    add_arguments(parser)
    args = parser.parse_args(argv)

    selected = tuple(t for t in (args.target or ["all"]) if t != "all")
    targets = selected or TARGETS
    launch = resolve_launch_command()

    print(f"MCP server : {args.server_name}")
    print(f"launch     : {' '.join(launch.as_list())}")
    print()
    plans, results = install_mcp(targets=targets, server_name=args.server_name, dry_run=args.dry_run)

    by_name = {result.name: result for result in results}
    for plan in plans:
        result = by_name.get(plan.name)
        if not plan.supported:
            print(f"  {plan.name:<7} skipped  {plan.detail}")
            continue
        if args.dry_run:
            print(f"  {plan.name:<7} would {plan.action}: {plan.detail}")
            continue
        if result is not None:
            print(f"  {plan.name:<7} {result.status:<9} {result.detail}")

    if args.dry_run:
        print("\nDry run: nothing was written.")
        return 0

    if not any(result.status == "failed" for result in results):
        print("\nRestart or reload a harness if the new tools do not appear:")
        print("  - Codex and the DeepSeek Harness read their server list at startup")
        print("  - Claude Code picks it up per session")
    else:
        print("\nSome harnesses could not be configured; see the failures above.")
    return 0


__all__ = ["add_arguments", "install_mcp", "main", "resolve_launch_command", "TARGETS"]
