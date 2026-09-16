"""Unified terminal UI, logging, interaction, and progress events.

Purpose: route Plan and Run output through one interface with interactive and
plain-terminal renderers.
Inputs: framework log/progress events and optional Plan conversation state.
Outputs: terminal rendering, user answers, confirmations, and progress updates.
Example: ``session = create_session(backend, ui=TerminalUi())``.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion, PathCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.filters import is_done
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.lexers import SimpleLexer
from prompt_toolkit.layout import HSplit, Window
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output import ColorDepth
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.shortcuts import ProgressBar
from prompt_toolkit.shortcuts.progress_bar.formatters import (
    IterationsPerSecond,
    Text,
    TimeLeft,
    create_default_formatters,
)
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from .banner import BANNER

_COMMANDS = ("/show", "/issues", "/undo", "/edit", "/save", "/help", "/quit")
_ANSI_RESET = "\033[0m"
_STATUS_PREFIXES = {
    "attention": ("ATTENTION", "Execution failed"),
    "llm": ("LLM usage", "Reasoning"),
    "running": ("Executing", "Running"),
}
_SHIFT_ENTER_SEQUENCES = {
    "\x1b[27;2;13~",  # xterm modifyOtherKeys
    "\x1b[13;2u",  # CSI u / extended keyboard protocol
}
_MINECRAFT_GRASS = 70
_MINECRAFT_DIRT = 137
_MINECRAFT_STONE = 242
_MINECRAFT_FIRE = 196
_MINECRAFT_DIAMOND = 80
_MINECRAFT_GOLD = 227
_LIGHT_CONE_COLOR = _MINECRAFT_STONE
_LIGHT_CONE_QUARKS_COLOR = _MINECRAFT_FIRE
_LIGHT_CONE_BLOCKS = frozenset(("▀", "▄", "█"))
_LETTER_PIXELS = frozenset(("█", "▒"))
_ANSI_STYLES = {
    "attention": f"\033[38;5;{_MINECRAFT_FIRE}m",
    "llm": f"\033[38;5;{_MINECRAFT_DIAMOND}m",
    "running": f"\033[38;5;{_MINECRAFT_GRASS}m",
}

_CONVERSATION_KEY_BINDINGS = KeyBindings()


@_CONVERSATION_KEY_BINDINGS.add("enter")
def _submit_or_insert_newline(event) -> None:
    """Submit on Enter while preserving distinct Shift+Enter sequences."""
    key_data = event.key_sequence[-1].data
    if key_data in _SHIFT_ENTER_SEQUENCES:
        event.current_buffer.newline()
    else:
        event.current_buffer.validate_and_handle()


_FOOTER_STYLE = {
    "bottom-toolbar": "fg:default bg:default reverse",
    "bottom-toolbar.text": "fg:default bg:default reverse",
}

_PROGRESS_STYLE = Style.from_dict(
    {
        "": "#ffff5f",
        **_FOOTER_STYLE,
    }
)

_INPUT_STYLE = Style.from_dict(
    {
        **_FOOTER_STYLE,
        "user-label": "bold #5fd7d7",
        "user-input": "",
        "prompt-continuation": "#6c6c6c",
        "frame.border": "#6c6c6c",
        "accepted user-label": "nobold #808080",
        "accepted user-input": "#808080",
        "accepted prompt-continuation": "#808080",
    }
)


def _progress_formatters():
    """Insert iteration speed immediately before the default ETA block."""
    formatters = create_default_formatters()
    time_left = next(index for index, formatter in enumerate(formatters) if isinstance(formatter, TimeLeft))
    eta_label = time_left - 1
    return [
        *formatters[:eta_label],
        IterationsPerSecond(),
        Text(" it/s "),
        *formatters[eta_label:],
    ]


def _render_minecraft_banner(message: str) -> str:
    """Render Minecraft-inspired banner layers and color bands with ANSI-256 colors."""
    lines = message.splitlines()
    height = len(lines)
    if height == 0:
        return message
    rendered = []
    base_positions = getattr(message, "base_positions", frozenset())
    overlay_positions = getattr(message, "overlay_positions", frozenset())
    endpoint_positions = set()
    if overlay_positions:
        first_row = min(row for row, _ in overlay_positions)
        last_row = max(row for row, _ in overlay_positions)
        endpoint_positions.add(
            max(
                (position for position in overlay_positions if position[0] == first_row),
                key=lambda position: position[1],
            )
        )
        endpoint_positions.add(
            min(
                (position for position in overlay_positions if position[0] == last_row),
                key=lambda position: position[1],
            )
        )
    glyph_row = None
    previous_glyph_line = False
    for row, line in enumerate(lines):
        has_glyph_line = any(character in _LETTER_PIXELS for character in line)
        if has_glyph_line:
            glyph_row = glyph_row + 1 if previous_glyph_line else 0
        else:
            glyph_row = None
        parts = []
        for column, character in enumerate(line):
            if character == " ":
                parts.append(character)
                continue
            position = (row, column)
            in_light_cone = position in base_positions or position in overlay_positions
            if in_light_cone and character not in _LIGHT_CONE_BLOCKS:
                parts.append(f"\033[39m{character}")
                continue
            if position in overlay_positions:
                character_color = _MINECRAFT_DIAMOND if position in endpoint_positions else _LIGHT_CONE_QUARKS_COLOR
            elif position in base_positions:
                character_color = _LIGHT_CONE_COLOR
            elif glyph_row is not None and character in _LETTER_PIXELS:
                upper_section = glyph_row < 2
                if character == "▒":
                    character_color = _MINECRAFT_STONE
                else:
                    character_color = _MINECRAFT_GRASS if upper_section else _MINECRAFT_DIRT
            else:
                parts.append(f"\033[39m{character}")
                continue
            parts.append(f"\033[38;5;{character_color}m{character}")
        rendered.append("".join(parts) + _ANSI_RESET)
        previous_glyph_line = has_glyph_line
    return "\n".join(rendered)


class UiCancelled(RuntimeError):
    """Raised when the user explicitly cancels an interactive workflow."""


@dataclass
class ProgressTask:
    """One renderer-neutral progress counter."""

    label: str
    total: int
    unit: str
    completed: int = 0
    native: Any | None = None


def _pointer_paths(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}/{str(key).replace('~', '~0').replace('/', '~1')}"
            yield path
            yield from _pointer_paths(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}/{index}"
            yield path
            yield from _pointer_paths(child, path)


class _ManifestCompleter(Completer):
    def __init__(self) -> None:
        self.state: Any | None = None
        self.path_completer = PathCompleter(expanduser=True)

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)
        if text.startswith("/show ") and self.state is not None:
            for path in _pointer_paths(self.state.candidate):
                if path.startswith(word):
                    yield Completion(path, start_position=-len(word))
            return
        if word.startswith((".", "~")) or "/" in word[1:]:
            yield from self.path_completer.get_completions(document, complete_event)
            return
        for command in _COMMANDS:
            if command.startswith(word):
                yield Completion(command, start_position=-len(word))


class PlainUi:
    """Non-full-screen UI for redirected output, tests, and simple terminals."""

    def show_banner(self) -> None:
        if not getattr(self, "_banner_shown", False):
            self.log(BANNER, style="banner")
            self.log()
            self._banner_shown = True

    def start(self) -> None:
        """Initialize the shared presentation once, independently of workflow phases."""
        self.show_banner()

    def set_phase(self, phase: str) -> None:
        """Update workflow status without rebuilding the UI."""
        self._phase = phase
        self.set_running_job(None, None)

    def log(self, message: str = "", *, level: str = "info", style: str | None = None) -> None:
        stream = sys.stderr if level == "error" else sys.stdout
        print(message, file=stream, flush=True)

    def warning(self, message: str) -> None:
        self.log(f"ATTENTION: {message}", level="warning", style="attention")

    def write(self, message: str) -> None:
        self.log(message)

    def manifest_updated(self, edits: list[dict[str, Any]], state: Any) -> None:
        self.log(
            f"Manifest updated ({len(edits)} field{'s' if len(edits) != 1 else ''}); "
            f"{len(state.issues)} validation issue{'s' if len(state.issues) != 1 else ''} remain."
        )

    def show_patch(self, edits: list[dict[str, Any]], state: Any) -> None:
        self.manifest_updated(edits, state)

    def ask(self, question: str, _state: Any | None = None, *, placeholder: str = "") -> str:
        self.log(f"\n{'Planner' if _state is not None else 'Agent'}: {question}\n")
        try:
            answer = input("You> ").strip()
        except (KeyboardInterrupt, EOFError) as exc:
            raise UiCancelled("interaction cancelled by user") from exc
        self.log()
        return answer

    def select_model(self, provider: str, models: list[str], requested: str | None = None) -> str:
        """Choose explicitly from a provider catalog in either terminal UI."""
        if requested is not None:
            self.log(f"Model {requested!r} is unavailable from {provider}.", level="warning")
        listing = "\n".join(f"  {index}. {model}" for index, model in enumerate(models, 1))
        while True:
            answer = self.ask(
                f"Available models for {provider}:\n{listing}\nChoose a number or model ID:",
                placeholder="Enter a model number or name…",
            )
            if answer in models:
                return answer
            if answer.isascii() and answer.isdigit() and 1 <= int(answer) <= len(models):
                return models[int(answer) - 1]
            self.log("Choose one of the listed models.", level="warning")

    def confirm(self, question: str) -> bool:
        try:
            return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}
        except (KeyboardInterrupt, EOFError) as exc:
            raise UiCancelled("interaction cancelled by user") from exc

    def ask_output_path(self, source: Path) -> str:
        directory = os.path.relpath(source.parent) + os.sep
        while True:
            answer = self.ask(
                "Output filename for the revised manifest?\n"
                f"Directory (fixed): {directory}\n"
                "Leave blank to save in place (requires overwrite confirmation).",
                placeholder="Enter an output filename…",
            ).strip()
            if answer and (Path(answer).name != answer or answer in {".", ".."}):
                self.log("Enter only a filename; the directory is fixed.", level="warning")
                continue
            return answer

    def review_manifest(self, state: Any, *, run_after: bool = False) -> bool | str | None:
        source, target = state.manifest_path, state.output_path
        output = target.name if target.parent == source.parent else str(target)
        if target == source:
            output += " (overwrites source)"
        elif target.exists():
            output += " (overwrites existing file)"
        question = f"● Manifest validated\n\n  {source.name}\n  Save as: {output}"
        return self.review_plan(question, state, accept_label="Save and run" if run_after else "Accept and save")

    def _plan_choice(self, question: str, accept_label: str) -> str:
        self.log(f"\n{question}\n")
        self.log(f"[y] {accept_label}  [N] Cancel  [?] Ask or revise\n")
        return input("Choose [y/N/?]: ")

    def review_plan(self, question: str, state: Any, *, accept_label: str = "Accept and save") -> bool | str | None:
        while True:
            try:
                answer = self._plan_choice(question, accept_label).strip().lower()
            except (KeyboardInterrupt, EOFError) as exc:
                raise UiCancelled("interaction cancelled by user") from exc
            if answer == "y":
                self.log()
                return True
            if answer in {"", "n"}:
                self.log()
                return None
            if answer == "?":
                while True:
                    request = self.ask(
                        "What would you like to know or change?", state,
                        placeholder="Ask a question or describe a change…",
                    )
                    if request.strip():
                        return request
            self.log("Please choose y, N, or ?.")

    def set_running_job(self, stage: str | None, job: str | None) -> None:
        pass

    def start_progress(self, label: str, *, total: int, unit: str) -> ProgressTask:
        return ProgressTask(label, total, unit)

    def advance_progress(self, task: ProgressTask, amount: int = 1) -> None:
        task.completed = min(task.total, task.completed + amount)

    def finish_progress(self, task: ProgressTask, *, success: bool = True) -> None:
        if success:
            task.completed = task.total

    def close(self) -> None:
        pass


class _InputSession(PromptSession):
    """Keep the composer compact while the footer stays at the terminal bottom."""

    def _create_layout(self):
        layout = super()._create_layout()
        # PromptSession's first section holds its input and completion menus.
        # Let a separate spacer absorb free rows, instead of stretching the frame.
        sections = layout.container.children
        composer = sections[0]

        def height():
            size = self.app.output.get_size()
            rows = composer.preferred_height(size.columns, size.rows).preferred
            return Dimension.exact(rows)

        sections[0] = HSplit([composer], height=height)
        sections.insert(1, Window())
        return layout


class TerminalUi(PlainUi):
    """Persistent prompt_toolkit conversation and progress renderer."""

    def __init__(self) -> None:
        self.completer = _ManifestCompleter()
        self.session = _InputSession(
            history=InMemoryHistory(),
            completer=self.completer,
            complete_while_typing=False,
            auto_suggest=AutoSuggestFromHistory(),
        )
        self._progress_bar: ProgressBar | None = None
        self._stdout_context: Any | None = None
        self._running_stage: str | None = None
        self._running_job: str | None = None
        self._phase = "startup"

    def log(self, message: str = "", *, level: str = "info", style: str | None = None) -> None:
        stream = sys.stderr if level == "error" else sys.stdout
        if style == "banner":
            print(_render_minecraft_banner(message), file=stream, flush=True)
            return
        color = _ANSI_STYLES.get(style or "")
        rendered = message
        if color:
            prefix = next(
                (candidate for candidate in _STATUS_PREFIXES.get(style or "", ()) if message.startswith(candidate)),
                None,
            )
            if prefix is not None:
                rendered = f"{color}{prefix}{_ANSI_RESET}{message[len(prefix):]}"
        print(rendered, file=stream, flush=True)

    def _status_toolbar(self) -> str:
        parts = [self._phase.upper()]
        if self._running_stage:
            parts.append(self._running_stage)
        if self._running_job:
            parts.append(self._running_job)
        activity = getattr(self, "_interaction", None) or {
            "startup": "Starting",
            "validate": "Validating",
            "plan": "Working",
            "run": "Running",
        }.get(self._phase, "Working")
        parts.append(activity)
        return " " + " · ".join(parts) + " "

    def _footer(self, hints: str = "") -> str:
        text = self._status_toolbar()
        if hints:
            text += " | " + hints.strip()
        text = " " + " ".join(text.split())
        width = max(1, self.session.app.output.get_size().columns - 1)
        if get_cwidth(text) <= width:
            return text
        clipped = ""
        used = 0
        for character in text:
            used += get_cwidth(character)
            if used > width - 1:
                break
            clipped += character
        return clipped + "…"

    def start(self) -> None:
        super().start()
        self._ensure_progress_bar()

    @contextmanager
    def _foreground(self) -> Iterator[None]:
        """Give input ownership to a prompt or editor, preserving live counters."""
        progress_bar = getattr(self, "_progress_bar", None)
        counters = list(progress_bar.counters) if progress_bar is not None else []
        if progress_bar is not None:
            self.close()
        try:
            yield
        finally:
            if progress_bar is not None:
                resumed = self._ensure_progress_bar()
                for counter in counters:
                    counter.progress_bar = resumed
                    resumed.counters.append(counter)
                resumed.invalidate()

    def _prompt(self, *args: Any, **kwargs: Any) -> str:
        hints = kwargs.pop("bottom_toolbar", " Enter confirm · Ctrl+C cancel ")
        self._interaction = kwargs.pop("interaction", "Waiting for input")
        try:
            with self._foreground():
                return self.session.prompt(
                    *args,
                    style=_INPUT_STYLE,
                    lexer=SimpleLexer("class:user-input"),
                    color_depth=ColorDepth.DEPTH_8_BIT,
                    show_frame=~is_done,
                    reserve_space_for_menu=0,
                    bottom_toolbar=lambda: self._footer(hints),
                    **kwargs,
                )
        finally:
            self._interaction = None

    def set_running_job(self, stage: str | None, job: str | None) -> None:
        self._running_stage = stage
        self._running_job = job
        if self._progress_bar is not None:
            self._progress_bar.app.invalidate()

    def _ensure_progress_bar(self) -> ProgressBar:
        if self._progress_bar is None:
            self._stdout_context = patch_stdout(raw=True)
            self._stdout_context.__enter__()
            self._progress_bar = ProgressBar(
                formatters=_progress_formatters(),
                bottom_toolbar=self._footer,
                style=_PROGRESS_STYLE,
                color_depth=ColorDepth.DEPTH_8_BIT,
            )
            self._progress_bar.__enter__()
        return self._progress_bar

    def start_progress(self, label: str, *, total: int, unit: str) -> ProgressTask:
        task = ProgressTask(label, total, unit)
        task.native = self._ensure_progress_bar()(label=f"{label} ({unit})", total=total, remove_when_done=True)
        return task

    def advance_progress(self, task: ProgressTask, amount: int = 1) -> None:
        remaining = min(amount, task.total - task.completed)
        task.completed += remaining
        if task.native is not None:
            for _ in range(remaining):
                task.native.item_completed()

    def finish_progress(self, task: ProgressTask, *, success: bool = True) -> None:
        if success:
            self.advance_progress(task, task.total - task.completed)
        if task.native is not None and not task.native.done:
            task.native.done = success
            task.native.stopped = True

    def _help(self) -> None:
        self.log(
            "Commands: /show [JSON pointer], /issues, /undo, /edit, /save, /help, /quit. "
            "Enter submits; Shift+Enter inserts a newline; Tab opens completion."
        )

    def _edit(self, state: Any) -> None:
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR") or "vi"
        descriptor, name = tempfile.mkstemp(suffix=".json", prefix="lamet-plan-")
        path = Path(name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(state.candidate, indent=2, ensure_ascii=False) + "\n")
            with self._foreground():
                subprocess.run([*shlex.split(editor), str(path)], check=True)
            document = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                raise ValueError("edited manifest root must be an object")
            state.replace_candidate(document, note="user external editor")
            self.log(f"Edited manifest loaded; {len(state.issues)} validation issues remain.")
        except (OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
            self.log(f"Editor changes were not applied: {exc}", level="error")
        finally:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _command(self, text: str, state: Any) -> bool:
        command, _, argument = text.partition(" ")
        if command == "/show":
            self.log(json.dumps(state.manifest_view(argument.strip()), indent=2, ensure_ascii=False))
        elif command == "/issues":
            if not state.issues:
                self.log("No validation issues remain.")
            for issue in state.issues:
                self.log(f"- {issue.path}: {issue.message}\n  {issue.physics}")
        elif command == "/undo":
            self.log(
                f"Previous manifest update undone; {len(state.issues)} validation issues remain."
                if state.undo()
                else "Nothing to undo."
            )
        elif command == "/edit":
            self._edit(state)
        elif command == "/save":
            self.log(f"Draft saved to {state.save()} with {len(state.issues)} validation issues remaining.")
        elif command == "/help":
            self._help()
        elif command == "/quit":
            raise UiCancelled("planning cancelled by user")
        else:
            return False
        return True

    def ask_output_path(self, source: Path) -> str:
        suggested = f"{source.stem}.planned.json"
        directory = os.path.relpath(source.parent) + os.sep
        self.log(
            "\n● Planner\n  Output filename for the revised manifest?\n"
            "  Clear the filename to save in place (requires overwrite confirmation).\n"
        )
        while True:
            try:
                answer = self._prompt(
                    [("class:user-label", " >  "), ("ansibrightblack", directory)],
                    multiline=False,
                    default=suggested,
                    placeholder=[("ansibrightblack", "Enter an output filename…")],
                    completer=PathCompleter(
                        get_paths=lambda: [str(source.parent)],
                        file_filter=lambda path: not Path(path).is_dir(),
                    ),
                    bottom_toolbar=" Enter submit · Tab complete · Ctrl+C cancel ",
                ).strip()
            except (KeyboardInterrupt, EOFError) as exc:
                raise UiCancelled("interaction cancelled by user") from exc
            if answer and (Path(answer).name != answer or answer in {".", ".."}):
                self.log("Enter only a filename; the directory is fixed.", level="warning")
                continue
            self.log()
            return answer

    def ask(self, question: str, state: Any | None = None, *, placeholder: str = "") -> str:
        role = "Planner" if state is not None or self._phase == "plan" else "Agent"
        self.log(f"\n● {role}\n" + "\n".join(f"  {line}" for line in question.splitlines()) + "\n")
        self.completer.state = state
        while True:
            try:
                answer = self._prompt(
                    HTML("<user-label> &gt; </user-label> "),
                    multiline=True,
                    completer=self.completer,
                    key_bindings=_CONVERSATION_KEY_BINDINGS,
                    prompt_continuation=[("class:prompt-continuation", "   ")],
                    placeholder=[("ansibrightblack", placeholder)],
                    bottom_toolbar=(
                        " Enter submit · Shift+Enter newline · /help "
                    ),
                ).strip()
            except (KeyboardInterrupt, EOFError) as exc:
                raise UiCancelled("interaction cancelled by user") from exc
            self.log()
            if not answer:
                continue
            if state is not None and answer.startswith("/") and self._command(answer, state):
                continue
            return answer

    def confirm(self, question: str) -> bool:
        while True:
            try:
                answer = self._prompt(
                    f"{question}\n\n[y] Yes  [N] No\n\n> ",
                    interaction="Waiting for confirmation",
                    multiline=False,
                    placeholder="",
                    completer=None,
                    auto_suggest=None,
                ).strip().lower()
            except (KeyboardInterrupt, EOFError) as exc:
                raise UiCancelled("interaction cancelled by user") from exc
            if answer in {"y", "yes"}:
                return True
            if answer in {"", "n", "no"}:
                return False
            self.log("Please answer yes or no.")

    def _plan_choice(self, question: str, accept_label: str) -> str:
        return self._prompt(
            f"{question}\n\n[y] {accept_label}  [N] Cancel  [?] Ask or revise\n\n> ",
            interaction="Waiting for confirmation",
            multiline=False,
            placeholder="",
            completer=None,
            auto_suggest=None,
            bottom_toolbar=" Enter confirm · Ctrl+C cancel ",
        )

    def close(self) -> None:
        if self._progress_bar is not None:
            self._progress_bar.__exit__(None, None, None)
            self._progress_bar = None
        if self._stdout_context is not None:
            self._stdout_context.__exit__(None, None, None)
            self._stdout_context = None


_ACTIVE_UI: ContextVar[PlainUi | None] = ContextVar("lamet_agent_ui", default=None)
_FALLBACK_UI = PlainUi()


def current_ui() -> PlainUi:
    return _ACTIVE_UI.get() or _FALLBACK_UI


@contextmanager
def use_ui(ui: PlainUi) -> Iterator[None]:
    token = _ACTIVE_UI.set(ui)
    try:
        yield
    finally:
        _ACTIVE_UI.reset(token)


def create_ui(*, interactive: bool | None = None) -> PlainUi:
    if interactive is None:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    return TerminalUi() if interactive else PlainUi()


def log(message: str = "", *, level: str = "info", style: str | None = None) -> None:
    if style is None and message.startswith("Running"):
        style = "running"
    current_ui().log(message, level=level, style=style)


def warning(message: str) -> None:
    current_ui().warning(message)


def track(iterable: Iterable[Any], *, label: str, unit: str, enabled: bool = True):
    """Yield an iterable while emitting renderer-neutral progress events."""
    if not enabled:
        yield from iterable
        return
    values = iterable if hasattr(iterable, "__len__") else list(iterable)
    task = current_ui().start_progress(label, total=len(values), unit=unit)
    success = False
    try:
        for value in values:
            yield value
            current_ui().advance_progress(task)
        success = True
    finally:
        current_ui().finish_progress(task, success=success)


__all__ = [
    "PlainUi",
    "ProgressTask",
    "TerminalUi",
    "UiCancelled",
    "create_ui",
    "current_ui",
    "log",
    "track",
    "use_ui",
    "warning",
]
