#!/usr/bin/env python3
"""Prove that MCP tool calls FAIL FAST instead of blocking on stdin.

A stdio MCP server keeps stdin open as its JSON-RPC channel. lamet-agent's
interactive paths call ``input()``, which on live-but-empty stdin blocks
forever. A tool that hangs is worse than a tool that fails, because the client
sees only a stall: the DeepSeek Harness reports an opaque timeout and the
session wedges.

This test drives the server with a live stdin (never closed, never fed
anything but protocol messages) and asserts that every path that *could* prompt
returns an error result within a bound.
"""

from __future__ import annotations

import json
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "examples" / "pion_pdf_cg_manifest.json"

FAIL_FAST_SECONDS = 120

EXIT_OK = 0
EXIT_FAIL = 1


class Timeout(Exception):
    pass


def _alarm(signum: int, frame: object) -> None:
    raise Timeout()


class Server:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "lamet_agent", "mcp"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(REPO),
        )
        self.id = 0

    def rpc(self, method: str, params: dict | None = None, *, notify: bool = False) -> dict:
        message: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notify:
            self.id += 1
            message["id"] = self.id
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        if notify:
            return {}
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            raise AssertionError("server closed stdout")
        return json.loads(line)

    def call(self, tool: str, arguments: dict, *, budget: float) -> tuple[dict, float]:
        signal.signal(signal.SIGALRM, _alarm)
        signal.alarm(int(budget))
        started = time.time()
        try:
            reply = self.rpc("tools/call", {"name": tool, "arguments": arguments})
        finally:
            signal.alarm(0)
        return reply.get("result", {}), time.time() - started

    def close(self) -> int:
        if self.proc.stdin:
            self.proc.stdin.close()
        return self.proc.wait(timeout=30)


def main() -> int:
    if not MANIFEST.is_file():
        print(f"cannot run: no manifest at {MANIFEST}")
        return EXIT_FAIL

    server = Server()
    failures = 0
    try:
        server.rpc(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "dsh-mcp-client", "version": "0.0.1"},
            },
        )
        server.rpc("notifications/initialized", notify=True)

        # Each case must produce an error result rather than blocking. NOTE the
        # server's stdin stays OPEN and empty throughout: that is exactly the
        # condition under which a stray input() would hang forever.
        cases = (
            ("run without provider/model", "run_manifest", {"manifest": str(MANIFEST)}),
            (
                "plan without output",
                "plan_manifest",
                {"manifest": str(MANIFEST), "provider": "codex", "model": "some-model"},
            ),
            (
                "run with unavailable manifest",
                "run_manifest",
                {"manifest": str(MANIFEST) + ".missing", "provider": "codex", "model": "m", "progress": "none"},
            ),
            (
                "validate a directory instead of a file",
                "validate_manifest",
                {"manifest": str(REPO / "examples")},
            ),
        )

        for label, tool, arguments in cases:
            try:
                result, elapsed = server.call(tool, arguments, budget=FAIL_FAST_SECONDS)
            except Timeout:
                print(f"FAIL    {label}: BLOCKED for more than {FAIL_FAST_SECONDS}s (the hang this guards against)")
                failures += 1
                continue
            text = (result.get("content") or [{}])[0].get("text", "")
            first = text.strip().splitlines()[0][:100] if text.strip() else "(empty)"
            ok = result.get("isError") is True
            print(f"{'PASS' if ok else 'FAIL'}    {label}: isError={result.get('isError')} in {elapsed:.1f}s")
            print(f"          {first}")
            if not ok:
                failures += 1

        # A read-only call must still succeed with live stdin.
        result, elapsed = server.call(
            "validate_manifest", {"manifest": str(MANIFEST)}, budget=FAIL_FAST_SECONDS
        )
        text = (result.get("content") or [{}])[0].get("text", "")
        ok = result.get("isError") is not True and text.startswith("VALID")
        print(f"{'PASS' if ok else 'FAIL'}    validate succeeds with live stdin in {elapsed:.1f}s")
        if not ok:
            failures += 1

    finally:
        code = server.close()
        print(f"\nserver exit after EOF: {code}")
        if code != 0:
            failures += 1
            if server.proc.stderr:
                print(server.proc.stderr.read()[-800:])

    print()
    if failures:
        print(f"{failures} check(s) failed.")
        return EXIT_FAIL
    print("No tool call blocked. Fail-fast behaviour holds.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
