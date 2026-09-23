#!/usr/bin/env python3
"""Drive the lamet-agent MCP server over real pipes and assert the protocol.

Purpose: prove the stdio server actually works as an MCP server, rather than
trusting that it does. It spawns ``lamet-agent mcp`` as a subprocess, speaks
newline-delimited JSON-RPC 2.0 exactly as a client would, and asserts:

* stdout carries protocol messages ONLY (a stray banner or log line is a
  protocol violation that a real client rejects);
* the initialize handshake succeeds and advertises the tools capability;
* tools/list returns unique, naming-contract-compliant names;
* a real tools/call round-trips, including an error case.

The naming assertions mirror the DeepSeek Harness contract, which builds the
model-facing name as ``mcp__<serverName>__<rawName>`` where the whole thing must
be at most 64 characters of ``[A-Za-z0-9_-]``. A name that violates this is
silently rewritten with a hash suffix, so the check is worth enforcing here.

Usage:
    python3 tests/unit/test_mcp_server.py [--server-name lamet]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

MAX_PUBLIC_NAME = 64
ALLOWED_NAME_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-")

EXIT_OK = 0
EXIT_FAIL = 1


class Server:
    """A running ``lamet-agent mcp`` child, spoken to over its pipes."""

    def __init__(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.next_id = 0

    def send(self, method: str, params: dict | None = None, *, notification: bool = False) -> None:
        message: dict = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            message["params"] = params
        if not notification:
            self.next_id += 1
            message["id"] = self.next_id
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(message) + "\n")
        self.process.stdin.flush()

    def read(self) -> dict:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            stderr = self.process.stderr.read() if self.process.stderr else ""
            raise AssertionError(f"server closed stdout unexpectedly. stderr:\n{stderr}")
        return json.loads(line)

    def call(self, method: str, params: dict | None = None) -> dict:
        self.send(method, params)
        return self.read()

    def close(self) -> tuple[int, str]:
        if self.process.stdin:
            self.process.stdin.close()
        code = self.process.wait(timeout=30)
        stderr = self.process.stderr.read() if self.process.stderr else ""
        return code, stderr


def check(condition: bool, label: str, detail: str = "") -> bool:
    if condition:
        print(f"PASS    {label}")
        return True
    print(f"FAIL    {label}" + (f"\n          {detail}" if detail else ""))
    return False


# The CLI renders an ASCII-art banner built from box-drawing glyphs. Its presence
# in a tool result means the UI was not replaced, which also means the
# interactive input() path is still live -- the hang this server must prevent.
BANNER_GLYPHS = set("▄▀█▒╔╗╚╝═║▌│└┘┌┐")


def pollution(text: str) -> str:
    found = sorted(BANNER_GLYPHS & set(text))
    return "".join(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end protocol test for the lamet-agent MCP server.")
    parser.add_argument("--server-name", default="lamet", help="serverName the client will configure")
    parser.add_argument(
        "--command", default=None,
        help="override the launch command (default: 'lamet-agent mcp')",
    )
    args = parser.parse_args(argv)

    command = args.command.split() if args.command else ["lamet-agent", "mcp"]
    print(f"launching: {' '.join(command)}\n")
    server = Server(command)
    failures = 0

    try:
        # 1. initialize
        reply = server.call(
            "initialize",
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {},
                "clientInfo": {"name": "dsh-mcp-client", "version": "0.0.1"},
            },
        )
        result = reply.get("result", {})
        failures += not check("result" in reply, "initialize returns a result", json.dumps(reply)[:300])
        failures += not check(
            result.get("protocolVersion") in {
                "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05", "2024-10-07",
            },
            "initialize negotiates a supported protocol version",
            f"got {result.get('protocolVersion')!r}",
        )
        failures += not check(
            "tools" in (result.get("capabilities") or {}),
            "initialize advertises the tools capability",
            json.dumps(result.get("capabilities")),
        )
        failures += not check(
            isinstance(result.get("serverInfo", {}).get("name"), str),
            "initialize reports serverInfo.name",
        )

        # 2. initialized notification must not produce a reply
        server.send("notifications/initialized", notification=True)
        server.send("tools/list")
        listing = server.read()
        failures += not check("result" in listing, "tools/list answers after the initialized notification")

        tools = listing["result"]["tools"]
        failures += not check(len(tools) >= 1, "tools/list returns at least one tool")

        names = [t["name"] for t in tools]
        failures += not check(len(names) == len(set(names)), "tool names are unique", str(names))

        bad_chars = [
            n for n in names
            if not set(n).issubset(ALLOWED_NAME_CHARS)
        ]
        failures += not check(not bad_chars, "tool names use only [A-Za-z0-9_-]", str(bad_chars))

        too_long = [
            n for n in names
            if len(f"mcp__{args.server_name}__{n}") > MAX_PUBLIC_NAME
        ]
        failures += not check(
            not too_long,
            f"public names mcp__{args.server_name}__<tool> stay within {MAX_PUBLIC_NAME} chars",
            str(too_long),
        )

        for tool in tools:
            schema = tool.get("inputSchema") or {}
            if schema.get("type") != "object":
                failures += not check(False, f"{tool['name']} has an object inputSchema")
            if not isinstance(tool.get("description"), str) or not tool["description"]:
                failures += not check(False, f"{tool['name']} has a description")

        # A tool that requires task-based execution is listed but throws on call.
        tasky = [t["name"] for t in tools if (t.get("execution") or {}).get("taskSupport") == "required"]
        failures += not check(not tasky, "no tool requires unsupported task-based execution", str(tasky))

        # 3. a real tools/call
        call = server.call("tools/call", {"name": "validate_manifest", "arguments": {}})
        result = call.get("result", {})
        failures += not check(
            result.get("isError") is True,
            "a tool call with missing arguments reports isError instead of crashing",
            json.dumps(call)[:300],
        )
        content = result.get("content")
        failures += not check(
            isinstance(content, list) and content and content[0].get("type") == "text",
            "tool results use the content-block array shape",
            json.dumps(result)[:300],
        )

        # 4. unknown tool and unknown method are proper JSON-RPC errors
        unknown = server.call("tools/call", {"name": "no_such_tool", "arguments": {}})
        failures += not check("error" in unknown, "an unknown tool returns a JSON-RPC error")
        unknown_method = server.call("definitely/not/a/method")
        failures += not check(
            unknown_method.get("error", {}).get("code") == -32601,
            "an unknown method returns -32601",
            json.dumps(unknown_method)[:200],
        )

        # 5. validate a real manifest when one is available
        manifest = Path(__file__).resolve().parents[2] / "examples" / "pion_pdf_cg_manifest.json"
        if manifest.is_file():
            good = server.call(
                "tools/call",
                {"name": "validate_manifest", "arguments": {"manifest": str(manifest)}},
            )
            text = good["result"]["content"][0]["text"]
            failures += not check(
                good["result"].get("isError") is not True and text.startswith("VALID"),
                "validate_manifest reports a real manifest as VALID",
                text[:300],
            )
            # The UI must be replaced, not merely quiet: a banner here means the
            # interactive path is still installed.
            marks = pollution(text)
            failures += not check(
                not marks,
                "no banner or UI decoration leaks into the tool result",
                f"found glyphs {marks!r} in: {text[:200]!r}",
            )
            missing = server.call(
                "tools/call",
                {"name": "validate_manifest", "arguments": {"manifest": str(manifest) + ".nope"}},
            )
            failures += not check(
                missing["result"].get("isError") is True,
                "validate_manifest reports a missing manifest as an error",
            )
            # A directory is an operator error; it must produce a diagnosable
            # message, not a bare exit code buried under decorative output.
            directory = server.call(
                "tools/call",
                {"name": "validate_manifest", "arguments": {"manifest": str(manifest.parent)}},
            )
            directory_text = directory["result"]["content"][0]["text"]
            failures += not check(
                directory["result"].get("isError") is True and not pollution(directory_text),
                "a directory passed as a manifest errors cleanly, with no banner",
                directory_text[:300],
            )
        else:
            print(f"SKIP    no bundled manifest at {manifest}")

    finally:
        code, stderr = server.close()
        failures += not check(code == 0, f"server exits cleanly on EOF (exit {code})", stderr[-400:])
        print()
        print("--- server stderr (diagnostics belong here) ---")
        print(stderr.strip() or "(empty)")

    print()
    if failures:
        print(f"{failures} check(s) failed.")
        return EXIT_FAIL
    print("All MCP protocol checks passed.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
