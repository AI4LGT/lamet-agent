"""Backend-neutral messages and unified provider-backed LLM construction."""

from __future__ import annotations

import asyncio
import hashlib
import http.client
import ipaddress
import json
import os
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse

from jsonschema import Draft7Validator, FormatChecker, ValidationError


class InvalidResponseError(ValueError):
    """A received model response failed decoding or contract validation."""

    recovery_reason = "Your previous response could not be decoded or validated. "

    def __init__(self, message: str, raw_response: str = "") -> None:
        super().__init__(message)
        self.raw_response = raw_response


class MissingResponseError(InvalidResponseError):
    """A transient connection failure prevented receiving a complete response."""

    recovery_reason = "A connection failure or missing response prevented completion of the previous request. "


class RetryableBackendError(InvalidResponseError):
    """The provider rejected a request transiently; no response is available."""

    recovery_reason = "The provider temporarily rejected the previous request; no complete response was received. "


def _retryable_status(status: Any) -> bool:
    return isinstance(status, int) and (status in {408, 429} or 500 <= status < 600)


@contextmanager
def _receive_response():
    try:
        yield
    except urllib.error.HTTPError as exc:
        if _retryable_status(exc.code):
            raise RetryableBackendError(f"HTTP {exc.code}: no complete response received") from exc
        raise
    except http.client.InvalidURL:
        raise
    except (ConnectionError, TimeoutError, urllib.error.URLError, http.client.HTTPException) as exc:
        raise MissingResponseError(f"{type(exc).__name__}: {exc}") from exc


@contextmanager
def _decode_response(raw: Any):
    try:
        yield
    except (ValueError, TypeError, KeyError, IndexError, AttributeError, ValidationError) as exc:
        text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False, default=str)
        raise InvalidResponseError(f"{type(exc).__name__}: {exc}", text) from exc


_OPENAI_COMPATIBLE_API = {
    "openai": ("https://api.openai.com/v1/", "OPENAI_API_KEY"),
    "anthropic": ("https://api.anthropic.com/v1/", "ANTHROPIC_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GEMINI_API_KEY"),
    "grok": ("https://api.x.ai/v1", "GROK_API_KEY"),
    "deepseek": ("https://api.deepseek.com/", "DEEPSEEK_API_KEY"),
}

_AGENT_CLI = {"codex", "claude"}


@dataclass(frozen=True)
class _ResolvedProvider:
    """Internal provider selection derived from the public inputs."""

    kind: str
    provider: str
    model: str | None
    base_url: str | None = None
    key_env: str | None = None


def _is_local_url(value: str) -> bool:
    hostname = urlparse(value).hostname
    if hostname is None:
        return False
    if hostname == "localhost" or hostname.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return address.is_loopback or address.is_unspecified


def _resolve_provider(provider: str, model: str | None = None) -> _ResolvedProvider:
    """Resolve a registered CLI/API provider or a custom compatible URL."""
    name = provider.strip()
    selected_model = model.strip() if model and model.strip() else None
    if not name:
        raise ValueError("provider must not be empty")
    if name in _AGENT_CLI:
        return _ResolvedProvider("cli", name, selected_model)
    if name in _OPENAI_COMPATIBLE_API:
        base_url, key_env = _OPENAI_COMPATIBLE_API[name]
        return _ResolvedProvider("api", name, selected_model, base_url, key_env)
    parsed = urlparse(name)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return _ResolvedProvider("api", name, selected_model, name)
    registered = sorted([*_AGENT_CLI, *_OPENAI_COMPATIBLE_API])
    raise ValueError(f"unknown provider {name!r}; use one of {registered} or an HTTP(S) OpenAI-compatible API URL")


def _api_models(provider: _ResolvedProvider, api_key: str) -> list[str]:
    """Read the provider model catalog without selecting a default."""
    if provider.kind != "api" or provider.base_url is None:
        raise ValueError("model validation requires an API provider")
    request = urllib.request.Request(
        f"{provider.base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        response_context = (
            urllib.request.urlopen(request)
            if _is_local_url(provider.base_url)
            else urllib.request.urlopen(request, timeout=180)
        )
        with response_context as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to query available models from {provider.base_url!r}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError(f"{provider.base_url!r}/models returned an invalid response")
    available = sorted(
        item["id"]
        for item in payload["data"]
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    )
    if not available:
        raise ValueError(f"{provider.base_url!r}/models returned no model IDs")
    return available


def _cli_models(provider: str) -> list[str]:
    """Discover models from the installed CLI using its control protocol."""
    if provider == "codex":
        from openai_codex.client import CodexClient
        from openai_codex.generated.v2_all import ModelListResponse

        with CodexClient() as client:
            client.initialize()
            page = client.model_list()
            models = [item.model for item in page.data]
            while page.next_cursor:
                page = client.request(
                    "model/list", {"cursor": page.next_cursor, "includeHidden": False},
                    response_model=ModelListResponse,
                )
                models.extend(item.model for item in page.data)
            return models

    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    async def discover() -> list[str]:
        async with ClaudeSDKClient(options=ClaudeAgentOptions(tools=[], mcp_servers={})) as client:
            info = await client.get_server_info()
            return [
                value for item in (info or {}).get("models", [])
                for value in (item.get("value"), item.get("resolvedModel"))
                if isinstance(value, str) and value != "default"
            ]

    return asyncio.run(discover())


def _select_model(
    provider: _ResolvedProvider,
    available: list[str],
    selector: Callable[[str, list[str], str | None], str] | None,
) -> str:
    models = sorted({item for item in available if isinstance(item, str) and item.strip()})
    if not models:
        raise ValueError(f"{provider.provider!r} returned no model IDs")
    selected = provider.model
    if selected not in models and selector is not None:
        selected = selector(provider.provider, models, selected)
    if selected not in models:
        raise ValueError(
            f"model {selected!r} is not available from {provider.provider!r}; "
            f"available models: {', '.join(models)}"
        )
    return selected


@dataclass(frozen=True)
class _ToolCall:
    """One already-parsed model tool call."""

    id: str
    name: str
    arguments: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("tool call id must be a nonempty string")
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("tool call name must be a nonempty string")
        if not isinstance(self.arguments, Mapping):
            raise TypeError("tool call arguments must be an object")


@dataclass(frozen=True)
class Message:
    """Neutral transcript message."""

    role: str
    content: str
    tool_call_id: str | None = None
    tool_call: _ToolCall | None = None
    tool_calls: tuple[_ToolCall, ...] = ()

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role '{self.role}'")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")
        if not isinstance(self.tool_calls, tuple) or any(not isinstance(call, _ToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must be a tuple of internal tool-call values")
        if self.tool_call is None and len(self.tool_calls) == 1:
            object.__setattr__(self, "tool_call", self.tool_calls[0])
            object.__setattr__(self, "tool_calls", ())
        if self.tool_call is not None and self.tool_calls:
            raise ValueError("assistant messages cannot mix tool_call and tool_calls")
        if len({call.id for call in self.calls}) != len(self.calls):
            raise ValueError("tool call ids must be unique within one assistant message")
        if self.role in {"system", "user"} and (self.tool_call_id is not None or self.calls):
            raise ValueError("system/user messages cannot contain tool fields")
        if self.role == "assistant" and self.tool_call_id is not None:
            raise ValueError("assistant messages cannot contain tool_call_id")
        if self.role == "tool" and (not self.tool_call_id or self.calls):
            raise ValueError("tool messages require only tool_call_id")
        if self.role == "tool":
            try:
                observation = json.loads(self.content)
            except json.JSONDecodeError as exc:
                raise ValueError("tool message content must be canonical JSON") from exc
            if not isinstance(observation, dict):
                raise ValueError("tool message content must encode an object")
            canonical = json.dumps(observation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            if canonical != self.content:
                raise ValueError("tool message content must use canonical JSON")

    @property
    def calls(self) -> tuple[_ToolCall, ...]:
        """Return the assistant calls in provider order."""
        return self.tool_calls or ((self.tool_call,) if self.tool_call is not None else ())


@dataclass(frozen=True)
class _AssistantResponse:
    """One assistant turn with zero or more ordered tool calls."""

    text: str
    tool_call: _ToolCall | None = None
    tool_calls: tuple[_ToolCall, ...] = ()
    structured: Mapping[str, Any] | None = None
    usage: Mapping[str, int] | None = None
    raw_response: str | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("assistant text must be a string")
        if not isinstance(self.tool_calls, tuple) or any(not isinstance(call, _ToolCall) for call in self.tool_calls):
            raise TypeError("tool_calls must be a tuple of internal tool-call values")
        if self.structured is not None and not isinstance(self.structured, Mapping):
            raise TypeError("structured response must be an object")
        if self.usage is not None:
            if not isinstance(self.usage, Mapping) or any(
                not isinstance(key, str) or not isinstance(value, int) or isinstance(value, bool)
                for key, value in self.usage.items()
            ):
                raise TypeError("response usage must be a mapping of token names to integers")
        if self.tool_call is None and len(self.tool_calls) == 1:
            object.__setattr__(self, "tool_call", self.tool_calls[0])
            object.__setattr__(self, "tool_calls", ())
        if self.tool_call is not None and self.tool_calls:
            raise ValueError("assistant responses cannot mix tool_call and tool_calls")
        if self.structured is not None and self.calls:
            raise ValueError("assistant responses cannot mix structured output and tool calls")
        if len({call.id for call in self.calls}) != len(self.calls):
            raise ValueError("tool call ids must be unique within one assistant response")

    @property
    def calls(self) -> tuple[_ToolCall, ...]:
        """Return all calls while preserving the single-call test interface."""
        return self.tool_calls or ((self.tool_call,) if self.tool_call is not None else ())


def _cli_response(
    payload: Mapping[str, Any], turn: int, usage: Mapping[str, int] | None, raw_response: str,
) -> _AssistantResponse:
    """Convert the shared CLI wire envelope after schema decoding."""
    return _AssistantResponse(
        payload["text"],
        tool_calls=tuple(
            _ToolCall(f"turn-{turn}-{index}", call["name"], call["arguments"])
            for index, call in enumerate(payload["tool_calls"], start=1)
        ),
        usage=usage,
        raw_response=raw_response,
    )


class ResponseContract:
    """Validate decoded responses consistently, independently of provider encoding."""

    def __init__(self, tools: list[dict[str, Any]], response_schema: Mapping[str, Any] | None) -> None:
        if tools and response_schema is not None:
            raise ValueError("structured responses cannot be combined with tools")
        self.tools = {
            tool["function"]["name"]: self._validator(tool["function"]["parameters"])
            for tool in tools
        }
        self.structured = self._validator(response_schema["schema"]) if response_schema is not None else None

    @staticmethod
    def _validator(schema: Mapping[str, Any]) -> Draft7Validator:
        Draft7Validator.check_schema(schema)
        return Draft7Validator(schema, format_checker=FormatChecker())

    def validate(self, response: _AssistantResponse) -> None:
        raw = {
            "text": response.text,
            "tool_calls": [{"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
                           for call in response.calls],
            "structured": response.structured,
        }
        with _decode_response(response.raw_response if response.raw_response is not None else raw):
            if self.structured is not None:
                if response.structured is None or response.calls:
                    raise ValueError("Expected a structured response without tool calls")
                self.structured.validate(response.structured)
                return
            if response.structured is not None:
                raise ValueError("Unexpected structured response")
            if not response.calls and not response.text.strip():
                raise ValueError("The response contains neither text nor tool calls.")
            for call in response.calls:
                if call.name not in self.tools:
                    raise ValueError(f"model requested unavailable tool '{call.name}'")
                self.tools[call.name].validate(call.arguments)


class LlmBackend(Protocol):
    """Synchronous interface used by an agent workflow session."""

    identity: str

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]],
        prompt_digest: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> _AssistantResponse: ...


def _chat_message(message: Message) -> dict[str, Any]:
    if message.role in {"system", "user"}:
        return {"role": message.role, "content": message.content}
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    payload: dict[str, Any] = {"role": "assistant", "content": message.content}
    if message.calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(dict(call.arguments), separators=(",", ":"))},
            }
            for call in message.calls
        ]
    return payload


class _OpenAICompatibleBackend:
    """One explicit non-streaming OpenAI-compatible chat adapter."""

    def __init__(self, base_url: str, model: str, api_key: str) -> None:
        if not base_url or not model or not api_key:
            raise ValueError("base_url, model, and api_key are required")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.identity = f"openai-compatible:{self.base_url}:{self.model}"

    @staticmethod
    def _normalise_usage(value: Any) -> dict[str, int] | None:
        """Normalize OpenAI-style chat token usage."""
        if not isinstance(value, Mapping):
            return None
        aliases = {
            "prompt_tokens": "input_tokens",
            "completion_tokens": "output_tokens",
        }
        result = {
            aliases.get(str(key), str(key)): raw
            for key, raw in value.items()
            if aliases.get(str(key), str(key)) in {"input_tokens", "output_tokens", "total_tokens"}
            and isinstance(raw, int)
            and not isinstance(raw, bool)
        }
        prompt_details = value.get("prompt_tokens_details")
        if isinstance(prompt_details, Mapping):
            cached_tokens = prompt_details.get("cached_tokens")
            if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool):
                result["cached_input_tokens"] = cached_tokens
        completion_details = value.get("completion_tokens_details")
        if isinstance(completion_details, Mapping):
            reasoning_tokens = completion_details.get("reasoning_tokens")
            if isinstance(reasoning_tokens, int) and not isinstance(reasoning_tokens, bool):
                result["reasoning_output_tokens"] = reasoning_tokens
        return result or None

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]],
        prompt_digest: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> _AssistantResponse:
        provider_messages = [_chat_message(message) for message in messages]
        body = {
            "model": self.model,
            "messages": provider_messages,
            "stream": False,
        }
        if tools:
            body["tools"] = tools
            body["parallel_tool_calls"] = False
        if response_schema is not None:
            if tools:
                raise ValueError("structured responses cannot be combined with tools")
            provider_messages.append(
                {
                    "role": "user",
                    "content": (
                        "Return only a JSON object matching this schema exactly:\n"
                        + json.dumps(response_schema["schema"], separators=(",", ":"), ensure_ascii=False)
                    ),
                }
            )
            body["response_format"] = {"type": "json_object"}
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, separators=(",", ":")).encode("utf-8"),
            headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            raw_bytes = response.read()
        with _decode_response(raw_bytes.decode("utf-8", errors="replace")):
            raw = raw_bytes.decode("utf-8")
            payload = json.loads(raw)
            choice = payload["choices"][0]
            message = choice["message"]
            provider_calls = message.get("tool_calls") or []
            if not isinstance(provider_calls, list):
                raise ValueError("provider returned malformed tool_calls")
            calls = []
            for provider_call in provider_calls:
                if not isinstance(provider_call, dict) or not isinstance(provider_call.get("function"), dict):
                    raise ValueError("provider returned a malformed tool call")
                function = provider_call["function"]
                if (
                    not isinstance(provider_call.get("id"), str)
                    or not provider_call["id"]
                    or not isinstance(function.get("name"), str)
                    or not function["name"]
                    or not isinstance(function.get("arguments"), str)
                ):
                    raise ValueError("provider returned a malformed tool call")
                arguments = json.loads(function["arguments"])
                if not isinstance(arguments, dict):
                    raise TypeError("provider tool arguments must decode to an object")
                calls.append(_ToolCall(provider_call["id"], function["name"], arguments))
            text = message.get("content")
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise TypeError("provider response content must be a string")
            structured = None
            if response_schema is not None:
                structured = json.loads(text)
                if not isinstance(structured, dict):
                    raise TypeError("provider structured response must decode to an object")
            return _AssistantResponse(
                text,
                tool_calls=tuple(calls),
                structured=structured,
                usage=self._normalise_usage(payload.get("usage")),
                raw_response=raw,
            )


_CLI_LOCAL_CONSTRAINTS = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "format",
    "minItems",
    "maxItems",
    "uniqueItems",
}


@dataclass
class _CliSchema:
    schema: dict[str, Any]
    decode: Callable[[Any], Any]


def _compile_cli_schema(original: Mapping[str, Any], provider: str, path: str = "$") -> _CliSchema:
    schema = {key: original[key] for key in ("title", "description", "default") if key in original}
    deferred = {}
    for key in _CLI_LOCAL_CONSTRAINTS & original.keys():
        supported = (
            key not in {"minLength", "maxLength", "uniqueItems"}
            if provider == "codex"
            else key == "minItems" and original[key] in (0, 1)
        )
        if supported:
            schema[key] = original[key]
        else:
            deferred[key] = original[key]
    if deferred:
        schema["description"] = (
            schema.get("description", "") + " Application validates: " + json.dumps(deferred, sort_keys=True)
        ).strip()

    kind = original.get("type")
    if kind == "array" and original.get("maxItems") == 0:
        return _CliSchema(
            {"type": "null", "description": "Return null to represent the required empty array."},
            lambda value: [],
        )

    # Native scalars avoid double-encoding strings. Arbitrary containers need
    # an explicit wrapper because provider objects must have closed properties.
    untyped = not any(key in original for key in ("type", "anyOf", "enum", "const"))
    open_object = kind == "object" and original.get("additionalProperties", True) is not False
    complex_enum = any(isinstance(value, (dict, list)) for value in original.get("enum", []))
    if untyped or open_object or complex_enum:
        def decode_json(value: str) -> Any:
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid serialized JSON: {exc}") from exc

        if untyped:
            def decode_value(value: Any) -> Any:
                if not isinstance(value, dict):
                    return value
                restored = decode_json(value["json"])
                if not isinstance(restored, (dict, list)):
                    raise ValueError(f"{path}: the json wrapper must contain an object or array")
                return restored

            return _CliSchema(
                {"anyOf": [
                    {"type": "string"},
                    {"type": "number"},
                    {"type": "boolean"},
                    {"type": "null"},
                    {
                        "type": "object",
                        "properties": {"json": {
                            "type": "string",
                            "description": "Serialized JSON object or array.",
                        }},
                        "required": ["json"],
                        "additionalProperties": False,
                    },
                ]},
                decode_value,
            )
        return _CliSchema(
            {
                "type": "string",
                "description": "Return a JSON-encoded value (not Markdown) satisfying this application schema: "
                + json.dumps(original, sort_keys=True, ensure_ascii=False),
            },
            decode_json,
        )

    if "anyOf" in original:
        children = [_compile_cli_schema(child, provider, path) for child in original["anyOf"]]
        schema["anyOf"] = [child.schema for child in children]

        def decode_union(value: Any) -> Any:
            child = next(child for child in children if Draft7Validator(child.schema).is_valid(value))
            return child.decode(value)

        decode = decode_union
    elif kind == "object":
        properties = original.get("properties", {})
        required = original.get("required", [])
        children = {key: _compile_cli_schema(child, provider, f"{path}.{key}") for key, child in properties.items()}
        wire_properties = {key: child.schema for key, child in children.items()}
        optional = set(properties) - set(required) if provider == "codex" else set()
        for key in optional:
            # A wrapper distinguishes an omitted property (null) from a present
            # property whose actual value is null ({"value": null}).
            wire_properties[key] = {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"value": children[key].schema},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                    {"type": "null"},
                ],
                "description": (
                    "Return null to omit this optional field; otherwise put its value in the value property."
                ),
            }
        schema.update(
            type="object",
            properties=wire_properties,
            required=list(properties) if provider == "codex" else list(required),
            additionalProperties=False,
        )

        def decode_object(value: Any) -> Any:
            result = {}
            for key, item in value.items():
                if key in optional:
                    if item is None:
                        continue
                    item = item["value"]
                result[key] = children[key].decode(item)
            return result

        decode = decode_object
    elif kind == "array":
        child = _compile_cli_schema(original.get("items", {}), provider, f"{path}[]")
        schema.update(type="array", items=child.schema)

        def decode_array(value: Any) -> Any:
            return [child.decode(item) for item in value]

        decode = decode_array
    else:

        def decode_scalar(value: Any) -> Any:
            return value

        decode = decode_scalar

    for key in ("type", "enum", "const"):
        if key in original:
            schema[key] = original[key]
    if "type" not in schema and ("enum" in schema or "const" in schema):
        values = schema.get("enum", [schema.get("const")])
        kinds = {
            "null"
            if value is None
            else "boolean"
            if isinstance(value, bool)
            else "string"
            if isinstance(value, str)
            else "number"
            for value in values
        }
        schema["type"] = next(iter(kinds)) if len(kinds) == 1 else sorted(kinds)
    return _CliSchema(schema, decode)


def _prepare_cli_schema(
    tools: list[dict[str, Any]], response_schema: Mapping[str, Any] | None, *, provider: str
) -> _CliSchema:
    """Build an ask or tool response schema, adapt it, and prepare validated decoding."""
    if response_schema is not None:
        original = response_schema["schema"]
    else:
        variants = [
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "enum": [tool["function"]["name"]]},
                    "arguments": tool["function"]["parameters"],
                },
                "required": ["name", "arguments"],
                "additionalProperties": False,
            }
            for tool in tools
        ]
        calls: dict[str, Any] = {
            "type": "array",
            "items": {"anyOf": variants} if variants else {"type": "string"},
        }
        if not variants:
            calls["maxItems"] = 0
        original = {
            "type": "object",
            "properties": {"text": {"type": "string"}, "tool_calls": calls},
            "required": ["text", "tool_calls"],
            "additionalProperties": False,
        }
    node = _compile_cli_schema(original, provider)
    wire_validator = Draft7Validator(node.schema)
    original_validator = Draft7Validator(original, format_checker=FormatChecker())

    def decode(value: Any) -> Any:
        wire_validator.validate(value)
        restored = node.decode(value)
        original_validator.validate(restored)
        return restored

    return _CliSchema(node.schema, decode)


def _cli_task_input(
    *,
    transcript: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    prompt_digest: str,
    structured: bool,
) -> str:
    """Build the common prompt envelope for CLI-agent backends."""
    task = {
        "messages": transcript,
        "tools": tools,
        "prompt_digest": prompt_digest,
    }
    sections = [
        "<TASK_INPUT>",
        json.dumps(task, separators=(",", ":"), ensure_ascii=False),
        "</TASK_INPUT>",
    ]
    if not structured:
        sections.append(
            "Request application tools through tool_calls; the application executes them. Do not call tools yourself."
        )
    sections.extend(
        [
            "<EXECUTION_CONSTRAINT>",
            "Do not run shell commands. Do not edit files.",
            "</EXECUTION_CONSTRAINT>",
        ]
    )
    return "\n\n".join(sections)


_CODEX_LLM_CONFIG = {
    "project_doc_max_bytes": 0,
    "include_environment_context": False,
    "include_permissions_instructions": False,
    "include_apps_instructions": False,
    "include_collaboration_mode_instructions": False,
    "skills": {"include_instructions": False},
    "agents": {"enabled": False},
    "orchestrator": {"skills": {"enabled": False}, "mcp": {"enabled": False}},
    "features": {
        "apps": False,
        "goals": False,
        "hooks": False,
        "image_generation": False,
        "memories": False,
        "plugins": False,
        "shell_tool": False,
        "skill_search": False,
        "view_image": False,
    },
    "memories": {"use_memories": False},
    "web_search": "disabled",
}


class _CodexBackend:
    """Persistent per-job adapter for the installed openai-codex thread SDK."""

    def __init__(self, model: str) -> None:
        if not model or not model.strip():
            raise ValueError("model is required")
        self.model = model
        self.identity = f"codex:{model}"
        self._turn = 0
        self._codex = None
        self._threads: dict[str, Any] = {}

    @staticmethod
    def _thread_key(messages: list[Message], prompt_digest: str) -> str:
        """Identify one job conversation from its stable prompt and first request."""
        first_user = next((message.content for message in messages if message.role == "user"), "")
        payload = f"{prompt_digest}\0{first_user}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _normalise_usage(value: Any) -> dict[str, int] | None:
        """Normalize the Codex SDK's last-turn token usage."""
        if value is None:
            return None
        if hasattr(value, "model_dump"):
            value = value.model_dump(by_alias=False)
        elif not isinstance(value, Mapping) and hasattr(value, "__dict__"):
            value = vars(value)
        if not isinstance(value, Mapping):
            return None
        if isinstance(value.get("last"), Mapping):
            value = value["last"]
        aliases = {
            "inputTokens": "input_tokens",
            "outputTokens": "output_tokens",
            "totalTokens": "total_tokens",
            "cachedInputTokens": "cached_input_tokens",
            "reasoningOutputTokens": "reasoning_output_tokens",
            "cacheWriteInputTokens": "cache_write_input_tokens",
        }
        fields = {
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cached_input_tokens",
            "reasoning_output_tokens",
            "cache_write_input_tokens",
        }
        result = {
            aliases.get(str(key), str(key)): raw
            for key, raw in value.items()
            if aliases.get(str(key), str(key)) in fields and isinstance(raw, int) and not isinstance(raw, bool)
        }
        return result or None

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]],
        prompt_digest: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> _AssistantResponse:
        if response_schema is not None and tools:
            raise ValueError("structured responses cannot be combined with tools")
        output_contract = _prepare_cli_schema(tools, response_schema, provider="codex")
        try:
            import openai_codex as sdk
            from openai_codex import Codex, Sandbox  # type: ignore
        except ImportError as exc:
            raise RuntimeError("the codex provider requires the openai-codex package") from exc

        thread_key = self._thread_key(messages, prompt_digest)
        existing_thread = thread_key in self._threads
        transcript = []
        source_messages = messages[-1:] if existing_thread else messages
        for message in source_messages:
            if message.role == "system":
                continue
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if len(message.calls) == 1:
                call = message.calls[0]
                item["tool_call"] = {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
            elif message.calls:
                item["tool_calls"] = [
                    {"id": call.id, "name": call.name, "arguments": dict(call.arguments)} for call in message.calls
                ]
            transcript.append(item)
        task_input = _cli_task_input(
            transcript=transcript,
            tools=tools,
            prompt_digest=prompt_digest,
            structured=response_schema is not None,
        )
        try:
            if self._codex is None:
                self._codex = Codex()
            if not existing_thread:
                system_prompt = "\n\n".join(message.content for message in messages if message.role == "system")
                thread = self._codex.thread_start(
                    base_instructions=system_prompt,
                    config=_CODEX_LLM_CONFIG,
                    sandbox=Sandbox.read_only,
                    ephemeral=True,
                    model=self.model,
                )
                self._threads[thread_key] = thread
            else:
                thread = self._threads[thread_key]
            run_options: dict[str, Any] = {
                "sandbox": Sandbox.read_only,
                "output_schema": output_contract.schema,
            }
            result = thread.run(task_input, **run_options)
        except Exception as exc:
            if getattr(sdk, "is_retryable_error", lambda error: False)(exc):
                # A rejected turn may never have recorded its input. Replay the
                # full application history instead of sending only repair text.
                self._threads.pop(thread_key, None)
                raise RetryableBackendError(f"Codex request failed: {exc}") from exc
            if isinstance(exc, getattr(sdk, "CodexError", ())) and not isinstance(
                exc, getattr(sdk, "TransportClosedError", ())
            ):
                raise RuntimeError(f"Codex request failed: {exc}") from exc
            if not isinstance(exc, (ConnectionError, TimeoutError)) and not isinstance(
                exc, getattr(sdk, "TransportClosedError", ())
            ):
                raise
            # A closed transport cannot resume its cached threads. The next
            # attempt recreates the client using the full request history.
            client, self._codex = self._codex, None
            self._threads.clear()
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass  # Preserve the original connection failure.
            raise MissingResponseError(f"Codex connection failed: {exc}") from exc
        raw = result.final_response
        with _decode_response(raw):
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError("Codex returned no final response")
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Codex returned malformed JSON: {raw}") from exc

            payload = output_contract.decode(payload)
            self._turn += 1
            if response_schema is not None:
                if not isinstance(payload, dict):
                    raise TypeError("Codex structured response must decode to an object")
                return _AssistantResponse(
                    raw, structured=payload, usage=self._normalise_usage(result.usage), raw_response=raw,
                )

            return _cli_response(payload, self._turn, self._normalise_usage(result.usage), raw)

    def close(self) -> None:
        """Close the persistent Codex client after the owning agent run."""
        if self._codex is not None:
            self._codex.close()
            self._codex = None
            self._threads.clear()


class _ClaudeCodeBackend:
    """Per-job Claude Code adapter using the installed Python Agent SDK."""

    def __init__(self, model: str) -> None:
        if not model or not model.strip():
            raise ValueError("model is required")
        self.model = model
        self.identity = f"claude:{model}"
        self._turn = 0
        self._sessions: dict[str, str] = {}

    @staticmethod
    def _thread_key(messages: list[Message], prompt_digest: str) -> str:
        """Identify one job conversation from its stable prompt and first request."""
        first_user = next((message.content for message in messages if message.role == "user"), "")
        payload = f"{prompt_digest}\0{first_user}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _normalise_usage(value: Any) -> dict[str, int] | None:
        """Normalize Claude's disjoint uncached, cache-write, and cache-read inputs."""
        if value is None:
            return None
        if not isinstance(value, Mapping) and hasattr(value, "__dict__"):
            value = vars(value)
        if not isinstance(value, Mapping):
            return None
        aliases = {
            "inputTokens": "input_tokens",
            "outputTokens": "output_tokens",
            "cacheCreationInputTokens": "cache_creation_input_tokens",
            "cacheReadInputTokens": "cache_read_input_tokens",
        }
        usage = {
            aliases.get(str(key), str(key)): raw
            for key, raw in value.items()
            if aliases.get(str(key), str(key))
            in {"input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"}
            and isinstance(raw, int)
            and not isinstance(raw, bool)
        }
        uncached_tokens = max(usage.get("input_tokens", 0), 0)
        cache_write_tokens = max(usage.get("cache_creation_input_tokens", 0), 0)
        cached_tokens = max(usage.get("cache_read_input_tokens", 0), 0)
        input_tokens = uncached_tokens + cache_write_tokens + cached_tokens
        output_tokens = max(usage.get("output_tokens", 0), 0)
        return {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_tokens,
            "cache_write_input_tokens": cache_write_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def complete(
        self,
        *,
        messages: list[Message],
        tools: list[dict[str, Any]],
        prompt_digest: str,
        response_schema: Mapping[str, Any] | None = None,
    ) -> _AssistantResponse:
        if response_schema is not None and tools:
            raise ValueError("structured responses cannot be combined with tools")
        output_contract = _prepare_cli_schema(tools, response_schema, provider="claude")
        try:
            import claude_agent_sdk as sdk
            from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKError, query  # type: ignore
        except ImportError as exc:
            raise RuntimeError("the claude provider requires the claude-agent-sdk package") from exc

        thread_key = self._thread_key(messages, prompt_digest)
        session_id = self._sessions.get(thread_key)
        existing_session = session_id is not None
        source_messages = messages[-1:] if existing_session else messages
        transcript = []
        for message in source_messages:
            if message.role == "system":
                continue
            item: dict[str, Any] = {"role": message.role, "content": message.content}
            if message.tool_call_id:
                item["tool_call_id"] = message.tool_call_id
            if len(message.calls) == 1:
                call = message.calls[0]
                item["tool_call"] = {"id": call.id, "name": call.name, "arguments": dict(call.arguments)}
            elif message.calls:
                item["tool_calls"] = [
                    {"id": call.id, "name": call.name, "arguments": dict(call.arguments)} for call in message.calls
                ]
            transcript.append(item)
        task_input = _cli_task_input(
            transcript=transcript,
            tools=tools,
            prompt_digest=prompt_digest,
            structured=response_schema is not None,
        )
        options_values: dict[str, Any] = {
            "model": self.model,
            "tools": [],
            "skills": [],
            "permission_mode": "dontAsk",
            "strict_mcp_config": True,
            # StructuredOutput can require internal tool-use turns before the
            # final result. This budget is separate from application Plan turns.
            "max_turns": 3,
        }
        if not existing_session:
            system_prompt = "\n\n".join(message.content for message in messages if message.role == "system")
            options_values["system_prompt"] = system_prompt
        else:
            options_values["resume"] = session_id
        options_values["output_format"] = {
            "type": "json_schema",
            "schema": output_contract.schema,
        }
        options = ClaudeAgentOptions(**options_values)

        async def run_query() -> Any:
            final_message = None
            async for message in query(prompt=task_input, options=options):
                if hasattr(message, "result") and hasattr(message, "session_id"):
                    final_message = message
            return final_message

        try:
            result = asyncio.run(run_query())
        except ClaudeSDKError as exc:
            if isinstance(exc, getattr(sdk, "CLIConnectionError", ())) and not isinstance(
                exc, getattr(sdk, "CLINotFoundError", ())
            ):
                self._sessions.pop(thread_key, None)
                raise MissingResponseError(f"Claude Code connection failed: {exc}") from exc
            if isinstance(exc, getattr(sdk, "CLIJSONDecodeError", ())):
                raise InvalidResponseError(str(exc), getattr(exc, "line", "")) from exc
            status = getattr(exc, "api_error_status", None)
            if _retryable_status(status):
                self._sessions.pop(thread_key, None)
                raise RetryableBackendError(f"Claude Code request failed: {exc}") from exc
            raise RuntimeError(f"Claude Code request failed: {exc}") from exc
        if result is None:
            self._sessions.pop(thread_key, None)
            raise MissingResponseError("Claude Code returned no final response")
        if getattr(result, "is_error", False):
            errors = getattr(result, "errors", None)
            detail = "; ".join(str(error) for error in errors) if isinstance(errors, (list, tuple)) else None
            message = "Claude Code returned an error"
            if detail:
                message += f": {detail}"
            raise RuntimeError(message)
        returned_session_id = getattr(result, "session_id", None)
        if not isinstance(returned_session_id, str) or not returned_session_id:
            raise RuntimeError(f"Claude Code returned no session ID: {result}")
        self._sessions[thread_key] = returned_session_id

        usage = self._normalise_usage(getattr(result, "usage", None))
        payload = getattr(result, "structured_output", None)
        raw_payload = json.dumps(payload, ensure_ascii=False, default=str)
        with _decode_response(payload if payload is not None else getattr(result, "result", None)):
            if not isinstance(payload, Mapping):
                raise TypeError("Claude Code structured response must be an object")
            payload = output_contract.decode(payload)
            if response_schema is not None:
                raw = getattr(result, "result", None)
                text = raw if isinstance(raw, str) and raw.strip() else json.dumps(payload, ensure_ascii=False)
                self._turn += 1
                return _AssistantResponse(text, structured=payload, usage=usage, raw_response=raw_payload)

            self._turn += 1
            return _cli_response(payload, self._turn, usage, raw_payload)

    def close(self) -> None:
        """Forget in-process Claude Code session handles after the owning run."""
        self._sessions.clear()


def create_backend(
    provider: str,
    model: str | None = None,
    api_key_file: str | Path | None = None,
    *,
    select_model: Callable[[str, list[str], str | None], str] | None = None,
) -> LlmBackend:
    """Create the one public LLM interface from provider-neutral inputs.

    Registered API providers obtain their key from ``api_key_file`` or their
    registered environment variable. Custom OpenAI-compatible URLs require a
    key file. Provider-specific authentication, model discovery, and adapter
    selection stay internal to this module.
    """
    resolved = _resolve_provider(provider, model)
    if resolved.kind == "cli":
        if api_key_file is not None:
            raise ValueError("api_key_file is only valid for API providers")
        try:
            available = _cli_models(resolved.provider)
        except Exception as exc:
            raise RuntimeError(f"failed to query models from {resolved.provider!r}: {exc}") from exc
        selected = _select_model(resolved, available, select_model)
        if resolved.provider == "codex":
            return _CodexBackend(selected)
        return _ClaudeCodeBackend(selected)

    if api_key_file is not None:
        key_path = Path(api_key_file)
        if not key_path.is_file():
            raise ValueError(f"api_key_file {str(key_path)!r} does not exist or is not a file")
        key = key_path.read_text(encoding="utf-8").strip()
        if not key:
            raise ValueError(f"api_key_file {str(key_path)!r} is empty")
    else:
        key = (os.environ.get(resolved.key_env, "") if resolved.key_env else "").strip()
    if not key:
        if resolved.key_env:
            raise ValueError(
                f"API provider {resolved.provider!r} requires api_key_file or "
                f"the {resolved.key_env} environment variable"
            )
        raise ValueError(f"API provider {resolved.provider!r} requires api_key_file")
    selected = _select_model(resolved, _api_models(resolved, key), select_model)
    return _OpenAICompatibleBackend(resolved.base_url or "", selected, key)


__all__ = [
    "Message",
    "LlmBackend",
    "create_backend",
]
