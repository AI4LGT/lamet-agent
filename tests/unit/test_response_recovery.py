"""Response repair stays in the original conversation and never executes bad output."""

import json
import http.client
import urllib.error
from io import BytesIO
import sys
from types import ModuleType, SimpleNamespace

import pytest

from lamet_agent.agent import LlmSession, _run_conversation
from lamet_agent.llm import InvalidResponseError, Message, _AssistantResponse, _OpenAICompatibleBackend, create_backend


class Backend:
    identity = "test:recovery"

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def complete(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_repair_keeps_original_request_context_schema_and_history(tmp_path):
    schema = {"schema": {"type": "object"}}
    backend = Backend([
        InvalidResponseError("JSON ended early", '{"value":'),
        _AssistantResponse("ok", structured={"value": 1}),
        _AssistantResponse("next"),
    ])
    session = LlmSession(backend, tmp_path / "llm.md", history=[Message("system", "rules")])
    session.add_context("data", {"x": 1})
    session.complete(label="ask", user_message="original question", response_schema=schema)

    first, retry = backend.requests
    assert first["messages"] == retry["messages"][:-1]
    feedback = json.loads(retry["messages"][-1].content)
    assert feedback["error"] == "JSON ended early"
    assert feedback["invalid_response"] == '{"value":'
    assert first["prompt_digest"] == retry["prompt_digest"]
    assert retry["response_schema"] is schema
    assert session.calls == 2
    assert session.recommendation_calls == 1
    assert not session._pending_context
    assert "rejected response" in (tmp_path / "llm.md").read_text()
    session.complete(label="next", user_message="follow-up")
    assert backend.requests[-1]["messages"][:3] == retry["messages"]


@pytest.mark.parametrize("error", [
    ValueError("invalid configuration"),
    http.client.InvalidURL("invalid URL"),
    urllib.error.HTTPError("https://example.test", 401, "Unauthorized", {}, None),
    urllib.error.HTTPError("https://example.test", 403, "Forbidden", {}, None),
])
def test_configuration_and_authorization_errors_are_not_retried(error):
    backend = Backend([error])
    with pytest.raises(type(error), match=str(error)):
        LlmSession(backend, None).complete(label="ask", user_message="question")
    assert len(backend.requests) == 1


@pytest.mark.parametrize("error", [
    ConnectionError("connection lost"), TimeoutError("timed out"),
    urllib.error.URLError("temporary DNS failure"), http.client.IncompleteRead(b"partial"),
    urllib.error.HTTPError("https://example.test", 503, "Unavailable", {}, None),
    urllib.error.HTTPError("https://example.test", 429, "Too many requests", {}, None),
])
def test_missing_response_is_reported_to_model_before_resending(error):
    backend = Backend([error, _AssistantResponse("ready")])
    session = LlmSession(backend, None)
    assert session.complete(label="ask", user_message="original request").text == "ready"
    feedback = json.loads(backend.requests[1]["messages"][-1].content)
    assert "previous request" in feedback["request"]
    assert feedback["invalid_response"] == ""
    assert backend.requests[1]["messages"][0].content == "original request"
    assert session.calls == 2


def test_repeated_connection_errors_stop_after_three_attempts():
    backend = Backend([ConnectionError("offline")] * 3)
    session = LlmSession(backend, None)
    with pytest.raises(InvalidResponseError, match="after 3 attempts"):
        session.complete(label="ask", user_message="question")
    assert session.calls == 3


def test_repeated_bad_responses_stop_after_three_attempts():
    backend = Backend([InvalidResponseError("bad JSON", "{")] * 3)
    session = LlmSession(backend, None)
    with pytest.raises(InvalidResponseError, match="after 3 attempts"):
        session.complete(label="ask", user_message="question")
    assert session.calls == 3
    assert session.history == []


def test_empty_response_is_reported_before_resending():
    backend = Backend([_AssistantResponse(""), _AssistantResponse("ready")])
    assert LlmSession(backend, None).complete(label="ask", user_message="question").text == "ready"
    assert "neither text nor tool calls" in backend.requests[1]["messages"][-1].content


@pytest.mark.parametrize("bad_content", ['{"value":', '{"value":"wrong type"}'])
def test_api_structured_reply_is_repaired_before_returning(monkeypatch, bad_content):
    contents = iter([bad_content, '{"value":7}'])
    requests = []

    def urlopen(request):
        requests.append(json.loads(request.data))
        return BytesIO(json.dumps({"choices": [{"message": {"content": next(contents)}}]}).encode())

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    backend = _OpenAICompatibleBackend("https://example.test/v1", "model", "unused")
    session = LlmSession(backend, None)
    response = session.complete(
        label="ask", user_message="Choose a value.", response_schema={
            "schema": {"type": "object", "properties": {"value": {"type": "integer"}}, "required": ["value"]},
        },
    )
    assert response.structured == {"value": 7}
    feedback = json.loads(requests[1]["messages"][-2]["content"])
    assert bad_content in json.loads(feedback["invalid_response"])["choices"][0]["message"]["content"]
    assert "Error" in feedback["error"]
    assert requests[0]["response_format"] == requests[1]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("max_turns, attempts", [(2, 2), (None, 3)])
def test_repair_respects_remaining_conversation_budget_and_executes_no_tools(max_turns, attempts):
    backend = Backend([InvalidResponseError("bad tool arguments", "{")] * 3)
    session = LlmSession(backend, None)
    executed = []
    with pytest.raises(InvalidResponseError, match=f"after {attempts} attempts"):
        _run_conversation(
            session=session, messages=[Message("user", "question")], tool_schemas=[], tool_names=set(),
            prompt_digest="digest", label="plan", invoke_call=lambda *args: executed.append(args),
            handle_text=lambda text: None, is_terminal=lambda: False, max_turns=max_turns, max_tool_steps=5,
        )
    assert session.calls == attempts
    assert executed == []


@pytest.mark.parametrize("provider", ["codex", "claude"])
def test_cli_response_repair_reuses_native_conversation(provider, monkeypatch):
    prompts, options, starts = [], [], []
    payload = {"text": "ready", "tool_calls": None}
    outputs = iter([{"text": 42, "tool_calls": None}, payload])

    class Thread:
        def run(self, prompt, **kwargs):
            prompts.append(prompt)
            return SimpleNamespace(final_response=json.dumps(next(outputs)), usage=None)

    class Codex:
        def thread_start(self, **kwargs):
            starts.append(kwargs)
            return Thread()

    class Options:
        def __init__(self, **kwargs):
            options.append(kwargs)

    async def query(*, prompt, options):
        prompts.append(prompt)
        yield SimpleNamespace(
            result="ignored", structured_output=next(outputs), session_id="same-session", is_error=False, usage=None,
        )

    sdk = ModuleType("openai_codex" if provider == "codex" else "claude_agent_sdk")
    if provider == "codex":
        sdk.Codex = Codex
        sdk.Sandbox = SimpleNamespace(read_only="read-only")
    else:
        sdk.ClaudeAgentOptions = Options
        sdk.ClaudeSDKError = type("ClaudeSDKError", (Exception,), {})
        sdk.query = query
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    session = LlmSession(create_backend(provider, "test-model"), None)
    assert session.complete(label="plan", user_message="question").text == "ready"
    assert session.calls == 2
    assert "ValidationError" in prompts[1]
    assert "Resend the complete response" in prompts[1]
    if provider == "codex":
        assert len(starts) == 1
    else:
        assert options[1]["resume"] == "same-session"


def test_codex_closed_transport_rebuilds_client_with_original_history(monkeypatch):
    starts, prompts, closed = [], [], []

    class TransportClosedError(Exception):
        pass

    class Thread:
        def run(self, prompt, **kwargs):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise TransportClosedError("connection closed")
            return SimpleNamespace(final_response='{"text":"ready","tool_calls":null}', usage=None)

    class Codex:
        def thread_start(self, **kwargs):
            starts.append(kwargs)
            return Thread()

        def close(self):
            closed.append(True)

    sdk = ModuleType("openai_codex")
    sdk.Codex, sdk.Sandbox = Codex, SimpleNamespace(read_only="read-only")
    sdk.TransportClosedError = TransportClosedError
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    session = LlmSession(create_backend("codex", "gpt-5.6-luna"), None, history=[Message("system", "original rules")])
    assert session.complete(label="plan", user_message="original question").text == "ready"
    assert len(starts) == 2
    assert starts[1]["base_instructions"] == "original rules"
    assert "original question" in prompts[1]
    assert "connection failure" in prompts[1]
    assert closed == [True]


@pytest.mark.parametrize("missing_cli", [False, True])
def test_claude_connection_failure_retries_but_missing_installation_does_not(monkeypatch, missing_cli):
    class ClaudeSDKError(Exception):
        pass

    class CLIConnectionError(ClaudeSDKError):
        pass

    class CLINotFoundError(CLIConnectionError):
        pass

    prompts = []

    async def query(*, prompt, options):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise CLINotFoundError("missing CLI") if missing_cli else CLIConnectionError("disconnected")
        yield SimpleNamespace(
            result="ready", structured_output={"text": "ready", "tool_calls": None},
            session_id="session", is_error=False, usage=None,
        )

    sdk = ModuleType("claude_agent_sdk")
    sdk.ClaudeSDKError, sdk.CLIConnectionError, sdk.CLINotFoundError = (
        ClaudeSDKError, CLIConnectionError, CLINotFoundError
    )
    sdk.ClaudeAgentOptions, sdk.query = lambda **kwargs: SimpleNamespace(**kwargs), query
    monkeypatch.setitem(sys.modules, sdk.__name__, sdk)
    session = LlmSession(create_backend("claude", "haiku"), None)
    if missing_cli:
        with pytest.raises(RuntimeError, match="missing CLI"):
            session.complete(label="plan", user_message="question")
        assert len(prompts) == 1
    else:
        assert session.complete(label="plan", user_message="question").text == "ready"
        assert "connection failure" in prompts[1]


@pytest.mark.parametrize('bad_name,bad_arguments', [('fit', {'window': 'bad'}), ('unknown', {'window': 3})])
def test_api_tool_contract_errors_are_repaired_before_execution(monkeypatch, bad_name, bad_arguments):
    requests = []
    outputs = iter([(bad_name, bad_arguments), ('fit', {'window': 3})])

    def urlopen(request):
        requests.append(json.loads(request.data))
        name, arguments = next(outputs)
        return BytesIO(json.dumps({'choices': [{'message': {'content': '', 'tool_calls': [{
            'id': 'call', 'function': {'name': name, 'arguments': json.dumps(arguments)},
        }]}}]}).encode())

    monkeypatch.setattr('urllib.request.urlopen', urlopen)
    tools = [{'type': 'function', 'function': {'name': 'fit', 'parameters': {
        'type': 'object', 'properties': {'window': {'type': 'integer'}}, 'required': ['window'],
    }}}]
    executed = []
    session = LlmSession(_OpenAICompatibleBackend('https://example.test', 'model', 'unused'), None)
    _run_conversation(
        session=session, messages=[Message('user', 'fit')], tool_schemas=tools, tool_names={'fit'},
        prompt_digest='digest', label='plan',
        invoke_call=lambda name, arguments: executed.append(arguments) or {'ok': True},
        handle_text=lambda text: None, is_terminal=lambda: bool(executed), max_turns=None, max_tool_steps=None,
    )
    assert executed == [{'window': 3}]
    assert session.calls == 2
    feedback = json.loads(requests[1]['messages'][-1]['content'])
    assert json.loads(feedback['invalid_response'])['choices'][0]['message']['tool_calls']


def test_whole_tool_batch_is_validated_before_any_execution():
    from lamet_agent.llm import _ToolCall

    backend = Backend([
        _AssistantResponse('', tool_calls=(
            _ToolCall('first', 'fit', {'window': 3}), _ToolCall('second', 'fit', {'window': 'bad'}),
        )),
        _AssistantResponse('', _ToolCall('fixed', 'fit', {'window': 7})),
    ])
    executed = []
    _run_conversation(
        session=LlmSession(backend, None), messages=[Message('user', 'fit')],
        tool_schemas=[{'type': 'function', 'function': {'name': 'fit', 'parameters': {
            'type': 'object', 'properties': {'window': {'type': 'integer'}}, 'required': ['window'],
        }}}], tool_names={'fit'}, prompt_digest='digest', label='plan',
        invoke_call=lambda name, arguments: executed.append(arguments) or {'ok': True},
        handle_text=lambda text: None, is_terminal=lambda: bool(executed), max_turns=None, max_tool_steps=None,
    )
    assert executed == [{'window': 7}]


@pytest.mark.parametrize('retryable', [True, False])
def test_codex_sdk_classification_replays_busy_requests_and_stops_fatal(monkeypatch, retryable):
    sdk = pytest.importorskip('openai_codex')
    from openai_codex.errors import InvalidParamsError, ServerBusyError

    prompts, starts = [], []
    error = ServerBusyError(-32000, 'server overloaded') if retryable else InvalidParamsError(-32602, 'bad model')

    class Thread:
        def run(self, prompt, **kwargs):
            prompts.append(prompt)
            if len(prompts) == 1:
                raise error
            return SimpleNamespace(final_response='{"text":"ready","tool_calls":null}', usage=None)

    class Codex:
        def thread_start(self, **kwargs):
            starts.append(kwargs)
            return Thread()

    monkeypatch.setattr(sdk, 'Codex', Codex)
    session = LlmSession(create_backend('codex', 'test-model'), None)
    if retryable:
        assert session.complete(label='plan', user_message='question').text == 'ready'
        assert len(prompts) == 2
        assert len(starts) == 2
        assert json.loads(prompts[1].split('<TASK_INPUT>\n\n')[1].split('\n\n</TASK_INPUT>')[0])["messages"][0] == {
            "role": "user", "content": "question",
        }
    else:
        with pytest.raises(RuntimeError, match='bad model'):
            session.complete(label='plan', user_message='question')
        assert len(prompts) == 1


def test_fatal_failure_is_recorded_with_elapsed_time(tmp_path):
    path = tmp_path / 'transcript.md'
    session = LlmSession(Backend([ValueError('invalid configuration')]), path)
    with pytest.raises(ValueError, match='invalid configuration'):
        session.complete(label='plan', user_message='question')
    text = path.read_text()
    assert 'request failed' in text
    assert 'invalid configuration' in text
    assert 'elapsed_seconds' in text


def test_invalid_contract_fails_before_request_or_retry():
    backend = Backend([])
    from jsonschema import SchemaError

    with pytest.raises(SchemaError):
        LlmSession(backend, None).complete(
            label='plan', user_message='question', response_schema={'schema': {'type': 'not-a-json-type'}},
        )
    assert backend.requests == []


def test_tool_registry_mismatch_fails_before_request():
    backend = Backend([])
    with pytest.raises(ValueError, match='schemas must match'):
        _run_conversation(
            session=LlmSession(backend, None), messages=[Message('user', 'question')],
            tool_schemas=[], tool_names={'undeclared'}, prompt_digest='digest', label='plan',
            invoke_call=lambda *args: {}, handle_text=lambda text: None,
            is_terminal=lambda: False, max_turns=None, max_tool_steps=None,
        )
    assert backend.requests == []


@pytest.fixture(autouse=True)
def mock_cli_model_catalog(monkeypatch):
    # Response tests exercise transport behavior independently of model discovery.
    monkeypatch.setattr("lamet_agent.llm._cli_models", lambda provider: [
        "gpt-5.6-luna", "gpt-test", "test-model", "haiku", "sonnet",
    ])
