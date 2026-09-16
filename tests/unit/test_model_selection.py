"""One catalog selection policy shared by API and CLI providers."""
from types import SimpleNamespace

import pytest

from lamet_agent import llm
from lamet_agent.ui import PlainUi, UiCancelled


@pytest.mark.parametrize('requested', [None, '', 'missing'])
@pytest.mark.parametrize('provider', ['deepseek', 'codex', 'claude'])
def test_factory_selects_missing_or_unavailable_model(monkeypatch, provider, requested):
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test')
    monkeypatch.setattr(llm, '_api_models', lambda *args: ['b', 'a', 'a'])
    monkeypatch.setattr(llm, '_cli_models', lambda *args: ['b', 'a', 'a'])
    calls = []

    def select(name, models, previous):
        calls.append((name, models, previous))
        return 'b'

    backend = llm.create_backend(provider, requested, select_model=select)
    assert backend.identity.endswith(':b')
    assert calls == [(provider, ['a', 'b'], requested or None)]


def test_valid_model_does_not_prompt(monkeypatch):
    monkeypatch.setattr(llm, '_cli_models', lambda _: ['valid'])

    def unexpected(*args):
        pytest.fail('valid model must not prompt')

    assert llm.create_backend('codex', 'valid', select_model=unexpected).model == 'valid'


@pytest.mark.parametrize('models', [[], ['only']])
def test_no_automatic_choice_without_ui(monkeypatch, models):
    monkeypatch.setattr(llm, '_cli_models', lambda _: models)
    with pytest.raises(ValueError):
        llm.create_backend('codex')


def test_ui_retries_empty_and_invalid_choices(monkeypatch):
    ui = PlainUi()
    answers = iter(['', 'unknown', '0', '3', '2'])
    monkeypatch.setattr(ui, 'ask', lambda _, **kwargs: next(answers))
    assert ui.select_model('test', ['a', 'b'], 'old') == 'b'


def test_ui_accepts_exact_name_and_cancellation(monkeypatch):
    ui = PlainUi()
    monkeypatch.setattr(ui, 'ask', lambda _, **kwargs: 'b')
    assert ui.select_model('test', ['a', 'b']) == 'b'

    def cancel(_, **kwargs):
        raise UiCancelled('cancelled')

    monkeypatch.setattr(ui, 'ask', cancel)
    with pytest.raises(UiCancelled):
        ui.select_model('test', ['a'])


def test_codex_catalog_pagination_and_cleanup(monkeypatch):
    import openai_codex.client
    events = []

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            events.append('closed')

        def initialize(self):
            events.append('initialized')

        def model_list(self):
            return SimpleNamespace(data=[SimpleNamespace(model='a')], next_cursor='next')

        def request(self, method, params, **kwargs):
            assert method == 'model/list'
            assert params == {'cursor': 'next', 'includeHidden': False}
            return SimpleNamespace(data=[SimpleNamespace(model='b')], next_cursor=None)

    monkeypatch.setattr(openai_codex.client, 'CodexClient', Client)
    assert llm._cli_models('codex') == ['a', 'b']
    assert events == ['initialized', 'closed']


def test_claude_catalog_aliases_and_cleanup(monkeypatch):
    import claude_agent_sdk
    events = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            events.append('closed')

        async def get_server_info(self):
            return {'models': [
                {'value': 'default', 'resolvedModel': 'model-a'},
                {'value': 'alias', 'resolvedModel': 'model-b'},
            ]}

    monkeypatch.setattr(claude_agent_sdk, 'ClaudeSDKClient', Client)
    assert llm._cli_models('claude') == ['model-a', 'alias', 'model-b']
    assert events == ['closed']
