"""Tests for GuardrailsOpenAI synchronous client behaviour."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import guardrails.client as client_module
import guardrails.context as guardrails_context
from guardrails._base_client import GuardrailsResponse
from guardrails.client import (
    GuardrailsAsyncAzureOpenAI,
    GuardrailsAzureOpenAI,
    GuardrailsOpenAI,
)
from guardrails.context import GuardrailsContext
from guardrails.exceptions import GuardrailTripwireTriggered
from guardrails.types import GuardrailResult


def _minimal_config() -> dict[str, Any]:
    """Return minimal pipeline config with no guardrails."""
    return {"version": 1, "output": {"version": 1, "guardrails": []}}


def _build_client(**kwargs: Any) -> GuardrailsOpenAI:
    """Instantiate GuardrailsOpenAI with deterministic defaults."""
    return GuardrailsOpenAI(config=_minimal_config(), **kwargs)


def _guardrail(name: str) -> Any:
    """Create a guardrail stub with a definition name."""
    return SimpleNamespace(definition=SimpleNamespace(name=name), ctx_requirements=SimpleNamespace())


@pytest.fixture(autouse=True)
def reset_context() -> None:
    guardrails_context.clear_context()
    yield
    guardrails_context.clear_context()


def test_default_context_uses_distinct_guardrail_client() -> None:
    """Default context should hold a fresh OpenAI instance mirroring config."""
    client = _build_client(api_key="secret-key", base_url="http://example.com")

    assert client.context is not None  # noqa: S101
    assert client.context.guardrail_llm is not client  # type: ignore[attr-defined]  # noqa: S101
    assert client.context.guardrail_llm.api_key == "secret-key"  # type: ignore[attr-defined]  # noqa: S101
    assert client.context.guardrail_llm.base_url == "http://example.com"  # type: ignore[attr-defined]  # noqa: S101


def test_conversation_context_exposes_history() -> None:
    """Conversation-aware context should surface conversation history only."""
    client = _build_client()
    conversation = [{"role": "user", "content": "Hello"}]

    conv_ctx = client._create_context_with_conversation(conversation)

    assert conv_ctx.get_conversation_history() == conversation  # noqa: S101
    assert not hasattr(conv_ctx, "update_injection_last_checked_index")  # noqa: S101


def test_create_default_context_uses_contextvar() -> None:
    """Existing context should be reused by derived client."""
    existing = GuardrailsContext(guardrail_llm="existing")
    guardrails_context.set_context(existing)
    try:
        client = _build_client()
        assert client._create_default_context() is existing  # noqa: S101
    finally:
        guardrails_context.clear_context()


def test_append_llm_response_handles_string_history() -> None:
    """String conversation history should be normalized before appending."""
    client = _build_client()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="assistant reply"))],
        output=None,
    )

    updated_history = client._append_llm_response_to_conversation("hi there", response)

    assert updated_history[0]["content"] == "hi there"  # noqa: S101
    assert updated_history[1].message.content == "assistant reply"  # type: ignore[union-attr]  # noqa: S101


def test_append_llm_response_handles_response_output() -> None:
    """Responses API output should be appended as-is."""
    client = _build_client()
    response = SimpleNamespace(
        choices=None,
        output=[{"role": "assistant", "content": "streamed"}],
    )

    updated_history = client._append_llm_response_to_conversation([], response)

    assert updated_history == [{"role": "assistant", "content": "streamed"}]  # noqa: S101


def test_append_llm_response_handles_none_history() -> None:
    """None conversation history should be converted to list."""
    client = _build_client()
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="assistant reply"))],
        output=None,
    )

    history = client._append_llm_response_to_conversation(None, response)

    assert history[-1].message.content == "assistant reply"  # type: ignore[union-attr]  # noqa: S101


def test_run_stage_guardrails_raises_on_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tripwire results should raise unless suppressed."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("basic guardrail")]
    captured_kwargs: dict[str, Any] = {}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        captured_kwargs.update(kwargs)
        return [GuardrailResult(tripwire_triggered=True)]

    monkeypatch.setattr("guardrails.client.run_guardrails", fake_run_guardrails)

    with pytest.raises(GuardrailTripwireTriggered):
        client._run_stage_guardrails("output", "payload")

    assert captured_kwargs["ctx"] is client.context  # noqa: S101
    assert captured_kwargs["stage_name"] == "output"  # noqa: S101


def test_run_stage_guardrails_uses_conversation_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt injection guardrail should trigger conversation-aware context."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("Prompt Injection Detection")]
    conversation = [{"role": "user", "content": "Hi"}]
    captured_kwargs: dict[str, Any] = {}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        captured_kwargs.update(kwargs)
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr("guardrails.client.run_guardrails", fake_run_guardrails)

    results = client._run_stage_guardrails("output", "payload", conversation_history=conversation)

    assert results == [GuardrailResult(tripwire_triggered=False)]  # noqa: S101
    ctx = captured_kwargs["ctx"]
    assert ctx.get_conversation_history() == conversation  # noqa: S101


def test_run_stage_guardrails_suppresses_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress flag should return results even when tripwire fires."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("basic guardrail")]
    result = GuardrailResult(tripwire_triggered=True)

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [result]

    monkeypatch.setattr("guardrails.client.run_guardrails", fake_run_guardrails)

    results = client._run_stage_guardrails("output", "payload", suppress_tripwire=True)

    assert results == [result]  # noqa: S101


def test_run_stage_guardrails_handles_empty_guardrails() -> None:
    """If no guardrails are configured for the stage, return empty list."""
    client = _build_client()
    client.guardrails["input"] = []

    assert client._run_stage_guardrails("input", "text") == []  # noqa: S101


def test_run_stage_guardrails_raises_on_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exceptions should propagate when raise_guardrail_errors is True."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("guard")]
    client.raise_guardrail_errors = True

    async def failing_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        raise RuntimeError("boom")

    monkeypatch.setattr(client_module, "run_guardrails", failing_run_guardrails)

    with pytest.raises(RuntimeError):
        client._run_stage_guardrails("output", "payload")


def test_run_stage_guardrails_creates_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """GuardrailsOpenAI should create a new loop when none is running."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("guard")]

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client_module, "run_guardrails", fake_run_guardrails)

    original_new_event_loop = asyncio.new_event_loop
    loops: list[asyncio.AbstractEventLoop] = []

    def fake_get_event_loop() -> asyncio.AbstractEventLoop:
        raise RuntimeError

    def fake_new_event_loop() -> asyncio.AbstractEventLoop:
        loop = original_new_event_loop()
        loops.append(loop)
        return loop

    monkeypatch.setattr(asyncio, "get_event_loop", fake_get_event_loop)
    monkeypatch.setattr(asyncio, "new_event_loop", fake_new_event_loop)
    monkeypatch.setattr(asyncio, "set_event_loop", lambda loop: None)

    try:
        result = client._run_stage_guardrails("output", "payload")
        assert result[0].tripwire_triggered is False  # noqa: S101
    finally:
        for loop in loops:
            loop.close()


def test_handle_llm_response_runs_output_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """_handle_llm_response should append conversation and return response wrapper."""
    client = _build_client()
    output_result = GuardrailResult(tripwire_triggered=False)
    captured_text: list[str] = []
    captured_history: list[list[Any]] = []

    def fake_run_stage(
        stage_name: str,
        text: str,
        conversation_history: list | None = None,
        suppress_tripwire: bool = False,
    ) -> list[GuardrailResult]:
        captured_text.append(text)
        if conversation_history is not None:
            captured_history.append(conversation_history)
        return [output_result]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]

    llm_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="LLM response"),
                delta=SimpleNamespace(content=None),
            )
        ],
        output_text=None,
    )

    response = client._handle_llm_response(
        llm_response,
        preflight_results=[GuardrailResult(tripwire_triggered=False)],
        input_results=[],
        conversation_history=[{"role": "user", "content": "hello"}],
    )

    assert captured_text == ["LLM response"]  # noqa: S101
    assert captured_history[-1][-1].message.content == "LLM response"  # type: ignore[index]  # noqa: S101
    assert response.guardrail_results.output == [output_result]  # noqa: S101


def test_handle_llm_response_suppresses_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress flag should return results even when output guardrail trips."""
    client = _build_client()

    def fake_run_stage(
        stage_name: str,
        text: str,
        conversation_history: list | None = None,
        suppress_tripwire: bool = False,
    ) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=True)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]

    response = client._handle_llm_response(
        llm_response=SimpleNamespace(output_text="value", choices=[]),
        preflight_results=[],
        input_results=[],
        conversation_history=[],
        suppress_tripwire=True,
    )

    assert response.guardrail_results.output[0].tripwire_triggered is True  # noqa: S101


def test_chat_completions_create_executes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat.completions.create should execute guardrail stages."""
    client = _build_client()
    client.guardrails = {"pre_flight": [_guardrail("Prompt Injection Detection")], "input": [_guardrail("Input")], "output": [_guardrail("Output")]}
    stages: list[str] = []

    def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        stages.append(stage_name)
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]
    monkeypatch.setattr(client, "_apply_preflight_modifications", lambda messages, results: messages)  # type: ignore[attr-defined]

    class _InlineExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def __enter__(self) -> _InlineExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def submit(self, fn, *args, **kwargs):
            class _ImmediateFuture:
                def __init__(self) -> None:
                    self._result = fn(*args, **kwargs)

                def result(self) -> Any:
                    return self._result

            return _ImmediateFuture()

    monkeypatch.setattr("guardrails.resources.chat.chat.ThreadPoolExecutor", _InlineExecutor)

    def fake_llm(**kwargs: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), delta=SimpleNamespace(content=None))],
            output_text=None,
        )

    client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=fake_llm))  # type: ignore[attr-defined]

    sentinel = object()

    def fake_handle_response(llm_response: Any, preflight_results: list[GuardrailResult], input_results: list[GuardrailResult], **kwargs: Any) -> Any:
        return sentinel

    monkeypatch.setattr(client, "_handle_llm_response", fake_handle_response)  # type: ignore[attr-defined]

    result = client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], model="gpt")

    assert "pre_flight" in stages and "input" in stages  # noqa: S101
    assert result is sentinel  # noqa: S101


def test_chat_completions_create_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming mode should use _stream_with_guardrails_sync."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [], "output": []}

    class _InlineExecutor:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _ = (args, kwargs)

        def __enter__(self) -> _InlineExecutor:
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

        def submit(self, fn, *args, **kwargs):
            class _ImmediateFuture:
                def __init__(self) -> None:
                    self._result = fn(*args, **kwargs)

                def result(self) -> Any:
                    return self._result

            return _ImmediateFuture()

    monkeypatch.setattr("guardrails.resources.chat.chat.ThreadPoolExecutor", _InlineExecutor)

    def fake_llm(**kwargs: Any) -> Any:
        return iter([SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="c"))])])

    client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=fake_llm))  # type: ignore[attr-defined]
    monkeypatch.setattr(client, "_stream_with_guardrails_sync", lambda *args, **kwargs: ["chunk"])  # type: ignore[attr-defined]

    result = client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], model="gpt", stream=True)

    assert result == ["chunk"]  # noqa: S101


def test_responses_create_executes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses.create should run stages and wrap response."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [_guardrail("Input")], "output": [_guardrail("Output")]}
    stages: list[str] = []

    def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        stages.append(stage_name)
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]
    monkeypatch.setattr(client, "_apply_preflight_modifications", lambda messages, results: messages)  # type: ignore[attr-defined]

    def fake_llm(**kwargs: Any) -> Any:
        return SimpleNamespace(output_text="text", choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client._resource_client.responses = SimpleNamespace(create=fake_llm)  # type: ignore[attr-defined]

    response = client.responses.create(input=[{"role": "user", "content": "hi"}], model="gpt")

    assert "input" in stages and "output" in stages  # noqa: S101
    assert isinstance(response, GuardrailsResponse)  # noqa: S101


def test_responses_parse_executes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses.parse should run guardrails and return wrapper."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [_guardrail("Input")], "output": []}

    def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]
    monkeypatch.setattr(client, "_apply_preflight_modifications", lambda messages, results: messages)  # type: ignore[attr-defined]

    def fake_parse(**kwargs: Any) -> Any:
        return SimpleNamespace(output_text="{}", output=[{"type": "message", "content": "parsed"}])

    client._resource_client.responses = SimpleNamespace(parse=fake_parse)  # type: ignore[attr-defined]

    sentinel = object()

    def fake_handle_parse(llm_response: Any, preflight_results: list[GuardrailResult], input_results: list[GuardrailResult], **kwargs: Any) -> Any:
        return sentinel

    monkeypatch.setattr(client, "_handle_llm_response", fake_handle_parse)  # type: ignore[attr-defined]

    response = client.responses.parse(input=[{"role": "user", "content": "hi"}], model="gpt", text_format=dict)

    assert response is sentinel  # noqa: S101


def test_responses_retrieve_executes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses.retrieve should run output guardrails."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [], "output": [_guardrail("Output")]}

    def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]

    client._resource_client.responses = SimpleNamespace(retrieve=lambda *args, **kwargs: SimpleNamespace(output_text="hi"))  # type: ignore[attr-defined]

    sentinel = object()

    def fake_create_response(
        response: Any, preflight: list[GuardrailResult], input_results: list[GuardrailResult], output_results: list[GuardrailResult]
    ) -> Any:
        return sentinel

    monkeypatch.setattr(client, "_create_guardrails_response", fake_create_response)  # type: ignore[attr-defined]

    response = client.responses.retrieve("resp")

    assert response is sentinel  # noqa: S101


def test_azure_clients_initialize() -> None:
    """Azure variants should initialize using azure kwargs."""
    async_client = GuardrailsAsyncAzureOpenAI(config=_minimal_config(), api_key="key", azure_param=1)
    sync_client = GuardrailsAzureOpenAI(config=_minimal_config(), api_key="key", azure_param=1)

    assert async_client._azure_kwargs["azure_param"] == 1  # type: ignore[attr-defined]  # noqa: S101
    assert sync_client._azure_kwargs["azure_param"] == 1  # type: ignore[attr-defined]  # noqa: S101


def test_azure_sync_run_stage_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure sync client should run guardrails with conversation context."""
    client = GuardrailsAzureOpenAI(config=_minimal_config(), api_key="key")
    client.guardrails = {"output": [_guardrail("Prompt Injection Detection")]}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client_module, "run_guardrails", fake_run_guardrails)

    result = client._run_stage_guardrails("output", "payload", conversation_history=[{"role": "user", "content": "hi"}])

    assert result[0].tripwire_triggered is False  # noqa: S101


def test_azure_sync_append_response() -> None:
    """Azure sync append helper should handle string history."""
    client = GuardrailsAzureOpenAI(config=_minimal_config(), api_key="key")
    history = client._append_llm_response_to_conversation(
        "hi", SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="reply"))], output=None)
    )

    assert history[-1].message.content == "reply"  # type: ignore[union-attr]  # noqa: S101


def test_azure_sync_handle_llm_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure sync _handle_llm_response should call output guardrails."""
    client = GuardrailsAzureOpenAI(config=_minimal_config(), api_key="key")
    client.guardrails = {"output": [_guardrail("Output")], "pre_flight": [], "input": []}

    def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]

    sentinel = object()

    def fake_create_response(*args: Any, **kwargs: Any) -> Any:
        return sentinel

    monkeypatch.setattr(client, "_create_guardrails_response", fake_create_response)  # type: ignore[attr-defined]

    result = client._handle_llm_response(
        llm_response=SimpleNamespace(output_text="text", choices=[]),
        preflight_results=[],
        input_results=[],
        conversation_history=[],
    )

    assert result is sentinel  # noqa: S101


def test_azure_sync_context_with_conversation() -> None:
    """Azure sync conversation context should surface history only."""
    client = GuardrailsAzureOpenAI(config=_minimal_config(), api_key="key")
    context = client._create_context_with_conversation([{"role": "user", "content": "hi"}])

    assert context.get_conversation_history()[0]["content"] == "hi"  # type: ignore[index]  # noqa: S101
    assert not hasattr(context, "update_injection_last_checked_index")  # noqa: S101


def test_azure_sync_run_stage_guardrails_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tripwire should be suppressed when requested for Azure sync client."""
    client = GuardrailsAzureOpenAI(config=_minimal_config(), api_key="key")
    client.guardrails = {"output": [_guardrail("Prompt Injection Detection")]}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=True)]

    monkeypatch.setattr(client_module, "run_guardrails", fake_run_guardrails)

    results = client._run_stage_guardrails(
        "output",
        "payload",
        conversation_history=[{"role": "user", "content": "hi"}],
        suppress_tripwire=True,
    )

    assert results[0].tripwire_triggered is True  # noqa: S101


def test_handle_llm_response_suppresses_tripwire_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppressed output guardrails should return triggered result."""
    client = _build_client()

    def fake_run_stage(
        stage_name: str,
        text: str,
        conversation_history: list | None = None,
        suppress_tripwire: bool = False,
    ) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=True)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]

    response = SimpleNamespace(output_text="text", choices=[])

    result = client._handle_llm_response(
        response,
        preflight_results=[],
        input_results=[],
        conversation_history=[],
        suppress_tripwire=True,
    )

    assert result.guardrail_results.output[0].tripwire_triggered is True  # noqa: S101


def test_override_resources_replaces_chat_and_responses() -> None:
    """_override_resources should swap chat and responses objects."""
    client = _build_client()
    # Manually call override to ensure replacement occurs
    client._override_resources()

    assert hasattr(client.chat, "completions")  # noqa: S101
    assert hasattr(client.responses, "create")  # noqa: S101
