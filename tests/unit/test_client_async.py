"""Tests for GuardrailsAsyncOpenAI core behaviour."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import guardrails.client as client_module
from guardrails.client import GuardrailsAsyncAzureOpenAI, GuardrailsAsyncOpenAI
from guardrails.exceptions import GuardrailTripwireTriggered
from guardrails.types import GuardrailResult


def _minimal_config() -> dict[str, Any]:
    """Return minimal pipeline config with no guardrails."""
    return {"version": 1, "output": {"version": 1, "guardrails": []}}


def _build_client(**kwargs: Any) -> GuardrailsAsyncOpenAI:
    """Instantiate GuardrailsAsyncOpenAI with deterministic defaults."""
    return GuardrailsAsyncOpenAI(config=_minimal_config(), **kwargs)


def _guardrail(name: str) -> Any:
    return SimpleNamespace(definition=SimpleNamespace(name=name), ctx_requirements=SimpleNamespace())


@pytest.mark.asyncio
async def test_default_context_uses_distinct_guardrail_client() -> None:
    """Default context should hold a fresh AsyncOpenAI instance mirroring config."""
    client = _build_client(api_key="secret-key", base_url="http://example.com")

    assert client.context is not None  # noqa: S101
    assert client.context.guardrail_llm is not client  # type: ignore[attr-defined]  # noqa: S101
    assert client.context.guardrail_llm.api_key == "secret-key"  # type: ignore[attr-defined]  # noqa: S101
    assert client.context.guardrail_llm.base_url == "http://example.com"  # type: ignore[attr-defined]  # noqa: S101


@pytest.mark.asyncio
async def test_conversation_context_exposes_history() -> None:
    """Conversation-aware context should surface conversation history only."""
    client = _build_client()
    conversation = [{"role": "user", "content": "Hello"}]

    conv_ctx = client._create_context_with_conversation(conversation)

    assert conv_ctx.get_conversation_history() == conversation  # noqa: S101
    assert not hasattr(conv_ctx, "update_injection_last_checked_index")  # noqa: S101


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


def _guardrail(name: str) -> Any:
    """Create a guardrail stub with a definition name."""
    return SimpleNamespace(definition=SimpleNamespace(name=name), ctx_requirements=SimpleNamespace())


@pytest.mark.asyncio
async def test_run_stage_guardrails_raises_on_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tripwire results should raise unless suppressed."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("basic guardrail")]
    captured_ctx = {}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        captured_ctx.update(kwargs)
        return [GuardrailResult(tripwire_triggered=True)]

    monkeypatch.setattr("guardrails.client.run_guardrails", fake_run_guardrails)

    with pytest.raises(GuardrailTripwireTriggered):
        await client._run_stage_guardrails("output", "payload")

    assert captured_ctx["ctx"] is client.context  # noqa: S101
    assert captured_ctx["stage_name"] == "output"  # noqa: S101


@pytest.mark.asyncio
async def test_run_stage_guardrails_uses_conversation_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prompt injection guardrail should trigger conversation-aware context."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("Prompt Injection Detection")]
    conversation = [{"role": "user", "content": "Hi"}]
    captured_ctx = {}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        captured_ctx.update(kwargs)
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr("guardrails.client.run_guardrails", fake_run_guardrails)

    results = await client._run_stage_guardrails("output", "payload", conversation_history=conversation)

    assert results == [GuardrailResult(tripwire_triggered=False)]  # noqa: S101
    ctx = captured_ctx["ctx"]
    assert ctx.get_conversation_history() == conversation  # noqa: S101


@pytest.mark.asyncio
async def test_run_stage_guardrails_suppresses_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress flag should return results even when tripwire fires."""
    client = _build_client()
    client.guardrails["output"] = [_guardrail("basic guardrail")]
    captured_kwargs = {}
    result = GuardrailResult(tripwire_triggered=True)

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        captured_kwargs.update(kwargs)
        return [result]

    monkeypatch.setattr("guardrails.client.run_guardrails", fake_run_guardrails)

    results = await client._run_stage_guardrails("output", "payload", suppress_tripwire=True)

    assert results == [result]  # noqa: S101
    assert captured_kwargs["suppress_tripwire"] is True  # noqa: S101


@pytest.mark.asyncio
async def test_handle_llm_response_runs_output_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """_handle_llm_response should append conversation and return response wrapper."""
    client = _build_client()
    output_result = GuardrailResult(tripwire_triggered=False)
    captured_text: list[str] = []
    captured_history: list[list[Any]] = []

    async def fake_run_stage(
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

    response = await client._handle_llm_response(
        llm_response,
        preflight_results=[GuardrailResult(tripwire_triggered=False)],
        input_results=[],
        conversation_history=[{"role": "user", "content": "hello"}],
    )

    assert captured_text == ["LLM response"]  # noqa: S101
    assert captured_history[-1][-1].message.content == "LLM response"  # type: ignore[index]  # noqa: S101
    assert response.guardrail_results.output == [output_result]  # noqa: S101


@pytest.mark.asyncio
async def test_chat_completions_create_runs_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """chat.completions.create should execute guardrail stages."""
    client = _build_client()
    client.guardrails = {
        "pre_flight": [_guardrail("Prompt Injection Detection")],
        "input": [_guardrail("Input Guard")],
        "output": [_guardrail("Output Guard")],
    }
    stage_calls: list[str] = []

    async def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        stage_calls.append(stage_name)
        return [GuardrailResult(tripwire_triggered=False, info={"stage": stage_name})]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]
    monkeypatch.setattr(client, "_apply_preflight_modifications", lambda messages, results: messages)  # type: ignore[attr-defined]

    async def fake_llm(**kwargs: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), delta=SimpleNamespace(content=None))],
            output=None,
            output_text=None,
        )

    client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=fake_llm))  # type: ignore[attr-defined]

    response = await client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], model="gpt")

    assert stage_calls[:2] == ["pre_flight", "input"]  # noqa: S101
    assert response.guardrail_results.output[0].info["stage"] == "output"  # noqa: S101


@pytest.mark.asyncio
async def test_chat_completions_create_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming path should defer to _stream_with_guardrails."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [], "output": []}

    def fake_stream_with_guardrails(*args: Any, **kwargs: Any):
        async def _gen():
            yield "chunk"

        return _gen()

    monkeypatch.setattr(client, "_stream_with_guardrails", fake_stream_with_guardrails)  # type: ignore[attr-defined]

    async def fake_llm(**kwargs: Any) -> Any:
        async def _aiter():
            yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="c"))])

        return _aiter()

    client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=fake_llm))  # type: ignore[attr-defined]

    stream = await client.chat.completions.create(messages=[{"role": "user", "content": "hi"}], model="gpt", stream=True)

    chunks = []
    async for value in stream:
        chunks.append(value)

    assert chunks == ["chunk"]  # noqa: S101


@pytest.mark.asyncio
async def test_responses_create_runs_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses.create should run guardrail stages and handle output."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [_guardrail("Input Guard")], "output": [_guardrail("Output Guard")]}
    stage_calls: list[str] = []

    async def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        stage_calls.append(stage_name)
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]
    monkeypatch.setattr(client, "_apply_preflight_modifications", lambda messages, results: messages)  # type: ignore[attr-defined]

    async def fake_llm(**kwargs: Any) -> Any:
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"), delta=SimpleNamespace(content=None))],
            output=None,
            output_text=None,
        )

    client._resource_client.responses = SimpleNamespace(create=fake_llm)  # type: ignore[attr-defined]

    result = await client.responses.create(input=[{"role": "user", "content": "hi"}], model="gpt")

    assert "input" in stage_calls  # noqa: S101
    assert result.guardrail_results.output[0].tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_responses_parse_runs_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses.parse should invoke guardrails and return wrapped response."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [_guardrail("Input Guard")], "output": []}

    async def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]
    monkeypatch.setattr(client, "_apply_preflight_modifications", lambda messages, results: messages)  # type: ignore[attr-defined]

    async def fake_llm(**kwargs: Any) -> Any:
        return SimpleNamespace(output_text="{}", output=[{"type": "message", "content": "parsed"}])

    client._resource_client.responses = SimpleNamespace(parse=fake_llm)  # type: ignore[attr-defined]

    result = await client.responses.parse(input=[{"role": "user", "content": "hi"}], model="gpt", text_format=dict)

    assert result.guardrail_results.input[0].tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_responses_retrieve_runs_output_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """responses.retrieve should execute output guardrails."""
    client = _build_client()
    client.guardrails = {"pre_flight": [], "input": [], "output": [_guardrail("Output Guard")]}

    async def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False, info={"stage": stage_name})]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]

    async def retrieve_response(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(output_text="hi")

    client._resource_client.responses = SimpleNamespace(retrieve=retrieve_response)  # type: ignore[attr-defined]

    result = await client.responses.retrieve("resp")

    assert result.guardrail_results.output[0].info["stage"] == "output"  # noqa: S101


@pytest.mark.asyncio
async def test_async_azure_run_stage_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure async client should reuse conversation context."""
    client = GuardrailsAsyncAzureOpenAI(config=_minimal_config(), api_key="key")
    client.guardrails = {"output": [_guardrail("Prompt Injection Detection")]}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client_module, "run_guardrails", fake_run_guardrails)

    results = await client._run_stage_guardrails("output", "payload", conversation_history=[{"role": "user", "content": "hi"}])

    assert results[0].tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_async_azure_default_context() -> None:
    """Azure async client should provide default context when needed."""
    client = GuardrailsAsyncAzureOpenAI(config=_minimal_config(), api_key="key")
    context = client._create_default_context()

    assert hasattr(context, "guardrail_llm")  # noqa: S101


@pytest.mark.asyncio
async def test_async_azure_append_response() -> None:
    """Azure async append helper should merge responses."""
    client = GuardrailsAsyncAzureOpenAI(config=_minimal_config(), api_key="key")
    history = client._append_llm_response_to_conversation(None, SimpleNamespace(output=[{"role": "assistant", "content": "data"}], choices=None))

    assert history[-1]["content"] == "data"  # type: ignore[index]  # noqa: S101


@pytest.mark.asyncio
async def test_async_azure_handle_llm_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """Azure async _handle_llm_response should call output guardrails."""
    client = GuardrailsAsyncAzureOpenAI(config=_minimal_config(), api_key="key")
    client.guardrails = {"output": [_guardrail("Output")], "pre_flight": [], "input": []}

    async def fake_run_stage(stage_name: str, text: str, **kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False)]

    monkeypatch.setattr(client, "_run_stage_guardrails", fake_run_stage)  # type: ignore[attr-defined]

    sentinel = object()

    def fake_create_response(*args: Any, **kwargs: Any) -> Any:
        return sentinel

    monkeypatch.setattr(client, "_create_guardrails_response", fake_create_response)  # type: ignore[attr-defined]

    result = await client._handle_llm_response(
        llm_response=SimpleNamespace(output_text="value", choices=[]),
        preflight_results=[],
        input_results=[],
        conversation_history=[],
    )

    assert result is sentinel  # noqa: S101


@pytest.mark.asyncio
async def test_async_azure_context_with_conversation() -> None:
    """Azure async conversation context should surface history only."""
    client = GuardrailsAsyncAzureOpenAI(config=_minimal_config(), api_key="key")
    ctx = client._create_context_with_conversation([{"role": "user", "content": "hi"}])

    assert ctx.get_conversation_history()[0]["content"] == "hi"  # type: ignore[index]  # noqa: S101
    assert not hasattr(ctx, "update_injection_last_checked_index")  # noqa: S101


@pytest.mark.asyncio
async def test_async_azure_run_stage_guardrails_suppressed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tripwire should be suppressed when requested."""
    client = GuardrailsAsyncAzureOpenAI(config=_minimal_config(), api_key="key")
    client.guardrails = {"output": [_guardrail("Prompt Injection Detection")]}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=True)]

    monkeypatch.setattr(client_module, "run_guardrails", fake_run_guardrails)

    results = await client._run_stage_guardrails(
        "output",
        "payload",
        conversation_history=[{"role": "user", "content": "hi"}],
        suppress_tripwire=True,
    )

    assert results[0].tripwire_triggered is True  # noqa: S101
