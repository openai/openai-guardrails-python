"""Public contracts using real SDK models and in-memory HTTP exchanges.

Invoked by test_real_sdk_contract.py in isolation from the parent conftest.
The filename deliberately avoids default collection in the stubbed interpreter.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import openai
import pytest
import pytest_asyncio
from pydantic import BaseModel

import guardrails
from guardrails.context import GuardrailsContext, clear_context, set_context

# Match the HTTP implementation used by the installed SDK (httpx or httpx2).
# Mocking its transport keeps SDK validation, routing and decoding intact.
http = importlib.import_module(openai.DefaultHttpxClient.__mro__[1].__module__.split(".")[0])


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def denied(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Real SDK contract tests must not open network connections")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket.socket, "connect_ex", denied)
    monkeypatch.setattr(socket, "create_connection", denied)


def pipeline(stage: str = "output", keyword: str | None = None) -> dict[str, Any]:
    checks = [] if keyword is None else [{"name": "Keyword Filter", "config": {"keywords": [keyword]}}]
    return {"version": 1, stage: {"version": 1, "guardrails": checks}}


def response_body(chat: bool, text: str = "hello") -> dict[str, Any]:
    if chat:
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}],
        }
    return {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "model": "test-model",
        "status": "completed",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
    }


@pytest_asyncio.fixture
async def client_case(request: pytest.FixtureRequest) -> AsyncIterator[tuple[Any, list[Any], bool, bool]]:
    asynchronous, azure = request.param
    requests: list[Any] = []

    def handler(req: Any) -> Any:
        requests.append(req)
        if req.headers.get("x-contract-error") == "yes":
            return http.Response(429, json={"error": {"message": "test rate limit", "type": "rate_limit_error"}})
        body = json.loads(req.content) if req.content else {}
        chat = "/chat/completions" in req.url.path
        if body.get("stream"):
            if chat:
                event = {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "test-model",
                    "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
                }
            else:
                event = {
                    "type": "response.output_text.delta",
                    "item_id": "msg_test",
                    "output_index": 0,
                    "content_index": 0,
                    "delta": "hello",
                    "sequence_number": 1,
                }
            return http.Response(
                200, headers={"content-type": "text/event-stream"}, content=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n".encode()
            )
        text = '{"answer":"hello"}' if "text" in body else "hello"
        return http.Response(200, json=response_body(chat, text))

    transport = http.MockTransport(handler)
    http_client = (http.AsyncClient if asynchronous else http.Client)(transport=transport, trust_env=False)
    kwargs: dict[str, Any] = {"api_key": "test-key", "max_retries": 0, "http_client": http_client}
    cls: Any
    if azure:
        kwargs.update(azure_endpoint="https://unit-test.openai.azure.com", api_version="2025-04-01-preview")
        cls = guardrails.GuardrailsAsyncAzureOpenAI if asynchronous else guardrails.GuardrailsAzureOpenAI
    else:
        kwargs["base_url"] = "https://api.openai.com/v1"
        cls = guardrails.GuardrailsAsyncOpenAI if asynchronous else guardrails.GuardrailsOpenAI
    # Use an explicit context to keep every possible provider call on the mock.
    sdk_cls = (openai.AsyncAzureOpenAI if asynchronous else openai.AzureOpenAI) if azure else (openai.AsyncOpenAI if asynchronous else openai.OpenAI)
    provider: Any = sdk_cls(**kwargs)
    set_context(GuardrailsContext(provider))
    clients: list[Any] = []

    def make(config: dict[str, Any] | None = None) -> Any:
        client = cls(config=config or pipeline(), **kwargs)
        clients.append(client)
        return client

    try:
        yield make, requests, asynchronous, azure
    finally:
        clear_context()
        for client in clients:
            if asynchronous:
                await client.close()
                await client._resource_client.close()
            else:
                client.close()
                client._resource_client.close()
        if asynchronous:
            await provider.close()
        else:
            provider.close()


CLIENTS = [(False, False), (True, False), (False, True), (True, True)]


async def call(function: Any, asynchronous: bool, *args: Any, **kwargs: Any) -> Any:
    # Sync Guardrails calls own their event loop; exercise them off the pytest loop.
    if asynchronous:
        return await function(*args, **kwargs)
    return await asyncio.to_thread(function, *args, **kwargs)


async def invoke(client: Any, asynchronous: bool, api: str, **kwargs: Any) -> Any:
    if api == "chat":
        return await call(client.chat.completions.create, asynchronous, messages=[{"role": "user", "content": "hello"}], model="test-model", **kwargs)
    return await call(client.responses.create, asynchronous, input="hello", model="test-model", **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_case", CLIENTS, indirect=True)
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("stream", [False, True])
async def test_create_wire_contract(client_case: Any, api: str, stream: bool) -> None:
    make, requests, asynchronous, azure = client_case
    result = await invoke(make(), asynchronous, api, stream=stream, temperature=0.2, extra_headers={"x-contract": "yes"})
    if stream:
        chunks = [chunk async for chunk in result] if asynchronous else await asyncio.to_thread(list, result)
        assert len(chunks) == 1
        result = chunks[0]
        assert (result.choices[0].delta.content if api == "chat" else result.delta) == "hello"
    else:
        assert (result.choices[0].message.content if api == "chat" else result.output_text) == "hello"
    assert isinstance(result, guardrails.GuardrailsResponse)
    assert result.guardrail_results.all_results == []
    assert len(requests) == 1
    req = requests[0]
    body = json.loads(req.content)
    assert body["temperature"] == 0.2
    assert body["stream"] is stream
    assert "suppress_tripwire" not in body
    assert req.headers["x-contract"] == "yes"
    assert ("safety_identifier" in body) is not azure
    if azure:
        assert req.url.host == "unit-test.openai.azure.com"
        assert req.url.params["api-version"] == "2025-04-01-preview"
        assert req.headers["api-key"] == "test-key"
        if api == "chat":
            assert "/deployments/test-model/chat/completions" in req.url.path
    else:
        assert req.headers["authorization"] == "Bearer test-key"


@pytest.mark.asyncio
@pytest.mark.parametrize("client_case", CLIENTS, indirect=True)
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("stage", ["pre_flight", "output"])
async def test_actual_check_tripwire_and_suppression(client_case: Any, api: str, stage: str) -> None:
    make, requests, asynchronous, _ = client_case
    client = make(pipeline(stage, "hello"))
    with pytest.raises(guardrails.GuardrailTripwireTriggered):
        await invoke(client, asynchronous, api)
    assert len(requests) == (0 if stage == "pre_flight" else 1)
    result = await invoke(client, asynchronous, api, suppress_tripwire=True)
    assert result.guardrail_results.tripwires_triggered
    assert len(result.guardrail_results.triggered_results) == 1
    assert len(requests) == (1 if stage == "pre_flight" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("client_case", CLIENTS, indirect=True)
async def test_retrieve_validates_actual_response(client_case: Any) -> None:
    make, requests, asynchronous, _ = client_case
    result = await call(make(pipeline("output", "hello")).responses.retrieve, asynchronous, "resp_test", suppress_tripwire=True)
    assert result.output_text == "hello"
    assert result.guardrail_results.output[0].tripwire_triggered
    assert requests[0].method == "GET"
    assert requests[0].url.path.endswith("/responses/resp_test")


class Answer(BaseModel):
    """Structured response expected by a caller."""

    answer: str


@pytest.mark.asyncio
@pytest.mark.parametrize("client_case", CLIENTS, indirect=True)
async def test_parse_actual_response(client_case: Any) -> None:
    make, requests, asynchronous, _ = client_case
    result = await call(make().responses.parse, asynchronous, input=[{"role": "user", "content": "hello"}], model="test-model", text_format=Answer)
    assert result.output_parsed == Answer(answer="hello")
    body = json.loads(requests[0].content)
    assert body["text"]["format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    "name",
    [
        "ConfiguredGuardrail",
        "GuardrailAgent",
        "GuardrailResult",
        "GuardrailResults",
        "GuardrailTripwireTriggered",
        "GuardrailsAsyncOpenAI",
        "GuardrailsOpenAI",
        "GuardrailsAsyncAzureOpenAI",
        "GuardrailsAzureOpenAI",
        "GuardrailsResponse",
        "check_plain_text",
        "checks",
        "JsonString",
        "ConfigSource",
        "run_guardrails",
        "GuardrailSpecMetadata",
        "instantiate_guardrails",
        "load_config_bundle",
        "load_pipeline_bundles",
        "default_spec_registry",
        "resources",
        "total_guardrail_token_usage",
    ],
)
def test_released_public_exports(name: str) -> None:
    assert name in guardrails.__all__
    assert getattr(guardrails, name) is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("source_kind", ["dict", "path", "json", "model"])
async def test_configuration_to_execution(tmp_path: Path, source_kind: str) -> None:
    bundle = pipeline("output", "hello")["output"]
    source: Any = bundle
    if source_kind == "path":
        source = tmp_path / "bundle.json"
        source.write_text(json.dumps(bundle))
    elif source_kind == "json":
        source = guardrails.JsonString(json.dumps(bundle))
    elif source_kind == "model":
        source = guardrails.load_config_bundle(bundle)
    configured = guardrails.instantiate_guardrails(guardrails.load_config_bundle(source))
    with pytest.raises(guardrails.GuardrailTripwireTriggered):
        await guardrails.run_guardrails({}, "hello", "text/plain", configured)
    results = await guardrails.run_guardrails({}, "hello", "text/plain", configured, suppress_tripwire=True)
    assert len(results) == 1
    assert results[0].tripwire_triggered
    assert results[0].info["guardrail_name"] == "Keyword Filter"


@pytest.mark.parametrize("config,code", [(pipeline("output", "hello"), 0), ({}, 1), (pipeline("output", ""), 1)])
def test_cli_validates_real_configuration(tmp_path: Path, capsys: pytest.CaptureFixture[str], config: dict[str, Any], code: int) -> None:
    from guardrails.cli import main

    path = tmp_path / "pipeline.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit) as exc:
        main(["validate", str(path), "--media-type", "text/plain"])
    assert exc.value.code == code
    output = capsys.readouterr()
    if code == 0:
        assert "1 guardrails loaded, 1 matching" in output.out
    else:
        assert "ERROR:" in output.err


@pytest.mark.asyncio
@pytest.mark.parametrize("client_case", CLIENTS, indirect=True)
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.xfail(
    strict=True, raises=AssertionError, reason="Preexisting: streaming discards output guardrail results, including suppressed tripwires"
)
async def test_stream_exposes_suppressed_output_violation(client_case: Any, api: str) -> None:
    make, _, asynchronous, _ = client_case
    stream = await invoke(make(pipeline("output", "hello")), asynchronous, api, stream=True, suppress_tripwire=True)
    chunks = [chunk async for chunk in stream] if asynchronous else await asyncio.to_thread(list, stream)
    assert any(chunk.guardrail_results.tripwires_triggered for chunk in chunks)


class CustomContext(BaseModel):
    """Application context for a registered custom check."""

    prefix: str


class CustomConfig(BaseModel):
    """Configuration consumed by the registered custom check."""

    suffix: str


def custom_sync(ctx: CustomContext, data: str, config: CustomConfig) -> guardrails.GuardrailResult:
    import threading

    return guardrails.GuardrailResult(tripwire_triggered=False, info={"value": ctx.prefix + data + config.suffix, "thread": threading.get_ident()})


async def custom_async(ctx: CustomContext, data: str, config: CustomConfig) -> guardrails.GuardrailResult:
    return custom_sync(ctx, data, config)


@pytest.mark.asyncio
@pytest.mark.parametrize("check", [custom_sync, custom_async])
async def test_custom_registry_to_execution(check: Any) -> None:
    from guardrails.registry import GuardrailRegistry

    registry = GuardrailRegistry()
    registry.register(name="custom", check_fn=check, description="Custom check", media_type="text/plain")
    bundle = guardrails.load_config_bundle({"guardrails": [{"name": "custom", "config": {"suffix": "!"}}]})
    configured = guardrails.instantiate_guardrails(bundle, registry)
    results = await guardrails.run_guardrails(CustomContext(prefix="say "), "hello", "text/plain", configured)
    assert len(results) == 1
    assert results[0].info["value"] == "say hello!"
    assert not results[0].tripwire_triggered


@pytest.mark.asyncio
@pytest.mark.xfail(strict=True, raises=AssertionError, reason="Preexisting: synchronous checks are invoked before asyncio.to_thread")
async def test_sync_custom_check_runs_off_event_loop() -> None:
    import threading

    from guardrails.registry import GuardrailRegistry

    registry = GuardrailRegistry()
    registry.register(name="custom", check_fn=custom_sync, description="Custom check", media_type="text/plain")
    configured = guardrails.instantiate_guardrails(
        guardrails.load_config_bundle({"guardrails": [{"name": "custom", "config": {"suffix": "!"}}]}), registry
    )
    results = await guardrails.run_guardrails(CustomContext(prefix=""), "hello", "text/plain", configured)
    assert results[0].info["thread"] != threading.get_ident()


@pytest.mark.asyncio
@pytest.mark.parametrize("client_case", CLIENTS, indirect=True)
@pytest.mark.parametrize("api", ["chat", "responses"])
async def test_provider_error_is_preserved(client_case: Any, api: str) -> None:
    make, requests, asynchronous, _ = client_case
    with pytest.raises(openai.RateLimitError) as exc:
        await invoke(make(), asynchronous, api, extra_headers={"x-contract-error": "yes"})
    assert exc.value.status_code == 429
    assert len(requests) == 1
