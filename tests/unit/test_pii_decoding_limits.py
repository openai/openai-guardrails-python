"""Decoded-size rejection must fail closed without decoding large inputs."""

from __future__ import annotations

import asyncio
import importlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from guardrails.client import GuardrailsAsyncOpenAI, GuardrailsOpenAI
from guardrails.exceptions import GuardrailTripwireTriggered
from guardrails.runtime import ConfigBundle, GuardrailConfig, instantiate_guardrails, run_guardrails

pii_module = importlib.import_module("guardrails.checks.text.pii")


@pytest.fixture
def analyzer(monkeypatch: pytest.MonkeyPatch) -> Mock:
    engine = Mock(analyze=Mock(return_value=[]))
    monkeypatch.setattr(pii_module, "_get_analyzer_engine", Mock(return_value=engine))
    return engine


@pytest.fixture(params=["base64", "hex"])
def size_rejection(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, analyzer: Mock) -> str:
    # Simulate only the decoder's length report. No large payload is allocated
    # or decoded, while the real limit check and caller pipeline still execute.
    decoded = MagicMock(spec=bytes)
    decoded.__len__.return_value = 10_001
    if request.param == "base64":
        monkeypatch.setattr(pii_module.base64, "b64decode", Mock(return_value=decoded))
        return "ZXhhbXBsZSBkb2N1bWVudA=="
    monkeypatch.setattr(pii_module, "bytes", Mock(fromhex=Mock(return_value=decoded)), raising=False)
    return "6578616d706c6520646f63756d656e74"


def pii_config(block: bool = False) -> dict[str, Any]:
    return {"entities": ["EMAIL_ADDRESS"], "block": block, "detect_encoded_pii": True}


@pytest.mark.asyncio
@pytest.mark.parametrize("block", [False, True])
async def test_limit_failure_retains_tripwire_and_strict_error(size_rejection: str, block: bool) -> None:
    guardrails = instantiate_guardrails(ConfigBundle(guardrails=[GuardrailConfig(name="Contains PII", config=pii_config(block))]))
    with pytest.raises(GuardrailTripwireTriggered):
        await run_guardrails(SimpleNamespace(), size_rejection, "text/plain", guardrails)

    # Suppression exposes the failure for manual handling without inventing PII.
    results = await run_guardrails(SimpleNamespace(), size_rejection, "text/plain", guardrails, suppress_tripwire=True)
    result = results[0]
    assert result.tripwire_triggered
    assert result.execution_failed
    assert result.info["checked_text"] == ""
    assert "pii_detected" not in result.info
    assert size_rejection not in str(result.info)
    error = result.original_exception
    assert isinstance(error, ValueError)
    assert "too large" in str(error)
    assert error.__traceback__ is None
    assert error.__context__ is None

    with pytest.raises(ValueError, match="too large"):
        await run_guardrails(SimpleNamespace(), size_rejection, "text/plain", guardrails, raise_guardrail_errors=True, suppress_tripwire=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_limit_failure_blocks_before_provider(size_rejection: str, asynchronous: bool, responses: bool, stream: bool) -> None:
    config = {"version": 1, "pre_flight": {"version": 1, "guardrails": [{"name": "Contains PII", "config": pii_config()}]}}
    client: Any = (GuardrailsAsyncOpenAI if asynchronous else GuardrailsOpenAI)(config=config, api_key="test-key")
    provider = AsyncMock() if asynchronous else Mock()
    if responses:
        client._resource_client.responses = SimpleNamespace(create=provider)
        resource = client.responses
        kwargs: dict[str, Any] = {"input": size_rejection}
    else:
        client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=provider))
        resource = client.chat.completions
        kwargs = {"messages": [{"role": "user", "content": [{"type": "text", "text": size_rejection}]}]}
    with pytest.raises(GuardrailTripwireTriggered):
        if asynchronous:
            await resource.create(model="test-model", stream=stream, **kwargs)
        else:
            await asyncio.to_thread(lambda: resource.create(model="test-model", stream=stream, **kwargs))
    provider.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["example document", "ZXhhbXBsZSBkb2N1bWVudA==", "6578616d706c6520646f63756d656e74", "////////////////"])
async def test_ordinary_content_keeps_existing_behavior(analyzer: Mock, text: str) -> None:
    result = await pii_module.pii(None, text, pii_module.PIIConfig(**pii_config()))
    assert not result.tripwire_triggered
    assert not result.execution_failed
    assert result.info["checked_text"] == text


@pytest.mark.asyncio
async def test_unrelated_errors_keep_existing_runtime_policy(analyzer: Mock) -> None:
    analyzer.analyze.side_effect = ValueError("analyzer unavailable")
    guardrails = instantiate_guardrails(ConfigBundle(guardrails=[GuardrailConfig(name="Contains PII", config=pii_config())]))
    results = await run_guardrails(SimpleNamespace(), "example document", "text/plain", guardrails)
    assert results[0].execution_failed
    assert not results[0].tripwire_triggered
