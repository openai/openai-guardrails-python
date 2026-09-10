"""Output classifiers consume the same masked current input as the provider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel

from guardrails.checks.text.llm_base import LLMOutput, run_llm
from guardrails.client import GuardrailsAsyncOpenAI, GuardrailsOpenAI
from guardrails.types import GuardrailResult


class _ParsedReply(BaseModel):
    answer: str


@pytest.mark.parametrize("client_type", [GuardrailsOpenAI, GuardrailsAsyncOpenAI])
@pytest.mark.parametrize("api,stream", [("chat", False), ("chat", True), ("responses", False), ("responses", True), ("parse", False)])
@pytest.mark.parametrize("mask", [True, False])
def test_output_classifier_receives_provider_input(monkeypatch: pytest.MonkeyPatch, client_type: Any, api: str, stream: bool, mask: bool) -> None:
    """Preserve masking through actual resource, response, and classifier pipelines."""
    import asyncio

    check = TestCase()
    is_async = client_type is GuardrailsAsyncOpenAI
    client = client_type(config={"version": 1, "output": {"version": 1, "guardrails": []}}, api_key="test-key")
    guardrail = SimpleNamespace(definition=SimpleNamespace(metadata=SimpleNamespace(uses_conversation_history=True)))
    client.guardrails = {stage: [guardrail] for stage in ("pre_flight", "input", "output")}
    original = "Please contact sample@example.test"
    sanitized = "Please contact <EMAIL_ADDRESS>" if mask else original
    prior = [{"role": "assistant", "content": "Earlier reply"}]
    messages = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": original}]
    # Exercise Responses string input and structured parse input separately.
    payload = original if api == "responses" else messages
    untouched = deepcopy(payload)
    prior_snapshot = deepcopy(prior)
    classifier = AsyncMock(
        return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"flagged": false, "confidence": 0.0}'))], usage=None)
    )
    monkeypatch.setattr("guardrails.checks.text.llm_base._request_chat_completion", classifier)
    stage_history: dict[str, Any] = {}

    async def run_checks(**kwargs: Any) -> list[GuardrailResult]:
        stage = kwargs["stage_name"]
        history = kwargs["ctx"].get_conversation_history()
        stage_history[stage] = deepcopy(history)
        if stage == "pre_flight":
            return [
                GuardrailResult(tripwire_triggered=False, info={"guardrail_name": "Contains PII", "pii_detected": mask, "checked_text": sanitized})
            ]
        if stage == "output":
            await run_llm(
                kwargs["data"], "Classify the conversation", client.context.guardrail_llm, "test-model", LLMOutput, conversation_history=history
            )
        return []

    monkeypatch.setattr("guardrails.client.run_guardrails", run_checks)
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content="Reply"))], output=None, output_text="Reply")
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Reply"))])

    async def chunks() -> AsyncIterator[Any]:
        yield chunk

    provider_result = (chunks() if is_async else iter([chunk])) if stream else reply
    provider = AsyncMock(return_value=provider_result) if is_async else Mock(return_value=provider_result)
    if api == "chat":
        client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=provider))
        call = client.chat.completions.create
        arguments = {"messages": payload, "model": "test-model", "stream": stream}
        expected_history = [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": sanitized}]
    else:
        client._resource_client.responses = SimpleNamespace(create=provider, parse=provider)
        loader = AsyncMock(return_value=prior) if is_async else Mock(return_value=prior)
        monkeypatch.setattr(client, "_load_conversation_history_from_previous_response", loader)
        call = client.responses.parse if api == "parse" else client.responses.create
        arguments = {"input": payload, "model": "test-model", "previous_response_id": "previous"}
        if api == "parse":
            arguments["text_format"] = cast(Any, _ParsedReply)
        else:
            arguments["stream"] = stream
        expected_history = (
            prior + ([{"role": "system", "content": "Be helpful"}] if api == "parse" else []) + [{"role": "user", "content": sanitized}]
        )

    async def invoke_async() -> None:
        result = await call(**arguments)
        if stream:
            async for _ in result:
                pass

    if is_async:
        asyncio.run(invoke_async())
    else:
        result = call(**arguments)
        if stream:
            list(result)

    expected_provider = sanitized if api == "responses" else [{"role": "system", "content": "Be helpful"}, {"role": "user", "content": sanitized}]
    check.assertEqual(provider.call_args.kwargs["messages" if api == "chat" else "input"], expected_provider)
    check.assertEqual(payload, untouched)
    check.assertEqual(prior, prior_snapshot)
    check.assertEqual(stage_history["pre_flight"][-1]["content"], original)
    check.assertEqual(stage_history["input"][-1]["content"], original)
    check.assertEqual(stage_history["output"], expected_history + [{"role": "assistant", "content": "Reply"}])
    classifier.assert_awaited_once()
    user_content = classifier.call_args.kwargs["messages"][1]["content"]
    analysis = json.loads(user_content.split("\n\n", 1)[1])
    check.assertEqual(analysis["conversation"], expected_history + [{"role": "assistant", "content": "Reply"}])
    if mask:
        check.assertNotIn(original, user_content)
