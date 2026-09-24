"""Empty classifier responses respect the public client's execution-error policy."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from guardrails import GuardrailsAsyncOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("chat", [True, False], ids=["chat", "responses"])
@pytest.mark.parametrize("strict", [True, False], ids=["strict", "default"])
@pytest.mark.parametrize(
    "content",
    [None, "", '{"flagged": false, "confidence": 0.9, "reason": "Benign input"}'],
    ids=["refusal", "empty", "valid-negative"],
)
async def test_jailbreak_preflight_classification_policy(monkeypatch: pytest.MonkeyPatch, chat: bool, strict: bool, content: str | None) -> None:
    """Strict preflight rejects missing classifications before calling the application model."""
    options: dict[str, Any] = {"raise_guardrail_errors": True} if strict else {}
    client = GuardrailsAsyncOpenAI(
        config={
            "version": 1,
            "pre_flight": {
                "version": 1,
                "guardrails": [{"name": "Jailbreak", "config": {"model": "classifier"}}],
            },
        },
        **options,
    )
    classifier = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, refusal="Cannot classify" if content is None else None))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=0, total_tokens=10),
        )
    )
    monkeypatch.setattr(client.context.guardrail_llm, "chat", SimpleNamespace(completions=SimpleNamespace(create=classifier)))
    application = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Application response"))] if chat else None,
            output=None,
            output_text="Application response" if not chat else None,
        )
    )
    client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=application))
    client._resource_client.responses = SimpleNamespace(create=application)

    request: Any
    if chat:
        request = client.chat.completions.create(messages=[{"role": "user", "content": "Hello"}], model="application")
    else:
        request = client.responses.create(input="Hello", model="application")

    if strict and not content:
        with pytest.raises(Exception, match="^LLM returned an empty classification response$"):
            await request
        application.assert_not_called()
    else:
        response = await request
        application.assert_awaited_once()
        result = response.guardrail_results.preflight[0]
        assert result.execution_failed is (not content)
        assert result.tripwire_triggered is False
        assert result.info["token_usage"]["prompt_tokens"] == 10

    classifier.assert_awaited_once()
