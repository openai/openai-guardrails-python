"""Conversation evaluation preserves the configured analysis client."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from openai import AsyncAzureOpenAI, AsyncOpenAI

from guardrails import instantiate_guardrails
from guardrails.checks.text import llm_base
from guardrails.evals.core.async_engine import AsyncRunEngine
from guardrails.evals.core.types import Context, Sample
from guardrails.runtime import ConfigBundle


@pytest.mark.asyncio
@pytest.mark.parametrize("azure", [False, True])
@pytest.mark.parametrize("multi_turn", [False, True])
async def test_conversation_evaluation_preserves_provider(monkeypatch: pytest.MonkeyPatch, azure: bool, multi_turn: bool) -> None:
    """The actual LLM check receives the selected client on every evaluated turn."""
    client: AsyncOpenAI
    if azure:
        client = AsyncAzureOpenAI(
            azure_endpoint="https://evaluation.azure.invalid",
            api_version="2025-01-01-preview",
            azure_ad_token="test-token",
        )
    else:
        client = AsyncOpenAI(api_key="test-key", base_url="https://evaluation.invalid/v1", max_retries=0)

    completion = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"flagged": false, "confidence": 0.1, "reason": "benign"}'))],
        usage=None,
    )
    request = AsyncMock(return_value=completion)
    monkeypatch.setattr(llm_base, "_request_chat_completion", request)
    guardrails = instantiate_guardrails(
        ConfigBundle.model_validate(
            {
                "version": 1,
                "guardrails": [{"name": "Jailbreak", "config": {"model": "test-model"}}],
            }
        )
    )
    engine = AsyncRunEngine(guardrails, multi_turn=multi_turn)
    sample = Sample(
        id="conversation",
        data='[{"role": "user", "content": "Hello"}, {"role": "user", "content": "Good morning"}]',
        expected_triggers={"Jailbreak": False},
    )
    results = await engine.run(Context(guardrail_llm=client), [sample], batch_size=1)
    assert request.await_count == (2 if multi_turn else 1)
    for call in request.await_args_list:
        assert call.kwargs["client"] is client
    assert "Good morning" in str(request.await_args_list[-1].kwargs["messages"])
    assert results[0].triggered == {"Jailbreak": False}
    assert "error" not in results[0].details
