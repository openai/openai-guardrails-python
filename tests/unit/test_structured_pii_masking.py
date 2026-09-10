"""Provider-bound structured PII masking regressions."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest

from guardrails.client import GuardrailsAsyncOpenAI, GuardrailsOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_repeated_encoded_pii_is_masked_before_provider_call(asynchronous: bool, stream: bool) -> None:
    """Mask every occurrence while preserving ordinary text and image parts."""
    config = {
        "version": 1,
        "pre_flight": {
            "version": 1,
            "guardrails": [{"name": "Contains PII", "config": {"entities": ["EMAIL_ADDRESS"], "block": False, "detect_encoded_pii": True}}],
        },
    }
    encoded = base64.b64encode(b"jane@example.com").decode()
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    messages: Any = [{"role": "user", "content": [{"type": "text", "text": f"First: {encoded}; second: {encoded}. End."}, image]}]
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))])
    if asynchronous:
        client = GuardrailsAsyncOpenAI(config=config, api_key="test-key")
        provider = AsyncMock(return_value=response)
        client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=provider))
        await client.chat.completions.create(messages=messages, model="test-model", stream=stream)
    else:
        sync_client = GuardrailsOpenAI(config=config, api_key="test-key")
        provider = Mock(return_value=response)
        sync_client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=provider))
        await asyncio.to_thread(lambda: sync_client.chat.completions.create(messages=messages, model="test-model", stream=stream))

    provider.assert_called_once()
    forwarded = provider.call_args.kwargs["messages"]
    assert forwarded[0]["content"] == [
        {"type": "text", "text": "First: <EMAIL_ADDRESS_ENCODED>; second: <EMAIL_ADDRESS_ENCODED>. End."},
        image,
    ]
    assert messages[0]["content"][0]["text"] == f"First: {encoded}; second: {encoded}. End."
