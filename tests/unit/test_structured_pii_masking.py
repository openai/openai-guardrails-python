"""Provider-bound structured PII masking regressions."""

from __future__ import annotations

import asyncio
import base64
import urllib.parse
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from guardrails.client import GuardrailsAsyncOpenAI, GuardrailsOpenAI


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("prefix", ["", (base64.b64encode(b"Snowman: %E2").decode() + "%98%83; ") * 10])
async def test_repeated_encoded_pii_is_masked_before_provider_call(asynchronous: bool, stream: bool, prefix: str) -> None:
    """Mask every occurrence while preserving ordinary text and image parts."""
    config = {
        "version": 1,
        "pre_flight": {
            "version": 1,
            "guardrails": [{"name": "Contains PII", "config": {"entities": ["EMAIL_ADDRESS"], "block": False, "detect_encoded_pii": True}}],
        },
    }
    encoded = base64.b64encode(b"jane@example.com").decode()
    note = base64.b64encode(b"example document").decode()
    image = {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}
    messages: Any = [
        {"role": "user", "content": [{"type": "text", "text": f"{prefix}Note: {note}; first: {encoded}; second: {encoded}. End."}, image]}
    ]
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
        {"type": "text", "text": f"{prefix}Note: {note}; first: <EMAIL_ADDRESS_ENCODED>; second: <EMAIL_ADDRESS_ENCODED>. End."},
        image,
    ]
    assert messages[0]["content"][0]["text"] == f"{prefix}Note: {note}; first: {encoded}; second: {encoded}. End."


@pytest.mark.parametrize("count", [100, 200, 400])
def test_percent_escape_decoding_work_is_bounded(count: int) -> None:
    """Preflight processes short URL escapes without repeatedly decoding prefixes."""
    config = {
        "version": 1,
        "pre_flight": {
            "version": 1,
            "guardrails": [{"name": "Contains PII", "config": {"entities": ["EMAIL_ADDRESS"], "block": False, "detect_encoded_pii": True}}],
        },
    }
    text = "%61 " * count
    messages: Any = [{"role": "user", "content": [{"type": "text", "text": text}]}]
    client = GuardrailsOpenAI(config=config, api_key="test-key")
    provider = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="OK"))]))
    client._resource_client.chat = SimpleNamespace(completions=SimpleNamespace(create=provider))

    # Count actual input processed by the decoder, avoiding timing-dependent
    # assertions and including every invocation made by public preflight.
    with patch("guardrails.checks.text.pii.urllib.parse.unquote", wraps=urllib.parse.unquote) as decode:
        client.chat.completions.create(messages=messages, model="test-model")

    assert sum(len(call.args[0]) for call in decode.call_args_list) <= 4 * len(text)
    provider.assert_called_once()
    assert provider.call_args.kwargs["messages"] == messages


def test_mixed_encoded_spans_preserve_unicode_and_unrelated_content() -> None:
    """Decoded offsets remain correct across Unicode, hex, Base64 and partial URLs."""
    from guardrails.checks.text.pii import _build_decoded_text

    note = base64.b64encode(b"example document").decode()
    email = b"jane@example.com".hex()
    nested = base64.b64encode(b"joe%40example.com").decode()
    text = f"Snowman: %E2%98%83; {note}; {email}; {nested}; jane%40example.com."
    decoded, candidates = _build_decoded_text(text)

    assert decoded == "Snowman: ☃; example document; jane@example.com; joe@example.com; jane@example.com."
    assert [decoded[c.decoded_start : c.decoded_end] for c in candidates] == [
        "☃",
        "example document",
        "jane@example.com",
        "joe@example.com",
        "@",
    ]


@pytest.mark.parametrize("text", ["☃ %E2%98%83 end", "%F0%9F%92%A9", "%E2%98text", "%FF%C3%A9", "%4x%41%", "%ED%A0%80"])
def test_url_offsets_match_standard_library_prefix_decoding(text: str) -> None:
    """Offsets retain unquote semantics even inside percent and UTF-8 sequences."""
    from guardrails.checks.text.pii import _url_decoded_offsets

    assert _url_decoded_offsets(text) == [len(urllib.parse.unquote(text[:end])) for end in range(len(text) + 1)]
