"""Shared pytest fixtures for guardrails tests.

These fixtures provide deterministic test environments by stubbing the OpenAI
client library, seeding environment variables, and preventing accidental live
network activity during the suite.
"""

from __future__ import annotations

import logging
import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest


class _StubOpenAIBase:
    """Base stub with attribute bag behaviour for OpenAI client classes."""

    def __init__(self, **kwargs: Any) -> None:
        self._client_kwargs = kwargs
        self.chat = SimpleNamespace()
        self.responses = SimpleNamespace()
        self.api_key = kwargs.get("api_key", "test-key")
        self.base_url = kwargs.get("base_url")
        self.organization = kwargs.get("organization")
        self.timeout = kwargs.get("timeout")
        self.max_retries = kwargs.get("max_retries")

    def __getattr__(self, item: str) -> Any:
        """Return None for unknown attributes to emulate real client laziness."""
        return None


class _StubAsyncOpenAI(_StubOpenAIBase):
    """Stub asynchronous OpenAI client."""


class _StubSyncOpenAI(_StubOpenAIBase):
    """Stub synchronous OpenAI client."""


@dataclass(frozen=True, slots=True)
class _DummyResponse:
    """Minimal response type with choices and output."""

    choices: list[Any] | None = None
    output: list[Any] | None = None
    output_text: str | None = None
    type: str | None = None
    delta: str | None = None


_STUB_OPENAI_MODULE = types.ModuleType("openai")
_STUB_OPENAI_MODULE.AsyncOpenAI = _StubAsyncOpenAI
_STUB_OPENAI_MODULE.OpenAI = _StubSyncOpenAI
_STUB_OPENAI_MODULE.AsyncAzureOpenAI = _StubAsyncOpenAI
_STUB_OPENAI_MODULE.AzureOpenAI = _StubSyncOpenAI
_STUB_OPENAI_MODULE.NOT_GIVEN = object()


class APITimeoutError(Exception):
    """Stub API timeout error."""


_STUB_OPENAI_MODULE.APITimeoutError = APITimeoutError

_OPENAI_TYPES_MODULE = types.ModuleType("openai.types")
_OPENAI_TYPES_MODULE.Completion = _DummyResponse
_OPENAI_TYPES_MODULE.Response = _DummyResponse

_OPENAI_CHAT_MODULE = types.ModuleType("openai.types.chat")
_OPENAI_CHAT_MODULE.ChatCompletion = _DummyResponse
_OPENAI_CHAT_MODULE.ChatCompletionChunk = _DummyResponse

_OPENAI_RESPONSES_MODULE = types.ModuleType("openai.types.responses")
_OPENAI_RESPONSES_MODULE.Response = _DummyResponse
_OPENAI_RESPONSES_MODULE.ResponseInputItemParam = dict  # type: ignore[attr-defined]
_OPENAI_RESPONSES_MODULE.ResponseOutputItem = dict  # type: ignore[attr-defined]
_OPENAI_RESPONSES_MODULE.ResponseStreamEvent = dict  # type: ignore[attr-defined]


_OPENAI_RESPONSES_RESPONSE_MODULE = types.ModuleType("openai.types.responses.response")
_OPENAI_RESPONSES_RESPONSE_MODULE.Response = _DummyResponse


class _ResponseTextConfigParam(dict):
    """Stub config param used for response formatting."""


_OPENAI_RESPONSES_MODULE.ResponseTextConfigParam = _ResponseTextConfigParam

sys.modules["openai"] = _STUB_OPENAI_MODULE
sys.modules["openai.types"] = _OPENAI_TYPES_MODULE
sys.modules["openai.types.chat"] = _OPENAI_CHAT_MODULE
sys.modules["openai.types.responses"] = _OPENAI_RESPONSES_MODULE
sys.modules["openai.types.responses.response"] = _OPENAI_RESPONSES_RESPONSE_MODULE


@pytest.fixture(autouse=True)
def stub_openai_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Provide stub OpenAI module so tests avoid real network-bound clients."""
    # Patch imported symbols in guardrails modules
    from guardrails import _base_client, client, types as guardrail_types  # type: ignore

    monkeypatch.setattr(_base_client, "AsyncOpenAI", _StubAsyncOpenAI, raising=False)
    monkeypatch.setattr(_base_client, "OpenAI", _StubSyncOpenAI, raising=False)
    monkeypatch.setattr(client, "AsyncOpenAI", _StubAsyncOpenAI, raising=False)
    monkeypatch.setattr(client, "OpenAI", _StubSyncOpenAI, raising=False)
    monkeypatch.setattr(client, "AsyncAzureOpenAI", _StubAsyncOpenAI, raising=False)
    monkeypatch.setattr(client, "AzureOpenAI", _StubSyncOpenAI, raising=False)
    monkeypatch.setattr(guardrail_types, "AsyncOpenAI", _StubAsyncOpenAI, raising=False)
    monkeypatch.setattr(guardrail_types, "OpenAI", _StubSyncOpenAI, raising=False)
    monkeypatch.setattr(guardrail_types, "AsyncAzureOpenAI", _StubAsyncOpenAI, raising=False)
    monkeypatch.setattr(guardrail_types, "AzureOpenAI", _StubSyncOpenAI, raising=False)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    yield _STUB_OPENAI_MODULE


@pytest.fixture(autouse=True)
def configure_logging() -> None:
    """Ensure logging defaults to DEBUG for deterministic assertions."""
    logging.basicConfig(level=logging.DEBUG)
