"""Tests for guardrails.context helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from dataclasses import FrozenInstanceError
from typing import cast

import pytest
from openai import AsyncOpenAI, OpenAI

from guardrails.context import GuardrailsContext, clear_context, get_context, has_context, set_context


def test_set_and_get_context_roundtrip() -> None:
    """set_context should make context available via get_context."""
    context = GuardrailsContext(guardrail_llm=AsyncOpenAI(api_key="test-key"))
    set_context(context)

    retrieved = get_context()
    assert retrieved is context  # noqa: S101
    assert has_context() is True  # noqa: S101

    clear_context()
    assert get_context() is None  # noqa: S101
    assert has_context() is False  # noqa: S101


def test_context_is_immutable() -> None:
    """GuardrailsContext should be frozen."""
    context = GuardrailsContext(guardrail_llm=AsyncOpenAI(api_key="test-key"))

    with pytest.raises(FrozenInstanceError):
        context.__setattr__("guardrail_llm", None)


def test_contextvar_propagates_with_copy_context() -> None:
    test_var: ContextVar[str | None] = ContextVar("test_var", default=None)
    test_var.set("test_value")

    def get_contextvar():
        return test_var.get()

    ctx = copy_context()
    result = ctx.run(get_contextvar)
    assert result == "test_value"  # noqa: S101


def test_contextvar_propagates_with_threadpool() -> None:
    test_var: ContextVar[str | None] = ContextVar("test_var", default=None)
    test_var.set("thread_test")

    def get_contextvar():
        return test_var.get()

    ctx = copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ctx.run, get_contextvar)
        result = future.result()

    assert result == "thread_test"  # noqa: S101


def test_guardrails_context_propagates_with_copy_context() -> None:
    context = GuardrailsContext(guardrail_llm=AsyncOpenAI(api_key="test-key"))
    set_context(context)

    def get_guardrails_context():
        return get_context()

    ctx = copy_context()
    result = ctx.run(get_guardrails_context)
    assert result is context  # noqa: S101

    clear_context()


def test_guardrails_context_propagates_with_threadpool() -> None:
    context = GuardrailsContext(guardrail_llm=AsyncOpenAI(api_key="test-key"))
    set_context(context)

    def get_guardrails_context():
        return get_context()

    ctx = copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ctx.run, get_guardrails_context)
        result = future.result()

    assert result is context  # noqa: S101

    clear_context()


def test_multiple_contextvars_propagate_with_threadpool() -> None:
    var1: ContextVar[str | None] = ContextVar("var1", default=None)
    var2: ContextVar[int | None] = ContextVar("var2", default=None)
    var1.set("value1")
    var2.set(42)

    def get_multiple_contextvars():
        return (var1.get(), var2.get())

    ctx = copy_context()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(ctx.run, get_multiple_contextvars)
        result = future.result()

    assert result == ("value1", 42)  # noqa: S101


def test_frozen_context_exposes_empty_history() -> None:
    """A client-only context returns no conversation history."""
    from guardrails.types import GuardrailLLMContextProto

    context = GuardrailsContext(guardrail_llm=AsyncOpenAI(api_key="test-key"))

    def read_client(value: GuardrailLLMContextProto) -> object:
        return value.guardrail_llm

    assert read_client(cast(GuardrailLLMContextProto, context)) is context.guardrail_llm
    assert cast(GuardrailLLMContextProto, context).get_conversation_history() is None


def test_explicit_protocol_subclass_can_store_client() -> None:
    """The protocol must not install a runtime property that blocks assignment."""
    from guardrails.types import GuardrailLLMContextProto

    class ExplicitContext(GuardrailLLMContextProto):
        guardrail_llm: AsyncOpenAI | OpenAI

        def __init__(self, client: AsyncOpenAI) -> None:
            self.guardrail_llm = client

    client = AsyncOpenAI(api_key="test-key")
    context = ExplicitContext(client)
    assert context.guardrail_llm is client
    assert context.get_conversation_history() is None
