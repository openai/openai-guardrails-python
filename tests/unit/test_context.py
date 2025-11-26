"""Tests for guardrails.context helpers."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar, copy_context
from dataclasses import FrozenInstanceError

import pytest

from guardrails.context import GuardrailsContext, clear_context, get_context, has_context, set_context


class _StubClient:
    """Minimal client placeholder for GuardrailsContext."""

    api_key = "stub"


def test_set_and_get_context_roundtrip() -> None:
    """set_context should make context available via get_context."""
    context = GuardrailsContext(guardrail_llm=_StubClient())
    set_context(context)

    retrieved = get_context()
    assert retrieved is context  # noqa: S101
    assert has_context() is True  # noqa: S101

    clear_context()
    assert get_context() is None  # noqa: S101
    assert has_context() is False  # noqa: S101


def test_context_is_immutable() -> None:
    """GuardrailsContext should be frozen."""
    context = GuardrailsContext(guardrail_llm=_StubClient())

    with pytest.raises(FrozenInstanceError):
        context.guardrail_llm = None


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
    context = GuardrailsContext(guardrail_llm=_StubClient())
    set_context(context)

    def get_guardrails_context():
        return get_context()

    ctx = copy_context()
    result = ctx.run(get_guardrails_context)
    assert result is context  # noqa: S101

    clear_context()


def test_guardrails_context_propagates_with_threadpool() -> None:
    context = GuardrailsContext(guardrail_llm=_StubClient())
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
