"""Tests for guardrails.context helpers."""

from __future__ import annotations

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
        context.guardrail_llm = None  # type: ignore[misc]
