"""Tests for moderation guardrail."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from guardrails.checks.text.moderation import Category, ModerationCfg, moderation


class _StubModerationClient:
    """Stub moderations client that returns prerecorded results."""

    def __init__(self, categories: dict[str, bool]) -> None:
        self._categories = categories

    async def create(self, model: str, input: str) -> Any:
        _ = (model, input)

        class _Result:
            def model_dump(self_inner) -> dict[str, Any]:
                return {"categories": self._categories}

        return SimpleNamespace(results=[_Result()])


@pytest.mark.asyncio
async def test_moderation_triggers_on_flagged_categories(monkeypatch: pytest.MonkeyPatch) -> None:
    """Requested categories flagged True should trigger the guardrail."""
    stub_client = SimpleNamespace(moderations=_StubModerationClient({"hate": True, "violence": False}))

    monkeypatch.setattr("guardrails.checks.text.moderation._get_moderation_client", lambda: stub_client)

    cfg = ModerationCfg(categories=[Category.HATE, Category.VIOLENCE])
    result = await moderation(None, "text", cfg)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["flagged_categories"] == ["hate"]  # noqa: S101


@pytest.mark.asyncio
async def test_moderation_handles_empty_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing results should return an informative error."""

    async def create_empty(**_: Any) -> Any:
        return SimpleNamespace(results=[])

    stub_client = SimpleNamespace(moderations=SimpleNamespace(create=create_empty))

    monkeypatch.setattr("guardrails.checks.text.moderation._get_moderation_client", lambda: stub_client)

    cfg = ModerationCfg(categories=[Category.HARASSMENT])
    result = await moderation(None, "text", cfg)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["error"] == "No moderation results returned"  # noqa: S101


@pytest.mark.asyncio
async def test_moderation_uses_context_client() -> None:
    """Moderation should use the client from context when available."""
    from openai import AsyncOpenAI

    # Track whether context client was used
    context_client_used = False

    async def track_create(**_: Any) -> Any:
        nonlocal context_client_used
        context_client_used = True

        class _Result:
            def model_dump(self) -> dict[str, Any]:
                return {"categories": {"hate": False, "violence": False}}

        return SimpleNamespace(results=[_Result()])

    # Create a context with a guardrail_llm client
    context_client = AsyncOpenAI(api_key="test-context-key", base_url="https://api.openai.com/v1")
    context_client.moderations = SimpleNamespace(create=track_create)

    ctx = SimpleNamespace(guardrail_llm=context_client)

    cfg = ModerationCfg(categories=[Category.HATE])
    result = await moderation(ctx, "test text", cfg)

    # Verify the context client was used
    assert context_client_used is True  # noqa: S101
    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_moderation_falls_back_for_third_party_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Moderation should fall back to environment client for third-party providers."""
    from openai import AsyncOpenAI, NotFoundError

    # Create fallback client that tracks usage
    fallback_used = False

    async def track_fallback_create(**_: Any) -> Any:
        nonlocal fallback_used
        fallback_used = True

        class _Result:
            def model_dump(self) -> dict[str, Any]:
                return {"categories": {"hate": False}}

        return SimpleNamespace(results=[_Result()])

    fallback_client = SimpleNamespace(moderations=SimpleNamespace(create=track_fallback_create))
    monkeypatch.setattr("guardrails.checks.text.moderation._get_moderation_client", lambda: fallback_client)

    # Create a context client that simulates a third-party provider
    # When moderation is called, it should raise NotFoundError
    async def raise_not_found(**_: Any) -> Any:
        raise NotFoundError("404 page not found")

    third_party_client = AsyncOpenAI(api_key="third-party-key", base_url="https://localhost:8080/v1")
    third_party_client.moderations = SimpleNamespace(create=raise_not_found)
    ctx = SimpleNamespace(guardrail_llm=third_party_client)

    cfg = ModerationCfg(categories=[Category.HATE])
    result = await moderation(ctx, "test text", cfg)

    # Verify the fallback client was used (not the third-party one)
    assert fallback_used is True  # noqa: S101
    assert result.tripwire_triggered is False  # noqa: S101
