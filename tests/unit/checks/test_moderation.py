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
