"""Tests for keyword-based guardrail helpers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from guardrails.checks.text.competitors import CompetitorCfg, competitors
from guardrails.checks.text.keywords import KeywordCfg, keywords, match_keywords
from guardrails.types import GuardrailResult


def test_match_keywords_sanitizes_trailing_punctuation() -> None:
    """Ensure keyword sanitization strips trailing punctuation before matching."""
    config = KeywordCfg(keywords=["token.", "secret!", "KEY?"])
    result = match_keywords("Leaked token appears here.", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["sanitized_keywords"] == ["token", "secret", "KEY"]  # noqa: S101
    assert result.info["matched"] == ["token"]  # noqa: S101
    assert result.info["guardrail_name"] == "Test Guardrail"  # noqa: S101
    assert result.info["checked_text"] == "Leaked token appears here."  # noqa: S101


def test_match_keywords_deduplicates_case_insensitive_matches() -> None:
    """Repeated matches differing by case should be deduplicated."""
    config = KeywordCfg(keywords=["Alert"])
    result = match_keywords("alert ALERT Alert", config, guardrail_name="Keyword Filter")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["matched"] == ["alert"]  # noqa: S101


@pytest.mark.asyncio
async def test_keywords_guardrail_wraps_match_keywords() -> None:
    """Async guardrail should mirror match_keywords behaviour."""
    config = KeywordCfg(keywords=["breach"])
    result = await keywords(ctx=None, data="Potential breach detected", config=config)

    assert isinstance(result, GuardrailResult)  # noqa: S101
    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["guardrail_name"] == "Keyword Filter"  # noqa: S101


@pytest.mark.asyncio
async def test_competitors_uses_keyword_matching() -> None:
    """Competitors guardrail delegates to keyword matching with distinct name."""
    config = CompetitorCfg(keywords=["ACME Corp"])
    result = await competitors(ctx=None, data="Comparing against ACME Corp today", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["guardrail_name"] == "Competitors"  # noqa: S101
    assert result.info["matched"] == ["ACME Corp"]  # noqa: S101


def test_keyword_cfg_requires_non_empty_keywords() -> None:
    """KeywordCfg should enforce at least one keyword."""
    with pytest.raises(ValidationError):
        KeywordCfg(keywords=[])


@pytest.mark.asyncio
async def test_keywords_does_not_trigger_on_benign_text() -> None:
    """Guardrail should not trigger when no keywords are present."""
    config = KeywordCfg(keywords=["restricted"])
    result = await keywords(ctx=None, data="Safe content", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
