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


def test_match_keywords_does_not_match_partial_words() -> None:
    """Ensure substrings embedded in larger words are ignored."""
    config = KeywordCfg(keywords=["orld"])
    result = match_keywords("Hello, world!", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is False  # noqa: S101


def test_match_keywords_handles_numeric_tokens() -> None:
    """Keywords containing digits should match exact tokens."""
    config = KeywordCfg(keywords=["world123"])
    result = match_keywords("Hello, world123", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["matched"] == ["world123"]  # noqa: S101


def test_match_keywords_rejects_partial_numeric_tokens() -> None:
    """Numeric keywords should not match when extra digits follow."""
    config = KeywordCfg(keywords=["world123"])
    result = match_keywords("Hello, world12345", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is False  # noqa: S101


def test_match_keywords_handles_underscored_tokens() -> None:
    """Underscored keywords should be detected exactly once."""
    config = KeywordCfg(keywords=["w_o_r_l_d"])
    result = match_keywords("Hello, w_o_r_l_d", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["matched"] == ["w_o_r_l_d"]  # noqa: S101


def test_match_keywords_rejects_words_embedded_in_underscores() -> None:
    """Words surrounded by underscores should not trigger partial matches."""
    config = KeywordCfg(keywords=["world"])
    result = match_keywords("Hello, test_world_test", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is False  # noqa: S101


def test_match_keywords_handles_chinese_characters() -> None:
    """Unicode keywords such as Chinese characters should match."""
    config = KeywordCfg(keywords=["你好"])
    result = match_keywords("你好", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["matched"] == ["你好"]  # noqa: S101


def test_match_keywords_handles_chinese_tokens_with_digits() -> None:
    """Unicode keywords that include digits should match whole tokens."""
    config = KeywordCfg(keywords=["你好123"])
    result = match_keywords("你好123", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["matched"] == ["你好123"]  # noqa: S101


def test_match_keywords_rejects_partial_chinese_tokens_with_digits() -> None:
    """Unicode keywords with trailing digits should not match supersets."""
    config = KeywordCfg(keywords=["你好123"])
    result = match_keywords("你好12345", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is False  # noqa: S101


def test_match_keywords_applies_boundaries_to_all_keywords() -> None:
    """Every keyword in a multi-token pattern should respect Unicode boundaries."""
    config = KeywordCfg(keywords=["test", "hello", "world"])
    result = match_keywords("testing hello world", config, guardrail_name="Test Guardrail")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["matched"] == ["hello", "world"]  # noqa: S101
