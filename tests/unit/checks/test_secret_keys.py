"""Tests for secret key detection guardrail."""

from __future__ import annotations

import pytest

from guardrails.checks.text.secret_keys import SecretKeysCfg, _detect_secret_keys, secret_keys


def test_detect_secret_keys_flags_high_entropy_strings() -> None:
    """High entropy tokens should be detected as potential secrets."""
    text = "API key sk-AAAABBBBCCCCDDDD"
    result = _detect_secret_keys(text, cfg={"min_length": 10, "min_entropy": 3.5, "min_diversity": 2, "strict_mode": True})

    assert result.tripwire_triggered is True  # noqa: S101
    assert "sk-AAAABBBBCCCCDDDD" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_with_custom_regex() -> None:
    """Custom regex patterns should trigger detection."""
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}"])
    result = await secret_keys(None, "internal-ab12 leaked", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_ignores_non_matching_input() -> None:
    """Benign inputs should not trigger the guardrail."""
    config = SecretKeysCfg(threshold="permissive")
    result = await secret_keys(None, "Hello world", config)

    assert result.tripwire_triggered is False  # noqa: S101
