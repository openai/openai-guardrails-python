"""Tests for secret key detection guardrail."""

from __future__ import annotations

import pytest

from guardrails.checks.text.secret_keys import SecretKeysCfg, _contains_allowed_pattern, _detect_secret_keys, secret_keys


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


def test_allowed_url_pattern_must_cover_entire_token() -> None:
    """A URL substring must not exempt unrelated text around it."""
    assert _contains_allowed_pattern("https://example.com/docs") is True  # noqa: S101
    assert _contains_allowed_pattern("prefixhttps://example.com/docs") is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_secret"),
    [
        ("https://attacker.example/?token=sk-AAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("https://attacker.example/sk-AAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("sk-AAAABBBBCCCCDDDD.png", "sk-AAAABBBBCCCCDDDD"),
    ],
)
async def test_secret_keys_checks_secrets_embedded_in_allowed_patterns(text: str, expected_secret: str) -> None:
    """Allowed URL/file shapes must not suppress embedded secret values."""
    config = SecretKeysCfg(threshold="balanced")
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected_secret in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/docs?id=1234",
        "artifact-123.png",
        "5F9a2B7c8D1e3F4a5B6c7D8e9F0a1B2c.png",
    ],
)
async def test_secret_keys_keeps_benign_allowed_patterns_exempt(text: str) -> None:
    """Benign URLs and filenames should keep their non-strict exemption."""
    config = SecretKeysCfg(threshold="balanced")
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101
