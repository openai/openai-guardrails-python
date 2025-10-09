"""Tests for URL guardrail helpers."""

from __future__ import annotations

import pytest

from guardrails.checks.text.urls import (
    URLConfig,
    _detect_urls,
    _is_url_allowed,
    _validate_url_security,
    urls,
)


def test_detect_urls_deduplicates_scheme_and_domain() -> None:
    """Ensure detection removes trailing punctuation and avoids duplicate domains."""
    text = "Visit https://example.com/, http://example.com/path, example.com should not duplicate, and 192.168.1.10:8080."
    detected = _detect_urls(text)

    assert "https://example.com/" in detected  # noqa: S101
    assert "http://example.com/path" in detected  # noqa: S101
    assert "example.com" not in detected  # noqa: S101
    assert "192.168.1.10:8080" in detected  # noqa: S101


def test_validate_url_security_blocks_bad_scheme() -> None:
    """Disallowed schemes should produce an error."""
    config = URLConfig()
    parsed, reason = _validate_url_security("http://blocked.com", config)

    assert parsed is None  # noqa: S101
    assert "Blocked scheme" in reason  # noqa: S101


def test_validate_url_security_blocks_userinfo_when_configured() -> None:
    """URLs with embedded credentials should be rejected when block_userinfo=True."""
    config = URLConfig(allowed_schemes={"https"}, block_userinfo=True)
    parsed, reason = _validate_url_security("https://user:pass@example.com", config)

    assert parsed is None  # noqa: S101
    assert "userinfo" in reason  # noqa: S101


def test_is_url_allowed_supports_subdomains_and_cidr() -> None:
    """Allow list should support subdomains and CIDR ranges."""
    config = URLConfig(
        url_allow_list=["example.com", "10.0.0.0/8"],
        allow_subdomains=True,
    )
    https_result, _ = _validate_url_security("https://api.example.com", config)
    ip_result, _ = _validate_url_security("https://10.1.2.3", config)

    assert https_result is not None  # noqa: S101
    assert ip_result is not None  # noqa: S101
    assert _is_url_allowed(https_result, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(ip_result, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_reports_allowed_and_blocked() -> None:
    """Urls guardrail should classify detected URLs based on config."""
    config = URLConfig(
        url_allow_list=["example.com"],
        allowed_schemes={"https", "data"},
        block_userinfo=True,
        allow_subdomains=False,
    )
    text = "Inline data URI data:text/plain;base64,QUJD. Use https://example.com/docs. Avoid http://attacker.com/login and https://sub.example.com."

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "https://example.com/docs" in result.info["allowed"]  # noqa: S101
    assert "data:text/plain;base64,QUJD" in result.info["allowed"]  # noqa: S101
    assert "http://attacker.com/login" in result.info["blocked"]  # noqa: S101
    assert "https://sub.example.com" in result.info["blocked"]  # noqa: S101
    assert any("Blocked scheme" in reason for reason in result.info["blocked_reasons"])  # noqa: S101
    assert any("Not in allow list" in reason for reason in result.info["blocked_reasons"])  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_allows_benign_input() -> None:
    """Benign text without URLs should not trigger."""
    config = URLConfig()
    result = await urls(ctx=None, data="No links here", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
