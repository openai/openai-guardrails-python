"""Regression tests for URL host normalization."""

from __future__ import annotations

import pytest

from guardrails.checks.text.urls import URLConfig, _detect_urls, _is_url_allowed, _validate_url_security, urls


def test_interior_www_label_is_not_removed_from_host() -> None:
    config = URLConfig(url_allow_list=["aexample.com"], allow_subdomains=False)
    parsed, reason, had_scheme = _validate_url_security("https://awww.example.com/path", config)

    assert parsed is not None, reason  # noqa: S101
    assert _is_url_allowed(parsed, config.url_allow_list, config.allow_subdomains, had_scheme) is False  # noqa: S101


def test_interior_www_label_is_not_removed_from_allow_entry() -> None:
    config = URLConfig(url_allow_list=["awww.example.com"], allow_subdomains=False)
    parsed, reason, had_scheme = _validate_url_security("https://aexample.com/", config)

    assert parsed is not None, reason  # noqa: S101
    assert _is_url_allowed(parsed, config.url_allow_list, config.allow_subdomains, had_scheme) is False  # noqa: S101


def test_leading_www_label_still_normalizes() -> None:
    config = URLConfig(url_allow_list=["example.com"], allow_subdomains=False)
    parsed, reason, had_scheme = _validate_url_security("https://www.example.com", config)

    assert parsed is not None, reason  # noqa: S101
    assert _is_url_allowed(parsed, config.url_allow_list, config.allow_subdomains, had_scheme) is True  # noqa: S101


def test_url_dedup_keeps_distinct_interior_www_host() -> None:
    detected = _detect_urls("https://awww.example.com and aexample.com")

    assert "https://awww.example.com" in detected  # noqa: S101
    assert "aexample.com" in detected  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_blocks_interior_www_lookalike() -> None:
    config = URLConfig(
        url_allow_list=["aexample.com"],
        allowed_schemes={"https"},
        allow_subdomains=False,
    )

    result = await urls(ctx=None, data="Visit https://awww.example.com/path", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert "https://awww.example.com/path" in result.info["blocked"]  # noqa: S101
