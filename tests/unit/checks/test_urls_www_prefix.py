"""Regression coverage for URL host normalization."""

from __future__ import annotations

import pytest

from guardrails.checks.text.urls import URLConfig, _detect_urls, _is_url_allowed, _strip_www_prefix, _validate_url_security, urls


def test_strip_www_prefix_removes_only_leading_label() -> None:
    assert _strip_www_prefix("www.example.com") == "example.com"  # noqa: S101
    assert _strip_www_prefix("api.www.example.com") == "api.www.example.com"  # noqa: S101
    assert _strip_www_prefix("example.com") == "example.com"  # noqa: S101


@pytest.mark.parametrize(
    ("url", "allowed_host", "expected"),
    [
        ("https://exwww.ample.com/path", "example.com", False),
        ("https://example.com/path", "exwww.ample.com", False),
        ("https://www.example.com/path", "example.com", True),
        ("https://example.com/path", "www.example.com", True),
    ],
)
def test_url_allow_list_preserves_host_identity(url: str, allowed_host: str, expected: bool) -> None:
    config = URLConfig(url_allow_list=[allowed_host], allow_subdomains=False)
    parsed, reason, had_scheme = _validate_url_security(url, config)

    assert parsed is not None, reason  # noqa: S101
    assert _is_url_allowed(parsed, config.url_allow_list, config.allow_subdomains, had_scheme) is expected  # noqa: S101


def test_url_detection_does_not_collapse_interior_www_host() -> None:
    detected = _detect_urls("https://exwww.ample.com and example.com")

    assert "https://exwww.ample.com" in detected  # noqa: S101
    assert "example.com" in detected  # noqa: S101


@pytest.mark.asyncio
async def test_url_guardrail_blocks_interior_www_lookalike() -> None:
    config = URLConfig(url_allow_list=["example.com"], allowed_schemes={"https"}, allow_subdomains=False)
    result = await urls(None, "Visit https://exwww.ample.com/path", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert "https://exwww.ample.com/path" in result.info["blocked"]  # noqa: S101
