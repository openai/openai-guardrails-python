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


def test_validate_url_security_blocks_password_without_username() -> None:
    """URLs that only include a password in userinfo must be blocked."""
    config = URLConfig(allowed_schemes={"https"}, block_userinfo=True)
    parsed, reason = _validate_url_security("https://:secret@example.com", config)

    assert parsed is None  # noqa: S101
    assert "userinfo" in reason  # noqa: S101


def test_url_config_normalizes_allowed_scheme_inputs() -> None:
    """URLConfig should accept schemes with delimiters and normalize them."""
    config = URLConfig(allowed_schemes={"HTTPS://", "http:", "  https  "})

    assert config.allowed_schemes == {"https", "http"}  # noqa: S101


def test_is_url_allowed_handles_full_urls_with_paths() -> None:
    """Allow list entries with schemes and paths should be honored."""
    config = URLConfig(
        url_allow_list=["https://suntropy.es", "https://api.example.com/v1"],
        allow_subdomains=False,
        allowed_schemes={"https://"},
    )
    root_url, _ = _validate_url_security("https://suntropy.es", config)
    path_url, _ = _validate_url_security("https://api.example.com/v1/resources?id=2", config)
    wrong_path_url, _ = _validate_url_security("https://api.example.com/v2", config)

    assert root_url is not None  # noqa: S101
    assert path_url is not None  # noqa: S101
    assert wrong_path_url is not None  # noqa: S101
    assert _is_url_allowed(root_url, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(path_url, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(wrong_path_url, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101


def test_is_url_allowed_respects_path_segment_boundaries() -> None:
    """Path matching should respect segment boundaries to prevent security issues."""
    config = URLConfig(
        url_allow_list=["https://example.com/api"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # These should be allowed
    exact_match, _ = _validate_url_security("https://example.com/api", config)
    valid_subpath, _ = _validate_url_security("https://example.com/api/users", config)

    # These should NOT be allowed (different path segments)
    similar_path1, _ = _validate_url_security("https://example.com/api2", config)
    similar_path2, _ = _validate_url_security("https://example.com/api-v2", config)

    assert exact_match is not None  # noqa: S101
    assert valid_subpath is not None  # noqa: S101
    assert similar_path1 is not None  # noqa: S101
    assert similar_path2 is not None  # noqa: S101

    # Exact match and valid subpath should be allowed
    assert _is_url_allowed(exact_match, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(valid_subpath, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101

    # Similar paths that don't respect segment boundaries should be blocked
    assert _is_url_allowed(similar_path1, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101
    assert _is_url_allowed(similar_path2, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101


def test_is_url_allowed_without_scheme_matches_multiple_protocols() -> None:
    """Scheme-less allow list entries should match any allowed scheme."""
    config = URLConfig(
        url_allow_list=["example.com"],
        allow_subdomains=False,
        allowed_schemes={"https", "http"},
    )
    https_result, https_reason = _validate_url_security("https://example.com", config)
    http_result, http_reason = _validate_url_security("http://example.com", config)

    assert https_result is not None, https_reason  # noqa: S101
    assert http_result is not None, http_reason  # noqa: S101
    assert _is_url_allowed(https_result, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(http_result, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101


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


@pytest.mark.asyncio
async def test_urls_guardrail_allows_full_url_configuration() -> None:
    """Reported regression: full URLs in config and schemes with delimiters should pass."""
    config = URLConfig(
        url_allow_list=["https://suntropy.es"],
        allowed_schemes={"https://"},
        block_userinfo=True,
        allow_subdomains=True,
    )
    text = "La url de la herramienta de estudios solares es: https://suntropy.es"

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["allowed"] == ["https://suntropy.es"]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


def test_url_config_rejects_invalid_scheme_types() -> None:
    """URLConfig should reject non-string scheme entries."""
    with pytest.raises(TypeError, match="allowed_schemes entries must be strings"):
        URLConfig(allowed_schemes={123, "https"})  # type: ignore[arg-type]


def test_url_config_rejects_empty_schemes() -> None:
    """URLConfig should reject empty scheme sets."""
    with pytest.raises(ValueError, match="must include at least one scheme"):
        URLConfig(allowed_schemes={"", "  "})


def test_validate_url_security_handles_malformed_urls() -> None:
    """Malformed URLs should be rejected with clear error messages."""
    config = URLConfig(allowed_schemes={"https"})
    parsed, reason = _validate_url_security("https://", config)

    assert parsed is None  # noqa: S101
    assert "Invalid URL" in reason  # noqa: S101


def test_is_url_allowed_handles_cidr_blocks() -> None:
    """CIDR blocks in allow list should match IP ranges."""
    config = URLConfig(
        url_allow_list=["10.0.0.0/8", "192.168.1.0/24"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # IPs within CIDR ranges
    ip_in_range1, _ = _validate_url_security("https://10.5.5.5", config)
    ip_in_range2, _ = _validate_url_security("https://192.168.1.100", config)
    # IP outside CIDR range
    ip_outside, _ = _validate_url_security("https://192.168.2.1", config)

    assert ip_in_range1 is not None  # noqa: S101
    assert ip_in_range2 is not None  # noqa: S101
    assert ip_outside is not None  # noqa: S101

    assert _is_url_allowed(ip_in_range1, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(ip_in_range2, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(ip_outside, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101


def test_is_url_allowed_handles_port_matching() -> None:
    """Allow list entries with explicit ports should require exact port match."""
    config = URLConfig(
        url_allow_list=["https://example.com:8443", "api.internal.com"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # Correct port
    correct_port, _ = _validate_url_security("https://example.com:8443", config)
    # Wrong port (implicit 443)
    wrong_port, _ = _validate_url_security("https://example.com", config)
    # Any port when not specified in allow list
    any_port, _ = _validate_url_security("https://api.internal.com:9000", config)

    assert correct_port is not None  # noqa: S101
    assert wrong_port is not None  # noqa: S101
    assert any_port is not None  # noqa: S101

    assert _is_url_allowed(correct_port, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(wrong_port, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101
    assert _is_url_allowed(any_port, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101


def test_is_url_allowed_handles_query_and_fragment() -> None:
    """Allow list entries with query/fragment should match exactly."""
    config = URLConfig(
        url_allow_list=["https://example.com/search?q=test", "https://example.com/docs#intro"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # Exact query match
    exact_query, _ = _validate_url_security("https://example.com/search?q=test", config)
    # Different query
    diff_query, _ = _validate_url_security("https://example.com/search?q=other", config)
    # Exact fragment match
    exact_fragment, _ = _validate_url_security("https://example.com/docs#intro", config)
    # Different fragment
    diff_fragment, _ = _validate_url_security("https://example.com/docs#outro", config)

    assert exact_query is not None  # noqa: S101
    assert diff_query is not None  # noqa: S101
    assert exact_fragment is not None  # noqa: S101
    assert diff_fragment is not None  # noqa: S101

    assert _is_url_allowed(exact_query, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(diff_query, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101
    assert _is_url_allowed(exact_fragment, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(diff_fragment, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101


def test_validate_url_security_allows_userinfo_when_disabled() -> None:
    """URLs with userinfo should be allowed when block_userinfo=False."""
    config = URLConfig(allowed_schemes={"https"}, block_userinfo=False)
    parsed, reason = _validate_url_security("https://user:pass@example.com", config)

    assert parsed is not None  # noqa: S101
    assert reason == ""  # noqa: S101


def test_is_url_allowed_enforces_scheme_when_explicitly_specified() -> None:
    """Scheme-qualified allow list entries must match scheme exactly (security)."""
    config = URLConfig(
        url_allow_list=["https://bank.example.com"],
        allow_subdomains=False,
        allowed_schemes={"https", "http"},  # Both schemes allowed globally
    )
    # HTTPS should be allowed (matches the scheme in allow list)
    https_url, _ = _validate_url_security("https://bank.example.com", config)
    # HTTP should be BLOCKED (doesn't match the explicit https:// in allow list)
    http_url, _ = _validate_url_security("http://bank.example.com", config)

    assert https_url is not None  # noqa: S101
    assert http_url is not None  # noqa: S101

    # This is the security-critical check: scheme-qualified entries must match exactly
    assert _is_url_allowed(https_url, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(http_url, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101


def test_is_url_allowed_enforces_scheme_for_ips() -> None:
    """Scheme-qualified IP addresses in allow list must match scheme exactly."""
    config = URLConfig(
        url_allow_list=["https://192.168.1.100"],
        allow_subdomains=False,
        allowed_schemes={"https", "http"},
    )
    # HTTPS should be allowed
    https_ip, _ = _validate_url_security("https://192.168.1.100", config)
    # HTTP should be BLOCKED
    http_ip, _ = _validate_url_security("http://192.168.1.100", config)

    assert https_ip is not None  # noqa: S101
    assert http_ip is not None  # noqa: S101

    assert _is_url_allowed(https_ip, config.url_allow_list, config.allow_subdomains) is True  # noqa: S101
    assert _is_url_allowed(http_ip, config.url_allow_list, config.allow_subdomains) is False  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_handles_malformed_ports_gracefully() -> None:
    """URLs with out-of-range or malformed ports should be blocked, not crash."""
    config = URLConfig(
        url_allow_list=["example.com"],
        allowed_schemes={"https"},
    )
    # Test various malformed ports
    text = "Visit https://example.com:99999 or https://example.com:abc or https://example.com:-1"

    result = await urls(ctx=None, data=text, config=config)

    # Should not crash; all should be blocked (either due to malformed ports or not in allow list)
    assert result.tripwire_triggered is True  # noqa: S101
    assert len(result.info["blocked"]) == 3  # noqa: S101
    # All three URLs should be blocked (the key is they don't crash the guardrail)
    assert len(result.info["blocked_reasons"]) == 3  # noqa: S101
