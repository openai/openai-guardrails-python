"""Regression coverage for Secret Keys URL/file exemptions."""

from __future__ import annotations

import pytest

from guardrails.checks.text.secret_keys import SecretKeysCfg, secret_keys


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://example.com/?token=sk-AAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("https://example.com/#access_token=sk-AAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("https://example.com/?client_secret=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/#refresh_token=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/?clientSecret=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/#refreshToken=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/?token=Aa0Bb1Cc2Dd3.json", "Aa0Bb1Cc2Dd3.json"),
        ("https://example.com/sk%2DAAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("https://example.com/foo%2Fsk-AAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("https://example.com/#foo%2Fsk-AAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("files/sk-AAAABBBBCCCCDDDD.png", "sk-AAAABBBBCCCCDDDD"),
        ("files/sk-AAAABBBBCCCCDDDD/image.png", "sk-AAAABBBBCCCCDDDD"),
        (r"files\sk-AAAABBBBCCCCDDDD\image.png", "sk-AAAABBBBCCCCDDDD"),
        ("files%2Fsk-AAAABBBBCCCCDDDD%2Fimage.png", "sk-AAAABBBBCCCCDDDD"),
        ("https://example.com/sk-AAAABBBBCCCCDDDD.png?download=1", "sk-AAAABBBBCCCCDDDD"),
        ("https://user:sk-AAAABBBBCCCCDDDD@example.com", "sk-AAAABBBBCCCCDDDD"),
        ("https://sk%2DAAAABBBBCCCCDDDD@example.com", "sk-AAAABBBBCCCCDDDD"),
        ("https://user:Aa0Bb1Cc2Dd3.json@example.com", "Aa0Bb1Cc2Dd3.json"),
    ],
)
async def test_secret_keys_checks_embedded_candidates(text: str, expected: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/internal-ab12",
        "https://example.com/?value=internal-ab12",
        "https://example.com/#value=internal-ab12",
        "files/internal-ab12.png",
        "files/internal-ab12/image.png",
    ],
)
async def test_secret_keys_preserves_custom_regex_inside_exempt_tokens(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/internal-ab12.png",
        "files/internal-ab12.png",
    ],
)
async def test_secret_keys_preserves_custom_regex_that_includes_extension(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}\.png$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12.png" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_extracts_embedded_candidates_in_strict_mode() -> None:
    result = await secret_keys(None, "https://a.co/sk-AAAAAAAAAAAA", SecretKeysCfg(threshold="strict"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert "sk-AAAAAAAAAAAA" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "(https://example.com/v1/docs)",
        "https://example.com/?file=Aa0Bb1Cc2Dd3.json",
        "https://example.com/Aa0Bb1Cc2Dd3Ee4F",
        "files/Aa0Bb1Cc2Dd3Ee4F.png",
        "files/archive/image.png",
    ],
)
async def test_secret_keys_keeps_benign_allowed_patterns_exempt(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101
