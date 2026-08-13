"""Regression coverage for Secret Keys URL/file exemptions."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from guardrails.checks.text.secret_keys import (
    SecretKeysCfg,
    _embedded_secret_candidates,
    _is_sensitive_parameter,
    secret_keys,
)

SECRET = "sk-AAAABBBBCCCCDDDD"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://example.com/?token=sk-AAAABBBBCCCCDDDD", SECRET),
        ("https://example.com/#access_token=sk-AAAABBBBCCCCDDDD", SECRET),
        ("https://example.com/?client_secret=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/#refresh_token=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/?clientSecret=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/#refreshToken=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/?token=Aa0Bb1Cc2Dd3.json", "Aa0Bb1Cc2Dd3.json"),
        ("https://example.com/token/Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/token=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/#token/Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/#token=Aa0Bb1Cc2Dd3Ee4Ff5", "Aa0Bb1Cc2Dd3Ee4Ff5"),
        ("https://example.com/sk%2DAAAABBBBCCCCDDDD", SECRET),
        ("https://example.com/foo%2Fsk-AAAABBBBCCCCDDDD", SECRET),
        ("https://example.com/#foo%2Fsk-AAAABBBBCCCCDDDD", SECRET),
        ("https://[bad]/sk-AAAABBBBCCCCDDDD", SECRET),
        ("files/sk-AAAABBBBCCCCDDDD.png", SECRET),
        ("files/sk-AAAABBBBCCCCDDDD/image.png", SECRET),
        (r"files\sk-AAAABBBBCCCCDDDD\image.png", SECRET),
        ("files%2Fsk-AAAABBBBCCCCDDDD%2Fimage.png", SECRET),
        ("https://example.com/sk-AAAABBBBCCCCDDDD.png?download=1", SECRET),
        ("https://user:sk-AAAABBBBCCCCDDDD@example.com", SECRET),
        ("https://sk%2DAAAABBBBCCCCDDDD@example.com", SECRET),
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
        "https://example.com/value/Aa0Bb1Cc2Dd3Ee4Ff5",
        "files/Aa0Bb1Cc2Dd3Ee4F.png",
        "files/archive/image.png",
    ],
)
async def test_secret_keys_keeps_benign_allowed_patterns_exempt(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?redirect=https%3A%2F%2Fother.example%2F%3Ftoken%3DAa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/?redirect=https%3A%2F%2Fother.example%2Ftoken%2FAa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/#redirect=https%3A%2F%2Fother.example%2F%3Ftoken%3DAa0Bb1Cc2Dd3Ee4Ff5",
    ],
)
async def test_secret_keys_checks_nested_url_parameter_values(text: str) -> None:
    expected = "Aa0Bb1Cc2Dd3Ee4Ff5"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_keeps_benign_nested_url_exempt() -> None:
    text = "https://example.com/?redirect=https%3A%2F%2Fother.example%2Fdocs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101


@given(
    separator=st.sampled_from(["/", "\\", "%2F", "%2f", "%5C", "%5c"]),
    location=st.sampled_from(["url_path", "fragment", "file"]),
    authority=st.sampled_from(["example.com", "[bad]"]),
)
def test_embedded_candidate_separator_encodings_are_equivalent(separator: str, location: str, authority: str) -> None:
    if location == "url_path":
        token = f"https://{authority}/foo{separator}{SECRET}"
    elif location == "fragment":
        token = f"https://{authority}/#foo{separator}{SECRET}"
    else:
        token = f"files{separator}{SECRET}{separator}image.png"

    assert SECRET in _embedded_secret_candidates(token)  # noqa: S101


@given(
    prefix=st.sampled_from(["client", "refresh", "access", "api"]),
    kind=st.sampled_from(["token", "secret", "key", "password"]),
    style=st.sampled_from(["snake", "kebab", "dot", "camel"]),
)
def test_sensitive_parameter_normalization_is_style_invariant(prefix: str, kind: str, style: str) -> None:
    if style == "camel":
        name = f"{prefix}{kind.title()}"
    else:
        separator = {"snake": "_", "kebab": "-", "dot": "."}[style]
        name = f"{prefix}{separator}{kind}"

    assert _is_sensitive_parameter(name) is True  # noqa: S101


@given(extension=st.sampled_from([".json", ".png", ".txt", ".md"]))
def test_sensitive_parameter_values_preserve_extensions(extension: str) -> None:
    value = f"Aa0Bb1Cc2Dd3{extension}"
    candidates = _embedded_secret_candidates(f"https://example.com/?token={value}")

    assert value in candidates  # noqa: S101


@given(
    name=st.sampled_from(["token", "client_secret", "refresh-token", "apiKey"]),
    form=st.sampled_from(["pair", "assignment"]),
    location=st.sampled_from(["path", "fragment"]),
)
def test_sensitive_path_labels_emit_associated_values(name: str, form: str, location: str) -> None:
    value = "Aa0Bb1Cc2Dd3Ee4Ff5"
    component = f"{name}/{value}" if form == "pair" else f"{name}={value}"
    token = f"https://example.com/{component}" if location == "path" else f"https://example.com/#{component}"

    assert value in _embedded_secret_candidates(token)  # noqa: S101
