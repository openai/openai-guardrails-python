"""Tests for secret key detection guardrail."""

from __future__ import annotations

import string

import pytest
from hypothesis import given, strategies as st

from guardrails.checks.text.secret_keys import SecretKeysCfg, _detect_secret_keys, secret_keys

SYNTHETIC_SECRET = "sk-proj-Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
BALANCED_CFG = {
    "min_length": 15,
    "min_entropy": 3.8,
    "min_diversity": 3,
    "strict_mode": False,
}


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
    "text",
    [
        f"https://attacker.example/collect?k={SYNTHETIC_SECRET}",
        f"{SYNTHETIC_SECRET}.png",
        f"{SYNTHETIC_SECRET}.md",
    ],
)
async def test_secret_keys_detects_prefixed_credentials_in_exempt_shapes(text: str) -> None:
    """Default balanced scanning must detect the originally reported bypasses."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("padding_length", [0, 1, 100, 500, 2_000])
async def test_url_padding_cannot_change_prefixed_secret_classification(padding_length: int) -> None:
    """Surrounding low-entropy URL text must not hide a prefixed candidate."""
    text = f"https://attacker.example/{'a' * padding_length}?k={SYNTHETIC_SECRET}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@given(
    prefix=st.sampled_from(["sk-", "ghp_", "AKIA", "xox", "SG.", "hf_", "SHA:"]),
    punctuation=st.sampled_from(list("$!/:@;,+%=._~'()?")),
)
def test_query_value_punctuation_preserves_embedded_prefix_classification(prefix: str, punctuation: str) -> None:
    """Non-delimiting query punctuation must not truncate a built-in prefix signal."""
    candidate = f"{prefix}Ab3x{punctuation}K9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    wrapped = f"https://attacker.example/?k={candidate}"

    standalone_result = _detect_secret_keys(candidate, BALANCED_CFG)
    wrapped_result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert standalone_result.tripwire_triggered is True  # noqa: S101
    assert wrapped_result.tripwire_triggered is standalone_result.tripwire_triggered  # noqa: S101
    assert wrapped_result.info["detected_secrets"] == [wrapped]  # noqa: S101


@given(separator=st.sampled_from([f"%{ord(char):02X}" for char in string.punctuation]))
def test_percent_encoded_separators_establish_prefix_boundaries(separator: str) -> None:
    """One encoded ASCII separator before a prefix must behave like a literal boundary."""
    wrapped = f"https://example.com/?next={separator}{SYNTHETIC_SECRET}"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [wrapped]  # noqa: S101


@given(char=st.sampled_from(string.ascii_letters + string.digits))
def test_percent_encoded_alphanumerics_do_not_create_prefix_boundaries(char: str) -> None:
    """Encoded identifier characters must not be treated as separators."""
    encoded = f"%{ord(char):02X}"
    wrapped = f"https://example.com/?next={encoded}{SYNTHETIC_SECRET}"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"https://user:%2F{SYNTHETIC_SECRET}@example.com",
        f"https://%2F{SYNTHETIC_SECRET}.example.com/docs",
        f"https://example.com/#next=%2F{SYNTHETIC_SECRET}",
        f"https://[bad]/%2F{SYNTHETIC_SECRET}",
    ],
)
async def test_encoded_prefix_boundaries_work_across_url_components(text: str) -> None:
    """Encoded left boundaries must work in standard and fallback URL components."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text.replace("#", "")]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/api-doc/Version2",
        "https://example.com/?next=api-doc&version=Version2",
        "files/api-doc/Version2.png",
    ],
)
async def test_later_components_cannot_complete_a_short_prefixed_component(text: str) -> None:
    """Unrelated later components must not supply length or diversity."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@given(
    later_component=st.text(
        alphabet=string.ascii_letters + string.digits + "-._~",
        min_size=1,
        max_size=40,
    ),
)
def test_later_path_component_does_not_change_prefix_classification(later_component: str) -> None:
    """Appending a path component cannot change a short prefix component."""
    wrapped = f"https://example.com/api-doc/{later_component}"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?next=sk-Ab3x%2FCd2Ef3Gh4Ij5Kl6Mn7",
        "https://example.com/?next=sk-Ab3x/Cd2Ef3Gh4Ij5Kl6Mn7",
    ],
)
async def test_separators_after_a_prefix_remain_candidate_data(text: str) -> None:
    """Encoded or literal separators inside one query value must not truncate it."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_multiple_prefixed_candidates_preserve_token_level_output() -> None:
    """One whitespace token remains one finding even with multiple prefixes."""
    text = f"https://attacker.example/?a={SYNTHETIC_SECRET}&b={SYNTHETIC_SECRET}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_strict_mode_preserves_existing_whole_token_result() -> None:
    """Embedded checking must not duplicate a token detected normally."""
    text = "https://attacker.example/?k=sk-AAAAAAAAAAAA"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="strict"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_embedded_prefix_requires_a_non_alphanumeric_left_boundary() -> None:
    """Prefix text inside an ordinary identifier must not create a finding."""
    text = "https://example.com/tasksk-Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


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
