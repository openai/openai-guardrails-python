"""Tests for secret key detection guardrail."""

from __future__ import annotations

import string

import pytest
from hypothesis import given, strategies as st

from guardrails.checks.text.secret_keys import (
    ALLOWED_EXTENSIONS,
    SecretKeysCfg,
    _detect_secret_keys,
    secret_keys,
)

SYNTHETIC_SECRET = "sk-proj-Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"
DIRECT_PREFIXES = ["sk-", "sk_", "ghp_", "xoxb-", "xoxp-", "SG.", "hf_"]
GENERIC_PREFIXES = ["key-", "api-", "apikey-", "token-", "secret-", "xox"]
BENIGN_SLUG_WORDS = ["documentation", "version", "release", "reference", "management", "guide", "archive"]
BALANCED_CFG = {
    "min_length": 15,
    "min_entropy": 3.8,
    "min_diversity": 3,
    "strict_mode": False,
}


def _encode_every_byte(value: str, *, lowercase: bool = False) -> str:
    """Percent-encode every UTF-8 byte in a string."""
    encoded = "".join(f"%{byte:02X}" for byte in value.encode("utf-8"))
    return encoded.lower() if lowercase else encoded


@st.composite
def _encoded_prefixes(draw) -> tuple[str, str]:
    """Generate literal/mixed-case percent-encoded forms of direct prefixes."""
    prefix = draw(st.sampled_from(DIRECT_PREFIXES))
    encoded_parts: list[str] = []
    for char in prefix:
        if draw(st.booleans()):
            hex_value = f"{ord(char):02X}"
            if draw(st.booleans()):
                hex_value = hex_value.lower()
            encoded_parts.append(f"%{hex_value}")
        else:
            encoded_parts.append(char)
    return prefix, "".join(encoded_parts)


@st.composite
def _encoded_allowed_extensions(draw) -> tuple[str, str]:
    """Generate once-encoded variants of allowed ASCII file extensions."""
    extension = draw(st.sampled_from(ALLOWED_EXTENSIONS))
    encoded_parts: list[str] = []
    for index, char in enumerate(extension):
        encode_char = index == 0 or draw(st.booleans())
        if encode_char:
            hex_value = f"{ord(char):02X}"
            if draw(st.booleans()):
                hex_value = hex_value.lower()
            encoded_parts.append(f"%{hex_value}")
        else:
            encoded_parts.append(char)
    return extension, "".join(encoded_parts)


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
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
async def test_aws_access_key_id_detected_standalone_and_embedded(threshold: str) -> None:
    """AWS access-key IDs bypass generic diversity through their fixed format."""
    standalone = await secret_keys(None, AWS_ACCESS_KEY_ID, SecretKeysCfg(threshold=threshold))
    wrapped = f"https://example.com/?key={AWS_ACCESS_KEY_ID}"
    embedded = await secret_keys(None, wrapped, SecretKeysCfg(threshold=threshold))
    encoded = await secret_keys(
        None,
        f"https://example.com/?key=%41KIAIOSFODNN7EXAMPLE",
        SecretKeysCfg(threshold=threshold),
    )

    assert standalone.tripwire_triggered is True  # noqa: S101
    assert standalone.info["detected_secrets"] == [AWS_ACCESS_KEY_ID]  # noqa: S101
    assert embedded.tripwire_triggered is True  # noqa: S101
    assert embedded.info["detected_secrets"] == [wrapped]  # noqa: S101
    assert encoded.tripwire_triggered is True  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/AKIA-DOCUMENTATION-2026",
        "https://example.com/AKIAIOSFODNN7EXAMPLEEXTRA",
    ],
)
async def test_non_key_akia_slugs_remain_exempt(text: str) -> None:
    """AKIA only lifts a container exemption when the AWS key shape matches."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/api-documentation-v2",
        "https://example.com/api-documentation-version-2026-release",
        "https://example.com/token-documentation-v2",
        "https://example.com/key-release-v2-2026",
        "https://example.com/apikey-reference-v2",
        "https://example.com/secret-management-v2",
        "https://example.com/xoxo-valentine-2026",
        "https://example.com/pk_documentation_v2",
        "https://example.com/SHA:deadbeef-v2",
    ],
)
async def test_ordinary_prefixed_url_slugs_remain_exempt(text: str) -> None:
    """Lexical prefixes alone must not turn ordinary URL slugs into secrets."""
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", GENERIC_PREFIXES)
async def test_high_entropy_generic_prefixes_do_not_lift_url_exemptions(prefix: str) -> None:
    """Generic lexical prefixes remain exempt even with random-looking suffixes."""
    candidate = f"{prefix}Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    text = f"https://example.com/{candidate}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@given(
    prefix=st.sampled_from(GENERIC_PREFIXES),
    words=st.lists(st.sampled_from(BENIGN_SLUG_WORDS), min_size=2, max_size=6),
    year=st.integers(min_value=2000, max_value=2099),
)
def test_generic_prefixed_benign_slug_property(prefix: str, words: list[str], year: int) -> None:
    """Natural-language URL slugs never lift exemptions through generic prefixes."""
    text = f"https://example.com/{prefix}{'-'.join(words)}-{year}"
    result = _detect_secret_keys(text, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@given(
    prefix=st.sampled_from(DIRECT_PREFIXES),
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


@given(encoded_prefix=_encoded_prefixes())
def test_percent_encoding_cannot_change_prefix_classification(encoded_prefix: tuple[str, str]) -> None:
    """Literal and once-encoded forms of a built-in prefix must classify alike."""
    prefix, encoded = encoded_prefix
    payload = "Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    literal = f"https://example.com/?k={prefix}{payload}"
    transformed = f"https://example.com/?k={encoded}{payload}"

    literal_result = _detect_secret_keys(literal, BALANCED_CFG)
    transformed_result = _detect_secret_keys(transformed, BALANCED_CFG)

    assert literal_result.tripwire_triggered is True  # noqa: S101
    assert transformed_result.tripwire_triggered is literal_result.tripwire_triggered  # noqa: S101
    assert transformed_result.info["detected_secrets"] == [transformed]  # noqa: S101


@pytest.mark.parametrize(
    "encoded_prefix",
    [
        "sk%2D",
        "sk%2d",
        "%73k-",
        "%73%6B%2D",
        "%73%6b%2d",
    ],
)
def test_encoded_prefix_regressions(encoded_prefix: str) -> None:
    """Known mixed literal/encoded prefix forms must be detected."""
    wrapped = f"https://example.com/?k={encoded_prefix}Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [wrapped]  # noqa: S101


@given(invalid_byte=st.integers(min_value=0x80, max_value=0xFF))
def test_invalid_utf8_escape_cannot_hide_later_encoded_prefix(invalid_byte: int) -> None:
    """Malformed UTF-8 escapes must not suppress later valid encoded data."""
    invalid_escape = f"%{invalid_byte:02X}"
    wrapped = (
        "https://example.com/?k="
        f"{invalid_escape}%2F%73%6B%2DAb3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    )
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [wrapped]  # noqa: S101


@pytest.mark.parametrize("malformed_escape", ["%GZ", "%A"])
def test_malformed_escape_cannot_hide_later_encoded_prefix(malformed_escape: str) -> None:
    """Malformed percent syntax must not suppress later valid encoded data."""
    wrapped = (
        "https://example.com/?k="
        f"{malformed_escape}%2F%73%6B%2DAb3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    )
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [wrapped]  # noqa: S101


@pytest.mark.parametrize(
    "double_encoded_prefix",
    [
        "sk%252D",
        "%2573k-",
        "%2573%256B%252D",
    ],
)
def test_percent_decoding_occurs_exactly_once(double_encoded_prefix: str) -> None:
    """Double-encoded prefixes must not be recursively normalized."""
    wrapped = f"https://example.com/?k={double_encoded_prefix}Ab3xK9mQ7zR2wT5vY8nL4pJ6hG1dF0sC"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


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


@pytest.mark.parametrize(
    "identifier",
    [
        "café",
        "咖啡",
        "Ω9",
        "١",
        "e\u0301",
        "join\u200c",
        "join\u200d",
    ],
)
def test_unicode_identifier_characters_do_not_create_prefix_boundaries(identifier: str) -> None:
    """Internationalized identifier continuations must not create a boundary."""
    wrapped = f"https://example.com/{identifier}{SYNTHETIC_SECRET}"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.parametrize("identifier", ["café", "咖啡", "Ω9", "e\u0301"])
def test_percent_encoded_unicode_identifiers_do_not_create_boundaries(identifier: str) -> None:
    """Percent-encoded Unicode identifier characters must remain non-boundaries."""
    encoded = _encode_every_byte(identifier)
    wrapped = f"https://example.com/{encoded}{SYNTHETIC_SECRET}"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.parametrize("separator", ["—", "、", "：", "💥"])
def test_unicode_punctuation_can_establish_prefix_boundaries(separator: str) -> None:
    """Unicode punctuation and symbols may separate a prefixed credential."""
    wrapped = f"https://example.com/?next={separator}{SYNTHETIC_SECRET}"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [wrapped]  # noqa: S101


@pytest.mark.parametrize("invalid_prefix", ["%FF", "%GZ", "%A"])
def test_invalid_percent_sequences_do_not_manufacture_boundaries(invalid_prefix: str) -> None:
    """Malformed escapes must remain raw and must not create replacement boundaries."""
    wrapped = f"https://example.com/?next={invalid_prefix}{SYNTHETIC_SECRET}"
    result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@given(
    extension=st.sampled_from(ALLOWED_EXTENSIONS),
    container=st.sampled_from(["{candidate}{extension}", "files/{candidate}{extension}", "https://example.com/{candidate}{extension}"]),
)
def test_allowed_extension_cannot_complete_short_prefixed_candidate(extension: str, container: str) -> None:
    """Allowed file syntax must not supply length or diversity to a short prefix."""
    candidate = "sk-ABCDEFGHIJ1"
    text = container.format(candidate=candidate, extension=extension)
    result = _detect_secret_keys(text, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@given(encoded_extension=_encoded_allowed_extensions())
def test_encoded_allowed_extension_matches_literal_extension_classification(encoded_extension: tuple[str, str]) -> None:
    """Once-encoding an allowed URL-path extension must not change classification."""
    extension, encoded = encoded_extension
    candidate = "sk-ABCDEFGHIJ1"
    literal = f"https://example.com/{candidate}{extension}"
    transformed = f"https://example.com/{candidate}{encoded}"

    literal_result = _detect_secret_keys(literal, BALANCED_CFG)
    transformed_result = _detect_secret_keys(transformed, BALANCED_CFG)

    assert literal_result.tripwire_triggered is False  # noqa: S101
    assert transformed_result.tripwire_triggered is literal_result.tripwire_triggered  # noqa: S101
    assert transformed_result.info["detected_secrets"] == []  # noqa: S101


@given(
    wrapper=st.sampled_from([("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"), ('"', '"'), ("'", "'")]),
    punctuation=st.sampled_from(list(".,;:!")),
)
def test_url_presentation_wrapping_cannot_complete_short_candidate(
    wrapper: tuple[str, str], punctuation: str
) -> None:
    """Confirmed presentation wrapping must not change a clean URL's classification."""
    opener, closer = wrapper
    base = "https://example.com/sk-ABCDEFGHIJ1"
    text = f"{opener}{base}{closer}{punctuation}"
    result = _detect_secret_keys(text, BALANCED_CFG)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


def test_terminal_uri_punctuation_preserves_prefixed_classification() -> None:
    """Terminal URI punctuation must remain credential data without a wrapper."""
    candidate = "sk-abcdefghij1!"
    wrapped = f"https://example.com/{candidate}"

    standalone_result = _detect_secret_keys(candidate, BALANCED_CFG)
    wrapped_result = _detect_secret_keys(wrapped, BALANCED_CFG)

    assert standalone_result.tripwire_triggered is True  # noqa: S101
    assert wrapped_result.tripwire_triggered is standalone_result.tripwire_triggered  # noqa: S101
    assert wrapped_result.info["detected_secrets"] == [wrapped]  # noqa: S101


@pytest.mark.asyncio
async def test_url_presentation_wrapping_preserves_strong_secret_detection() -> None:
    """Presentation trimming must not hide a genuine embedded credential."""
    text = f"({f'https://example.com/{SYNTHETIC_SECRET}'})."
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101


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
        "https://example.com/?next=api-doc%20Version2",
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
    text = f"https://attacker.example/?k={SYNTHETIC_SECRET}"
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
