"""Focused regression coverage for parser edge cases."""

from __future__ import annotations

import re
import sys
from importlib import import_module
from urllib.parse import quote

import pytest
from hypothesis import given, strategies as st

from guardrails.checks.text.secret_keys import (
    SecretKeysCfg,
    _contains_allowed_pattern,
    _embedded_secret_candidates,
    _normalize_allowed_token,
    _strip_closed_presentation_regions,
    secret_keys,
)

SYNTHETIC_VALUE = "Aa0Bb1Cc2Dd3Ee4Ff5"
PREFIXED_SECRET = "sk-AAAABBBBCCCCDDDD"
ENCODED_SEPARATORS = (("%2F", "/"), ("%2f", "/"), ("%5C", "\\"), ("%5c", "\\"))


def test_long_unmatched_presentation_prefix_is_normalized_in_one_pass() -> None:
    assert _normalize_allowed_token("!" * 20_000 + "a.png") == "a.png"  # noqa: S101


def test_long_markdown_like_prefix_is_normalized_in_one_pass() -> None:
    assert _normalize_allowed_token("](" * 20_000 + "a.png") == "a.png"  # noqa: S101


def test_long_non_file_token_uses_linear_extension_check() -> None:
    assert _contains_allowed_pattern("!" * 40_000 + "plain") is False  # noqa: S101


def test_many_nonsensitive_assignments_are_scanned_linearly() -> None:
    token = "files/" + "a:" * 2_000 + "benign.json"

    assert _embedded_secret_candidates(token) == ()  # noqa: S101


@given(separator=st.sampled_from(["/", "\\", "%2F", "%2f", "%5C", "%5c"]))
def test_file_path_label_pairs_with_following_value(separator: str) -> None:
    token = f"files{separator}token{separator}{SYNTHETIC_VALUE}{separator}image.png"

    assert SYNTHETIC_VALUE in _embedded_secret_candidates(token)  # noqa: S101


@given(
    encoded_and_decoded_separator=st.sampled_from(ENCODED_SEPARATORS),
    location=st.sampled_from(["url_path", "fragment", "file"]),
)
def test_encoded_separator_within_secret_preserves_whole_candidate(
    encoded_and_decoded_separator: tuple[str, str],
    location: str,
) -> None:
    encoded_separator, decoded_separator = encoded_and_decoded_separator
    encoded_secret = f"sk-Ab1{encoded_separator}Cd2Ef3Gh4Ij5Kl6"
    decoded_secret = f"sk-Ab1{decoded_separator}Cd2Ef3Gh4Ij5Kl6"
    if location == "url_path":
        token = f"https://example.com/{encoded_secret}"
    elif location == "fragment":
        token = f"https://example.com/#{encoded_secret}"
    else:
        token = f"files/{encoded_secret}/image.png"

    assert decoded_secret in _embedded_secret_candidates(token)  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
@pytest.mark.parametrize(("encoded_separator", "decoded_separator"), ENCODED_SEPARATORS)
async def test_secret_keys_detects_encoded_separator_within_secret(
    threshold: str,
    encoded_separator: str,
    decoded_separator: str,
) -> None:
    encoded_secret = f"sk-Ab1{encoded_separator}Cd2Ef3Gh4Ij5Kl6"
    decoded_secret = f"sk-Ab1{decoded_separator}Cd2Ef3Gh4Ij5Kl6"
    result = await secret_keys(
        None,
        f"https://example.com/{encoded_secret}",
        SecretKeysCfg(threshold=threshold, custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [decoded_secret]  # noqa: S101


@given(
    boundary=st.sampled_from(ENCODED_SEPARATORS),
    internal=st.sampled_from(ENCODED_SEPARATORS),
    location=st.sampled_from(["url_path", "fragment", "file"]),
)
def test_mixed_separator_roles_preserve_secret_suffix(
    boundary: tuple[str, str],
    internal: tuple[str, str],
    location: str,
) -> None:
    encoded_boundary, _ = boundary
    encoded_internal, decoded_internal = internal
    encoded_secret = f"sk-Ab1{encoded_internal}Cd2Ef3Gh4Ij5Kl6"
    decoded_secret = f"sk-Ab1{decoded_internal}Cd2Ef3Gh4Ij5Kl6"
    component = f"prefix{encoded_boundary}{encoded_secret}"
    if location == "url_path":
        token = f"https://example.com/{component}"
    elif location == "fragment":
        token = f"https://example.com/#{component}"
    else:
        token = f"files/{component}/image.png"

    assert decoded_secret in _embedded_secret_candidates(token)  # noqa: S101


@given(
    boundary=st.sampled_from(ENCODED_SEPARATORS),
    internal=st.sampled_from(ENCODED_SEPARATORS),
)
def test_custom_pattern_selects_mixed_separator_range(
    boundary: tuple[str, str],
    internal: tuple[str, str],
) -> None:
    encoded_boundary, _ = boundary
    encoded_internal, decoded_internal = internal
    expected = f"internal{decoded_internal}ab12"
    token = f"https://example.com/prefix{encoded_boundary}internal{encoded_internal}ab12"
    candidates = _embedded_secret_candidates(token, [f"{re.escape(expected)}$"])

    assert expected in candidates  # noqa: S101


@given(
    label=st.sampled_from(["token", "client_secret", "refresh-token", "apiKey"]),
    boundary=st.sampled_from(ENCODED_SEPARATORS),
    internal=st.sampled_from(ENCODED_SEPARATORS),
)
def test_sensitive_label_selects_mixed_separator_range(
    label: str,
    boundary: tuple[str, str],
    internal: tuple[str, str],
) -> None:
    encoded_boundary, _ = boundary
    encoded_internal, decoded_internal = internal
    encoded_value = f"Aa0Bb1{encoded_internal}Cd2Ef3Gh4Ij5Kl6"
    decoded_value = f"Aa0Bb1{decoded_internal}Cd2Ef3Gh4Ij5Kl6"
    token = f"https://example.com/{label}{encoded_boundary}{encoded_value}"

    assert decoded_value in _embedded_secret_candidates(token)  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/internal%252Fab12",
        "https://example.com/?token=internal%252Fab12",
        "https://example.com/#token=internal%252Fab12",
        "files/internal%252Fab12/image.png",
    ],
)
async def test_each_syntax_layer_is_percent_decoded_once(text: str) -> None:
    expected = "internal%2Fab12"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[rf"{re.escape(expected)}$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("https://example.com/?value=prefix%2Fsk-AAAABBBBCCCCDDDD", "sk-AAAABBBBCCCCDDDD"),
        ("https://user:prefix%2Fsk-AAAABBBBCCCCDDDD@example.com", "sk-AAAABBBBCCCCDDDD"),
        ("https://prefix.sk-AAAABBBBCCCCDDDD.example.com", "sk-AAAABBBBCCCCDDDD"),
        ("https://example.com:sk-AAAABBBBCCCCDDDD/docs", "sk-AAAABBBBCCCCDDDD"),
        ("https://SHA:Aa0Bb1Cc2Dd3Ee4Ff5/docs", "SHA:Aa0Bb1Cc2Dd3Ee4Ff5"),
    ],
)
async def test_secret_keys_scans_all_url_component_boundaries(text: str, expected: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?sk-AAAABBBBCCCCDDDD=1",
        "https://example.com/?sk-AAAABBBBCCCCDDDD",
        "https://example.com/?prefix%2Fsk-AAAABBBBCCCCDDDD=1",
        "https://example.com/#sk-AAAABBBBCCCCDDDD=1",
    ],
)
async def test_secret_keys_scans_query_and_fragment_names(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert PREFIXED_SECRET in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?internal-ab12=1",
        "https://example.com/?internal-ab12",
        "https://example.com/#internal-ab12=1",
    ],
)
async def test_custom_pattern_scans_query_and_fragment_names(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?client_secret[0]=Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/?client_secret/Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/#refresh_token[0]=Aa0Bb1Cc2Dd3Ee4Ff5",
    ],
)
async def test_sensitive_parameter_variants_emit_values(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert SYNTHETIC_VALUE in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '"https://example.com/internal-ab12"',
        "'https://example.com/internal-ab12'",
        "<https://example.com/internal-ab12>",
        "`https://example.com/internal-ab12`",
    ],
)
async def test_wrapped_urls_preserve_custom_pattern_boundaries(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == ["internal-ab12"]  # noqa: S101


@pytest.mark.asyncio
async def test_nested_url_worklist_has_no_predictable_depth_bypass() -> None:
    text = "https://inner.example/?value=internal-ab12"
    for _ in range(5):
        text = f"https://outer.example/?redirect={quote(text, safe='')}"

    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "config", "expected"),
    [
        (
            "https://SG.Aa0Bb1Cc2Dd3Ee4Ff5.example.com/docs",
            SecretKeysCfg(threshold="balanced", custom_regex=None),
            "SG.Aa0Bb1Cc2Dd3Ee4Ff5",
        ),
        (
            "https://Case.Ab12.example.com/docs",
            SecretKeysCfg(threshold="balanced", custom_regex=[r"Case\.Ab12$"]),
            "Case.Ab12",
        ),
        (
            "https://[bad]sk-AAAABBBBCCCCDDDD/docs",
            SecretKeysCfg(threshold="balanced", custom_regex=None),
            PREFIXED_SECRET,
        ),
    ],
)
async def test_raw_authority_candidates_preserve_exact_spelling(
    text: str,
    config: SecretKeysCfg,
    expected: str,
) -> None:
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/sk-x/Docs2026/guide",
        "https://example.com/token/docs/ApiV2-Release2026/reference",
        "files/token/docs/ApiV2-Release2026/reference/image.png",
    ],
)
async def test_literal_path_boundaries_do_not_create_secret_candidates(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/sk-AAAABBBBCCCCDDDD/docs/index.html",
        "https://prefix.sk-AAAABBBBCCCCDDDD.example.com/docs",
    ],
)
async def test_literal_boundaries_do_not_pollute_detected_secret_values(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
async def test_duplicate_standalone_secrets_preserve_occurrence_count() -> None:
    result = await secret_keys(
        None,
        f"{PREFIXED_SECRET} {PREFIXED_SECRET}",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.info["detected_secrets"] == [PREFIXED_SECRET, PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        (
            "https://example.com/prefix%2FInternal%2FAb12%2Ftail",
            r"(?i)^internal/ab12$",
            "Internal/Ab12",
        ),
        (
            "https://example.com/prefix%2Finternal%5Cab12%2Ftail",
            r"\Ainternal\\ab12\Z",
            r"internal\ab12",
        ),
    ],
)
async def test_custom_patterns_keep_anchor_semantics_across_encoded_boundaries(
    text: str,
    pattern: str,
    expected: str,
) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=[pattern]))

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_custom_pattern_spans_three_hostname_labels() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"Case\.Ab12\.Xy34$"])
    result = await secret_keys(None, "https://Case.Ab12.Xy34.example.com/docs", config)

    assert result.info["detected_secrets"] == ["Case.Ab12.Xy34"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        'https://example.com"sk-AAAABBBBCCCCDDDD',
        "https://example.com|sk-AAAABBBBCCCCDDDD",
        "https://example.com,sk-AAAABBBBCCCCDDDD",
        "<https://example.com>sk-AAAABBBBCCCCDDDD",
    ],
)
async def test_text_outside_url_span_is_not_exempt(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert PREFIXED_SECRET in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "“https://example.com/internal-ab12”",
        "«https://example.com/internal-ab12»",
        "https://example.com/internal-ab12…",
        "https://example.com/internal-ab12。",
        "https://example.com/internal-ab12）",
    ],
)
async def test_unicode_prose_wrappers_do_not_change_custom_boundaries(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal-ab12"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/internal-ab12，下一步",
        "https://example.com/internal-ab12—next",
        "https://example.com/?value=internal-ab12，&next=1",
        "https://example.com/internal-ab12，/next",
        "files/internal-ab12，/next/image.png",
    ],
)
async def test_unicode_sentence_punctuation_ends_custom_candidate(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, text, config)

    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://[bad:sk-AAAABBBBCCCCDDDD]/docs",
        "https://[::sk-AAAABBBBCCCCDDDD]/docs",
        "https://[bad%3Ask-AAAABBBBCCCCDDDD]/docs",
        "https://[bad]foo:sk-AAAABBBBCCCCDDDD/docs",
    ],
)
async def test_malformed_authority_boundaries_do_not_hide_prefixes(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert PREFIXED_SECRET in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com:Aa0Bb1Cc2Dd3Ee4Ff5/docs",
        "https://[::1]:Aa0Bb1Cc2Dd3Ee4Ff5/docs",
        "https://[bad]Aa0Bb1Cc2Dd3Ee4Ff5/docs",
    ],
)
async def test_malformed_authority_boundaries_do_not_hide_generic_secrets(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?client_secret[foo]=Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/?client_secret[0][foo]=Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/?client_secret.0=Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/?clientSecret0=Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/client_secret/0/Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/#client_secret/[foo]/Aa0Bb1Cc2Dd3Ee4Ff5",
        "files/client_secret/0/Aa0Bb1Cc2Dd3Ee4Ff5/image.png",
    ],
)
async def test_sensitive_container_indexes_preserve_associated_value(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert SYNTHETIC_VALUE in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("leading", ["%09", "%20", "%22"])
async def test_nested_url_values_allow_leading_wrappers(leading: str) -> None:
    inner = "https%3A%2F%2Finner.example%2Finternal-ab12"
    text = f"https://outer.example/?redirect={leading}{inner}"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, text, config)

    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_nested_url_candidates_preserve_text_order() -> None:
    first = "https://inner.example/?value=custom-one"
    second = "https://inner.example/?value=custom-two"
    for _ in range(3):
        first = f"https://wrapper.example/?redirect={quote(first, safe='')}"
    for _ in range(1):
        second = f"https://wrapper.example/?redirect={quote(second, safe='')}"
    text = f"https://outer.example/?first={quote(first, safe='')}&second={quote(second, safe='')}"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"custom-(?:one|two)$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["custom-one", "custom-two"]  # noqa: S101


@pytest.mark.asyncio
async def test_nested_candidate_precedes_later_direct_value() -> None:
    first = "https://inner.example/?value=custom-one"
    text = f"https://outer.example/?first={quote(first, safe='')}&second=custom-two"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"custom-(?:one|two)$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["custom-one", "custom-two"]  # noqa: S101


@pytest.mark.asyncio
async def test_repeated_nested_url_values_preserve_occurrences() -> None:
    """Scan equal sibling URLs separately while retaining cycle protection."""
    secret = "Aa0Bb1Cc2Dd3Ee4Ff5"
    nested = f"https://inner.example/?token={secret}"
    encoded = quote(nested, safe="")
    text = f"https://outer.example/?first={encoded}&second={encoded}"

    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [secret, secret]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("threshold", "credential"),
    [("balanced", SYNTHETIC_VALUE), ("permissive", f"{SYNTHETIC_VALUE}Gh")],
)
async def test_percent_encoded_url_path_segment_preserves_nested_authority(
    threshold: str,
    credential: str,
) -> None:
    nested = quote(f"https://{credential}@second.example/docs", safe="")
    text = f"https://first.example/{nested}"

    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [credential]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("threshold", "credential"),
    [("balanced", SYNTHETIC_VALUE), ("permissive", f"{SYNTHETIC_VALUE}Gh")],
)
async def test_percent_encoded_scheme_relative_value_preserves_nested_authority(
    threshold: str,
    credential: str,
) -> None:
    nested = quote(f"//{credential}@second.example/docs", safe="")
    text = f"https://first.example/?redirect={nested}"

    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [credential]  # noqa: S101


@pytest.mark.asyncio
async def test_nested_url_scan_budget_fails_closed() -> None:
    text = "https://inner.example/docs"
    for _ in range(400):
        text = f"https://outer.example/?redirect={text}"

    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_encoded_route_suffix_preserves_definite_candidate() -> None:
    text = "https://example.com/sk-AAAABBBBCCCCDDDD%2Fdocs%2Findex.html"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
async def test_long_prefix_before_encoded_separator_preserves_full_secret() -> None:
    expected = "sk-AAAABBBBCCCCDDDD/Cd2Ef3Gh4Ij5Kl6"
    text = "https://example.com/sk-AAAABBBBCCCCDDDD%2FCd2Ef3Gh4Ij5Kl6"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
async def test_query_value_prefers_whole_decoded_candidate() -> None:
    expected = "sk-AAAABBBBCCCCDDDD/docs"
    text = "https://example.com/?token=sk-AAAABBBBCCCCDDDD%2Fdocs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
async def test_candidate_canonicalization_is_scoped_to_each_component() -> None:
    encoded = "sk-Aa1Bb2Cc3Dd4Ee5Ff6%2Fdocs"
    text = f"https://example.com/{encoded}?token={encoded}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [
        "sk-Aa1Bb2Cc3Dd4Ee5Ff6",
        "sk-Aa1Bb2Cc3Dd4Ee5Ff6/docs",
    ]  # noqa: S101


@pytest.mark.asyncio
async def test_encoded_candidate_preserves_order_before_later_path_secret() -> None:
    first = "sk-Aa1Bb2Cc3Dd4Ee5Ff6/Gg7Hh8Ii9Jj0Kk1Ll2"
    second = "sk-Qq1Ww2Ee3Rr4Tt5Yy6"
    text = f"https://example.com/{first.replace('/', '%2F')}/{second}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [first, second]  # noqa: S101


@pytest.mark.asyncio
async def test_hostname_candidates_preserve_text_order() -> None:
    text = "https://SG.Aa0Bb1Cc2Dd3Ee4Ff5.sk-AAAABBBBCCCCDDDD.example.com/docs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["SG.Aa0Bb1Cc2Dd3Ee4Ff5", PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
async def test_sendgrid_hostname_candidate_preserves_three_label_key() -> None:
    first_payload = "Aa0Bb1Cc2Dd3Ee4Ff5Gh6I"
    second_payload = "Jj7Kk8Ll9Mm0Nn1Oo2Pp3Qq4Rr5Ss6Tt7Uu8Vv9Ww0X"
    expected = f"SG.{first_payload}.{second_payload}"
    result = await secret_keys(
        None,
        f"https://{expected}.example.com/docs",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
async def test_sendgrid_prefix_does_not_consume_long_domain_label() -> None:
    expected = "SG.Aa0Bb1Cc2Dd3Ee4Ff5"
    text = f"https://{expected}.averylongdomainlabel123.example.com/docs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["。", "．", "｡"])
async def test_idna_dot_equivalents_preserve_hostname_candidates(separator: str) -> None:
    text = f"https://Case{separator}Ab12{separator}Xy34.example.com/docs"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"Case\.Ab12\.Xy34$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["Case.Ab12.Xy34"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
async def test_uri_subdelimiter_inside_hostname_does_not_force_entropy_candidate(threshold: str) -> None:
    text = "https://foo!Aa0Bb1Cc2Dd3Ee4Ff5.example.com/docs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    ["auth[token]", "credentials[api_key]", "user[password]"],
)
async def test_nested_sensitive_parameter_names_preserve_inner_key(name: str) -> None:
    text = f"https://example.com/?{name}=Aa0Bb1Cc2Dd3Ee4Ff5"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert SYNTHETIC_VALUE in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "Aa0Bb1Cc2Dd3Ee4Ff5,https://example.com",
        "https://example.com,Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com|Aa0Bb1Cc2Dd3Ee4Ff5",
    ],
)
async def test_generic_secret_outside_url_span_is_not_exempt(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert SYNTHETIC_VALUE in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/foo=sk-AAAABBBBCCCCDDDD",
        "https://example.com/foo&sk-AAAABBBBCCCCDDDD",
        "https://example.com/foo+sk-AAAABBBBCCCCDDDD",
    ],
)
async def test_uri_component_delimiters_do_not_hide_prefixes(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert PREFIXED_SECRET in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com,Case.Ab12.Xy34",
        "https://[bad:Case.Ab12.Xy34]/docs",
    ],
)
async def test_prose_boundary_preserves_dotted_custom_candidate(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"Case\.Ab12\.Xy34$"])
    result = await secret_keys(None, text, config)

    assert "Case.Ab12.Xy34" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern_count", [1, 100, 500])
@pytest.mark.parametrize("pattern_template", [r"^never{index}$", r"^[z]never{index}$", r"^(?=z)never{index}$", r"(?m)^never{index}$"])
async def test_valid_many_label_hostname_does_not_exhaust_range_budget(
    pattern_count: int,
    pattern_template: str,
) -> None:
    host = ".".join(["a"] * 120 + ["com"])
    config = SecretKeysCfg(
        threshold="balanced",
        custom_regex=[pattern_template.format(index=index) for index in range(pattern_count)],
    )
    result = await secret_keys(None, f"https://{host}/docs", config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "custom_regex", "expected"),
    [
        ("Aa0Bb1Cc2Dd3Ee4Ff5,files/archive/image.png", None, SYNTHETIC_VALUE),
        ("Aa0Bb1Cc2Dd3Ee4Ff5=files/archive/image.png", None, SYNTHETIC_VALUE),
        ("internal-ab12,files/archive/image.png", [r"internal-[a-z0-9]{4}$"], "internal-ab12"),
        ("https://example.com**sk-AAAABBBBCCCCDDDD", None, PREFIXED_SECRET),
    ],
)
async def test_allowed_pattern_does_not_exempt_adjacent_secret(
    text: str,
    custom_regex: list[str] | None,
    expected: str,
) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=custom_regex))

    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("punctuation", [".", ",", ";", ":", "!", "?", ")", "]"])
async def test_url_trailing_punctuation_is_removed_before_custom_match(punctuation: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, f"https://example.com/internal-ab12{punctuation}", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?redirect=https%3A%2F%2Fother.example%2F%3Ftoken%3Dinternal%252Bab12",
        "https://example.com/#redirect=https%3A%2F%2Fother.example%2F%3Ftoken%3Dinternal%252Bab12",
    ],
)
async def test_nested_url_escape_is_preserved(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal\+ab12$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal+ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["query", "fragment"])
@pytest.mark.parametrize("separator", ["/", "%2F", "\\", "%5C"])
async def test_parameter_names_preserve_literal_and_encoded_separators(
    location: str,
    separator: str,
) -> None:
    raw_name = f"sk-Ab1{separator}Cd2Ef3Gh4Ij5Kl6"
    marker = "?" if location == "query" else "#"
    text = f"https://example.com/{marker}{raw_name}=1"
    expected = "sk-Ab1/Cd2Ef3Gh4Ij5Kl6" if separator.lower() == "%2f" or separator == "/" else "sk-Ab1\\Cd2Ef3Gh4Ij5Kl6"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["auth.token.value", "client_secret.value", "client_secret[0].value"])
async def test_sensitive_parameter_components_can_precede_other_fields(name: str) -> None:
    text = f"https://example.com/?{name}={SYNTHETIC_VALUE}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://[Aa0Bb1Cc2Dd3Ee4Ff5]/docs",
        "https://[bad:Aa0Bb1:Cc2Dd3:Ee4Ff5]/docs",
        "https://example.com:Aa0Bb1:Cc2Dd3:Ee4Ff5:443/docs",
    ],
)
async def test_malformed_host_literals_do_not_hide_generic_secrets(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "custom_regex", "expected"),
    [
        ("https://example.com/foo.sk-Aa1Bb2Cc3Dd4Ee5Ff6", None, "sk-Aa1Bb2Cc3Dd4Ee5Ff6"),
        ("files/foo.sk-Aa1Bb2Cc3Dd4Ee5Ff6.png", None, "sk-Aa1Bb2Cc3Dd4Ee5Ff6"),
        ("https://example.com/prefix.internal-ab12", [r"internal-[a-z0-9]{4}$"], "internal-ab12"),
        ("files/prefix.internal-ab12.png", [r"internal-[a-z0-9]{4}$"], "internal-ab12"),
    ],
)
async def test_dotted_path_boundaries_do_not_hide_secrets(
    text: str,
    custom_regex: list[str] | None,
    expected: str,
) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=custom_regex))

    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_file_residual_preserves_order_before_path_secret() -> None:
    text = f"{SYNTHETIC_VALUE}=files/sk-Qq1Ww2Ee3Rr4Tt5Yy6/image.png"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE, "sk-Qq1Ww2Ee3Rr4Tt5Yy6"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"https://example.com/?value=({SYNTHETIC_VALUE})",
        f"files/archive/({SYNTHETIC_VALUE})/image.png",
    ],
)
async def test_enclosed_generic_values_remain_allowed_pattern_data(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fragment",
    ["value=internal+ab12", "value=internal%2Bab12", "internal+ab12", "internal%2Bab12"],
)
async def test_fragment_preserves_literal_and_encoded_plus(fragment: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal\+ab12$"])
    result = await secret_keys(None, f"https://example.com/#{fragment}", config)

    assert result.info["detected_secrets"] == ["internal+ab12"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "VF"])
async def test_ipvfuture_literal_is_not_treated_as_malformed_host(version: str) -> None:
    text = f"https://[{version}.Aa0Bb1Cc2Dd3Ee4Ff5]/docs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/internal-ab12!",
        "https://example.com/internal-ab12%21",
        "https://example.com/?value=internal-ab12!&next=1",
        "https://example.com/?value=internal-ab12%21&next=1",
    ],
)
async def test_explicit_custom_pattern_outranks_prose_trimming(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-ab12!$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal-ab12!"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["authorization", "credential", "credentials", "auth.value"])
async def test_common_credential_parameter_names_force_value_checks(name: str) -> None:
    text = f"https://example.com/?{name}={SYNTHETIC_VALUE}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["token_endpoint", "authorization_endpoint", "credentials_url"])
async def test_credential_metadata_names_do_not_force_public_urls(name: str) -> None:
    public_url = quote("https://identity.example/oauth2/authorize", safe="")
    text = f"https://example.com/?{name}={public_url}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(("opening", "closing"), [("“", "”"), ("«", "»"), ("（", "）"), ("【", "】"), ("「", "」")])
@pytest.mark.parametrize("location", ["query", "file"])
async def test_unicode_enclosed_generic_values_remain_exempt(
    opening: str,
    closing: str,
    location: str,
) -> None:
    enclosed = f"{opening}{SYNTHETIC_VALUE}{closing}"
    text = f"https://example.com/?value={enclosed}" if location == "query" else f"files/{enclosed}/image.png"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "custom_regex", "expected"),
    [
        ("https://example.com/backup-sk-AAAABBBBCCCCDDDD", None, PREFIXED_SECRET),
        ("files/backup_sk-AAAABBBBCCCCDDDD.json", None, PREFIXED_SECRET),
        ("https://example.com/backup-internal-ab12", [r"internal-[a-z0-9]{4}$"], "internal-ab12"),
        ("files/backup_internal-ab12.json", [r"internal-[a-z0-9]{4}$"], "internal-ab12"),
    ],
)
async def test_slug_boundaries_do_not_hide_secret_suffixes(
    text: str,
    custom_regex: list[str] | None,
    expected: str,
) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=custom_regex))

    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "**https://example.com/sk-AAAABBBBCCCCDDDD**",
        "_https://example.com/sk-AAAABBBBCCCCDDDD_",
        "~~https://example.com/sk-AAAABBBBCCCCDDDD~~",
        "[https://example.com/sk-AAAABBBBCCCCDDDD](target)",
    ],
)
async def test_markdown_wrappers_do_not_pollute_candidate_values(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["https://example.com/internal%2Fab12)", "https://example.com/?value=internal%2Fab12)"])
async def test_encoded_separator_custom_candidate_precedes_prose_trim(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal/ab12\)$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal/ab12)"]  # noqa: S101


@pytest.mark.asyncio
async def test_untrimmed_terminal_custom_candidate_preserves_text_order() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"custom-one$", r"custom-two\)$"])
    result = await secret_keys(None, "https://example.com/custom-one/custom-two)", config)

    assert result.info["detected_secrets"] == ["custom-one", "custom-two)"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("opening", ["%28", "%5B", "%60"])
async def test_encoded_opening_wrapper_pairs_with_literal_closer(opening: str) -> None:
    closing = {"%28": ")", "%5B": "]", "%60": "`"}[opening]
    text = f"https://example.com/?value={opening}{SYNTHETIC_VALUE}{closing}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["accessTokenValue", "clientSecretValue", "apiKeyValue", "refreshTokenData", "authorizationCode"])
async def test_credential_payload_suffix_names_force_value_checks(name: str) -> None:
    text = f"https://example.com/?{name}={SYNTHETIC_VALUE}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "authorization.endpoint",
        "authorization[endpoint]",
        "token.endpoint",
        "token[endpoint]",
        "credentials.url",
        "credentials[url]",
        "auth.type",
        "auth[type]",
    ],
)
async def test_nested_credential_metadata_names_do_not_force_public_urls(name: str) -> None:
    public_url = quote("https://identity.example/oauth2/authorize", safe="")
    result = await secret_keys(
        None,
        f"https://example.com/?{name}={public_url}",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["~", "%7E", "%25"])
@pytest.mark.parametrize("location", ["path", "query", "file"])
async def test_unreserved_slug_boundaries_do_not_hide_prefixed_secrets(
    separator: str,
    location: str,
) -> None:
    component = f"backup{separator}sk-Aa1Bb2Cc3Dd4Ee5Ff6"
    if location == "path":
        text = f"https://example.com/{component}"
    elif location == "query":
        text = f"https://example.com/?value={component}"
    else:
        text = f"files/{component}.json"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert "sk-Aa1Bb2Cc3Dd4Ee5Ff6" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["<", '"', "'", "{", "|", "^"])
async def test_broad_url_exemption_without_parser_span_scans_residual(boundary: str) -> None:
    text = f"https://{boundary}{PREFIXED_SECRET}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert PREFIXED_SECRET in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
async def test_custom_secret_spans_url_matcher_punctuation(threshold: str) -> None:
    text = "https://example.com/prefix-internal{ab12"
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold=threshold, custom_regex=[r"internal\{ab12$"]),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == ["internal{ab12"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("threshold", "credential"),
    [("balanced", SYNTHETIC_VALUE), ("permissive", f"{SYNTHETIC_VALUE}Gh")],
)
async def test_adjacent_urls_parse_each_authority(threshold: str, credential: str) -> None:
    text = f"[first](https://first.example)[second](https://{credential}@second.example)"

    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [credential]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", [",", ";", "("])
@pytest.mark.parametrize(
    ("threshold", "credential"),
    [("balanced", SYNTHETIC_VALUE), ("permissive", f"{SYNTHETIC_VALUE}Gh")],
)
async def test_adjacent_non_markdown_urls_parse_each_authority(
    separator: str,
    threshold: str,
    credential: str,
) -> None:
    text = f"https://first.example{separator}https://{credential}@second.example"

    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [credential]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/Aa0Bb1Cc2Dd3Ee4Ff5%20Gg6Hh7Ii8Jj9Kk0Ll1",
        "files/Aa0Bb1Cc2Dd3Ee4Ff5%20Gg6Hh7Ii8Jj9Kk0Ll1/image.png",
        "https://example.com/?value=Aa0Bb1Cc2Dd3Ee4Ff5+Gg6Hh7Ii8Jj9Kk0Ll1",
    ],
)
async def test_decoded_uri_whitespace_does_not_force_opaque_candidates(
    threshold: str,
    text: str,
) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/Aa0Bb1Cc2Dd3Ee4Ff5%7BGg6Hh7Ii8Jj9Kk0Ll1",
        "files/Aa0Bb1Cc2Dd3Ee4Ff5%7BGg6Hh7Ii8Jj9Kk0Ll1/image.png",
        "https://example.com/?value=Aa0Bb1Cc2Dd3Ee4Ff5%7BGg6Hh7Ii8Jj9Kk0Ll1",
    ],
)
async def test_percent_encoded_uri_punctuation_does_not_force_opaque_candidates(
    threshold: str,
    text: str,
) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
@pytest.mark.parametrize("name", ["public_key", "verification_key", "verify_key", "client_public_key"])
async def test_explicit_public_key_fields_do_not_force_opaque_values(
    threshold: str,
    name: str,
) -> None:
    text = f"https://example.com/?{name}={SYNTHETIC_VALUE}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold=threshold, custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://e.com/prefix-internal_-route",
        "files/prefix-internal_-route/image.png",
    ],
)
async def test_top_level_custom_alternative_preserves_punctuation_end(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal_$|NEVER"])

    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == ["internal_"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("threshold", ["balanced", "permissive"])
@pytest.mark.parametrize(
    "text",
    [
        "item+Aa0Bb1Cc2Dd3Ee4Ff5.png",
        "files/item+Aa0Bb1Cc2Dd3Ee4Ff5.png",
    ],
)
async def test_file_root_uri_subdelimiter_does_not_force_opaque_candidate(
    threshold: str,
    text: str,
) -> None:
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold=threshold, custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


def test_custom_regex_pattern_count_remains_backward_compatible() -> None:
    config = SecretKeysCfg(
        threshold="balanced",
        custom_regex=[rf"^x{index}$" for index in range(33)],
    )

    assert len(config.custom_regex or []) == 33  # noqa: S101


@pytest.mark.asyncio
async def test_custom_range_budget_does_not_turn_benign_url_into_secret() -> None:
    text = "https://example.com/" + "%2F".join(["a"] * 400)
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"never-match$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_custom_truncation_does_not_override_url_exemption() -> None:
    text = "https://example.com/" + "-".join(f"a{index}" for index in range(600))
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r".*NEVER_MATCH$"])

    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101
    assert result.info["custom_scan_incomplete"] is True  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [r"^.ECRET$", r"^(?=.ECRET).ECRET$"],
)
@pytest.mark.parametrize(
    "carrier",
    [
        "https://example.com/{value}",
        "files/{value}/image.png",
        "https://example.com/?value={value}",
        "https://example.com/#value={value}",
    ],
)
async def test_custom_range_budget_checks_late_atomic_candidates(pattern: str, carrier: str) -> None:
    parts = [f"a{index}" for index in range(600)]
    parts[300] = "SECRET"
    text = carrier.format(value="-".join(parts))
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])

    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "SECRET" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_custom_range_budget_checks_late_two_boundary_candidate_once() -> None:
    parts = [f"a{index}" for index in range(722)]
    parts[700:702] = ["TARGET", "a"]
    text = "https://example.com/" + "-".join(parts)
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^.ARGET-a$"])

    result = await secret_keys(None, text, config)

    assert "TARGET-a" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "carrier",
    [
        "https://example.com/{long}/prefix-SECRET-route",
        "files/{long}/prefix-SECRET-route/image.png",
        "https://example.com/{long}?value=prefix-SECRET-route",
        "https://example.com/{long}#value=prefix-SECRET-route",
    ],
)
async def test_custom_truncation_keeps_short_scan_for_later_components(carrier: str) -> None:
    long_component = "-".join(f"a{index}" for index in range(600))
    text = carrier.format(long=long_component)
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^.ECRET$"])

    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["SECRET"]  # noqa: S101
    assert result.info["custom_scan_incomplete"] is True  # noqa: S101


@pytest.mark.asyncio
async def test_priority_custom_scan_uses_one_token_wide_budget() -> None:
    text = "https://example.com/" + "/".join(["a-b"] * 3_000)
    config = SecretKeysCfg(
        threshold="balanced",
        custom_regex=[rf"^(?:z)never{index}$" for index in range(500)],
    )

    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["custom_scan_incomplete"] is True  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern_template",
    [
        r"^[zZ]never{index}$",
        r"^(?:z)never{index}$",
        r"^.never{index}$",
        r"^(z)never{index}$",
        r"^(?P<kind>z)never{index}$",
        r"^(?=(z))znever{index}$",
        r"(?x)^znever{index}$ # benign",
    ],
)
async def test_many_nonmatching_custom_patterns_do_not_exhaust_short_path_budget(pattern_template: str) -> None:
    text = "https://example.com/" + "%2F".join(["a"] * 130)
    pattern_count = 16 if re.compile(pattern_template.format(index=0)).groups else 500
    config = SecretKeysCfg(
        threshold="balanced",
        custom_regex=[pattern_template.format(index=index) for index in range(pattern_count)],
    )
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_repeated_tokens_reuse_embedded_custom_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("guardrails.checks.text.secret_keys")
    original = module._extract_embedded_secret_candidates
    calls = 0

    def counted_extract(token: str, custom_regex: list[str] | None = None):
        """Count extractor calls while preserving its result.

        Args:
            token: Exempt token to inspect.
            custom_regex: Configured custom expressions.

        Returns:
            The original extractor result.
        """
        nonlocal calls
        calls += 1
        return original(token, custom_regex)

    monkeypatch.setattr(module, "_extract_embedded_secret_candidates", counted_extract)
    token = "https://example.com/" + "%2F".join(["a"] * 130)
    config = SecretKeysCfg(
        threshold="balanced",
        custom_regex=[rf"^(?:z)never{index}$" for index in range(500)],
    )

    result = await secret_keys(None, " ".join([token] * 8), config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert calls == 1  # noqa: S101


def test_canonical_classification_reuses_combined_custom_matchers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid re-running every original pattern for every candidate."""
    module = import_module("guardrails.checks.text.secret_keys")
    candidates = tuple(f"sk-Aa1Bb2Cc3Dd4Ee5Ff6-{index}" for index in range(1_000))
    patterns = [rf"^never{index}$" for index in range(1_000)]
    original_match = module.re.match
    raw_match_calls = 0

    def counted_match(pattern: str, value: str, flags: int = 0) -> re.Match[str] | None:
        """Count raw string-pattern matches.

        Args:
            pattern: Regex source passed to ``re.match``.
            value: Candidate string to match.
            flags: Optional regular-expression flags.

        Returns:
            The original ``re.match`` result.
        """
        nonlocal raw_match_calls
        raw_match_calls += 1
        return original_match(pattern, value, flags)

    monkeypatch.setattr(module.re, "match", counted_match)

    findings = module._canonical_embedded_findings(candidates, (), module.CONFIGS["balanced"], patterns)

    assert findings == list(candidates)  # noqa: S101
    assert raw_match_calls == 0  # noqa: S101


@pytest.mark.asyncio
async def test_unique_tokens_are_not_retained_in_embedded_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("guardrails.checks.text.secret_keys")
    original = module._extract_embedded_secret_candidates
    calls = 0

    def counted_extract(token: str, custom_regex: list[str] | None = None):
        """Count unique-token extractor calls.

        Args:
            token: Exempt token to inspect.
            custom_regex: Configured custom expressions.

        Returns:
            The original extractor result.
        """
        nonlocal calls
        calls += 1
        return original(token, custom_regex)

    monkeypatch.setattr(module, "_extract_embedded_secret_candidates", counted_extract)
    tokens = [f"https://example.com/{index}%2Fa" for index in range(8)]
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^never$"])

    result = await secret_keys(None, " ".join(tokens), config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert calls == len(tokens)  # noqa: S101


@pytest.mark.asyncio
async def test_repeated_markdown_labels_reuse_embedded_custom_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    module = import_module("guardrails.checks.text.secret_keys")
    original = module._extract_embedded_secret_candidates
    calls = 0

    def counted_extract(token: str, custom_regex: list[str] | None = None):
        """Count Markdown label and target extraction calls.

        Args:
            token: Exempt token to inspect.
            custom_regex: Configured custom expressions.

        Returns:
            The original extractor result.
        """
        nonlocal calls
        calls += 1
        return original(token, custom_regex)

    monkeypatch.setattr(module, "_extract_embedded_secret_candidates", counted_extract)
    label = "https://example.com/" + "%2F".join(["a"] * 130)
    token = f"[{label}](x)"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^never$"])

    result = await secret_keys(None, " ".join([token] * 8), config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert calls == 2  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "patterns",
    [
        ["(?x)# [\n(x)\\1$", "(?x)# [\n(z)\\1$"],
        ["(?x)# [\n(z)\\1$", "(?x)# [\n(x)\\1$"],
    ],
)
@pytest.mark.parametrize("text", ["https://example.com/prefix%2Fzz", "files/prefix%2Fzz/image.png"])
async def test_combined_custom_matchers_preserve_verbose_backreferences(patterns: list[str], text: str) -> None:
    assert any(re.match(pattern, "xx") for pattern in patterns)  # noqa: S101
    assert any(re.match(pattern, "zz") for pattern in patterns)  # noqa: S101

    config = SecretKeysCfg(threshold="balanced", custom_regex=patterns)
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["zz"]  # noqa: S101


@pytest.mark.asyncio
async def test_combined_custom_matchers_preserve_leading_bracket_class_semantics() -> None:
    pattern = r"[]()][A-Za-z0-9]{30}$"
    value = ":Ab3dEf5hIj7lMn9pQr2tUv4xYz6Bcd"
    assert re.match(pattern, value) is None  # noqa: S101

    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, f"https://example.com/{value}", config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [r"(?# (?P<)(?P<real>x)", "(?x:# (?P<\n(?P<real>x)\n)"])
async def test_custom_matcher_does_not_rewrite_capture_syntax_inside_comments(pattern: str) -> None:
    value = "Aa1Bb2Cc3Dd4Ee5Ff6"
    assert re.match(pattern, value) is None  # noqa: S101

    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, f"https://example.com/{value}", config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("component", "pattern", "expected"),
    [
        ("%2F".join(["a"] * 400 + ["internal", "ab12"]), r"internal/ab12$", "internal/ab12"),
        ("-".join(["a"] * 400 + ["internal-ab12"]), r"internal-[a-z0-9]{4}$", "internal-ab12"),
    ],
)
async def test_custom_range_budget_still_checks_trailing_boundaries(
    component: str,
    pattern: str,
    expected: str,
) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, f"https://example.com/{component}", config)

    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "authorization.redirect_uri",
        "authorization[redirect_uri]",
        "token.introspection_endpoint",
        "token[introspection_endpoint]",
        "credentials.metadata_url",
        "credentials[metadata_url]",
        "auth.jwks_uri",
        "auth[jwks_uri]",
    ],
)
async def test_compound_credential_metadata_names_do_not_force_public_urls(name: str) -> None:
    public_url = quote("https://identity.example/oauth2/openid-configuration", safe="")
    result = await secret_keys(
        None,
        f"https://example.com/?{name}={public_url}",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "token_v2",
        "tokenV2",
        "api_key_v2",
        "client_secret_v3",
        "password_v1",
        "passwordConfirmation",
        "password_repeat",
    ],
)
async def test_versioned_and_payload_credential_names_force_value_checks(name: str) -> None:
    result = await secret_keys(
        None,
        f"https://example.com/?{name}={SYNTHETIC_VALUE}",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", ["__", "___", "``", "```", "***"])
async def test_repeated_markdown_wrappers_do_not_pollute_candidate_values(marker: str) -> None:
    text = f"{marker}https://example.com/{PREFIXED_SECRET}{marker}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "custom_regex", "expected"),
    [
        ("files/backup#sk-Aa1Bb2Cc3Dd4Ee5Ff6.json", None, "sk-Aa1Bb2Cc3Dd4Ee5Ff6"),
        ("files/backup*sk-Aa1Bb2Cc3Dd4Ee5Ff6.json", None, "sk-Aa1Bb2Cc3Dd4Ee5Ff6"),
        ("files/backup#internal-ab12.json", [r"internal-[a-z0-9]{4}$"], "internal-ab12"),
    ],
)
async def test_file_component_markers_do_not_concatenate_away_secrets(
    text: str,
    custom_regex: list[str] | None,
    expected: str,
) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=custom_regex))

    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"https://example.com/token:{SYNTHETIC_VALUE}",
        f"https://example.com/client_secret:{SYNTHETIC_VALUE}",
        f"files/token:{SYNTHETIC_VALUE}/image.png",
        f"https://example.com/?token:{SYNTHETIC_VALUE}",
        f"https://example.com/#token:{SYNTHETIC_VALUE}",
    ],
)
async def test_colon_sensitive_assignments_force_value_checks(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert SYNTHETIC_VALUE in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_encoded_bracketed_malformed_host_does_not_hide_generic_secret() -> None:
    text = f"https://%5B{SYNTHETIC_VALUE}%5D/docs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"https://example.com/item;{SYNTHETIC_VALUE}/docs",
        f"https://example.com/?value=item;{SYNTHETIC_VALUE}",
        f"https://example.com/#value=item;{SYNTHETIC_VALUE}",
    ],
)
async def test_uri_semicolon_data_does_not_force_generic_candidates(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("delimiter", list("!$&'()*+,;="))
async def test_uri_subdelimiter_data_does_not_force_generic_path_candidates(delimiter: str) -> None:
    text = f"https://example.com/item{delimiter}{SYNTHETIC_VALUE}/docs"

    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "password_policy",
        "token_bucket",
        "token_lifetime",
        "auth_provider",
        "authorization_server",
        "credential_store",
        "secret_rotation",
        "key_usage",
        "api_key_documentation",
    ],
)
async def test_noncredential_names_containing_sensitive_words_do_not_force_values(name: str) -> None:
    public_url = quote("https://identity.example/oauth2/policies/v3", safe="")
    result = await secret_keys(
        None,
        f"https://example.com/?{name}={public_url}",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "authorization.endpoint.value",
        "token_endpoint_value",
        "credentials.metadata_url.value",
        "auth.jwks_uri.value",
    ],
)
async def test_metadata_value_wrappers_do_not_force_public_urls(name: str) -> None:
    public_url = quote("https://identity.example/oauth2/token", safe="")
    result = await secret_keys(
        None,
        f"https://example.com/?{name}={public_url}",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/backup-internalab12",
        "https://example.com/?value=backup-internalab12",
        "files/backup-internalab12.json",
    ],
)
async def test_optional_custom_regex_prefix_remains_detectable_inside_allowed_tokens(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-?ab12$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internalab12"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"**__https://example.com/{PREFIXED_SECRET}__**",
        f"__**https://example.com/{PREFIXED_SECRET}**__",
        f"~~`https://example.com/{PREFIXED_SECRET}`~~",
    ],
)
async def test_nested_markdown_wrappers_do_not_pollute_url_candidates(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("opening", ["[", "%5B"])
@pytest.mark.parametrize("closing", ["]", "%5D"])
async def test_mixed_encoded_brackets_preserve_malformed_host_candidate(
    opening: str,
    closing: str,
) -> None:
    text = f"https://{opening}{SYNTHETIC_VALUE}{closing}/docs"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"[https://example.com/{PREFIXED_SECRET}](sk-QQQQWWWWEEEERRRR)", [PREFIXED_SECRET, "sk-QQQQWWWWEEEERRRR"]),
        (f"[{PREFIXED_SECRET}](a.png)", [PREFIXED_SECRET]),
        (f"![{PREFIXED_SECRET}](a.png)", [PREFIXED_SECRET]),
        (f"[{PREFIXED_SECRET}](a/sk-QQQQWWWWEEEERRRR.png)", [PREFIXED_SECRET, "sk-QQQQWWWWEEEERRRR"]),
    ],
)
async def test_markdown_link_regions_preserve_candidate_order(text: str, expected: list[str]) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == expected  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "a/sk-AAAAAAAAAAAA%2Epng",
        "a%2Fsk-AAAAAAAAAAAA%2Epng",
        "a/sk-AAAAAAAAAAAA.png%29",
        "%28a/sk-AAAAAAAAAAAA.png%29",
        "a/sk-AAAAAAAAAAAA.png%2C",
    ],
)
async def test_encoded_file_extension_and_wrappers_do_not_hide_prefixed_secrets(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "~~a/sk-AAAAAAAAAAAA.png~~",
        "_a/sk-AAAAAAAAAAAA.png_",
        "`a/sk-AAAAAAAAAAAA.png`",
        "<a/sk-AAAAAAAAAAAA.png>",
        "a/sk-AAAAAAAAAAAA.png,",
    ],
)
async def test_file_presentation_wrappers_do_not_hide_prefixed_secrets(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [
        r"(?=foo)foo/qux|bar/baz$",
        r"[x]foo/qux|bar/baz$",
        r"[x]?bar/baz$",
        r"(?=foo)?bar/baz$",
        r"(?x)bar / baz $",
    ],
)
async def test_complex_custom_regex_prefixes_are_not_pruned(pattern: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, "https://example.com/prefix%2Fbar%2Fbaz", config)

    assert result.info["detected_secrets"] == ["bar/baz"]  # noqa: S101


@pytest.mark.asyncio
async def test_verbose_terminal_comment_preserves_punctuation_ended_custom_candidate() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"(?x)^internal_$  # comment"])
    result = await secret_keys(None, "https://e.com/prefix-internal_-route", config)

    assert result.info["detected_secrets"] == ["internal_"]  # noqa: S101


@pytest.mark.asyncio
async def test_scoped_flags_preserve_punctuation_ended_custom_candidate() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"(?i:^internal_$)"])
    result = await secret_keys(None, "https://e.com/prefix-internal_-route", config)

    assert result.info["detected_secrets"] == ["internal_"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.skipif(sys.version_info < (3, 14), reason=r"Python before 3.14 does not support the \z anchor")
async def test_python_314_terminal_anchor_preserves_punctuation_ended_custom_candidate() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal_\z"])
    result = await secret_keys(None, "https://e.com/prefix-internal_-route", config)

    assert result.info["detected_secrets"] == ["internal_"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/prefix-TARGET-route.png",
        "files/prefix-TARGET-route/image.png",
        "https://example.com/?value=prefix-TARGET-route",
        "https://example.com/#value=prefix-TARGET-route",
    ],
)
async def test_lookahead_wildcard_is_not_treated_as_a_literal_prefix(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^(?=.ARGET).ARGET$"])

    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["TARGET"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pattern", "value"),
    [
        (r"^(SE.){2}$", "SECSET"),
        (r"^(?:tok[A-Z0-9]){4}$", "tokAtokBtokCtokD"),
        (r"^(?:ab|ac){2}Z$", "ababZ"),
    ],
)
async def test_repeated_custom_subpatterns_are_not_used_as_fixed_prefixes(pattern: str, value: str) -> None:
    text = f"https://example.com/prefix-{value}-route.png"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])

    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == [value]  # noqa: S101


@pytest.mark.asyncio
async def test_custom_range_findings_preserve_source_order_after_budget_planning() -> None:
    parts = [f"a{index:03d}" for index in range(800)]
    text = "https://example.com/prefix%2F" + "%2F".join(parts)
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"[a-z]\d{3}/[a-z]\d{3}$"])
    result = await secret_keys(None, text, config)

    expected = [f"a{index:03d}/a{index + 1:03d}" for index in range(799)]
    assert result.info["detected_secrets"] == expected  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"_See https://example.com/{PREFIXED_SECRET}_",
        f"**_See https://example.com/{PREFIXED_SECRET}_**",
    ],
)
async def test_cross_word_markdown_wrappers_do_not_pollute_url_candidates(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [PREFIXED_SECRET]  # noqa: S101


@pytest.mark.asyncio
async def test_markdown_state_does_not_change_raw_custom_regex_semantics() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-ab12_$"])
    result = await secret_keys(None, "_identifier internal-ab12_", config)

    assert result.info["detected_secrets"] == ["internal-ab12_"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[click sk-AAAAAAAAAAAA](a.png)", "sk-AAAAAAAAAAAA"),
        ("![alt sk-AAAAAAAAAAAA](a.png)", "sk-AAAAAAAAAAAA"),
        ("[https://example.com](a%2Fsk-AAAAAAAAAAAA%2Epng)", "sk-AAAAAAAAAAAA"),
    ],
)
async def test_multiword_markdown_labels_and_targets_are_scanned(text: str, expected: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "[a/sk-AAAAAAAAAAAA.png]()",
        "![a/sk-AAAAAAAAAAAA.png]()",
        "[a/sk-AAAAAAAAAAAA.png](<>)",
        "![a/sk-AAAAAAAAAAAA.png](<>)",
        '[a/sk-AAAAAAAAAAAA.png](<a.png> "title")',
        '![a/sk-AAAAAAAAAAAA.png](<a.png> "title")',
    ],
)
async def test_markdown_file_labels_are_scanned_with_empty_or_titled_targets(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '"See a/sk-AAAAAAAAAAAA.png"',
        "'See a/sk-AAAAAAAAAAAA.png'",
        "<See a/sk-AAAAAAAAAAAA.png>",
    ],
)
async def test_cross_word_enclosures_do_not_hide_file_candidates(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '(See public.png) "See a/sk-AAAAAAAAAAAA.png"',
        "{See public.png} <See a/sk-AAAAAAAAAAAA.png>",
        '（See public.png） "See a/sk-AAAAAAAAAAAA.png"',
    ],
)
async def test_cross_word_enclosure_state_closes_before_the_next_region(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '(See "some a/sk-AAAAAAAAAAAA.png")',
        "<See 'some a/sk-AAAAAAAAAAAA.png'>",
    ],
)
async def test_nested_cross_word_enclosures_do_not_hide_file_candidates(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        '("See a/sk-AAAAAAAAAAAA.png")',
        "<'See a/sk-AAAAAAAAAAAA.png'>",
        '{"See a/sk-AAAAAAAAAAAA.png"}',
    ],
)
async def test_same_token_nested_openers_do_not_hide_file_candidates(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


PRESENTATION_WRAPPERS = (
    ('"', '"'),
    ("'", "'"),
    ("<", ">"),
    ("(", ")"),
    ("{", "}"),
    ("[", "]"),
    ("_", "_"),
    ("~~", "~~"),
    ("**", "**"),
    ("`", "`"),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["", " "])
@pytest.mark.parametrize(("outer_open", "outer_close"), PRESENTATION_WRAPPERS)
@pytest.mark.parametrize(("inner_open", "inner_close"), PRESENTATION_WRAPPERS)
async def test_nested_presentation_wrapper_matrix_does_not_hide_file_candidates(
    separator: str,
    outer_open: str,
    outer_close: str,
    inner_open: str,
    inner_close: str,
) -> None:
    text = f"{outer_open}{inner_open}See{separator}a/sk-AAAAAAAAAAAA.png{inner_close}{outer_close}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
async def test_closed_nested_presentation_does_not_leak_into_following_tokens() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-ab12_$"])
    result = await secret_keys(None, '**"public.png"** internal-ab12_', config)

    assert result.info["detected_secrets"] == ["internal-ab12_"]  # noqa: S101


def test_closed_nested_presentation_clears_all_closer_state() -> None:
    closers = ["**", '"']

    assert _strip_closed_presentation_regions('**"public.png"**', closers, 3) == '**"public.png'  # noqa: S101
    assert closers == []  # noqa: S101


@pytest.mark.asyncio
async def test_punctuation_after_an_opener_does_not_close_its_region() -> None:
    result = await secret_keys(
        None,
        "**!!! a/sk-AAAAAAAAAAAA.png**",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.info["detected_secrets"] == ["sk-AAAAAAAAAAAA"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(("opening", "closing"), [('"', '"'), ("'", "'"), ("<", ">")])
async def test_presentation_depth_overflow_fails_closed(opening: str, closing: str) -> None:
    text = opening * 17 + "See a/sk-AAAAAAAAAAAA.png" + closing * 17
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text.split()[0]]  # noqa: S101


@pytest.mark.asyncio
async def test_presentation_overflow_findings_are_bounded_to_source_tokens() -> None:
    tokens = ['"' * 17 + f"value{index}" + '"' * 17 for index in range(64)]
    text = " ".join(tokens)

    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))
    findings = result.info["detected_secrets"]

    assert findings == tokens  # noqa: S101
    assert sum(map(len, findings)) <= len(text)  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name",
    [
        "password.policy",
        "password[policy]",
        "token.bucket",
        "auth.provider",
        "authorization.server",
        "credential.store",
        "key.usage",
    ],
)
async def test_structural_metadata_fields_do_not_force_public_urls(name: str) -> None:
    public_url = quote("https://identity.example/oauth2/policies/v3", safe="")
    result = await secret_keys(
        None,
        f"https://example.com/?{name}={public_url}",
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_custom_range_prefers_longest_match_from_same_source_start() -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"foo(?:-bar)?$"])
    result = await secret_keys(None, "https://example.com/prefix-foo-bar", config)

    assert result.info["detected_secrets"] == ["foo-bar"]  # noqa: S101


@pytest.mark.asyncio
async def test_custom_range_checks_longer_nearby_match_before_route_tail() -> None:
    route_tail = "%2F".join(f"a{index}" for index in range(100))
    text = f"https://example.com/prefix%2Ffoo%2Fbar%2F{route_tail}"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"foo(?:/bar)?$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["foo/bar"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "pattern",
    [
        r"foo(?:/(?:a/){40}bar)?$",
        r"foo(?:/(?:a/){40}bar)??$",
        r"foo(?:/a|/(?:a/){40}bar)?$",
        r"^(?:foo)(?:/(?:a/){40}bar)?$",
        r"(?i)^foo(?:/(?:a/){40}bar)?$",
        r"^[fF]oo(?:/(?:a/){40}bar)?$",
    ],
)
async def test_custom_range_uses_regex_match_end_beyond_nearby_window(pattern: str) -> None:
    secret_tail = "/".join(["a"] * 40 + ["bar"])
    route_tail = "%2F".join(f"r{index}" for index in range(100))
    text = f"https://example.com/prefix%2Ffoo%2F{quote(secret_tail, safe='')}%2F{route_tail}"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == [f"foo/{secret_tail}"]  # noqa: S101


@pytest.mark.asyncio
async def test_complete_strict_word_does_not_duplicate_embedded_finding() -> None:
    text = f"https://example.com/{PREFIXED_SECRET}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="strict", custom_regex=None))

    assert result.info["detected_secrets"] == [text]  # noqa: S101
