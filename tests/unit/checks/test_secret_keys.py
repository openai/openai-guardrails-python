"""Tests for secret key detection guardrail."""

from __future__ import annotations

import pytest

from guardrails.checks.text.secret_keys import (
    SecretKeysCfg,
    _detect_secret_keys,
    _find_nested_url_start,
    secret_keys,
)


def test_detect_secret_keys_flags_high_entropy_strings() -> None:
    """High entropy tokens should be detected as potential secrets."""
    text = "API key sk-AAAABBBBCCCCDDDD"
    result = _detect_secret_keys(text, cfg={"min_length": 10, "min_entropy": 3.5, "min_diversity": 2, "strict_mode": True})

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
    config = SecretKeysCfg(threshold="permissive", custom_regex=None)
    result = await secret_keys(None, "Hello world", config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?token=sk-AAAABBBBCCCCDDDD",
        "https://example.com/?token=sk%2DAAAABBBBCCCCDDDD",
        "files/sk-AAAABBBBCCCCDDDD.png",
        "files/sk%2DAAAABBBBCCCCDDDD.md",
        "https://example.com/?key=AKIA1234567890ABCDEF",
        "files/xoxb-AAAABBBBCCCC1.md",
    ],
)
async def test_secret_keys_detects_supported_exempt_container_values(
    text: str,
) -> None:
    """Provider credentials in query values and file basenames are detected."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", [",", "("])
async def test_adjacent_url_does_not_extend_a_short_query_candidate(separator: str) -> None:
    """An adjacent URL cannot supply length or diversity to a query value."""
    first_url = "https://example.com/?token=sk-ABCDEFGHIJ1"
    adjacent_urls = f"{first_url}{separator}https://example.com/x"

    first_result = await secret_keys(
        None,
        first_url,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )
    adjacent_result = await secret_keys(
        None,
        adjacent_urls,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert first_result.tripwire_triggered is False  # noqa: S101
    assert adjacent_result.tripwire_triggered is False  # noqa: S101
    assert adjacent_result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_adjacent_url_does_not_suppress_a_file_candidate() -> None:
    """A later URL cannot suppress a supported file-basename candidate."""
    text = "files/sk-AAAABBBBCCCCDDDD.png,https://example.com"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_period_delimited_url_does_not_suppress_file_candidate() -> None:
    """A broad scheme-like match cannot hide a later explicit URL."""
    text = "files/sk-AAAABBBBCCCCDDDD.png.https://example.com"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "example.com/sk-AAAABBBBCCCCDDDD.png",
        "www.example.com/sk-AAAABBBBCCCCDDDD.png",
        "192.0.2.1/sk-AAAABBBBCCCCDDDD.png",
        "//example.com/sk-AAAABBBBCCCCDDDD.png",
    ],
)
async def test_scheme_less_url_path_is_not_a_file_candidate(text: str) -> None:
    """Shared URL detection prevents scheme-less path reclassification."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["example.com/path", "192.0.2.1/path"])
async def test_adjacent_scheme_less_url_does_not_suppress_file_candidate(url: str) -> None:
    """A presentation-delimited scheme-less URL preserves the file prefix."""
    text = f"files/sk-AAAABBBBCCCCDDDD.png,{url}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", ["blocked.example/path", "192.0.2.1/path"])
async def test_adjacent_url_does_not_pad_nonempty_query_value(destination: str) -> None:
    """An adjacent destination cannot pad a short value ending in equals."""
    text = f"https://outer.example/?token=sk-ABCDEFGHIJ=,{destination}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix",
    [
        "https://example.com:bad",
        "https://example.com:70000",
        "https://example.com:blocked.example/path",
        "https://example.com:192.0.2.1/path",
        "https://[bad]",
    ],
)
async def test_malformed_url_suffix_does_not_expose_file_candidate(suffix: str) -> None:
    """Only a valid adjacent URL can delimit a preceding file candidate."""
    text = f"files/sk-AAAABBBBCCCCDDDD.png,{suffix}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_malformed_url_prefix_cannot_alias_a_later_valid_url() -> None:
    """A later valid URL cannot validate an earlier malformed prefix."""
    text = "files/sk-AAAABBBBCCCCDDDD.png,https://example.com:bad,https://example.com"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("hard_stop", ["\\", "|"])
@pytest.mark.parametrize(
    "text_template",
    [
        "https://example.com/?token=sk-AAAABBBBCCCCDDDD{hard_stop}tail",
        "files/sk-AAAABBBBCCCCDDDD.png,https://example.com{hard_stop}tail",
    ],
)
async def test_url_regex_hard_stop_does_not_expose_candidates(
    hard_stop: str,
    text_template: str,
) -> None:
    """A regex-truncated malformed URL cannot expose container contents."""
    text = text_template.format(hard_stop=hard_stop)

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_detects_bracketed_ipv6_query_values() -> None:
    """A valid bracketed IPv6 host exposes its query values."""
    text = "https://[::1]/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_suffix", ["%", "%F", "%GG", "%FF"])
async def test_query_value_decoding_rejects_malformed_encoding_atomically(
    invalid_suffix: str,
) -> None:
    """Malformed escapes cannot pad an otherwise-short query candidate."""
    text = f"https://example.com/?token=sk%2DABCDEFGHIJ1{invalid_suffix}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_query_value_decoding_accepts_valid_utf8_atomically() -> None:
    """A completely valid encoded query credential remains detectable."""
    text = "https://example.com/?token=sk%2DAAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_query_value_decoding_rejects_raw_invalid_unicode_atomically() -> None:
    """Raw invalid Unicode cannot pad an otherwise-short query candidate."""
    text = "https://example.com/?token=sk-ABCDEFGHIJ1\ud800"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix",
    [
        "/https://example.com/x",
        "%2Fhttps%3A%2F%2Fexample.com%2Fx",
    ],
)
async def test_nested_url_does_not_extend_a_short_query_candidate(suffix: str) -> None:
    """An unsupported nested URL cannot contribute to candidate scoring."""
    text = f"https://example.com/?token=sk-ABCDEFGHIJ1{suffix}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://outer.example/?next=https://[::1]/?token=sk-AAAABBBBCCCCDDDD",
        "https://outer.example/#https://[::1]/?token=sk-AAAABBBBCCCCDDDD",
        "https://outer.example/?next=,https://inner.example/?token=sk-AAAABBBBCCCCDDDD",
    ],
)
async def test_nested_url_query_values_remain_unsupported(text: str) -> None:
    """Nested URL spans cannot expose their query credentials."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "ftp://allow.example/sk-AAAABBBBCCCCDDDD.png",
        "data:text/plain/sk-AAAABBBBCCCCDDDD.png",
        "javascript:/payload/sk-AAAABBBBCCCCDDDD.png",
        "vbscript:/payload/sk-AAAABBBBCCCCDDDD.png",
    ],
)
async def test_non_http_url_cannot_fall_through_to_file_candidate(text: str) -> None:
    """A recognized URL scheme keeps its path out of file extraction."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix",
    [
        "ftp://other.example/path",
        "data:text/plain,x",
        "javascript:alert(1)",
        "vbscript:x",
    ],
)
async def test_adjacent_non_http_url_cannot_suppress_query_secret(suffix: str) -> None:
    """A later explicit URL cannot suppress a supported query candidate."""
    text = f"https://example.com/?token=sk-AAAABBBBCCCCDDDD,{suffix}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        "sk-AAAABBBBdata:textCCCC1",
        "sk-AAAABBBBftp://CCCC1",
        "sk-AAAABBBBhttps://CCCC1",
    ],
)
async def test_scheme_substring_inside_query_candidate_is_not_structural(value: str) -> None:
    """Scheme text without a presentation boundary remains credential data."""
    text = f"https://example.com/?token={value}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "mailto:/sk-AAAABBBBCCCCDDDD.png",
        "custom://host/sk-AAAABBBBCCCCDDDD.png",
        "custom:/path/sk-AAAABBBBCCCCDDDD.png",
        "file:///tmp/sk-AAAABBBBCCCCDDDD.png",
        "file:/tmp/sk-AAAABBBBCCCCDDDD.png",
        "ssh://host/sk-AAAABBBBCCCCDDDD.png",
    ],
)
async def test_generic_url_cannot_fall_through_to_file_candidate(text: str) -> None:
    """An unsupported hierarchical URI is not a non-URL file token."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["%23", "%26", "%3D", "%20", "%40"])
async def test_once_decoded_nested_url_separator_is_structural(separator: str) -> None:
    """A once-decoded URL separator isolates a nested URL suffix."""
    text = f"https://example.com/?token=sk-ABCDEFGHIJ1{separator}https%3A%2F%2Fblocked.example%2Fx"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        "sk-Ahttps://B/https://inner.example/x",
        "sk-Ahttps%3A%2F%2FB%2Fhttps%3A%2F%2Finner.example%2Fx",
    ],
)
async def test_later_structural_nested_url_follows_scheme_text(value: str) -> None:
    """Earlier credential text cannot hide a later nested URL boundary."""
    text = f"https://outer.example/?token={value}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "suffix",
    [
        ",mailto:user@example.com",
        ",custom://host/path",
        "%2Cmailto%3Auser%40example.com",
        "%2Ccustom%3A%2F%2Fhost%2Fpath",
    ],
)
async def test_generic_nested_url_suffix_cannot_pad_query_candidate(suffix: str) -> None:
    """Generic nested URL shapes remain outside credential scoring."""
    text = f"https://example.com/?token=sk-ABCDEFGHIJ1{suffix}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "files/sk-AAAAmailto:BBBBCCCC.png",
        "files/sk-AAAAdata:textBBBBCCCC.png",
    ],
)
async def test_scheme_text_inside_file_basename_is_not_structural(text: str) -> None:
    """Embedded scheme text remains part of a supported file candidate."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("descendant", ["inner.example/path", "192.0.2.1/path"])
async def test_scheme_less_nested_descendant_preserves_query_ownership(descendant: str) -> None:
    """A nested scheme-less URL cannot expose a later nested credential."""
    text = f"https://outer.example/?next=,{descendant},https://inner.example/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("bridge", ['"inner.example/path,', ",(inner.example/path,"])
async def test_nested_query_ownership_crosses_scheme_less_descendant(bridge: str) -> None:
    """A nested scheme-less URL cannot expose a later nested credential."""
    text = f"https://outer.example/?next={bridge}https://inner.example/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "first_url",
    [
        "https://outer.example/path=",
        "https://outer.example/#label=",
        "https://outer.example:bad/?next=",
        "https://outer.example/?next==",
    ],
)
async def test_non_query_text_cannot_own_an_adjacent_credential_url(first_url: str) -> None:
    """Only a valid active query value can own an adjacent URL."""
    text = f"{first_url},https://[::1]/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text.replace("#", "")]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nested_urls",
    [
        "https://inner.example/,https://[::1]/?token=sk-AAAABBBBCCCCDDDD",
        "(https://inner.example/),(https://[::1]/?token=sk-AAAABBBBCCCCDDDD)",
    ],
)
async def test_nested_query_ownership_covers_adjacent_descendants(nested_urls: str) -> None:
    """Nested descendants remain inside the unsupported query value."""
    text = f"https://outer.example/?next=,{nested_urls}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://outer.example/https:?//inner&token=sk-AAAABBBBCCCCDDDD",
        "https://outer.example/https%3A?%2F%2Finner&token=sk-AAAABBBBCCCCDDDD",
        "https://outer.example/?token=sk-AAAABBBBCCCCDDDD&next=https:#//inner",
    ],
)
async def test_component_boundaries_cannot_fabricate_a_nested_url(text: str) -> None:
    """Unsupported URL components cannot suppress a supported query value."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text.replace("#", "")]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nested_value",
    [
        "https://inner.example/",
        "https%3A%2F%2Finner.example%2F",
    ],
)
async def test_nested_query_value_cannot_suppress_a_sibling_secret(nested_value: str) -> None:
    """A nested value cannot suppress a supported sibling query value."""
    text = f"https://outer.example/?token=sk-AAAABBBBCCCCDDDD&next={nested_value}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "nested_value",
    [
        "https://inner.example/",
        "inner.example/path",
        "192.0.2.1/path",
    ],
)
async def test_literal_nested_value_preserves_a_later_query_secret(
    nested_value: str,
) -> None:
    """A literal nested value cannot consume a later outer query field."""
    text = f"https://outer.example/?next=,{nested_value}&token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("opening", "closing"),
    [("[", "]"), ("(", ")"), ("{", "}"), ("<", ">"), ("'", "'"), ('"', '"')],
)
async def test_wrapped_open_query_cannot_suppress_a_later_query_secret(
    opening: str,
    closing: str,
) -> None:
    """A presentation wrapper ends unsupported nested-value ownership."""
    text = f"{opening}https://outer.example/?next={closing},https://[::1]/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_matched_single_quotes_do_not_pad_a_short_query_candidate() -> None:
    """Matched single quotes remain presentation rather than query data."""
    text = "'https://outer.example/?token=sk-ABCDEFGHIJ1'"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded_descendant",
    [
        "inner.example%2Fpath",
        "192.0.2.1%2Fpath",
    ],
)
async def test_encoded_scheme_less_descendant_does_not_pad_a_short_candidate(
    encoded_descendant: str,
) -> None:
    """Encoded scheme-less descendants match literal URL boundaries."""
    text = f"https://outer.example/?token=sk-ABCDEFGHIJ1%2C{encoded_descendant}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["inner.example", "192.0.2.1"])
@pytest.mark.parametrize(
    ("separator", "path_separator"),
    [
        ("/", "/"),
        ("\\", "/"),
        ("=", "/"),
        ("@", "/"),
        ("%2F", "%2F"),
        ("%5C", "%2F"),
        ("%3D", "%2F"),
        ("%40", "%2F"),
    ],
)
async def test_structural_descendant_does_not_pad_a_short_candidate(
    host: str,
    separator: str,
    path_separator: str,
) -> None:
    """All approved structural boundaries isolate scheme-less descendants."""
    text = f"https://outer.example/?token=sk-ABCDEFGHIJ1{separator}{host}{path_separator}path"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value",
    [
        "sk-AAAABBBB%40example.comCCCCDDDD",
        "sk-example.com-AAAABBBBCCCCDDDD",
    ],
)
async def test_domain_like_credential_data_remains_owned(value: str) -> None:
    """Email-like and prefix-internal domains remain credential data."""
    text = f"https://outer.example/?token={value}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "encoded_descendant",
    [
        "%09https%3A%2F%2Finner.example%2Fpath",
        "%09inner.example%2Fpath",
        "%09192.0.2.1%2Fpath",
        "%2Fsk-inner.example%2Fpath",
    ],
)
async def test_nested_url_cannot_hide_exact_aws_access_key(
    encoded_descendant: str,
) -> None:
    """A decoded descendant cannot erase its credential owner."""
    text = f"https://outer.example/?token=AKIA1234567890ABCDEF{encoded_descendant}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("closer", [")", "]", "}", ">"])
async def test_unmatched_closer_cannot_hide_sibling_query_secret(
    closer: str,
) -> None:
    """An unmatched closer terminates open-query ownership."""
    text = f"https://allow.example/?next={closer}https://blocked.example/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_nested_duplicate_cannot_hide_sibling_query_secret() -> None:
    """A detected URL is evaluated at every non-overlapping occurrence."""
    credential_url = "https://inner.example/?token=sk-AAAABBBBCCCCDDDD"
    text = f"https://outer.example/?next={credential_url}&x=1,{credential_url}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


def test_nested_url_scan_is_linear_at_the_container_limit() -> None:
    """Plain exact-limit query data is scanned once without URL retries."""
    assert _find_nested_url_start("a" * (64 * 1024)) is None  # noqa: S101


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-a.foo:/foo:/", 8),
        ("=sk-.mailto:", 5),
    ],
)
def test_nested_url_scan_preserves_non_overlapping_match_order(
    value: str,
    expected: int,
) -> None:
    """Linear scanning preserves legacy non-overlapping regex semantics."""
    assert _find_nested_url_start(value) == expected  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://[bad]/sk-AAAABBBBCCCCDDDD.png",
        "https://[bad]/sk-AAAABBBBCCCCDDDD.png,https://example.com",
        "https://[bad]/sk-AAAABBBBCCCCDDDD.png(https://example.com",
    ],
)
async def test_malformed_url_path_is_not_a_file_candidate(text: str) -> None:
    """A malformed URL cannot fall through to file-basename handling."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "label=https://example.com/sk-AAAABBBBCCCCDDDD.png",
        "label=https://[bad]/sk-AAAABBBBCCCCDDDD.png",
    ],
)
async def test_assignment_prefixed_url_path_is_not_a_file_candidate(text: str) -> None:
    """Assignment syntax cannot turn a URL path into a file candidate."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_assignment_prefixed_url_exposes_query_candidate() -> None:
    """Assignment syntax preserves supported HTTP query extraction."""
    text = "label=https://example.com/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authority",
    [
        "example.com:bad",
        "example.com:70000",
        "example.com:https://good.example",
        "[::1]:https://good.example",
    ],
)
async def test_malformed_url_authority_does_not_expose_query_values(authority: str) -> None:
    """Malformed ports cannot opt query values into candidate scoring."""
    text = f"https://{authority}/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_query_candidate_after_many_fields_is_detected() -> None:
    """Query parsing remains streaming without imposing a field-count bypass."""
    benign_fields = "&".join("field=value" for _ in range(1_000))
    text = f"https://example.com/?{benign_fields}&token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_oversized_exempt_container_fails_closed() -> None:
    """Oversized URL tokens cannot cause unbounded container parsing."""
    text = f"https://example.com/?padding={'a' * (64 * 1024)}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
async def test_exact_limit_exempt_container_remains_bounded() -> None:
    """The largest accepted exempt container is classified without rescans."""
    prefix = "https://example.com/?padding="
    text = prefix + "a" * (64 * 1024 - len(prefix))

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_oversized_marker_padded_container_fails_closed() -> None:
    """Presentation markers cannot bypass the raw container-size bound."""
    text = f"https://example.com/?padding={'*' * (64 * 1024)}"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text.replace("*", "")]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "files/#sk-AAAABBBBCCCCDDDD#.png",
        "#files/sk-AAAABBBBCCCCDDDD.png#",
        "files/*sk-AAAABBBBCCCCDDDD*.png",
        "*files/sk-AAAABBBBCCCCDDDD.png*",
        "files/sk-AAAABBBBCCCCDDDD.png*#",
        "#*files/sk-AAAABBBBCCCCDDDD.png*#",
    ],
)
async def test_file_presentation_markers_do_not_hide_candidates(text: str) -> None:
    """Equivalent hash and star markers preserve file-candidate detection."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text.replace("*", "").replace("#", "")]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "files/s*k-AAAABBBBCCCCDDDD.png",
        "files/s#k-AAAABBBBCCCCDDDD.png",
    ],
)
async def test_internal_file_markers_remain_candidate_content(text: str) -> None:
    """Literal markers inside a basename are content rather than wrappers."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", ["%2F", "%2f", "%5C", "%5c"])
async def test_encoded_file_separator_remains_in_raw_basename(separator: str) -> None:
    """An encoded separator remains content in the raw final basename."""
    text = f"files/sk-AAAABBBBCCCCDDDD{separator}tail.png"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected_secrets"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_encoding",
    ["%", "%F", "%GG", "%FF", "%C0%AF", "%ED%A0%80"],
)
async def test_invalid_file_encoding_is_rejected_atomically(invalid_encoding: str) -> None:
    """Invalid escapes and UTF-8 cannot synthesize candidate characters."""
    text = f"files/sk-ABCDEFGHIJ1{invalid_encoding}.png"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "files/sk%*2DAAAABBBBCCCCDDDD.png",
        "files/sk%#2DAAAABBBBCCCCDDDD.png",
        "https://example.com/?token=sk%*2DAAAABBBBCCCCDDDD",
    ],
)
async def test_presentation_markers_cannot_repair_percent_escapes(text: str) -> None:
    """Marker removal cannot synthesize a valid percent escape."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "files/sk-ABCDEFGHIJ1%C3*%A9.png",
        "files/sk-ABCDEFGHIJ1%C3#%A9.png",
        "https://example.com/?token=sk-ABCDEFGHIJ1%C3*%A9",
    ],
)
async def test_presentation_markers_cannot_repair_invalid_utf8(text: str) -> None:
    """Marker removal occurs only after strict raw-component validation."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "files/%23sk-AAAABBBBCCCCDDDD%23.png",
        "files/%2Ask-AAAABBBBCCCCDDDD%2A.png",
        "https://example.com/?token=%2Ask-AAAABBBBCCCCDDDD%2A",
    ],
)
async def test_encoded_markers_remain_candidate_content(text: str) -> None:
    """Encoded marker characters are not presentation syntax."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_mid_token_url_does_not_expose_query_candidates() -> None:
    """An HTTP substring without a container boundary remains unsupported."""
    text = "files/prefixhttps://example.com/?token=sk-AAAABBBBCCCCDDDD.md"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_double_slash_prefixed_http_url_does_not_expose_query_candidates() -> None:
    """A protocol-relative prefix cannot alias a later explicit scheme."""
    text = "//https://example.com/?token=sk-AAAABBBBCCCCDDDD"

    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "files/prefixhttps://example.com/sk-AAAABBBBCCCCDDDD.png",
        "//https://example.com/sk-AAAABBBBCCCCDDDD.png",
        "files/prefixssh://example.com/sk-AAAABBBBCCCCDDDD.png",
    ],
)
async def test_rejected_url_shape_cannot_fall_through_to_file_candidate(text: str) -> None:
    """A rejected URL-shaped span blocks standalone file extraction."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/sk-AAAABBBBCCCCDDDD",
        "https://example.com/#token=sk-AAAABBBBCCCCDDDD",
        "https://user:sk-AAAABBBBCCCCDDDD@example.com",
        "https://example.com/sk-AAAABBBBCCCCDDDD.png",
        "https://example.com/?token=token-AAAABBBBCCCCDDDD",
        "https://example.com/?client_secret=Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/?token=sk%252DAAAABBBBCCCCDDDD",
        "files/sk%252DAAAABBBBCCCCDDDD.png",
        ("https://example.com/?next=https%3A%2F%2Finner.example%2F%3Ftoken%3Dsk-AAAABBBBCCCCDDDD"),
        "files/sk-AAAABBBBCCCCDDDD/image.png",
        "https://[bad]/?token=sk-AAAABBBBCCCCDDDD",
    ],
)
async def test_secret_keys_keeps_unsupported_exempt_containers_exempt(
    text: str,
) -> None:
    """The narrow container contract does not grow into URL interpretation."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_custom_regex_does_not_expand_exempt_container_contract() -> None:
    """Custom patterns retain their existing whole-token matching semantics."""
    result = await secret_keys(
        None,
        "files/internal-ab12.png",
        SecretKeysCfg(
            threshold="balanced",
            custom_regex=[r"internal-[a-z0-9]{4}"],
        ),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?next=release-notes",
        "files/archive.png",
    ],
)
async def test_secret_keys_keeps_benign_supported_containers_exempt(
    text: str,
) -> None:
    """Benign values remain exempt in the two supported containers."""
    result = await secret_keys(
        None,
        text,
        SecretKeysCfg(threshold="balanced", custom_regex=None),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101
