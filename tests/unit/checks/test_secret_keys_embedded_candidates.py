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
SYNTHETIC_VALUE = "Aa0Bb1Cc2Dd3Ee4Ff5"
PUNCTUATED_SECRET = "sk-Ab1+Cd2Ef3Gh4Ij5Kl6"


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
        (
            "https%3A%2F%2Fexample.com%2Ftoken%2FAa0Bb1Cc2Dd3Ee4Ff5",
            "Aa0Bb1Cc2Dd3Ee4Ff5",
        ),
        ("https://[bad]/sk-AAAABBBBCCCCDDDD", SECRET),
        ("files/sk-AAAABBBBCCCCDDDD.png", SECRET),
        ("files/sk-AAAABBBBCCCCDDDD/image.png", SECRET),
        (r"files\sk-AAAABBBBCCCCDDDD\image.png", SECRET),
        ("files%2Fsk-AAAABBBBCCCCDDDD%2Fimage.png", SECRET),
        ("https://example.com/sk-AAAABBBBCCCCDDDD.png?download=1", SECRET),
        ("https://user:sk-AAAABBBBCCCCDDDD@example.com", SECRET),
        ("https://sk%2DAAAABBBBCCCCDDDD@example.com", SECRET),
        ("https://user:Aa0Bb1Cc2Dd3.json@example.com", "Aa0Bb1Cc2Dd3.json"),
        ("https://example.com/?token=Aa0Bb1'Cc2Dd3Ee4", "Aa0Bb1'Cc2Dd3Ee4"),
    ],
)
async def test_secret_keys_checks_embedded_candidates(text: str, expected: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", ["Aa0Bb1Cc2Dd3Ee4Ff5", SECRET])
async def test_repeated_embedded_secret_preserves_occurrences(secret: str) -> None:
    """Keep distinct source occurrences even when their values match."""
    text = f"https://example.com/?token={secret}&token={secret}"

    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [secret, secret]  # noqa: S101


@pytest.mark.asyncio
async def test_repeated_path_secret_preserves_occurrences() -> None:
    """Keep equal candidates from separate definite path segments."""
    text = f"https://example.com/{SECRET}/{SECRET}"

    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.info["detected_secrets"] == [SECRET, SECRET]  # noqa: S101


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


@pytest.mark.asyncio
@pytest.mark.parametrize(("opening", "closing"), [("[", "]"), ("%5B", "%5D"), ("%5b", "%5d")])
@pytest.mark.parametrize(
    "template",
    [
        "https://example.com/token/{opening}{value}{closing}",
        "https://example.com/#token/{opening}{value}{closing}",
        "files/token/{opening}{value}{closing}/image.png",
    ],
)
async def test_sensitive_path_labels_check_ambiguous_bracket_payloads(
    opening: str,
    closing: str,
    template: str,
) -> None:
    text = template.format(opening=opening, closing=closing, value=SYNTHETIC_VALUE)
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/token/[docs]",
        "https://example.com/token/[]/docs",
        "https://example.com/token/[0]/docs",
        "files/token/[docs]/image.png",
        f"https://example.com/catalog/[{SYNTHETIC_VALUE}]",
    ],
)
async def test_benign_bracket_indexes_and_payloads_remain_exempt(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("index", ["[docs]", "[0]", "[]"])
async def test_ambiguous_bracket_index_keeps_following_associated_value(index: str) -> None:
    text = f"https://example.com/client_secret/{index}/{SYNTHETIC_VALUE}"
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.info["detected_secrets"] == [SYNTHETIC_VALUE]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        f"https://example.com/foo+{PUNCTUATED_SECRET}",
        "https://example.com/foo%2Bsk-Ab1%2BCd2Ef3Gh4Ij5Kl6",
        "https://example.com/?value=foo%2Bsk-Ab1%2BCd2Ef3Gh4Ij5Kl6",
        f"https://example.com/#value=foo+{PUNCTUATED_SECRET}",
        f"files/foo+{PUNCTUATED_SECRET}/image.png",
    ],
)
async def test_component_prefix_scan_preserves_internal_uri_punctuation(text: str) -> None:
    result = await secret_keys(None, text, SecretKeysCfg(threshold="balanced"))

    assert result.info["detected_secrets"] == [PUNCTUATED_SECRET]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/foo+internal+ab12",
        "https://example.com/?value=foo%2Binternal%2Bab12",
        "files/foo+internal+ab12/image.png",
    ],
)
async def test_component_prefix_scan_preserves_custom_regex_suffix(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^internal\+ab12$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal+ab12"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        ("https://example.com/foo+Internal+Ab12", r"(?i)^internal\+ab12$", "Internal+Ab12"),
        ("https://example.com/?value=foo%2BInternal%2BAb12", r"(?i)^internal\+ab12$", "Internal+Ab12"),
        ("https://example.com/#value=foo+Internal+ab12", r"^[iI]nternal\+ab12$", "Internal+ab12"),
        ("files/foo+Internal+Ab12/image.png", r"(?x)^Internal \+ Ab12$", "Internal+Ab12"),
        ("https://example.com/foo+Internal%2FAb12", r"(?i)^internal/ab12$", "Internal/Ab12"),
        ("https://example.com/foo+a+b", r"^a(?!\+b)$", "a"),
        (
            "https://example.com/internal-bad,internal+good,internal-bad.png",
            r"^internal\+good$",
            "internal+good",
        ),
        (
            "files/internal+zzzz,internal+ab12,internal+yyyy/image.png",
            r"^internal\+[a][b]12$",
            "internal+ab12",
        ),
    ],
)
async def test_custom_component_ranges_cover_every_punctuation_boundary(
    text: str,
    pattern: str,
    expected: str,
) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "pattern", "expected"),
    [
        (
            "https://example.com/prefix+internal+ab12!+route.png",
            r"^internal\+ab12!$",
            "internal+ab12!",
        ),
        (
            "files/prefix+internal+ab12!+route/image.png",
            r"^internal\+ab12!$",
            "internal+ab12!",
        ),
        (
            "https://example.com/?value=prefix%2Binternal%2Bab12!&next=1",
            r"^internal\+ab12!$",
            "internal+ab12!",
        ),
        (
            "https://example.com/prefix+internal+ab12)+route.png",
            r"^internal\+ab12\)$",
            "internal+ab12)",
        ),
    ],
)
async def test_custom_component_match_may_end_with_punctuation(
    text: str,
    pattern: str,
    expected: str,
) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == [expected]  # noqa: S101


@pytest.mark.asyncio
async def test_custom_component_match_cannot_end_inside_a_word() -> None:
    text = "https://example.com/prefix+foobar+route.png"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^foo$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/foo+$internal+ab12",
        "https://example.com/foo%2B%24internal%2Bab12",
        "https://example.com/?value=foo%2B%24internal%2Bab12",
        "https://example.com/#value=foo+$internal+ab12",
        "files/foo+$internal+ab12/image.png",
    ],
)
async def test_custom_component_match_may_start_with_punctuation(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^\$internal\+ab12$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["$internal+ab12"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [r"^\+internal$", r"^\Winternal$"])
async def test_custom_component_punctuation_start_uses_python_match_semantics(pattern: str) -> None:
    text = "https://example.com/foo++internal+route.png"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["+internal"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/foo+$internalTail",
        "https://example.com/foo%2B%24internalTail",
        "https://example.com/?value=foo%2B%24internalTail",
        "https://example.com/#value=foo+$internalTail",
        "files/foo+$internalTail/image.png",
    ],
)
async def test_leading_punctuation_custom_prefix_extends_to_lexical_end(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^\$internal"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["$internalTail"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/foo-internal-ab12!/docs",
        "https://example.com/?value=foo-internal-ab12!&next=1",
        "files/foo-internal-ab12!/image.png",
    ],
)
async def test_unanchored_custom_match_preserves_trailing_punctuation(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^internal-ab12!"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal-ab12!"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("pattern", [r"^internal-ab12!", r"^internal-ab12!$"])
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/foo-internal-ab12!",
        "https://example.com/?value=foo-internal-ab12!",
        "https://example.com/#value=foo-internal-ab12!",
    ],
)
async def test_terminal_url_punctuation_keeps_embedded_custom_match(text: str, pattern: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[pattern])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal-ab12!"]  # noqa: S101


@pytest.mark.asyncio
async def test_terminal_query_uses_form_plus_semantics_before_prose_trim() -> None:
    text = "https://example.com/?value=foo+internal+ab12!"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^internal\+ab12!$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/internal=ab12!",
        "https://example.com/foo-internal=ab12!",
        "https://example.com/foo-internal%3Dab12!",
        "https://example.com/?value=foo-internal=ab12!",
        "https://example.com/#value=foo-internal=ab12!",
    ],
)
async def test_terminal_custom_scan_preserves_layer_local_equals(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^internal=ab12!$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal=ab12!"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/foo-internal&ab12!",
        "https://example.com/#foo-internal&ab12!",
    ],
)
async def test_terminal_path_and_fragment_preserve_ampersand_data(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^internal&ab12!$"])
    result = await secret_keys(None, text, config)

    assert result.info["detected_secrets"] == ["internal&ab12!"]  # noqa: S101


@pytest.mark.asyncio
async def test_nonmatching_terminal_custom_regex_does_not_exhaust_slug_budget() -> None:
    text = f"https://example.com/{'-'.join(['a'] * 800)}/image.png"
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"(?i)^never-match$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_terminal_prose_recovery_does_not_repeat_full_range_scan() -> None:
    base = "https://example.com/" + "%2F".join(["a"] * 130)
    config = SecretKeysCfg(
        threshold="balanced",
        custom_regex=[rf"^(?:z)never{index}$" for index in range(500)],
    )

    for text in (base, f"{base}!"):
        result = await secret_keys(None, text, config)
        assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/whiskey-Aa0Bb1Cc2Dd3Ee4Ff5",
        "https://example.com/monkey-Ab1Cd2Ef3Gh4Ij5Kl6",
        "https://example.com/foo+public+docs",
        "https://example.com/foo/sk-Ab1/Cd2Ef3Gh4Ij5Kl6/docs",
    ],
)
async def test_component_prefix_scan_requires_a_boundary_and_stays_in_segment(text: str) -> None:
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
