"""Tests for URL guardrail helpers."""

from __future__ import annotations

import asyncio
import re
import string
from datetime import timedelta
from statistics import median
from timeit import repeat

import pytest
from hypothesis import given, settings, strategies as st
from hypothesis.strategies import DrawFn

from guardrails.checks.text.urls import (
    AMBIGUOUS_DOMAIN_HOST_CANDIDATE_RE,
    AMBIGUOUS_IP_URL_RE,
    AMBIGUOUS_URL_REASON,
    DOMAIN_HOST_CANDIDATE_RE,
    IP_URL_RE,
    URLConfig,
    _clean_ambiguous_url_candidate,
    _detect_domain_like_url_spans,
    _detect_domain_like_urls,
    _detect_urls,
    _find_ambiguous_url_candidates,
    _is_url_allowed,
    _mark_unmatched_closing_delimiters,
    _token_end,
    _truncate_before_adjacent_scheme_less_url,
    _validate_url_security,
    urls,
)

REFERENCE_DOMAIN_PATTERN = re.compile(
    r"\b(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9.-]*\.[a-zA-Z]{2,}(?:/[^\s]*)?",
    re.IGNORECASE,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("foo.com-", ["foo.com"]),
        ("foo.com123", ["foo.com"]),
        ("foo.c0m", []),
        ("foo.com.a", ["foo.com"]),
        ("foo..example.com", ["foo..example.com"]),
        ("-example.com", ["example.com"]),
        ("_example.com", []),
        ("foo.com-evil/path", ["foo.com"]),
        ("foo.com/path?x=1", ["foo.com/path?x=1"]),
        ("abc/example.com", ["example.com"]),
        ("example.cİ", ["example.cİ"]),
        ("example.cı", ["example.cı"]),
        ("example.cſ", ["example.cſ"]),
        ("example.cK", ["example.cK"]),
        ("K.example.com", ["K.example.com"]),
        ("foo.ſſ/path", ["foo.ſſ/path"]),
    ],
)
def test_detect_domain_like_urls_preserves_existing_matches(
    text: str,
    expected: list[str],
) -> None:
    """The linear scanner should preserve established domain matching."""
    assert _detect_domain_like_urls(text) == expected  # noqa: S101


@given(
    text=st.text(
        alphabet=string.ascii_letters + string.digits + string.punctuation + "İıſK \t\n\r",
        max_size=100,
    )
)
def test_detect_domain_like_urls_matches_reference_pattern(text: str) -> None:
    """The linear scanner should match the previous regex on bounded text."""
    assert _detect_domain_like_urls(text) == REFERENCE_DOMAIN_PATTERN.findall(text)  # noqa: S101


@given(
    text=st.text(
        alphabet=string.ascii_letters + string.digits + string.punctuation + "é中 ",
        max_size=100,
    )
)
def test_ascii_boundary_detectors_cover_ordinary_matches(text: str) -> None:
    """ASCII-boundary scans conservatively cover ordinary URL matches."""
    ordinary_spans = _detect_domain_like_url_spans(text, DOMAIN_HOST_CANDIDATE_RE)
    ordinary_ip_matches = IP_URL_RE.finditer(text)
    ordinary_spans.extend(match.span() for match in ordinary_ip_matches)
    ambiguous_spans = _detect_domain_like_url_spans(
        text,
        AMBIGUOUS_DOMAIN_HOST_CANDIDATE_RE,
    )
    ambiguous_ip_matches = AMBIGUOUS_IP_URL_RE.finditer(text)
    ambiguous_spans.extend(match.span() for match in ambiguous_ip_matches)

    for ordinary_start, ordinary_end in ordinary_spans:
        ordinary_span_is_covered = False
        for ambiguous_start, ambiguous_end in ambiguous_spans:
            if ambiguous_start <= ordinary_start and ordinary_end <= ambiguous_end:
                ordinary_span_is_covered = True
                break
        assert ordinary_span_is_covered  # noqa: S101


def test_detect_urls_scales_linearly_for_invalid_domain_input() -> None:
    """Doubling invalid dotted input should not cause quadratic growth."""
    small_text = "a." * 2_000 + "-"
    large_text = "a." * 4_000 + "-"

    small_duration = median(repeat(lambda: _detect_urls(small_text), number=1, repeat=7))
    large_duration = median(repeat(lambda: _detect_urls(large_text), number=1, repeat=7))

    assert _detect_urls(small_text) == []  # noqa: S101
    assert _detect_urls(large_text) == []  # noqa: S101
    assert large_duration < small_duration * 3  # noqa: S101


def test_adjacent_scanner_validates_malformed_authority_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Many suffix candidates cannot repeatedly parse one bad authority."""
    parse_count = 0

    def reject_authority(_url: str) -> bool:
        nonlocal parse_count
        parse_count += 1
        return False

    monkeypatch.setattr(
        "guardrails.checks.text.urls._has_valid_http_authority",
        reject_authority,
    )
    text = "https://[bad/" + ",a.example" * 2_000

    assert _truncate_before_adjacent_scheme_less_url(text) == text  # noqa: S101
    assert parse_count == 1  # noqa: S101


def test_detect_urls_reuses_whitespace_token_end(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owned candidates in one token share a single token-end scan."""
    scan_count = 0

    def count_token_end(text: str, start: int) -> int:
        nonlocal scan_count
        scan_count += 1
        return _token_end(text, start)

    monkeypatch.setattr("guardrails.checks.text.urls._token_end", count_token_end)
    text = ",".join(f"https://h{index}.example/?next=,inner.example#frag" for index in range(500))

    _detect_urls(text)

    assert scan_count == 1  # noqa: S101


def test_adjacent_query_scanner_scales_linearly() -> None:
    """Adjacent query candidates do not rescan growing prefixes."""
    small_text = "https://valid/?q=x" + ",a.example" * 1_000
    large_text = "https://valid/?q=x" + ",a.example" * 4_000

    small_duration = median(repeat(lambda: _detect_urls(small_text), number=1, repeat=7))
    large_duration = median(repeat(lambda: _detect_urls(large_text), number=1, repeat=7))

    assert large_duration < small_duration * 6  # noqa: S101


def test_detect_urls_preserves_unicode_casefolded_domain() -> None:
    """Unicode characters matched by the previous regex remain detectable."""
    assert _detect_urls("Visit attacker.cK now") == ["attacker.cK"]  # noqa: S101


def test_ambiguous_scanner_ignores_control_only_text() -> None:
    """Control-only text does not become a URL candidate."""
    assert _find_ambiguous_url_candidates("\t\n\r") == []  # noqa: S101


@settings(max_examples=100)
@given(
    pairs=st.lists(
        st.sampled_from([("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]),
        max_size=50,
    )
)
def test_closing_delimiter_scanner_accepts_generated_nested_pairs(
    pairs: list[tuple[str, str]],
) -> None:
    """Properly nested generated delimiters have no unmatched offsets."""
    openings = "".join(opening for opening, _ in pairs)
    closings = "".join(closing for _, closing in reversed(pairs))
    text = f"{openings}payload{closings}"

    assert _mark_unmatched_closing_delimiters(text) is None  # noqa: S101


def test_closing_delimiter_scanner_allocates_mask_for_mismatch() -> None:
    """A mismatched closing delimiter is marked at its source offset."""
    unmatched_mask = _mark_unmatched_closing_delimiters("([)]")

    assert unmatched_mask is not None  # noqa: S101
    assert list(unmatched_mask) == [0, 0, 1, 0]  # noqa: S101


ASCII_URL_CONTROLS = pytest.mark.parametrize("control", ["\t", "\n", "\r"])

_AMBIGUOUS_URL_BASES = (
    ("http", "http://", "allowed.example/path_(x)"),
    ("https", "https://", "allowed.example/path_(x)"),
    ("ftp", "ftp://", "allowed.example/path_(x)"),
    ("http", "http://", "[::1]/internal"),
    ("data", "data:", "text/plain,hello"),
    ("javascript", "javascript:", "alert(1)"),
    ("vbscript", "vbscript:", "msgbox(1)"),
    ("mailto", "mailto:", "user@allowed.example"),
)
_AMBIGUOUS_URL_BOUNDARIES = (
    ("", ""),
    ("Use ", " now"),
    ("(", ")."),
    ("[", "]"),
)


@st.composite
def _ambiguous_url_cases(draw: DrawFn) -> tuple[str, str, str]:
    """Generate ambiguous URLs with varied controls and boundaries.

    Args:
        draw: Hypothesis draw function for composing generated values.

    Returns:
        The normalized scheme, raw candidate, and surrounding source text.
    """
    scheme, scheme_prefix, payload = draw(st.sampled_from(_AMBIGUOUS_URL_BASES))
    uppercase_flags = draw(
        st.lists(
            st.booleans(),
            min_size=len(scheme_prefix),
            max_size=len(scheme_prefix),
        )
    )
    cased_scheme_prefix = "".join(
        character.upper() if uppercase else character
        for character, uppercase in zip(
            scheme_prefix,
            uppercase_flags,
            strict=True,
        )
    )
    base_candidate = cased_scheme_prefix + payload
    control_position = draw(st.integers(min_value=1, max_value=len(base_candidate) - 1))
    control_run = draw(st.text(alphabet="\t\n\r", min_size=1, max_size=32))
    candidate = base_candidate[:control_position] + control_run + base_candidate[control_position:]
    prefix, suffix = draw(st.sampled_from(_AMBIGUOUS_URL_BOUNDARIES))

    return scheme, candidate, f"{prefix}{candidate}{suffix}"


@st.composite
def _delimiter_heavy_ambiguous_url_cases(draw: DrawFn) -> tuple[str, int]:
    """Generate control-bearing URLs with parser-valid delimiters.

    Args:
        draw: Hypothesis draw function for composing generated values.

    Returns:
        The raw URL-like segment and the first inserted control offset.
    """
    scheme = draw(st.sampled_from([value[1] for value in _AMBIGUOUS_URL_BASES]))
    body = draw(
        st.text(
            alphabet='abcXYZ019./:@?&#[](){}|^`\\<>"!;,-_+=',
            max_size=80,
        )
    )
    base_candidate = f"{scheme}{body}z"
    control_position = draw(st.integers(min_value=1, max_value=len(base_candidate) - 1))
    control_run = draw(st.text(alphabet="\t\n\r", min_size=1, max_size=8))
    candidate = base_candidate[:control_position] + control_run + base_candidate[control_position:]

    return candidate, control_position


@st.composite
def _scheme_less_ambiguous_url_cases(draw: DrawFn) -> str:
    """Generate scheme-less domain and IP control insertions.

    Args:
        draw: Hypothesis draw function for composing generated values.

    Returns:
        A raw scheme-less URL containing an ASCII URL control run.
    """
    base_candidate = draw(
        st.sampled_from(
            [
                "trusted.com",
                "sub.trusted.example/path",
                "192.168.1.12",
                "192.168.1.12:8080/internal",
            ]
        )
    )
    control_position = draw(st.integers(min_value=1, max_value=len(base_candidate) - 1))
    control_run = draw(st.text(alphabet="\t\n\r", min_size=1, max_size=16))

    return base_candidate[:control_position] + control_run + base_candidate[control_position:]


@st.composite
def _ambiguous_cleanup_cases(draw: DrawFn) -> tuple[str, int]:
    """Generate raw control-bearing candidates and protected lengths.

    Args:
        draw: Hypothesis draw function for composing generated values.

    Returns:
        A raw candidate containing an ASCII URL control and a prefix length
        that cleanup must preserve.
    """
    alphabet = string.ascii_letters + string.digits + string.punctuation + " \t\n\r"
    prefix = draw(st.text(alphabet=alphabet, max_size=50))
    control = draw(st.sampled_from(["\t", "\n", "\r"]))
    suffix = draw(st.text(alphabet=alphabet, max_size=50))
    candidate = prefix + control + suffix
    minimum_length = draw(st.integers(min_value=0, max_value=len(candidate)))

    return candidate, minimum_length


@settings(max_examples=200)
@given(case=_ambiguous_cleanup_cases())
def test_ambiguous_cleanup_preserves_generated_invariants(
    case: tuple[str, int],
) -> None:
    """Cleanup remains prefix-preserving, control-bearing, and idempotent."""
    candidate, minimum_length = case

    cleaned = _clean_ambiguous_url_candidate(candidate, minimum_length)

    assert candidate.startswith(cleaned)  # noqa: S101
    assert len(cleaned) >= minimum_length  # noqa: S101
    assert any(control in cleaned for control in "\t\n\r")  # noqa: S101
    assert _clean_ambiguous_url_candidate(cleaned, minimum_length) == cleaned  # noqa: S101


def test_detect_urls_deduplicates_scheme_and_domain() -> None:
    """Ensure detection removes trailing punctuation and avoids duplicate domains."""
    text = " ".join(
        (
            "Visit https://example.com/, http://example.com/path,",
            "example.com should not duplicate, and 192.168.1.10:8080.",
        )
    )
    detected = _detect_urls(text)

    assert "https://example.com/" in detected  # noqa: S101
    assert "http://example.com/path" in detected  # noqa: S101
    assert "example.com" not in detected  # noqa: S101
    assert "192.168.1.10:8080" in detected  # noqa: S101


@pytest.mark.parametrize("separator", [",", "("])
def test_detect_urls_splits_presentation_separated_scheme_urls(separator: str) -> None:
    """A later scheme starts a new URL after presentation punctuation."""
    first_url = "https://example.com/?token=sk-ABCDEFGHIJ1"
    second_url = "https://example.com/x"

    detected = _detect_urls(f"{first_url}{separator}{second_url}")

    assert detected == [first_url, second_url]  # noqa: S101


def test_detect_urls_recognizes_bracketed_ipv6_hosts() -> None:
    """A valid bracketed IPv6 host remains one explicit URL."""
    url = "https://[::1]/?token=sk-AAAABBBBCCCCDDDD"

    assert _detect_urls(url) == [url]  # noqa: S101


def test_detect_urls_stops_before_bracketed_presentation_text() -> None:
    """Ordinary URLs retain the released bracket presentation boundary."""
    url = "https://allow.example/path"

    assert _detect_urls(f"{url}[annotation]") == [url]  # noqa: S101


@pytest.mark.parametrize("suffix", ["", ",", ")", "."])
def test_detect_urls_preserves_bare_bracketed_ipv6_host(suffix: str) -> None:
    """Presentation cleanup preserves the structural authority bracket."""
    url = "https://[::1]"

    assert _detect_urls(f"{url}{suffix}") == [url]  # noqa: S101


@pytest.mark.parametrize(
    "explicit_url",
    [
        "https://normal.example",
        "https://normal.example:8443/path",
        "https://normal.example/path?x=1",
        "https://normal.example/path#fragment",
        "https://[::1]",
        "https://[::1]:8443/path",
        "https://[::1]/path?x=1",
        "https://[::1]/path#fragment",
    ],
)
@pytest.mark.parametrize(
    "scheme_less_url",
    [
        "blocked.example/path",
        "192.0.2.1:8080/path",
    ],
)
def test_detect_urls_splits_scheme_less_candidate_after_explicit_url(
    explicit_url: str,
    scheme_less_url: str,
) -> None:
    """An explicit URL cannot absorb an adjacent scheme-less candidate."""
    assert _detect_urls(f"{explicit_url},{scheme_less_url}") == [  # noqa: S101
        explicit_url,
        scheme_less_url,
    ]


@pytest.mark.parametrize("separator", [",", "("])
@pytest.mark.parametrize("destination", ["blocked.example/path", "192.0.2.1/path"])
def test_nonempty_query_value_ending_in_equals_does_not_own_url(
    separator: str,
    destination: str,
) -> None:
    """A trailing equals inside value content does not imply emptiness."""
    outer_url = "https://outer.example/?token=sk-ABCDEFGHIJ="

    assert _detect_urls(f"{outer_url}{separator}{destination}") == [  # noqa: S101
        outer_url,
        destination,
    ]


def test_nested_query_ownership_resets_at_next_field() -> None:
    """An owned empty value cannot authorize a later query field."""
    outer_url = "https://outer.example/?next=,inner.example/path&token=x"
    destination = "blocked.example/path"

    assert _detect_urls(f"{outer_url},{destination}") == [  # noqa: S101
        outer_url,
        destination,
    ]


@pytest.mark.parametrize(
    "nested_value",
    [
        "inner.example/path",
        "inner.example/path,second.example/path",
        "192.0.2.1/path,second.example/path",
    ],
)
def test_detect_urls_keeps_scheme_less_url_in_owned_query_value(nested_value: str) -> None:
    """An active empty query value retains nested scheme-less descendants."""
    text = f"https://outer.example/?next=,{nested_value}"

    assert _detect_urls(text) == [text]  # noqa: S101


def test_detect_urls_keeps_domain_text_inside_data_url() -> None:
    """HTTP adjacency rules cannot split a data URL payload."""
    url = "data:text/plain,example.com"

    assert _detect_urls(url) == [url]  # noqa: S101


@pytest.mark.parametrize("host", ["allow.example", "[::1]"])
def test_detect_urls_splits_period_delimited_domain_after_fragment(host: str) -> None:
    """A hostname suffix after fragment punctuation starts a new URL."""
    explicit_url = f"https://{host}/path?x=1#frag"
    scheme_less_url = "blocked.example/path"

    assert _detect_urls(f"{explicit_url}.{scheme_less_url}") == [  # noqa: S101
        explicit_url,
        scheme_less_url,
    ]


@pytest.mark.parametrize(
    "url",
    [
        "https://allow.example/path/foo.blocked.example/archive",
        "https://allow.example/?label=foo.blocked.example/path",
        "https://[::1]/path/foo.blocked.example/archive",
        "https://[::1]/?label=foo.blocked.example/path",
    ],
)
def test_detect_urls_keeps_dotted_path_and_query_components(url: str) -> None:
    """Domain-shaped path and query text stays inside its explicit URL."""
    assert _detect_urls(url) == [url]  # noqa: S101


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/items/alpha,foo.com/detail",
        "https://example.com/?labels=alpha,foo.com",
        "https://example.com/#labels=alpha,foo.com",
    ],
)
def test_detect_urls_preserves_domain_text_after_component_punctuation(url: str) -> None:
    """Valid URL components retain punctuation followed by domain text."""
    assert _detect_urls(  # noqa: S101
        url,
        preserved_component_urls=frozenset((url,)),
    ) == [url]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/items/alpha,foo.com/detail",
        "https://example.com/?labels=alpha,foo.com",
        "https://example.com/#labels=alpha,foo.com",
    ],
)
def test_exact_component_url_prefix_survives_wrappers_and_adjacency(url: str) -> None:
    """Presentation text cannot expose punctuation inside an exact URL."""
    preserved_urls = frozenset((url,))
    blocked_url = "blocked.example/path"

    assert _detect_urls(  # noqa: S101
        f"({url})",
        preserved_component_urls=preserved_urls,
    ) == [url]
    assert _detect_urls(  # noqa: S101
        f"{url},{blocked_url}",
        preserved_component_urls=preserved_urls,
    ) == [url, blocked_url]


@pytest.mark.parametrize(
    "url, source",
    [
        ("https://allow.example/?", "https://allow.example/?,blocked.example/path"),
        (
            "https://allow.example/path/foo.com)",
            "(https://allow.example/path/foo.com)),blocked.example/path",
        ),
    ],
)
def test_exact_component_url_skips_presentation_cleanup(url: str, source: str) -> None:
    """Exact source punctuation remains part of the preserved URL."""
    assert _detect_urls(  # noqa: S101
        source,
        preserved_component_urls=frozenset((url,)),
    ) == [url, "blocked.example/path"]


@pytest.mark.parametrize(
    "url",
    [
        "https://allow.example/path?",
        "https://allow.example/path)",
        "https://allow.example/path]",
    ],
)
@pytest.mark.parametrize(
    "sibling",
    [
        "https://blocked.example/path",
        "ftp://blocked.example/path",
        "blocked.example/path",
        "192.0.2.1/path",
    ],
)
def test_exact_terminal_component_preserves_zero_bridge_sibling(
    url: str,
    sibling: str,
) -> None:
    """Exact terminal punctuation and an adjacent URL retain source spans."""
    assert _detect_urls(  # noqa: S101
        f"{url}{sibling}",
        frozenset(("https", "ftp")),
        preserved_component_urls=frozenset((url,)),
    ) == [url, sibling]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/items/alpha,foo.com/detail",
        "https://example.com/?labels=alpha,foo.com",
    ],
)
async def test_urls_guardrail_exactly_allows_component_punctuation(url: str) -> None:
    """Exact allow-list entries preserve path and query punctuation."""
    result = await urls(
        ctx=None,
        data=url,
        config=URLConfig(url_allow_list=[url], allowed_schemes={"https"}),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/items/alpha,foo.com/detail",
        "https://example.com/?labels=alpha,foo.com",
        "https://example.com/#labels=alpha,foo.com",
    ],
)
async def test_exact_component_url_remains_allowed_before_adjacent_url(url: str) -> None:
    """An exact URL stays allowed while a separate destination is blocked."""
    blocked_url = "blocked.example/path"
    result = await urls(
        ctx=None,
        data=f"({url}),{blocked_url}",
        config=URLConfig(url_allow_list=[url], allowed_schemes={"https"}),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/items/alpha,foo.com/detail",
        "https://example.com/?labels=alpha,foo.com",
        "https://example.com/#labels=alpha,foo.com",
    ],
)
async def test_normalized_exact_url_remains_allowed_before_adjacent_url(url: str) -> None:
    """Allow-list normalization applies before component preservation."""
    blocked_url = "blocked.example/path"
    result = await urls(
        ctx=None,
        data=f"({url}),{blocked_url}",
        config=URLConfig(
            url_allow_list=[f"  {url.replace('example.com', 'EXAMPLE.COM')}  "],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, configured_url",
    [
        (
            "https://example.com/items/ALPHA,foo.com/detail",
            "https://example.com/items/alpha,foo.com/detail",
        ),
        (
            "https://example.com/?labels=ALPHA,foo.com",
            "https://example.com/?labels=alpha,foo.com",
        ),
        (
            "https://example.com/#labels=ALPHA,foo.com",
            "https://example.com/#labels=alpha,foo.com",
        ),
    ],
)
async def test_component_preservation_keeps_case_sensitive_source_identity(
    url: str,
    configured_url: str,
) -> None:
    """Preservation cannot rewrite case-sensitive URL components."""
    blocked_url = "blocked.example/path"
    result = await urls(
        ctx=None,
        data=f"({url}),{blocked_url}",
        config=URLConfig(
            url_allow_list=[configured_url],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert url not in result.info["allowed"]  # noqa: S101
    assert blocked_url in result.info["blocked"]  # noqa: S101
    assert configured_url not in result.info["detected"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/items/ALPHA,foo.com/detail",
        "https://example.com/?labels=ALPHA,foo.com",
        "https://example.com/#labels=ALPHA,foo.com",
    ],
)
async def test_exact_component_allow_list_preserves_matching_case(url: str) -> None:
    """An exact allow-list entry retains case-sensitive components."""
    result = await urls(
        ctx=None,
        data=url,
        config=URLConfig(
            url_allow_list=[url],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_exact_component_allow_list_normalizes_scheme_and_host_case() -> None:
    """Scheme and host case do not change exact component identity."""
    url = "https://example.com/?labels=ALPHA,foo.com"
    result = await urls(
        ctx=None,
        data=url,
        config=URLConfig(
            url_allow_list=["HTTPS://EXAMPLE.COM/?labels=ALPHA,foo.com"],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_userinfo, configured_userinfo",
    [
        ("alice", "bobby"),
        ("alice:secret", "alice:other"),
    ],
)
async def test_component_preservation_requires_exact_userinfo(
    source_userinfo: str,
    configured_userinfo: str,
) -> None:
    """Exact prefix preservation includes username and password identity."""
    suffix = "example.com/items/alpha,foo.com/detail"
    source_url = f"https://{source_userinfo}@{suffix}"
    configured_url = f"https://{configured_userinfo}@{suffix}"
    result = await urls(
        ctx=None,
        data=source_url,
        config=URLConfig(
            url_allow_list=[configured_url],
            allowed_schemes={"https"},
            block_userinfo=False,
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert configured_url not in result.info["detected"]  # noqa: S101
    assert source_url not in result.info["allowed"]  # noqa: S101


@pytest.mark.asyncio
async def test_preserved_open_query_cannot_hide_adjacent_url() -> None:
    """A separated URL cannot become content of an exact empty query value."""
    allowed_url = "https://allow.example/?next="
    blocked_url = "blocked.example/path"
    result = await urls(
        ctx=None,
        data=f"{allowed_url},{blocked_url}",
        config=URLConfig(
            url_allow_list=[allowed_url],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [allowed_url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [allowed_url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_url, allowed_schemes",
    [
        ("https://blocked.example/path", {"https"}),
        ("ftp://blocked.example/path", {"https", "ftp"}),
    ],
)
async def test_preserved_open_query_cannot_hide_adjacent_explicit_url(
    blocked_url: str,
    allowed_schemes: set[str],
) -> None:
    """A preserved empty value cannot own a separated explicit URL."""
    allowed_url = "https://allow.example/?next="
    result = await urls(
        ctx=None,
        data=f"{allowed_url},{blocked_url}",
        config=URLConfig(
            url_allow_list=[allowed_url],
            allowed_schemes=allowed_schemes,
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [allowed_url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [allowed_url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_url, mismatched_allow_url",
    [
        ("http://blocked.example/path", "https://blocked.example/path"),
        ("https://blocked.example/path", "http://blocked.example/path"),
    ],
)
async def test_colon_adjacent_url_preserves_explicit_scheme(
    blocked_url: str,
    mismatched_allow_url: str,
) -> None:
    """A separated explicit URL retains scheme-qualified validation."""
    allowed_url = "https://allow.example/?next="
    result = await urls(
        ctx=None,
        data=f"{allowed_url}:{blocked_url}",
        config=URLConfig(
            url_allow_list=[allowed_url, mismatched_allow_url],
            allowed_schemes={"http", "https"},
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [allowed_url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [allowed_url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
async def test_colon_adjacent_url_preserves_userinfo_validation() -> None:
    """A separated explicit URL retains userinfo validation."""
    allowed_url = "https://allow.example/?next="
    blocked_url = "https://user:pass@allowed.example/path"
    result = await urls(
        ctx=None,
        data=f"{allowed_url}:{blocked_url}",
        config=URLConfig(
            url_allow_list=[allowed_url, "allowed.example"],
            allowed_schemes={"https"},
            block_userinfo=True,
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [allowed_url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [allowed_url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "allowed_url",
    [
        "https://allow.example/path:",
        "https://allow.example/?label=value:",
        "https://allow.example/#label:",
    ],
)
async def test_terminal_boundary_preserves_adjacent_explicit_url(allowed_url: str) -> None:
    """A boundary owned by an exact URL cannot erase its adjacent sibling."""
    blocked_url = "https://user:pass@blocked.example/path"
    result = await urls(
        ctx=None,
        data=f"{allowed_url}{blocked_url}",
        config=URLConfig(
            url_allow_list=[allowed_url, "blocked.example"],
            allowed_schemes={"https"},
            block_userinfo=True,
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [allowed_url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [allowed_url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


def test_detect_urls_keeps_distinct_same_host_path() -> None:
    """Host deduplication cannot discard a distinct adjacent path."""
    explicit_url = "https://example.com/allowed"
    scheme_less_url = "example.com/blocked"

    assert _detect_urls(f"{explicit_url},{scheme_less_url}") == [  # noqa: S101
        explicit_url,
        scheme_less_url,
    ]


@pytest.mark.parametrize("scheme_less_url", ["example.com", "www.example.com"])
def test_detect_urls_keeps_adjacent_bare_same_host(scheme_less_url: str) -> None:
    """A source-distinct bare host remains a separate destination."""
    explicit_url = "https://example.com/allowed"

    assert _detect_urls(f"{explicit_url},{scheme_less_url}") == [  # noqa: S101
        explicit_url,
        scheme_less_url,
    ]


def test_detect_urls_preserves_released_candidate_category_order() -> None:
    """HTTP candidates remain ahead of scheme-less candidates."""
    first_url = "https://allowed.example/a"
    second_url = "blocked.example/x"
    third_url = "https://second.example/y"

    assert _detect_urls(f"{first_url},{second_url} {third_url}") == [  # noqa: S101
        first_url,
        third_url,
        second_url,
    ]


def test_detect_urls_splits_adjacent_ipv4_after_explicit_ipv4() -> None:
    """An explicit IPv4 path cannot absorb an adjacent IPv4 URL."""
    explicit_url = "https://192.0.2.1/allowed"
    scheme_less_url = "203.0.113.2/blocked"

    assert _detect_urls(f"{explicit_url},{scheme_less_url}") == [  # noqa: S101
        explicit_url,
        scheme_less_url,
    ]


@pytest.mark.parametrize("separator", [",", "("])
@pytest.mark.parametrize(
    ("first_url", "second_url", "expected"),
    [
        (
            "first.example/path",
            "second.example/path",
            ["first.example/path", "second.example/path"],
        ),
        (
            "first.example/path",
            "203.0.113.2/path",
            ["first.example/path", "203.0.113.2/path"],
        ),
        (
            "192.0.2.1/path",
            "second.example/path",
            ["second.example/path", "192.0.2.1/path"],
        ),
        (
            "192.0.2.1/path",
            "203.0.113.2/path",
            ["192.0.2.1/path", "203.0.113.2/path"],
        ),
    ],
)
def test_detect_urls_splits_adjacent_scheme_less_paths(
    separator: str,
    first_url: str,
    second_url: str,
    expected: list[str],
) -> None:
    """Presentation boundaries split every scheme-less producer pair."""
    assert _detect_urls(f"{first_url}{separator}{second_url}") == expected  # noqa: S101


def test_detect_urls_preserves_released_fallback_category_order() -> None:
    """Domain candidates remain ahead of IPv4 candidates."""
    explicit_url = "https://outer.example"
    ipv4_url = "192.0.2.1"
    domain_url = "blocked.example"

    assert _detect_urls(f"{explicit_url},{ipv4_url},{domain_url}") == [  # noqa: S101
        explicit_url,
        domain_url,
        ipv4_url,
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "blocked.example https://later.example",
            ["https://later.example", "blocked.example"],
        ),
        (
            "192.0.2.1 https://later.example",
            ["https://later.example", "192.0.2.1"],
        ),
        (
            "ftp://ftp.example https://later.example",
            ["https://later.example", "ftp://ftp.example"],
        ),
        (
            "data:text/plain,x https://later.example ftp://ftp.example javascript:alert",
            [
                "https://later.example",
                "ftp://ftp.example",
                "data:text/plain,x",
                "javascript:alert",
            ],
        ),
    ],
)
def test_detect_urls_preserves_released_mixed_category_order(
    text: str,
    expected: list[str],
) -> None:
    """Unrelated inputs retain the released detector ordering."""
    assert _detect_urls(text) == expected  # noqa: S101


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "trusted.co\tm data:text/plain,x",
            ["trusted.co\tm", "data:text/plain,x"],
        ),
        (
            "trusted.co\tm normal.example https://later.example",
            ["https://later.example", "trusted.co\tm", "normal.example"],
        ),
        (
            "trusted.co\tm ftp://later.example javascript:alert blocked.example 192.0.2.1",
            [
                "ftp://later.example",
                "trusted.co\tm",
                "javascript:alert",
                "blocked.example",
                "192.0.2.1",
            ],
        ),
    ],
)
def test_detect_urls_preserves_released_ambiguous_candidate_order(
    text: str,
    expected: list[str],
) -> None:
    """Ambiguous scheme-less candidates lead the fallback pass."""
    assert _detect_urls(text) == expected  # noqa: S101


def test_ambiguous_explicit_url_keeps_source_distinct_same_host_fallback() -> None:
    """Ambiguous hierarchical URLs do not hide separate same-host input."""
    ambiguous_url = "https://allowed.example/pa\tth"

    assert _detect_urls(f"{ambiguous_url} allowed.example") == [  # noqa: S101
        ambiguous_url,
        "allowed.example",
    ]


def test_host_deduplication_removes_only_a_leading_www_label() -> None:
    """Embedded www labels cannot hide a source-distinct bare host."""
    explicit_url = "https://notwww.example.com/path"
    bare_host = "notexample.com"

    assert _detect_urls(f"{explicit_url} {bare_host}") == [  # noqa: S101
        explicit_url,
        bare_host,
    ]


def test_scheme_less_path_scanner_scales_linearly() -> None:
    """Adjacent scheme-less paths do not rescan the remaining suffix."""
    small_text = ",".join(f"d{index}.example/path" for index in range(1_000))
    large_text = ",".join(f"d{index}.example/path" for index in range(4_000))

    small_duration = median(repeat(lambda: _detect_urls(small_text), number=1, repeat=5))
    large_duration = median(repeat(lambda: _detect_urls(large_text), number=1, repeat=5))

    assert large_duration < small_duration * 6  # noqa: S101


@pytest.mark.parametrize("bridge", ['"inner.example/path,', ",(inner.example/path,"])
def test_detect_urls_keeps_nested_scheme_after_scheme_less_descendant(bridge: str) -> None:
    """An open query value owns chained URLs throughout its source token."""
    text = f"https://outer.example/?next={bridge}https://inner.example/path"

    assert _detect_urls(text) == ["https://outer.example/?next="]  # noqa: S101


@pytest.mark.parametrize(
    ("opening", "closing"),
    [("[", "]"), ("(", ")"), ("{", "}"), ("<", ">"), ("'", "'"), ('"', '"')],
)
@pytest.mark.parametrize(
    "sibling_url",
    [
        "https://[::1]/path",
        "blocked.example/path",
        "192.0.2.1/path",
    ],
)
def test_wrapped_open_query_does_not_own_a_sibling_url(
    opening: str,
    closing: str,
    sibling_url: str,
) -> None:
    """A matching presentation closer ends nested query ownership."""
    outer_url = "https://outer.example/?next="
    text = f"{opening}{outer_url}{closing},{sibling_url}"

    assert _detect_urls(text) == [outer_url, sibling_url]  # noqa: S101


def test_matched_single_quotes_do_not_extend_a_detected_url() -> None:
    """A matched single quote remains outside the detected URL span."""
    url = "https://outer.example/?token=sk-ABCDEFGHIJ1"

    assert _detect_urls(f"'{url}'") == [url]  # noqa: S101


@pytest.mark.parametrize(
    "nested_value",
    [
        "https://inner.example/",
        "inner.example/path",
        "192.0.2.1/path",
    ],
)
def test_nested_query_ownership_resumes_at_the_next_field(nested_value: str) -> None:
    """A raw field separator resumes the accepted outer URL."""
    text = f"https://outer.example/?next=,{nested_value}&token=value"

    assert _detect_urls(text) == [text]  # noqa: S101


@pytest.mark.parametrize("host", ["outer.example", "[::1]"])
def test_nested_query_ownership_ends_before_fragment_destination(host: str) -> None:
    """Query ownership cannot suppress a later fragment-adjacent URL."""
    outer_url = f"https://{host}/?next=,inner.example/path#frag"
    blocked_url = "blocked.example/path"

    assert _detect_urls(f"{outer_url}.{blocked_url}") == [  # noqa: S101
        outer_url,
        blocked_url,
    ]


@pytest.mark.parametrize(
    "suffix",
    [
        "ftp://blocked.example/path",
        "data:text/plain,x",
        "javascript:alert",
        "vbscript:x",
    ],
)
def test_detect_urls_splits_adjacent_non_http_scheme(suffix: str) -> None:
    """A presentation-adjacent explicit scheme starts a new URL span."""
    outer_url = "https://allow.example/#frag"

    assert _detect_urls(f"{outer_url},{suffix}") == [outer_url, suffix]  # noqa: S101


@pytest.mark.parametrize(
    "authority",
    [
        "example.com:blocked.example/path",
        "example.com:192.0.2.1/path",
        "example.com:https://good.example/path",
        "[::1]:https://good.example/path",
    ],
)
def test_detect_urls_preserves_malformed_port_candidate(authority: str) -> None:
    """Invalid authority ports remain intact for fail-closed validation."""
    url = f"https://{authority}"

    assert _detect_urls(url) == [url]  # noqa: S101


@pytest.mark.parametrize("prefix", ["data:text/plain,payload", "javascript:alert", "vbscript:x"])
@pytest.mark.parametrize("suffix", ["blocked.example/path", "192.0.2.1/path"])
def test_detect_urls_splits_destination_after_hostless_payload(prefix: str, suffix: str) -> None:
    """A hostless payload cannot mask a later destination."""
    assert _detect_urls(f"{prefix},{suffix}") == [prefix, suffix]  # noqa: S101


@pytest.mark.parametrize("host", ["localhost", "intranet", "devbox"])
def test_detect_urls_splits_destination_after_single_label_host(host: str) -> None:
    """Every released-valid HTTP host supports adjacency splitting."""
    explicit_url = f"https://{host}/ok"
    blocked_url = "blocked.example/path"

    assert _detect_urls(f"{explicit_url},{blocked_url}") == [  # noqa: S101
        explicit_url,
        blocked_url,
    ]


@pytest.mark.parametrize("descendant", ["inner.example/path", "192.0.2.1/path"])
def test_query_owner_covers_scheme_less_then_explicit_descendant(descendant: str) -> None:
    """A scheme-less descendant cannot end empty-query ownership."""
    text = f"https://outer.example/?next=,{descendant},https://inner.example/?token=value"

    assert _detect_urls(text) == [f"https://outer.example/?next=,{descendant}"]  # noqa: S101


def test_detect_urls_preserves_source_order_for_bracketed_hosts() -> None:
    """Bracketed-host matches remain ordered by their source offsets."""
    first_url = "https://normal.example/a"
    second_url = "https://[::1]/b"

    assert _detect_urls(f"{first_url} {second_url}") == [first_url, second_url]  # noqa: S101


@pytest.mark.parametrize("prefix", ["label=(", "label=,"])
def test_detect_urls_preserves_top_level_urls_after_assignment_text(prefix: str) -> None:
    """Assignment-like prose cannot claim ownership of a following URL."""
    url = "https://[::1]/path"

    assert _detect_urls(f"{prefix}{url}") == [url]  # noqa: S101


@pytest.mark.parametrize(
    "first_url",
    [
        "https://outer.example/path=",
        "https://outer.example/#label=",
        "https://outer.example:bad/?next=",
        "https://outer.example/?next==",
    ],
)
def test_detect_urls_requires_valid_query_ownership(first_url: str) -> None:
    """Unsupported components and malformed URLs cannot own a sibling URL."""
    second_url = "https://[::1]/?token=value"

    assert _detect_urls(f"{first_url},{second_url}") == [first_url, second_url]  # noqa: S101


@pytest.mark.parametrize(
    "authority",
    [
        "outer.example:bad",
        "outer.example:70000",
        "outer.example:https://good.example",
        "[bad]",
    ],
)
@pytest.mark.parametrize("bridge", ["inner.example/path", "192.0.2.1/path"])
def test_malformed_query_owner_cannot_bridge_to_sibling_url(
    authority: str,
    bridge: str,
) -> None:
    """A malformed authority cannot suppress a later valid URL."""
    malformed_url = f"https://{authority}/?next="
    valid_url = "https://inner.example/?token=sk-AAAABBBBCCCCDDDD"
    text = f"{malformed_url},{bridge},{valid_url}"

    assert _detect_urls(text) == [f"{malformed_url},{bridge}", valid_url]  # noqa: S101


@pytest.mark.parametrize("closer", [")", "]", "}", ">"])
@pytest.mark.parametrize(
    "sibling_url",
    [
        "blocked.example/path",
        "192.0.2.1/path",
        "https://blocked.example/path",
    ],
)
def test_unmatched_closer_terminates_open_query_ownership(
    closer: str,
    sibling_url: str,
) -> None:
    """An unmatched closer leaves the following URL independently visible."""
    outer_url = "https://allow.example/?next="

    assert _detect_urls(f"{outer_url}{closer},{sibling_url}") == [  # noqa: S101
        outer_url,
        sibling_url,
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "https://outer.example/?next=https://[::1]/?token=value",
            ["https://outer.example/?next=https://[::1]/?token=value"],
        ),
        (
            "https://outer.example/x/https://[fe80::1%25eth0]:8443/?token=value",
            ["https://outer.example/x/https://[fe80::1%25eth0]:8443/?token=value"],
        ),
        (
            "https://outer.example/?next=,https://inner.example/?token=value",
            ["https://outer.example/?next="],
        ),
        (
            "https://outer.example/?next=,https://inner.example/,https://[::1]/?token=value",
            ["https://outer.example/?next="],
        ),
        (
            "https://outer.example/?next=(https://inner.example/),(https://[::1]/?token=value)",
            ["https://outer.example/?next="],
        ),
        (
            "https://outer.example/?next=,https://127.0.0.1/?token=value",
            ["https://outer.example/?next="],
        ),
        (
            "https://outer.example/?next=,https://192.168.1.10:8080/?token=value",
            ["https://outer.example/?next="],
        ),
    ],
)
def test_detect_urls_does_not_promote_nested_urls(
    text: str,
    expected: list[str],
) -> None:
    """Nested schemes do not become independent top-level URL spans."""
    assert _detect_urls(text) == expected  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_ignores_bracketed_presentation_suffix() -> None:
    """Presentation text cannot invalidate an exact-path allow-list entry."""
    url = "https://allow.example/path"
    config = URLConfig(
        url_allow_list=[url],
        allowed_schemes={"https"},
    )

    result = await urls(ctx=None, data=f"{url}[annotation]", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_allows_allowlisted_bracketed_ipv6_host() -> None:
    """Normal bracketed IPv6 URLs reach the shared validation path."""
    url = "https://[::1]/docs"
    config = URLConfig(
        url_allow_list=["[::1]"],
        allowed_schemes={"https"},
    )

    result = await urls(ctx=None, data=url, config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_allows_bare_allowlisted_bracketed_ipv6_host() -> None:
    """A bare bracketed IPv6 authority reaches allow-list validation intact."""
    url = "https://[::1]"
    config = URLConfig(
        url_allow_list=["[::1]"],
        allowed_schemes={"https"},
    )

    result = await urls(ctx=None, data=f"{url},", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_does_not_promote_nested_ipv4_fallback() -> None:
    """Scheme-less fallback cannot reintroduce an owned nested IPv4 URL."""
    outer_url = "https://outer.example/?next="
    text = f"{outer_url},https://127.0.0.1/?token=value"
    config = URLConfig(
        url_allow_list=["outer.example"],
        allowed_schemes={"https"},
    )

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [outer_url]  # noqa: S101
    assert result.info["allowed"] == [outer_url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "explicit_url, allow_entry",
    [
        ("https://normal.example", "normal.example"),
        ("https://normal.example:8443/path", "normal.example"),
        ("https://normal.example/path?x=1", "normal.example"),
        ("https://normal.example/path#fragment", "normal.example"),
        ("https://[::1]", "[::1]"),
        ("https://[::1]:8443/path", "[::1]"),
        ("https://[::1]/path?x=1", "[::1]"),
        ("https://[::1]/path#fragment", "[::1]"),
    ],
)
@pytest.mark.parametrize(
    "scheme_less_url",
    [
        "blocked.example/path",
        "192.0.2.1:8080/path",
    ],
)
async def test_urls_guardrail_blocks_scheme_less_url_after_allowlisted_url(
    explicit_url: str,
    allow_entry: str,
    scheme_less_url: str,
) -> None:
    """An allowlisted URL cannot hide an adjacent blocked candidate."""
    config = URLConfig(
        url_allow_list=[allow_entry],
        allowed_schemes={"https"},
    )

    result = await urls(
        ctx=None,
        data=f"{explicit_url},{scheme_less_url}",
        config=config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [explicit_url, scheme_less_url]  # noqa: S101
    assert result.info["allowed"] == [explicit_url]  # noqa: S101
    assert result.info["blocked"] == [scheme_less_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("closer", [")", "]", "}", ">"])
async def test_urls_guardrail_blocks_sibling_after_unmatched_closer(
    closer: str,
) -> None:
    """An allowlisted open query cannot own past an unmatched closer."""
    outer_url = "https://allow.example/?next="
    blocked_url = "blocked.example/path"

    result = await urls(
        ctx=None,
        data=f"{outer_url}{closer},{blocked_url}",
        config=URLConfig(
            url_allow_list=["allow.example"],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [outer_url, blocked_url]  # noqa: S101
    assert result.info["allowed"] == [outer_url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, allow_entry, blocked_url",
    [
        (
            "https://example.com/allowed,example.com/blocked",
            "https://example.com/allowed",
            "example.com/blocked",
        ),
        (
            "https://192.0.2.1/allowed,203.0.113.2/blocked",
            "192.0.2.1",
            "203.0.113.2/blocked",
        ),
    ],
)
async def test_urls_guardrail_blocks_distinct_adjacent_destination(
    text: str,
    allow_entry: str,
    blocked_url: str,
) -> None:
    """An allowlisted URL cannot hide a distinct adjacent destination."""
    result = await urls(
        ctx=None,
        data=text,
        config=URLConfig(url_allow_list=[allow_entry], allowed_schemes={"https"}),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "blocked_url, allowed_schemes",
    [
        ("ftp://blocked.example/path", {"https", "ftp"}),
    ],
)
async def test_urls_guardrail_blocks_adjacent_non_http_url(
    blocked_url: str,
    allowed_schemes: set[str],
) -> None:
    """An allowlisted HTTP URL cannot hide another explicit scheme."""
    outer_url = "https://allow.example/#frag"
    result = await urls(
        ctx=None,
        data=f"{outer_url},{blocked_url}",
        config=URLConfig(
            url_allow_list=["allow.example"],
            allowed_schemes=allowed_schemes,
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["allowed"] == [outer_url]  # noqa: S101
    assert result.info["blocked"] == [blocked_url]  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_preserves_userinfo_for_validation() -> None:
    """Domain-shaped userinfo cannot be reinterpreted as adjacent URLs."""
    text = "https://user:allowed.example@user/path"
    result = await urls(
        ctx=None,
        data=text,
        config=URLConfig(
            url_allow_list=["user", "allowed.example"],
            allowed_schemes={"https"},
            block_userinfo=True,
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [text]  # noqa: S101
    assert result.info["blocked"] == [text]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme_less_url", ["example.com", "www.example.com"])
async def test_urls_guardrail_validates_adjacent_bare_same_host(scheme_less_url: str) -> None:
    """A path-restricted allowlist cannot discard an adjacent bare host."""
    explicit_url = "https://example.com/allowed"
    result = await urls(
        ctx=None,
        data=f"{explicit_url},{scheme_less_url}",
        config=URLConfig(
            url_allow_list=[explicit_url],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["allowed"] == [explicit_url]  # noqa: S101
    assert result.info["blocked"] == [scheme_less_url]  # noqa: S101


@pytest.mark.asyncio
async def test_ambiguous_url_does_not_hide_source_distinct_blocked_host() -> None:
    """A distinct bare host remains visible to allowlist validation."""
    ambiguous_url = "https://allowed.example/pa\tth"
    bare_host = "allowed.example"
    result = await urls(
        ctx=None,
        data=f"{ambiguous_url} {bare_host}",
        config=URLConfig(
            url_allow_list=["https://allowed.example/safe"],
            allowed_schemes={"https"},
        ),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous_url, bare_host]  # noqa: S101
    assert result.info["blocked"] == [ambiguous_url, bare_host]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("prefix", ["label=(", "label=,"])
async def test_urls_guardrail_blocks_top_level_url_after_assignment_text(prefix: str) -> None:
    """Assignment-like prose cannot suppress a blocked bracketed URL."""
    url = "https://[::1]/path"
    config = URLConfig(
        url_allow_list=[],
        allowed_schemes={"https"},
    )

    result = await urls(ctx=None, data=f"{prefix}{url}", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [url]  # noqa: S101


@ASCII_URL_CONTROLS
def test_detect_urls_preserves_control_obfuscated_http_scheme(control: str) -> None:
    """Python-normalized scheme obfuscation must remain raw."""
    candidate = f"htt{control}p://2130706433:8000/internal/credentials"

    detected = _detect_urls(candidate)

    assert detected == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("candidate_template", "allowed_prefix"),
    [
        ("trusted.co{control}m", "trusted.co"),
        ("192.168.1.1{control}2", "192.168.1.1"),
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_control_bearing_scheme_less_url(
    control: str,
    candidate_template: str,
    allowed_prefix: str,
) -> None:
    """A scheme-less safe prefix cannot hide a normalized continuation."""
    candidate = candidate_template.format(control=control)
    config = URLConfig(url_allow_list=[allowed_prefix], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("candidate_template", "allowed_prefix"),
    [
        ("é{control}trusted.com", "trusted.com"),
        ("é{control}trusted.co{control}m", "trusted.co"),
        ("é{control}192.168.1.1{control}2", "192.168.1.1"),
        ("a1{control}92.168.1.12", "92.168.1.12"),
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_preserves_raw_scheme_less_boundaries(
    control: str,
    candidate_template: str,
    allowed_prefix: str,
) -> None:
    """Control-created boundaries survive scheme-less normalization."""
    candidate = candidate_template.format(control=control)
    config = URLConfig(
        url_allow_list=[allowed_prefix],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@given(
    prefix=st.text(alphabet=string.ascii_letters + string.digits, min_size=1, max_size=16),
    control=st.sampled_from(["\t", "\n", "\r"]),
    suffix=st.sampled_from(
        [
            "92.168.1.12",
            "92.168.1.12:8080",
            "92.168.1.12/internal/credentials",
        ]
    ),
)
def test_ambiguous_scanner_preserves_control_created_ip_boundaries(
    prefix: str,
    control: str,
    suffix: str,
) -> None:
    """Raw IP boundaries remain visible when normalization joins a prefix."""
    candidate = prefix + control + suffix

    detected = _find_ambiguous_url_candidates(candidate)

    assert detected == [(candidate, (0, len(candidate)))]  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(prefix_length=st.integers(min_value=10_000, max_value=50_000))
def test_ambiguous_scanner_handles_long_prefix_before_control_created_ip(
    prefix_length: int,
) -> None:
    """Raw IP boundary fallback remains bounded for long joined prefixes."""
    candidate = "a" * prefix_length + "\t92.168.1.12"

    detected = _find_ambiguous_url_candidates(candidate)

    assert detected == [(candidate, (0, len(candidate)))]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("candidate_template", "expected_template", "allowed_prefix"),
    [
        ("étrusted.co{control}m", "trusted.co{control}m", "trusted.co"),
        ("étrusted.c{control}K", "trusted.c{control}K", "trusted.cK"),
        ("étrusted.ſ{control}ſ", "trusted.ſ{control}ſ", "trusted.ſſ"),
        ("é192.168.1.1{control}2", "192.168.1.1{control}2", "192.168.1.1"),
        ("_trusted.co{control}m", "trusted.co{control}m", "trusted.co"),
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_uses_ascii_boundaries_for_ambiguous_urls(
    control: str,
    candidate_template: str,
    expected_template: str,
    allowed_prefix: str,
) -> None:
    """Unicode word boundaries cannot suppress ambiguous URL detection."""
    candidate = candidate_template.format(control=control)
    expected = expected_template.format(control=control)
    config = URLConfig(url_allow_list=[allowed_prefix], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [expected]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [expected]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{expected}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("host", ["trusted.com", "192.168.1.1"])
@pytest.mark.asyncio
async def test_urls_guardrail_keeps_control_bearing_scheme_less_userinfo_prefix(
    control: str,
    host: str,
) -> None:
    """Controls in scheme-less userinfo remain part of the raw candidate."""
    candidate = f"user{control}@{host}"
    config = URLConfig(url_allow_list=[host], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_treats_leading_control_as_whitespace(control: str) -> None:
    """A leading control remains a separator from a normal URL."""
    candidate = "trusted.com"
    config = URLConfig(url_allow_list=[candidate], allowed_schemes={"http"})

    result = await urls(ctx=None, data=f"{control}{candidate}", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == [candidate]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_ignores_only_leading_control_prefix(control: str) -> None:
    """An internal control remains ambiguous after a leading separator."""
    candidate = f"é{control}trusted.com"
    config = URLConfig(url_allow_list=["trusted.com"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=f"{control}{candidate}", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_merges_duplicate_shadow_boundaries(control: str) -> None:
    """Shadow and ordinary matches produce one complete raw candidate."""
    candidate = f"prefix/{control}trusted.com"
    config = URLConfig(url_allow_list=["trusted.com"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("delimiter", ["[", "]", "\\", "<", ">", '"', "{", "}", "|", "^", "`"])
@pytest.mark.asyncio
async def test_urls_guardrail_keeps_scheme_less_delimiter_continuation(
    control: str,
    delimiter: str,
) -> None:
    """A scheme-less delimiter cannot hide a later URL control."""
    candidate = f"trusted.co{delimiter}{control}m"
    config = URLConfig(url_allow_list=["trusted.co"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_does_not_bridge_scheme_less_prose() -> None:
    """Scheme-less URLs do not absorb later prose containing controls."""
    text = "See trusted.co for notes about user@example.com before\ta demo"
    config = URLConfig(
        url_allow_list=["trusted.co", "example.com"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == ["trusted.co", "example.com"]  # noqa: S101
    assert result.info["allowed"] == ["trusted.co", "example.com"]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_separates_adjacent_scheme_less_candidates() -> None:
    """A normal domain remains separate from an ambiguous neighbor."""
    normal = "normal.example"
    ambiguous = "trusted.co\tm"
    config = URLConfig(
        url_allow_list=[normal, "trusted.co"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=f"{normal},{ambiguous}", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous, normal]  # noqa: S101
    assert result.info["allowed"] == [normal]  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101


@pytest.mark.parametrize(
    "text_template",
    [
        "[{normal}]({ambiguous})",
        "[{ambiguous}]({normal})",
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_separates_mixed_markdown_candidates(
    text_template: str,
) -> None:
    """Scheme-less normal and explicit ambiguous URLs stay independent."""
    normal = "normal.example"
    ambiguous = "http://trusted.co\tm"
    config = URLConfig(
        url_allow_list=[normal, "trusted.co"],
        allowed_schemes={"http"},
    )

    result = await urls(
        ctx=None,
        data=text_template.format(normal=normal, ambiguous=ambiguous),
        config=config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous, normal]  # noqa: S101
    assert result.info["allowed"] == [normal]  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_trims_scheme_less_markdown_boundary(control: str) -> None:
    """A scheme-less link label excludes adjacent Markdown syntax."""
    ambiguous = f"trusted.co{control}m"
    normal = "normal.example"
    text = f"[{ambiguous}](destination)[{normal}](destination)"
    config = URLConfig(
        url_allow_list=["trusted.co", normal],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous, normal]  # noqa: S101
    assert result.info["allowed"] == [normal]  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{ambiguous}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("destination_control", ["\t", "\n", "\r"])
@pytest.mark.asyncio
async def test_urls_guardrail_trims_markdown_before_later_control(
    control: str,
    destination_control: str,
) -> None:
    """A later destination control cannot extend the link-label candidate."""
    ambiguous = f"trusted.co{control}m"
    text = f"[{ambiguous}](destination{destination_control}text)"
    config = URLConfig(
        url_allow_list=["trusted.co"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{ambiguous}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_preserves_explicit_markdown_destination(
    control: str,
) -> None:
    """Trimming a scheme-less label preserves the destination URL shape."""
    ambiguous = f"trusted.co{control}m"
    normal = "https://normal.example/docs"
    text = f"[{ambiguous}]({normal})"
    config = URLConfig(
        url_allow_list=["trusted.co", normal],
        allowed_schemes={"http", "https"},
    )

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [normal, ambiguous]  # noqa: S101
    assert result.info["allowed"] == [normal]  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{ambiguous}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("opening", "closing"),
    [
        ("(", ")"),
        ("[", "]"),
        ("<", ">"),
        ("{", "}"),
        ('"', '"'),
        ("`", "`"),
        ("|", "|"),
        ("^", "^"),
        ("\\", "\\"),
    ],
)
@pytest.mark.parametrize("wrapper_width", [1, 2])
@pytest.mark.asyncio
async def test_urls_guardrail_trims_unmatched_wrapping_delimiter(
    control: str,
    opening: str,
    closing: str,
    wrapper_width: int,
) -> None:
    """A wrapping prose delimiter is excluded from the raw candidate."""
    ambiguous = f"trusted.co{control}m"
    config = URLConfig(
        url_allow_list=["trusted.co"],
        allowed_schemes={"http"},
    )

    result = await urls(
        ctx=None,
        data=f"{opening * wrapper_width}{ambiguous}{closing * wrapper_width}",
        config=config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{ambiguous}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("delimiter", ["[", "{", "<"])
@pytest.mark.asyncio
async def test_urls_guardrail_trims_unmatched_trailing_opening_delimiter(
    control: str,
    delimiter: str,
) -> None:
    """An unmatched trailing opener is excluded from the raw candidate."""
    ambiguous = f"trusted.co{control}m"
    config = URLConfig(
        url_allow_list=["trusted.co"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=ambiguous + delimiter, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{ambiguous}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("prefix_delimiter", "trailing_delimiter"),
    [("]", "["), ("}", "{"), (">", "<")],
)
@pytest.mark.asyncio
async def test_urls_guardrail_does_not_pair_inverted_delimiters(
    control: str,
    prefix_delimiter: str,
    trailing_delimiter: str,
) -> None:
    """Oppositely ordered delimiters cannot retain a trailing boundary."""
    ambiguous = f"trusted.co{prefix_delimiter}{control}m"
    config = URLConfig(
        url_allow_list=["trusted.co"],
        allowed_schemes={"http"},
    )

    result = await urls(
        ctx=None,
        data=ambiguous + trailing_delimiter,
        config=config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{ambiguous}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    "boundary",
    ["[", "{", "<", '"', "`", "|", "^", "\\", ")", "]", "}", ">"],
)
@pytest.mark.asyncio
async def test_urls_guardrail_stops_at_post_control_hard_boundary(
    control: str,
    boundary: str,
) -> None:
    """Adjacent text after a hard boundary stays outside the candidate."""
    ambiguous = f"trusted.co{control}m"
    config = URLConfig(
        url_allow_list=["trusted.co", "adjacent.example"],
        allowed_schemes={"http"},
    )

    result = await urls(
        ctx=None,
        data=f"{ambiguous}{boundary}adjacent.example",
        config=config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [ambiguous, "adjacent.example"]  # noqa: S101
    assert result.info["allowed"] == ["adjacent.example"]  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101
    assert result.info["blocked_reasons"][0] == f"{ambiguous}: {AMBIGUOUS_URL_REASON}"  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_preserves_matched_ipv6_delimiters(control: str) -> None:
    """A control-bearing IPv6 URL retains its matched address brackets."""
    candidate = f"http://[::{control}1]"
    config = URLConfig(
        url_allow_list=["[::1]"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_keeps_unconfirmed_scheme_less_markdown_boundary(
    control: str,
) -> None:
    """A Markdown-like delimiter before a control remains in the candidate."""
    candidate = f"trusted.example/safe](junk{control}admin"
    config = URLConfig(
        url_allow_list=["trusted.example/safe"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_preserves_final_lf_after_punctuation() -> None:
    """Cleanup cannot remove a final LF from an ambiguous candidate."""
    candidate = "http://trusted.example!\n"
    config = URLConfig(
        url_allow_list=["trusted.example"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_preserves_scheme_less_userinfo_candidate(
    control: str,
) -> None:
    """Scheme-less userinfo remains one raw ambiguous candidate."""
    candidate = f"trusted.com{control}@evil.example/path"
    config = URLConfig(
        url_allow_list=["trusted.com", "evil.example"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_configured_scheme_obfuscation(
    control: str,
) -> None:
    """Configured hierarchical schemes cannot hide controls."""
    candidate = f"cust{control}om://trusted.example/path"
    config = URLConfig(
        url_allow_list=["trusted.example"],
        allowed_schemes={"custom"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_configured_hostless_scheme_obfuscation(
    control: str,
) -> None:
    """Configured hostless schemes cannot hide controls."""
    candidate = f"cust{control}om:user@trusted.example"
    config = URLConfig(
        url_allow_list=["trusted.example"],
        allowed_schemes={"custom"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_blocks_long_configured_scheme_obfuscation() -> None:
    """Arbitrarily long configured schemes remain fail-closed."""
    scheme = "a" * 4_096
    candidate = f"{scheme[:2048]}\t{scheme[2048:]}://trusted.example/path"
    config = URLConfig(
        url_allow_list=["trusted.example"],
        allowed_schemes={scheme},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


def test_urls_guardrail_scales_linearly_for_long_configured_scheme() -> None:
    """Public scanning avoids repeated long configured-scheme matches."""

    def scan_scheme(scheme_length: int) -> None:
        scheme = "a" * scheme_length
        text = scheme + "\tvalue"
        config = URLConfig(allowed_schemes={scheme})

        result = asyncio.run(urls(ctx=None, data=text, config=config))

        assert result.tripwire_triggered is False  # noqa: S101
        assert result.info["detected"] == []  # noqa: S101

    small_duration = median(repeat(lambda: scan_scheme(4_000), number=1, repeat=5))
    large_duration = median(repeat(lambda: scan_scheme(8_000), number=1, repeat=5))

    assert large_duration < small_duration * 3  # noqa: S101


def test_urls_guardrail_scales_linearly_for_many_configured_schemes() -> None:
    """Public scanning avoids sorting the configured-scheme collection."""

    def build_config(scheme_count: int) -> URLConfig:
        schemes = {f"s{index:08x}" for index in range(scheme_count)}
        schemes.update(("data", "https"))
        return URLConfig(allowed_schemes=schemes)

    small_config = build_config(5_000)
    large_config = build_config(10_000)

    def scan_config(config: URLConfig) -> None:
        result = asyncio.run(urls(ctx=None, data="plain\ttext", config=config))

        assert result.tripwire_triggered is False  # noqa: S101
        assert result.info["detected"] == []  # noqa: S101

    small_duration = median(repeat(lambda: scan_config(small_config), number=1, repeat=5))
    large_duration = median(repeat(lambda: scan_config(large_config), number=1, repeat=5))

    assert large_duration < small_duration * 3  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_mailto_scheme_obfuscation(
    control: str,
) -> None:
    """Mailto obfuscation fails closed even when mailto is disallowed."""
    candidate = f"mail{control}to:user@trusted.example"
    config = URLConfig(
        url_allow_list=["trusted.example"],
        allowed_schemes={"https"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("hard_boundary", ['"', "<", ">"])
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_obfuscated_scheme_before_hard_boundary(
    control: str,
    hard_boundary: str,
) -> None:
    """A hard boundary cannot hide the body of an obfuscated scheme."""
    candidate = f"htt{control}p://{hard_boundary}@trusted.example/path"
    config = URLConfig(url_allow_list=["trusted.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("prefix", ["+", "-", ".", "a", "0"])
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_ambiguous_scheme_after_prefix(
    control: str,
    prefix: str,
) -> None:
    """Ordinary scheme boundaries must not hide an ambiguous URL."""
    candidate = f"https://trusted.co{control}m"
    config = URLConfig(url_allow_list=["trusted.co"], allowed_schemes={"https"})

    result = await urls(ctx=None, data=prefix + candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("scheme", "split_at", "payload"),
    [
        ("data", 2, "text/plain,hello"),
        ("javascript", 4, "alert(1)"),
        ("vbscript", 2, "msgbox(1)"),
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_control_obfuscated_special_scheme(
    control: str,
    scheme: str,
    split_at: int,
    payload: str,
) -> None:
    """Special schemes normalized by urllib.parse must fail closed."""
    candidate = f"{scheme[:split_at]}{control}{scheme[split_at:]}:{payload}"
    config = URLConfig(allowed_schemes={scheme})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("scheme", "split_at", "suffix"),
    [
        ("data", 2, ""),
        ("data", 2, ";"),
        ("javascript", 4, ""),
        ("javascript", 4, "!"),
        ("vbscript", 2, ""),
        ("vbscript", 2, "..."),
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_preserves_obfuscated_hostless_scheme_delimiter(
    control: str,
    scheme: str,
    split_at: int,
    suffix: str,
) -> None:
    """Candidate cleanup must preserve the structural scheme colon."""
    raw_scheme = f"{scheme[:split_at]}{control}{scheme[split_at:]}:"
    config = URLConfig(allowed_schemes={scheme})

    result = await urls(ctx=None, data=raw_scheme + suffix, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [raw_scheme]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [raw_scheme]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{raw_scheme}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@pytest.mark.parametrize(
    "text",
    [
        "javaſcript:\talert(1)",
        "javascrİpt:\nalert(1)",
        "vbſcript:\rmsgbox(1)",
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_ignores_non_ascii_scheme_confusables(text: str) -> None:
    """Unicode casefolding must not create a Python URL scheme."""
    result = await urls(ctx=None, data=text, config=URLConfig())

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == []  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    "outer_scheme",
    ["http://", "https://", "ftp://", "data:", "javascript:", "vbscript:"],
)
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_nested_scheme_after_control(
    control: str,
    outer_scheme: str,
) -> None:
    """A nested allowed URL must not hide a control-bearing outer scheme."""
    candidate = f"{outer_scheme}{control}https://trusted.example/path"
    config = URLConfig(url_allow_list=["trusted.example"], allowed_schemes={"https"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_nested_scheme_after_path_control(control: str) -> None:
    """Path text before a control must remain joined to a nested scheme."""
    outer_url = "http://outer.example/foo"
    inner_url = "https://inner.example"
    candidate = f"{outer_url}{control}{inner_url}"
    config = URLConfig(
        url_allow_list=[outer_url, inner_url],
        allowed_schemes={"http", "https"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    "candidate_template",
    [
        "http://[::1]/inter{control}nal",
        "htt{control}p://[::1]/internal",
        "http://{control}[::1]/internal",
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_control_bearing_ipv6_url(
    control: str,
    candidate_template: str,
) -> None:
    """Control-bearing IPv6 URLs must not evade scheme detection."""
    candidate = candidate_template.format(control=control)
    config = URLConfig(url_allow_list=["[::1]"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
def test_detect_urls_preserves_controls_in_scheme_separator(control: str) -> None:
    """Python-normalized scheme separators must remain raw."""
    candidate = f"http:{control}//2130706433:8000/internal/credentials"

    detected = _detect_urls(candidate)

    assert detected == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
def test_detect_urls_preserves_controls_in_canonical_url(control: str) -> None:
    """Controls inside a canonical URL must remain raw."""
    candidate = f"http://2130706433:8000/inter{control}nal/credentials"

    detected = _detect_urls(candidate)

    assert detected == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
def test_detect_urls_preserves_balanced_candidate_delimiters(control: str) -> None:
    """Balanced URL delimiters remain while sentence punctuation is removed."""
    candidate = f"http://allowed.example/pa{control}th_(x)"

    detected = _detect_urls(f"Visit ({candidate}).")

    assert detected == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("delimiter", ["[", "]", "\\", "<", ">", '"', "{", "}", "|", "^", "`"])
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_control_after_excluded_delimiter(
    control: str,
    delimiter: str,
) -> None:
    """An excluded delimiter must not hide a later URL control."""
    candidate = f"http://trusted.example/safe{delimiter}{control}admin"
    config = URLConfig(
        url_allow_list=["http://trusted.example/safe"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@pytest.mark.parametrize("whitespace", ["\u0085", "\u00a0", "\u2003", "\u2028", "\u2029"])
@pytest.mark.asyncio
async def test_urls_guardrail_stops_before_non_control_whitespace(whitespace: str) -> None:
    """Unrelated controls after Unicode whitespace remain outside the URL."""
    url = "http://allowed.example"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=f"{url}{whitespace}Notes\tvalue", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == [url]  # noqa: S101
    assert result.info["allowed"] == [url]  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize(
    ("whitespace", "userinfo"),
    [
        (" ", ""),
        ("\v", ":password"),
        ("\f", "user"),
        ("\u0085", "user:password"),
        ("\u00a0", "first last"),
        ("\u2003", "user"),
        ("\u2028", ""),
        ("\u2029", ":password"),
    ],
)
@pytest.mark.asyncio
async def test_urls_guardrail_bridges_whitespace_before_userinfo_host(
    control: str,
    whitespace: str,
    userinfo: str,
) -> None:
    """Whitespace before userinfo cannot hide a control-bearing host."""
    candidate = f"http://allowed.example{whitespace}{userinfo}@2130706433{control}/internal"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_bridges_nested_scheme_userinfo_host(control: str) -> None:
    """An outer path must not hide an inner scheme's userinfo host."""
    outer_url = "http://outer.example/path"
    inner_url = f"http://allowed.example user@2130706433{control}/internal"
    candidate = outer_url + inner_url
    config = URLConfig(
        url_allow_list=["outer.example", "allowed.example"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_uses_last_userinfo_host(control: str) -> None:
    """Repeated userinfo markers cannot hide the final parsed host."""
    candidate = " ".join(
        (
            "http://allowed.example first@intermediate.example",
            f"second@2130706433{control}/internal",
        )
    )
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
def test_detect_urls_preserves_controls_after_path_separator(control: str) -> None:
    """A standalone URL may continue after a controlled separator."""
    candidate = f"http://2130706433:8000/{control}internal/credentials"

    detected = _detect_urls(candidate)

    assert detected == [candidate]  # noqa: S101


def test_detect_urls_handles_long_control_run() -> None:
    """A long control run is consumed as one URL candidate token."""
    candidate = "http://allowed.example/" + "\n" * 30_000 + "internal"

    detected = _detect_urls(candidate)

    assert detected == [candidate]  # noqa: S101


def test_detect_urls_skips_suffix_after_unmatched_closing_boundary() -> None:
    """An unmatched closing boundary prevents scanning an irrelevant suffix."""
    candidate = "trusted.co\tm"
    text = candidate + ")" + "(" * 30_000

    detected = _detect_urls(text)

    assert detected == [candidate]  # noqa: S101


@settings(max_examples=75)
@given(case=_ambiguous_url_cases())
def test_urls_guardrail_blocks_generated_ambiguous_candidates(
    case: tuple[str, str, str],
) -> None:
    """Generated ambiguous URLs remain raw and fail closed."""
    scheme, candidate, text = case
    config = URLConfig(
        url_allow_list=["allowed.example", "[::1]"],
        allowed_schemes={scheme},
    )

    detected = _detect_urls(text)
    result = asyncio.run(urls(ctx=None, data=text, config=config))

    assert detected == [candidate]  # noqa: S101
    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [  # noqa: S101
        f"{candidate}: {AMBIGUOUS_URL_REASON}"
    ]


@settings(max_examples=100)
@given(case=_delimiter_heavy_ambiguous_url_cases())
def test_ambiguous_scanner_covers_generated_control_offset(
    case: tuple[str, int],
) -> None:
    """Delimiter combinations cannot hide an inserted URL control."""
    candidate, control_position = case

    detected = _find_ambiguous_url_candidates(candidate)

    inserted_control_is_covered = False
    for raw_candidate, (start, end) in detected:
        has_url_control = any(control in raw_candidate for control in "\t\n\r")
        if start <= control_position < end and has_url_control:
            inserted_control_is_covered = True
            break
    assert inserted_control_is_covered  # noqa: S101
    assert all(  # noqa: S101
        any(control in raw_candidate for control in "\t\n\r") for raw_candidate, _ in detected
    )


@settings(max_examples=100)
@given(candidate=_scheme_less_ambiguous_url_cases())
def test_ambiguous_scanner_preserves_generated_scheme_less_candidate(
    candidate: str,
) -> None:
    """Generated scheme-less candidates remain raw and ambiguous."""
    detected = _find_ambiguous_url_candidates(candidate)

    assert detected == [(candidate, (0, len(candidate)))]  # noqa: S101


@settings(max_examples=100)
@given(
    candidate=_scheme_less_ambiguous_url_cases(),
    destination=st.text(
        alphabet=string.ascii_letters + string.digits + "_-/",
        min_size=1,
        max_size=30,
    ),
    destination_control=st.sampled_from(["", "\t", "\n", "\r"]),
)
def test_ambiguous_scanner_trims_generated_markdown_boundary(
    candidate: str,
    destination: str,
    destination_control: str,
) -> None:
    """Generated scheme-less link labels exclude Markdown syntax."""
    text = f"[{candidate}]({destination}{destination_control}text)"

    detected = _find_ambiguous_url_candidates(text)

    assert detected[0] == (candidate, (1, len(candidate) + 1))  # noqa: S101


@settings(max_examples=100)
@given(
    candidate=_scheme_less_ambiguous_url_cases(),
    trailing_boundaries=st.text(
        alphabet='.,;:!?)]}>"`|^\\',
        min_size=1,
        max_size=16,
    ),
)
def test_ambiguous_scanner_trims_generated_trailing_boundaries(
    candidate: str,
    trailing_boundaries: str,
) -> None:
    """Mixed generated trailing boundaries remain outside the span."""
    text = candidate + trailing_boundaries

    detected = _find_ambiguous_url_candidates(text)

    assert detected == [(candidate, (0, len(candidate)))]  # noqa: S101


@settings(max_examples=100)
@given(
    prefix=st.sampled_from(["é", "١", "_", "é_name@"]),
    base_candidate=st.sampled_from(["trusted.com", "trusted.cK", "trusted.ſſ", "192.168.1.12"]),
    control=st.sampled_from(["\t", "\n", "\r"]),
    control_position=st.integers(min_value=1, max_value=8),
)
def test_ambiguous_scanner_handles_non_ascii_scheme_less_boundaries(
    prefix: str,
    base_candidate: str,
    control: str,
    control_position: int,
) -> None:
    """Non-ASCII boundaries cannot hide a control-bearing URL suffix."""
    insertion = min(control_position, len(base_candidate) - 1)
    controlled_candidate = base_candidate[:insertion] + control + base_candidate[insertion:]
    text = prefix + controlled_candidate

    detected = _find_ambiguous_url_candidates(text)

    candidate_is_covered = False
    for raw_candidate, (start, end) in detected:
        raw_span_is_preserved = raw_candidate == text[start:end]
        if raw_span_is_preserved and control in raw_candidate:
            candidate_is_covered = controlled_candidate in raw_candidate
            if candidate_is_covered:
                break
    assert candidate_is_covered  # noqa: S101


@settings(max_examples=12, deadline=timedelta(milliseconds=500))
@given(
    control=st.sampled_from(["\t", "\n", "\r"]),
    run_length=st.integers(min_value=10_000, max_value=50_000),
)
def test_detect_urls_handles_generated_long_control_runs(
    control: str,
    run_length: int,
) -> None:
    """Long generated control runs are detected within a bounded time."""
    candidate = "http://allowed.example/" + control * run_length + "internal"

    detected = _detect_urls(candidate)

    assert detected == [candidate]  # noqa: S101


@settings(max_examples=12, deadline=timedelta(milliseconds=500))
@given(
    control=st.sampled_from(["\t", "\n", "\r"]),
    run_length=st.integers(min_value=10_000, max_value=50_000),
)
def test_detect_urls_handles_long_control_runs_before_nested_scheme(
    control: str,
    run_length: int,
) -> None:
    """Nested schemes after long control runs are detected in bounded time."""
    candidate = "javascript:" + control * run_length + "https://trusted.example/path"

    detected = _detect_urls(candidate)

    assert detected == [candidate]  # noqa: S101


@settings(max_examples=12, deadline=timedelta(milliseconds=500))
@given(
    control=st.sampled_from(["\t", "\n", "\r"]),
    fragment_count=st.integers(min_value=1_000, max_value=10_000),
)
def test_ambiguous_scanner_skips_large_control_free_segments(
    control: str,
    fragment_count: int,
) -> None:
    """Unrelated controls do not cause repeated scans of prior segments."""
    control_free_segment = "http://allowed.example[" * fragment_count
    text = f"{control_free_segment} prose{control}value"

    candidates = _find_ambiguous_url_candidates(text)

    assert candidates == []  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(suffix_length=st.integers(min_value=10_000, max_value=50_000))
def test_ambiguous_scanner_skips_suffix_after_last_control(
    suffix_length: int,
) -> None:
    """A large control-free suffix is skipped after the last control."""
    text = "plain\tprose " + "x" * suffix_length

    candidates = _find_ambiguous_url_candidates(text)

    assert candidates == []  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(word_count=st.integers(min_value=1_000, max_value=10_000))
def test_ambiguous_scanner_handles_long_userinfo_in_linear_time(
    word_count: int,
) -> None:
    """Whitespace-bearing userinfo is scanned within a bounded time."""
    candidate = "http://allowed.example " + "user " * word_count + "name@2130706433\t/internal"

    detected = _find_ambiguous_url_candidates(candidate)

    assert detected == [(candidate, (0, len(candidate)))]  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(authority_count=st.integers(min_value=1_000, max_value=10_000))
def test_ambiguous_scanner_handles_repeated_authorities_in_linear_time(
    authority_count: int,
) -> None:
    """Repeated whitespace userinfo authorities remain linear."""
    candidate = "http://a user@" * authority_count + "2130706433\t/internal"

    detected = _find_ambiguous_url_candidates(candidate)

    assert detected == [(candidate, (0, len(candidate)))]  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(fragment_count=st.integers(min_value=1_000, max_value=10_000))
def test_ambiguous_scanner_handles_scheme_less_prose_in_linear_time(
    fragment_count: int,
) -> None:
    """Repeated scheme-less prose boundaries remain linear."""
    text = "trusted.co word " * fragment_count + "tail\tvalue"

    detected = _find_ambiguous_url_candidates(text)

    assert detected == []  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(candidate_count=st.integers(min_value=1_000, max_value=10_000))
def test_ambiguous_scanner_handles_many_scheme_less_candidates(
    candidate_count: int,
) -> None:
    """Many scheme-less candidates are collected within a bounded time."""
    candidate = "a.co\tm," * candidate_count + "z.co\tm"

    detected = _find_ambiguous_url_candidates(candidate)

    assert len(detected) == candidate_count + 1  # noqa: S101
    assert detected[0][0] == "a.co\tm"  # noqa: S101
    assert detected[-1][0] == "z.co\tm"  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(candidate_count=st.integers(min_value=1_000, max_value=10_000))
def test_ambiguous_scanner_handles_many_non_ascii_boundaries(
    candidate_count: int,
) -> None:
    """Unicode-prefixed scheme-less candidates remain linear."""
    candidate = "étrusted.co\tm " * candidate_count

    detected = _find_ambiguous_url_candidates(candidate)

    assert len(detected) == candidate_count  # noqa: S101
    assert all(raw_candidate == "trusted.co\tm" for raw_candidate, _ in detected)  # noqa: S101


@settings(max_examples=8, deadline=timedelta(milliseconds=500))
@given(candidate_count=st.integers(min_value=1_000, max_value=5_000))
def test_ambiguous_scanner_handles_many_markdown_boundaries(
    candidate_count: int,
) -> None:
    """Markdown cleanup with later controls remains linear."""
    text = "[trusted.co\tm](destination\ntext) " * candidate_count

    detected = _find_ambiguous_url_candidates(text)

    assert len(detected) == candidate_count  # noqa: S101
    assert all(raw_candidate == "trusted.co\tm" for raw_candidate, _ in detected)  # noqa: S101


def test_validate_url_security_blocks_bad_scheme() -> None:
    """Disallowed schemes should produce an error."""
    config = URLConfig()
    parsed, reason, _ = _validate_url_security("http://blocked.com", config)

    assert parsed is None  # noqa: S101
    assert "Blocked scheme" in reason  # noqa: S101


def test_validate_url_security_blocks_userinfo_when_configured() -> None:
    """URLs with embedded credentials should be rejected when block_userinfo=True."""
    config = URLConfig(allowed_schemes={"https"}, block_userinfo=True)
    parsed, reason, _ = _validate_url_security("https://user:pass@example.com", config)

    assert parsed is None  # noqa: S101
    assert "userinfo" in reason  # noqa: S101


def test_validate_url_security_blocks_password_without_username() -> None:
    """URLs that only include a password in userinfo must be blocked."""
    config = URLConfig(allowed_schemes={"https"}, block_userinfo=True)
    parsed, reason, _ = _validate_url_security("https://:secret@example.com", config)

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
    root_url, _, had_scheme1 = _validate_url_security("https://suntropy.es", config)
    path_url, _, had_scheme2 = _validate_url_security("https://api.example.com/v1/resources?id=2", config)
    wrong_path_url, _, had_scheme3 = _validate_url_security("https://api.example.com/v2", config)

    assert root_url is not None  # noqa: S101
    assert path_url is not None  # noqa: S101
    assert wrong_path_url is not None  # noqa: S101
    assert _is_url_allowed(root_url, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(path_url, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101
    assert _is_url_allowed(wrong_path_url, config.url_allow_list, config.allow_subdomains, had_scheme3) is False  # noqa: S101


def test_is_url_allowed_respects_path_segment_boundaries() -> None:
    """Path matching should respect segment boundaries to prevent security issues."""
    config = URLConfig(
        url_allow_list=["https://example.com/api"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # These should be allowed
    exact_match, _, had_scheme1 = _validate_url_security("https://example.com/api", config)
    valid_subpath, _, had_scheme2 = _validate_url_security("https://example.com/api/users", config)

    # These should NOT be allowed (different path segments)
    similar_path1, _, had_scheme3 = _validate_url_security("https://example.com/api2", config)
    similar_path2, _, had_scheme4 = _validate_url_security("https://example.com/api-v2", config)

    assert exact_match is not None  # noqa: S101
    assert valid_subpath is not None  # noqa: S101
    assert similar_path1 is not None  # noqa: S101
    assert similar_path2 is not None  # noqa: S101

    # Exact match and valid subpath should be allowed
    assert _is_url_allowed(exact_match, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(valid_subpath, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101

    # Similar paths that don't respect segment boundaries should be blocked
    assert _is_url_allowed(similar_path1, config.url_allow_list, config.allow_subdomains, had_scheme3) is False  # noqa: S101
    assert _is_url_allowed(similar_path2, config.url_allow_list, config.allow_subdomains, had_scheme4) is False  # noqa: S101


def test_is_url_allowed_without_scheme_matches_multiple_protocols() -> None:
    """Scheme-less allow list entries should match any allowed scheme."""
    config = URLConfig(
        url_allow_list=["example.com"],
        allow_subdomains=False,
        allowed_schemes={"https", "http"},
    )
    https_result, https_reason, had_scheme1 = _validate_url_security("https://example.com", config)
    http_result, http_reason, had_scheme2 = _validate_url_security("http://example.com", config)

    assert https_result is not None, https_reason  # noqa: S101
    assert http_result is not None, http_reason  # noqa: S101
    assert _is_url_allowed(https_result, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(http_result, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101


def test_is_url_allowed_supports_subdomains_and_cidr() -> None:
    """Allow list should support subdomains and CIDR ranges."""
    config = URLConfig(
        url_allow_list=["example.com", "10.0.0.0/8"],
        allow_subdomains=True,
    )
    https_result, _, had_scheme1 = _validate_url_security("https://api.example.com", config)
    ip_result, _, had_scheme2 = _validate_url_security("https://10.1.2.3", config)

    assert https_result is not None  # noqa: S101
    assert ip_result is not None  # noqa: S101
    assert _is_url_allowed(https_result, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(ip_result, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101


@pytest.mark.parametrize(
    ("url", "allowed_host", "expected"),
    [
        ("https://www.example.com", "example.com", True),
        ("https://example.com", "www.example.com", True),
        ("https://notwww.example.com", "notexample.com", False),
        ("https://sub.www.example.com", "sub.example.com", False),
    ],
)
def test_is_url_allowed_removes_only_a_leading_www_label(
    url: str,
    allowed_host: str,
    expected: bool,
) -> None:
    """Only the conventional leading www label is normalized."""
    config = URLConfig(url_allow_list=[allowed_host], allowed_schemes={"https"})
    parsed_url, _, had_scheme = _validate_url_security(url, config)

    assert parsed_url is not None  # noqa: S101
    assert _is_url_allowed(parsed_url, config.url_allow_list, False, had_scheme) is expected  # noqa: S101


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


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_raw_control_obfuscated_scheme(control: str) -> None:
    """Scheme controls interpreted by urllib.parse must fail closed."""
    candidate = f"htt{control}p://2130706433:8000/internal/credentials"
    config = URLConfig(url_allow_list=["2130706433"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_allowlisted_host_before_control_userinfo(control: str) -> None:
    """A safe-looking host prefix must not hide decimal loopback userinfo."""
    candidate = f"http://allowed.example{control}@2130706433:8000/internal/credentials"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_controls_in_canonical_url(control: str) -> None:
    """Canonical schemes do not make control-bearing URLs safe."""
    candidate = f"http://2130706433:8000/inter{control}nal/credentials"
    config = URLConfig(url_allow_list=["2130706433"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_controls_inside_allowlisted_path(control: str) -> None:
    """A safe path prefix must not hide a control-bearing continuation."""
    candidate = f"http://allowed.example/pa{control}th"
    config = URLConfig(
        url_allow_list=["http://allowed.example/pa"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.parametrize("continuation", ["th", "Th"])
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_embedded_controls_inside_allowlisted_path(
    control: str,
    continuation: str,
) -> None:
    """Surrounding prose must not hide a control-bearing URL path."""
    candidate = f"http://allowed.example/pa{control}{continuation}"
    config = URLConfig(
        url_allow_list=["http://allowed.example/pa"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=f"Use {candidate} now", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_embedded_controls_after_path_separator(control: str) -> None:
    """A line break after a slash may still continue the URL path."""
    candidate = f"http://allowed.example/{control}internal"
    config = URLConfig(
        url_allow_list=["http://allowed.example/"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=f"Use {candidate} now", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_controls_continuing_allowlisted_authority(
    control: str,
) -> None:
    """A safe host prefix must not hide an authority continuation."""
    candidate = f"http://allowed.example{control}.evil/path"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=f"Use {candidate} now", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_plain_standalone_authority_continuation(control: str) -> None:
    """A standalone host continuation cannot be treated as prose."""
    candidate = f"http://allowed.exa{control}mple"
    config = URLConfig(url_allow_list=["allowed.exa"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_classifies_normal_and_ambiguous_urls_independently() -> None:
    """Normal URLs stay allowed while raw ambiguous candidates are blocked."""
    ambiguous = "http://allowed.example\t@2130706433:8000/internal/credentials"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})
    text = f"Normal http://allowed.example/docs and ambiguous {ambiguous}"

    result = await urls(ctx=None, data=text, config=config)

    assert result.info["detected"] == [ambiguous, "http://allowed.example/docs"]  # noqa: S101
    assert result.info["allowed"] == ["http://allowed.example/docs"]  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_keeps_markdown_adjacent_url_independent() -> None:
    """A closing Markdown bracket remains a URL candidate boundary."""
    ambiguous = "http://allowed.example/pa\tth"
    normal = "http://allowed.example/docs"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=f"[{ambiguous}]({normal})", config=config)

    assert result.info["detected"] == [ambiguous, normal]  # noqa: S101
    assert result.info["allowed"] == [normal]  # noqa: S101
    assert result.info["blocked"] == [ambiguous]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_does_not_trust_unconfirmed_markdown_boundary(
    control: str,
) -> None:
    """A Markdown-like delimiter cannot hide a later URL control."""
    candidate = f"http://trusted.example/safe](junk{control}admin"
    config = URLConfig(
        url_allow_list=["http://trusted.example/safe"],
        allowed_schemes={"http"},
    )

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101
    assert result.info["blocked_reasons"] == [f"{candidate}: {AMBIGUOUS_URL_REASON}"]  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_allows_benign_multiline_prose_without_urls() -> None:
    """Control characters in prose alone must not trigger URL filtering."""
    result = await urls(
        ctx=None,
        data="First line\nSecond line\r\nThird\tcolumn",
        config=URLConfig(),
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected"] == []  # noqa: S101
    assert result.info["blocked"] == []  # noqa: S101


@pytest.mark.parametrize("control", ["\n", "\r", "\r\n"])
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_url_followed_by_prose_line(control: str) -> None:
    """A URL-like span crossing a line break must fail closed."""
    candidate = f"http://allowed.example/{control}Next"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=f"Visit {candidate} paragraph", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@ASCII_URL_CONTROLS
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_control_joined_urls(control: str) -> None:
    """A control followed by another scheme remains one raw candidate."""
    first_url = "http://allowed.example/"
    second_url = "http://allowed.example/docs"
    config = URLConfig(url_allow_list=["allowed.example"], allowed_schemes={"http"})
    candidate = f"{first_url}{control}{second_url}"

    result = await urls(ctx=None, data=candidate, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


@pytest.mark.parametrize("control", ["\n", "\r", "\r\n"])
@pytest.mark.asyncio
async def test_urls_guardrail_blocks_single_label_host_crossing_line_break(control: str) -> None:
    """A single-label host crossing a line break must fail closed."""
    candidate = f"http://intranet{control}next"
    config = URLConfig(url_allow_list=["intranet"], allowed_schemes={"http"})

    result = await urls(ctx=None, data=f"Visit {candidate} paragraph", config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["detected"] == [candidate]  # noqa: S101
    assert result.info["allowed"] == []  # noqa: S101
    assert result.info["blocked"] == [candidate]  # noqa: S101


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
    parsed, reason, _ = _validate_url_security("https://", config)

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
    ip_in_range1, _, had_scheme1 = _validate_url_security("https://10.5.5.5", config)
    ip_in_range2, _, had_scheme2 = _validate_url_security("https://192.168.1.100", config)
    # IP outside CIDR range
    ip_outside, _, had_scheme3 = _validate_url_security("https://192.168.2.1", config)

    assert ip_in_range1 is not None  # noqa: S101
    assert ip_in_range2 is not None  # noqa: S101
    assert ip_outside is not None  # noqa: S101

    assert _is_url_allowed(ip_in_range1, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(ip_in_range2, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101
    assert _is_url_allowed(ip_outside, config.url_allow_list, config.allow_subdomains, had_scheme3) is False  # noqa: S101


def test_is_url_allowed_handles_port_matching() -> None:
    """Port matching: enforced if allow list has explicit port, otherwise any port allowed."""
    config = URLConfig(
        url_allow_list=["https://example.com:8443", "api.internal.com"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # Explicit port 8443 matches allow list's explicit port → ALLOWED
    correct_port, _, had_scheme1 = _validate_url_security("https://example.com:8443", config)
    # Implicit 443 doesn't match allow list's explicit 8443 → BLOCKED
    wrong_port, _, had_scheme2 = _validate_url_security("https://example.com", config)
    # Explicit 9000 with no port restriction in allow list → ALLOWED
    explicit_port_no_restriction, _, had_scheme3 = _validate_url_security("https://api.internal.com:9000", config)
    # Implicit 443 with no port restriction in allow list → ALLOWED
    implicit_match, _, had_scheme4 = _validate_url_security("https://api.internal.com", config)
    # Explicit default 443 with no port restriction in allow list → ALLOWED (regression fix)
    explicit_default_port, _, had_scheme5 = _validate_url_security("https://api.internal.com:443", config)

    assert correct_port is not None  # noqa: S101
    assert wrong_port is not None  # noqa: S101
    assert explicit_port_no_restriction is not None  # noqa: S101
    assert implicit_match is not None  # noqa: S101
    assert explicit_default_port is not None  # noqa: S101

    assert _is_url_allowed(correct_port, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(wrong_port, config.url_allow_list, config.allow_subdomains, had_scheme2) is False  # noqa: S101
    assert _is_url_allowed(explicit_port_no_restriction, config.url_allow_list, config.allow_subdomains, had_scheme3) is True  # noqa: S101
    assert _is_url_allowed(implicit_match, config.url_allow_list, config.allow_subdomains, had_scheme4) is True  # noqa: S101
    assert _is_url_allowed(explicit_default_port, config.url_allow_list, config.allow_subdomains, had_scheme5) is True  # noqa: S101


def test_is_url_allowed_handles_query_and_fragment() -> None:
    """Allow list entries with query/fragment should match exactly."""
    config = URLConfig(
        url_allow_list=["https://example.com/search?q=test", "https://example.com/docs#intro"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # Exact query match
    exact_query, _, had_scheme1 = _validate_url_security("https://example.com/search?q=test", config)
    # Different query
    diff_query, _, had_scheme2 = _validate_url_security("https://example.com/search?q=other", config)
    # Exact fragment match
    exact_fragment, _, had_scheme3 = _validate_url_security("https://example.com/docs#intro", config)
    # Different fragment
    diff_fragment, _, had_scheme4 = _validate_url_security("https://example.com/docs#outro", config)

    assert exact_query is not None  # noqa: S101
    assert diff_query is not None  # noqa: S101
    assert exact_fragment is not None  # noqa: S101
    assert diff_fragment is not None  # noqa: S101

    assert _is_url_allowed(exact_query, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(diff_query, config.url_allow_list, config.allow_subdomains, had_scheme2) is False  # noqa: S101
    assert _is_url_allowed(exact_fragment, config.url_allow_list, config.allow_subdomains, had_scheme3) is True  # noqa: S101
    assert _is_url_allowed(diff_fragment, config.url_allow_list, config.allow_subdomains, had_scheme4) is False  # noqa: S101


def test_validate_url_security_allows_userinfo_when_disabled() -> None:
    """URLs with userinfo should be allowed when block_userinfo=False."""
    config = URLConfig(allowed_schemes={"https"}, block_userinfo=False)
    parsed, reason, _ = _validate_url_security("https://user:pass@example.com", config)

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
    https_url, _, had_scheme1 = _validate_url_security("https://bank.example.com", config)
    # HTTP should be BLOCKED (doesn't match the explicit https:// in allow list)
    http_url, _, had_scheme2 = _validate_url_security("http://bank.example.com", config)

    assert https_url is not None  # noqa: S101
    assert http_url is not None  # noqa: S101

    # This is the security-critical check: scheme-qualified entries must match exactly
    assert _is_url_allowed(https_url, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(http_url, config.url_allow_list, config.allow_subdomains, had_scheme2) is False  # noqa: S101


def test_is_url_allowed_enforces_scheme_for_ips() -> None:
    """Scheme-qualified IP addresses in allow list must match scheme exactly."""
    config = URLConfig(
        url_allow_list=["https://192.168.1.100"],
        allow_subdomains=False,
        allowed_schemes={"https", "http"},
    )
    # HTTPS should be allowed
    https_ip, _, had_scheme1 = _validate_url_security("https://192.168.1.100", config)
    # HTTP should be BLOCKED
    http_ip, _, had_scheme2 = _validate_url_security("http://192.168.1.100", config)

    assert https_ip is not None  # noqa: S101
    assert http_ip is not None  # noqa: S101

    assert _is_url_allowed(https_ip, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(http_ip, config.url_allow_list, config.allow_subdomains, had_scheme2) is False  # noqa: S101


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


@pytest.mark.asyncio
async def test_urls_guardrail_rejects_scheme_shaped_malformed_port() -> None:
    """An invalid authority cannot become two independently allowed URLs."""
    text = "https://example.com:https://good.example/path"
    config = URLConfig(
        url_allow_list=["example.com", "good.example"],
        allowed_schemes={"https"},
    )

    result = await urls(ctx=None, data=text, config=config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["blocked"] == [text]  # noqa: S101


def test_is_url_allowed_handles_trailing_slash_in_path() -> None:
    """Allow list entries with trailing slashes should match subpaths correctly."""
    config = URLConfig(
        url_allow_list=["https://example.com/api/"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # URL with subpath should be allowed
    subpath_url, _, had_scheme1 = _validate_url_security("https://example.com/api/users", config)
    # Exact match (with trailing slash) should be allowed
    exact_url, _, had_scheme2 = _validate_url_security("https://example.com/api/", config)

    assert subpath_url is not None  # noqa: S101
    assert exact_url is not None  # noqa: S101

    # Both should be allowed
    assert _is_url_allowed(subpath_url, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(exact_url, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_scheme_matching_with_qualified_allow_list() -> None:
    """Test exact behavior: scheme-qualified allow list vs scheme-less/explicit URLs."""
    config = URLConfig(
        url_allow_list=["https://suntropy.es"],
        allowed_schemes={"https"},
        allow_subdomains=False,
    )

    # Test schemeless URL
    result1 = await urls(ctx=None, data="Visit suntropy.es", config=config)
    assert "suntropy.es" in result1.info["allowed"]  # noqa: S101
    assert result1.tripwire_triggered is False  # noqa: S101

    # Test HTTPS URL (should match allow list scheme)
    result2 = await urls(ctx=None, data="Visit https://suntropy.es", config=config)
    assert "https://suntropy.es" in result2.info["allowed"]  # noqa: S101
    assert result2.tripwire_triggered is False  # noqa: S101

    # Test HTTP URL (wrong explicit scheme should be blocked)
    result3 = await urls(ctx=None, data="Visit http://suntropy.es", config=config)
    assert "http://suntropy.es" in result3.info["blocked"]  # noqa: S101
    assert result3.tripwire_triggered is True  # noqa: S101


def test_is_url_allowed_handles_ipv6_addresses() -> None:
    """IPv6 addresses should be handled correctly (colons are not ports)."""
    config = URLConfig(
        url_allow_list=["[2001:db8::1]", "ftp://[2001:db8::2]"],
        allow_subdomains=False,
        allowed_schemes={"https", "ftp"},
    )
    # IPv6 without scheme
    ipv6_no_scheme, _, had_scheme1 = _validate_url_security("[2001:db8::1]", config)
    # IPv6 with ftp scheme
    ipv6_with_ftp, _, had_scheme2 = _validate_url_security("ftp://[2001:db8::2]", config)

    assert ipv6_no_scheme is not None  # noqa: S101
    assert ipv6_with_ftp is not None  # noqa: S101

    # Both should be allowed
    assert _is_url_allowed(ipv6_no_scheme, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(ipv6_with_ftp, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101


def test_is_url_allowed_handles_ipv6_cidr_notation() -> None:
    """IPv6 CIDR blocks should be handled correctly (brackets stripped, path concatenated)."""
    config = URLConfig(
        url_allow_list=["[2001:db8::]/64", "[fe80::]/10"],
        allow_subdomains=False,
        allowed_schemes={"https"},
    )
    # IP within first CIDR range
    ip_in_range1, _, had_scheme1 = _validate_url_security("https://[2001:db8::1234]", config)
    # IP within second CIDR range
    ip_in_range2, _, had_scheme2 = _validate_url_security("https://[fe80::5678]", config)
    # IP outside CIDR ranges
    ip_outside, _, had_scheme3 = _validate_url_security("https://[2001:db9::1]", config)

    assert ip_in_range1 is not None  # noqa: S101
    assert ip_in_range2 is not None  # noqa: S101
    assert ip_outside is not None  # noqa: S101

    # IPs within CIDR ranges should be allowed
    assert _is_url_allowed(ip_in_range1, config.url_allow_list, config.allow_subdomains, had_scheme1) is True  # noqa: S101
    assert _is_url_allowed(ip_in_range2, config.url_allow_list, config.allow_subdomains, had_scheme2) is True  # noqa: S101
    # IP outside should be blocked
    assert _is_url_allowed(ip_outside, config.url_allow_list, config.allow_subdomains, had_scheme3) is False  # noqa: S101


@pytest.mark.asyncio
async def test_urls_guardrail_blocks_subdomains_and_paths_correctly() -> None:
    """Verify subdomains and paths are still blocked according to allow list rules."""
    config = URLConfig(
        url_allow_list=["https://suntropy.es"],
        allowed_schemes={"https"},
        allow_subdomains=False,
    )
    # Test blocked cases - different domains and subdomains
    text = "Visit help-suntropy.es and help.suntropy.es"

    result = await urls(ctx=None, data=text, config=config)

    # Both should be blocked - not in allow list
    assert result.tripwire_triggered is True  # noqa: S101
    assert len(result.info["blocked"]) == 2  # noqa: S101
    assert "help-suntropy.es" in result.info["blocked"]  # noqa: S101
    assert "help.suntropy.es" in result.info["blocked"]  # noqa: S101
