"""URL detection guardrail.

This guardrail detects URLs in text and validates them against an allow list of
permitted domains, IP addresses, and full URLs. It provides security features
to prevent credential injection, typosquatting attacks, and unauthorized schemes.

The guardrail uses regex patterns for URL detection and Pydantic for robust
URL parsing and validation.

Example Usage:
    Default configuration:
        config = URLConfig(url_allow_list=["example.com"])

    Custom configuration:
        config = URLConfig(
            url_allow_list=["company.com", "10.0.0.0/8"],
            allowed_schemes={"http", "https"},
            allow_subdomains=True
        )
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from heapq import merge
from ipaddress import AddressValueError, ip_address, ip_network
from itertools import chain
from typing import Any
from urllib.parse import ParseResult, urlparse

from pydantic import BaseModel, Field, field_validator

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["urls"]

DEFAULT_PORTS = {
    "http": 80,
    "https": 443,
}

SCHEME_PREFIX_RE = re.compile(r"^[a-z][a-z0-9+.-]*://", re.IGNORECASE)
_DOMAIN_HOST_CANDIDATE_PATTERN = r"[a-zA-Z0-9][a-zA-Z0-9.-]*"
DOMAIN_HOST_CANDIDATE_RE = re.compile(rf"\b{_DOMAIN_HOST_CANDIDATE_PATTERN}", re.IGNORECASE)
_AMBIGUOUS_DOMAIN_PATTERN = rf"(?<![A-Za-z0-9])(?i:{_DOMAIN_HOST_CANDIDATE_PATTERN})"
AMBIGUOUS_DOMAIN_HOST_CANDIDATE_RE = re.compile(_AMBIGUOUS_DOMAIN_PATTERN)
_IP_HOST_PATTERN = r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?::[0-9]+)?"
_IP_URL_PATTERN = rf"{_IP_HOST_PATTERN}(?:/[^\s]*)?"
IP_URL_RE = re.compile(rf"\b{_IP_URL_PATTERN}")
_IP_HOST_RE = re.compile(rf"\b{_IP_HOST_PATTERN}")
AMBIGUOUS_IP_URL_RE = re.compile(rf"(?<![A-Za-z0-9]){_IP_URL_PATTERN}")
ASCII_LETTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
CASE_INSENSITIVE_ASCII_LETTERS = ASCII_LETTERS | frozenset("İıſK")
ASCII_URL_CONTROL_CHARACTERS = "\t\n\r"
AMBIGUOUS_URL_REASON = "Ambiguous URL containing ASCII control characters"

_HIERARCHICAL_CONTROL_SENSITIVE_SCHEMES = frozenset(("http", "https", "ftp"))
_HOSTLESS_CONTROL_SENSITIVE_SCHEMES = frozenset(("data", "javascript", "vbscript", "mailto"))
_ASCII_URL_CONTROL_SET = frozenset(ASCII_URL_CONTROL_CHARACTERS)
_ASCII_URL_CONTROL_RE = re.compile(r"[\t\n\r]")
_ASCII_URL_CONTROL_TRANSLATION = str.maketrans("", "", ASCII_URL_CONTROL_CHARACTERS)
_VALID_SCHEME_RE = re.compile(r"[a-z][a-z0-9+.-]*", re.ASCII | re.IGNORECASE)
_ASCII_SCHEME_CHARACTER_SET = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+.-")
_TRAILING_SENTENCE_PUNCTUATION = frozenset(".,;:!?")
_POST_CONTROL_HARD_URL_BOUNDARIES = frozenset(('"', "`", "|", "^", "\\", "[", "{", "<"))
_POST_CONTROL_HARD_URL_BOUNDARY_RE = re.compile(r'["`|^\\\[{<]')
_PAIRED_DELIMITER_OPEN_CODES = {"(": 1, "[": 2, "{": 3, "<": 4}
_PAIRED_DELIMITER_CLOSE_TO_OPEN_CODE = {")": 1, "]": 2, "}": 3, ">": 4}
_PRESENTATION_CLOSERS = {"(": ")", "[": "]", "{": "}", "<": ">", "'": "'", '"': '"', "`": "`"}
_PAIRED_DELIMITER_RE = re.compile(r"[()\[\]{}<>]")
_ADJACENT_SCHEME_URL_BOUNDARIES = (
    _TRAILING_SENTENCE_PUNCTUATION
    | frozenset(_PAIRED_DELIMITER_OPEN_CODES)
    | frozenset(_PAIRED_DELIMITER_CLOSE_TO_OPEN_CODE)
    | frozenset(("'", '"', "`"))
)
_ADJACENT_SCHEME_URL_BOUNDARY_CLASS = re.escape("".join(sorted(_ADJACENT_SCHEME_URL_BOUNDARIES)))
_HIERARCHICAL_ADJACENT_SCHEME_URL_BOUNDARIES = _ADJACENT_SCHEME_URL_BOUNDARIES - frozenset(":")
_HIERARCHICAL_ADJACENT_SCHEME_URL_BOUNDARY_CLASS = re.escape("".join(sorted(_HIERARCHICAL_ADJACENT_SCHEME_URL_BOUNDARIES)))
_ADJACENT_DOMAIN_START_RE = re.compile(
    rf"(?<=[{_ADJACENT_SCHEME_URL_BOUNDARY_CLASS}]){_DOMAIN_HOST_CANDIDATE_PATTERN}",
    re.IGNORECASE,
)
_ADJACENT_IP_START_RE = re.compile(rf"(?<=[{_ADJACENT_SCHEME_URL_BOUNDARY_CLASS}])(?:[0-9]{{1,3}}\.){{3}}[0-9]{{1,3}}")
_EXPLICIT_URL_SCHEME_PATTERN = r"(?:https?|ftp)://|(?:data|javascript|vbscript):"
_EXPLICIT_URL_SCHEME_RE = re.compile(_EXPLICIT_URL_SCHEME_PATTERN, re.IGNORECASE)
_HTTP_URL_CHARACTER_PATTERN = r'[^\s<>"{}|\\^`\[\]]'
_BRACKETED_HTTP_AUTHORITY_PATTERN = r"https?://\[[^\]\s/?#]+\]"
_HIERARCHICAL_URL_COMPONENT_PATTERN = (
    r"(?:(?!["
    + _HIERARCHICAL_ADJACENT_SCHEME_URL_BOUNDARY_CLASS
    + r"](?:"
    + _EXPLICIT_URL_SCHEME_PATTERN
    + r"))(?:"
    + _BRACKETED_HTTP_AUTHORITY_PATTERN
    + "|"
    + _HTTP_URL_CHARACTER_PATTERN
    + r"))"
)
_HTTP_URL_COMPONENT_PATTERN = _HIERARCHICAL_URL_COMPONENT_PATTERN
_HTTP_URL_PATTERN = r"https?://" + _HTTP_URL_COMPONENT_PATTERN + "+"
_FTP_URL_PATTERN = r"ftp://" + _HIERARCHICAL_URL_COMPONENT_PATTERN + "+"
_BRACKETED_HTTP_URL_PATTERN = _BRACKETED_HTTP_AUTHORITY_PATTERN + _HTTP_URL_COMPONENT_PATTERN + "*"
_BRACKETED_HTTP_AUTHORITY_RE = re.compile(_BRACKETED_HTTP_AUTHORITY_PATTERN, re.IGNORECASE)
_TRAILING_DETECTED_URL_PUNCTUATION_RE = re.compile(r"[.,;:!?)\]]+$")
_DENSE_DELIMITER_SAMPLE_SIZE = 64
_DENSE_DELIMITER_SAMPLE_WINDOW = 4_096
_POST_CONTROL_UNMATCHED_URL_BOUNDARIES = frozenset(")]}>")
_TRAILING_URL_BOUNDARIES = _TRAILING_SENTENCE_PUNCTUATION | _POST_CONTROL_HARD_URL_BOUNDARIES


@dataclass(frozen=True, slots=True)
class _AmbiguousSchemeMatcher:
    """Reversed trie for linear control-bearing scheme detection.

    Args:
        transitions: Character transitions indexed by trie node.
        hierarchical_nodes: Terminal nodes that accept a ``://`` suffix.
        hostless_nodes: Terminal nodes that accept a single colon suffix.
        hostless_first_nodes: Built-in hostless schemes whose colon form has
            precedence over a configured hierarchical form.
    """

    transitions: tuple[dict[str, int], ...]
    hierarchical_nodes: frozenset[int]
    hostless_nodes: frozenset[int]
    hostless_first_nodes: frozenset[int]


@dataclass(frozen=True, slots=True)
class _AmbiguousSchemeMatch:
    """One raw scheme-prefix match in source text.

    Args:
        start: Inclusive raw offset of the scheme name.
        end: Exclusive raw offset of the matched scheme prefix.
        opens_authority: Whether the prefix ends in ``://``.
    """

    start: int
    end: int
    opens_authority: bool


def _build_ambiguous_scheme_matcher(
    allowed_schemes: set[str] | frozenset[str],
) -> _AmbiguousSchemeMatcher:
    """Build a reversed trie for built-in and configured schemes.

    Args:
        allowed_schemes: Normalized schemes allowed by the URL filter.

    Returns:
        An immutable matcher description for raw scheme scanning.
    """
    hierarchical_schemes: set[str] = set(_HIERARCHICAL_CONTROL_SENSITIVE_SCHEMES)
    hostless_schemes: set[str] = set(_HOSTLESS_CONTROL_SENSITIVE_SCHEMES)
    configured_schemes = {scheme.lower() for scheme in allowed_schemes if _VALID_SCHEME_RE.fullmatch(scheme) is not None}
    hierarchical_schemes.update(configured_schemes)
    hostless_schemes.update(configured_schemes)

    transitions: list[dict[str, int]] = [{}]
    hierarchical_nodes: set[int] = set()
    hostless_nodes: set[int] = set()
    hostless_first_nodes: set[int] = set()
    all_schemes = hierarchical_schemes | hostless_schemes
    for scheme in all_schemes:
        node = 0
        for character in reversed(scheme):
            next_node = transitions[node].get(character)
            if next_node is None:
                next_node = len(transitions)
                transitions[node][character] = next_node
                transitions.append({})
            node = next_node
        if scheme in hierarchical_schemes:
            hierarchical_nodes.add(node)
        if scheme in hostless_schemes:
            hostless_nodes.add(node)
        if scheme in _HOSTLESS_CONTROL_SENSITIVE_SCHEMES:
            hostless_first_nodes.add(node)

    return _AmbiguousSchemeMatcher(
        transitions=tuple(transitions),
        hierarchical_nodes=frozenset(hierarchical_nodes),
        hostless_nodes=frozenset(hostless_nodes),
        hostless_first_nodes=frozenset(hostless_first_nodes),
    )


def _find_hierarchical_scheme_end(text: str, colon_position: int) -> int | None:
    """Find a control-interleaved ``//`` suffix after a scheme colon.

    Args:
        text: Source text containing the scheme colon.
        colon_position: Offset of the scheme colon.

    Returns:
        The exclusive offset after the second slash, or None when absent.
    """
    position = colon_position + 1
    for _ in range(2):
        while position < len(text) and text[position] in _ASCII_URL_CONTROL_SET:
            position += 1
        if position == len(text) or text[position] != "/":
            return None
        position += 1
    return position


def _find_ambiguous_scheme_matches(
    text: str,
    matcher: _AmbiguousSchemeMatcher,
) -> list[_AmbiguousSchemeMatch]:
    """Find scheme prefixes without repeated regex rescans.

    Each colon terminates the backward scan for the preceding colon, so the
    total amount of backward work is linear in the input length.

    Args:
        text: Source text to scan for scheme prefixes.
        matcher: Reversed scheme trie and terminal metadata.

    Returns:
        Scheme-prefix matches ordered by source position.
    """
    matches: list[_AmbiguousSchemeMatch] = []
    for colon_position, character in enumerate(text):
        if character != ":":
            continue

        hierarchical_end = _find_hierarchical_scheme_end(text, colon_position)
        node = 0
        position = colon_position - 1
        selected_match: _AmbiguousSchemeMatch | None = None
        while position >= 0:
            scheme_character = text[position]
            if scheme_character in _ASCII_URL_CONTROL_SET:
                position -= 1
                continue
            if scheme_character not in _ASCII_SCHEME_CHARACTER_SET:
                break

            next_node = matcher.transitions[node].get(scheme_character.lower())
            if next_node is None:
                break
            node = next_node

            if node in matcher.hostless_first_nodes:
                match_end = colon_position + 1
                opens_authority = False
            elif hierarchical_end is not None and node in matcher.hierarchical_nodes:
                match_end = hierarchical_end
                opens_authority = True
            elif node in matcher.hostless_nodes:
                match_end = colon_position + 1
                opens_authority = False
            else:
                position -= 1
                continue

            selected_match = _AmbiguousSchemeMatch(
                start=position,
                end=match_end,
                opens_authority=opens_authority,
            )
            position -= 1

        if selected_match is not None:
            matches.append(selected_match)

    return matches


@dataclass(frozen=True, slots=True)
class UrlDetectionResult:
    """Result structure for URL detection and filtering."""

    detected: list[str]
    allowed: list[str]
    blocked: list[str]
    blocked_reasons: list[str] = field(default_factory=list)


class URLConfig(BaseModel):
    """Direct URL configuration with explicit parameters."""

    url_allow_list: list[str] = Field(
        default_factory=list,
        description="Allowed URLs, domains, or IP addresses",
    )
    allowed_schemes: set[str] = Field(
        default={"https"},
        description="Allowed URL schemes/protocols (default: HTTPS only for security)",
    )
    block_userinfo: bool = Field(
        default=True,
        description="Block URLs with userinfo (user:pass@domain) to prevent credential injection",
    )
    allow_subdomains: bool = Field(
        default=False,
        description="Allow subdomains of allowed domains (e.g. api.example.com if example.com is allowed)",
    )

    @field_validator("allowed_schemes", mode="before")
    @classmethod
    def normalize_allowed_schemes(cls, value: Any) -> set[str]:
        """Normalize allowed schemes to bare identifiers without delimiters."""
        if value is None:
            return {"https"}

        if isinstance(value, str):
            raw_values = [value]
        else:
            raw_values = list(value)

        normalized: set[str] = set()
        for entry in raw_values:
            if not isinstance(entry, str):
                raise TypeError("allowed_schemes entries must be strings")
            cleaned = entry.strip().lower()
            if not cleaned:
                continue
            # Support inputs like "https://", "HTTPS:", or " https "
            if cleaned.endswith("://"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.removesuffix(":")
            if cleaned:
                normalized.add(cleaned)

        if not normalized:
            raise ValueError("allowed_schemes must include at least one scheme")

        return normalized


def _find_domain_end(host_candidate: str) -> int | None:
    """Find the end of the last valid domain suffix in a host candidate.

    Args:
        host_candidate: Maximal ASCII hostname-shaped text to inspect.

    Returns:
        The exclusive end offset of the last dot followed by at least two
        ASCII letters, or ``None`` when no valid suffix exists.
    """
    domain_end = None

    for index, character in enumerate(host_candidate):
        if character != ".":
            continue

        suffix_end = index + 1
        while suffix_end < len(host_candidate):
            if host_candidate[suffix_end] not in CASE_INSENSITIVE_ASCII_LETTERS:
                break
            suffix_end += 1

        if suffix_end - index > 2:
            domain_end = suffix_end

    return domain_end


def _detect_domain_like_url_spans(
    text: str,
    candidate_re: re.Pattern[str] = DOMAIN_HOST_CANDIDATE_RE,
) -> list[tuple[int, int]]:
    """Detect scheme-less domain-like URL spans in linear time.

    Args:
        text: Text to scan for domain-like URLs.
        candidate_re: Pattern locating maximal hostname-shaped text.

    Returns:
        Inclusive start and exclusive end offsets for domain-like URLs.
    """
    detected_spans: list[tuple[int, int]] = []
    search_position = 0

    while match := candidate_re.search(text, search_position):
        host_candidate = match.group(0)
        domain_end = _find_domain_end(host_candidate)
        if domain_end is None:
            search_position = match.end()
            continue

        match_end = match.start() + domain_end
        if domain_end == len(host_candidate) and match_end < len(text) and text[match_end] == "/":
            match_end = _find_scheme_less_path_end(text, match_end)

        detected_spans.append((match.start(), match_end))
        search_position = match_end

    return detected_spans


def _detect_ip_url_spans(text: str) -> list[tuple[int, int]]:
    """Detect IPv4 URL spans with adjacent-URL boundaries.

    Args:
        text: Text to scan for IPv4 URLs.

    Returns:
        Inclusive start and exclusive end offsets for IPv4 URLs.
    """
    detected_spans: list[tuple[int, int]] = []
    search_position = 0

    while match := _IP_HOST_RE.search(text, search_position):
        match_end = match.end()
        if match_end < len(text) and text[match_end] == "/":
            match_end = _find_scheme_less_path_end(text, match_end)
        detected_spans.append((match.start(), match_end))
        search_position = match_end

    return detected_spans


def _detect_domain_like_urls(text: str) -> list[str]:
    """Detect scheme-less domain-like URLs in linear time.

    Args:
        text: Text to scan for domain-like URLs.

    Returns:
        Domain-like URL matches using the existing detection semantics.
    """
    return [text[start:end] for start, end in _detect_domain_like_url_spans(text)]


def _use_dense_delimiter_scan(text: str) -> bool:
    """Choose direct scanning when delimiters are densely clustered.

    Args:
        text: Candidate text whose delimiter density should be sampled.

    Returns:
        True when the initial delimiter sample fits within a small window.
    """
    first_match = _PAIRED_DELIMITER_RE.search(text)
    if first_match is None:
        return False

    search_position = first_match.end()
    for _ in range(_DENSE_DELIMITER_SAMPLE_SIZE):
        match = _PAIRED_DELIMITER_RE.search(text, search_position)
        if match is None:
            return False
        if match.start() - first_match.start() > _DENSE_DELIMITER_SAMPLE_WINDOW:
            return False
        search_position = match.end()

    return True


def _paired_delimiter_positions(text: str) -> Iterator[tuple[int, str]]:
    """Iterate delimiter positions with a density-appropriate scan.

    Args:
        text: Candidate text whose delimiters should be located.

    Returns:
        An iterator over source offsets and delimiter characters.
    """
    if _use_dense_delimiter_scan(text):
        return enumerate(text)
    return ((match.start(), match.group(0)) for match in _PAIRED_DELIMITER_RE.finditer(text))


def _mark_unmatched_closing_delimiters(text: str) -> bytearray | None:
    """Mark closing delimiters without properly nested openers.

    Args:
        text: Candidate text whose paired delimiters should be inspected.

    Returns:
        A one-byte-per-character mask marking unmatched closing delimiters,
        or None when every closing delimiter has a matching opener.
    """
    opening_delimiters = bytearray()
    unmatched_positions: bytearray | None = None

    for position, character in _paired_delimiter_positions(text):
        opening_code = _PAIRED_DELIMITER_OPEN_CODES.get(character)
        if opening_code is not None:
            opening_delimiters.append(opening_code)
            continue

        expected_opening_code = _PAIRED_DELIMITER_CLOSE_TO_OPEN_CODE.get(character)
        if expected_opening_code is None:
            continue
        if opening_delimiters and opening_delimiters[-1] == expected_opening_code:
            opening_delimiters.pop()
        else:
            if unmatched_positions is None:
                unmatched_positions = bytearray(len(text))
            unmatched_positions[position] = 1

    return unmatched_positions


def _truncate_at_post_control_hard_boundary(
    candidate: str,
    minimum_length: int,
    control_is_in_protected_prefix: bool,
) -> str:
    """Truncate an unprotected candidate at its first hard boundary.

    Args:
        candidate: Raw URL-like candidate to inspect.
        minimum_length: Prefix length that must remain intact.
        control_is_in_protected_prefix: Whether URL syntax after a controlled
            scheme prefix must remain part of the candidate.

    Returns:
        The candidate prefix ending before the first applicable hard boundary.
    """
    if control_is_in_protected_prefix:
        return candidate
    hard_boundary_match = _POST_CONTROL_HARD_URL_BOUNDARY_RE.search(
        candidate,
        minimum_length,
    )
    if hard_boundary_match is None:
        return candidate
    return candidate[: hard_boundary_match.start()]


def _truncate_at_unmatched_closing_boundary(
    candidate: str,
    minimum_length: int,
    control_is_in_protected_prefix: bool,
) -> str:
    """Truncate an unprotected candidate at an unmatched closing delimiter.

    Args:
        candidate: Raw URL-like candidate to inspect.
        minimum_length: Prefix length that must remain intact.
        control_is_in_protected_prefix: Whether URL syntax after a controlled
            scheme prefix must remain part of the candidate.

    Returns:
        The candidate prefix ending before the first applicable unmatched
        closing delimiter.
    """
    if control_is_in_protected_prefix:
        return candidate

    opening_delimiters = bytearray()
    for position, character in _paired_delimiter_positions(candidate):
        opening_code = _PAIRED_DELIMITER_OPEN_CODES.get(character)
        if opening_code is not None:
            opening_delimiters.append(opening_code)
            continue

        expected_opening_code = _PAIRED_DELIMITER_CLOSE_TO_OPEN_CODE.get(character)
        if expected_opening_code is None:
            continue
        if opening_delimiters and opening_delimiters[-1] == expected_opening_code:
            opening_delimiters.pop()
        elif position >= minimum_length:
            return candidate[:position]

    return candidate


def _clean_ambiguous_url_candidate(candidate: str, minimum_length: int) -> str:
    """Remove Markdown boundaries and trailing sentence punctuation.

    Args:
        candidate: Raw URL-like candidate to clean.
        minimum_length: Prefix length that must remain intact.

    Returns:
        The candidate without confirmed Markdown syntax, trailing sentence
        punctuation, or unmatched trailing delimiters. The final ASCII URL
        control is always preserved.
    """
    url_controls = ASCII_URL_CONTROL_CHARACTERS
    first_control_position = min(
        (candidate.find(control) for control in url_controls if control in candidate),
        default=-1,
    )
    markdown_boundary = candidate.find(
        "](",
        max(minimum_length, first_control_position + 1),
    )
    if first_control_position != -1 and markdown_boundary != -1:
        candidate = candidate[:markdown_boundary]

    last_control_position = max(
        (candidate.rfind(control) for control in ASCII_URL_CONTROL_CHARACTERS),
        default=-1,
    )
    control_is_in_protected_prefix = minimum_length > 0 and last_control_position < minimum_length
    minimum_length = max(minimum_length, last_control_position + 1)
    unmatched_boundaries = _POST_CONTROL_UNMATCHED_URL_BOUNDARIES

    candidate = _truncate_at_post_control_hard_boundary(
        candidate,
        minimum_length,
        control_is_in_protected_prefix,
    )
    candidate = _truncate_at_unmatched_closing_boundary(
        candidate,
        minimum_length,
        control_is_in_protected_prefix,
    )

    if not control_is_in_protected_prefix:
        unmatched_mask: bytearray | None = None
    else:
        unmatched_mask = _mark_unmatched_closing_delimiters(candidate)

    candidate_end = len(candidate)
    while candidate_end > minimum_length:
        trailing_position = candidate_end - 1
        delimiter = candidate[trailing_position]
        is_unmatched = unmatched_mask is not None and bool(unmatched_mask[trailing_position])
        if is_unmatched and unmatched_mask is not None:
            previous_matched_position = unmatched_mask.rfind(
                b"\x00",
                minimum_length,
                candidate_end,
            )
            candidate_end = max(minimum_length, previous_matched_position + 1)
            continue
        has_unmatched_delimiter = is_unmatched and delimiter in unmatched_boundaries
        is_trailing_boundary = delimiter in _TRAILING_URL_BOUNDARIES
        is_trailing_boundary = is_trailing_boundary or has_unmatched_delimiter
        if not is_trailing_boundary:
            break
        candidate_end -= 1

    return candidate[:candidate_end]


def _is_non_control_whitespace(character: str) -> bool:
    """Check whether a character terminates an ambiguous URL candidate.

    Args:
        character: Character to classify.

    Returns:
        True for whitespace other than ASCII TAB, LF, and CR.
    """
    return character.isspace() and character not in _ASCII_URL_CONTROL_SET


def _find_userinfo_marker_before_path_boundary(
    text: str,
    search_start: int,
) -> int | None:
    """Find the next userinfo marker in the current authority.

    Args:
        text: Source text containing the authority.
        search_start: Offset where the forward search begins.

    Returns:
        The next userinfo marker offset, or None if a path, query, fragment,
        or the end of the text is reached first.
    """
    for position in range(search_start, len(text)):
        character = text[position]
        if character in "/?#":
            return None
        if character == "@":
            return position

    return None


def _refresh_userinfo_marker(
    text: str,
    whitespace_position: int,
    cached_marker: int | None,
) -> int | None:
    """Reuse or refresh the next userinfo marker for an authority.

    Args:
        text: Source text containing the authority.
        whitespace_position: Current non-control whitespace offset.
        cached_marker: Previously located userinfo marker, if any.

    Returns:
        A cached or newly located userinfo marker, or None before a path
        boundary or the end of the text.
    """
    if cached_marker is not None and whitespace_position < cached_marker:
        return cached_marker
    return _find_userinfo_marker_before_path_boundary(
        text,
        whitespace_position + 1,
    )


def _scan_url_like_segment(
    text: str,
    segment_start: int,
    scheme_matches_by_start: dict[int, _AmbiguousSchemeMatch],
) -> tuple[int, bool]:
    """Find a URL-like segment end without splitting userinfo syntax.

    Args:
        text: Source text to scan.
        segment_start: Inclusive offset where the segment begins.
        scheme_matches_by_start: Scheme-prefix matches keyed by source start.

    Returns:
        The exclusive segment end and whether it contains an ASCII URL
        control character.
    """
    segment_end = segment_start
    has_control = False
    authority_is_open = False
    authority_opens_at: int | None = None
    next_userinfo_marker: int | None = None

    while segment_end < len(text):
        if authority_opens_at is None:
            scheme_match = scheme_matches_by_start.get(segment_end)
            if scheme_match is not None and scheme_match.opens_authority:
                authority_opens_at = scheme_match.end

        character = text[segment_end]
        if not _is_non_control_whitespace(character):
            if character in _ASCII_URL_CONTROL_SET:
                has_control = True
            if character in "/?#":
                authority_is_open = False
                next_userinfo_marker = None
            segment_end += 1
            if segment_end == authority_opens_at:
                authority_is_open = True
                authority_opens_at = None
            continue

        if not authority_is_open:
            break

        next_userinfo_marker = _refresh_userinfo_marker(
            text,
            segment_end,
            next_userinfo_marker,
        )
        if next_userinfo_marker is None:
            break
        segment_end += 1

    return segment_end, has_control


def _scan_ambiguous_url_candidate(
    text: str,
    scheme_start: int,
    scheme_end: int,
    segment_end: int,
    scheme_matches_by_start: dict[int, _AmbiguousSchemeMatch],
) -> tuple[str | None, int]:
    """Scan one possible control-bearing URL candidate in linear time.

    Args:
        text: Source text containing the matched scheme.
        scheme_start: Inclusive offset of the matched scheme.
        scheme_end: Exclusive offset of the matched scheme.
        segment_end: Exclusive offset of the non-whitespace segment.
        scheme_matches_by_start: Scheme-prefix matches keyed by source start.

    Returns:
        The cleaned raw candidate when ambiguous, plus the exclusive offset
        scanned. The candidate is None when no ASCII URL control was included.
    """
    matched_scheme = text[scheme_start:scheme_end]
    has_control = not _ASCII_URL_CONTROL_SET.isdisjoint(matched_scheme)
    candidate_end = scheme_end

    while candidate_end < segment_end:
        character = text[candidate_end]
        if character == "]" and candidate_end + 1 < segment_end and text[candidate_end + 1] == "(":
            nested_scheme = scheme_matches_by_start.get(candidate_end + 2)
            if has_control or nested_scheme is not None:
                break
        if character in _ASCII_URL_CONTROL_SET:
            has_control = True
        candidate_end += 1

    if not has_control:
        return None, candidate_end

    raw_candidate = text[scheme_start:candidate_end]
    protected_prefix_end = scheme_end
    while protected_prefix_end < candidate_end:
        if text[protected_prefix_end] not in _ASCII_URL_CONTROL_SET:
            break
        protected_prefix_end += 1
    candidate = _clean_ambiguous_url_candidate(
        raw_candidate,
        minimum_length=protected_prefix_end - scheme_start,
    )
    return candidate, candidate_end


def _normalize_url_control_segment(
    text: str,
    segment_start: int,
    segment_end: int,
) -> tuple[str, list[int]]:
    """Remove URL controls while preserving source position mappings.

    Args:
        text: Source text containing the segment.
        segment_start: Inclusive segment offset.
        segment_end: Exclusive segment offset.

    Returns:
        Control-stripped text and the raw positions of removed controls.
    """
    segment = text[segment_start:segment_end]
    normalized_text = segment.translate(_ASCII_URL_CONTROL_TRANSLATION)
    control_matches = _ASCII_URL_CONTROL_RE.finditer(segment)
    control_positions = [segment_start + match.start() for match in control_matches]
    return normalized_text, control_positions


def _find_raw_scheme_less_candidate_starts(
    text: str,
    normalized_text: str,
    segment_start: int,
    segment_end: int,
    control_positions: list[int],
) -> list[int]:
    """Find raw starts from normalized and source URL boundaries.

    Args:
        text: Source text containing the raw segment.
        normalized_text: Segment text with ASCII URL controls removed.
        segment_start: Raw offset where the normalized segment begins.
        segment_end: Exclusive raw offset where the segment ends.
        control_positions: Raw positions of removed controls.

    Returns:
        Sorted, unique raw starts for scheme-less URL candidates.
    """
    if not normalized_text:
        return []

    # These ASCII-boundary patterns cover the ordinary ``\b`` matches and
    # may conservatively start earlier after a Unicode word character.
    domain_spans = _detect_domain_like_url_spans(
        normalized_text,
        AMBIGUOUS_DOMAIN_HOST_CANDIDATE_RE,
    )
    ambiguous_ip_matches = AMBIGUOUS_IP_URL_RE.finditer(normalized_text)
    ambiguous_ip_spans = ((match.start(), match.end()) for match in ambiguous_ip_matches)
    raw_starts: list[int] = []
    first_content_position = segment_start
    for control_position in control_positions:
        if control_position != first_content_position:
            break
        first_content_position += 1
    first_inner_control = next(
        (position for position in control_positions if position > first_content_position),
        None,
    )
    previous_normalized_start: int | None = None
    control_index = 0
    control_count = len(control_positions)
    for normalized_start, _ in merge(
        domain_spans,
        ambiguous_ip_spans,
    ):
        if normalized_start == previous_normalized_start:
            continue
        previous_normalized_start = normalized_start
        raw_start = segment_start + normalized_start + control_index
        while control_index < control_count and control_positions[control_index] <= raw_start:
            raw_start += 1
            control_index += 1
        if not raw_starts or raw_starts[-1] != raw_start:
            raw_starts.append(raw_start)

    if not raw_starts:
        # A control can create an ordinary raw boundary that disappears when
        # normalization joins an ASCII prefix to an IP-shaped suffix.
        raw_segment = text[segment_start:segment_end]
        raw_domain_spans = _detect_domain_like_url_spans(raw_segment)
        raw_ip_matches = IP_URL_RE.finditer(raw_segment)
        raw_ip_spans = ((match.start(), match.end()) for match in raw_ip_matches)
        for raw_start, _ in merge(raw_domain_spans, raw_ip_spans):
            source_start = segment_start + raw_start
            if not raw_starts or raw_starts[-1] != source_start:
                raw_starts.append(source_start)

    if raw_starts and first_inner_control is not None and first_inner_control < raw_starts[0]:
        raw_starts[0] = first_content_position

    return raw_starts


def _join_scheme_less_userinfo_spans(
    text: str,
    raw_starts: list[int],
    segment_end: int,
) -> Iterator[tuple[int, int]]:
    """Join scheme-less domains separated by userinfo markers.

    Args:
        text: Source text containing the URL-like segment.
        raw_starts: Sorted raw starts of domain and IP matches.
        segment_end: Exclusive end of the URL-like segment.

    Returns:
        An iterator of candidate spans with domains around ``@`` kept
        together.
    """
    start_index = 0
    while start_index < len(raw_starts):
        raw_start = raw_starts[start_index]
        next_start_index = start_index + 1
        while (
            next_start_index < len(raw_starts)
            and text.find(
                "@",
                raw_starts[next_start_index - 1],
                raw_starts[next_start_index],
            )
            != -1
        ):
            next_start_index += 1
        if next_start_index < len(raw_starts):
            raw_end = raw_starts[next_start_index]
        else:
            raw_end = segment_end
        yield raw_start, raw_end
        start_index = next_start_index


def _find_control_bearing_scheme_less_candidates(
    text: str,
    segment_start: int,
    segment_end: int,
    occupied_spans: list[tuple[int, int]],
) -> list[tuple[str, tuple[int, int]]]:
    """Map control-stripped domain and IP matches back to raw spans.

    The control-stripped text is used only to discover candidates. Returned
    candidates always preserve the raw source text for fail-closed handling.

    Args:
        text: Source text containing the segment.
        segment_start: Inclusive segment offset.
        segment_end: Exclusive segment offset.
        occupied_spans: Explicit-scheme spans already claimed in the segment.

    Returns:
        Raw control-bearing scheme-less candidates and their source spans.
    """
    if occupied_spans:
        first_occupied_start, first_occupied_end = occupied_spans[0]
        if first_occupied_start <= segment_start and segment_end <= first_occupied_end:
            return []

    normalized_text, control_positions = _normalize_url_control_segment(
        text,
        segment_start,
        segment_end,
    )
    raw_starts = _find_raw_scheme_less_candidate_starts(
        text,
        normalized_text,
        segment_start,
        segment_end,
        control_positions,
    )
    raw_spans = _join_scheme_less_userinfo_spans(text, raw_starts, segment_end)

    candidates: list[tuple[str, tuple[int, int]]] = []
    control_index = 0
    occupied_index = 0
    occupied_count = len(occupied_spans)
    control_count = len(control_positions)
    for raw_start, raw_end in raw_spans:
        while occupied_index < occupied_count and occupied_spans[occupied_index][1] <= raw_start:
            occupied_index += 1
        if occupied_index < occupied_count:
            occupied_start, occupied_end = occupied_spans[occupied_index]
            if occupied_start <= raw_start < occupied_end:
                continue
            raw_end = min(raw_end, occupied_start)

        while control_index < control_count and control_positions[control_index] < raw_start:
            control_index += 1
        if control_index == control_count or control_positions[control_index] >= raw_end:
            continue

        raw_candidate = text[raw_start:raw_end]
        candidate = _clean_ambiguous_url_candidate(raw_candidate, minimum_length=0)
        candidates.append((candidate, (raw_start, raw_start + len(candidate))))

    return candidates


def _find_ambiguous_url_candidates(
    text: str,
    allowed_schemes: set[str] | frozenset[str] = frozenset(),
) -> list[tuple[str, tuple[int, int]]]:
    """Find raw control-bearing URL candidates and their source spans.

    Args:
        text: Text to scan for ambiguous URL-like candidates.
        allowed_schemes: Additional configured scheme names to recognize.

    Returns:
        Raw candidates paired with their source spans.
    """
    next_control_match = _ASCII_URL_CONTROL_RE.search(text)
    if next_control_match is None:
        return []

    scheme_matcher = _build_ambiguous_scheme_matcher(allowed_schemes)
    scheme_matches = _find_ambiguous_scheme_matches(text, scheme_matcher)
    scheme_matches_by_start = {match.start: match for match in scheme_matches}
    candidates: list[tuple[str, tuple[int, int]]] = []
    segment_start = 0
    scheme_match_index = 0

    while segment_start < len(text) and next_control_match is not None:
        while segment_start < len(text) and _is_non_control_whitespace(text[segment_start]):
            segment_start += 1
        if segment_start == len(text):
            break

        segment_end, has_segment_control = _scan_url_like_segment(
            text,
            segment_start,
            scheme_matches_by_start,
        )

        if not has_segment_control:
            segment_start = segment_end
            continue

        segment_candidates: list[tuple[str, tuple[int, int]]] = []
        search_position = segment_start
        while scheme_match_index < len(scheme_matches) and scheme_matches[scheme_match_index].start < segment_start:
            scheme_match_index += 1
        while scheme_match_index < len(scheme_matches) and scheme_matches[scheme_match_index].start < segment_end:
            match = scheme_matches[scheme_match_index]
            scheme_match_index += 1
            if match.start < search_position:
                continue
            candidate, candidate_end = _scan_ambiguous_url_candidate(
                text,
                match.start,
                match.end,
                segment_end,
                scheme_matches_by_start,
            )
            search_position = max(match.end, candidate_end)
            if candidate is not None:
                candidate_span = (match.start, match.start + len(candidate))
                segment_candidates.append((candidate, candidate_span))

        occupied_spans = [span for _, span in segment_candidates]
        scheme_less_candidates = _find_control_bearing_scheme_less_candidates(
            text,
            segment_start,
            segment_end,
            occupied_spans,
        )
        candidates.extend(
            merge(
                segment_candidates,
                scheme_less_candidates,
                key=lambda item: item[1],
            )
        )

        next_control_match = _ASCII_URL_CONTROL_RE.search(text, segment_end)
        segment_start = segment_end

    return candidates


def _is_nested_url_value_start(
    text: str,
    owner_end: int,
    start: int,
) -> bool:
    """Return whether a scheme continues an owned URL query value.

    Args:
        text: Original text containing the scheme.
        owner_end: Exclusive end of the previous owned URL span.
        start: Inclusive scheme offset.

    Returns:
        True when the scheme remains inside the previous query-value span.
    """
    _ = text
    return owner_end >= 0 and start < owner_end


def _token_end(text: str, start: int) -> int:
    """Return the end of the whitespace-delimited token at ``start``.

    Args:
        text: Source text containing the token.
        start: Inclusive offset inside the token.

    Returns:
        The exclusive token end.
    """
    end = start
    while end < len(text) and not text[end].isspace():
        end += 1
    return end


def _presentation_wrapper_end(text: str, start: int, end: int) -> int | None:
    """Return the end of an immediately enclosing presentation wrapper.

    Args:
        text: Source text containing the URL.
        start: Inclusive URL offset.
        end: Exclusive cleaned URL offset.

    Returns:
        The URL end when the next character closes an immediate wrapper, or
        ``None`` when the URL is not immediately wrapped.
    """
    if start == 0 or end >= len(text):
        return None
    expected_closer = _PRESENTATION_CLOSERS.get(text[start - 1])
    if expected_closer is None or text[end] != expected_closer:
        return None
    return end


def _has_valid_http_authority(url: str) -> bool:
    """Return whether text has a released-valid HTTP authority.

    Args:
        url: Candidate HTTP URL prefix.

    Returns:
        True when the prefix has an HTTP(S) scheme, nonempty host, and valid
        port syntax.
    """
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        _port = parsed_url.port
    except (ValueError, UnicodeError):
        return False
    return parsed_url.scheme.lower() in {"http", "https"} and hostname is not None


def _has_open_http_query_value(url: str) -> bool:
    """Return whether a valid HTTP URL ends with an empty query value.

    Args:
        url: Accepted explicit URL candidate.

    Returns:
        True when the URL has no fragment and its final query field ends in
        an equals sign.
    """
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname
        _port = parsed_url.port
    except (ValueError, UnicodeError):
        return False

    final_query_field = parsed_url.query.rsplit("&", 1)[-1]
    value_separator = final_query_field.find("=")
    return (
        parsed_url.scheme.lower() in {"http", "https"}
        and hostname is not None
        and not parsed_url.fragment
        and bool(parsed_url.query)
        and value_separator == len(final_query_field) - 1
    )


def _clean_detected_scheme_url(url: str) -> str:
    """Remove presentation punctuation from an explicit URL candidate.

    Args:
        url: Raw explicit-scheme candidate.

    Returns:
        Candidate without trailing presentation punctuation. A closing bracket
        that terminates an IPv6 authority is preserved.
    """
    bracketed_authority = _BRACKETED_HTTP_AUTHORITY_RE.match(url)
    structural_prefix_end = bracketed_authority.end() if bracketed_authority else 0
    return url[:structural_prefix_end] + _TRAILING_DETECTED_URL_PUNCTUATION_RE.sub(
        "",
        url[structural_prefix_end:],
    )


def _iter_adjacent_url_starts(url: str, search_start: int) -> Iterator[int]:
    """Yield boundary-aligned URL starts in source order.

    Args:
        url: Explicit URL candidate to scan.
        search_start: Inclusive offset for nested candidates.

    Yields:
        Unique source offsets for explicit schemes, domains, and IPv4 hosts.
    """
    explicit_starts = (
        match.start()
        for match in _EXPLICIT_URL_SCHEME_RE.finditer(url, search_start)
        if match.start() > search_start and url[match.start() - 1] in _ADJACENT_SCHEME_URL_BOUNDARIES
    )
    domain_starts = (match.start() for match in _ADJACENT_DOMAIN_START_RE.finditer(url, search_start) if _find_domain_end(match.group()) is not None)
    ip_starts = (match.start() for match in _ADJACENT_IP_START_RE.finditer(url, search_start))
    previous_start = -1
    for start in merge(explicit_starts, domain_starts, ip_starts):
        if start != previous_start:
            yield start
            previous_start = start


def _is_adjacent_url_start(text: str, start: int) -> bool:
    """Return whether presentation punctuation introduces a URL.

    Args:
        text: Source text containing the possible URL.
        start: Inclusive offset immediately after presentation punctuation.

    Returns:
        True when an explicit, domain, or IPv4 URL starts at the offset.
    """
    if _EXPLICIT_URL_SCHEME_RE.match(text, start) is not None:
        return True
    domain_match = _ADJACENT_DOMAIN_START_RE.match(text, start)
    if domain_match is not None and _find_domain_end(domain_match.group()) is not None:
        return True
    return _ADJACENT_IP_START_RE.match(text, start) is not None


def _find_scheme_less_path_end(text: str, path_start: int) -> int:
    """Find a scheme-less path end without rescanning later suffixes.

    Args:
        text: Source text containing the path.
        path_start: Offset of the slash that begins the path.

    Returns:
        The exclusive path end at whitespace, token end, or an adjacent URL
        presentation boundary.
    """
    position = path_start + 1
    while position < len(text):
        character = text[position]
        if character.isspace():
            return position
        if character in _ADJACENT_SCHEME_URL_BOUNDARIES and position + 1 < len(text) and _is_adjacent_url_start(text, position + 1):
            return position
        position += 1
    return position


def _has_owned_http_query_value(url: str) -> bool:
    """Return whether an empty query value owns a nested URL suffix.

    Args:
        url: Raw HTTP(S) candidate before presentation cleanup.

    Returns:
        True when a query field ending in ``=`` is followed only by nesting
        punctuation before a URL-shaped descendant.
    """
    scheme_prefix_end = url.find("://") + 3
    query_start = url.find("?", scheme_prefix_end)
    if query_start < 0:
        return False
    fragment_start = url.find("#", query_start)
    query_end = fragment_start if fragment_start >= 0 else len(url)
    nested_boundaries = _ADJACENT_SCHEME_URL_BOUNDARIES | frozenset(("/", "\\"))

    explicit_starts = (match.start() for match in _EXPLICIT_URL_SCHEME_RE.finditer(url, query_start + 1))
    domain_starts = (
        match.start() for match in _ADJACENT_DOMAIN_START_RE.finditer(url, query_start + 1) if _find_domain_end(match.group()) is not None
    )
    ip_starts = (match.start() for match in _ADJACENT_IP_START_RE.finditer(url, query_start + 1))
    scan_position = query_start + 1
    value_separator_found = False
    separator_is_nested = True
    separator_length = 0
    separator_start = -1
    field_owns_nested_urls = False
    for start in chain(merge(explicit_starts, domain_starts, ip_starts), (query_end,)):
        scan_end = min(start, query_end)
        while scan_position < scan_end:
            character = url[scan_position]
            if character == "&":
                value_separator_found = False
                separator_is_nested = True
                separator_length = 0
                separator_start = -1
                field_owns_nested_urls = False
            elif not field_owns_nested_urls and not value_separator_found:
                if character == "=":
                    value_separator_found = True
                    separator_is_nested = True
                    separator_length = 0
                    separator_start = scan_position + 1
            elif not field_owns_nested_urls:
                separator_length += 1
                if character not in nested_boundaries:
                    separator_is_nested = False
            scan_position += 1
        if start >= query_end:
            break
        if (
            not field_owns_nested_urls
            and value_separator_found
            and separator_length
            and separator_is_nested
            and separator_start >= 0
            and _mark_unmatched_closing_delimiters(url[separator_start:scan_end]) is None
        ):
            field_owns_nested_urls = True
    return field_owns_nested_urls


def _truncate_before_adjacent_scheme_less_url(url: str) -> str:
    """Split a presentation-adjacent scheme-less URL from a scheme URL.

    Args:
        url: Raw explicit-scheme candidate.

    Returns:
        The explicit URL prefix, unless an active empty query value owns the
        adjacent scheme-less candidate.
    """
    lowered_url = url.lower()
    if not lowered_url.startswith(("http://", "https://")):
        scheme_prefix_end = url.find(":") + 1
        if scheme_prefix_end <= 0:
            return url
        data_payload_separator = url.find(",", scheme_prefix_end) if lowered_url.startswith("data:") else -1
        for start in _iter_adjacent_url_starts(url, scheme_prefix_end):
            boundary = start - 1
            if boundary == data_payload_separator:
                continue
            return url[:boundary]
        return url

    scheme_prefix_end = url.find("://") + 3
    component_starts = [position for delimiter in "/?#" if (position := url.find(delimiter, scheme_prefix_end)) >= 0]
    first_component_start = min(component_starts, default=len(url))
    userinfo_end = url.rfind("@", scheme_prefix_end, first_component_start)
    query_start = url.find("?", first_component_start)
    fragment_start = url.find("#", first_component_start)
    authority_is_valid: bool | None = None
    query_scan_position = query_start + 1
    query_value_separator_found = False
    query_value_has_content = False
    query_field_owns_nested_urls = False
    for start in _iter_adjacent_url_starts(url, scheme_prefix_end):
        boundary = start - 1
        boundary_character = url[boundary]
        if start <= userinfo_end:
            continue
        if boundary_character == ":" and boundary < first_component_start:
            return url
        if boundary_character == "." and boundary < first_component_start:
            continue
        if authority_is_valid is None:
            authority_is_valid = _has_valid_http_authority(url[:boundary])
            if not authority_is_valid:
                return url

        in_query = query_start >= 0 and boundary > query_start and (fragment_start < 0 or boundary < fragment_start)
        if in_query:
            while query_scan_position < boundary:
                character = url[query_scan_position]
                if character == "&":
                    query_value_separator_found = False
                    query_value_has_content = False
                    query_field_owns_nested_urls = False
                elif not query_field_owns_nested_urls and not query_value_separator_found:
                    if character == "=":
                        query_value_separator_found = True
                elif not query_field_owns_nested_urls:
                    query_value_has_content = True
                query_scan_position += 1
            if query_field_owns_nested_urls:
                continue
            if query_value_separator_found and not query_value_has_content:
                query_field_owns_nested_urls = True
                continue
        if boundary_character == "." and (fragment_start < 0 or boundary < fragment_start):
            continue
        return url[:boundary]

    return url


def _has_same_preserved_url_identity(source_url: str, preserved_url: str) -> bool:
    """Compare URL identity while normalizing only scheme and host case.

    Args:
        source_url: URL spelling recovered from the source text.
        preserved_url: Exact configured URL spelling.

    Returns:
        True when normalized authority and case-sensitive components match.
    """
    try:
        source_parsed = urlparse(source_url)
        preserved_parsed = urlparse(preserved_url)
        same_authority = (
            source_parsed.scheme.lower() == preserved_parsed.scheme.lower()
            and source_parsed.hostname is not None
            and preserved_parsed.hostname is not None
            and source_parsed.hostname.lower() == preserved_parsed.hostname.lower()
            and source_parsed.port == preserved_parsed.port
        )
    except (ValueError, UnicodeError):
        return False
    same_components = (
        source_parsed.username == preserved_parsed.username
        and source_parsed.password == preserved_parsed.password
        and source_parsed.path == preserved_parsed.path
        and source_parsed.params == preserved_parsed.params
        and source_parsed.query == preserved_parsed.query
        and source_parsed.fragment == preserved_parsed.fragment
    )
    return same_authority and same_components


def _find_preserved_component_url_prefix(
    raw_candidate: str,
    preserved_component_urls: tuple[str, ...],
) -> str | None:
    """Find an exact allow-list URL at the start of a longer source span.

    Args:
        raw_candidate: Raw explicit-scheme regex candidate.
        preserved_component_urls: Exact allow-list URL strings.

    Returns:
        The longest exact URL prefix ending before presentation punctuation or
        an adjacent URL, or ``None`` when the candidate must use normal
        adjacency splitting.
    """
    for preserved_url in preserved_component_urls:
        source_prefix = raw_candidate[: len(preserved_url)]
        if not _has_same_preserved_url_identity(source_prefix, preserved_url):
            continue
        if len(raw_candidate) == len(preserved_url):
            return source_prefix

        suffix = raw_candidate[len(preserved_url) :]
        if (
            source_prefix
            and source_prefix[-1] in _ADJACENT_SCHEME_URL_BOUNDARIES
            and _EXPLICIT_URL_SCHEME_RE.match(raw_candidate, len(preserved_url)) is not None
        ):
            return source_prefix
        if suffix and all(character in _ADJACENT_SCHEME_URL_BOUNDARIES for character in suffix):
            return source_prefix

        for adjacent_start in _iter_adjacent_url_starts(raw_candidate, len(preserved_url)):
            bridge = raw_candidate[len(preserved_url) : adjacent_start]
            prefix_ends_with_boundary = source_prefix[-1] in _ADJACENT_SCHEME_URL_BOUNDARIES
            if (bridge and all(character in _ADJACENT_SCHEME_URL_BOUNDARIES for character in bridge)) or (not bridge and prefix_ends_with_boundary):
                return source_prefix
            break
    return None


def _iter_preserved_component_url_spans(
    text: str,
    preserved_component_urls: tuple[str, ...],
) -> Iterator[tuple[int, int, str, bool]]:
    """Yield exact configured URL spans directly from source text.

    Args:
        text: Source text scanned for URLs.
        preserved_component_urls: Exact allow-list URL strings.

    Yields:
        Scheme-candidate tuples preserving the original source spelling.
    """
    for scheme_match in _EXPLICIT_URL_SCHEME_RE.finditer(text):
        start = scheme_match.start()
        if start > 0 and not text[start - 1].isspace() and text[start - 1] not in _ADJACENT_SCHEME_URL_BOUNDARIES:
            continue
        for preserved_url in preserved_component_urls:
            end = start + len(preserved_url)
            if end > len(text):
                continue
            source_url = text[start:end]
            if not _has_same_preserved_url_identity(source_url, preserved_url):
                continue
            if end < len(text):
                next_character = text[end]
                source_ends_with_boundary = source_url[-1] in _ADJACENT_SCHEME_URL_BOUNDARIES
                adjacent_url = source_ends_with_boundary and _is_adjacent_url_start(text, end)
                if not next_character.isspace() and next_character not in _ADJACENT_SCHEME_URL_BOUNDARIES and not adjacent_url:
                    continue
            yield start, end, source_url, False
            break


def _find_adjacent_explicit_url_after_prefix(raw_candidate: str, prefix_end: int) -> int | None:
    """Find an explicit URL separated from a preserved source prefix.

    Args:
        raw_candidate: Raw explicit-scheme regex candidate.
        prefix_end: Exclusive end of the preserved URL prefix.

    Returns:
        The adjacent URL start, or ``None`` when intervening text is content.
    """
    match = _EXPLICIT_URL_SCHEME_RE.search(raw_candidate, prefix_end)
    if match is None:
        return None
    bridge = raw_candidate[prefix_end : match.start()]
    prefix_ends_with_boundary = prefix_end > 0 and raw_candidate[prefix_end - 1] in _ADJACENT_SCHEME_URL_BOUNDARIES
    if (bridge and all(character in _ADJACENT_SCHEME_URL_BOUNDARIES for character in bridge)) or (not bridge and prefix_ends_with_boundary):
        return match.start()
    return None


def _released_url_detection_order(candidate: tuple[int, int, str]) -> tuple[int, int]:
    """Return the released detector category and source order.

    Args:
        candidate: Source start, source end, and detected URL text.

    Returns:
        A stable ordering key that preserves the released pattern-category
        order while retaining source order inside each category.
    """
    start, _, url = candidate
    lowered_url = url.lower()
    if lowered_url.startswith(("http://", "https://")):
        category = 0
    elif lowered_url.startswith("ftp://"):
        category = 1
    elif lowered_url.startswith("data:"):
        category = 2
    elif lowered_url.startswith("javascript:"):
        category = 3
    elif lowered_url.startswith("vbscript:"):
        category = 4
    elif IP_URL_RE.fullmatch(url) is not None:
        category = 6
    else:
        category = 5
    return category, start


def _detect_urls(
    text: str,
    allowed_schemes: set[str] | frozenset[str] = frozenset(),
    *,
    preserved_component_urls: frozenset[str] = frozenset(),
) -> list[str]:
    """Detect URLs using regex patterns with deduplication.

    Detects URLs with explicit schemes (http, https, ftp, data, javascript,
    vbscript), domain-like patterns without schemes, and IP addresses.
    Deduplicates to avoid returning both scheme-ful and scheme-less versions
    of the same URL.

    Args:
        text: The text to scan for URLs.
        allowed_schemes: Additional configured scheme names to recognize.
        preserved_component_urls: Exact URL strings whose valid components may
            retain presentation punctuation followed by domain-shaped text.

    Returns:
        List of unique URL strings found in the text, with trailing
        punctuation removed.
    """
    ambiguous_candidates = _find_ambiguous_url_candidates(text, allowed_schemes)
    preserved_component_url_prefixes = tuple(sorted(preserved_component_urls, key=len, reverse=True))
    ambiguous_urls = [candidate for candidate, _ in ambiguous_candidates]
    if ambiguous_candidates:
        detection_characters = list(text)
        for _, (start, end) in ambiguous_candidates:
            detection_characters[start:end] = " " * (end - start)
        text_without_ambiguous_candidates = "".join(detection_characters)
    else:
        text_without_ambiguous_candidates = text

    # Pattern 1: URLs with schemes (highest priority)
    scheme_patterns = [
        _BRACKETED_HTTP_URL_PATTERN,
        # Let a presentation-delimited later explicit scheme start a new match.
        _HTTP_URL_PATTERN,
        _FTP_URL_PATTERN,
        r'data:[^\s<>"{}|\\^`\[\]]+',
        r'javascript:[^\s<>"{}|\\^`\[\]]+',
        r'vbscript:[^\s<>"{}|\\^`\[\]]+',
    ]

    scheme_candidates = list(
        _iter_preserved_component_url_spans(
            text_without_ambiguous_candidates,
            preserved_component_url_prefixes,
        )
    )
    for pattern in scheme_patterns:
        matches = re.finditer(pattern, text_without_ambiguous_candidates, re.IGNORECASE)
        for scheme_match in matches:
            # Clean trailing punctuation
            raw_candidate = scheme_match.group()
            preserved_prefix = _find_preserved_component_url_prefix(
                raw_candidate,
                preserved_component_url_prefixes,
            )
            separated = preserved_prefix or _truncate_before_adjacent_scheme_less_url(raw_candidate)
            cleaned = preserved_prefix or _clean_detected_scheme_url(separated)
            candidate_start = scheme_match.start()
            if (
                preserved_prefix is None
                and candidate_start > 0
                and text_without_ambiguous_candidates[candidate_start - 1] == "'"
                and cleaned.endswith("'")
            ):
                cleaned = cleaned[:-1]
            if cleaned:
                trailing_presentation = separated[len(cleaned) :]
                source_suffix_start = candidate_start + len(cleaned)
                source_suffix_starts_with_closer = (
                    source_suffix_start < len(text_without_ambiguous_candidates)
                    and text_without_ambiguous_candidates[source_suffix_start] in _PAIRED_DELIMITER_CLOSE_TO_OPEN_CODE
                )
                open_query_owns_suffix = (
                    _has_open_http_query_value(cleaned)
                    and not source_suffix_starts_with_closer
                    and _mark_unmatched_closing_delimiters(trailing_presentation) is None
                )
                owns_query_value = preserved_prefix is None and (
                    open_query_owns_suffix or (_has_valid_http_authority(cleaned) and _has_owned_http_query_value(raw_candidate))
                )
                scheme_candidates.append(
                    (
                        candidate_start,
                        candidate_start + len(cleaned),
                        cleaned,
                        owns_query_value,
                    )
                )
                if preserved_prefix is not None:
                    adjacent_start = _find_adjacent_explicit_url_after_prefix(
                        raw_candidate,
                        len(preserved_prefix),
                    )
                    if adjacent_start is not None:
                        adjacent_raw = raw_candidate[adjacent_start:]
                        adjacent_url = _clean_detected_scheme_url(_truncate_before_adjacent_scheme_less_url(adjacent_raw))
                        if adjacent_url:
                            adjacent_source_start = candidate_start + adjacent_start
                            scheme_candidates.append(
                                (
                                    adjacent_source_start,
                                    adjacent_source_start + len(adjacent_url),
                                    adjacent_url,
                                    False,
                                )
                            )

    covered_end = -1
    nested_value_owner_end = -1
    owned_query_spans: list[tuple[int, int]] = []
    detected_candidates: list[tuple[int, int, str]] = []
    current_token_end = -1
    for start, end, cleaned, owns_query_value in sorted(
        scheme_candidates,
        key=lambda candidate: (candidate[0], -(candidate[1] - candidate[0])),
    ):
        if start < covered_end:
            continue
        if _is_nested_url_value_start(text, nested_value_owner_end, start):
            covered_end = end
            continue
        detected_candidates.append((start, end, cleaned))
        covered_end = end
        if owns_query_value:
            if end > current_token_end:
                current_token_end = _token_end(text, end)
            wrapper_end = _presentation_wrapper_end(text, start, end)
            if wrapper_end is not None:
                nested_value_owner_end = -1
                continue
            fragment_start = text.find("#", start, current_token_end)
            field_end = text.find("&", end, current_token_end)
            owner_boundaries = [boundary for boundary in (fragment_start, field_end) if boundary >= 0]
            nested_value_owner_end = min(owner_boundaries, default=current_token_end)
            if field_end >= 0 and (fragment_start < 0 or field_end < fragment_start):
                extended_raw = text[start:current_token_end]
                extended_url = _clean_detected_scheme_url(_truncate_before_adjacent_scheme_less_url(extended_raw))
                extended_end = start + len(extended_url)
                if extended_end > end:
                    detected_candidates[-1] = (start, extended_end, extended_url)
                    covered_end = extended_end
            if end < nested_value_owner_end:
                owned_query_spans.append((end, nested_value_owner_end))
        else:
            nested_value_owner_end = -1
    fallback_characters = list(text_without_ambiguous_candidates)
    for start, end, _, _ in scheme_candidates:
        fallback_characters[start:end] = " " * (end - start)
    for start, end in owned_query_spans:
        fallback_characters[start:end] = " " * (end - start)
    text_without_explicit_scheme_candidates = "".join(fallback_characters)

    # Pattern 2: Domain-like patterns (scheme-less) - but skip if already found with scheme
    domain_spans = _detect_domain_like_url_spans(text_without_explicit_scheme_candidates)
    domain_candidates = ((start, end, text_without_explicit_scheme_candidates[start:end]) for start, end in domain_spans)

    # Pattern 3: IP addresses - merge with domains to preserve source order.
    ip_spans = _detect_ip_url_spans(text_without_explicit_scheme_candidates)
    ip_candidates = ((start, end, text_without_explicit_scheme_candidates[start:end]) for start, end in ip_spans)
    for start, end, fallback_url in merge(
        domain_candidates,
        ip_candidates,
        key=lambda candidate: candidate[0],
    ):
        _ = start, end
        cleaned = _TRAILING_DETECTED_URL_PUNCTUATION_RE.sub("", fallback_url)
        if cleaned:
            detected_candidates.append((start, start + len(cleaned), cleaned))

    # Remove only non-adjacent bare hosts already represented by an explicit URL.
    final_urls: list[str] = []
    scheme_url_domains: set[str] = set()
    ambiguous_explicit_urls = [url for url in ambiguous_urls if "://" in url]
    ambiguous_scheme_less_urls = [url for url in ambiguous_urls if "://" not in url]

    # First pass: collect all domains from ordinary hierarchical URLs. Ambiguous
    # candidates have already been removed from fallback detection by source span,
    # so a surviving same-host fallback is a distinct occurrence.
    ordinary_explicit_urls = (url for _, _, url in detected_candidates if "://" in url)
    for url in ordinary_explicit_urls:
        try:
            parsed = urlparse(url)
            if parsed.hostname:
                scheme_url_domains.add(parsed.hostname.lower())
                # Also add www-stripped version
                bare_domain = parsed.hostname.lower().removeprefix("www.")
                scheme_url_domains.add(bare_domain)
        except (ValueError, UnicodeError):
            # Skip URLs with parsing errors (malformed URLs, encoding issues)
            # This is expected for edge cases and doesn't require logging
            pass
    # Second pass: retain source order while removing only legacy bare-host duplicates.
    for start, _, url in sorted(detected_candidates, key=_released_url_detection_order):
        if _EXPLICIT_URL_SCHEME_RE.match(url) is not None:
            final_urls.append(url)
            continue
        url_lower = url.lower().removeprefix("www.")
        presentation_adjacent = start > 0 and text[start - 1] in _ADJACENT_SCHEME_URL_BOUNDARIES
        if presentation_adjacent or url_lower not in scheme_url_domains:
            final_urls.append(url)

    hierarchical_urls = [url for url in final_urls if "://" in url]
    scheme_less_urls = [url for url in final_urls if "://" not in url]
    final_urls = [
        *ambiguous_explicit_urls,
        *hierarchical_urls,
        *ambiguous_scheme_less_urls,
        *scheme_less_urls,
    ]

    # Remove empty URLs and return unique list
    return list(dict.fromkeys([url for url in final_urls if url]))


def _validate_url_security(url_string: str, config: URLConfig) -> tuple[ParseResult | None, str, bool]:
    """Validate URL security properties using urllib.parse.

    Checks URL structure, validates the scheme is allowed, and ensures no
    credentials are embedded in userinfo if block_userinfo is enabled.

    Args:
        url_string: The URL string to validate.
        config: Configuration specifying allowed schemes and userinfo policy.

    Returns:
        A tuple of (parsed_url, error_reason, had_explicit_scheme). If validation
        succeeds, parsed_url is a ParseResult, error_reason is empty, and
        had_explicit_scheme indicates if the original URL included a scheme.
        If validation fails, parsed_url is None and error_reason describes the failure.
    """
    try:
        # Parse URL - track whether scheme was explicit
        has_explicit_scheme = False
        if "://" in url_string:
            # Standard URL with double-slash scheme (http://, https://, ftp://, etc.)
            parsed_url = urlparse(url_string)
            original_scheme = parsed_url.scheme
            has_explicit_scheme = True
        elif ":" in url_string and url_string.split(":", 1)[0] in {"data", "javascript", "vbscript", "mailto"}:
            # Special single-colon schemes
            parsed_url = urlparse(url_string)
            original_scheme = parsed_url.scheme
            has_explicit_scheme = True
        else:
            # Add http scheme for parsing only (user didn't specify a scheme)
            parsed_url = urlparse(f"http://{url_string}")
            original_scheme = None  # No explicit scheme
            has_explicit_scheme = False

        # Basic validation: must have scheme and netloc (except for special schemes)
        if not parsed_url.scheme:
            return None, "Invalid URL format", False

        # Special schemes like data: and javascript: don't need netloc
        special_schemes = {"data", "javascript", "vbscript", "mailto"}
        if parsed_url.scheme not in special_schemes and not parsed_url.netloc:
            return None, "Invalid URL format", False

        # Security validations - only validate scheme if it was explicitly provided
        if has_explicit_scheme and original_scheme not in config.allowed_schemes:
            return None, f"Blocked scheme: {original_scheme}", has_explicit_scheme

        if config.block_userinfo and (parsed_url.username or parsed_url.password):
            return None, "Contains userinfo (potential credential injection)", has_explicit_scheme

        # Everything else (IPs, localhost, private IPs) goes through allow list logic
        return parsed_url, "", has_explicit_scheme

    except (ValueError, UnicodeError, AttributeError) as e:
        # Common URL parsing errors:
        # - ValueError: Invalid URL structure, invalid port, etc.
        # - UnicodeError: Invalid encoding in URL
        # - AttributeError: Unexpected URL structure
        return None, f"Invalid URL format: {str(e)}", False
    except Exception as e:
        # Catch any unexpected errors but provide debugging info
        return None, f"URL parsing error: {type(e).__name__}: {str(e)}", False


def _safe_get_port(parsed: ParseResult, scheme: str) -> int | None:
    """Safely extract port from ParseResult, handling malformed ports.

    Args:
        parsed: The parsed URL.
        scheme: The URL scheme (for default port lookup).

    Returns:
        The port number, the default port for the scheme, or None if invalid.
    """
    try:
        return parsed.port or DEFAULT_PORTS.get(scheme.lower())
    except ValueError:
        # Port is out of range (0-65535) or malformed
        return None


def _is_url_allowed(
    parsed_url: ParseResult,
    allow_list: list[str],
    allow_subdomains: bool,
    url_had_explicit_scheme: bool,
) -> bool:
    """Check if parsed URL matches any entry in the allow list.

    Supports domain names, IP addresses, CIDR blocks, and full URLs with
    paths/ports/query strings. Allow list entries without explicit schemes
    match any scheme. Entries with schemes must match exactly against URLs
    with explicit schemes, but match any scheme-less URL.

    Args:
        parsed_url: The parsed URL to check.
        allow_list: List of allowed URL patterns (domains, IPs, CIDR, full URLs).
        allow_subdomains: If True, subdomains of allowed domains are permitted.
        url_had_explicit_scheme: Whether the original URL included an explicit scheme.

    Returns:
        True if the URL matches any allow list entry, False otherwise.
    """
    if not allow_list:
        return False

    url_host = parsed_url.hostname
    if not url_host:
        return False

    url_host = url_host.lower()
    url_domain = url_host.removeprefix("www.")
    scheme_lower = parsed_url.scheme.lower() if parsed_url.scheme else ""
    # Safely get port (rejects malformed ports)
    url_port = _safe_get_port(parsed_url, scheme_lower)
    # Early rejection of malformed ports
    try:
        _ = parsed_url.port  # This will raise ValueError for malformed ports
    except ValueError:
        return False
    url_path = parsed_url.path or "/"
    url_query = parsed_url.query
    url_fragment = parsed_url.fragment

    try:
        url_ip = ip_address(url_host)
    except (AddressValueError, ValueError):
        url_ip = None

    for allowed_entry in allow_list:
        allowed_entry = allowed_entry.strip()

        has_explicit_scheme = bool(SCHEME_PREFIX_RE.match(allowed_entry))
        if has_explicit_scheme:
            parsed_allowed = urlparse(allowed_entry)
        else:
            parsed_allowed = urlparse(f"//{allowed_entry}")
        allowed_host = (parsed_allowed.hostname or "").lower()
        allowed_scheme = parsed_allowed.scheme.lower() if parsed_allowed.scheme else ""
        # Check if port was explicitly specified (safely)
        try:
            allowed_port_explicit = parsed_allowed.port
        except ValueError:
            allowed_port_explicit = None
        allowed_port = _safe_get_port(parsed_allowed, allowed_scheme)
        allowed_path = parsed_allowed.path
        allowed_query = parsed_allowed.query
        allowed_fragment = parsed_allowed.fragment

        # Handle IP addresses and CIDR blocks (including schemes)
        try:
            allowed_ip = ip_address(allowed_host)
        except (AddressValueError, ValueError):
            allowed_ip = None

        if allowed_ip is not None:
            if url_ip is None:
                continue
            # Scheme matching for IPs: if both allow list and URL have explicit schemes, they must match
            if has_explicit_scheme and url_had_explicit_scheme and allowed_scheme and allowed_scheme != scheme_lower:
                continue
            # Port matching: enforce if allow list has explicit port
            if allowed_port_explicit is not None and allowed_port != url_port:
                continue
            if allowed_ip == url_ip:
                return True

            network_spec = allowed_host
            if parsed_allowed.path not in ("", "/"):
                network_spec = f"{network_spec}{parsed_allowed.path}"
            try:
                if network_spec and "/" in network_spec and url_ip in ip_network(network_spec, strict=False):
                    return True
            except (AddressValueError, ValueError):
                # Path segment might not represent a CIDR mask; ignore.
                pass
            continue

        if not allowed_host:
            continue

        allowed_domain = allowed_host.removeprefix("www.")

        # Port matching: enforce if allow list has explicit port
        if allowed_port_explicit is not None and allowed_port != url_port:
            continue

        host_matches = url_domain == allowed_domain or (allow_subdomains and url_domain.endswith(f".{allowed_domain}"))
        if not host_matches:
            continue

        # Scheme matching: if both allow list and URL have explicit schemes, they must match
        if has_explicit_scheme and url_had_explicit_scheme and allowed_scheme and allowed_scheme != scheme_lower:
            continue

        # Path matching with segment boundary respect
        if allowed_path not in ("", "/"):
            # Normalize trailing slashes to prevent issues with entries like "/api/"
            # which should match "/api/users" but would fail with double-slash check
            normalized_allowed_path = allowed_path.rstrip("/")
            # Ensure path matching respects segment boundaries to prevent
            # "/api" from matching "/api2" or "/api-v2"
            if url_path != allowed_path and url_path != normalized_allowed_path and not url_path.startswith(f"{normalized_allowed_path}/"):
                continue

        if allowed_query and allowed_query != url_query:
            continue

        if allowed_fragment and allowed_fragment != url_fragment:
            continue

        return True

    return False


async def urls(ctx: Any, data: str, config: URLConfig) -> GuardrailResult:
    """Detects URLs using regex patterns, validates them with Pydantic, and checks against the allow list.

    Args:
        ctx: Context object.
        data: Text to scan for URLs.
        config: Configuration object.
    """
    _ = ctx

    # Detect URLs using regex patterns
    normalized_allow_list_urls = (allowed_url.strip() for allowed_url in config.url_allow_list)
    explicit_allow_list_urls = frozenset(
        allowed_url for allowed_url in normalized_allow_list_urls if _EXPLICIT_URL_SCHEME_RE.match(allowed_url) is not None
    )
    detected_urls = _detect_urls(
        data,
        config.allowed_schemes,
        preserved_component_urls=explicit_allow_list_urls,
    )

    allowed, blocked = [], []
    blocked_reasons = []

    for url_string in detected_urls:
        if any(character in url_string for character in ASCII_URL_CONTROL_CHARACTERS):
            blocked.append(url_string)
            blocked_reasons.append(f"{url_string}: {AMBIGUOUS_URL_REASON}")
            continue

        # Validate URL with security checks
        parsed_url, error_reason, url_had_explicit_scheme = _validate_url_security(url_string, config)

        if parsed_url is None:
            blocked.append(url_string)
            blocked_reasons.append(f"{url_string}: {error_reason}")
            continue

        # Check against allow list
        # Special schemes (data:, javascript:, mailto:) don't have meaningful hosts
        # so they only need scheme validation, not host-based allow list checking
        hostless_schemes = {"data", "javascript", "vbscript", "mailto"}
        if parsed_url.scheme in hostless_schemes:
            # For hostless schemes, only scheme permission matters (no allow list needed)
            # They were already validated for scheme permission in _validate_url_security
            allowed.append(url_string)
        elif _is_url_allowed(parsed_url, config.url_allow_list, config.allow_subdomains, url_had_explicit_scheme):
            allowed.append(url_string)
        else:
            blocked.append(url_string)
            blocked_reasons.append(f"{url_string}: Not in allow list")

    return GuardrailResult(
        tripwire_triggered=bool(blocked),
        info={
            "guardrail_name": "URL Filter",
            "config": {
                "allowed_schemes": list(config.allowed_schemes),
                "block_userinfo": config.block_userinfo,
                "allow_subdomains": config.allow_subdomains,
                "url_allow_list": config.url_allow_list,
            },
            "detected": detected_urls,
            "allowed": allowed,
            "blocked": blocked,
            "blocked_reasons": blocked_reasons,
        },
    )


# Register the URL filter
default_spec_registry.register(
    name="URL Filter",
    check_fn=urls,
    description="URL filtering using regex + Pydantic with direct configuration.",
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)
