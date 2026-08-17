"""Secret key detection guardrail module.

This module provides functions and configuration for detecting potential API keys,
secrets, and credentials in text. It includes entropy and diversity checks, pattern
recognition, and a guardrail check_fn for runtime enforcement. File extensions and
URLs are optionally excluded, and custom detection criteria are supported.

Classes:
    SecretKeysCfg: Pydantic configuration for specifying secret key detection rules.

Functions:
    secret_keys: Async guardrail function for secret key detection.

Configuration Parameters:
    `threshold` (str): Detection sensitivity level. One of:

    - "strict": Most sensitive, may have more false positives
    - "balanced": Default setting, balanced between sensitivity and specificity
    - "permissive": Least sensitive, may have more false negatives

    `custom_regex` (list[str] | None): Optional list of custom regex patterns to check for secrets.
        If provided, these patterns will be used in addition to the default checks.
        Each pattern must be a valid regex string.

Constants:
    COMMON_KEY_PREFIXES: Common prefixes used in secret keys.
    ALLOWED_EXTENSIONS: File extensions to ignore when strict_mode is False.

Examples:
```python
    >>> cfg = SecretKeysCfg(
    ...     threshold="balanced",
    ...     custom_regex=["my-custom-[a-zA-Z0-9]{32}", "internal-[a-zA-Z0-9]{16}-key"]
    ... )
    >>> result = await secret_keys(None, "my-custom-abc123xyz98765", cfg)
    >>> result.tripwire_triggered
    True
```
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator
from heapq import merge
from typing import Any, TypedDict
from urllib.parse import ParseResult, unquote_to_bytes, urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from guardrails.checks.text.urls import (  # noqa: PLC2701
    _ADJACENT_SCHEME_URL_BOUNDARIES,
    _detect_domain_like_url_spans,
    _detect_ip_url_spans,
    _detect_urls,
)
from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["secret_keys"]


class SecretCfg(TypedDict, total=False):
    strict_mode: bool
    min_length: int
    min_diversity: int
    min_entropy: float


# Define common key prefixes
COMMON_KEY_PREFIXES = (
    "key-",
    "sk-",
    "sk_",
    "pk_",
    "pk-",
    "ghp_",
    "AKIA",
    "xox",
    "SG.",
    "hf_",
    "api-",
    "apikey-",
    "token-",
    "secret-",
    "SHA:",
    "Bearer ",
)


_EMBEDDED_KEY_PREFIXES = (
    "sk-",
    "sk_",
    "ghp_",
    "xoxb-",
    "xoxp-",
    "SG.",
    "hf_",
)
_EMBEDDED_OWNER_PREFIXES = (*_EMBEDDED_KEY_PREFIXES, "AKIA")
_AWS_ACCESS_KEY_ID_RE = re.compile(r"AKIA[A-Z0-9]{16}\Z")
_HTTP_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE)
_EXPLICIT_URL_SCHEME_RE = re.compile(
    r"(?:https?|ftp)://|(?:data|javascript|vbscript):",
    re.IGNORECASE,
)
_URL_SHAPED_SCHEME_RE = re.compile(
    r"[a-z][a-z0-9+.-]*:/",
    re.IGNORECASE,
)
_ASCII_SCHEME_START_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
_ASCII_SCHEME_CHARACTERS = _ASCII_SCHEME_START_CHARACTERS | frozenset("0123456789+.-")
_HOSTLESS_URL_SCHEMES = frozenset(("data", "javascript", "mailto", "vbscript"))
_NESTED_URL_VALUE_BOUNDARIES = _ADJACENT_SCHEME_URL_BOUNDARIES | frozenset(("/", "\\", "#", "&", "=", "@"))
_STRUCTURAL_URL_SHAPE_BOUNDARIES = _ADJACENT_SCHEME_URL_BOUNDARIES | frozenset("=")
_MAX_EXEMPT_CONTAINER_LENGTH = 64 * 1024


# Define allowed file extensions
ALLOWED_EXTENSIONS = (
    # Common file extensions
    ".py",
    ".js",
    ".html",
    ".css",
    ".json",
    ".md",
    ".txt",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".conf",
    ".config",
    ".log",
    ".sql",
    ".sh",
    ".bat",
    ".dll",
    ".so",
    ".dylib",
    ".jar",
    ".war",
    ".php",
    ".rb",
    ".go",
    ".rs",
    ".ts",
    ".jsx",
    ".vue",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".fs",
    ".vb",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
)

CONFIGS: dict[str, SecretCfg] = {
    "strict": {
        "min_length": 10,
        "min_entropy": 3.5,
        "min_diversity": 2,
        "strict_mode": True,
    },
    "balanced": {
        "min_length": 15,
        "min_entropy": 3.8,
        "min_diversity": 3,
        "strict_mode": False,
    },
    "permissive": {
        "min_length": 20,
        "min_entropy": 4.0,
        "min_diversity": 3,
        "strict_mode": False,
    },
}


class SecretKeysCfg(BaseModel):
    """Configuration for secret key and credential detection.

    This configuration allows fine-tuning of secret detection sensitivity and
    adding custom patterns for project-specific secrets.

    Attributes:
        threshold (str): Detection sensitivity level. One of:

            - "strict": Most sensitive, may have more false positives
            - "balanced": Default setting, balanced between sensitivity and specificity
            - "permissive": Least sensitive, may have more false negatives

        custom_regex (list[str] | None): Optional list of custom regex patterns to check for secrets.
            If provided, these patterns will be used in addition to the default checks.
            Each pattern must be a valid regex string.
    """

    threshold: str = Field(
        "balanced",
        description="Threshold level to use (strict, balanced, or permissive)",
        pattern="^(strict|balanced|permissive)$",
    )
    custom_regex: list[str] | None = Field(
        None, description="Optional list of custom regex patterns to check for secrets. Each pattern must be a valid regex string."
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("custom_regex")
    def validate_custom_regex(cls, v):
        """Validate that all custom regex patterns are valid."""
        if v is not None:
            for pattern in v:
                if not isinstance(pattern, str):
                    raise ValueError("Each regex pattern must be a string")
                try:
                    re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"Invalid regex pattern '{pattern!r}': {exc}") from exc
        return v


def _entropy(s: str) -> float:
    """Calculate the Shannon entropy of a string.

    Args:
        s (str): The input string.

    Returns:
        float: The Shannon entropy of the string.
    """
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1

    return -sum((n := counts[c]) / len(s) * math.log2(n / len(s)) for c in counts)


def _char_diversity(s: str) -> int:
    """Count the number of character types present in a string.

    Returns the sum of booleans for presence of lowercase, uppercase, digits, and specials.

    Args:
        s (str): Input string.

    Returns:
        int: Number of unique character types in the string (1-4).
    """
    return sum(
        (
            any(c.islower() for c in s),
            any(c.isupper() for c in s),
            any(c.isdigit() for c in s),
            any(not c.isalnum() for c in s),
        )
    )


def _contains_allowed_pattern(text: str) -> bool:
    """Return True if text contains allowed URL or file extension patterns.

    Args:
        text (str): Input string.

    Returns:
        bool: True if text matches URL or allowed extension; otherwise False.
    """
    # Simple regex for URLs
    url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    if url_pattern.search(text):
        return True

    # Regex for allowed file extensions
    # Build a pattern like: ".*\\.(py|js|html|...|png)$"
    ext_pattern = re.compile(
        r"[^\s]+(" + "|".join(re.escape(ext) for ext in ALLOWED_EXTENSIONS) + r")$",
        re.IGNORECASE,
    )
    if ext_pattern.search(text):
        return True

    return False


def _is_secret_candidate(
    s: str,
    cfg: SecretCfg,
    custom_regex: list[str] | None = None,
    *,
    allow_pattern_exemption: bool = True,
) -> bool:
    """Check if a string is a secret key using the specified criteria.

    Skips candidates matching allowed patterns (when strict_mode=False),
    enforces minimum length, character diversity, common prefix, and entropy.
    Also checks against custom patterns if provided.

    Args:
        s (str): String to analyze.
        cfg (SecretCfg): Detection configuration.
        custom_regex (Optional[List[str]]): List of custom regex patterns to check.
        allow_pattern_exemption: Whether URL and file exemptions may suppress
            this candidate.

    Returns:
        bool: True if the string is a secret key; otherwise False.
    """
    # Check custom patterns first if provided
    if custom_regex:
        for pattern in custom_regex:
            if re.match(pattern, s):
                return True

    if allow_pattern_exemption and not cfg.get("strict_mode", False) and _contains_allowed_pattern(s):
        return False

    long_enough = len(s) >= cfg.get("min_length", 15)
    diverse = _char_diversity(s) >= cfg.get("min_diversity", 2)

    if not (long_enough and diverse):
        return False

    if any(s.startswith(prefix) for prefix in COMMON_KEY_PREFIXES):
        return True

    return _entropy(s) >= cfg.get("min_entropy", 3.7)


def _is_supported_embedded_candidate(value: str, cfg: SecretCfg) -> bool:
    """Check a candidate from a supported exempt container.

    Args:
        value: Once-decoded query value or final file basename.
        cfg: Secret-detection thresholds.

    Returns:
        True when the value has a supported provider-specific shape.
    """
    if _AWS_ACCESS_KEY_ID_RE.fullmatch(value):
        return True
    if not value.startswith(_EMBEDDED_KEY_PREFIXES):
        return False
    return _is_secret_candidate(
        value,
        cfg,
        allow_pattern_exemption=False,
    )


def _find_url_shaped_scheme_start(
    value: str,
    boundaries: frozenset[str],
    *,
    preserve_embedded_prefix: bool,
    prefer_explicit_within_generic: bool,
) -> int | None:
    """Find a boundary-aligned URL-shaped scheme in one pass.

    Args:
        value: Text to scan.
        boundaries: Characters that may precede an independent scheme.
        preserve_embedded_prefix: Whether a provider-prefixed value at offset
            zero remains credential data rather than a URL shape.
        prefer_explicit_within_generic: Whether a known explicit scheme inside
            a broader generic scheme-shaped span takes precedence.

    Returns:
        The scheme offset, or ``None`` when no structural scheme exists.
    """
    explicit_start = next(
        (
            match.start()
            for match in _EXPLICIT_URL_SCHEME_RE.finditer(value)
            if (match.start() == 0 or value[match.start() - 1].isspace() or value[match.start() - 1] in boundaries)
            and not (preserve_embedded_prefix and match.start() == 0 and value.startswith(_EMBEDDED_KEY_PREFIXES))
        ),
        None,
    )
    value_length = len(value)
    position = 0
    while position < value_length:
        if value[position] not in _ASCII_SCHEME_START_CHARACTERS:
            position += 1
            continue

        scheme_start = position
        position += 1
        while position < value_length and value[position] in _ASCII_SCHEME_CHARACTERS:
            position += 1
        if position >= value_length or value[position] != ":":
            position += 1
            continue

        scheme = value[scheme_start:position].lower()
        has_path_separator = position + 1 < value_length and value[position + 1] == "/"
        matched_start: int | None = None
        if len(scheme) >= 2 and has_path_separator:
            matched_start = scheme_start
        else:
            for hostless_scheme in _HOSTLESS_URL_SCHEMES:
                hostless_start = position - len(hostless_scheme)
                if hostless_start < scheme_start or value[hostless_start:position].lower() != hostless_scheme:
                    continue
                hostless_preceding_character = value[hostless_start - 1] if hostless_start else ""
                if hostless_start == 0 or hostless_preceding_character.isspace() or hostless_preceding_character in boundaries:
                    matched_start = hostless_start
                    break
        if matched_start is not None:
            matched_preceding_character = value[matched_start - 1] if matched_start else ""
            starts_at_boundary = matched_start == 0 or matched_preceding_character.isspace() or matched_preceding_character in boundaries
            if not starts_at_boundary:
                matched_start = None
        if matched_start is not None:
            if explicit_start is not None and explicit_start < matched_start:
                return explicit_start
            if prefer_explicit_within_generic and explicit_start is not None and explicit_start < position:
                return explicit_start
            if not preserve_embedded_prefix or matched_start > 0 or not value.startswith(_EMBEDDED_KEY_PREFIXES):
                return matched_start
        position += 1
        if has_path_separator:
            while position < value_length and value[position] == "/":
                position += 1
    return explicit_start


def _find_nested_url_start(value: str) -> int | None:
    """Find the first structurally delimited nested URL.

    Args:
        value: Once-decoded query value.

    Returns:
        The nested scheme offset, or ``None`` when every scheme-like
        occurrence is part of credential data.
    """
    nested_start = _find_url_shaped_scheme_start(
        value,
        _NESTED_URL_VALUE_BOUNDARIES,
        preserve_embedded_prefix=True,
        prefer_explicit_within_generic=False,
    )
    domain_spans = _detect_domain_like_url_spans(value)
    ip_spans = _detect_ip_url_spans(value)
    for detected_start, detected_end in merge(domain_spans, ip_spans):
        if nested_start is not None and detected_start < nested_start < detected_end:
            continue

        preceding_character = value[detected_start - 1] if detected_start else ""
        starts_at_boundary = detected_start == 0 or preceding_character.isspace() or preceding_character in _NESTED_URL_VALUE_BOUNDARIES
        detected_url = value[detected_start:detected_end]
        preserves_email_data = preceding_character == "@" and "/" not in detected_url and "\\" not in detected_url
        preserves_owner_prefix = detected_start == 0 and value.startswith(_EMBEDDED_OWNER_PREFIXES)
        if starts_at_boundary and not preserves_email_data and not preserves_owner_prefix:
            if nested_start is None or detected_start < nested_start:
                nested_start = detected_start
    return nested_start


def _find_structural_url_shape(text: str) -> int | None:
    """Find the first token-level URL shape.

    Args:
        text: Raw whitespace-delimited token.

    Returns:
        The earliest independent scheme offset at the token start, after
        presentation punctuation, or after an assignment delimiter. Returns
        ``None`` when scheme-like text is ordinary basename data.
    """
    return _find_url_shaped_scheme_start(
        text,
        _STRUCTURAL_URL_SHAPE_BOUNDARIES,
        preserve_embedded_prefix=False,
        prefer_explicit_within_generic=True,
    )


def _has_supported_url_termination(text: str, start: int, url: str) -> bool:
    """Return whether a detected URL ends at a supported source boundary.

    Args:
        text: Original whitespace-delimited token.
        start: Inclusive source offset of the detected URL.
        url: Cleaned URL returned by the shared detector.

    Returns:
        True when the URL reaches the token end or presentation punctuation.
    """
    end = start + len(url)
    return end == len(text) or text[end].isspace() or text[end] in _ADJACENT_SCHEME_URL_BOUNDARIES


def _get_detected_url_container_start(text: str, start: int, url: str) -> int | None:
    """Return the independent container start for a detected URL.

    Args:
        text: Original whitespace-delimited token.
        start: Inclusive source offset of the detected URL.
        url: Cleaned URL returned by the shared detector.

    Returns:
        The container start, or ``None`` when the URL is embedded in ordinary
        token text.
    """
    if start == 0:
        return 0
    if start == 2 and text.startswith("//") and _EXPLICIT_URL_SCHEME_RE.match(url) is None:
        return 0

    preceding_character = text[start - 1]
    if preceding_character not in _STRUCTURAL_URL_SHAPE_BOUNDARIES:
        return None
    if preceding_character == ":" and _EXPLICIT_URL_SCHEME_RE.match(url) is None:
        return None
    return start


def _iter_exempt_container_candidates(text: str) -> Iterator[str]:
    """Yield candidates from explicitly supported exempt containers.

    HTTP(S) URLs reuse the URL guardrail detector and expose query values only.
    Non-URL file tokens expose only the final basename before an allowed
    extension. URL paths, fragments, userinfo, nested URLs, intermediate path
    segments, and malformed URL recovery are intentionally out of scope.

    Args:
        text: Raw whitespace-delimited token.

    Yields:
        Once-decoded query values or a final file basename.
    """
    container_text = text.strip("*")
    first_scheme_start = _find_structural_url_shape(container_text)
    detected_urls = _detect_urls(container_text, frozenset(("http", "https")))
    located_urls: list[tuple[int, str]] = []
    for detected_url in detected_urls:
        search_start = 0
        while (detected_start := container_text.find(detected_url, search_start)) >= 0:
            located_urls.append((detected_start, detected_url))
            search_start = detected_start + len(detected_url)
    located_urls.sort(key=lambda located_url: (located_url[0], -len(located_url[1])))

    first_detected_container: tuple[int, int, str] | None = None
    for detected_start, detected_url in located_urls:
        container_start = _get_detected_url_container_start(container_text, detected_start, detected_url)
        if container_start is None:
            continue
        if first_detected_container is None or container_start < first_detected_container[0]:
            first_detected_container = (container_start, detected_start, detected_url)

    first_container_start = first_scheme_start
    if first_detected_container is not None and (first_container_start is None or first_detected_container[0] <= first_container_start):
        first_container_start = first_detected_container[0]

    first_url_supports_file_prefix = first_container_start is None
    if first_detected_container is not None and first_detected_container[0] == first_container_start:
        _, detected_start, first_url = first_detected_container
        first_url_supports_file_prefix = _has_supported_url_termination(
            container_text,
            detected_start,
            first_url,
        )
        if first_url_supports_file_prefix and _HTTP_SCHEME_RE.match(first_url) is not None:
            try:
                parsed_first_url = urlparse(first_url)
            except (ValueError, UnicodeError):
                first_url_supports_file_prefix = False
            else:
                first_url_supports_file_prefix = _is_valid_http_url(parsed_first_url)
    elif first_scheme_start is not None:
        urls_at_first_scheme = [
            detected_url
            for detected_start, detected_url in located_urls
            if detected_start == first_scheme_start and _EXPLICIT_URL_SCHEME_RE.match(detected_url) is not None
        ]
        if urls_at_first_scheme:
            first_url = max(urls_at_first_scheme, key=len)
            first_url_supports_file_prefix = _has_supported_url_termination(
                container_text,
                first_scheme_start,
                first_url,
            )
            if first_url_supports_file_prefix and _HTTP_SCHEME_RE.match(first_url) is not None:
                try:
                    parsed_first_url = urlparse(first_url)
                except (ValueError, UnicodeError):
                    first_url_supports_file_prefix = False
                else:
                    first_url_supports_file_prefix = _is_valid_http_url(parsed_first_url)

    for detected_start, detected_url in located_urls:
        if _get_detected_url_container_start(container_text, detected_start, detected_url) is None:
            continue
        if _HTTP_SCHEME_RE.match(detected_url) is None:
            continue
        if not _has_supported_url_termination(container_text, detected_start, detected_url):
            continue
        try:
            parsed_url = urlparse(detected_url)
        except (ValueError, UnicodeError):
            continue
        if not _is_valid_http_url(parsed_url):
            continue

        for value in _iter_query_values(parsed_url.query):
            nested_scheme_start = _find_nested_url_start(value)
            if nested_scheme_start is not None:
                value = value[:nested_scheme_start].rstrip("".join(_NESTED_URL_VALUE_BOUNDARIES)).rstrip()
            if value:
                yield value

    file_text = container_text
    if first_container_start is not None:
        if not first_url_supports_file_prefix:
            return
        prefix = container_text[:first_container_start]
        if not prefix or prefix[-1] not in _ADJACENT_SCHEME_URL_BOUNDARIES:
            return
        file_text = prefix.strip("".join(_ADJACENT_SCHEME_URL_BOUNDARIES))

    file_text = file_text.strip("*#")
    if _URL_SHAPED_SCHEME_RE.search(file_text) is not None:
        return
    lowered = file_text.lower()
    for extension in ALLOWED_EXTENSIONS:
        if not lowered.endswith(extension):
            continue
        raw_stem = file_text[: -len(extension)]
        raw_basename = re.split(r"[\\/]", raw_stem)[-1]
        basename = _decode_percent_encoded_component(
            raw_basename,
            plus_as_space=False,
            literal_wrapper_characters="*#",
        )
        if basename:
            yield basename
        return


def _is_valid_http_url(parsed_url: ParseResult) -> bool:
    """Return whether a parsed HTTP(S) URL has a valid authority.

    Args:
        parsed_url: Parsed URL returned by ``urlparse``.

    Returns:
        True when the URL has a supported scheme, hostname, and port.
    """
    if parsed_url.scheme.lower() not in {"http", "https"}:
        return False
    try:
        hostname = parsed_url.hostname
        _port = parsed_url.port
    except ValueError:
        return False
    return hostname is not None


def _iter_query_values(query: str) -> Iterator[str]:
    """Yield non-empty query values without materializing every field.

    Args:
        query: Raw URL query without the leading question mark.

    Yields:
        Once-decoded values from fields containing a non-empty value.
    """
    field_start = 0
    while field_start <= len(query):
        field_end = query.find("&", field_start)
        if field_end < 0:
            field_end = len(query)

        value_separator = query.find("=", field_start, field_end)
        if value_separator >= 0 and value_separator + 1 < field_end:
            raw_value = query[value_separator + 1 : field_end]
            value = _decode_percent_encoded_component(
                raw_value,
                plus_as_space=True,
                literal_wrapper_characters="*",
            )
            if value is not None:
                yield value

        if field_end == len(query):
            return
        field_start = field_end + 1


def _decode_percent_encoded_component(
    value: str,
    *,
    plus_as_space: bool,
    literal_wrapper_characters: str = "",
) -> str | None:
    """Decode one component only when all percent escapes form valid UTF-8.

    Args:
        value: Raw component text.
        plus_as_space: Whether query-form plus signs represent spaces.
        literal_wrapper_characters: Literal presentation characters to strip
            from the component edges only after raw validation succeeds.

    Returns:
        The decoded component, or ``None`` when an escape or UTF-8 is invalid.
    """
    if re.search(r"%(?![0-9A-Fa-f]{2})", value) is not None:
        return None

    encoded_value = value.replace("+", " ") if plus_as_space else value
    try:
        decoded_value = unquote_to_bytes(encoded_value).decode("utf-8", errors="strict")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return None

    if not literal_wrapper_characters:
        return decoded_value

    normalized_value = value.strip(literal_wrapper_characters)
    normalized_encoded_value = normalized_value.replace("+", " ") if plus_as_space else normalized_value
    return unquote_to_bytes(normalized_encoded_value).decode("utf-8", errors="strict")


def _detect_secret_keys(
    text: str,
    cfg: SecretCfg,
    custom_regex: list[str] | None = None,
) -> GuardrailResult:
    """Detect potential secret keys in text.

    Args:
        text: Input text to scan.
        cfg: Secret-detection thresholds.
        custom_regex: Optional project-specific secret patterns.

    Returns:
        Guardrail result containing any detected secret tokens.
    """
    secrets: list[str] = []
    for raw_word in re.findall(r"\S+", text):
        structural_word = raw_word.replace("*", "")
        word = structural_word.replace("#", "")
        if _is_secret_candidate(word, cfg, custom_regex):
            secrets.append(word)
            continue

        if not _contains_allowed_pattern(word):
            continue
        if len(raw_word) > _MAX_EXEMPT_CONTAINER_LENGTH:
            secrets.append(word)
            continue
        candidates = _iter_exempt_container_candidates(raw_word)
        if any(_is_supported_embedded_candidate(candidate, cfg) for candidate in candidates):
            secrets.append(word)

    return GuardrailResult(
        tripwire_triggered=bool(secrets),
        info={
            "guardrail_name": "Secret Keys",
            "detected_secrets": secrets,
        },
    )


async def secret_keys(
    ctx: Any,
    data: str,
    config: SecretKeysCfg,
) -> GuardrailResult:
    """Async guardrail function for secret key and credential detection.

    Scans the input for likely secrets or credentials (e.g., API keys, tokens)
    using entropy, diversity, and pattern rules.

    Args:
        ctx (Any): Guardrail context (unused).
        data (str): Input text to scan.
        config (SecretKeysCfg): Configuration for secret detection.

    Returns:
        GuardrailResult: Indicates if secrets were detected, with findings in info.
    """
    _ = ctx
    cfg = CONFIGS[config.threshold]
    return _detect_secret_keys(data, cfg, config.custom_regex)


default_spec_registry.register(
    name="Secret Keys",
    check_fn=secret_keys,
    description=("Checks that the text does not contain potential API keys, secrets, or other credentials."),
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)
