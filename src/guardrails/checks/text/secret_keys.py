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
import unicodedata
from typing import Any, Iterator, TypedDict
from urllib.parse import SplitResult, urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["secret_keys"]


class SecretCfg(TypedDict, total=False):
    strict_mode: bool
    min_length: int
    min_diversity: int
    min_entropy: float


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

ALLOWED_EXTENSIONS = (
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

# Only provider-specific credential forms may lift an existing URL/file exemption.
# Generic lexical stems such as api-/key-/token-/secret-/xox remain covered by the
# standalone detector, but are deliberately not sufficient evidence inside an
# otherwise exempt URL or filename.
_EMBEDDED_DIRECT_PREFIXES = (
    "sk-",
    "sk_",
    "ghp_",
    "AKIA",
    "xoxb-",
    "xoxp-",
    "SG.",
    "hf_",
)

_PREFIX_RE = re.compile(
    r"(?:"
    + "|".join(
        re.escape(prefix)
        for prefix in sorted(_EMBEDDED_DIRECT_PREFIXES, key=len, reverse=True)
    )
    + r")"
)
_URL_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_DOTTED_PREFIXES = tuple(prefix for prefix in _EMBEDDED_DIRECT_PREFIXES if prefix.endswith("."))

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
    """Configuration for secret key and credential detection."""

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
    """Calculate the Shannon entropy of a string."""
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return -sum((n := counts[c]) / len(s) * math.log2(n / len(s)) for c in counts)


def _char_diversity(s: str) -> int:
    """Count the number of character types present in a string."""
    return sum(
        (
            any(c.islower() for c in s),
            any(c.isupper() for c in s),
            any(c.isdigit() for c in s),
            any(not c.isalnum() for c in s),
        )
    )


def _contains_allowed_pattern(text: str) -> bool:
    """Return True if text contains allowed URL or file extension patterns."""
    url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    if url_pattern.search(text):
        return True

    ext_pattern = re.compile(
        r"[^\s]+(" + "|".join(re.escape(ext) for ext in ALLOWED_EXTENSIONS) + r")$",
        re.IGNORECASE,
    )
    if ext_pattern.search(text):
        return True

    return False


def _decode_percent_once(text: str) -> str:
    """Decode valid percent escapes exactly once without replacement characters.

    Percent-encoded bytes are collected into contiguous runs and decoded as
    strict UTF-8. Invalid byte sequences remain in their original ``%XX`` form,
    while literal characters are preserved as-is. The decoded result is never
    recursively decoded.

    Args:
        text: Raw URL or file component.

    Returns:
        A one-pass semantic view of the component.
    """
    output: list[str] = []
    index = 0
    text_length = len(text)

    while index < text_length:
        if (
            text[index] != "%"
            or index + 2 >= text_length
            or text[index + 1] not in _HEX_DIGITS
            or text[index + 2] not in _HEX_DIGITS
        ):
            output.append(text[index])
            index += 1
            continue

        encoded_chunks: list[str] = []
        encoded_bytes = bytearray()
        while (
            index + 2 < text_length
            and text[index] == "%"
            and text[index + 1] in _HEX_DIGITS
            and text[index + 2] in _HEX_DIGITS
        ):
            encoded_chunks.append(text[index : index + 3])
            encoded_bytes.append(int(text[index + 1 : index + 3], 16))
            index += 3

        try:
            decoded = bytes(encoded_bytes).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            output.extend(encoded_chunks)
            continue

        output.append(decoded)

    return "".join(output)


def _is_identifier_continuation(char: str) -> bool:
    """Check whether a character continues an identifier-like token.

    Args:
        char: Character immediately preceding a candidate prefix.

    Returns:
        True for Unicode letters/numbers, combining marks, connector
        punctuation, or zero-width join controls.
    """
    if char.isalnum():
        return True
    if char in {"\u200c", "\u200d"}:
        return True
    return unicodedata.category(char) in {"Mn", "Mc", "Me", "Pc"}


def _has_prefix_boundary(text: str, start: int) -> bool:
    """Check whether a built-in prefix starts at a semantic boundary."""
    if start == 0:
        return True
    return not _is_identifier_continuation(text[start - 1])


def _component_has_detectable_prefixed_candidate(component: str, cfg: SecretCfg) -> bool:
    """Check one structural component for a detectable provider-specific prefix."""
    semantic_component = _decode_percent_once(component)
    prefix_matches = [
        match
        for match in _PREFIX_RE.finditer(semantic_component)
        if _has_prefix_boundary(semantic_component, match.start())
    ]
    if not prefix_matches:
        return False

    min_length = cfg.get("min_length", 15)
    min_diversity = cfg.get("min_diversity", 2)
    suffix_class_mask = 0
    suffix_length = 0
    match_starts = {match.start() for match in prefix_matches}

    for index in range(len(semantic_component) - 1, -1, -1):
        char = semantic_component[index]
        suffix_length += 1

        if char.islower():
            suffix_class_mask |= 1
        elif char.isupper():
            suffix_class_mask |= 2
        elif char.isdigit():
            suffix_class_mask |= 4
        else:
            suffix_class_mask |= 8

        if index not in match_starts:
            continue

        if suffix_length >= min_length and suffix_class_mask.bit_count() >= min_diversity:
            return True

    return False


def _iter_query_components(query: str) -> Iterator[str]:
    """Yield query-style names and values as separate components."""
    for field in query.split("&"):
        if not field:
            continue
        name, separator, value = field.partition("=")
        if name:
            yield name
        if separator and value:
            yield value


def _iter_netloc_components(netloc: str) -> Iterator[str]:
    """Yield userinfo and host labels from a raw URL authority."""
    userinfo, separator, host_port = netloc.rpartition("@")
    if separator:
        username, password_separator, password = userinfo.partition(":")
        if username:
            yield username
        if password_separator and password:
            yield password
    else:
        host_port = netloc

    host = host_port
    if host.startswith("["):
        bracket_end = host.find("]")
        if bracket_end >= 0:
            if bracket_end > 1:
                yield host[1:bracket_end]
            port = host[bracket_end + 1 :]
            if port.startswith(":") and len(port) > 1:
                yield port[1:]
            return
    else:
        possible_host, port_separator, possible_port = host.rpartition(":")
        if port_separator and possible_port.isdigit():
            host = possible_host
            if possible_port:
                yield possible_port

    labels = [label for label in host.strip("[]").split(".") if label]
    for index, label in enumerate(labels):
        yield label
        if index + 1 >= len(labels):
            continue
        for prefix in _DOTTED_PREFIXES:
            if label == prefix[:-1]:
                yield f"{label}.{labels[index + 1]}"


def _iter_fragment_components(fragment: str) -> Iterator[str]:
    """Yield components from an opaque URL fragment."""
    if "=" in fragment or "&" in fragment:
        yield from _iter_query_components(fragment)
        return
    yield from (component for component in re.split(r"[\\/]", fragment) if component)


def _iter_parsed_url_components(parsed: SplitResult) -> Iterator[str]:
    """Yield component-local views from a parsed URL."""
    yield from _iter_netloc_components(parsed.netloc)
    yield from (component for component in re.split(r"[\\/]", parsed.path) if component)
    yield from _iter_query_components(parsed.query)
    yield from _iter_fragment_components(parsed.fragment)


def _iter_fallback_components(text: str) -> Iterator[str]:
    """Yield conservative components when URL parsing fails."""
    yield from (component for component in re.split(r"[\\/?#&=@.]", text) if component)


def _iter_candidate_components(text: str) -> Iterator[str]:
    """Yield structural components from an exempt URL or file token."""
    scheme_matches = list(_URL_SCHEME_RE.finditer(text))
    if not scheme_matches:
        yield from (component for component in re.split(r"[\\/]", text) if component)
        return

    first_start = scheme_matches[0].start()
    if first_start:
        yield from _iter_fallback_components(text[:first_start])

    for index, match in enumerate(scheme_matches):
        end = scheme_matches[index + 1].start() if index + 1 < len(scheme_matches) else len(text)
        url_text = text[match.start() : end]
        try:
            parsed = urlsplit(url_text)
        except (ValueError, UnicodeError):
            yield from _iter_fallback_components(url_text)
        else:
            yield from _iter_parsed_url_components(parsed)


def _contains_detectable_prefixed_candidate(text: str, cfg: SecretCfg) -> bool:
    """Check an exempt token for a detectable component-local direct prefix."""
    return any(_component_has_detectable_prefixed_candidate(component, cfg) for component in _iter_candidate_components(text))


def _is_secret_candidate(s: str, cfg: SecretCfg, custom_regex: list[str] | None = None) -> bool:
    """Check if a string is a secret key using the specified criteria."""
    if custom_regex:
        for pattern in custom_regex:
            if re.match(pattern, s):
                return True

    if not cfg.get("strict_mode", False) and _contains_allowed_pattern(s):
        return False

    long_enough = len(s) >= cfg.get("min_length", 15)
    diverse = _char_diversity(s) >= cfg.get("min_diversity", 2)
    if not (long_enough and diverse):
        return False

    if any(s.startswith(prefix) for prefix in COMMON_KEY_PREFIXES):
        return True

    return _entropy(s) >= cfg.get("min_entropy", 3.7)


def _detect_secret_keys(text: str, cfg: SecretCfg, custom_regex: list[str] | None = None) -> GuardrailResult:
    """Detect potential secret keys in text."""
    secrets: list[str] = []
    for raw_word in re.findall(r"\S+", text):
        structural_word = raw_word.replace("*", "")
        word = structural_word.replace("#", "")
        if _is_secret_candidate(word, cfg, custom_regex):
            secrets.append(word)
            continue

        if _contains_allowed_pattern(word) and _contains_detectable_prefixed_candidate(structural_word, cfg):
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
    """Async guardrail function for secret key and credential detection."""
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
