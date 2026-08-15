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

# Only provider-specific token stems may lift URL/file exemptions. Generic lexical
# prefixes such as api-/token-/key- remain part of standalone detection semantics,
# but are intentionally excluded here because natural URL slugs can satisfy length,
# diversity, and even entropy thresholds without containing a credential. AKIA is
# handled with its fixed AWS access-key-ID shape rather than generic diversity.
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
_AWS_ACCESS_KEY_ID_RE = re.compile(r"AKIA[A-Z0-9]{16}")
_AWS_ACCESS_KEY_ID_LENGTH = 20
_URL_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_DOTTED_PREFIXES = tuple(prefix for prefix in _EMBEDDED_DIRECT_PREFIXES if prefix.endswith("."))
_SENTENCE_TRAILING_PUNCTUATION = ".,;:!?"
_PRESENTATION_CLOSERS = {
    "(": ")",
    "[": "]",
    "{": "}",
    "<": ">",
    '"': '"',
    "'": "'",
}

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
        threshold: Detection sensitivity level.
        custom_regex: Optional project-specific secret patterns.
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
        s: Input string.

    Returns:
        Shannon entropy of the string.
    """
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1
    return -sum((n := counts[c]) / len(s) * math.log2(n / len(s)) for c in counts)


def _char_diversity(s: str) -> int:
    """Count the number of character types present in a string.

    Args:
        s: Input string.

    Returns:
        Number of lowercase, uppercase, digit, and special categories present.
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
    """Check whether text matches an existing URL or file exemption.

    Args:
        text: Input token.

    Returns:
        True when the token matches an allowed URL or file-extension pattern.
    """
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

    Each valid UTF-8 sequence is decoded independently. Invalid escapes or byte
    sequences remain in their original ``%XX`` form, but do not prevent later
    independently valid escapes from being decoded. The result is never
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

        first_byte = int(text[index + 1 : index + 3], 16)
        if first_byte < 0x80:
            output.append(chr(first_byte))
            index += 3
            continue

        if 0xC2 <= first_byte <= 0xDF:
            width = 2
        elif 0xE0 <= first_byte <= 0xEF:
            width = 3
        elif 0xF0 <= first_byte <= 0xF4:
            width = 4
        else:
            output.append(text[index : index + 3])
            index += 3
            continue

        encoded_bytes = bytearray([first_byte])
        cursor = index + 3
        for _ in range(width - 1):
            if (
                cursor + 2 >= text_length
                or text[cursor] != "%"
                or text[cursor + 1] not in _HEX_DIGITS
                or text[cursor + 2] not in _HEX_DIGITS
            ):
                break
            encoded_bytes.append(int(text[cursor + 1 : cursor + 3], 16))
            cursor += 3

        if len(encoded_bytes) != width:
            output.append(text[index : index + 3])
            index += 3
            continue

        try:
            decoded = bytes(encoded_bytes).decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            output.append(text[index : index + 3])
            index += 3
            continue

        output.append(decoded)
        index = cursor

    return "".join(output)


def _strip_allowed_extension(component: str) -> str:
    """Remove one literal or once-percent-encoded allowed file extension.

    The returned value stays in its raw representation so the component can
    still receive exactly one normal percent-decoding pass during scoring.

    Args:
        component: Final raw path or filename component.

    Returns:
        The raw filename stem when its semantic one-pass suffix is an allowed
        extension, otherwise the original component.
    """
    for extension in ALLOWED_EXTENSIONS:
        raw_index = len(component)
        for expected in reversed(extension):
            if (
                raw_index >= 3
                and component[raw_index - 3] == "%"
                and component[raw_index - 2] in _HEX_DIGITS
                and component[raw_index - 1] in _HEX_DIGITS
                and chr(int(component[raw_index - 2 : raw_index], 16)).lower() == expected.lower()
            ):
                raw_index -= 3
                continue

            if raw_index >= 1 and component[raw_index - 1].lower() == expected.lower():
                raw_index -= 1
                continue

            break
        else:
            return component[:raw_index]

    return component


def _trim_url_presentation_suffix(url_text: str, leading_context: str) -> str:
    """Remove only confirmed presentation punctuation outside a URL.

    Terminal URI punctuation is valid credential data, so it is preserved unless
    the URL is immediately preceded by a known presentation opener and the span
    contains that opener's matching closer. Sentence punctuation may be removed
    only after such a confirmed closer.

    Args:
        url_text: Raw URL span extending to the whitespace-token boundary or
            the next URL scheme.
        leading_context: Raw text immediately preceding this URL span.

    Returns:
        The URL span with a confirmed outer presentation closer and any sentence
        punctuation after that closer removed; otherwise the original span.
    """
    if not leading_context:
        return url_text

    closer = _PRESENTATION_CLOSERS.get(leading_context[-1])
    if closer is None:
        return url_text

    without_sentence_punctuation = url_text.rstrip(_SENTENCE_TRAILING_PUNCTUATION)
    if not without_sentence_punctuation.endswith(closer):
        return url_text

    return without_sentence_punctuation[: -len(closer)]


def _iter_path_components(path: str, *, strip_final_extension: bool) -> Iterator[str]:
    """Yield literal path components with optional final-extension removal.

    Args:
        path: Raw URL or local-file path.
        strip_final_extension: Whether an allowed extension on the final path
            component should be excluded from secret scoring.

    Yields:
        Non-empty path components in source order.
    """
    components = [component for component in re.split(r"[\\/]", path) if component]
    for index, component in enumerate(components):
        if strip_final_extension and index == len(components) - 1:
            component = _strip_allowed_extension(component)
        if component:
            yield component


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
    """Check whether a built-in prefix starts at a semantic boundary.

    Args:
        text: Semantic component being scanned.
        start: Start offset of the candidate prefix.

    Returns:
        True when the prefix is not embedded in an identifier-like token.
    """
    if start == 0:
        return True
    return not _is_identifier_continuation(text[start - 1])


def _is_aws_access_key_id(value: str) -> bool:
    """Check whether a value is a complete long-term AWS access-key ID.

    Args:
        value: Candidate value.

    Returns:
        True for exactly ``AKIA`` followed by 16 uppercase letters or digits.
    """
    return _AWS_ACCESS_KEY_ID_RE.fullmatch(value) is not None


def _has_embedded_aws_access_key_id(text: str, start: int) -> bool:
    """Check for an AWS access-key ID at a component offset.

    Args:
        text: Once-decoded component text.
        start: Offset where the ``AKIA`` prefix begins.

    Returns:
        True when the next 20 characters form an AWS access-key ID and the
        identifier does not continue with another identifier character.
    """
    end = start + _AWS_ACCESS_KEY_ID_LENGTH
    if end > len(text) or not _is_aws_access_key_id(text[start:end]):
        return False
    return end == len(text) or not _is_identifier_continuation(text[end])


def _component_has_detectable_prefixed_candidate(component: str, cfg: SecretCfg) -> bool:
    """Check one component for a provider-specific embedded credential.

    Args:
        component: Raw URL or file component.
        cfg: Secret-detection thresholds for the active mode.

    Returns:
        True when a provider-specific candidate satisfies its format-specific
        rule or the existing length/diversity policy.
    """
    semantic_component = _decode_percent_once(component)
    prefix_matches = {
        match.start(): match.group(0)
        for match in _PREFIX_RE.finditer(semantic_component)
        if _has_prefix_boundary(semantic_component, match.start())
    }
    if not prefix_matches:
        return False

    min_length = cfg.get("min_length", 15)
    min_diversity = cfg.get("min_diversity", 2)
    suffix_class_mask = 0
    suffix_length = 0

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

        matched_prefix = prefix_matches.get(index)
        if matched_prefix is None:
            continue

        if matched_prefix == "AKIA":
            if _has_embedded_aws_access_key_id(semantic_component, index):
                return True
            continue

        if suffix_length >= min_length and suffix_class_mask.bit_count() >= min_diversity:
            return True

    return False


def _iter_query_components(query: str) -> Iterator[str]:
    """Yield query-style names and values as separate components.

    Args:
        query: Raw query or query-shaped fragment text.

    Yields:
        Non-empty query names and values in source order.
    """
    for field in query.split("&"):
        if not field:
            continue
        name, separator, value = field.partition("=")
        if name:
            yield name
        if separator and value:
            yield value


def _iter_netloc_components(netloc: str) -> Iterator[str]:
    """Yield userinfo and host labels from a raw URL authority.

    Args:
        netloc: Raw URL authority component.

    Yields:
        Userinfo fields, port text, host labels, and dotted-prefix host pairs.
    """
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
    """Yield component-local views from an opaque URL fragment.

    Args:
        fragment: Raw fragment text.

    Yields:
        Query-style fields or slash-delimited fragment components.
    """
    if "=" in fragment or "&" in fragment:
        yield from _iter_query_components(fragment)
        return
    yield from _iter_path_components(fragment, strip_final_extension=False)


def _iter_parsed_url_components(parsed: SplitResult) -> Iterator[str]:
    """Yield component-local views from a parsed URL.

    Args:
        parsed: Result returned by ``urlsplit``.

    Yields:
        Authority, path, query, and fragment components.
    """
    yield from _iter_netloc_components(parsed.netloc)
    yield from _iter_path_components(parsed.path, strip_final_extension=True)
    yield from _iter_query_components(parsed.query)
    yield from _iter_fragment_components(parsed.fragment)


def _iter_fallback_components(text: str) -> Iterator[str]:
    """Yield conservative components when URL parsing fails.

    Args:
        text: Raw malformed URL text.

    Yields:
        Non-empty pieces separated by structural URL delimiters.
    """
    yield from (component for component in re.split(r"[\\/?#&=@.]", text) if component)


def _iter_candidate_components(text: str) -> Iterator[str]:
    """Yield structural components from an exempt URL or file token.

    Args:
        text: Raw whitespace-delimited token.

    Yields:
        URL or file components eligible for provider-prefix checks.
    """
    scheme_matches = list(_URL_SCHEME_RE.finditer(text))
    if not scheme_matches:
        yield from _iter_path_components(text, strip_final_extension=True)
        return

    first_start = scheme_matches[0].start()
    if first_start:
        yield from _iter_fallback_components(text[:first_start])

    for index, match in enumerate(scheme_matches):
        end = scheme_matches[index + 1].start() if index + 1 < len(scheme_matches) else len(text)
        leading_context = text[: match.start()]
        url_text = _trim_url_presentation_suffix(text[match.start() : end], leading_context)
        if not url_text:
            continue
        try:
            parsed = urlsplit(url_text)
        except (ValueError, UnicodeError):
            yield from _iter_fallback_components(url_text)
        else:
            yield from _iter_parsed_url_components(parsed)


def _contains_detectable_prefixed_candidate(text: str, cfg: SecretCfg) -> bool:
    """Check an exempt token for provider-specific embedded credentials.

    Args:
        text: Raw token that matched a URL or allowed-file exemption.
        cfg: Secret-detection thresholds for the active mode.

    Returns:
        True when any structural component contains a provider-specific
        candidate satisfying its format-specific or length/diversity policy.
    """
    return any(_component_has_detectable_prefixed_candidate(component, cfg) for component in _iter_candidate_components(text))


def _is_secret_candidate(s: str, cfg: SecretCfg, custom_regex: list[str] | None = None) -> bool:
    """Check whether a string satisfies the existing secret policy.

    Args:
        s: Candidate string.
        cfg: Secret-detection thresholds.
        custom_regex: Optional project-specific secret patterns.

    Returns:
        True when the candidate is considered a secret.
    """
    if custom_regex:
        for pattern in custom_regex:
            if re.match(pattern, s):
                return True

    if not cfg.get("strict_mode", False) and _contains_allowed_pattern(s):
        return False

    if _is_aws_access_key_id(s):
        return True

    long_enough = len(s) >= cfg.get("min_length", 15)
    diverse = _char_diversity(s) >= cfg.get("min_diversity", 2)
    if not (long_enough and diverse):
        return False

    if any(s.startswith(prefix) for prefix in COMMON_KEY_PREFIXES):
        return True

    return _entropy(s) >= cfg.get("min_entropy", 3.7)


def _detect_secret_keys(text: str, cfg: SecretCfg, custom_regex: list[str] | None = None) -> GuardrailResult:
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
    """Run secret key and credential detection.

    Args:
        ctx: Guardrail context.
        data: Input text to scan.
        config: Secret Keys guardrail configuration.

    Returns:
        Guardrail result containing any detected secret tokens.
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
