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

_PREFIX_RE = re.compile(
    r"(?:"
    + "|".join(
        re.escape(prefix)
        for prefix in sorted(COMMON_KEY_PREFIXES, key=len, reverse=True)
        if not any(char.isspace() for char in prefix)
    )
    + r")"
)
_URL_SCHEME_RE = re.compile(r"https?://", re.IGNORECASE)
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_DOTTED_PREFIXES = tuple(prefix for prefix in COMMON_KEY_PREFIXES if prefix.endswith("."))

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


def _decode_percent_once(text: str) -> str:
    """Decode valid percent escapes exactly once without replacement characters.

    ASCII bytes are decoded individually. Valid multi-byte UTF-8 escape runs are
    decoded strictly; malformed or non-canonical bytes remain in their original
    ``%XX`` form so they cannot manufacture semantic boundaries.

    Args:
        text: Raw URL or file component.

    Returns:
        A one-pass semantic view of the component.
    """
    output: list[str] = []
    index = 0

    while index < len(text):
        if (
            text[index] != "%"
            or index + 2 >= len(text)
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
        elif 0xE0$ÑPÐ€L@