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

import functools
import math
import re
from typing import Any, TypedDict

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer
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


@functools.lru_cache(maxsize=1)
def _get_analyzer_engine() -> AnalyzerEngine:
    """Return a singleton, configured Presidio AnalyzerEngine for pattern detection.

    Includes a recognizer for file extensions to allow filtering in non-strict mode.

    Returns:
        AnalyzerEngine: Initialized Presidio analyzer engine.
    """
    engine = AnalyzerEngine()

    # Recognise file extensions so we can filter them out in non‑strict mode.
    pattern = Pattern(
        name="file_extension",
        regex=f"\\S+({'|'.join(re.escape(ext) for ext in ALLOWED_EXTENSIONS)})",
        score=1.0,
    )
    engine.registry.add_recognizer(PatternRecognizer(supported_entity="FILE_EXTENSION", patterns=[pattern]))

    return engine


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


def _is_secret_candidate(s: str, cfg: SecretCfg, custom_regex: list[str] | None = None) -> bool:
    """Check if a string is a secret key using the specified criteria.

    Skips candidates matching allowed patterns (when strict_mode=False),
    enforces minimum length, character diversity, common prefix, and entropy.
    Also checks against custom patterns if provided.

    Args:
        s (str): String to analyze.
        cfg (SecretCfg): Detection configuration.
        custom_regex (Optional[List[str]]): List of custom regex patterns to check.

    Returns:
        bool: True if the string is a secret key; otherwise False.
    """
    # Check custom patterns first if provided
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
    """Detect potential secret keys in text.

    Args:
        text (str): Input text to scan.
        cfg (SecretCfg): Secret detection criteria.
        custom_regex (Optional[List[str]]): List of custom regex patterns to check.

    Returns:
        GuardrailResult: Result containing flag status and detected secrets.
    """
    words = (w.replace("*", "").replace("#", "") for w in re.findall(r"\S+", text))
    secrets = [w for w in words if _is_secret_candidate(w, cfg, custom_regex)]

    return GuardrailResult(
        tripwire_triggered=bool(secrets),
        info={
            "guardrail_name": "Secret Keys",
            "detected_secrets": secrets,
            "checked_text": text,  # Secret key detection doesn't modify text, pass through unchanged
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
