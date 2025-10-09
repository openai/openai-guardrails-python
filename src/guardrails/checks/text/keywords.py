"""Keyword-based guardrail for detecting banned terms in text.

This module provides a guardrail for detecting specific keywords or phrases in text.
It uses case-insensitive matching against a configurable list of keywords.

Classes:
    KeywordCfg: Pydantic config model for specifying banned keywords.

Functions:
    match_keywords: Match forbidden keywords in a given text sample.
    keywords: Guardrail check_fn for detecting banned terms in user input.

Configuration Parameters:
    This guardrail uses the following configuration parameters:

    - `keywords` (list[str]): A list of keywords or phrases to detect. Matching is case-insensitive.
        Example: ["confidential", "internal use only", "do not share"]

Example:
```python
    >>> config = KeywordCfg(keywords=["confidential", "internal use only"])
    >>> result = await match_keywords(None, "This is confidential information", config)
    >>> result.tripwire_triggered
    True
```
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["KeywordCfg", "keywords", "match_keywords"]


class KeywordCfg(BaseModel):
    """Configuration schema for banned keyword matching.

    This Pydantic model is used to specify a list of keywords that will be
    flagged if detected in the analyzed text.

    Attributes:
        keywords (list[str]): List of forbidden keywords to flag if found.
    """

    keywords: list[str] = Field(
        ...,
        min_length=1,
        description="Banned keywords to match in text.",
    )

    model_config = ConfigDict(extra="forbid")


# TODO: Use AhoCorasick algorithm instead
@lru_cache(maxsize=256)
def _compile_pattern(keywords: tuple[str, ...]) -> re.Pattern[str]:
    """Compile and cache a case‐insensitive regex.

    The regex matches any keyword as a separate word
    (i.e. "foo" won't match "food", but will match "Foo").

    Args:
        keywords (tuple[str, ...]): Tuple of keywords to build the pattern.

    Returns:
        re.Pattern[str]: Compiled regex pattern to match any given keyword.
    """
    escaped = (re.escape(k) for k in keywords)
    pattern_text = r"\b(?:" + "|".join(escaped) + r")\b"

    return re.compile(pattern_text, re.IGNORECASE)


def match_keywords(
    data: str,
    config: KeywordCfg,
    guardrail_name: str,
) -> GuardrailResult:
    """Detect banned keywords in the provided text.

    Matches text against the configured keywords and returns a `GuardrailResult`
    indicating whether any forbidden keyword was found.

    Args:
        data (str): Input text to analyze.
        config (KeywordCfg): Configuration specifying which keywords to detect.
        guardrail_name (str): Name of the guardrail for result metadata.

    Returns:
        GuardrailResult: Result containing match details and status.
    """
    # Sanitize keywords by stripping trailing punctuation
    sanitized_keywords = [re.sub(r"[.,!?;:]+$", "", keyword) for keyword in config.keywords]

    pat = _compile_pattern(tuple(sorted(sanitized_keywords)))
    matches = [m.group(0) for m in pat.finditer(data)]
    seen: set[str] = set()
    # Take unique matches, ignoring case
    unique = []
    for m in matches:
        if m.lower() not in seen:
            unique.append(m)
            seen.add(m.lower())
    return GuardrailResult(
        tripwire_triggered=bool(unique),
        info={
            "guardrail_name": guardrail_name,
            "matched": unique,
            "checked": config.keywords,
            "sanitized_keywords": sanitized_keywords,
            "checked_text": data,  # Keyword filtering doesn't modify text, pass through unchanged
        },
    )


async def keywords(
    ctx: Any,
    data: str,
    config: KeywordCfg,
) -> GuardrailResult:
    """Guardrail function to check for banned keywords in user text.

    This is the main check_fn for keyword-based moderation guardrails.
    It flags the input if any forbidden keyword is found.

    Args:
        ctx (Any): Context object (not used in this implementation).
        data (str): Input text to validate.
        config (KeywordCfg): Configuration with list of banned keywords.

    Returns:
        GuardrailResult: Indicates whether any banned keyword was detected.
    """
    _ = ctx

    return match_keywords(data, config, guardrail_name="Keyword Filter")


default_spec_registry.register(
    name="Keyword Filter",
    check_fn=keywords,
    description="Triggers when any keyword appears in text.",
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)
