"""Competitor detection guardrail module.

This module provides a guardrail for detecting mentions of competitors in text.
It uses case-insensitive keyword matching against a configurable list of competitor names.

Classes:
    CompetitorCfg: Configuration schema for competitor detection.

Functions:
    competitors: Async guardrail function for competitor detection.

Configuration Parameters:
    `keywords` (list[str]): A list of competitor names to detect. Matching is case-insensitive.

Example:
```python
    >>> config = CompetitorCfg(keywords=["ACME Corp", "Competitor Inc"])
    >>> result = await competitors(None, "We are better than ACME Corp", config)
    >>> result.tripwire_triggered
    True
```
"""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

from .keywords import KeywordCfg, match_keywords

__all__ = ["competitors"]


class CompetitorCfg(KeywordCfg):
    """Configuration schema for competitor detection.

    This Pydantic model is used to specify a list of competitor names that will be
    flagged if detected in the analyzed text. Matching is case-insensitive.

    Attributes:
        keywords (list[str]): List of competitor names to detect. Matching is case-insensitive.
            Example: ["ACME Corp", "Competitor Inc"]
    """

    keywords: list[str] = Field(
        ...,
        min_length=1,
        description="List of competitor names to detect. Matching is case-insensitive.",
    )

    model_config = ConfigDict(extra="forbid")


async def competitors(
    ctx: Any,
    data: str,
    config: CompetitorCfg,
) -> GuardrailResult:
    """Guardrail function to flag competitor mentions in text.

    Checks the provided text for the presence of any competitor names specified
    in the configuration. Returns a `GuardrailResult` indicating whether any
    competitor keyword was found.

    Args:
        ctx (Any): Context object for the guardrail runtime (unused).
        data (str): Text to analyze for competitor mentions.
        config (CompetitorCfg): Configuration specifying competitor keywords.

    Returns:
        GuardrailResult: Result of the keyword match, with metadata describing
            which keywords (if any) were detected.
    """
    _ = ctx

    return match_keywords(data, config, guardrail_name="Competitors")


default_spec_registry.register(
    name="Competitors",
    check_fn=competitors,
    description=("Checks if the model output mentions any competitors from the provided list."),
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)
