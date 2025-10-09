"""Moderation guardrail for text content using OpenAI's moderation API.

This module provides a guardrail for detecting harmful or policy-violating content
using OpenAI's moderation API. It supports filtering by specific content categories
and provides detailed analysis of detected violations.

Classes:
    ModerationCfg: Configuration schema for moderation categories.
    Category: Enum of supported moderation categories.

Functions:
    moderation: Async guardrail function for content moderation.

Configuration Parameters:
    `categories` (list[Category]): List of moderation categories to check.

    Available categories listed below. If not specified, all categories are checked by default.

Example:
```python
    >>> cfg = ModerationCfg(categories=["hate", "harassment", "self-harm"])
    >>> result = await moderation(None, "harmful content here", cfg)
    >>> result.tripwire_triggered
    True
```
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import cache
from typing import Any

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

from ..._openai_utils import prepare_openai_kwargs

logger = logging.getLogger(__name__)

__all__ = ["moderation", "Category", "ModerationCfg"]


class Category(str, Enum):
    """Enumeration of supported moderation categories.

    These categories correspond to types of harmful or restricted content
    recognized by the OpenAI moderation endpoint.

    Members:

        sexual: Sexually explicit or suggestive content.
        sexual/minors: Sexual content that includes individuals under the age of 18.
        hate: Hateful or discriminatory language.
        hate/threatening: Hateful content that also includes violence or serious harm.
        harassment: Content involving harassment or bullying.
        harassment/threatening: Harassment content that also includes violence or serious harm.
        self/harm: Content promoting or depicting self-harm.
        self/harm/intent: Content where the speaker expresses intent to harm oneself.
        self/harm/instructions: Content that provides instructions for self-harm or encourages self-harm.
        violence: Content that depicts death, violence, or physical injury.
        violence/graphic: Content that depicts death, violence, or physical injury in graphic detail.
        illicit: Content that gives advice or instruction on how to commit illicit acts (e.g., "how to shoplift").
        illicit/violent: Illicit content but also includes references to violence or procuring a weapon.
    """  # noqa: E501

    SEXUAL = "sexual"
    SEXUAL_MINORS = "sexual/minors"
    HATE = "hate"
    HATE_THREATENING = "hate/threatening"
    HARASSMENT = "harassment"
    HARASSMENT_THREATENING = "harassment/threatening"
    SELF_HARM = "self-harm"
    SELF_HARM_INTENT = "self-harm/intent"
    SELF_HARM_INSTRUCTIONS = "self-harm/instructions"
    VIOLENCE = "violence"
    VIOLENCE_GRAPHIC = "violence/graphic"
    ILLICIT = "illicit"
    ILLICIT_VIOLENT = "illicit/violent"


class ModerationCfg(BaseModel):
    """Configuration schema for the moderation guardrail.

    This configuration allows selection of specific moderation categories to check.
    If no categories are specified, all supported categories will be checked.

    Attributes:
        categories (list[Category]): List of moderation categories to check.

            Available categories:

            - "hate": Hate speech and discriminatory content
            - "harassment": Harassing or bullying content
            - "self-harm": Content promoting self-harm
            - "violence": Violent content
            - "sexual": Sexual content
            - "sexual/minors": Sexual content involving minors
            - "hate/threatening": Threatening hate speech
            - "harassment/threatening": Threatening harassment
            - "self-harm/intent": Content expressing self-harm intent
            - "self-harm/instructions": Instructions for self-harm
            - "violence/graphic": Graphic violent content
            - "illicit": Illegal activities
            - "illicit/violent": Violent illegal activities

            Defaults to all supported categories if not specified.
    """

    categories: list[Category] = Field(
        default_factory=lambda: list(Category),
        description="Moderation categories to check. Defaults to all categories if not specified.",
    )

    model_config = ConfigDict(extra="forbid")


@cache
def _get_moderation_client() -> AsyncOpenAI:
    """Return a cached instance of the OpenAI async client.

    This prevents redundant client instantiation across multiple checks.

    Returns:
        AsyncOpenAI: Cached OpenAI API client for moderation checks.
    """
    return AsyncOpenAI(**prepare_openai_kwargs({}))


async def moderation(
    ctx: Any,
    data: str,
    config: ModerationCfg,
) -> GuardrailResult:
    """Guardrail check_fn to flag disallowed content categories using OpenAI moderation API.

    Calls the OpenAI moderation endpoint on input text and flags if any of the
    configured categories are detected. Returns a result containing flagged
    categories, details, and tripwire status.

    Args:
        ctx (GuardrailLLMContextProto): Guardrail runtime context (unused).
        data (str): User or model text to analyze.
        config (ModerationCfg): Moderation config specifying categories to flag.

    Returns:
        GuardrailResult: Indicates if tripwire was triggered, and details of flagged categories.
    """

    # Prefer reusing an existing OpenAI client from context ONLY if it targets the
    # official OpenAI API. If it's any other provider (e.g., Ollama via base_url),
    # fall back to the default OpenAI moderation client.
    def _maybe_reuse_openai_client_from_ctx(context: Any) -> AsyncOpenAI | None:
        try:
            candidate = getattr(context, "guardrail_llm", None)
            if not isinstance(candidate, AsyncOpenAI):
                return None

            # Attempt to discover the effective base URL in a best-effort way
            base_url = getattr(candidate, "base_url", None)
            if base_url is None:
                inner = getattr(candidate, "_client", None)
                base_url = getattr(inner, "base_url", None) or getattr(inner, "_base_url", None)

            # Reuse only when clearly the official OpenAI endpoint
            if base_url is None:
                return candidate
            if isinstance(base_url, str) and "api.openai.com" in base_url:
                return candidate
            return None
        except Exception:
            return None

    client = _maybe_reuse_openai_client_from_ctx(ctx) or _get_moderation_client()
    resp = await client.moderations.create(
        model="omni-moderation-latest",
        input=data,
    )
    results = resp.results or []
    if not results:
        return GuardrailResult(
            tripwire_triggered=False,
            info={"error": "No moderation results returned"},
        )

    outcome = results[0].model_dump()
    categories = outcome["categories"]

    # Check only the categories specified in config and collect results
    flagged_categories = []
    category_details = {}

    for cat in config.categories:
        cat_value = cat.value
        if categories.get(cat_value, False):
            flagged_categories.append(cat_value)
        category_details[cat_value] = categories.get(cat_value, False)

    # Only trigger if the requested categories are flagged
    is_flagged = bool(flagged_categories)

    return GuardrailResult(
        tripwire_triggered=is_flagged,
        info={
            "guardrail_name": "Moderation",
            "flagged_categories": flagged_categories,
            "categories_checked": config.categories,
            "category_details": category_details,
            "checked_text": data,  # Moderation doesn't modify text, pass through unchanged
        },
    )


default_spec_registry.register(
    name="Moderation",
    check_fn=moderation,
    description="Flags text containing disallowed content categories.",
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="API"),
)
