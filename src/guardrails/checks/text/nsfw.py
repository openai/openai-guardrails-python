"""NSFW content detection guardrail module.

This module provides a guardrail for detecting Not Safe For Work (NSFW) content in text,
including explicit sexual content, adult themes, profanity,and inappropriate material. It uses an
LLM to analyze text for various categories of NSFW content.

Performance Metrics:
    For detailed performance benchmarks and evaluation results, see our
    [benchmarking page](/benchmarking/nsfw/).

Constants:
    SYSTEM_PROMPT: Detailed LLM prompt guiding NSFW detection and classification.

Attributes:
    nsfw_content: Registered guardrail check_fn for NSFW text content detection.

Configuration Parameters:
    This guardrail uses the base LLM configuration (see LLMConfig) with these parameters:

    - `model` (str): The name of the LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    - `confidence_threshold` (float): Minimum confidence score (0.0 to 1.0) required to
        trigger the guardrail. Defaults to 0.7.

Examples:
```python
    >>> config = LLMConfig(
    ...     model="gpt-4.1-mini",
    ...     confidence_threshold=0.8
    ... )
    >>> result = await nsfw_content(None, "Explicit adult content...", config)
    >>> result.tripwire_triggered
    True
```
"""

from __future__ import annotations

import textwrap

from guardrails.types import CheckFn, GuardrailLLMContextProto

from .llm_base import (
    LLMConfig,
    LLMOutput,
    create_llm_check_fn,
)

__all__ = ["nsfw_content"]


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a content moderation assistant.
    Your task is to analyze text and determine if it contains NSFW (Not Safe For Work) content.

    Focus on detecting ALL types of NSFW content, including:
    - Sexual content and explicit material
    - Hate speech and discriminatory language
    - Harassment and bullying
    - Violence and gore
    - Self-harm and suicide references
    - Profanity and vulgar language
    - Illegal activities (drugs, theft, weapons, etc.)
    - Adult themes and mature content
    - Inappropriate workplace content
    - Extremist or radical content
    - Exploitation or abuse
    - Graphic medical content
    - Other potentially offensive or inappropriate content

    Flag any content that matches these criteria.
    """,
).strip()


nsfw_content: CheckFn[GuardrailLLMContextProto, str, LLMConfig] = create_llm_check_fn(
    name="NSFW Text",
    description=(
        "Detects NSFW (Not Safe For Work) content in text, including sexual content, "
        "hate speech, violence, profanity, illegal activities, and other inappropriate material."
    ),
    system_prompt=SYSTEM_PROMPT,
    output_model=LLMOutput,
    config_model=LLMConfig,
)
