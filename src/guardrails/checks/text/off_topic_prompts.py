"""Off Topic Prompts guardrail module.

This module provides a guardrail for ensuring content stays within a specified
business scope or topic domain. It uses an LLM to analyze text against a defined
context to detect off-topic or irrelevant content.

Classes:
    TopicalAlignmentConfig: Configuration schema including business scope details.

Constants:
    SYSTEM_PROMPT: Parameterized LLM prompt for on-topic / off-topic analysis.

Attributes:
    topical_alignment: Registered guardrail check_fn for off topic prompts enforcement.

Configuration Parameters:
    This guardrail uses the base LLM configuration (see LLMConfig) with these parameters:

    - `model` (str): The name of the LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    - `confidence_threshold` (float): Minimum confidence score (0.0 to 1.0) required to
        trigger the guardrail. Defaults to 0.7.
    - `system_prompt_details` (str): Description of the allowed business scope or topic
        domain. This is used to evaluate content relevance.

Examples:
```python
    >>> config = TopicalAlignmentConfig(
    ...     model="gpt-4.1-mini",
    ...     confidence_threshold=0.8,,
    ...     system_prompt_details="Customer support for cloud services."
    ... )
    >>> result = await topical_alignment(None, "My VM won't start...", config)
    >>> result.tripwire_triggered
    False
```
"""

from __future__ import annotations

import textwrap

from pydantic import ConfigDict, Field

from guardrails.types import CheckFn, GuardrailLLMContextProto

from .llm_base import (
    LLMConfig,
    LLMOutput,
    create_llm_check_fn,
)

__all__ = ["topical_alignment"]


class TopicalAlignmentConfig(LLMConfig):
    """Configuration for off topic prompts guardrail.

    Extends LLMConfig with a required business scope for content checks.

    Attributes:
        system_prompt_details (str): Description of the allowed business scope
            or topics. Used to parameterize the LLM's system prompt.
        model (str): The LLM model to use for content checking (inherited).
        confidence_threshold (float): Minimum confidence to trigger (inherited).
    """

    system_prompt_details: str = Field(
        ...,
        description="Description of the allowed business scope or on-topic context.",
    )

    model_config = ConfigDict(extra="forbid")


SYSTEM_PROMPT = textwrap.dedent(
    """
    You are a content analysis system that determines if text stays on topic.

    BUSINESS SCOPE: {system_prompt_details}

    Determine if the text stays within the defined business scope. Flag any content
    that strays from the allowed topics.
    """
).strip()


topical_alignment: CheckFn[GuardrailLLMContextProto, str, TopicalAlignmentConfig] = create_llm_check_fn(
    name="Off Topic Prompts",
    description="Checks that the content stays within the defined business scope.",
    system_prompt=SYSTEM_PROMPT,  # business_scope supplied at runtime
    output_model=LLMOutput,
    config_model=TopicalAlignmentConfig,
)
