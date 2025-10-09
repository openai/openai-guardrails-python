"""User-defined LLM guardrail for custom content moderation.

This module provides a guardrail for implementing custom content checks using
Large Language Models (LLMs). It allows users to define their own system prompts
for content moderation, enabling flexible and domain-specific guardrail enforcement.

Classes:
    UserDefinedConfig: Pydantic configuration schema for user-defined prompts.

Attributes:
    user_defined_llm: Registered guardrail check_fn for user-defined LLM moderation.

Configuration Parameters:
    This guardrail uses the base LLM configuration (see LLMConfig) with these parameters:

    - `model` (str): The name of the LLM model to use (e.g., "gpt-4", "gpt-3.5-turbo")
    - `confidence_threshold` (float): Minimum confidence score (0.0 to 1.0) required to
        trigger the guardrail. Defaults to 0.7.
    - `system_prompt_details` (str): Custom instructions defining the content detection
        criteria. This prompt should clearly specify what content to detect and how
        to evaluate it.

Examples:
```python
    >>> config = UserDefinedConfig(
    ...     model="gpt-4.1-mini",
    ...     confidence_threshold=0.7,
    ...     system_prompt_details="Do not allow any political advocacy."
    ... )
    >>> result = await user_defined_llm(None, "Content to check...", config)
    >>> result.tripwire_triggered
    True
```
"""

import textwrap

from pydantic import ConfigDict, Field

from guardrails.types import CheckFn, GuardrailLLMContextProto

from .llm_base import (
    LLMConfig,
    LLMOutput,
    create_llm_check_fn,
)

__all__ = ["user_defined_llm"]


class UserDefinedConfig(LLMConfig):
    """Configuration schema for user-defined LLM moderation checks.

    Extends the base LLMConfig with a required field for custom prompt details.

    Attributes:
        system_prompt_details (str): Free-form instructions describing content
            moderation requirements. Injected into the LLM system prompt at runtime.
        model (str): The LLM model to use (inherited).
        confidence_threshold (float): Threshold to flag (inherited).
    """

    system_prompt_details: str = Field(
        ...,
        description="Details of the user-defined system prompt to be used for moderation.",
    )

    model_config = ConfigDict(extra="forbid")


SYSTEM_PROMPT = textwrap.dedent(
    """\
    # Instructions

    {system_prompt_details}

    Flag any content that matches the criteria defined above.
    """
).strip()

user_defined_llm: CheckFn[GuardrailLLMContextProto, str, UserDefinedConfig] = create_llm_check_fn(
    name="Custom Prompt Check",
    description=(
        "Runs a user-defined guardrail based on a custom system prompt. Allows for flexible content moderation based on specific requirements."
    ),
    system_prompt=SYSTEM_PROMPT,
    output_model=LLMOutput,
    config_model=UserDefinedConfig,
)
