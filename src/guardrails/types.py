"""Type definitions, Protocols, and result types for Guardrails.

This module provides core types for implementing Guardrails, including:

- The `TokenUsage` dataclass, representing token consumption from LLM-based guardrails.
- The `GuardrailResult` dataclass, representing the outcome of a guardrail check.
- The `CheckFn` Protocol, a callable interface for all guardrail functions.

"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, TypeVar, runtime_checkable

from openai import AsyncOpenAI, OpenAI

try:
    # Available in OpenAI Python SDK when Azure features are installed
    from openai import AsyncAzureOpenAI, AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AsyncAzureOpenAI = object  # type: ignore
    AzureOpenAI = object  # type: ignore
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token usage statistics from an LLM-based guardrail.

    This dataclass encapsulates token consumption data from OpenAI API responses.
    For providers that don't return usage data, the unavailable_reason field
    will contain an explanation.

    Attributes:
        prompt_tokens: Number of tokens in the prompt. None if unavailable.
        completion_tokens: Number of tokens in the completion. None if unavailable.
        total_tokens: Total tokens used. None if unavailable.
        unavailable_reason: Explanation when token usage is not available
            (e.g., third-party models). None when usage data is present.
    """

    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    unavailable_reason: str | None = None


@runtime_checkable
class GuardrailLLMContextProto(Protocol):
    """Protocol for context types providing an OpenAI client.

    Classes implementing this protocol must expose an OpenAI client
    via the `guardrail_llm` attribute. For conversation-aware guardrails
    (like prompt injection detection), they can also access `conversation_history`
    containing the full conversation history.

    Attributes:
        guardrail_llm (AsyncOpenAI | OpenAI): The OpenAI client used by the guardrail.
        conversation_history (list, optional): Full conversation history for conversation-aware guardrails.
    """

    guardrail_llm: AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI

    def get_conversation_history(self) -> list | None:
        """Get conversation history if available, None otherwise."""
        return getattr(self, "conversation_history", None)


@dataclass(frozen=True, slots=True)
class GuardrailResult:
    """Result returned from a guardrail check.

    This dataclass encapsulates the outcome of a guardrail function,
    including whether a tripwire was triggered, execution failure status,
    and any supplementary metadata.

    Attributes:
        tripwire_triggered (bool): True if the guardrail identified a critical failure.
        execution_failed (bool): True if the guardrail failed to execute properly.
        original_exception (Exception | None): The original exception if execution failed.
        info (dict[str, Any]): Additional structured data about the check result,
            such as error details, matched patterns, or diagnostic messages.
            Implementations may include a 'checked_text' field containing the
            processed/validated text when applicable. Defaults to an empty dict.
    """

    tripwire_triggered: bool
    execution_failed: bool = False
    original_exception: Exception | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required fields and consistency."""
        # Ensure consistency: if execution_failed=True, original_exception should be present
        if self.execution_failed and self.original_exception is None:
            raise ValueError("When execution_failed=True, original_exception must be provided")


TContext = TypeVar("TContext")
TIn = TypeVar("TIn")
TCfg = TypeVar("TCfg", bound=BaseModel)
MaybeAwaitableResult = GuardrailResult | Awaitable[GuardrailResult]
CheckFn = Callable[[TContext, TIn, TCfg], MaybeAwaitableResult]
"""Type alias for a guardrail function.

A guardrail function accepts a context object, input data, and a configuration object,
returning either a `GuardrailResult` or an awaitable resolving to `GuardrailResult`.

Args:
    TContext (TypeVar): The context type (often includes resources used by a guardrail).
    TIn (TypeVar): The input data to validate or check.
    TCfg (TypeVar): The configuration type, usually a Pydantic model.
Returns:
    GuardrailResult or Awaitable[GuardrailResult]: The outcome of the guardrail check.
"""


def extract_token_usage(response: Any) -> TokenUsage:
    """Extract token usage from an OpenAI API response.

    Attempts to extract token usage data from the response's `usage` attribute.
    Works with both Chat Completions API and Responses API responses.
    For third-party models or responses without usage data, returns a TokenUsage
    with None values and an explanation in unavailable_reason.

    Args:
        response: An OpenAI API response object (ChatCompletion, Response, etc.)

    Returns:
        TokenUsage: Token usage statistics extracted from the response.
    """
    usage = getattr(response, "usage", None)

    if usage is None:
        return TokenUsage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            unavailable_reason="Token usage not available for this model provider",
        )

    # Extract token counts - handle both attribute access and dict-like access
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    if prompt_tokens is None:
        # Try Responses API format
        prompt_tokens = getattr(usage, "input_tokens", None)

    completion_tokens = getattr(usage, "completion_tokens", None)
    if completion_tokens is None:
        # Try Responses API format
        completion_tokens = getattr(usage, "output_tokens", None)

    total_tokens = getattr(usage, "total_tokens", None)

    # If all values are None, the response has a usage object but no data
    if prompt_tokens is None and completion_tokens is None and total_tokens is None:
        return TokenUsage(
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            unavailable_reason="Token usage data not populated in response",
        )

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        unavailable_reason=None,
    )


def token_usage_to_dict(token_usage: TokenUsage) -> dict[str, Any]:
    """Convert a TokenUsage dataclass to a dictionary for inclusion in info dicts.

    Args:
        token_usage: TokenUsage instance to convert.

    Returns:
        Dictionary representation suitable for GuardrailResult.info.
    """
    result: dict[str, Any] = {
        "prompt_tokens": token_usage.prompt_tokens,
        "completion_tokens": token_usage.completion_tokens,
        "total_tokens": token_usage.total_tokens,
    }
    if token_usage.unavailable_reason is not None:
        result["unavailable_reason"] = token_usage.unavailable_reason
    return result


def aggregate_token_usage_from_infos(
    info_dicts: Iterable[dict[str, Any] | None],
) -> dict[str, Any]:
    """Aggregate token usage from multiple guardrail info dictionaries.

    Args:
        info_dicts: Iterable of guardrail info dicts (each may contain a
            ``token_usage`` entry) or None.

    Returns:
        Dictionary mirroring GuardrailResults.total_token_usage output.
    """
    total_prompt = 0
    total_completion = 0
    total = 0
    has_any_data = False

    for info in info_dicts:
        if not info:
            continue

        usage = info.get("token_usage")
        if usage is None:
            continue

        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total_val = usage.get("total_tokens")

        if prompt is None and completion is None and total_val is None:
            continue

        has_any_data = True
        if prompt is not None:
            total_prompt += prompt
        if completion is not None:
            total_completion += completion
        if total_val is not None:
            total += total_val

    return {
        "prompt_tokens": total_prompt if has_any_data else None,
        "completion_tokens": total_completion if has_any_data else None,
        "total_tokens": total if has_any_data else None,
    }


# Attribute names used by Agents SDK RunResult for guardrail results
_AGENTS_SDK_RESULT_ATTRS = (
    "input_guardrail_results",
    "output_guardrail_results",
    "tool_input_guardrail_results",
    "tool_output_guardrail_results",
)


def total_guardrail_token_usage(result: Any) -> dict[str, Any]:
    """Get aggregated token usage from any guardrails result object.

    This is a unified interface that works across all guardrails surfaces:
    - GuardrailsResponse (from GuardrailsAsyncOpenAI, GuardrailsOpenAI, etc.)
    - GuardrailResults (direct access to organized results)
    - Agents SDK RunResult (from Runner.run with GuardrailAgent)

    Args:
        result: A result object from any guardrails client. Can be:
            - GuardrailsResponse with guardrail_results attribute
            - GuardrailResults with total_token_usage property
            - Agents SDK RunResult with *_guardrail_results attributes

    Returns:
        Dictionary with aggregated token usage:
            - prompt_tokens: Sum of all prompt tokens (or None if no data)
            - completion_tokens: Sum of all completion tokens (or None if no data)
            - total_tokens: Sum of all total tokens (or None if no data)

    Example:
        ```python
        # Works with OpenAI client responses
        response = await client.responses.create(...)
        tokens = total_guardrail_token_usage(response)

        # Works with Agents SDK results
        result = await Runner.run(agent, input)
        tokens = total_guardrail_token_usage(result)

        print(f"Used {tokens['total_tokens']} guardrail tokens")
        ```
    """
    # Check for GuardrailsResponse (has guardrail_results with total_token_usage)
    guardrail_results = getattr(result, "guardrail_results", None)
    if guardrail_results is not None and hasattr(guardrail_results, "total_token_usage"):
        return guardrail_results.total_token_usage

    # Check for GuardrailResults directly (has total_token_usage property/descriptor)
    class_attr = getattr(type(result), "total_token_usage", None)
    if class_attr is not None and hasattr(class_attr, "__get__"):
        return result.total_token_usage

    # Check for Agents SDK RunResult (has *_guardrail_results attributes)
    infos: list[dict[str, Any] | None] = []
    for attr in _AGENTS_SDK_RESULT_ATTRS:
        stage_results = getattr(result, attr, None)
        if stage_results:
            infos.extend(_extract_agents_sdk_infos(stage_results))

    if infos:
        return aggregate_token_usage_from_infos(infos)

    # Fallback: no recognized result type
    return {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
    }


def _extract_agents_sdk_infos(
    stage_results: Iterable[Any],
) -> Iterable[dict[str, Any] | None]:
    """Extract info dicts from Agents SDK guardrail results.

    Args:
        stage_results: List of GuardrailResultResult objects from Agents SDK.

    Yields:
        Info dictionaries containing token_usage data.
    """
    for gr_result in stage_results:
        output = getattr(gr_result, "output", None)
        if output is not None:
            output_info = getattr(output, "output_info", None)
            if isinstance(output_info, dict):
                yield output_info
