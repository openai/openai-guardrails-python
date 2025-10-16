"""Type definitions, Protocols, and result types for Guardrails.

This module provides core types for implementing Guardrails, including:

- The `GuardrailResult` dataclass, representing the outcome of a guardrail check.
- The `CheckFn` Protocol, a callable interface for all guardrail functions.

"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
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
