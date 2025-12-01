"""Context management using Python ContextVars for guardrails.

This module provides a clean way to manage guardrail execution context
using Python's built-in ContextVars, which automatically propagate through
async/await boundaries and execution contexts.
"""

from contextvars import ContextVar
from dataclasses import dataclass

from openai import AsyncOpenAI, OpenAI

try:
    from openai import AsyncAzureOpenAI, AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    AsyncAzureOpenAI = object  # type: ignore
    AzureOpenAI = object  # type: ignore

# Main context variable for guardrails
CTX = ContextVar("guardrails_context", default=None)


@dataclass(frozen=True, slots=True)
class GuardrailsContext:
    """Context for guardrail execution.

    This dataclass defines the resources and configuration needed
    for guardrail execution, including the LLM client to use.

    The guardrail_llm can be either:
    - AsyncOpenAI: For async guardrail execution
    - OpenAI: For sync guardrail execution

    Both client types work seamlessly with the guardrails system.
    """

    guardrail_llm: AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI
    # Add other context fields as needed
    # user_id: str
    # session_data: dict
    # etc.


def set_context(context: GuardrailsContext) -> None:
    """Set the guardrails context for the current execution context.

    Args:
        context: The context object containing guardrail resources
    """
    CTX.set(context)


def get_context() -> GuardrailsContext | None:
    """Get the current guardrails context.

    Returns:
        The current context if set, None otherwise
    """
    return CTX.get()


def has_context() -> bool:
    """Check if a guardrails context is currently set.

    Returns:
        True if context is set, False otherwise
    """
    return CTX.get() is not None


def clear_context() -> None:
    """Clear the current guardrails context."""
    CTX.set(None)
