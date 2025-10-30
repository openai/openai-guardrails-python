"""OpenAI safety identifier utilities.

This module provides utilities for handling the OpenAI safety_identifier parameter,
which is used to track guardrails library usage for monitoring and abuse detection.

The safety identifier is only supported by the official OpenAI API and should not
be sent to Azure OpenAI or other OpenAI-compatible providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI, AsyncOpenAI, AzureOpenAI, OpenAI
else:
    try:
        from openai import AsyncAzureOpenAI, AzureOpenAI
    except ImportError:
        AsyncAzureOpenAI = None  # type: ignore[assignment, misc]
        AzureOpenAI = None  # type: ignore[assignment, misc]

__all__ = ["SAFETY_IDENTIFIER", "supports_safety_identifier"]

# OpenAI safety identifier for tracking guardrails library usage
SAFETY_IDENTIFIER = "openai-guardrails-python"


def supports_safety_identifier(
    client: AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI | Any,
) -> bool:
    """Check if the client supports the safety_identifier parameter.

    Only the official OpenAI API supports this parameter.
    Azure OpenAI and local/alternative providers (Ollama, vLLM, etc.) do not.

    Args:
        client: The OpenAI client instance to check.

    Returns:
        True if safety_identifier should be included in API calls, False otherwise.

    Examples:
        >>> from openai import AsyncOpenAI
        >>> client = AsyncOpenAI()
        >>> supports_safety_identifier(client)
        True

        >>> from openai import AsyncOpenAI
        >>> local_client = AsyncOpenAI(base_url="http://localhost:11434")
        >>> supports_safety_identifier(local_client)
        False
    """
    # Azure clients don't support it
    if AsyncAzureOpenAI is not None and AzureOpenAI is not None:
        if isinstance(client, AsyncAzureOpenAI | AzureOpenAI):
            return False

    # Check if using a custom base_url (local or alternative provider)
    base_url = getattr(client, "base_url", None)
    if base_url is not None:
        base_url_str = str(base_url)
        # Only official OpenAI API endpoints support safety_identifier
        return "api.openai.com" in base_url_str

    # Default OpenAI client (no custom base_url) supports it
    return True

