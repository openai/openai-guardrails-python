"""Chat completions with guardrails."""

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..._base_client import GuardrailsBaseClient

# OpenAI safety identifier for tracking guardrails library usage
# Only supported by official OpenAI API (not Azure or local/alternative providers)
_SAFETY_IDENTIFIER = "oai_guardrails"


def _supports_safety_identifier(client: Any) -> bool:
    """Check if the client supports the safety_identifier parameter.

    Only the official OpenAI API supports this parameter.
    Azure OpenAI and local/alternative providers do not.

    Args:
        client: The OpenAI client instance.

    Returns:
        True if safety_identifier should be included, False otherwise.
    """
    # Azure clients don't support it
    client_type = type(client).__name__
    if "Azure" in client_type:
        return False

    # Check if using a custom base_url (local or alternative provider)
    base_url = getattr(client, "base_url", None)
    if base_url is not None:
        base_url_str = str(base_url)
        # Only official OpenAI API endpoints support safety_identifier
        return "api.openai.com" in base_url_str

    # Default OpenAI client (no custom base_url) supports it
    return True


class Chat:
    """Chat completions with guardrails (sync)."""

    def __init__(self, client: GuardrailsBaseClient) -> None:
        """Initialize Chat resource.

        Args:
            client: GuardrailsBaseClient instance with configured guardrails.
        """
        self._client = client

    @property
    def completions(self):
        """Access chat completions API with guardrails.

        Returns:
            ChatCompletions: Chat completions interface with guardrail support.
        """
        return ChatCompletions(self._client)


class AsyncChat:
    """Chat completions with guardrails (async)."""

    def __init__(self, client: GuardrailsBaseClient) -> None:
        """Initialize AsyncChat resource.

        Args:
            client: GuardrailsBaseClient instance with configured guardrails.
        """
        self._client = client

    @property
    def completions(self):
        """Access async chat completions API with guardrails.

        Returns:
            AsyncChatCompletions: Async chat completions with guardrail support.
        """
        return AsyncChatCompletions(self._client)


class ChatCompletions:
    """Chat completions interface with guardrails (sync)."""

    def __init__(self, client: GuardrailsBaseClient) -> None:
        """Initialize ChatCompletions interface.

        Args:
            client: GuardrailsBaseClient instance with configured guardrails.
        """
        self._client = client

    def create(self, messages: list[dict[str, str]], model: str, stream: bool = False, suppress_tripwire: bool = False, **kwargs):
        """Create chat completion with guardrails (synchronous).

        Runs preflight first, then executes input guardrails concurrently with the LLM call.
        """
        normalized_conversation = self._client._normalize_conversation(messages)
        latest_message, _ = self._client._extract_latest_user_message(messages)

        # Preflight first (synchronous wrapper)
        preflight_results = self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=normalized_conversation,
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_messages = self._client._apply_preflight_modifications(messages, preflight_results)

        # Run input guardrails and LLM call concurrently using a thread for the LLM
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Only include safety_identifier for OpenAI clients (not Azure)
            llm_kwargs = {
                "messages": modified_messages,
                "model": model,
                "stream": stream,
                **kwargs,
            }
            if _supports_safety_identifier(self._client._resource_client):
                llm_kwargs["safety_identifier"] = _SAFETY_IDENTIFIER

            llm_future = executor.submit(
                self._client._resource_client.chat.completions.create,
                **llm_kwargs,
            )
            input_results = self._client._run_stage_guardrails(
                "input",
                latest_message,
                conversation_history=normalized_conversation,
                suppress_tripwire=suppress_tripwire,
            )
            llm_response = llm_future.result()

        # Handle streaming vs non-streaming
        if stream:
            return self._client._stream_with_guardrails_sync(
                llm_response,
                preflight_results,
                input_results,
                conversation_history=normalized_conversation,
                suppress_tripwire=suppress_tripwire,
            )
        else:
            return self._client._handle_llm_response(
                llm_response,
                preflight_results,
                input_results,
                conversation_history=normalized_conversation,
                suppress_tripwire=suppress_tripwire,
            )


class AsyncChatCompletions:
    """Async chat completions interface with guardrails."""

    def __init__(self, client):
        """Initialize AsyncChatCompletions interface.

        Args:
            client: GuardrailsBaseClient instance with configured guardrails.
        """
        self._client = client

    async def create(
        self, messages: list[dict[str, str]], model: str, stream: bool = False, suppress_tripwire: bool = False, **kwargs
    ) -> Any | AsyncIterator[Any]:
        """Create chat completion with guardrails."""
        normalized_conversation = self._client._normalize_conversation(messages)
        latest_message, _ = self._client._extract_latest_user_message(messages)

        # Run pre-flight guardrails
        preflight_results = await self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=normalized_conversation,
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_messages = self._client._apply_preflight_modifications(messages, preflight_results)

        # Run input guardrails and LLM call concurrently for both streaming and non-streaming
        input_check = self._client._run_stage_guardrails(
            "input",
            latest_message,
            conversation_history=normalized_conversation,
            suppress_tripwire=suppress_tripwire,
        )
        # Only include safety_identifier for OpenAI clients (not Azure)
        llm_kwargs = {
            "messages": modified_messages,
            "model": model,
            "stream": stream,
            **kwargs,
        }
        if _supports_safety_identifier(self._client._resource_client):
            llm_kwargs["safety_identifier"] = _SAFETY_IDENTIFIER

        llm_call = self._client._resource_client.chat.completions.create(**llm_kwargs)

        input_results, llm_response = await asyncio.gather(input_check, llm_call)

        if stream:
            return self._client._stream_with_guardrails(
                llm_response,
                preflight_results,
                input_results,
                conversation_history=normalized_conversation,
                suppress_tripwire=suppress_tripwire,
            )
        else:
            return await self._client._handle_llm_response(
                llm_response,
                preflight_results,
                input_results,
                conversation_history=normalized_conversation,
                suppress_tripwire=suppress_tripwire,
            )
