"""Chat completions with guardrails."""

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..._base_client import GuardrailsBaseClient


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
        latest_message, _ = self._client._extract_latest_user_message(messages)

        # Preflight first (synchronous wrapper)
        preflight_results = self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=messages,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_messages = self._client._apply_preflight_modifications(messages, preflight_results)

        # Run input guardrails and LLM call concurrently using a thread for the LLM
        with ThreadPoolExecutor(max_workers=1) as executor:
            llm_future = executor.submit(
                self._client._resource_client.chat.completions.create,
                messages=modified_messages,  # Use messages with any preflight modifications
                model=model,
                stream=stream,
                **kwargs,
            )
            input_results = self._client._run_stage_guardrails(
                "input",
                latest_message,
                conversation_history=messages,  # Pass full conversation for prompt injection detection
                suppress_tripwire=suppress_tripwire,
            )
            llm_response = llm_future.result()

        # Handle streaming vs non-streaming
        if stream:
            return self._client._stream_with_guardrails_sync(
                llm_response,
                preflight_results,
                input_results,
                suppress_tripwire=suppress_tripwire,
            )
        else:
            return self._client._handle_llm_response(
                llm_response,
                preflight_results,
                input_results,
                conversation_history=messages,
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
        latest_message, _ = self._client._extract_latest_user_message(messages)

        # Run pre-flight guardrails
        preflight_results = await self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=messages,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_messages = self._client._apply_preflight_modifications(messages, preflight_results)

        # Run input guardrails and LLM call concurrently for both streaming and non-streaming
        input_check = self._client._run_stage_guardrails(
            "input",
            latest_message,
            conversation_history=messages,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )
        llm_call = self._client._resource_client.chat.completions.create(
            messages=modified_messages,  # Use messages with any preflight modifications
            model=model,
            stream=stream,
            **kwargs,
        )

        input_results, llm_response = await asyncio.gather(input_check, llm_call)

        if stream:
            return self._client._stream_with_guardrails(
                llm_response,
                preflight_results,
                input_results,
                suppress_tripwire=suppress_tripwire,
            )
        else:
            return await self._client._handle_llm_response(
                llm_response,
                preflight_results,
                input_results,
                conversation_history=messages,
                suppress_tripwire=suppress_tripwire,
            )
