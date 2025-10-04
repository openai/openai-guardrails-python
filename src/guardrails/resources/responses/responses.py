"""Responses API with guardrails."""

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel

from ..._base_client import GuardrailsBaseClient


class Responses:
    """Responses API with guardrails (sync)."""

    def __init__(self, client: GuardrailsBaseClient) -> None:
        """Initialize Responses resource.

        Args:
            client: GuardrailsBaseClient instance with configured guardrails.
        """
        self._client = client

    def create(
        self,
        input: str | list[dict[str, str]],
        model: str,
        stream: bool = False,
        tools: list[dict] | None = None,
        suppress_tripwire: bool = False,
        **kwargs,
    ):
        """Create response with guardrails (synchronous).

        Runs preflight first, then executes input guardrails concurrently with the LLM call.
        """
        # Determine latest user message text when a list of messages is provided
        if isinstance(input, list):
            latest_message, _ = self._client._extract_latest_user_message(input)
        else:
            latest_message = input

        # Preflight first (run checks on the latest user message text, with full conversation)
        preflight_results = self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=input,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_input = self._client._apply_preflight_modifications(input, preflight_results)

        # Input guardrails and LLM call concurrently
        with ThreadPoolExecutor(max_workers=1) as executor:
            llm_future = executor.submit(
                self._client._resource_client.responses.create,
                input=modified_input,  # Use preflight-modified input
                model=model,
                stream=stream,
                tools=tools,
                **kwargs,
            )
            input_results = self._client._run_stage_guardrails(
                "input",
                latest_message,
                conversation_history=input,  # Pass full conversation for prompt injection detection
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
                conversation_history=input,
                suppress_tripwire=suppress_tripwire,
            )

    def parse(self, input: list[dict[str, str]], model: str, text_format: type[BaseModel], suppress_tripwire: bool = False, **kwargs):
        """Parse response with structured output and guardrails (synchronous)."""
        latest_message, _ = self._client._extract_latest_user_message(input)

        # Preflight first
        preflight_results = self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=input,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_input = self._client._apply_preflight_modifications(input, preflight_results)

        # Input guardrails and LLM call concurrently
        with ThreadPoolExecutor(max_workers=1) as executor:
            llm_future = executor.submit(
                self._client._resource_client.responses.parse,
                input=modified_input,  # Use modified input with preflight changes
                model=model,
                text_format=text_format,
                **kwargs,
            )
            input_results = self._client._run_stage_guardrails(
                "input",
                latest_message,
                conversation_history=input,  # Pass full conversation for prompt injection detection
                suppress_tripwire=suppress_tripwire,
            )
            llm_response = llm_future.result()

        return self._client._handle_llm_response(
            llm_response,
            preflight_results,
            input_results,
            conversation_data=input,
            suppress_tripwire=suppress_tripwire,
        )

    def retrieve(self, response_id: str, suppress_tripwire: bool = False, **kwargs):
        """Retrieve response with output guardrail validation (synchronous)."""
        # Get the response using the original OpenAI client
        response = self._client._resource_client.responses.retrieve(response_id, **kwargs)

        # Run output guardrails on the retrieved content
        output_text = response.output_text if hasattr(response, "output_text") else ""
        output_results = self._client._run_stage_guardrails("output", output_text, suppress_tripwire=suppress_tripwire)

        # Return wrapped response with guardrail results
        return self._client._create_guardrails_response(
            response,
            [],
            [],
            output_results,  # preflight  # input
        )


class AsyncResponses:
    """Responses API with guardrails (async)."""

    def __init__(self, client):
        """Initialize AsyncResponses resource.

        Args:
            client: GuardrailsBaseClient instance with configured guardrails.
        """
        self._client = client

    async def create(
        self,
        input: str | list[dict[str, str]],
        model: str,
        stream: bool = False,
        tools: list[dict] | None = None,
        suppress_tripwire: bool = False,
        **kwargs,
    ) -> Any | AsyncIterator[Any]:
        """Create response with guardrails."""
        # Determine latest user message text when a list of messages is provided
        if isinstance(input, list):
            latest_message, _ = self._client._extract_latest_user_message(input)
        else:
            latest_message = input

        # Run pre-flight guardrails (on latest user message text, with full conversation)
        preflight_results = await self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=input,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_input = self._client._apply_preflight_modifications(input, preflight_results)

        # Run input guardrails and LLM call in parallel
        input_check = self._client._run_stage_guardrails(
            "input",
            latest_message,
            conversation_history=input,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )
        llm_call = self._client._resource_client.responses.create(
            input=modified_input,  # Use preflight-modified input
            model=model,
            stream=stream,
            tools=tools,
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
                conversation_history=input,
                suppress_tripwire=suppress_tripwire,
            )

    async def parse(
        self, input: list[dict[str, str]], model: str, text_format: type[BaseModel], stream: bool = False, suppress_tripwire: bool = False, **kwargs
    ) -> Any | AsyncIterator[Any]:
        """Parse response with structured output and guardrails."""
        latest_message, _ = self._client._extract_latest_user_message(input)

        # Run pre-flight guardrails
        preflight_results = await self._client._run_stage_guardrails(
            "pre_flight",
            latest_message,
            conversation_history=input,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )

        # Apply pre-flight modifications (PII masking, etc.)
        modified_input = self._client._apply_preflight_modifications(input, preflight_results)

        # Run input guardrails and LLM call in parallel
        input_check = self._client._run_stage_guardrails(
            "input",
            latest_message,
            conversation_history=input,  # Pass full conversation for prompt injection detection
            suppress_tripwire=suppress_tripwire,
        )
        llm_call = self._client._resource_client.responses.parse(
            input=modified_input,  # Use modified input with preflight changes
            model=model,
            text_format=text_format,
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
                conversation_history=input,
                suppress_tripwire=suppress_tripwire,
            )

    async def retrieve(self, response_id: str, suppress_tripwire: bool = False, **kwargs):
        """Retrieve response with output guardrail validation."""
        # Get the response using the original OpenAI client
        response = await self._client._resource_client.responses.retrieve(response_id, **kwargs)

        # Run output guardrails on the retrieved content
        output_text = response.output_text if hasattr(response, "output_text") else ""
        output_results = await self._client._run_stage_guardrails("output", output_text, suppress_tripwire=suppress_tripwire)

        # Return wrapped response with guardrail results
        return self._client._create_guardrails_response(
            response,
            [],
            [],
            output_results,  # preflight  # input
        )
