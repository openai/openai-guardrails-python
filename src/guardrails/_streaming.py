"""Streaming functionality for guardrails integration.

This module contains streaming-related logic for handling LLM responses
with periodic guardrail checks.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from ._base_client import GuardrailsResponse
from .exceptions import GuardrailTripwireTriggered
from .types import GuardrailResult

logger = logging.getLogger(__name__)


class StreamingMixin:
    """Mixin providing streaming functionality for guardrails clients."""

    async def _stream_with_guardrails(
        self,
        llm_stream: Any,  # coroutine or async iterator of OpenAI chunks
        preflight_results: list[GuardrailResult],
        input_results: list[GuardrailResult],
        check_interval: int = 100,
        suppress_tripwire: bool = False,
    ) -> AsyncIterator[GuardrailsResponse]:
        """Stream with periodic guardrail checks (async)."""
        accumulated_text = ""
        chunk_count = 0

        # Handle case where llm_stream is a coroutine
        if hasattr(llm_stream, "__await__"):
            llm_stream = await llm_stream

        async for chunk in llm_stream:
            # Extract text from chunk
            chunk_text = self._extract_response_text(chunk)
            if chunk_text:
                accumulated_text += chunk_text
                chunk_count += 1

                # Run output guardrails periodically
                if chunk_count % check_interval == 0:
                    try:
                        await self._run_stage_guardrails("output", accumulated_text, suppress_tripwire=suppress_tripwire)
                    except GuardrailTripwireTriggered:
                        # Clear accumulated output and re-raise
                        accumulated_text = ""
                        raise

            # Yield chunk with guardrail results
            yield self._create_guardrails_response(chunk, preflight_results, input_results, [])

        # Final output check
        if accumulated_text:
            await self._run_stage_guardrails("output", accumulated_text, suppress_tripwire=suppress_tripwire)
            # Note: This final result won't be yielded since stream is complete
            # but the results are available in the last chunk

    def _stream_with_guardrails_sync(
        self,
        llm_stream: Any,  # iterator of OpenAI chunks
        preflight_results: list[GuardrailResult],
        input_results: list[GuardrailResult],
        check_interval: int = 100,
        suppress_tripwire: bool = False,
    ):
        """Stream with periodic guardrail checks (sync)."""
        accumulated_text = ""
        chunk_count = 0

        for chunk in llm_stream:
            # Extract text from chunk
            chunk_text = self._extract_response_text(chunk)
            if chunk_text:
                accumulated_text += chunk_text
                chunk_count += 1

                # Run output guardrails periodically
                if chunk_count % check_interval == 0:
                    try:
                        self._run_stage_guardrails("output", accumulated_text, suppress_tripwire=suppress_tripwire)
                    except GuardrailTripwireTriggered:
                        # Clear accumulated output and re-raise
                        accumulated_text = ""
                        raise

            # Yield chunk with guardrail results
            yield self._create_guardrails_response(chunk, preflight_results, input_results, [])

        # Final output check
        if accumulated_text:
            self._run_stage_guardrails("output", accumulated_text, suppress_tripwire=suppress_tripwire)
            # Note: This final result won't be yielded since stream is complete
            # but the results are available in the last chunk
