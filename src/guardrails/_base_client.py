"""Base client functionality for guardrails integration.

This module contains the shared base class and data structures used by both
async and sync guardrails clients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response

from .context import has_context
from .runtime import load_pipeline_bundles
from .types import GuardrailLLMContextProto, GuardrailResult
from .utils.context import validate_guardrail_context

logger = logging.getLogger(__name__)

# Type alias for OpenAI response types
OpenAIResponseType = Union[Completion, ChatCompletion, ChatCompletionChunk, Response]  # noqa: UP007


@dataclass(frozen=True, slots=True)
class GuardrailResults:
    """Organized guardrail results by pipeline stage."""

    preflight: list[GuardrailResult]
    input: list[GuardrailResult]
    output: list[GuardrailResult]

    @property
    def all_results(self) -> list[GuardrailResult]:
        """Get all guardrail results combined."""
        return self.preflight + self.input + self.output

    @property
    def tripwires_triggered(self) -> bool:
        """Check if any guardrails triggered tripwires."""
        return any(r.tripwire_triggered for r in self.all_results)

    @property
    def triggered_results(self) -> list[GuardrailResult]:
        """Get only the guardrail results that triggered tripwires."""
        return [r for r in self.all_results if r.tripwire_triggered]


@dataclass(frozen=True, slots=True)
class GuardrailsResponse:
    """Wrapper around any OpenAI response with guardrail results.

    This class provides the same interface as OpenAI responses, with additional
    guardrail results accessible via the guardrail_results attribute.

    Users should access content the same way as with OpenAI responses:
    - For chat completions: response.choices[0].message.content
    - For responses: response.output_text
    - For streaming: response.choices[0].delta.content
    """

    llm_response: OpenAIResponseType  # OpenAI response object (chat completion, response, etc.)
    guardrail_results: GuardrailResults


class GuardrailsBaseClient:
    """Base class with shared functionality for guardrails clients."""

    def _extract_latest_user_message(self, messages: list) -> tuple[str, int]:
        """Extract the latest user message text and its index from a list of message-like items.

        Supports both dict-based messages (OpenAI) and object models with
        role/content attributes. Handles Responses API content-part format.

        Returns:
            Tuple of (message_text, message_index). Index is -1 if no user message found.
        """

        def _get_attr(obj, key: str):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        def _content_to_text(content) -> str:
            # String content
            if isinstance(content, str):
                return content.strip()
            # List of content parts (Responses API)
            if isinstance(content, list):
                parts: list[str] = []
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        text_val = part.get("text", "")
                        if part_type in {"input_text", "text", "output_text", "summary_text"} and isinstance(text_val, str):
                            parts.append(text_val)
                    else:
                        # Object-like content part
                        ptype = getattr(part, "type", None)
                        ptext = getattr(part, "text", "")
                        if ptype in {"input_text", "text", "output_text", "summary_text"} and isinstance(ptext, str):
                            parts.append(ptext)
                return " ".join(parts).strip()
            return ""

        for i in range(len(messages) - 1, -1, -1):
            message = messages[i]
            role = _get_attr(message, "role")
            if role == "user":
                content = _get_attr(message, "content")
                message_text = _content_to_text(content)
                return message_text, i

        return "", -1

    def _create_guardrails_response(
        self,
        llm_response: OpenAIResponseType,
        preflight_results: list[GuardrailResult],
        input_results: list[GuardrailResult],
        output_results: list[GuardrailResult],
    ) -> GuardrailsResponse:
        """Create a GuardrailsResponse with organized results."""
        guardrail_results = GuardrailResults(
            preflight=preflight_results,
            input=input_results,
            output=output_results,
        )
        return GuardrailsResponse(
            llm_response=llm_response,
            guardrail_results=guardrail_results,
        )

    def _setup_guardrails(self, config: str | Path | dict[str, Any], context: Any | None = None) -> None:
        """Setup guardrail infrastructure."""
        self.pipeline = load_pipeline_bundles(config)
        self.guardrails = self._instantiate_all_guardrails()
        self.context = self._create_default_context() if context is None else context
        self._validate_context(self.context)

    def _apply_preflight_modifications(
        self, data: list[dict[str, str]] | str, preflight_results: list[GuardrailResult]
    ) -> list[dict[str, str]] | str:
        """Apply pre-flight modifications to messages or text.

        Args:
            data: Either a list of messages or a text string
            preflight_results: Results from pre-flight guardrails

        Returns:
            Modified data with pre-flight changes applied
        """
        if not preflight_results:
            return data

        # Get PII mappings from preflight results for individual text processing
        pii_mappings = {}
        for result in preflight_results:
            if "detected_entities" in result.info:
                detected = result.info["detected_entities"]
                for entity_type, entities in detected.items():
                    for entity in entities:
                        # Map original PII to masked token
                        pii_mappings[entity] = f"<{entity_type}>"

        if not pii_mappings:
            return data

        def _mask_text(text: str) -> str:
            """Apply PII masking to individual text with robust replacement."""
            if not isinstance(text, str):
                return text

            masked_text = text

            # Sort PII entities by length (longest first) to avoid partial replacements
            # (shouldn't need this as Presidio should handle this, but just in case)
            sorted_pii = sorted(pii_mappings.items(), key=lambda x: len(x[0]), reverse=True)

            for original_pii, masked_token in sorted_pii:
                if original_pii in masked_text:
                    # Use replace() which handles special characters safely
                    masked_text = masked_text.replace(original_pii, masked_token)

            return masked_text

        if isinstance(data, str):
            # Handle string input (for responses API)
            return _mask_text(data)
        else:
            # Handle message list input (primarily for chat API and structured Responses API)
            _, latest_user_idx = self._extract_latest_user_message(data)
            if latest_user_idx == -1:
                return data

            # Use shallow copy for efficiency - we only modify the content field of one message
            modified_messages = data.copy()

            # Extract current content safely
            current_content = (
                data[latest_user_idx]["content"] if isinstance(data[latest_user_idx], dict) else getattr(data[latest_user_idx], "content", None)
            )

            # Apply modifications based on content type
            if isinstance(current_content, str):
                # Plain string content - mask individually
                modified_content = _mask_text(current_content)
            elif isinstance(current_content, list):
                # Structured content - mask each text part individually
                modified_content = []
                for part in current_content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type in {"input_text", "text", "output_text", "summary_text"} and "text" in part:
                            # Mask this specific text part individually
                            original_text = part["text"]
                            masked_text = _mask_text(original_text)
                            modified_content.append({**part, "text": masked_text})
                        else:
                            # Keep non-text parts unchanged
                            modified_content.append(part)
                    else:
                        # Keep unknown parts unchanged
                        modified_content.append(part)
            else:
                # Unknown content type - skip modifications
                return data

            # Only modify the specific message that needs content changes
            if modified_content != current_content:
                if isinstance(modified_messages[latest_user_idx], dict):
                    modified_messages[latest_user_idx] = {
                        **modified_messages[latest_user_idx],
                        "content": modified_content,
                    }
                else:
                    # Fallback: if it's an object-like, set attribute when possible
                    try:
                        modified_messages[latest_user_idx].content = modified_content
                    except Exception:
                        return data

            return modified_messages

    def _instantiate_all_guardrails(self) -> dict[str, list]:
        """Instantiate guardrails for all stages."""
        from .registry import default_spec_registry
        from .runtime import instantiate_guardrails

        guardrails = {}
        for stage_name in ["pre_flight", "input", "output"]:
            stage = getattr(self.pipeline, stage_name)
            guardrails[stage_name] = instantiate_guardrails(stage, default_spec_registry) if stage else []
        return guardrails

    def _validate_context(self, context: Any) -> None:
        """Validate context against all guardrails."""
        for stage_guardrails in self.guardrails.values():
            for guardrail in stage_guardrails:
                validate_guardrail_context(guardrail, context)

    def _extract_response_text(self, response: Any) -> str:
        """Extract text content from various response types."""
        choice0 = response.choices[0] if getattr(response, "choices", None) else None
        candidates: tuple[str | None, ...] = (
            getattr(getattr(choice0, "delta", None), "content", None),
            getattr(getattr(choice0, "message", None), "content", None),
            getattr(response, "output_text", None),
            getattr(response, "delta", None),
        )
        for value in candidates:
            if isinstance(value, str):
                return value or ""
        if getattr(response, "type", None) == "response.output_text.delta":
            return getattr(response, "delta", "") or ""
        return ""

    def _create_default_context(self) -> GuardrailLLMContextProto:
        """Create default context with guardrail_llm client.

        This method checks for existing ContextVars context first.
        If none exists, it creates a default context using the main client.
        """
        # Check if there's a context set via ContextVars
        if has_context():
            from .context import get_context

            context = get_context()
            if context and hasattr(context, "guardrail_llm"):
                # Use the context's guardrail_llm
                return context

        # Fall back to using the main client (self) for guardrails
        # Note: This will be overridden by subclasses to provide the correct type
        raise NotImplementedError("Subclasses must implement _create_default_context")

    def _initialize_client(self, config: str | Path | dict[str, Any], openai_kwargs: dict[str, Any], client_class: type) -> None:
        """Initialize client with common setup.

        Args:
            config: Pipeline configuration
            openai_kwargs: OpenAI client arguments
            client_class: The OpenAI client class to instantiate for resources
        """
        # Create a separate OpenAI client instance for resource access
        # This avoids circular reference issues when overriding OpenAI's resource properties
        # Note: This is NOT used for LLM calls or guardrails - it's just for resource access
        self._resource_client = client_class(**openai_kwargs)

        # Setup guardrails after OpenAI initialization
        # Check for existing ContextVars context, otherwise use default
        self._setup_guardrails(config, None)

        # Override chat and responses after parent initialization
        self._override_resources()
