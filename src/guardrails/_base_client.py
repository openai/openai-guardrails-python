"""Base client functionality for guardrails integration.

This module contains the shared base class and data structures used by both
async and sync guardrails clients.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Union

from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response

from .context import has_context
from .runtime import load_pipeline_bundles
from .types import GuardrailLLMContextProto, GuardrailResult
from .utils.context import validate_guardrail_context
from .utils.conversation import append_assistant_response, normalize_conversation

logger = logging.getLogger(__name__)

# Type alias for OpenAI response types
OpenAIResponseType = Union[Completion, ChatCompletion, ChatCompletionChunk, Response]  # noqa: UP007

# Text content types recognized in message content parts
_TEXT_CONTENT_TYPES: Final[set[str]] = {"text", "input_text", "output_text"}


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
                        if part_type in _TEXT_CONTENT_TYPES and isinstance(text_val, str):
                            parts.append(text_val)
                    else:
                        # Object-like content part
                        ptype = getattr(part, "type", None)
                        ptext = getattr(part, "text", "")
                        if ptype in _TEXT_CONTENT_TYPES and isinstance(ptext, str):
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
            Modified data with PII masking applied if PII was detected
        """
        if not preflight_results:
            return data

        # Look specifically for PII guardrail results with actual modifications
        pii_result = None
        for result in preflight_results:
            # Only PII guardrail modifies text - check name first (faster)
            if result.info.get("guardrail_name") == "Contains PII" and result.info.get("pii_detected"):
                pii_result = result
                break  # PII is the only guardrail that modifies text

        # If no PII modifications were made, return original data
        if pii_result is None:
            return data

        # Apply PII-masked text to data
        if isinstance(data, str):
            # Simple case: string input (Responses API)
            checked_text = pii_result.info.get("checked_text")
            return checked_text if checked_text is not None else data

        # Complex case: message list (Chat API)
        _, latest_user_idx = self._extract_latest_user_message(data)
        if latest_user_idx == -1:
            return data

        # Get current content
        current_content = (
            data[latest_user_idx]["content"] if isinstance(data[latest_user_idx], dict) else getattr(data[latest_user_idx], "content", None)
        )

        # Apply PII-masked text based on content type
        if isinstance(current_content, str):
            # Plain string content - replace with masked version
            checked_text = pii_result.info.get("checked_text")
            if checked_text is None:
                return data
            return self._update_message_content(data, latest_user_idx, checked_text)

        if isinstance(current_content, list):
            # Structured content - mask each text part individually using Presidio
            return self._apply_pii_masking_to_structured_content(data, pii_result, latest_user_idx, current_content)

        # Unknown content type, return unchanged
        return data

    def _update_message_content(self, data: list[dict[str, str]], user_idx: int, new_content: Any) -> list[dict[str, str]]:
        """Update message content at the specified index.

        Args:
            data: Message list
            user_idx: Index of message to update
            new_content: New content value

        Returns:
            Modified message list or original if update fails
        """
        modified_messages = data.copy()
        try:
            if isinstance(modified_messages[user_idx], dict):
                modified_messages[user_idx] = {
                    **modified_messages[user_idx],
                    "content": new_content,
                }
            else:
                modified_messages[user_idx].content = new_content
        except Exception:
            return data
        return modified_messages

    def _apply_pii_masking_to_structured_content(
        self,
        data: list[dict[str, str]],
        pii_result: GuardrailResult,
        user_idx: int,
        current_content: list,
    ) -> list[dict[str, str]]:
        """Apply PII masking to structured content parts using Presidio.

        Args:
            data: Message list with structured content
            pii_result: PII guardrail result containing detected entities
            user_idx: Index of the user message to modify
            current_content: The structured content list (already extracted)

        Returns:
            Modified messages with PII masking applied to each text part
        """
        from presidio_anonymizer import AnonymizerEngine
        from presidio_anonymizer.entities import OperatorConfig

        # Extract detected entity types and config
        detected = pii_result.info.get("detected_entities", {})
        if not detected:
            return data

        detect_encoded_pii = pii_result.info.get("detect_encoded_pii", False)

        # Get Presidio engines - entity types are guaranteed valid from detection
        from .checks.text.pii import _get_analyzer_engine

        analyzer = _get_analyzer_engine()
        anonymizer = AnonymizerEngine()
        entity_types = list(detected.keys())

        # Create operators for each entity type
        operators = {entity_type: OperatorConfig("replace", {"new_value": f"<{entity_type}>"}) for entity_type in entity_types}

        def _mask_text(text: str) -> str:
            """Mask using Presidio's analyzer and anonymizer with Unicode normalization.

            Handles both plain and encoded PII consistently with main detection path.
            """
            if not text:
                return text

            # Import functions from pii module
            from .checks.text.pii import _build_decoded_text, _normalize_unicode

            # Normalize to prevent bypasses
            normalized = _normalize_unicode(text)

            # Check for plain PII
            analyzer_results = analyzer.analyze(normalized, entities=entity_types, language="en")
            has_plain_pii = bool(analyzer_results)

            # Check for encoded PII if enabled
            has_encoded_pii = False
            encoded_candidates = []

            if detect_encoded_pii:
                decoded_text, encoded_candidates = _build_decoded_text(normalized)
                if encoded_candidates:
                    # Analyze decoded text
                    decoded_results = analyzer.analyze(decoded_text, entities=entity_types, language="en")
                    has_encoded_pii = bool(decoded_results)

            # If no PII found at all, return original text
            if not has_plain_pii and not has_encoded_pii:
                return text

            # Mask plain PII
            masked = normalized
            if has_plain_pii:
                masked = anonymizer.anonymize(text=masked, analyzer_results=analyzer_results, operators=operators).text

            # Mask encoded PII if found
            if has_encoded_pii:
                # Re-analyze to get positions in the (potentially) masked text
                decoded_text_for_masking, candidates_for_masking = _build_decoded_text(masked)
                decoded_results = analyzer.analyze(decoded_text_for_masking, entities=entity_types, language="en")

                if decoded_results:
                    # Map detections back to mask encoded chunks
                    for result in decoded_results:
                        detected_value = decoded_text_for_masking[result.start : result.end]
                        entity_type = result.entity_type

                        # Find candidate that contains this PII
                        for candidate in candidates_for_masking:
                            if detected_value in candidate.decoded_text:
                                # Mask the encoded version
                                entity_marker = f"<{entity_type}_ENCODED>"
                                masked = masked[: candidate.start] + entity_marker + masked[candidate.end :]
                                break

            return masked

        # Mask each text part
        modified_content = []
        for part in current_content:
            if isinstance(part, dict):
                part_text = part.get("text")
                if part.get("type") in _TEXT_CONTENT_TYPES and isinstance(part_text, str) and part_text:
                    modified_content.append({**part, "text": _mask_text(part_text)})
                else:
                    modified_content.append(part)
            else:
                # Handle object-based content parts
                if (
                    hasattr(part, "type")
                    and hasattr(part, "text")
                    and part.type in _TEXT_CONTENT_TYPES
                    and isinstance(part.text, str)
                    and part.text
                ):
                    try:
                        part.text = _mask_text(part.text)
                    except Exception:
                        pass
                    modified_content.append(part)
                else:
                    # Preserve non-dict, non-object parts (e.g., raw strings)
                    modified_content.append(part)

        return self._update_message_content(data, user_idx, modified_content)

    def _instantiate_all_guardrails(self) -> dict[str, list]:
        """Instantiate guardrails for all stages."""
        from .registry import default_spec_registry
        from .runtime import instantiate_guardrails

        guardrails = {}
        for stage_name in ["pre_flight", "input", "output"]:
            stage = getattr(self.pipeline, stage_name)
            guardrails[stage_name] = instantiate_guardrails(stage, default_spec_registry) if stage else []
        return guardrails

    def _normalize_conversation(self, payload: Any) -> list[dict[str, Any]]:
        """Normalize arbitrary conversation payloads."""
        return normalize_conversation(payload)

    def _conversation_with_response(
        self,
        conversation: list[dict[str, Any]],
        response: Any,
    ) -> list[dict[str, Any]]:
        """Append the assistant response to a normalized conversation."""
        return append_assistant_response(conversation, response)

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
