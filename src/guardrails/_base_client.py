"""Base client functionality for guardrails integration.

This module contains the shared base class and data structures used by both
async and sync guardrails clients.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Union
from weakref import WeakValueDictionary

from openai.types import Completion
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from openai.types.responses import Response

from .context import has_context
from .runtime import load_pipeline_bundles
from .types import GuardrailLLMContextProto, GuardrailResult, aggregate_token_usage_from_infos
from .utils.context import validate_guardrail_context
from .utils.conversation import append_assistant_response, normalize_conversation

logger = logging.getLogger(__name__)

# Track instances that have emitted deprecation warnings
_warned_instance_ids: WeakValueDictionary[int, Any] = WeakValueDictionary()


def _warn_llm_response_deprecation(instance: Any) -> None:
    """Emit deprecation warning for llm_response access.

    Args:
        instance: The GuardrailsResponse instance.
    """
    instance_id = id(instance)
    if instance_id not in _warned_instance_ids:
        warnings.warn(
            "Accessing 'llm_response' is deprecated. "
            "Access response attributes directly instead (e.g., use 'response.output_text' "
            "instead of 'response.llm_response.output_text'). "
            "The 'llm_response' attribute will be removed in future versions.",
            DeprecationWarning,
            stacklevel=3,
        )
        _warned_instance_ids[instance_id] = instance

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

    @property
    def total_token_usage(self) -> dict[str, Any]:
        """Aggregate token usage across all LLM-based guardrails.

        Sums prompt_tokens, completion_tokens, and total_tokens from all
        guardrail results that include token_usage in their info dict.
        Non-LLM guardrails (which don't have token_usage) are skipped.

        Returns:
            Dictionary with:
            - prompt_tokens: Sum of all prompt tokens (or None if no data)
            - completion_tokens: Sum of all completion tokens (or None if no data)
            - total_tokens: Sum of all total tokens (or None if no data)
        """
        infos = (result.info for result in self.all_results)
        return aggregate_token_usage_from_infos(infos)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class GuardrailsResponse:
    """OpenAI response with guardrail results.

    Access OpenAI response attributes directly:
        response.output_text
        response.choices[0].message.content

    Access guardrail results:
        response.guardrail_results.preflight
        response.guardrail_results.input
        response.guardrail_results.output
    """

    _llm_response: OpenAIResponseType
    guardrail_results: GuardrailResults

    def __init__(
        self,
        llm_response: OpenAIResponseType | None = None,
        guardrail_results: GuardrailResults | None = None,
        *,
        _llm_response: OpenAIResponseType | None = None,
    ) -> None:
        """Initialize GuardrailsResponse.

        Args:
            llm_response: OpenAI response object.
            guardrail_results: Guardrail results.
            _llm_response: OpenAI response object (keyword-only alias).

        Raises:
            TypeError: If arguments are invalid.
        """
        if llm_response is not None and _llm_response is not None:
            msg = "Cannot specify both 'llm_response' and '_llm_response'"
            raise TypeError(msg)

        if llm_response is None and _llm_response is None:
            msg = "Must specify either 'llm_response' or '_llm_response'"
            raise TypeError(msg)

        if guardrail_results is None:
            msg = "Missing required argument: 'guardrail_results'"
            raise TypeError(msg)

        response_obj = llm_response if llm_response is not None else _llm_response

        object.__setattr__(self, "_llm_response", response_obj)
        object.__setattr__(self, "guardrail_results", guardrail_results)

    @property
    def llm_response(self) -> OpenAIResponseType:
        """Access underlying OpenAI response (deprecated).

        Returns:
            OpenAI response object.
        """
        _warn_llm_response_deprecation(self)
        return self._llm_response

    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to underlying OpenAI response.

        Args:
            name: Attribute name.

        Returns:
            Attribute value from OpenAI response.

        Raises:
            AttributeError: If attribute doesn't exist.
        """
        return getattr(self._llm_response, name)

    def __dir__(self) -> list[str]:
        """List all available attributes including delegated ones.

        Returns:
            Sorted list of attribute names.
        """
        own_attrs = set(object.__dir__(self))
        delegated_attrs = set(dir(self._llm_response))
        return sorted(own_attrs | delegated_attrs)


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
            _llm_response=llm_response,
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
        from guardrails.utils.anonymizer import OperatorConfig, anonymize

        # Extract detected entity types and config
        detected = pii_result.info.get("detected_entities", {})
        if not detected:
            return data

        detect_encoded_pii = pii_result.info.get("detect_encoded_pii", False)

        # Get analyzer engine - entity types are guaranteed valid from detection
        from .checks.text.pii import _get_analyzer_engine

        analyzer = _get_analyzer_engine()
        entity_types = list(detected.keys())

        # Create operators for each entity type
        operators = {entity_type: OperatorConfig("replace", {"new_value": f"<{entity_type}>"}) for entity_type in entity_types}

        def _mask_text(text: str) -> str:
            """Mask using custom anonymizer with Unicode normalization.

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
                masked = anonymize(text=masked, analyzer_results=analyzer_results, operators=operators).text

            # Mask encoded PII if found
            if has_encoded_pii:
                # Re-analyze to get positions in the (potentially) masked text
                decoded_text_for_masking, candidates_for_masking = _build_decoded_text(masked)
                decoded_results = analyzer.analyze(decoded_text_for_masking, entities=entity_types, language="en")

                if decoded_results:
                    # Build list of (candidate, entity_type) pairs to mask
                    candidates_to_mask = []

                    for result in decoded_results:
                        detected_value = decoded_text_for_masking[result.start : result.end]
                        entity_type = result.entity_type

                        # Find candidate that overlaps with this PII
                        # Use comprehensive overlap logic matching pii.py implementation
                        for candidate in candidates_for_masking:
                            if not candidate.decoded_text:
                                continue

                            candidate_lower = candidate.decoded_text.lower()
                            detected_lower = detected_value.lower()

                            # Check if candidate's decoded text overlaps with the detection
                            # Handle partial encodings where encoded span may include extra characters
                            # e.g., %3A%6a%6f%65%40 → ":joe@" but only "joe@" is in email "joe@domain.com"
                            has_overlap = (
                                candidate_lower in detected_lower  # Candidate is substring of detection
                                or detected_lower in candidate_lower  # Detection is substring of candidate
                                or (
                                    len(candidate_lower) >= 3
                                    and any(  # Any 3-char chunk overlaps
                                        candidate_lower[i : i + 3] in detected_lower for i in range(len(candidate_lower) - 2)
                                    )
                                )
                            )

                            if has_overlap:
                                candidates_to_mask.append((candidate, entity_type))
                                break

                    # Sort by position (reverse) to mask from end to start
                    # This preserves position validity for subsequent replacements
                    candidates_to_mask.sort(key=lambda x: x[0].start, reverse=True)

                    # Mask from end to start
                    for candidate, entity_type in candidates_to_mask:
                        entity_marker = f"<{entity_type}_ENCODED>"
                        masked = masked[: candidate.start] + entity_marker + masked[candidate.end :]

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
                if hasattr(part, "type") and hasattr(part, "text") and part.type in _TEXT_CONTENT_TYPES and isinstance(part.text, str) and part.text:
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
