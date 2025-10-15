"""High-level GuardrailsClient for easy integration with OpenAI APIs.

This module provides GuardrailsAsyncOpenAI and GuardrailsOpenAI classes that
subclass OpenAI's clients to provide full API compatibility while automatically
applying guardrails to text-based methods that could benefit from validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI, OpenAI

try:  # Optional Azure support
    from openai import AsyncAzureOpenAI, AzureOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncAzureOpenAI = None  # type: ignore
    AzureOpenAI = None  # type: ignore

from ._base_client import (
    GuardrailResults,
    GuardrailsBaseClient,
    GuardrailsResponse,
    OpenAIResponseType,
)
from ._openai_utils import prepare_openai_kwargs
from ._streaming import StreamingMixin
from .exceptions import GuardrailTripwireTriggered
from .runtime import run_guardrails
from .types import GuardrailLLMContextProto, GuardrailResult

# Re-export for backward compatibility
__all__ = [
    "GuardrailsAsyncOpenAI",
    "GuardrailsOpenAI",
    "GuardrailsAsyncAzureOpenAI",
    "GuardrailsAzureOpenAI",
    "GuardrailsResponse",
    "GuardrailResults",
]

logger = logging.getLogger(__name__)

# Stage name constants
PREFLIGHT_STAGE = "pre_flight"
INPUT_STAGE = "input"
OUTPUT_STAGE = "output"


class GuardrailsAsyncOpenAI(AsyncOpenAI, GuardrailsBaseClient, StreamingMixin):
    """AsyncOpenAI subclass with automatic guardrail integration.

    This class provides full OpenAI API compatibility while automatically
    applying guardrails to text-based methods that could benefit from validation.

    Methods with guardrails:
    - chat.completions.create() - Input/output validation
    - responses.create() - Input/output validation
    - responses.parse() - Input/output validation
    - responses.retrieve() - Output validation (retrieved content)

    All other methods pass through unchanged for full API compatibility.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        raise_guardrail_errors: bool = False,
        **openai_kwargs,
    ) -> None:
        """Initialize GuardrailsAsyncOpenAI client.

        Args:
            config: Path to pipeline configuration file or configuration dict.
            raise_guardrail_errors: If True, raise exceptions when guardrails fail to execute.
                If False (default), treat guardrail execution errors as safe and continue.
                Note: Tripwires (guardrail violations) are handled separately and not affected
                by this parameter.
            **openai_kwargs: Additional arguments passed to AsyncOpenAI constructor.
        """
        openai_kwargs = prepare_openai_kwargs(openai_kwargs)
        # Initialize OpenAI client first
        super().__init__(**openai_kwargs)

        # Store the error handling preference
        self.raise_guardrail_errors = raise_guardrail_errors

        # Use base client initialization helper (guardrail LLM client)
        from openai import AsyncOpenAI

        self._initialize_client(config, openai_kwargs, AsyncOpenAI)

    def _create_default_context(self) -> GuardrailLLMContextProto:
        """Create default context with guardrail_llm client."""
        # First check base implementation for ContextVars
        try:
            return super()._create_default_context()
        except NotImplementedError:
            pass

        # Create a separate client instance for guardrails (not the same as main client)
        @dataclass
        class DefaultContext:
            guardrail_llm: AsyncOpenAI

        # Create separate instance with same configuration
        from openai import AsyncOpenAI

        guardrail_kwargs = {
            "api_key": self.api_key,
            "base_url": getattr(self, "base_url", None),
            "organization": getattr(self, "organization", None),
            "timeout": getattr(self, "timeout", None),
            "max_retries": getattr(self, "max_retries", None),
        }
        default_headers = getattr(self, "default_headers", None)
        if default_headers is not None:
            guardrail_kwargs["default_headers"] = default_headers
        guardrail_client = AsyncOpenAI(**prepare_openai_kwargs(guardrail_kwargs))

        return DefaultContext(guardrail_llm=guardrail_client)

    def _create_context_with_conversation(self, conversation_history: list) -> GuardrailLLMContextProto:
        """Create a context with conversation history for prompt injection detection guardrail."""

        # Create a new context that includes conversation history
        @dataclass
        class ConversationContext:
            guardrail_llm: AsyncOpenAI
            conversation_history: list

            def get_conversation_history(self) -> list:
                return self.conversation_history

        return ConversationContext(
            guardrail_llm=self.context.guardrail_llm,
            conversation_history=conversation_history,
        )

    def _append_llm_response_to_conversation(self, conversation_history: list | str, llm_response: Any) -> list:
        """Append LLM response to conversation history as-is."""
        if conversation_history is None:
            conversation_history = []

        # Handle case where conversation_history is a string (from single input)
        if isinstance(conversation_history, str):
            conversation_history = [{"role": "user", "content": conversation_history}]

        # Make a copy to avoid modifying the original
        updated_history = conversation_history.copy()

        # For responses API: append the output directly
        if hasattr(llm_response, "output") and llm_response.output:
            updated_history.extend(llm_response.output)
        # For chat completions: append the choice message directly (prompt injection detection check will parse)
        elif hasattr(llm_response, "choices") and llm_response.choices:
            updated_history.append(llm_response.choices[0])

        return updated_history

    def _override_resources(self):
        """Override chat and responses with our guardrail-enhanced versions."""
        from .resources.chat import AsyncChat
        from .resources.responses import AsyncResponses

        # Replace the chat and responses attributes with our versions
        object.__setattr__(self, "chat", AsyncChat(self))
        object.__setattr__(self, "responses", AsyncResponses(self))

    async def _run_stage_guardrails(
        self,
        stage_name: str,
        text: str,
        conversation_history: list = None,
        suppress_tripwire: bool = False,
    ) -> list[GuardrailResult]:
        """Run guardrails for a specific pipeline stage."""
        if not self.guardrails[stage_name]:
            return []

        try:
            # Check if prompt injection detection guardrail is present and we have conversation history
            has_injection_detection = any(
                guardrail.definition.name.lower() == "prompt injection detection" for guardrail in self.guardrails[stage_name]
            )

            if has_injection_detection and conversation_history:
                ctx = self._create_context_with_conversation(conversation_history)
            else:
                ctx = self.context

            results = await run_guardrails(
                ctx=ctx,
                data=text,
                media_type="text/plain",
                guardrails=self.guardrails[stage_name],
                suppress_tripwire=suppress_tripwire,
                stage_name=stage_name,
                raise_guardrail_errors=self.raise_guardrail_errors,
            )

            # Check for tripwire triggers unless suppressed
            if not suppress_tripwire:
                for result in results:
                    if result.tripwire_triggered:
                        raise GuardrailTripwireTriggered(result)

            return results

        except GuardrailTripwireTriggered:
            if suppress_tripwire:
                return results
            raise

    async def _handle_llm_response(
        self,
        llm_response: OpenAIResponseType,
        preflight_results: list[GuardrailResult],
        input_results: list[GuardrailResult],
        conversation_history: list = None,
        suppress_tripwire: bool = False,
    ) -> GuardrailsResponse:
        """Handle non-streaming LLM response with output guardrails."""
        # Create complete conversation history including the LLM response
        complete_conversation = self._append_llm_response_to_conversation(conversation_history, llm_response)

        response_text = self._extract_response_text(llm_response)
        output_results = await self._run_stage_guardrails(
            "output",
            response_text,
            conversation_history=complete_conversation,
            suppress_tripwire=suppress_tripwire,
        )

        return self._create_guardrails_response(llm_response, preflight_results, input_results, output_results)


class GuardrailsOpenAI(OpenAI, GuardrailsBaseClient, StreamingMixin):
    """OpenAI subclass with automatic guardrail integration.

    Synchronous version of GuardrailsAsyncOpenAI with the same functionality.
    """

    def __init__(
        self,
        config: str | Path | dict[str, Any],
        raise_guardrail_errors: bool = False,
        **openai_kwargs,
    ) -> None:
        """Initialize GuardrailsOpenAI client.

        Args:
            config: Path to pipeline configuration file or configuration dict.
            raise_guardrail_errors: If True, raise exceptions when guardrails fail to execute.
                If False (default), treat guardrail execution errors as safe and continue.
                Note: Tripwires (guardrail violations) are handled separately and not affected
                by this parameter.
            **openai_kwargs: Additional arguments passed to OpenAI constructor.
        """
        openai_kwargs = prepare_openai_kwargs(openai_kwargs)
        # Initialize OpenAI client first
        super().__init__(**openai_kwargs)

        # Store the error handling preference
        self.raise_guardrail_errors = raise_guardrail_errors

        # Use base client initialization helper
        from openai import OpenAI

        self._initialize_client(config, openai_kwargs, OpenAI)

    def _create_default_context(self) -> GuardrailLLMContextProto:
        """Create default context with guardrail_llm client."""
        # First check base implementation for ContextVars
        try:
            return super()._create_default_context()
        except NotImplementedError:
            pass

        # Create a separate client instance for guardrails (not the same as main client)
        @dataclass
        class DefaultContext:
            guardrail_llm: OpenAI

        # Create separate instance with same configuration
        from openai import OpenAI

        guardrail_kwargs = {
            "api_key": self.api_key,
            "base_url": getattr(self, "base_url", None),
            "organization": getattr(self, "organization", None),
            "timeout": getattr(self, "timeout", None),
            "max_retries": getattr(self, "max_retries", None),
        }
        default_headers = getattr(self, "default_headers", None)
        if default_headers is not None:
            guardrail_kwargs["default_headers"] = default_headers
        guardrail_client = OpenAI(**prepare_openai_kwargs(guardrail_kwargs))

        return DefaultContext(guardrail_llm=guardrail_client)

    def _create_context_with_conversation(self, conversation_history: list) -> GuardrailLLMContextProto:
        """Create a context with conversation history for prompt injection detection guardrail."""

        # Create a new context that includes conversation history
        @dataclass
        class ConversationContext:
            guardrail_llm: OpenAI
            conversation_history: list

            def get_conversation_history(self) -> list:
                return self.conversation_history

        return ConversationContext(
            guardrail_llm=self.context.guardrail_llm,
            conversation_history=conversation_history,
        )

    def _append_llm_response_to_conversation(self, conversation_history: list | str, llm_response: Any) -> list:
        """Append LLM response to conversation history as-is."""
        if conversation_history is None:
            conversation_history = []

        # Handle case where conversation_history is a string (from single input)
        if isinstance(conversation_history, str):
            conversation_history = [{"role": "user", "content": conversation_history}]

        # Make a copy to avoid modifying the original
        updated_history = conversation_history.copy()

        # For responses API: append the output directly
        if hasattr(llm_response, "output") and llm_response.output:
            updated_history.extend(llm_response.output)
        # For chat completions: append the choice message directly (prompt injection detection check will parse)
        elif hasattr(llm_response, "choices") and llm_response.choices:
            updated_history.append(llm_response.choices[0])

        return updated_history

    def _override_resources(self):
        """Override chat and responses with our guardrail-enhanced versions."""
        from .resources.chat import Chat
        from .resources.responses import Responses

        # Replace the chat and responses attributes with our versions
        object.__setattr__(self, "chat", Chat(self))
        object.__setattr__(self, "responses", Responses(self))

    def _run_stage_guardrails(
        self,
        stage_name: str,
        text: str,
        conversation_history: list = None,
        suppress_tripwire: bool = False,
    ) -> list[GuardrailResult]:
        """Run guardrails for a specific pipeline stage (synchronous)."""
        if not self.guardrails[stage_name]:
            return []

        # For sync version, we need to run async guardrails in sync context
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        async def _run_async():
            # Check if prompt injection detection guardrail is present and we have conversation history
            has_injection_detection = any(
                guardrail.definition.name.lower() == "prompt injection detection" for guardrail in self.guardrails[stage_name]
            )

            if has_injection_detection and conversation_history:
                ctx = self._create_context_with_conversation(conversation_history)
            else:
                ctx = self.context

            results = await run_guardrails(
                ctx=ctx,
                data=text,
                media_type="text/plain",
                guardrails=self.guardrails[stage_name],
                suppress_tripwire=suppress_tripwire,
                stage_name=stage_name,
                raise_guardrail_errors=self.raise_guardrail_errors,
            )

            # Check for tripwire triggers unless suppressed
            if not suppress_tripwire:
                for result in results:
                    if result.tripwire_triggered:
                        raise GuardrailTripwireTriggered(result)

            return results

        try:
            return loop.run_until_complete(_run_async())
        except GuardrailTripwireTriggered:
            if suppress_tripwire:
                return []
            raise

    def _handle_llm_response(
        self,
        llm_response: OpenAIResponseType,
        preflight_results: list[GuardrailResult],
        input_results: list[GuardrailResult],
        conversation_history: list = None,
        suppress_tripwire: bool = False,
    ) -> GuardrailsResponse:
        """Handle LLM response with output guardrails."""
        # Create complete conversation history including the LLM response
        complete_conversation = self._append_llm_response_to_conversation(conversation_history, llm_response)

        response_text = self._extract_response_text(llm_response)
        output_results = self._run_stage_guardrails(
            "output",
            response_text,
            conversation_history=complete_conversation,
            suppress_tripwire=suppress_tripwire,
        )

        return self._create_guardrails_response(llm_response, preflight_results, input_results, output_results)


# ---------------- Azure OpenAI Variants -----------------

if AsyncAzureOpenAI is not None:

    class GuardrailsAsyncAzureOpenAI(AsyncAzureOpenAI, GuardrailsBaseClient, StreamingMixin):  # type: ignore
        """AsyncAzureOpenAI subclass with automatic guardrail integration."""

        def __init__(
            self,
            config: str | Path | dict[str, Any],
            raise_guardrail_errors: bool = False,
            **azure_kwargs: Any,
        ) -> None:
            """Initialize GuardrailsAsyncAzureOpenAI client.

            Args:
                config: Path to pipeline configuration file or configuration dict.
                raise_guardrail_errors: If True, raise exceptions when guardrails fail to execute.
                    If False (default), treat guardrail execution errors as safe and continue.
                    Note: Tripwires (guardrail violations) are handled separately and not affected
                by this parameter.
                **azure_kwargs: Additional arguments passed to AsyncAzureOpenAI constructor.
            """
            azure_kwargs = prepare_openai_kwargs(azure_kwargs)
            # Initialize Azure client first
            super().__init__(**azure_kwargs)

            # Store the error handling preference
            self.raise_guardrail_errors = raise_guardrail_errors

            # Initialize common guardrails infra; resource client should also be Azure
            from openai import AsyncAzureOpenAI as _AsyncAzureOpenAI  # type: ignore

            # Persist azure kwargs so we can mirror them when creating the context client
            self._azure_kwargs: dict[str, Any] = dict(azure_kwargs)
            self._initialize_client(config, azure_kwargs, _AsyncAzureOpenAI)

        def _create_default_context(self) -> GuardrailLLMContextProto:
            # Try ContextVars first
            try:
                return super()._create_default_context()
            except NotImplementedError:
                pass

            # Create a separate Azure client instance for guardrails
            @dataclass
            class DefaultContext:
                guardrail_llm: Any  # AsyncAzureOpenAI

            from openai import AsyncAzureOpenAI as _AsyncAzureOpenAI  # type: ignore

            # Use the same kwargs the main Azure client was constructed with
            guardrail_client = _AsyncAzureOpenAI(**self._azure_kwargs)
            return DefaultContext(guardrail_llm=guardrail_client)

        def _create_context_with_conversation(self, conversation_history: list) -> GuardrailLLMContextProto:
            """Create a context with conversation history for prompt injection detection guardrail."""

            # Create a new context that includes conversation history
            @dataclass
            class ConversationContext:
                guardrail_llm: Any  # AsyncAzureOpenAI
                conversation_history: list

                def get_conversation_history(self) -> list:
                    return self.conversation_history

            return ConversationContext(
                guardrail_llm=self.context.guardrail_llm,
                conversation_history=conversation_history,
            )

        def _append_llm_response_to_conversation(self, conversation_history: list | str, llm_response: Any) -> list:
            """Append LLM response to conversation history as-is."""
            if conversation_history is None:
                conversation_history = []

            # Handle case where conversation_history is a string (from single input)
            if isinstance(conversation_history, str):
                conversation_history = [{"role": "user", "content": conversation_history}]

            # Make a copy to avoid modifying the original
            updated_history = conversation_history.copy()

            # For responses API: append the output directly
            if hasattr(llm_response, "output") and llm_response.output:
                updated_history.extend(llm_response.output)
            # For chat completions: append the choice message directly (prompt injection detection check will parse)
            elif hasattr(llm_response, "choices") and llm_response.choices:
                updated_history.append(llm_response.choices[0])

            return updated_history

        def _override_resources(self):
            from .resources.chat import AsyncChat
            from .resources.responses import AsyncResponses

            object.__setattr__(self, "chat", AsyncChat(self))
            object.__setattr__(self, "responses", AsyncResponses(self))

        async def _run_stage_guardrails(
            self,
            stage_name: str,
            text: str,
            conversation_history: list = None,
            suppress_tripwire: bool = False,
        ) -> list[GuardrailResult]:
            """Run guardrails for a specific pipeline stage."""
            if not self.guardrails[stage_name]:
                return []

            try:
                # Check if prompt injection detection guardrail is present and we have conversation history
                has_injection_detection = any(
                    guardrail.definition.name.lower() == "prompt injection detection" for guardrail in self.guardrails[stage_name]
                )

                if has_injection_detection and conversation_history:
                    ctx = self._create_context_with_conversation(conversation_history)
                else:
                    ctx = self.context

                results = await run_guardrails(
                    ctx=ctx,
                    data=text,
                    media_type="text/plain",
                    guardrails=self.guardrails[stage_name],
                    suppress_tripwire=suppress_tripwire,
                    stage_name=stage_name,
                    raise_guardrail_errors=self.raise_guardrail_errors,
                )

                # Check for tripwire triggers unless suppressed
                if not suppress_tripwire:
                    for result in results:
                        if result.tripwire_triggered:
                            raise GuardrailTripwireTriggered(result)

                return results

            except GuardrailTripwireTriggered:
                if suppress_tripwire:
                    return results
                raise

        async def _handle_llm_response(
            self,
            llm_response: OpenAIResponseType,
            preflight_results: list[GuardrailResult],
            input_results: list[GuardrailResult],
            conversation_history: list = None,
            suppress_tripwire: bool = False,
        ) -> GuardrailsResponse:
            """Handle non-streaming LLM response with output guardrails (async)."""
            # Create complete conversation history including the LLM response
            complete_conversation = self._append_llm_response_to_conversation(conversation_history, llm_response)

            response_text = self._extract_response_text(llm_response)
            output_results = await self._run_stage_guardrails(
                "output",
                response_text,
                conversation_history=complete_conversation,
                suppress_tripwire=suppress_tripwire,
            )

            return self._create_guardrails_response(llm_response, preflight_results, input_results, output_results)


if AzureOpenAI is not None:

    class GuardrailsAzureOpenAI(AzureOpenAI, GuardrailsBaseClient, StreamingMixin):  # type: ignore
        """AzureOpenAI subclass with automatic guardrail integration (sync)."""

        def __init__(
            self,
            config: str | Path | dict[str, Any],
            raise_guardrail_errors: bool = False,
            **azure_kwargs: Any,
        ) -> None:
            """Initialize GuardrailsAzureOpenAI client.

            Args:
                config: Path to pipeline configuration file or configuration dict.
                raise_guardrail_errors: If True, raise exceptions when guardrails fail to execute.
                    If False (default), treat guardrail execution errors as safe and continue.
                    Note: Tripwires (guardrail violations) are handled separately and not affected
                    by this parameter.
                **azure_kwargs: Additional arguments passed to AzureOpenAI constructor.
            """
            azure_kwargs = prepare_openai_kwargs(azure_kwargs)
            super().__init__(**azure_kwargs)

            # Store the error handling preference
            self.raise_guardrail_errors = raise_guardrail_errors

            from openai import AzureOpenAI as _AzureOpenAI  # type: ignore

            # Persist azure kwargs
            self._azure_kwargs: dict[str, Any] = dict(azure_kwargs)
            self._initialize_client(config, azure_kwargs, _AzureOpenAI)

        def _create_default_context(self) -> GuardrailLLMContextProto:
            try:
                return super()._create_default_context()
            except NotImplementedError:
                pass

            @dataclass
            class DefaultContext:
                guardrail_llm: Any  # AzureOpenAI

            from openai import AzureOpenAI as _AzureOpenAI  # type: ignore

            guardrail_client = _AzureOpenAI(**self._azure_kwargs)
            return DefaultContext(guardrail_llm=guardrail_client)

        def _create_context_with_conversation(self, conversation_history: list) -> GuardrailLLMContextProto:
            """Create a context with conversation history for prompt injection detection guardrail."""

            # Create a new context that includes conversation history
            @dataclass
            class ConversationContext:
                guardrail_llm: Any  # AzureOpenAI
                conversation_history: list

                def get_conversation_history(self) -> list:
                    return self.conversation_history

            return ConversationContext(
                guardrail_llm=self.context.guardrail_llm,
                conversation_history=conversation_history,
            )

        def _append_llm_response_to_conversation(self, conversation_history: list | str, llm_response: Any) -> list:
            """Append LLM response to conversation history as-is."""
            if conversation_history is None:
                conversation_history = []

            # Handle case where conversation_history is a string (from single input)
            if isinstance(conversation_history, str):
                conversation_history = [{"role": "user", "content": conversation_history}]

            # Make a copy to avoid modifying the original
            updated_history = conversation_history.copy()

            # For responses API: append the output directly
            if hasattr(llm_response, "output") and llm_response.output:
                updated_history.extend(llm_response.output)
            # For chat completions: append the choice message directly (prompt injection detection check will parse)
            elif hasattr(llm_response, "choices") and llm_response.choices:
                updated_history.append(llm_response.choices[0])

            return updated_history

        def _override_resources(self):
            from .resources.chat import Chat
            from .resources.responses import Responses

            object.__setattr__(self, "chat", Chat(self))
            object.__setattr__(self, "responses", Responses(self))

        def _run_stage_guardrails(
            self,
            stage_name: str,
            text: str,
            conversation_history: list = None,
            suppress_tripwire: bool = False,
        ) -> list[GuardrailResult]:
            """Run guardrails for a specific pipeline stage (synchronous)."""
            if not self.guardrails[stage_name]:
                return []

            # For sync version, we need to run async guardrails in sync context
            import asyncio

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            async def _run_async():
                # Check if prompt injection detection guardrail is present and we have conversation history
                has_injection_detection = any(
                    guardrail.definition.name.lower() == "prompt injection detection" for guardrail in self.guardrails[stage_name]
                )

                if has_injection_detection and conversation_history:
                    ctx = self._create_context_with_conversation(conversation_history)
                else:
                    ctx = self.context

                results = await run_guardrails(
                    ctx=ctx,
                    data=text,
                    media_type="text/plain",
                    guardrails=self.guardrails[stage_name],
                    suppress_tripwire=suppress_tripwire,
                    stage_name=stage_name,
                    raise_guardrail_errors=self.raise_guardrail_errors,
                )

                # Check for tripwire triggers unless suppressed
                if not suppress_tripwire:
                    for result in results:
                        if result.tripwire_triggered:
                            raise GuardrailTripwireTriggered(result)

                return results

            try:
                return loop.run_until_complete(_run_async())
            except GuardrailTripwireTriggered:
                if suppress_tripwire:
                    return []
                raise

        def _handle_llm_response(
            self,
            llm_response: OpenAIResponseType,
            preflight_results: list[GuardrailResult],
            input_results: list[GuardrailResult],
            conversation_history: list = None,
            suppress_tripwire: bool = False,
        ) -> GuardrailsResponse:
            """Handle LLM response with output guardrails (sync)."""
            # Create complete conversation history including the LLM response
            complete_conversation = self._append_llm_response_to_conversation(conversation_history, llm_response)

            response_text = self._extract_response_text(llm_response)
            output_results = self._run_stage_guardrails(
                "output",
                response_text,
                conversation_history=complete_conversation,
                suppress_tripwire=suppress_tripwire,
            )

            return self._create_guardrails_response(llm_response, preflight_results, input_results, output_results)
