"""LLM-based guardrail content checking.

This module enables the creation and registration of content moderation guardrails
using Large Language Models (LLMs). It provides configuration and output schemas,
prompt helpers, a utility for executing LLM-based checks, and a factory for generating
guardrail check functions leveraging LLMs.

Simply add your own system prompt to create a new guardrail. See `Off Topic Prompts` for an example.

Classes:
    LLMConfig: Configuration schema for parameterizing LLM-based guardrails.
    LLMOutput: Output schema for results from LLM analysis.
    LLMErrorOutput: Extended LLM output schema with error information.

Functions:
    run_llm: Run an LLM analysis and return structured output.
    create_llm_check_fn: Factory for building and registering LLM-based guardrails.

Examples:
```python
    from guardrails.types import CheckFn
    class MyLLMOutput(LLMOutput):
        my_guardrail = create_llm_check_fn(
            name="MyCheck",
            description="Checks for risky language.",
            system_prompt="Check for risky content.",
            output_model=MyLLMOutput,
    )
```
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import logging
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import (
    CheckFn,
    GuardrailLLMContextProto,
    GuardrailResult,
    TokenUsage,
    extract_token_usage,
    token_usage_to_dict,
)
from guardrails.utils.output import OutputSchema

from ...utils.safety_identifier import SAFETY_IDENTIFIER, supports_safety_identifier

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI, AzureOpenAI  # type: ignore[unused-import]
else:
    try:
        from openai import AsyncAzureOpenAI, AzureOpenAI  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        AsyncAzureOpenAI = object  # type: ignore[assignment]
        AzureOpenAI = object  # type: ignore[assignment]

logger = logging.getLogger(__name__)


__all__ = [
    "LLMConfig",
    "LLMErrorOutput",
    "LLMOutput",
    "create_error_result",
    "create_llm_check_fn",
]


class LLMConfig(BaseModel):
    """Configuration schema for LLM-based content checks.

    Used to specify the LLM model, confidence threshold, and conversation history
    settings for triggering a tripwire.

    Attributes:
        model (str): The LLM model to use for checking the text.
        confidence_threshold (float): Minimum confidence required to trigger the guardrail,
            as a float between 0.0 and 1.0.
        max_turns (int): Maximum number of conversation turns to include in analysis.
            Set to 1 for single-turn behavior. Defaults to 10.
    """

    model: str = Field(..., description="LLM model to use for checking the text")
    confidence_threshold: float = Field(
        0.7,
        description="Minimum confidence threshold to trigger the guardrail (0.0 to 1.0). Defaults to 0.7.",
        ge=0.0,
        le=1.0,
    )
    max_turns: int = Field(
        10,
        description="Maximum conversation turns to include in analysis. Set to 1 for single-turn. Defaults to 10.",
        ge=1,
    )

    model_config = ConfigDict(extra="forbid")


TLLMCfg = TypeVar("TLLMCfg", bound=LLMConfig)


class LLMOutput(BaseModel):
    """Output schema for LLM content checks.

    Used for structured results returned by LLM-based moderation guardrails.

    Attributes:
        flagged (bool): Indicates whether the content was flagged.
        confidence (float): LLM's confidence in the flagging decision (0.0 to 1.0).
    """

    flagged: bool
    confidence: float


class LLMErrorOutput(LLMOutput):
    """Extended LLM output schema with error information.

    Extends LLMOutput to include additional information about errors that occurred
    during LLM processing, such as content filter triggers.

    Attributes:
        info (dict): Additional information about the error.
    """

    info: dict


def create_error_result(
    guardrail_name: str,
    analysis: LLMErrorOutput,
    additional_info: dict[str, Any] | None = None,
    token_usage: TokenUsage | None = None,
) -> GuardrailResult:
    """Create a standardized GuardrailResult from an LLM error output.

    Args:
        guardrail_name: Name of the guardrail that failed.
        analysis: The LLM error output.
        additional_info: Optional additional fields to include in info dict.
        token_usage: Optional token usage statistics from the LLM call.

    Returns:
        GuardrailResult with execution_failed=True.
    """
    error_info = getattr(analysis, "info", {})
    error_message = error_info.get("error_message", "LLM execution failed")

    result_info: dict[str, Any] = {
        "guardrail_name": guardrail_name,
        "error": error_message,
        **analysis.model_dump(),
    }

    if additional_info:
        result_info.update(additional_info)

    # Include token usage if provided
    if token_usage is not None:
        result_info["token_usage"] = token_usage_to_dict(token_usage)

    return GuardrailResult(
        tripwire_triggered=False,
        execution_failed=True,
        original_exception=Exception(error_message),
        info=result_info,
    )


def _format_field_instruction(name: str, annotation: Any) -> str:
    """Return a natural language description for a response field."""
    if name == "flagged":
        return '- "flagged": boolean (true if detected and false otherwise)'
    if name == "confidence":
        return '- "confidence": float (0.0 to 1.0)'
    if name == "reason":
        return '- "reason": string explaining the rationale for the decision'

    type_name = getattr(annotation, "__name__", None) or str(annotation)
    normalized_type = type_name.lower()
    if "bool" in normalized_type:
        value_type = "boolean"
    elif "float" in normalized_type:
        value_type = "float"
    elif "int" in normalized_type:
        value_type = "integer"
    elif "str" in normalized_type:
        value_type = "string"
    else:
        value_type = normalized_type
    return f'- "{name}": {value_type}'


def _build_full_prompt(system_prompt: str, output_model: type[LLMOutput]) -> str:
    """Assemble a complete LLM prompt with instructions and response schema.

    Incorporates the supplied system prompt and specifies the required JSON response fields.

    Args:
        system_prompt (str): The instructions describing analysis criteria.
        output_model (type[LLMOutput]): Pydantic model defining expected response structure.

    Returns:
        str: Formatted prompt string for LLM input.
    """
    template = """
    {system_prompt}

    Respond with a json object containing:
    {field_instructions}

    Only respond with the json object, nothing else.

    **IMPORTANT:**
    You must output a confidence score reflecting how likely the input is violative of the guardrail:
    - 1.0 = Certain violative
    - 0.0 = Certain not violative
    - Use the full range [0.0 - 1.0] to reflect your level of certainty

    Analyze the following text according to the instructions above.
    """
    field_instructions = "\n".join(_format_field_instruction(name, field.annotation) for name, field in output_model.model_fields.items())
    return (
        textwrap.dedent(template)
        .strip()
        .format(
            system_prompt=system_prompt,
            field_instructions=field_instructions,
        )
    )


def _strip_json_code_fence(text: str) -> str:
    """Remove JSON code fencing (```json ... ```) from a response, if present.

    This function is defensive: it returns the input string unchanged unless
    a valid JSON code fence is detected and parseable.

    Args:
        text (str): LLM output, possibly wrapped in a JSON code fence.

    Returns:
        str: Extracted JSON string or the original string.
    """
    lines = text.strip().splitlines()
    if len(lines) < 3:
        return text

    first, *body, last = lines
    if not first.startswith("```json") or last != "```":
        return text

    candidate = "\n".join(body)
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return text

    return candidate


async def _invoke_openai_callable(
    method: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Invoke OpenAI SDK methods that may be sync or async."""
    if inspect.iscoroutinefunction(method):
        return await method(*args, **kwargs)

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        functools.partial(method, *args, **kwargs),
    )
    if inspect.isawaitable(result):
        return await result
    return result


async def _request_chat_completion(
    client: AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI,
    *,
    messages: list[dict[str, str]],
    model: str,
    response_format: dict[str, Any],
) -> Any:
    """Invoke chat.completions.create on sync or async OpenAI clients."""
    # Only include safety_identifier for official OpenAI API
    kwargs: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "response_format": response_format,
    }

    # Only official OpenAI API supports safety_identifier (not Azure or local models)
    if supports_safety_identifier(client):
        kwargs["safety_identifier"] = SAFETY_IDENTIFIER

    return await _invoke_openai_callable(client.chat.completions.create, **kwargs)


def _build_analysis_payload(
    conversation_history: list[dict[str, Any]] | None,
    latest_input: str,
    max_turns: int,
) -> str:
    """Build a JSON payload with conversation history and latest input.

    Args:
        conversation_history: List of normalized conversation entries.
        latest_input: The current text being analyzed.
        max_turns: Maximum number of conversation turns to include.

    Returns:
        JSON string with conversation context and latest input.
    """
    trimmed_input = latest_input.strip()
    recent_turns = (conversation_history or [])[-max_turns:]
    payload = {
        "conversation": recent_turns,
        "latest_input": trimmed_input,
    }
    return json.dumps(payload, ensure_ascii=False)


async def run_llm(
    text: str,
    system_prompt: str,
    client: AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI,
    model: str,
    output_model: type[LLMOutput],
    conversation_history: list[dict[str, Any]] | None = None,
    max_turns: int = 10,
) -> tuple[LLMOutput, TokenUsage]:
    """Run an LLM analysis for a given prompt and user input.

    Invokes the OpenAI LLM, enforces prompt/response contract, parses the LLM's
    output, and returns a validated result along with token usage statistics.

    When conversation_history is provided and max_turns > 1, the analysis
    includes conversation context formatted as a JSON payload with the
    structure: {"conversation": [...], "latest_input": "..."}.

    Args:
        text (str): Text to analyze.
        system_prompt (str): Prompt instructions for the LLM.
        client (AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI): OpenAI client used for guardrails.
        model (str): Identifier for which LLM model to use.
        output_model (type[LLMOutput]): Model for parsing and validating the LLM's response.
        conversation_history (list[dict[str, Any]] | None): Optional normalized
            conversation history for multi-turn analysis. Defaults to None.
        max_turns (int): Maximum number of conversation turns to include.
            Defaults to 10. Set to 1 for single-turn behavior.

    Returns:
        tuple[LLMOutput, TokenUsage]: A tuple containing:
            - Structured output with the detection decision and confidence.
            - Token usage statistics from the LLM call.
    """
    full_prompt = _build_full_prompt(system_prompt, output_model)

    # Default token usage for error cases
    no_usage = TokenUsage(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        unavailable_reason="LLM call failed before usage could be recorded",
    )

    # Build user content based on whether conversation history is available
    # and whether we're in multi-turn mode (max_turns > 1)
    has_conversation = conversation_history and len(conversation_history) > 0
    use_multi_turn = has_conversation and max_turns > 1

    if use_multi_turn:
        # Multi-turn: build JSON payload with conversation context
        analysis_payload = _build_analysis_payload(conversation_history, text, max_turns)
        user_content = f"# Analysis Input\n\n{analysis_payload}"
    else:
        # Single-turn: use text directly (strip whitespace for consistency)
        user_content = f"# Text\n\n{text.strip()}"

    try:
        response = await _request_chat_completion(
            client=client,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": user_content},
            ],
            model=model,
            response_format=OutputSchema(output_model).get_completions_format(),  # type: ignore[arg-type, unused-ignore]
        )

        # Extract token usage from the response
        token_usage = extract_token_usage(response)

        result = response.choices[0].message.content
        if not result:
            return (
                output_model(
                    flagged=False,
                    confidence=0.0,
                ),
                token_usage,
            )
        result = _strip_json_code_fence(result)
        return output_model.model_validate_json(result), token_usage

    except Exception as exc:
        logger.exception("LLM guardrail failed for prompt: %s", system_prompt)

        # Check if this is a content filter error - Azure OpenAI
        if "content_filter" in str(exc):
            logger.warning("Content filter triggered by provider: %s", exc)
            return (
                LLMErrorOutput(
                    flagged=True,
                    confidence=1.0,
                    info={
                        "third_party_filter": True,
                        "error_message": str(exc),
                    },
                ),
                no_usage,
            )
        # Always return error information for other LLM failures
        return (
            LLMErrorOutput(
                flagged=False,
                confidence=0.0,
                info={
                    "error_message": str(exc),
                },
            ),
            no_usage,
        )


def create_llm_check_fn(
    name: str,
    description: str,
    system_prompt: str,
    output_model: type[LLMOutput] = LLMOutput,
    config_model: type[TLLMCfg] = LLMConfig,  # type: ignore[assignment]
) -> CheckFn[GuardrailLLMContextProto, str, TLLMCfg]:
    """Factory for constructing and registering an LLM-based guardrail check_fn.

    This helper registers the guardrail with the default registry and returns a
    check_fn suitable for use in guardrail pipelines. The returned function will
    use the configured LLM to analyze text, validate the result, and trigger if
    confidence exceeds the provided threshold.

    All guardrails created with this factory automatically support multi-turn
    conversation analysis. Conversation history is extracted from the context
    and trimmed to the configured max_turns. Set max_turns=1 in config for
    single-turn behavior.

    Args:
        name (str): Name under which to register the guardrail.
        description (str): Short explanation of the guardrail's logic.
        system_prompt (str): Prompt passed to the LLM to control analysis.
        output_model (type[LLMOutput]): Schema for parsing the LLM output.
        config_model (type[LLMConfig]): Configuration schema for the check_fn.

    Returns:
        CheckFn[GuardrailLLMContextProto, str, TLLMCfg]: Async check function
            to be registered as a guardrail.
    """

    async def guardrail_func(
        ctx: GuardrailLLMContextProto,
        data: str,
        config: TLLMCfg,
    ) -> GuardrailResult:
        """Runs an LLM-based check_fn on text using the configured system prompt.

        Args:
            ctx (GuardrailLLMContextProto | Any): The guardrail context.
            data (str): The text data to analyze.
            config (LLMConfig): Configuration for the LLM check_fn.

        Returns:
            GuardrailResult: The result of the guardrail check_fn.
        """
        if spd := getattr(config, "system_prompt_details", None):
            rendered_system_prompt = system_prompt.format(system_prompt_details=spd)
        else:
            rendered_system_prompt = system_prompt

        # Extract conversation history from context if available
        conversation_history = getattr(ctx, "get_conversation_history", lambda: None)() or []

        # Get max_turns from config (default to 10 if not present for backward compat)
        max_turns = getattr(config, "max_turns", 10)

        analysis, token_usage = await run_llm(
            data,
            rendered_system_prompt,
            ctx.guardrail_llm,
            config.model,
            output_model,
            conversation_history=conversation_history,
            max_turns=max_turns,
        )

        # Check if this is an error result
        if isinstance(analysis, LLMErrorOutput):
            return create_error_result(
                guardrail_name=name,
                analysis=analysis,
                token_usage=token_usage,
            )

        # Compare severity levels
        is_trigger = analysis.flagged and analysis.confidence >= config.confidence_threshold
        return GuardrailResult(
            tripwire_triggered=is_trigger,
            info={
                "guardrail_name": name,
                **analysis.model_dump(),
                "threshold": config.confidence_threshold,
                "token_usage": token_usage_to_dict(token_usage),
            },
        )

    guardrail_func.__annotations__["config"] = config_model

    default_spec_registry.register(
        name=name,
        check_fn=guardrail_func,
        description=description,
        media_type="text/plain",
        metadata=GuardrailSpecMetadata(engine="LLM", uses_conversation_history=True),
    )

    return guardrail_func
