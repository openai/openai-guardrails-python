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
from guardrails.types import CheckFn, GuardrailLLMContextProto, GuardrailResult
from guardrails.utils.output import OutputSchema

if TYPE_CHECKING:
    from openai import AsyncAzureOpenAI, AzureOpenAI  # type: ignore[unused-import]
else:
    try:
        from openai import AsyncAzureOpenAI, AzureOpenAI  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        AsyncAzureOpenAI = object  # type: ignore[assignment]
        AzureOpenAI = object  # type: ignore[assignment]

logger = logging.getLogger(__name__)


__all__ = ["LLMConfig", "LLMOutput", "LLMErrorOutput", "create_llm_check_fn"]


class LLMConfig(BaseModel):
    """Configuration schema for LLM-based content checks.

    Used to specify the LLM model and confidence threshold for triggering a tripwire.

    Attributes:
        model (str): The LLM model to use for checking the text.
        confidence_threshold (float): Minimum confidence required to trigger the guardrail,
            as a float between 0.0 and 1.0.
    """

    model: str = Field(..., description="LLM model to use for checking the text")
    confidence_threshold: float = Field(
        0.7,
        description="Minimum confidence threshold to trigger the guardrail (0.0 to 1.0). Defaults to 0.7.",
        ge=0.0,
        le=1.0,
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


def _build_full_prompt(system_prompt: str) -> str:
    """Assemble a complete LLM prompt with instructions and response schema.

    Incorporates the supplied system prompt and specifies the required JSON response fields.

    Args:
        system_prompt (str): The instructions describing analysis criteria.

    Returns:
        str: Formatted prompt string for LLM input.
    """
    template = """
    {system_prompt}

    Respond with a json object containing:
    - "flagged": boolean (true if detected and false otherwise)
    - "confidence": float (0.0 to 1.0)

    Only respond with the json object, nothing else.

    **IMPORTANT:**
    You must output a confidence score reflecting how likely the input is violative of the guardrail:
    - 1.0 = Certain violative
    - 0.0 = Certain not violative
    - Use the full range [0.0 - 1.0] to reflect your level of certainty

    Analyze the following text according to the instructions above.
    """
    return textwrap.dedent(template).strip().format(system_prompt=system_prompt)


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
    return await _invoke_openai_callable(
        client.chat.completions.create,
        messages=messages,
        model=model,
        response_format=response_format,
    )


async def run_llm(
    text: str,
    system_prompt: str,
    client: AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI,
    model: str,
    output_model: type[LLMOutput],
) -> LLMOutput:
    """Run an LLM analysis for a given prompt and user input.

    Invokes the OpenAI LLM, enforces prompt/response contract, parses the LLM's
    output, and returns a validated result.

    Args:
        text (str): Text to analyze.
        system_prompt (str): Prompt instructions for the LLM.
        client (AsyncOpenAI | OpenAI | AsyncAzureOpenAI | AzureOpenAI): OpenAI client used for guardrails.
        model (str): Identifier for which LLM model to use.
        output_model (type[LLMOutput]): Model for parsing and validating the LLM's response.

    Returns:
        LLMOutput: Structured output containing the detection decision and confidence.
    """
    full_prompt = _build_full_prompt(system_prompt)

    try:
        response = await _request_chat_completion(
            client=client,
            messages=[
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": f"# Text\n\n{text}"},
            ],
            model=model,
            response_format=OutputSchema(output_model).get_completions_format(),  # type: ignore[arg-type, unused-ignore]
        )
        result = response.choices[0].message.content
        if not result:
            return output_model(
                flagged=False,
                confidence=0.0,
            )
        result = _strip_json_code_fence(result)
        return output_model.model_validate_json(result)

    except Exception as exc:
        logger.exception("LLM guardrail failed for prompt: %s", system_prompt)

        # Check if this is a content filter error - Azure OpenAI
        if "content_filter" in str(exc):
            logger.warning("Content filter triggered by provider: %s", exc)
            return LLMErrorOutput(
                flagged=True,
                confidence=1.0,
                info={
                    "third_party_filter": True,
                    "error_message": str(exc),
                },
            )
        # Always return error information for other LLM failures
        return LLMErrorOutput(
            flagged=False,
            confidence=0.0,
            info={
                "error_message": str(exc),
            },
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

        analysis = await run_llm(
            data,
            rendered_system_prompt,
            ctx.guardrail_llm,
            config.model,
            output_model,
        )

        # Check if this is an error result
        if isinstance(analysis, LLMErrorOutput):
            # Extract error information from the LLMErrorOutput
            error_info = analysis.info if hasattr(analysis, "info") else {}
            error_message = error_info.get("error_message", "LLM execution failed")

            return GuardrailResult(
                tripwire_triggered=False,  # Don't trigger tripwire on execution errors
                execution_failed=True,
                original_exception=Exception(error_message),  # Create exception from error message
                info={
                    "guardrail_name": name,
                    "checked_text": data,
                    "error": error_message,
                    **analysis.model_dump(),
                },
            )

        # Compare severity levels
        is_trigger = analysis.flagged and analysis.confidence >= config.confidence_threshold
        return GuardrailResult(
            tripwire_triggered=is_trigger,
            info={
                "guardrail_name": name,
                **analysis.model_dump(),
                "threshold": config.confidence_threshold,
                "checked_text": data,  # LLM-based guardrails don't modify text, pass through unchanged
            },
        )

    guardrail_func.__annotations__["config"] = config_model

    default_spec_registry.register(
        name=name,
        check_fn=guardrail_func,
        description=description,
        media_type="text/plain",
        metadata=GuardrailSpecMetadata(engine="LLM"),
    )

    return guardrail_func
