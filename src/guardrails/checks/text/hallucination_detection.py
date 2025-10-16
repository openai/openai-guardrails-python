"""Hallucination Detection guardrail module.

This module provides a guardrail for detecting when an LLM generates content that
may be factually incorrect, unsupported, or "hallucinated." It uses the OpenAI
Responses API with file search to validate claims against actual documents.

**IMPORTANT: A valid OpenAI vector store must be created before using this guardrail.**

To create an OpenAI vector store, you can:

1. **Use the Guardrails Wizard**: Configure the guardrail through the [Guardrails Wizard](https://guardrails.openai.com/), which provides an option to create a vector store if you don't already have one.
2. **Use the OpenAI Dashboard**: Create a vector store directly in the [OpenAI Dashboard](https://platform.openai.com/storage/vector_stores/).
3. **Follow OpenAI Documentation**: Refer to the "Create a vector store and upload a file" section of the [File Search documentation](https://platform.openai.com/docs/guides/tools-file-search) for detailed instructions.
4. **Use the provided utility script**: Use the `create_vector_store.py` script provided in the [repo](https://github.com/openai/openai-guardrails-python/blob/main/src/guardrails/utils/create_vector_store.py) to create a vector store from local files or directories.

**Pricing**: For pricing details on file search and vector storage, see the [Built-in tools section](https://openai.com/api/pricing/) of the OpenAI pricing page.

Classes:
    HallucinationDetectionConfig: Configuration schema for hallucination detection.
    HallucinationDetectionOutput: Output schema for hallucination analysis results.

Constants:
    VALIDATION_PROMPT: Pre-defined prompt for hallucination detection.

Configuration Parameters:
    - `model` (str): The LLM model to use for analysis (e.g., "gpt-4o-mini")
    - `confidence_threshold` (float): Minimum confidence score (0.0 to 1.0) required to
        trigger the guardrail. Defaults to 0.7.
    - `knowledge_source` (str): Vector store ID to use for document validation.

Examples:
```python
    >>> config = HallucinationDetectionConfig(
    ...     model="gpt-4.1-mini",
    ...     confidence_threshold=0.8,
    ...     knowledge_source="vs_abc123"
    ... )
    >>> result = await hallucination_detection(ctx, "Our premium plan is $199/month...", config)
    >>> result.tripwire_triggered
    True
```
"""  # noqa: E501

from __future__ import annotations

import logging
import textwrap

from pydantic import ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailLLMContextProto, GuardrailResult

from .llm_base import LLMConfig, LLMOutput, _invoke_openai_callable

logger = logging.getLogger(__name__)

__all__ = ["hallucination_detection"]


class HallucinationDetectionConfig(LLMConfig):
    """Configuration schema for hallucination detection.

    Extends the base LLM configuration with file search validation parameters.

    Attributes:
        knowledge_source: Vector store ID to use for document validation.
    """

    knowledge_source: str = Field(
        ...,
        description="Vector store ID to use for document validation (must start with 'vs_').",
    )

    model_config = ConfigDict(extra="forbid")


class HallucinationDetectionOutput(LLMOutput):
    """Output schema for hallucination analysis.

    Extends the base LLM output with hallucination-specific details.

    Attributes:
        hallucination_type (str | None): Type of hallucination detected.
        hallucinated_statements (list[str] | None): Specific statements flagged as potentially hallucinated.
        verified_statements (list[str] | None): Specific statements that are supported by the documents.
    """

    hallucination_type: str | None = Field(
        None,
        description="Type of hallucination detected (e.g., 'factual_error', 'unsupported_claim').",
    )
    hallucinated_statements: list[str] | None = Field(
        None,
        description="Specific statements or claims that may be hallucinated.",
    )
    verified_statements: list[str] | None = Field(
        None,
        description="Specific statements that are supported by the documents.",
    )


VALIDATION_PROMPT = textwrap.dedent(
    """\
    You are an expert at detecting hallucinations in AI-generated text. Your task is to analyze the given text and determine if it contains any FACTUAL CLAIMS that are clearly contradicted or completely unsupported by the provided documents.

    **IMPORTANT: Flag content that contains CLEAR CONTRADICTIONS or COMPLETELY UNSUPPORTED claims.**

    A factual claim is a specific statement about:
    - Product features, specifications, or capabilities
    - Prices, costs, fees, or pricing information
    - Policies, terms, conditions, or rules
    - Dates, times, schedules, or availability
    - Names, titles, contact information, or credentials
    - Statistics, numbers, percentages, or measurements
    - Specific offers, discounts, or promotions
    - Technical details, requirements, or limitations
    - Company information, history, or facts
    - Legal or regulatory information

    **DO NOT flag:**
    - General greetings, pleasantries, or conversational content
    - Opinions, preferences, or subjective statements
    - General advice or suggestions without specific claims
    - Questions or requests for information
    - Emotional expressions or personal statements
    - Generic statements that don't contain verifiable facts
    - **ANY FACTUAL CLAIMS THAT ARE SUPPORTED BY THE DOCUMENTS**
    - **INCOMPLETE BUT ACCURATE STATEMENTS** (statements that are true but don't mention all available options or details)
    - **MINOR PHRASING DIFFERENCES** (if the core meaning is correct, don't flag)
    - **SIMILAR BUT NOT IDENTICAL WORDING** (if the claim is essentially true, don't flag)

    Examples of content that SHOULD be flagged:
    - "Our premium plan costs $299/month" (when documents say $199/month - clear contradiction)
    - "We offer 24/7 customer support" (when documents don't mention this at all - completely unsupported)
    - "The warranty covers 60 days" (when documents say 30 days - clear contradiction)
    - **Statements that contain any contradictions or unsupported claims. Even if some claims are supported.**
    - "Our jacket is $59.95 and comes in red, and blue" (When the document confirms the price but contradicts the available colors)

    Examples of content that should NOT be flagged:
    - "Hello, how can I help you today?" (conversational)
    - "I'd be happy to assist you" (conversational)
    - "That sounds like a great idea" (opinion)
    - "Let me check our documentation" (intention)
    - "Thank you for your patience" (courtesy)
    - "Our premium plan costs $199/month" (when documents confirm this price - supported)
    - "You earn 1% back" (when documents confirm this rate - supported, even if incomplete)

    Analyze the text and determine if it contains factual claims that are:
    1. **Supported by the documents** - Claims that are directly supported by citations (even if not complete) → DO NOT FLAG
    2. **Essentially correct** - Claims that have minor phrasing differences but same core meaning → DO NOT FLAG
    3. **Clearly contradicted by the documents** - Claims that directly contradict the documents → FLAG
    4. **Completely unsupported by the documents** - Claims that cannot be verified from the documents → FLAG

    Respond with a JSON object containing:
    - "flagged": boolean (true if ANY factual claims are clearly contradicted or completely unsupported)
    - "confidence": float (0.0 to 1.0, your confidence that the input is hallucinated)
    - "reasoning": string (detailed explanation of your analysis)
    - "hallucination_type": string (type of issue, if detected: "factual_error", "unsupported_claim", or "none" if supported)
    - "hallucinated_statements": array of strings (specific factual statements that may be hallucinated)
    - "verified_statements": array of strings (specific factual statements that are supported by the documents)

    **CRITICAL GUIDELINES**:
    - Flag content if ANY factual claims are unsupported or contradicted (even if some claims are supported)
    - Allow conversational, opinion-based, or general content to pass through
    - Allow content to pass through ONLY if ALL factual claims are supported by documents
    - Allow minor phrasing differences if the core meaning is correct
    - Focus on claims that could mislead users about products, services, policies, or facts
    - You must output a confidence score reflecting how likely the input is hallucinated:
        - 1.0 = Certain hallucinated
        - 0.0 = Certain not hallucinated
        - Use the full range [0.0 - 1.0] to reflect your level of certainty
    """  # noqa: E501
).strip()


async def hallucination_detection(
    ctx: GuardrailLLMContextProto,
    candidate: str,
    config: HallucinationDetectionConfig,
) -> GuardrailResult:
    """Detect potential hallucinations in text by validating against documents.

    This function uses the OpenAI Responses API with file search and structured output
    to validate factual claims in the candidate text against the provided knowledge source.
    It flags content that contains any unsupported or contradicted factual claims.

    Args:
        ctx: Guardrail context containing the LLM client.
        candidate: Text to analyze for potential hallucinations.
        config: Configuration for hallucination detection.

    Returns:
        GuardrailResult containing hallucination analysis with flagged status
        and confidence score.

    Raises:
        ValueError: If knowledge_source is invalid or LLM response is malformed.
        Exception: For API errors or processing failures.
    """
    if not config.knowledge_source or not config.knowledge_source.startswith("vs_"):
        raise ValueError("knowledge_source must be a valid vector store ID starting with 'vs_'")

    try:
        # Create the validation query
        validation_query = f"{VALIDATION_PROMPT}\n\nText to validate:\n{candidate}"

        # Use the Responses API with file search and structured output
        response = await _invoke_openai_callable(
            ctx.guardrail_llm.responses.parse,
            input=validation_query,
            model=config.model,
            text_format=HallucinationDetectionOutput,
            tools=[{"type": "file_search", "vector_store_ids": [config.knowledge_source]}],
        )

        # Get the parsed output directly
        analysis = response.output_parsed

        # Determine if tripwire should be triggered
        is_trigger = analysis.flagged and analysis.confidence >= config.confidence_threshold

        return GuardrailResult(
            tripwire_triggered=is_trigger,
            info={
                "guardrail_name": "Hallucination Detection",
                **analysis.model_dump(),
                "threshold": config.confidence_threshold,
                "checked_text": candidate,  # Hallucination Detection doesn't modify text, pass through unchanged
            },
        )

    except ValueError as e:
        # Log validation errors but return safe default
        logger.warning(f"Validation error in hallucination_detection: {e}")
        return GuardrailResult(
            tripwire_triggered=False,
            info={
                "guardrail_name": "Hallucination Detection",
                "flagged": False,
                "confidence": 0.0,
                "reasoning": f"Validation failed: {str(e)}",
                "hallucination_type": None,
                "hallucinated_statements": None,
                "verified_statements": None,
                "threshold": config.confidence_threshold,
                "error": str(e),
                "checked_text": candidate,  # Hallucination Detection doesn't modify text, pass through unchanged
            },
        )
    except Exception as e:
        # Log unexpected errors and return safe default
        logger.exception("Unexpected error in hallucination_detection")
        return GuardrailResult(
            tripwire_triggered=False,
            info={
                "guardrail_name": "Hallucination Detection",
                "flagged": False,
                "confidence": 0.0,
                "reasoning": f"Analysis failed: {str(e)}",
                "hallucination_type": None,
                "hallucinated_statements": None,
                "verified_statements": None,
                "threshold": config.confidence_threshold,
                "error": str(e),
                "checked_text": candidate,  # Hallucination Detection doesn't modify text, pass through unchanged
            },
        )


# Register the guardrail
default_spec_registry.register(
    name="Hallucination Detection",
    check_fn=hallucination_detection,
    description=(
        "Detects potential hallucinations in AI-generated text using OpenAI "
        "Responses API with file search. Validates claims against actual documents "
        "and flags factually incorrect, unsupported, or potentially fabricated information."
    ),
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="FileSearch"),
)
