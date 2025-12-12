"""Tests for hallucination detection guardrail."""

from __future__ import annotations

from typing import Any

import pytest

from guardrails.checks.text.hallucination_detection import (
    HallucinationDetectionConfig,
    HallucinationDetectionOutput,
    hallucination_detection,
)
from guardrails.checks.text.llm_base import LLMOutput
from guardrails.types import TokenUsage


def _mock_token_usage() -> TokenUsage:
    """Return a mock TokenUsage for tests."""
    return TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


class _FakeResponse:
    """Fake response from responses.parse."""

    def __init__(self, parsed_output: Any, usage: TokenUsage) -> None:
        self.output_parsed = parsed_output
        self.usage = usage


class _FakeGuardrailLLM:
    """Fake guardrail LLM client."""

    def __init__(self, response: _FakeResponse) -> None:
        self._response = response
        self.responses = self

    async def parse(self, **kwargs: Any) -> _FakeResponse:
        """Mock parse method."""
        return self._response


class _FakeContext:
    """Context stub providing LLM client."""

    def __init__(self, llm_response: _FakeResponse) -> None:
        self.guardrail_llm = _FakeGuardrailLLM(llm_response)


@pytest.mark.asyncio
async def test_hallucination_detection_includes_reasoning_when_enabled() -> None:
    """When include_reasoning=True, output should include reasoning and detail fields."""
    parsed_output = HallucinationDetectionOutput(
        flagged=True,
        confidence=0.95,
        reasoning="The claim contradicts documented information",
        hallucination_type="factual_error",
        hallucinated_statements=["Premium plan costs $299/month"],
        verified_statements=["Customer support available"],
    )
    response = _FakeResponse(parsed_output, _mock_token_usage())
    context = _FakeContext(response)

    config = HallucinationDetectionConfig(
        model="gpt-test",
        confidence_threshold=0.7,
        knowledge_source="vs_test123",
        include_reasoning=True,
    )

    result = await hallucination_detection(context, "Test claim", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["flagged"] is True  # noqa: S101
    assert result.info["confidence"] == 0.95  # noqa: S101
    assert "reasoning" in result.info  # noqa: S101
    assert result.info["reasoning"] == "The claim contradicts documented information"  # noqa: S101
    assert "hallucination_type" in result.info  # noqa: S101
    assert result.info["hallucination_type"] == "factual_error"  # noqa: S101
    assert "hallucinated_statements" in result.info  # noqa: S101
    assert result.info["hallucinated_statements"] == ["Premium plan costs $299/month"]  # noqa: S101
    assert "verified_statements" in result.info  # noqa: S101
    assert result.info["verified_statements"] == ["Customer support available"]  # noqa: S101


@pytest.mark.asyncio
async def test_hallucination_detection_excludes_reasoning_when_disabled() -> None:
    """When include_reasoning=False (default), output should only include flagged and confidence."""
    parsed_output = LLMOutput(
        flagged=False,
        confidence=0.2,
    )
    response = _FakeResponse(parsed_output, _mock_token_usage())
    context = _FakeContext(response)

    config = HallucinationDetectionConfig(
        model="gpt-test",
        confidence_threshold=0.7,
        knowledge_source="vs_test123",
        include_reasoning=False,
    )

    result = await hallucination_detection(context, "Test claim", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["flagged"] is False  # noqa: S101
    assert result.info["confidence"] == 0.2  # noqa: S101
    assert "reasoning" not in result.info  # noqa: S101
    assert "hallucination_type" not in result.info  # noqa: S101
    assert "hallucinated_statements" not in result.info  # noqa: S101
    assert "verified_statements" not in result.info  # noqa: S101


@pytest.mark.asyncio
async def test_hallucination_detection_requires_valid_vector_store() -> None:
    """Should raise ValueError if knowledge_source is invalid."""
    context = _FakeContext(_FakeResponse(LLMOutput(flagged=False, confidence=0.0), _mock_token_usage()))

    # Missing vs_ prefix
    config = HallucinationDetectionConfig(
        model="gpt-test",
        confidence_threshold=0.7,
        knowledge_source="invalid_id",
    )

    with pytest.raises(ValueError, match="knowledge_source must be a valid vector store ID starting with 'vs_'"):
        await hallucination_detection(context, "Test", config)

    # Empty string
    config_empty = HallucinationDetectionConfig(
        model="gpt-test",
        confidence_threshold=0.7,
        knowledge_source="",
    )

    with pytest.raises(ValueError, match="knowledge_source must be a valid vector store ID starting with 'vs_'"):
        await hallucination_detection(context, "Test", config_empty)

