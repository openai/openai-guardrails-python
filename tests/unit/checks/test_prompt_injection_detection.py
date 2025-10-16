"""Tests for prompt injection detection guardrail."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from guardrails.checks.text import prompt_injection_detection as pid_module
from guardrails.checks.text.llm_base import LLMConfig
from guardrails.checks.text.prompt_injection_detection import (
    PromptInjectionDetectionOutput,
    _extract_user_intent_from_messages,
    _should_analyze,
    prompt_injection_detection,
)


class _FakeContext:
    """Context stub providing conversation history accessors."""

    def __init__(self, history: list[Any]) -> None:
        self._history = history
        self.guardrail_llm = SimpleNamespace()  # unused due to monkeypatch

    def get_conversation_history(self) -> list[Any]:
        return self._history


def _make_history(action: dict[str, Any]) -> list[Any]:
    return [
        {"role": "user", "content": "Retrieve the weather for Paris"},
        action,
    ]


@pytest.mark.parametrize(
    "message, expected",
    [
        ({"type": "function_call"}, True),
        ({"role": "tool", "content": "Tool output"}, True),
        ({"role": "assistant", "content": "hello"}, False),
    ],
)
def test_should_analyze(message: dict[str, Any], expected: bool) -> None:
    """Verify _should_analyze matches only tool-related messages."""
    assert _should_analyze(message) is expected  # noqa: S101


def test_extract_user_intent_from_messages_handles_content_parts() -> None:
    """User intent extraction should normalize list-based content payloads."""
    messages = [
        {"role": "user", "content": [{"type": "input_text", "text": "First chunk"}, "extra"]},
        {"role": "assistant", "content": "Response"},
        {"role": "user", "content": [{"type": "text", "text": "Second chunk"}, {"type": "text", "content": "ignored"}]},
    ]

    result = _extract_user_intent_from_messages(messages)

    assert result["previous_context"] == ["First chunk extra"]  # noqa: S101
    assert result["most_recent_message"] == "Second chunk ignored"  # noqa: S101


def test_extract_user_intent_from_messages_handles_object_messages() -> None:
    """User intent extraction should support message objects with content attributes."""

    class Message:
        def __init__(self, role: str, content: Any) -> None:
            self.role = role
            self.content = content

    messages = [
        Message(role="user", content="Plain text content"),
        Message(role="assistant", content="Assistant text"),
        Message(role="user", content=[{"text": "Nested dict text"}, {"content": "secondary"}]),
    ]

    result = _extract_user_intent_from_messages(messages)

    assert result["previous_context"] == ["Plain text content"]  # noqa: S101
    assert result["most_recent_message"] == "Nested dict text secondary"  # noqa: S101


@pytest.mark.asyncio
async def test_prompt_injection_detection_triggers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guardrail should trigger when analysis flags misalignment above threshold."""
    history = _make_history({"type": "function_call", "tool_name": "delete_files", "arguments": "{}"})
    context = _FakeContext(history)

    async def fake_call_llm(ctx: Any, prompt: str, config: LLMConfig) -> PromptInjectionDetectionOutput:
        assert "delete_files" in prompt  # noqa: S101
        assert hasattr(ctx, "guardrail_llm")  # noqa: S101
        return PromptInjectionDetectionOutput(flagged=True, confidence=0.95, observation="Deletes user files")

    monkeypatch.setattr(pid_module, "_call_prompt_injection_detection_llm", fake_call_llm)

    config = LLMConfig(model="gpt-test", confidence_threshold=0.9)
    result = await prompt_injection_detection(context, data="{}", config=config)

    assert result.tripwire_triggered is True  # noqa: S101


@pytest.mark.asyncio
async def test_prompt_injection_detection_no_trigger(monkeypatch: pytest.MonkeyPatch) -> None:
    """Low confidence results should not trigger the guardrail."""
    history = _make_history({"type": "function_call", "tool_name": "get_weather", "arguments": "{}"})
    context = _FakeContext(history)

    async def fake_call_llm(ctx: Any, prompt: str, config: LLMConfig) -> PromptInjectionDetectionOutput:
        return PromptInjectionDetectionOutput(flagged=True, confidence=0.3, observation="Aligned")

    monkeypatch.setattr(pid_module, "_call_prompt_injection_detection_llm", fake_call_llm)

    config = LLMConfig(model="gpt-test", confidence_threshold=0.9)
    result = await prompt_injection_detection(context, data="{}", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert "Aligned" in result.info["observation"]  # noqa: S101


@pytest.mark.asyncio
async def test_prompt_injection_detection_skips_without_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """When no conversation history is present, guardrail should skip."""
    context = _FakeContext([])
    config = LLMConfig(model="gpt-test", confidence_threshold=0.9)

    result = await prompt_injection_detection(context, data="{}", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["observation"] == "No conversation history available"  # noqa: S101


@pytest.mark.asyncio
async def test_prompt_injection_detection_handles_analysis_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exceptions during analysis should return a skip result."""
    history = _make_history({"type": "function_call", "tool_name": "get_weather", "arguments": "{}"})
    context = _FakeContext(history)

    async def failing_llm(*_args: Any, **_kwargs: Any) -> PromptInjectionDetectionOutput:
        raise RuntimeError("LLM failed")

    monkeypatch.setattr(pid_module, "_call_prompt_injection_detection_llm", failing_llm)

    config = LLMConfig(model="gpt-test", confidence_threshold=0.7)
    result = await prompt_injection_detection(context, data="{}", config=config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert "Error during prompt injection detection check" in result.info["observation"]  # noqa: S101


@pytest.mark.asyncio
async def test_prompt_injection_detection_llm_supports_sync_responses() -> None:
    """Underlying responses.parse may be synchronous for some clients."""
    analysis = PromptInjectionDetectionOutput(flagged=True, confidence=0.4, observation="Action summary")

    class _SyncResponses:
        def parse(self, **kwargs: Any) -> Any:
            _ = kwargs
            return SimpleNamespace(output_parsed=analysis)

    context = SimpleNamespace(guardrail_llm=SimpleNamespace(responses=_SyncResponses()))
    config = LLMConfig(model="gpt-test", confidence_threshold=0.5)

    parsed = await pid_module._call_prompt_injection_detection_llm(context, "prompt", config)

    assert parsed is analysis  # noqa: S101
