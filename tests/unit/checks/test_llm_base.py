"""Tests for LLM-based guardrail helpers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from guardrails.checks.text import llm_base
from guardrails.checks.text.llm_base import (
    LLMConfig,
    LLMErrorOutput,
    LLMOutput,
    _build_full_prompt,
    _strip_json_code_fence,
    create_llm_check_fn,
    run_llm,
)
from guardrails.types import GuardrailResult


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content

    async def create(self, **kwargs: Any) -> Any:
        _ = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


class _FakeAsyncClient:
    def __init__(self, content: str | None) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


class _FakeSyncCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content

    def create(self, **kwargs: Any) -> Any:
        _ = kwargs
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])


class _FakeSyncClient:
    def __init__(self, content: str | None) -> None:
        self.chat = SimpleNamespace(completions=_FakeSyncCompletions(content))


def test_strip_json_code_fence_removes_wrapping() -> None:
    """Valid JSON code fences should be removed."""
    fenced = """```json
{"flagged": false, "confidence": 0.2}
```"""
    assert _strip_json_code_fence(fenced) == '{"flagged": false, "confidence": 0.2}'  # noqa: S101


def test_build_full_prompt_includes_instructions() -> None:
    """Generated prompt should embed system instructions and schema guidance."""
    prompt = _build_full_prompt("Analyze text")
    assert "Analyze text" in prompt  # noqa: S101
    assert "Respond with a json object" in prompt  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_returns_valid_output() -> None:
    """run_llm should parse the JSON response into the provided output model."""
    client = _FakeAsyncClient('{"flagged": true, "confidence": 0.9}')
    result = await run_llm(
        text="Sensitive text",
        system_prompt="Detect problems.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
    )
    assert isinstance(result, LLMOutput)  # noqa: S101
    assert result.flagged is True and result.confidence == 0.9  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_supports_sync_clients() -> None:
    """run_llm should invoke synchronous clients without awaiting them."""
    client = _FakeSyncClient('{"flagged": false, "confidence": 0.25}')

    result = await run_llm(
        text="General text",
        system_prompt="Assess text.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
    )

    assert isinstance(result, LLMOutput)  # noqa: S101
    assert result.flagged is False and result.confidence == 0.25  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_handles_content_filter_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content filter errors should return LLMErrorOutput with flagged=True."""

    class _FailingClient:
        class _Chat:
            class _Completions:
                async def create(self, **kwargs: Any) -> Any:
                    raise RuntimeError("content_filter triggered by provider")

            completions = _Completions()

        chat = _Chat()

    result = await run_llm(
        text="Sensitive",
        system_prompt="Detect.",
        client=_FailingClient(),  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
    )

    assert isinstance(result, LLMErrorOutput)  # noqa: S101
    assert result.flagged is True  # noqa: S101
    assert result.info["third_party_filter"] is True  # noqa: S101


@pytest.mark.asyncio
async def test_create_llm_check_fn_triggers_on_confident_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated guardrail function should trip when confidence exceeds the threshold."""

    async def fake_run_llm(
        text: str,
        system_prompt: str,
        client: Any,
        model: str,
        output_model: type[LLMOutput],
    ) -> LLMOutput:
        assert system_prompt == "Check with details"  # noqa: S101
        return LLMOutput(flagged=True, confidence=0.95)

    monkeypatch.setattr(llm_base, "run_llm", fake_run_llm)

    class DetailedConfig(LLMConfig):
        system_prompt_details: str = "details"

    guardrail_fn = create_llm_check_fn(
        name="HighConfidence",
        description="Test guardrail",
        system_prompt="Check with {system_prompt_details}",
        output_model=LLMOutput,
        config_model=DetailedConfig,
    )

    config = DetailedConfig(model="gpt-test", confidence_threshold=0.9)
    context = SimpleNamespace(guardrail_llm="fake-client")

    result = await guardrail_fn(context, "content", config)

    assert isinstance(result, GuardrailResult)  # noqa: S101
    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["threshold"] == 0.9  # noqa: S101


@pytest.mark.asyncio
async def test_create_llm_check_fn_handles_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM error results should mark execution_failed without triggering tripwire."""

    async def fake_run_llm(
        text: str,
        system_prompt: str,
        client: Any,
        model: str,
        output_model: type[LLMOutput],
    ) -> LLMErrorOutput:
        return LLMErrorOutput(flagged=False, confidence=0.0, info={"error_message": "timeout"})

    monkeypatch.setattr(llm_base, "run_llm", fake_run_llm)

    guardrail_fn = create_llm_check_fn(
        name="Resilient",
        description="Test guardrail",
        system_prompt="Prompt",
    )

    config = LLMConfig(model="gpt-test", confidence_threshold=0.5)
    context = SimpleNamespace(guardrail_llm="fake-client")
    result = await guardrail_fn(context, "text", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.execution_failed is True  # noqa: S101
    assert "timeout" in str(result.original_exception)  # noqa: S101
