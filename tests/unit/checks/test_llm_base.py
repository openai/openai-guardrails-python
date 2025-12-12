"""Tests for LLM-based guardrail helpers."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from guardrails.checks.text import llm_base
from guardrails.checks.text.llm_base import (
    LLMConfig,
    LLMErrorOutput,
    LLMOutput,
    _build_analysis_payload,
    _build_full_prompt,
    _strip_json_code_fence,
    create_llm_check_fn,
    run_llm,
)
from guardrails.types import GuardrailResult, TokenUsage


def _mock_token_usage() -> TokenUsage:
    """Return a mock TokenUsage for tests."""
    return TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)


def _mock_usage_object() -> SimpleNamespace:
    """Return a mock usage object for fake API responses."""
    return SimpleNamespace(prompt_tokens=100, completion_tokens=50, total_tokens=150)


class _FakeCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content

    async def create(self, **kwargs: Any) -> Any:
        _ = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=_mock_usage_object(),
        )


class _FakeAsyncClient:
    def __init__(self, content: str | None) -> None:
        self.chat = SimpleNamespace(completions=_FakeCompletions(content))


class _FakeSyncCompletions:
    def __init__(self, content: str | None) -> None:
        self._content = content

    def create(self, **kwargs: Any) -> Any:
        _ = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=_mock_usage_object(),
        )


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
    prompt = _build_full_prompt("Analyze text", LLMOutput)
    assert "Analyze text" in prompt  # noqa: S101
    assert "Respond with a json object" in prompt  # noqa: S101
    assert "flagged" in prompt  # noqa: S101
    assert "confidence" in prompt  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_returns_valid_output() -> None:
    """run_llm should parse the JSON response into the provided output model."""
    client = _FakeAsyncClient('{"flagged": true, "confidence": 0.9}')
    result, token_usage = await run_llm(
        text="Sensitive text",
        system_prompt="Detect problems.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
    )
    assert isinstance(result, LLMOutput)  # noqa: S101
    assert result.flagged is True and result.confidence == 0.9  # noqa: S101
    # Verify token usage is returned
    assert token_usage.prompt_tokens == 100  # noqa: S101
    assert token_usage.completion_tokens == 50  # noqa: S101
    assert token_usage.total_tokens == 150  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_supports_sync_clients() -> None:
    """run_llm should invoke synchronous clients without awaiting them."""
    client = _FakeSyncClient('{"flagged": false, "confidence": 0.25}')

    result, token_usage = await run_llm(
        text="General text",
        system_prompt="Assess text.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
    )

    assert isinstance(result, LLMOutput)  # noqa: S101
    assert result.flagged is False and result.confidence == 0.25  # noqa: S101
    # Verify token usage is returned
    assert isinstance(token_usage, TokenUsage)  # noqa: S101


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

    result, token_usage = await run_llm(
        text="Sensitive",
        system_prompt="Detect.",
        client=_FailingClient(),  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
    )

    assert isinstance(result, LLMErrorOutput)  # noqa: S101
    assert result.flagged is True  # noqa: S101
    assert result.info["third_party_filter"] is True  # noqa: S101
    # Token usage should indicate failure
    assert token_usage.unavailable_reason is not None  # noqa: S101


@pytest.mark.asyncio
async def test_create_llm_check_fn_triggers_on_confident_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Generated guardrail function should trip when confidence exceeds the threshold."""

    async def fake_run_llm(
        text: str,
        system_prompt: str,
        client: Any,
        model: str,
        output_model: type[LLMOutput],
        conversation_history: list[dict[str, Any]] | None = None,
        max_turns: int = 10,
    ) -> tuple[LLMOutput, TokenUsage]:
        assert system_prompt == "Check with details"  # noqa: S101
        return LLMOutput(flagged=True, confidence=0.95), _mock_token_usage()

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
    # Verify token usage is included in the result
    assert "token_usage" in result.info  # noqa: S101
    assert result.info["token_usage"]["total_tokens"] == 150  # noqa: S101


@pytest.mark.asyncio
async def test_create_llm_check_fn_handles_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """LLM error results should mark execution_failed without triggering tripwire."""
    error_usage = TokenUsage(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        unavailable_reason="LLM call failed",
    )

    async def fake_run_llm(
        text: str,
        system_prompt: str,
        client: Any,
        model: str,
        output_model: type[LLMOutput],
        conversation_history: list[dict[str, Any]] | None = None,
        max_turns: int = 10,
    ) -> tuple[LLMErrorOutput, TokenUsage]:
        return LLMErrorOutput(flagged=False, confidence=0.0, info={"error_message": "timeout"}), error_usage

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
    # Verify token usage is included even in error results
    assert "token_usage" in result.info  # noqa: S101


# ==================== Multi-Turn Functionality Tests ====================


def test_llm_config_has_max_turns_field() -> None:
    """LLMConfig should have max_turns field with default of 10."""
    config = LLMConfig(model="gpt-test")
    assert config.max_turns == 10  # noqa: S101


def test_llm_config_max_turns_can_be_set() -> None:
    """LLMConfig.max_turns should be configurable."""
    config = LLMConfig(model="gpt-test", max_turns=5)
    assert config.max_turns == 5  # noqa: S101


def test_llm_config_max_turns_minimum_is_one() -> None:
    """LLMConfig.max_turns should have minimum value of 1."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        LLMConfig(model="gpt-test", max_turns=0)


def test_build_analysis_payload_formats_correctly() -> None:
    """_build_analysis_payload should create JSON with conversation and latest_input."""
    conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    latest_input = "What's the weather?"

    payload_str = _build_analysis_payload(conversation_history, latest_input, max_turns=10)
    payload = json.loads(payload_str)

    assert payload["conversation"] == conversation_history  # noqa: S101
    assert payload["latest_input"] == "What's the weather?"  # noqa: S101


def test_build_analysis_payload_trims_to_max_turns() -> None:
    """_build_analysis_payload should trim conversation to max_turns."""
    conversation_history = [
        {"role": "user", "content": f"Message {i}"} for i in range(15)
    ]

    payload_str = _build_analysis_payload(conversation_history, "latest", max_turns=5)
    payload = json.loads(payload_str)

    # Should only have the last 5 turns
    assert len(payload["conversation"]) == 5  # noqa: S101
    assert payload["conversation"][0]["content"] == "Message 10"  # noqa: S101
    assert payload["conversation"][4]["content"] == "Message 14"  # noqa: S101


def test_build_analysis_payload_handles_none_conversation() -> None:
    """_build_analysis_payload should handle None conversation gracefully."""
    payload_str = _build_analysis_payload(None, "latest input", max_turns=10)
    payload = json.loads(payload_str)

    assert payload["conversation"] == []  # noqa: S101
    assert payload["latest_input"] == "latest input"  # noqa: S101


def test_build_analysis_payload_handles_empty_conversation() -> None:
    """_build_analysis_payload should handle empty conversation list."""
    payload_str = _build_analysis_payload([], "latest input", max_turns=10)
    payload = json.loads(payload_str)

    assert payload["conversation"] == []  # noqa: S101
    assert payload["latest_input"] == "latest input"  # noqa: S101


def test_build_analysis_payload_strips_whitespace() -> None:
    """_build_analysis_payload should strip whitespace from latest_input."""
    payload_str = _build_analysis_payload([], "  trimmed text  ", max_turns=10)
    payload = json.loads(payload_str)

    assert payload["latest_input"] == "trimmed text"  # noqa: S101


class _FakeCompletionsCapture:
    """Captures the messages sent to the LLM for verification."""

    def __init__(self, content: str | None) -> None:
        self._content = content
        self.captured_messages: list[dict[str, str]] | None = None

    async def create(self, **kwargs: Any) -> Any:
        self.captured_messages = kwargs.get("messages")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))],
            usage=_mock_usage_object(),
        )


class _FakeAsyncClientCapture:
    """Fake client that captures messages for testing."""

    def __init__(self, content: str | None) -> None:
        self._completions = _FakeCompletionsCapture(content)
        self.chat = SimpleNamespace(completions=self._completions)

    @property
    def captured_messages(self) -> list[dict[str, str]] | None:
        return self._completions.captured_messages


@pytest.mark.asyncio
async def test_run_llm_single_turn_without_conversation() -> None:
    """run_llm without conversation_history should use single-turn format."""
    client = _FakeAsyncClientCapture('{"flagged": false, "confidence": 0.1}')

    await run_llm(
        text="Test input",
        system_prompt="Analyze.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
        conversation_history=None,
        max_turns=10,
    )

    # Should use single-turn format "# Text\n\n..."
    user_message = client.captured_messages[1]["content"]
    assert user_message.startswith("# Text")  # noqa: S101
    assert "Test input" in user_message  # noqa: S101
    # Should NOT have JSON payload format
    assert "latest_input" not in user_message  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_single_turn_with_max_turns_one() -> None:
    """run_llm with max_turns=1 should use single-turn format even with conversation."""
    client = _FakeAsyncClientCapture('{"flagged": false, "confidence": 0.1}')
    conversation_history = [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"},
    ]

    await run_llm(
        text="Test input",
        system_prompt="Analyze.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
        conversation_history=conversation_history,
        max_turns=1,  # Single-turn mode
    )

    # Should use single-turn format "# Text\n\n..."
    user_message = client.captured_messages[1]["content"]
    assert user_message.startswith("# Text")  # noqa: S101
    assert "Test input" in user_message  # noqa: S101
    # Should NOT have JSON payload format
    assert "latest_input" not in user_message  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_multi_turn_with_conversation() -> None:
    """run_llm with conversation_history and max_turns>1 should use multi-turn format."""
    client = _FakeAsyncClientCapture('{"flagged": false, "confidence": 0.1}')
    conversation_history = [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"},
    ]

    await run_llm(
        text="Test input",
        system_prompt="Analyze.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
        conversation_history=conversation_history,
        max_turns=10,
    )

    # Should use multi-turn format "# Analysis Input\n\n..."
    user_message = client.captured_messages[1]["content"]
    assert user_message.startswith("# Analysis Input")  # noqa: S101
    # Should have JSON payload format
    assert "latest_input" in user_message  # noqa: S101
    assert "conversation" in user_message  # noqa: S101
    # Parse the JSON to verify structure
    json_start = user_message.find("{")
    payload = json.loads(user_message[json_start:])
    assert payload["latest_input"] == "Test input"  # noqa: S101
    assert len(payload["conversation"]) == 2  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_empty_conversation_uses_single_turn() -> None:
    """run_llm with empty conversation_history should use single-turn format."""
    client = _FakeAsyncClientCapture('{"flagged": false, "confidence": 0.1}')

    await run_llm(
        text="Test input",
        system_prompt="Analyze.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
        conversation_history=[],  # Empty list
        max_turns=10,
    )

    # Should use single-turn format
    user_message = client.captured_messages[1]["content"]
    assert user_message.startswith("# Text")  # noqa: S101
    assert "latest_input" not in user_message  # noqa: S101


@pytest.mark.asyncio
async def test_create_llm_check_fn_extracts_conversation_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory-created guardrail should extract conversation history from context."""
    captured_args: dict[str, Any] = {}

    async def fake_run_llm(
        text: str,
        system_prompt: str,
        client: Any,
        model: str,
        output_model: type[LLMOutput],
        conversation_history: list[dict[str, Any]] | None = None,
        max_turns: int = 10,
    ) -> tuple[LLMOutput, TokenUsage]:
        captured_args["conversation_history"] = conversation_history
        captured_args["max_turns"] = max_turns
        return LLMOutput(flagged=False, confidence=0.1), _mock_token_usage()

    monkeypatch.setattr(llm_base, "run_llm", fake_run_llm)

    guardrail_fn = create_llm_check_fn(
        name="ConvoTest",
        description="Test guardrail",
        system_prompt="Prompt",
    )

    # Create context with conversation history
    conversation = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]

    class ContextWithHistory:
        guardrail_llm = "fake-client"

        def get_conversation_history(self) -> list:
            return conversation

    config = LLMConfig(model="gpt-test", max_turns=5)
    await guardrail_fn(ContextWithHistory(), "text", config)

    # Verify conversation history was passed to run_llm
    assert captured_args["conversation_history"] == conversation  # noqa: S101
    assert captured_args["max_turns"] == 5  # noqa: S101


@pytest.mark.asyncio
async def test_create_llm_check_fn_handles_missing_conversation_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory-created guardrail should handle context without get_conversation_history."""
    captured_args: dict[str, Any] = {}

    async def fake_run_llm(
        text: str,
        system_prompt: str,
        client: Any,
        model: str,
        output_model: type[LLMOutput],
        conversation_history: list[dict[str, Any]] | None = None,
        max_turns: int = 10,
    ) -> tuple[LLMOutput, TokenUsage]:
        captured_args["conversation_history"] = conversation_history
        return LLMOutput(flagged=False, confidence=0.1), _mock_token_usage()

    monkeypatch.setattr(llm_base, "run_llm", fake_run_llm)

    guardrail_fn = create_llm_check_fn(
        name="NoConvoTest",
        description="Test guardrail",
        system_prompt="Prompt",
    )

    # Context without get_conversation_history method
    context = SimpleNamespace(guardrail_llm="fake-client")
    config = LLMConfig(model="gpt-test")
    await guardrail_fn(context, "text", config)

    # Should pass empty list when no conversation history
    assert captured_args["conversation_history"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_strips_whitespace_in_single_turn_mode() -> None:
    """run_llm should strip whitespace from input in single-turn mode."""
    client = _FakeAsyncClientCapture('{"flagged": false, "confidence": 0.1}')

    await run_llm(
        text="  Test input with whitespace  \n",
        system_prompt="Analyze.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
        conversation_history=None,
        max_turns=10,
    )

    # Should strip whitespace in single-turn mode
    user_message = client.captured_messages[1]["content"]
    assert "# Text\n\nTest input with whitespace" in user_message  # noqa: S101
    assert "  Test input" not in user_message  # noqa: S101


@pytest.mark.asyncio
async def test_run_llm_strips_whitespace_in_multi_turn_mode() -> None:
    """run_llm should strip whitespace from input in multi-turn mode."""
    client = _FakeAsyncClientCapture('{"flagged": false, "confidence": 0.1}')
    conversation_history = [
        {"role": "user", "content": "Previous message"},
    ]

    await run_llm(
        text="  Test input with whitespace  \n",
        system_prompt="Analyze.",
        client=client,  # type: ignore[arg-type]
        model="gpt-test",
        output_model=LLMOutput,
        conversation_history=conversation_history,
        max_turns=10,
    )

    # Should strip whitespace in multi-turn mode
    user_message = client.captured_messages[1]["content"]
    import json
    json_start = user_message.find("{")
    payload = json.loads(user_message[json_start:])
    assert payload["latest_input"] == "Test input with whitespace"  # noqa: S101
