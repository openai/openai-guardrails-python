"""Tests for chat resource wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from guardrails.resources.chat.chat import AsyncChatCompletions, ChatCompletions


class _InlineExecutor:
    """Minimal executor that runs submitted callables synchronously."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _ = (args, kwargs)

    def __enter__(self) -> _InlineExecutor:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def submit(self, fn, *args, **kwargs):
        class _ImmediateFuture:
            def __init__(self) -> None:
                self._result = fn(*args, **kwargs)

            def result(self) -> Any:
                return self._result

        return _ImmediateFuture()


class _SyncClient:
    """Fake synchronous guardrails client for ChatCompletions tests."""

    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, Any]] = []
        self.input_calls: list[dict[str, Any]] = []
        self.applied: list[Any] = []
        self.handle_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.latest_messages: list[Any] = []
        self._resource_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=self._llm_call),
            )
        )
        self._llm_response = SimpleNamespace(type="llm")
        self._stream_result = "stream"
        self._handle_result = "handled"

    def _llm_call(self, **kwargs: Any) -> Any:
        self.llm_kwargs = kwargs
        return self._llm_response

    def _extract_latest_user_message(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        self.latest_messages.append(messages)
        return messages[-1]["content"], len(messages) - 1

    def _run_stage_guardrails(
        self,
        stage_name: str,
        text: str,
        conversation_history: list | None = None,
        suppress_tripwire: bool = False,
    ) -> list[Any]:
        call = {
            "stage": stage_name,
            "text": text,
            "history": conversation_history,
            "suppress": suppress_tripwire,
        }
        if stage_name == "pre_flight":
            self.preflight_calls.append(call)
            return ["preflight"]
        self.input_calls.append(call)
        return ["input"]

    def _apply_preflight_modifications(self, messages: list[dict[str, str]], results: list[Any]) -> list[Any]:
        self.applied.append((messages, results))
        return [{"role": "user", "content": "modified"}]

    def _handle_llm_response(
        self,
        llm_response: Any,
        preflight_results: list[Any],
        input_results: list[Any],
        conversation_history: list | None = None,
        suppress_tripwire: bool = False,
    ) -> Any:
        self.handle_calls.append(
            {
                "response": llm_response,
                "preflight": preflight_results,
                "input": input_results,
                "history": conversation_history,
            }
        )
        return self._handle_result

    def _stream_with_guardrails_sync(
        self,
        llm_stream: Any,
        preflight_results: list[Any],
        input_results: list[Any],
        suppress_tripwire: bool = False,
    ) -> Any:
        self.stream_calls.append(
            {
                "stream": llm_stream,
                "preflight": preflight_results,
                "input": input_results,
                "suppress": suppress_tripwire,
            }
        )
        return self._stream_result


class _AsyncClient:
    """Fake asynchronous client for AsyncChatCompletions tests."""

    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, Any]] = []
        self.input_calls: list[dict[str, Any]] = []
        self.applied: list[Any] = []
        self.handle_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.latest_messages: list[Any] = []
        self._resource_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=self._llm_call),
            )
        )
        self._llm_response = SimpleNamespace(type="llm")
        self._stream_result = "async-stream"
        self._handle_result = "async-handled"

    async def _llm_call(self, **kwargs: Any) -> Any:
        self.llm_kwargs = kwargs
        return self._llm_response

    def _extract_latest_user_message(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        self.latest_messages.append(messages)
        return messages[-1]["content"], len(messages) - 1

    async def _run_stage_guardrails(
        self,
        stage_name: str,
        text: str,
        conversation_history: list | None = None,
        suppress_tripwire: bool = False,
    ) -> list[Any]:
        call = {
            "stage": stage_name,
            "text": text,
            "history": conversation_history,
            "suppress": suppress_tripwire,
        }
        if stage_name == "pre_flight":
            self.preflight_calls.append(call)
            return ["preflight"]
        self.input_calls.append(call)
        return ["input"]

    def _apply_preflight_modifications(self, messages: list[dict[str, str]], results: list[Any]) -> list[Any]:
        self.applied.append((messages, results))
        return [{"role": "user", "content": "modified"}]

    async def _handle_llm_response(
        self,
        llm_response: Any,
        preflight_results: list[Any],
        input_results: list[Any],
        conversation_history: list | None = None,
        suppress_tripwire: bool = False,
    ) -> Any:
        self.handle_calls.append(
            {
                "response": llm_response,
                "preflight": preflight_results,
                "input": input_results,
                "history": conversation_history,
            }
        )
        return self._handle_result

    def _stream_with_guardrails(
        self,
        llm_stream: Any,
        preflight_results: list[Any],
        input_results: list[Any],
        suppress_tripwire: bool = False,
    ) -> Any:
        self.stream_calls.append(
            {
                "stream": llm_stream,
                "preflight": preflight_results,
                "input": input_results,
                "suppress": suppress_tripwire,
            }
        )
        return self._stream_result


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hello"},
    ]


def test_chat_completions_create_invokes_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """ChatCompletions.create should run guardrails and forward modified messages."""
    client = _SyncClient()
    completions = ChatCompletions(client)

    monkeypatch.setattr("guardrails.resources.chat.chat.ThreadPoolExecutor", _InlineExecutor)

    result = completions.create(messages=_messages(), model="gpt-test")

    assert result == "handled"  # noqa: S101
    assert client.preflight_calls[0]["stage"] == "pre_flight"  # noqa: S101
    assert client.input_calls[0]["stage"] == "input"  # noqa: S101
    assert client.llm_kwargs["messages"][0]["content"] == "modified"  # noqa: S101
    assert client.handle_calls[0]["preflight"] == ["preflight"]  # noqa: S101


def test_chat_completions_stream_returns_streaming_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming mode should defer to _stream_with_guardrails_sync."""
    client = _SyncClient()
    completions = ChatCompletions(client)

    monkeypatch.setattr("guardrails.resources.chat.chat.ThreadPoolExecutor", _InlineExecutor)

    result = completions.create(messages=_messages(), model="gpt-test", stream=True, suppress_tripwire=True)

    assert result == "stream"  # noqa: S101
    stream_call = client.stream_calls[0]
    assert stream_call["suppress"] is True  # noqa: S101
    assert stream_call["preflight"] == ["preflight"]  # noqa: S101


@pytest.mark.asyncio
async def test_async_chat_completions_create_invokes_guardrails() -> None:
    """AsyncChatCompletions.create should await guardrails and LLM call."""
    client = _AsyncClient()
    completions = AsyncChatCompletions(client)

    result = await completions.create(messages=_messages(), model="gpt-test")

    assert result == "async-handled"  # noqa: S101
    assert client.preflight_calls[0]["stage"] == "pre_flight"  # noqa: S101
    assert client.input_calls[0]["stage"] == "input"  # noqa: S101
    assert client.llm_kwargs["messages"][0]["content"] == "modified"  # noqa: S101
    assert client.handle_calls[0]["preflight"] == ["preflight"]  # noqa: S101


@pytest.mark.asyncio
async def test_async_chat_completions_stream_returns_wrapper() -> None:
    """Async streaming mode should defer to _stream_with_guardrails."""
    client = _AsyncClient()
    completions = AsyncChatCompletions(client)

    result = await completions.create(
        messages=_messages(),
        model="gpt-test",
        stream=True,
        suppress_tripwire=False,
    )

    assert result == "async-stream"  # noqa: S101
    stream_call = client.stream_calls[0]
    assert stream_call["preflight"] == ["preflight"]  # noqa: S101
    assert stream_call["input"] == ["input"]  # noqa: S101
