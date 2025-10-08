"""Tests for responses resource wrappers."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from guardrails.resources.responses.responses import AsyncResponses, Responses


class _SyncResponsesClient:
    """Fake synchronous guardrails client for Responses tests."""

    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, Any]] = []
        self.input_calls: list[dict[str, Any]] = []
        self.output_calls: list[dict[str, Any]] = []
        self.applied: list[Any] = []
        self.handle_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self.parse_calls: list[dict[str, Any]] = []
        self.retrieve_calls: list[dict[str, Any]] = []
        self._llm_response = SimpleNamespace(output_text="result", type="llm")
        self._stream_result = "stream"
        self._handle_result = "handled"
        self._resource_client = SimpleNamespace(
            responses=SimpleNamespace(
                create=self._llm_create,
                parse=self._llm_parse,
                retrieve=self._llm_retrieve,
            )
        )

    def _llm_create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self._llm_response

    def _llm_parse(self, **kwargs: Any) -> Any:
        self.parse_calls.append(kwargs)
        return self._llm_response

    def _llm_retrieve(self, response_id: str, **kwargs: Any) -> Any:
        self.retrieve_calls.append({"id": response_id, "kwargs": kwargs})
        return self._llm_response

    def _extract_latest_user_message(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        return messages[-1]["content"], len(messages) - 1

    def _run_stage_guardrails(
        self,
        stage: str,
        text: str,
        conversation_history: list | str | None = None,
        suppress_tripwire: bool = False,
    ) -> list[str]:
        call = {
            "stage": stage,
            "text": text,
            "history": conversation_history,
            "suppress": suppress_tripwire,
        }
        if stage == "pre_flight":
            self.preflight_calls.append(call)
            return ["preflight"]
        if stage == "input":
            self.input_calls.append(call)
            return ["input"]
        self.output_calls.append(call)
        return ["output"]

    def _apply_preflight_modifications(self, data: Any, results: list[Any]) -> Any:
        self.applied.append((data, results))
        if isinstance(data, list):
            return [{"role": "user", "content": "modified"}]
        return "modified"

    def _handle_llm_response(
        self,
        llm_response: Any,
        preflight_results: list[Any],
        input_results: list[Any],
        conversation_history: Any = None,
        suppress_tripwire: bool = False,
        **kwargs: Any,
    ) -> Any:
        self.handle_calls.append(
            {
                "response": llm_response,
                "preflight": preflight_results,
                "input": input_results,
                "history": conversation_history,
                "extra": kwargs,
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

    def _create_guardrails_response(
        self,
        response: Any,
        preflight_results: list[Any],
        input_results: list[Any],
        output_results: list[Any],
    ) -> Any:
        self.output_calls.append({"stage": "output", "results": output_results})
        return {
            "response": response,
            "preflight": preflight_results,
            "input": input_results,
            "output": output_results,
        }


class _AsyncResponsesClient:
    """Fake asynchronous guardrails client for AsyncResponses tests."""

    def __init__(self) -> None:
        self.preflight_calls: list[dict[str, Any]] = []
        self.input_calls: list[dict[str, Any]] = []
        self.output_calls: list[dict[str, Any]] = []
        self.applied: list[Any] = []
        self.handle_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.create_calls: list[dict[str, Any]] = []
        self._llm_response = SimpleNamespace(output_text="async", type="llm")
        self._stream_result = "async-stream"
        self._handle_result = "async-handled"
        self._resource_client = SimpleNamespace(
            responses=SimpleNamespace(
                create=self._llm_create,
            )
        )

    async def _llm_create(self, **kwargs: Any) -> Any:
        self.create_calls.append(kwargs)
        return self._llm_response

    def _extract_latest_user_message(self, messages: list[dict[str, str]]) -> tuple[str, int]:
        return messages[-1]["content"], len(messages) - 1

    async def _run_stage_guardrails(
        self,
        stage: str,
        text: str,
        conversation_history: list | str | None = None,
        suppress_tripwire: bool = False,
    ) -> list[str]:
        call = {
            "stage": stage,
            "text": text,
            "history": conversation_history,
            "suppress": suppress_tripwire,
        }
        if stage == "pre_flight":
            self.preflight_calls.append(call)
            return ["preflight"]
        if stage == "input":
            self.input_calls.append(call)
            return ["input"]
        self.output_calls.append(call)
        return ["output"]

    def _apply_preflight_modifications(self, data: Any, results: list[Any]) -> Any:
        self.applied.append((data, results))
        if isinstance(data, list):
            return [{"role": "user", "content": "modified"}]
        return "modified"

    async def _handle_llm_response(
        self,
        llm_response: Any,
        preflight_results: list[Any],
        input_results: list[Any],
        conversation_history: Any = None,
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


def _inline_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    class _InlineExecutor:
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

    monkeypatch.setattr("guardrails.resources.responses.responses.ThreadPoolExecutor", _InlineExecutor)


def test_responses_create_runs_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Responses.create should apply guardrails and forward modified input."""
    client = _SyncResponsesClient()
    responses = Responses(client)
    _inline_executor(monkeypatch)

    result = responses.create(input=_messages(), model="gpt-test")

    assert result == "handled"  # noqa: S101
    assert client.preflight_calls[0]["stage"] == "pre_flight"  # noqa: S101
    assert client.input_calls[0]["stage"] == "input"  # noqa: S101
    assert client.create_calls[0]["input"][0]["content"] == "modified"  # noqa: S101


def test_responses_create_stream_returns_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming mode should call _stream_with_guardrails_sync."""
    client = _SyncResponsesClient()
    responses = Responses(client)
    _inline_executor(monkeypatch)

    result = responses.create(input=_messages(), model="gpt-test", stream=True, suppress_tripwire=True)

    assert result == "stream"  # noqa: S101
    stream_call = client.stream_calls[0]
    assert stream_call["suppress"] is True  # noqa: S101
    assert stream_call["preflight"] == ["preflight"]  # noqa: S101


def test_responses_parse_runs_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Responses.parse should run guardrails and pass modified input."""
    client = _SyncResponsesClient()
    responses = Responses(client)
    _inline_executor(monkeypatch)

    class _Schema(BaseModel):
        text: str

    messages = _messages()
    result = responses.parse(input=messages, model="gpt-test", text_format=_Schema)

    assert result == "handled"  # noqa: S101
    assert client.parse_calls[0]["input"][0]["content"] == "modified"  # noqa: S101
    assert client.handle_calls[0]["extra"]["conversation_data"] == messages  # noqa: S101


def test_responses_retrieve_wraps_output() -> None:
    """Responses.retrieve should run output guardrails and wrap the response."""
    client = _SyncResponsesClient()
    responses = Responses(client)

    wrapped = responses.retrieve("resp-1", suppress_tripwire=False)

    assert wrapped["response"].output_text == "result"  # noqa: S101
    assert wrapped["output"] == ["output"]  # noqa: S101
    assert client.retrieve_calls[0]["id"] == "resp-1"  # noqa: S101


@pytest.mark.asyncio
async def test_async_responses_create_runs_guardrails() -> None:
    """AsyncResponses.create should await guardrails and modify input."""
    client = _AsyncResponsesClient()
    responses = AsyncResponses(client)

    result = await responses.create(input=_messages(), model="gpt-test")

    assert result == "async-handled"  # noqa: S101
    assert client.preflight_calls[0]["stage"] == "pre_flight"  # noqa: S101
    assert client.input_calls[0]["stage"] == "input"  # noqa: S101
    assert client.create_calls[0]["input"][0]["content"] == "modified"  # noqa: S101


@pytest.mark.asyncio
async def test_async_responses_stream_returns_wrapper() -> None:
    """AsyncResponses streaming mode should defer to _stream_with_guardrails."""
    client = _AsyncResponsesClient()
    responses = AsyncResponses(client)

    result = await responses.create(input=_messages(), model="gpt-test", stream=True)

    assert result == "async-stream"  # noqa: S101
    stream_call = client.stream_calls[0]
    assert stream_call["preflight"] == ["preflight"]  # noqa: S101
    assert stream_call["input"] == ["input"]  # noqa: S101
