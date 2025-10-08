"""Tests for StreamingMixin behaviour."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from typing import Any

import pytest

from guardrails._base_client import GuardrailsBaseClient, GuardrailsResponse
from guardrails._streaming import StreamingMixin
from guardrails.exceptions import GuardrailTripwireTriggered
from guardrails.types import GuardrailResult


@dataclass(frozen=True, slots=True)
class _Chunk:
    """Simple chunk carrying text content."""

    text: str


class _StreamingCollector(StreamingMixin, GuardrailsBaseClient):
    """Minimal client exposing hooks required by StreamingMixin."""

    def __init__(self) -> None:
        self.run_calls: list[tuple[str, bool]] = []
        self.responses: list[GuardrailsResponse] = []
        self._next_results: list[GuardrailResult] = []
        self._should_raise = False

    def set_results(self, results: list[GuardrailResult]) -> None:
        self._next_results = results

    def trigger_tripwire(self) -> None:
        self._should_raise = True

    def _extract_response_text(self, chunk: _Chunk) -> str:
        return chunk.text

    def _run_stage_guardrails(
        self,
        stage_name: str,
        text: str,
        suppress_tripwire: bool = False,
        **kwargs: Any,
    ) -> list[GuardrailResult]:
        self.run_calls.append((text, suppress_tripwire))
        if self._should_raise:
            from guardrails.exceptions import GuardrailTripwireTriggered

            raise GuardrailTripwireTriggered(GuardrailResult(tripwire_triggered=True))
        return self._next_results

    async def _run_stage_guardrails_async(
        self,
        stage_name: str,
        text: str,
        suppress_tripwire: bool = False,
        **kwargs: Any,
    ) -> list[GuardrailResult]:
        return self._run_stage_guardrails(stage_name, text, suppress_tripwire=suppress_tripwire)

    def _create_guardrails_response(
        self,
        llm_response: Any,
        preflight_results: list[GuardrailResult],
        input_results: list[GuardrailResult],
        output_results: list[GuardrailResult],
    ) -> GuardrailsResponse:
        response = super()._create_guardrails_response(llm_response, preflight_results, input_results, output_results)
        self.responses.append(response)
        return response


async def _aiter(items: list[_Chunk]) -> AsyncIterator[_Chunk]:
    for item in items:
        yield item


def test_stream_with_guardrails_sync_emits_results() -> None:
    """Synchronous streaming should yield GuardrailsResponse objects with accumulated results."""
    client = _StreamingCollector()
    client.set_results([GuardrailResult(tripwire_triggered=False)])
    chunks: Iterator[_Chunk] = iter([_Chunk("a"), _Chunk("b")])

    responses = list(
        client._stream_with_guardrails_sync(
            chunks,
            preflight_results=[GuardrailResult(tripwire_triggered=False)],
            input_results=[],
            check_interval=1,
        )
    )

    assert [resp.guardrail_results.output for resp in responses] == [[], []]  # noqa: S101
    assert client.run_calls == [("a", False), ("ab", False), ("ab", False)]  # noqa: S101


@pytest.mark.asyncio
async def test_stream_with_guardrails_async_emits_results() -> None:
    """Async streaming should yield GuardrailsResponse objects and run final checks."""
    client = _StreamingCollector()

    async def fake_run_stage(
        stage_name: str,
        text: str,
        suppress_tripwire: bool = False,
        **kwargs: Any,
    ) -> list[GuardrailResult]:
        client.run_calls.append((text, suppress_tripwire))
        return []

    client._run_stage_guardrails = fake_run_stage  # type: ignore[assignment]

    responses = [
        response
        async for response in client._stream_with_guardrails(
            _aiter([_Chunk("a"), _Chunk("b")]),
            preflight_results=[],
            input_results=[],
            check_interval=2,
        )
    ]

    assert len(responses) == 2  # noqa: S101
    # Final guardrail run should consume aggregated text "ab"
    assert client.run_calls[-1][0] == "ab"  # noqa: S101


@pytest.mark.asyncio
async def test_stream_with_guardrails_async_raises_on_tripwire() -> None:
    """Tripwire should abort streaming and clear accumulated text."""
    client = _StreamingCollector()

    async def fake_run_stage(
        stage_name: str,
        text: str,
        suppress_tripwire: bool = False,
        **kwargs: Any,
    ) -> list[GuardrailResult]:
        raise_guardrail = text == "chunk"
        if raise_guardrail:
            from guardrails.exceptions import GuardrailTripwireTriggered

            raise GuardrailTripwireTriggered(GuardrailResult(tripwire_triggered=True))
        return []

    client._run_stage_guardrails = fake_run_stage  # type: ignore[assignment]

    async def chunk_stream() -> AsyncIterator[_Chunk]:
        yield _Chunk("chunk")

    with pytest.raises(GuardrailTripwireTriggered):
        async for _ in client._stream_with_guardrails(
            chunk_stream(),
            preflight_results=[],
            input_results=[],
            check_interval=1,
        ):
            pass
