"""Tests for async evaluation engine prompt injection helpers."""

from __future__ import annotations

import json
from typing import Any

import pytest

from guardrails.evals.core.async_engine import (
    _parse_conversation_payload,
    _run_incremental_prompt_injection,
)
from guardrails.types import GuardrailResult


class _FakeClient:
    """Minimal stub mimicking GuardrailsAsyncOpenAI for testing."""

    def __init__(self, results_sequence: list[list[GuardrailResult]], histories: list[list[Any]]) -> None:
        self._results_sequence = results_sequence
        self._histories = histories
        self._call_index = 0

    async def _run_stage_guardrails(
        self,
        *,
        stage_name: str,
        text: str,
        conversation_history: list[Any],
        suppress_tripwire: bool,
    ) -> list[GuardrailResult]:
        """Return pre-seeded results while recording provided history."""
        assert stage_name == "output"  # noqa: S101
        assert text == ""  # noqa: S101
        assert suppress_tripwire is True  # noqa: S101
        self._histories.append(conversation_history)
        result = self._results_sequence[self._call_index]
        self._call_index += 1
        return result


def _make_result(triggered: bool) -> GuardrailResult:
    return GuardrailResult(
        tripwire_triggered=triggered,
        info={"guardrail_name": "Prompt Injection Detection"},
    )


@pytest.mark.asyncio
async def test_incremental_prompt_injection_stops_on_trigger() -> None:
    """Prompt injection helper should halt once the guardrail triggers."""
    conversation = [
        {"role": "user", "content": "Plan a trip."},
        {"role": "assistant", "type": "function_call", "tool_calls": [{"id": "call_1"}]},
    ]
    sequences = [
        [_make_result(triggered=False)],
        [_make_result(triggered=True)],
    ]
    histories: list[list[Any]] = []
    client = _FakeClient(sequences, histories)

    results = await _run_incremental_prompt_injection(client, conversation)

    assert client._call_index == 2  # noqa: S101
    assert histories[0] == conversation[:1]  # noqa: S101
    assert histories[1] == conversation[:2]  # noqa: S101
    assert results == sequences[1]  # noqa: S101
    info = results[0].info
    assert info["trigger_turn_index"] == 1  # noqa: S101
    assert info["trigger_role"] == "assistant"  # noqa: S101
    assert info["trigger_message"] == conversation[1]  # noqa: S101


@pytest.mark.asyncio
async def test_incremental_prompt_injection_returns_last_result_when_no_trigger() -> None:
    """Prompt injection helper should return last non-empty result when no trigger."""
    conversation = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Need weather update."},
        {"role": "assistant", "type": "function_call", "tool_calls": [{"id": "call_2"}]},
    ]
    sequences = [
        [],  # first turn: nothing to analyse
        [_make_result(triggered=False)],  # second turn: still safe
        [_make_result(triggered=False)],  # third turn: safe action analysed
    ]
    histories: list[list[Any]] = []
    client = _FakeClient(sequences, histories)

    results = await _run_incremental_prompt_injection(client, conversation)

    assert client._call_index == 3  # noqa: S101
    assert results == sequences[-1]  # noqa: S101
    info = results[0].info
    assert info["last_checked_turn_index"] == 2  # noqa: S101
    assert info["last_checked_role"] == "assistant"  # noqa: S101


def test_parse_conversation_payload_supports_object_with_messages() -> None:
    """Conversation payload parser should extract message lists from dicts."""
    payload = {
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
    }
    parsed = _parse_conversation_payload(json.dumps(payload))

    assert parsed == payload["messages"]  # noqa: S101


def test_parse_conversation_payload_returns_none_for_invalid_json() -> None:
    """Parser should return None when sample data is not valid JSON."""
    parsed = _parse_conversation_payload("not-json")

    assert parsed is None  # noqa: S101
