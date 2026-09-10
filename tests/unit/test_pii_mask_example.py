"""Conversation-history contract for the interactive PII masking example."""

import runpy
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from guardrails import GuardrailTripwireTriggered
from guardrails.types import GuardrailResult


@pytest.fixture
def example() -> dict[str, Any]:
    return runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/basic/pii_mask_example.py"))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("preflight", "expected"),
    [
        (
            [GuardrailResult(False, info={"guardrail_name": "Contains PII", "pii_detected": True, "checked_text": "<EMAIL_ADDRESS>"})],
            "<EMAIL_ADDRESS>",
        ),
        ([GuardrailResult(False, info={"guardrail_name": "Contains PII", "pii_detected": False})], "sample input"),
        ([], "sample input"),
    ],
)
async def test_successful_turn_retains_checked_input(example: dict[str, Any], preflight: list[GuardrailResult], expected: str) -> None:
    response = MagicMock()
    response.choices[0].message.content = "How can I help?"
    response.guardrail_results.preflight = preflight
    response.guardrail_results.output = []
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    history = [{"role": "system", "content": "Be helpful."}]

    await example["process_input"](client, "sample input", history)

    assert history == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": expected},
        {"role": "assistant", "content": "How can I help?"},
    ]
    response.guardrail_results.preflight = []
    await example["process_input"](client, "next turn", history)

    assert client.chat.completions.create.await_args_list[1].kwargs["messages"] == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": expected},
        {"role": "assistant", "content": "How can I help?"},
        {"role": "user", "content": "next turn"},
    ]
    assert history[-2:] == [
        {"role": "user", "content": "next turn"},
        {"role": "assistant", "content": "How can I help?"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("provider unavailable"), GuardrailTripwireTriggered(GuardrailResult(True))])
async def test_failed_turn_preserves_history(example: dict[str, Any], error: Exception) -> None:
    client = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=error)
    history = [{"role": "system", "content": "Be helpful."}]

    with pytest.raises(type(error)):
        await example["process_input"](client, "sample input", history)

    assert history == [{"role": "system", "content": "Be helpful."}]
