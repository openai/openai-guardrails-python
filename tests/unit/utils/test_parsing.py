"""Tests for guardrails.utils.parsing."""

from __future__ import annotations

from guardrails.utils.parsing import Entry, format_entries, parse_response_items, parse_response_items_as_json


def test_parse_response_items_handles_messages() -> None:
    items = [
        {
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Hello"},
                "!",
            ],
        },
        {
            "type": "function_call",
            "arguments": "{}",
        },
    ]

    entries = parse_response_items(items)

    assert entries == [Entry(role="user", content="Hello!"), Entry(role="function_call", content="{}")]


def test_parse_response_items_filters_by_role() -> None:
    items = [{"type": "message", "role": "assistant", "content": "Hi"}, {"type": "message", "role": "user", "content": "Bye"}]
    entries = parse_response_items(items, filter_role="assistant")

    assert entries == [Entry(role="assistant", content="Hi")]


def test_parse_response_items_as_json() -> None:
    entries_json = parse_response_items_as_json(
        [{"type": "message", "role": "assistant", "content": "Hi"}],
    )

    assert "assistant" in entries_json  # noqa: S101


def test_format_entries_text() -> None:
    text = format_entries([Entry("assistant", "Hi"), Entry("user", "Bye")], fmt="text")

    assert text == "assistant: Hi\nuser: Bye"
