"""Utilities for normalizing conversation history across providers.

The helpers in this module transform arbitrary chat/response payloads into a
consistent list of dictionaries that guardrails can consume. The structure is
intended to capture the semantic roles of user/assistant turns as well as tool
calls and outputs regardless of the originating API.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    """Normalized representation of a conversation item.

    Attributes:
        role: Logical speaker role (user, assistant, system, tool, etc.).
        type: Optional type discriminator for non-message items such as
            ``function_call`` or ``function_call_output``.
        content: Primary text payload for message-like items.
        tool_name: Name of the tool/function associated with the entry.
        arguments: Serialized tool/function arguments when available.
        output: Serialized tool result payload when available.
        call_id: Identifier that links tool calls and outputs.
    """

    role: str | None = None
    type: str | None = None
    content: str | None = None
    tool_name: str | None = None
    arguments: str | None = None
    output: str | None = None
    call_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        """Convert entry to a plain dict, omitting null fields."""
        payload: dict[str, Any] = {}
        if self.role is not None:
            payload["role"] = self.role
        if self.type is not None:
            payload["type"] = self.type
        if self.content is not None:
            payload["content"] = self.content
        if self.tool_name is not None:
            payload["tool_name"] = self.tool_name
        if self.arguments is not None:
            payload["arguments"] = self.arguments
        if self.output is not None:
            payload["output"] = self.output
        if self.call_id is not None:
            payload["call_id"] = self.call_id
        return payload


def normalize_conversation(
    conversation: str | Mapping[str, Any] | Sequence[Any] | None,
) -> list[dict[str, Any]]:
    """Normalize arbitrary conversation payloads to guardrail-friendly dicts.

    Args:
        conversation: Conversation history expressed as a raw string (single
            user turn), a mapping/object representing a message, or a sequence
            of messages/items.

    Returns:
        List of dictionaries describing the conversation in chronological order.
    """
    if conversation is None:
        return []

    if isinstance(conversation, str):
        entry = ConversationEntry(role="user", content=conversation)
        return [entry.to_payload()]

    if isinstance(conversation, Mapping):
        entries = _normalize_item(conversation)
        return [entry.to_payload() for entry in entries]

    if isinstance(conversation, Sequence):
        normalized: list[ConversationEntry] = []
        for item in conversation:
            normalized.extend(_normalize_item(item))
        return [entry.to_payload() for entry in normalized]

    # Fallback: treat the value as a message-like object.
    entries = _normalize_item(conversation)
    return [entry.to_payload() for entry in entries]


def append_assistant_response(
    conversation: Iterable[dict[str, Any]],
    llm_response: Any,
) -> list[dict[str, Any]]:
    """Append the assistant response to a normalized conversation copy.

    Args:
        conversation: Existing normalized conversation.
        llm_response: Response object returned from the model call.

    Returns:
        New conversation list containing the assistant's response entries.
    """
    base = [entry.copy() for entry in conversation]
    response_entries = _normalize_model_response(llm_response)
    base.extend(entry.to_payload() for entry in response_entries)
    return base


def merge_conversation_with_items(
    conversation: Iterable[dict[str, Any]],
    items: Sequence[Any],
) -> list[dict[str, Any]]:
    """Return a new conversation list with additional items appended.

    Args:
        conversation: Existing normalized conversation.
        items: Additional items (tool calls, tool outputs, messages) to append.

    Returns:
        List representing the combined conversation.
    """
    base = [entry.copy() for entry in conversation]
    for entry in _normalize_sequence(items):
        base.append(entry.to_payload())
    return base


def _normalize_sequence(items: Sequence[Any]) -> list[ConversationEntry]:
    entries: list[ConversationEntry] = []
    for item in items:
        entries.extend(_normalize_item(item))
    return entries


def _normalize_item(item: Any) -> list[ConversationEntry]:
    """Normalize a single message or tool record."""
    if item is None:
        return []

    if isinstance(item, Mapping):
        return _normalize_mapping(item)

    if hasattr(item, "model_dump"):
        return _normalize_mapping(item.model_dump(exclude_unset=True))

    if hasattr(item, "__dict__"):
        return _normalize_mapping(vars(item))

    if isinstance(item, str):
        return [ConversationEntry(role="user", content=item)]

    return [ConversationEntry(content=_stringify(item))]


def _normalize_mapping(item: Mapping[str, Any]) -> list[ConversationEntry]:
    entries: list[ConversationEntry] = []
    item_type = item.get("type")

    if item_type in {"function_call", "tool_call"}:
        entries.append(
            ConversationEntry(
                type="function_call",
                tool_name=_extract_tool_name(item),
                arguments=_stringify(item.get("arguments") or item.get("function", {}).get("arguments")),
                call_id=_stringify(item.get("call_id") or item.get("id")),
            )
        )
        return entries

    if item_type == "function_call_output":
        entries.append(
            ConversationEntry(
                type="function_call_output",
                tool_name=_extract_tool_name(item),
                arguments=_stringify(item.get("arguments")),
                output=_extract_text(item.get("output")),
                call_id=_stringify(item.get("call_id")),
            )
        )
        return entries

    role = item.get("role")
    if role is not None:
        entries.extend(_normalize_role_message(role, item))
        return entries

    # Fallback path for message-like objects without explicit role/type.
    entries.append(
        ConversationEntry(
            content=_extract_text(item.get("content") if "content" in item else item),
            type=item_type if isinstance(item_type, str) else None,
        )
    )
    return entries


def _normalize_role_message(role: str, item: Mapping[str, Any]) -> list[ConversationEntry]:
    entries: list[ConversationEntry] = []
    text = _extract_text(item.get("content"))
    if role != "tool":
        entries.append(ConversationEntry(role=role, content=text))

    # Normalize inline tool calls/functions.
    tool_calls = item.get("tool_calls")
    if isinstance(tool_calls, Sequence):
        entries.extend(_normalize_tool_calls(tool_calls))

    function_call = item.get("function_call")
    if isinstance(function_call, Mapping):
        entries.append(
            ConversationEntry(
                type="function_call",
                tool_name=_stringify(function_call.get("name")),
                arguments=_stringify(function_call.get("arguments")),
                call_id=_stringify(function_call.get("call_id")),
            )
        )

    if role == "tool":
        tool_output = ConversationEntry(
            type="function_call_output",
            tool_name=_extract_tool_name(item),
            output=text,
            arguments=_stringify(item.get("arguments")),
            call_id=_stringify(item.get("tool_call_id") or item.get("call_id")),
        )
        return [entry for entry in [tool_output] if any(entry.to_payload().values())]

    return [entry for entry in entries if any(entry.to_payload().values())]


def _normalize_tool_calls(tool_calls: Sequence[Any]) -> list[ConversationEntry]:
    entries: list[ConversationEntry] = []
    for call in tool_calls:
        if hasattr(call, "model_dump"):
            call_mapping = call.model_dump(exclude_unset=True)
        elif isinstance(call, Mapping):
            call_mapping = call
        else:
            call_mapping = {}

        entries.append(
            ConversationEntry(
                type="function_call",
                tool_name=_extract_tool_name(call_mapping),
                arguments=_stringify(call_mapping.get("arguments") or call_mapping.get("function", {}).get("arguments")),
                call_id=_stringify(call_mapping.get("id") or call_mapping.get("call_id")),
            )
        )
    return entries


def _extract_tool_name(item: Mapping[str, Any]) -> str | None:
    if "tool_name" in item and isinstance(item["tool_name"], str):
        return item["tool_name"]
    if "name" in item and isinstance(item["name"], str):
        return item["name"]
    function = item.get("function")
    if isinstance(function, Mapping):
        name = function.get("name")
        if isinstance(name, str):
            return name
    return None


def _extract_text(content: Any) -> str | None:
    if content is None:
        return None

    if isinstance(content, str):
        return content

    if isinstance(content, Mapping):
        text = content.get("text")
        if isinstance(text, str):
            return text
        return _extract_text(content.get("content"))

    if isinstance(content, Sequence) and not isinstance(content, bytes | bytearray):
        parts: list[str] = []
        for item in content:
            extracted = _extract_text(item)
            if extracted:
                parts.append(extracted)
        joined = " ".join(part for part in parts if part)
        return joined or None

    return _stringify(content)


def _normalize_model_response(response: Any) -> list[ConversationEntry]:
    if response is None:
        return []

    if hasattr(response, "output"):
        output = response.output
        if isinstance(output, Sequence):
            return _normalize_sequence(output)

    if hasattr(response, "choices"):
        choices = response.choices
        if isinstance(choices, Sequence) and choices:
            choice = choices[0]
            message = getattr(choice, "message", choice)
            return _normalize_item(message)

    # Streaming deltas often expose ``delta`` with message fragments.
    delta = getattr(response, "delta", None)
    if delta:
        return _normalize_item(delta)

    return []


def _stringify(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)
