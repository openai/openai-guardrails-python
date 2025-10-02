"""Utilities for parsing OpenAI response items into `Entry` objects and formatting them.

It provides:
  - Entry: a record of role and content.
  - parse_response_items: flatten responses into entries with optional filtering.
  - format_entries: render entries as JSON or plain text.
"""

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from openai.types.responses import (
    Response,
    ResponseInputItemParam,
    ResponseOutputItem,
    ResponseStreamEvent,
)
from openai.types.responses.response import Response as OpenAIResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)

__all__ = [
    "Entry",
    "TResponse",
    "TResponseInputItem",
    "TResponseOutputItem",
    "TResponseStreamEvent",
    "format_entries",
    "parse_response_items",
    "parse_response_items_as_json",
]

TResponse = Response
"""A type alias for the Response type from the OpenAI SDK."""

TResponseInputItem = ResponseInputItemParam
"""A type alias for the ResponseInputItemParam type from the OpenAI SDK."""

TResponseOutputItem = ResponseOutputItem
"""A type alias for the ResponseOutputItem type from the OpenAI SDK."""

TResponseStreamEvent = ResponseStreamEvent
"""A type alias for the ResponseStreamEvent type from the OpenAI SDK."""


@dataclass(frozen=True, slots=True)
class Entry:
    """Parsed text entry with role metadata."""

    role: str
    content: str


def _to_mapping(item: Any) -> Mapping[str, Any] | None:
    """Convert BaseModel to dict or pass through Mapping."""
    if isinstance(item, BaseModel):
        return item.model_dump(exclude_none=True)
    if isinstance(item, Mapping):
        return item
    return None


def _parse_message(item: Mapping[str, Any]) -> list[Entry]:
    """Parse both input and output messages (type='message')."""
    role = item["role"]
    contents = item["content"]
    if isinstance(contents, str):
        return [Entry(role, contents)]

    parts: list[str] = []
    for part in contents:
        match part:
            case {"type": t, "text": txt} if t in {"input_text", "output_text"}:
                parts.append(txt)
            case s if isinstance(s, str):
                parts.append(s)
            case _:
                logger.warning("Unknown message part: %s", part)
    return [Entry(role, "".join(parts))]


def _scalar_handler(role: str, key: str) -> Callable[[Mapping[str, Any]], list[Entry]]:
    """Generate handler for single-string fields."""

    def handler(item: Mapping[str, Any]) -> list[Entry]:
        val = item.get(key)
        return [Entry(role, val)] if isinstance(val, str) else []

    return handler


def _list_handler(
    role: str,
    list_key: str,
    text_key: str,
) -> Callable[[Mapping[str, Any]], list[Entry]]:
    """Generate handler for lists of dicts with a text field."""

    def handler(item: Mapping[str, Any]) -> list[Entry]:
        entries: list[Entry] = []
        for elem in item.get(list_key) or []:
            if isinstance(elem, Mapping):
                text = elem.get(text_key)
                if isinstance(text, str):
                    entries.append(Entry(role, text))
        return entries

    return handler


def _default_handler(item: Mapping[str, Any]) -> list[Entry]:
    """Fallback: no text entries."""
    _ = item
    return []


_HANDLER_MAP: dict[str, Callable[[Mapping[str, Any]], list[Entry]]] = {
    "message": _parse_message,
    "function_call": _scalar_handler("function_call", "arguments"),
    "function_call_output": _scalar_handler("function_response", "output"),
    "file_search_call": _list_handler("file_search", "results", "text"),
    "reasoning": _list_handler("reasoning", "summary", "text"),
    "web_search_call": _default_handler,
    "computer_call": _default_handler,
    "computer_call_output": _default_handler,
    "item_reference": _default_handler,
}


def parse_response_items(
    items: OpenAIResponse | list[TResponseInputItem] | list[TResponseOutputItem],
    *,
    filter_role: str | None = None,
    last_n: int | None = None,
    include_types: set[str] | None = None,
    exclude_types: set[str] | None = None,
) -> list[Entry]:
    """Convert response items into a list of Entry objects.

    This function accepts a single `ResponseInputItemParam`, an `OpenAIResponse`
    model (whose `.output` attribute is parsed), or a list of these types.
    It normalizes all inputs to a flat list of `Entry` dataclasses, then applies:

      - Inclusion or exclusion of specific item types.
      - Role-based filtering.
      - Slicing to keep only the last `last_n` entries, if set.

    Args:
        items: A `ResponseInputItemParam`, an `OpenAIResponse`, or a list of these.
        filter_role: Only include entries whose `role` matches this value.
        last_n: If set, only the last `last_n` parsed entries are retained.
        include_types: Only process items whose `type` is in this set.
        exclude_types: Skip items whose `type` is in this set.

    Returns:
        A list of `Entry` objects, each with `role` and `content` fields.
    """
    if not isinstance(items, list):
        items = [items]  # type: ignore[assignment]

    entries: list[Entry] = []
    for raw in items:
        if isinstance(raw, OpenAIResponse):
            raw = raw.output
        else:
            raw = [raw]  # type: ignore[assignment]

        for item in raw:
            mapping = _to_mapping(item)
            if mapping is None:
                logger.warning("Skipped non-mapping item: %s", item)
                continue

            typ = mapping.get("type", "message")
            if include_types and typ not in include_types:
                continue
            if exclude_types and typ in exclude_types:
                continue

            handler = _HANDLER_MAP.get(typ)
            if handler:
                entries.extend(handler(mapping))
            else:
                logger.warning("Unrecognized item type: %s, %s", typ, mapping)

    if filter_role is not None:
        entries = [e for e in entries if e.role == filter_role]

    return entries[-last_n:] if last_n else entries


def format_entries(
    entries: list[Entry],
    fmt: Literal["json", "text"],
    json_kwargs: dict[str, Any] | None = None,
) -> str:
    """Render a list of Entry objects as JSON or plain text.

    Args:
        entries: The list of `Entry` instances to serialize.
        fmt: Output format specifier. `"json"` returns a JSON string;
            `"text"` returns newline-delimited plain text.
        json_kwargs: Keyword arguments to pass through to `json.dumps`
            when `fmt == "json"`. Defaults to {"indent": 2, "ensure_ascii": False}

    Returns:
        A `str` containing the formatted entries.
    """
    if json_kwargs is None:
        json_kwargs = {"indent": 2, "ensure_ascii": False}

    if fmt == "json":
        return json.dumps([asdict(e) for e in entries], **json_kwargs)
    return "\n".join(f"{e.role}: {e.content}" for e in entries)


def parse_response_items_as_json(
    items: OpenAIResponse | list[TResponseInputItem] | list[TResponseOutputItem],
    *,
    filter_role: str | None = None,
    last_n: int | None = None,
    include_types: set[str] | None = None,
    exclude_types: set[str] | None = None,
    json_kwargs: dict[str, Any] | None = None,
) -> str:
    """Parse response items and render as a JSON string."""
    entries = parse_response_items(
        items,
        filter_role=filter_role,
        last_n=last_n,
        include_types=include_types,
        exclude_types=exclude_types,
    )
    return format_entries(entries, "json", json_kwargs=json_kwargs)
