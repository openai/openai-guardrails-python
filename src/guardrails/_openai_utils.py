"""Utilities for configuring OpenAI clients used by guardrails."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

SAFETY_IDENTIFIER_HEADER = "OpenAI-Safety-Identifier"
SAFETY_IDENTIFIER_VALUE = "oai_guardrails"


def ensure_safety_identifier_header(default_headers: Mapping[str, str] | None) -> dict[str, str]:
    """Return headers with the Guardrails safety identifier applied."""
    headers = dict(default_headers or {})
    headers[SAFETY_IDENTIFIER_HEADER] = SAFETY_IDENTIFIER_VALUE
    return headers


def prepare_openai_kwargs(openai_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return OpenAI constructor kwargs that include the safety identifier header."""
    prepared_kwargs = dict(openai_kwargs)
    default_headers = prepared_kwargs.get("default_headers")
    headers = ensure_safety_identifier_header(default_headers if isinstance(default_headers, Mapping) else None)
    prepared_kwargs["default_headers"] = headers
    return prepared_kwargs
