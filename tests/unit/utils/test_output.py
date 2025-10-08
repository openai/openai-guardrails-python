"""Tests for guardrails.utils.output module."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from guardrails.exceptions import ModelBehaviorError, UserError
from guardrails.utils.output import OutputSchema


@dataclass(frozen=True, slots=True)
class _Payload:
    message: str
    count: int


def test_output_schema_wraps_non_text_types() -> None:
    schema = OutputSchema(_Payload)
    json_schema = schema.json_schema()
    assert json_schema["type"] == "object"  # noqa: S101

    validated = schema.validate_json('{"response": {"message": "hi", "count": 2}}')
    assert validated == _Payload(message="hi", count=2)  # noqa: S101


def test_output_schema_plain_text() -> None:
    schema = OutputSchema(str)
    assert schema.is_plain_text() is True  # noqa: S101
    with pytest.raises(UserError):
        schema.json_schema()


def test_output_schema_invalid_json_raises() -> None:
    schema = OutputSchema(_Payload)
    with pytest.raises(ModelBehaviorError):
        schema.validate_json("not-json")
