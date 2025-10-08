"""Tests for guardrails.utils.schema utilities."""

from __future__ import annotations

import pytest
from pydantic import BaseModel, TypeAdapter

from guardrails.exceptions import ModelBehaviorError, UserError
from guardrails.utils.schema import ensure_strict_json_schema, validate_json


class _Payload(BaseModel):
    message: str


def test_validate_json_success() -> None:
    adapter = TypeAdapter(_Payload)
    result = validate_json('{"message": "hi"}', adapter, partial=False)

    assert result.message == "hi"  # noqa: S101


def test_validate_json_error() -> None:
    adapter = TypeAdapter(_Payload)
    with pytest.raises(ModelBehaviorError):
        validate_json('{"message": 5}', adapter, partial=False)


def test_ensure_strict_json_schema_enforces_constraints() -> None:
    schema = {
        "type": "object",
        "properties": {
            "message": {"type": "string"},
        },
    }

    strict = ensure_strict_json_schema(schema)

    assert strict["additionalProperties"] is False  # noqa: S101
    assert strict["required"] == ["message"]  # noqa: S101


def test_ensure_strict_json_schema_rejects_additional_properties() -> None:
    schema = {"type": "object", "additionalProperties": True}
    with pytest.raises(UserError):
        ensure_strict_json_schema(schema)
