"""This module provides utilities for handling and validating JSON schema output.

It includes the `OutputSchema` class, which captures, validates, and parses the
JSON schema of the output, and helper functions for type checking and string
representation of types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, get_args, get_origin

from openai.types.responses import ResponseTextConfigParam
from pydantic import BaseModel, TypeAdapter
from typing_extensions import TypedDict

from guardrails.exceptions import ModelBehaviorError, UserError

from .schema import ensure_strict_json_schema, validate_json

_WRAPPER_DICT_KEY = "response"


@dataclass(init=False)
class OutputSchema:
    """An object that captures and validates/parses the JSON schema of the output."""

    output_type: type[Any]
    """The type of the output."""

    _type_adapter: TypeAdapter[Any]
    """A type adapter that wraps the output type, so that we can validate JSON."""

    _is_wrapped: bool
    """Whether the output type is wrapped in a dictionary. This is generally done if the base
    output type cannot be represented as a JSON Schema object.
    """

    _output_schema: dict[str, Any]
    """The JSON schema of the output."""

    strict_json_schema: bool
    """Whether the JSON schema is in strict mode. We **strongly** recommend setting this to True,
    as it increases the likelihood of correct JSON input.
    """

    def __init__(self, output_type: type[Any], strict_json_schema: bool = True):
        """Initialize an OutputSchema for the given output type.

        Args:
            output_type: The target Python type of the LLM output.
            strict_json_schema: Whether to enforce strict JSON schema generation.
        """
        self.output_type = output_type
        self.strict_json_schema = strict_json_schema

        if output_type is None or output_type is str:
            self._is_wrapped = False
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()
            return

        # We should wrap for things that are not plain text, and for things that would definitely
        # not be a JSON Schema object.
        self._is_wrapped = not _is_subclass_of_base_model_or_dict(output_type)

        if self._is_wrapped:
            OutputType = TypedDict(
                "OutputType",
                {
                    _WRAPPER_DICT_KEY: output_type,  # type: ignore
                },
            )
            self._type_adapter = TypeAdapter(OutputType)
            self._output_schema = self._type_adapter.json_schema()
        else:
            self._type_adapter = TypeAdapter(output_type)
            self._output_schema = self._type_adapter.json_schema()

        if self.strict_json_schema:
            self._output_schema = ensure_strict_json_schema(self._output_schema)

    def is_plain_text(self) -> bool:
        """Whether the output type is plain text (versus a JSON object)."""
        return self.output_type is None or self.output_type is str

    def json_schema(self) -> dict[str, Any]:
        """The JSON schema of the output type."""
        if self.is_plain_text():
            raise UserError("Output type is plain text, so no JSON schema is available")
        return self._output_schema

    def validate_json(self, json_str: str, partial: bool = False) -> Any:
        """Validate a JSON string against the output type.

        Returns the validated object, or raises a `ModelBehaviorError` if the JSON is invalid.
        """
        validated = validate_json(json_str, self._type_adapter, partial)
        if self._is_wrapped:
            if not isinstance(validated, dict):
                # TODO: (ovallis) add logging here
                raise ModelBehaviorError(
                    f"Expected a dict, got {type(validated)} for JSON: {json_str}",
                )

            if _WRAPPER_DICT_KEY not in validated:
                # TODO: (ovallis) add logging here
                raise ModelBehaviorError(
                    f"Could not find key {_WRAPPER_DICT_KEY} in JSON: {json_str}",
                )
            return validated[_WRAPPER_DICT_KEY]
        return validated

    def output_type_name(self) -> str:
        """The name of the output type."""
        return _type_to_str(self.output_type)

    def get_response_format(self) -> ResponseTextConfigParam:
        """Return the OpenAI completion parameters for JSON schema output."""
        return {
            "format": {
                "type": "json_schema",
                "name": "final_output",
                "schema": self.json_schema(),
                "strict": self.strict_json_schema,
            },
        }

    # TODO: (ovallis) Add output type.
    def get_completions_format(self):
        """Return the completions API format spec for JSON schema output."""
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "final_output",
                "schema": self.json_schema(),
                "strict": True,
            },
        }


def _is_subclass_of_base_model_or_dict(t: Any) -> bool:
    if not isinstance(t, type):
        return False

    # If it's a generic alias, 'origin' will be the actual type, e.g. 'list'
    origin = get_origin(t)

    allowed_types = (BaseModel, dict)
    # If it's a generic alias e.g. list[str], then we should check the origin type i.e. list
    return issubclass(origin or t, allowed_types)


def _type_to_str(t: type[Any]) -> str:
    origin = get_origin(t)
    args = get_args(t)

    if origin is None:
        # It's a simple type like `str`, `int`, etc.
        return t.__name__
    if args:
        args_str = ", ".join(_type_to_str(arg) for arg in args)
        return f"{origin.__name__}[{args_str}]"
    return str(t)
