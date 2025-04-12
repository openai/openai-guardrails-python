from __future__ import annotations

from typing import Literal

from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypeVar

from ..exceptions import ModelBehaviorError

T = TypeVar("T")


def validate_json(json_str: str, type_adapter: TypeAdapter[T], partial: bool) -> T:
    partial_setting: bool | Literal["off", "on", "trailing-strings"] = (
        "trailing-strings" if partial else False
    )
    try:
        validated = type_adapter.validate_json(
            json_str, experimental_allow_partial=partial_setting
        )
        return validated
    except ValidationError as e:
        # TODO: (ovallis) add logging here
        raise ModelBehaviorError(
            f"Invalid JSON when parsing {json_str} for {type_adapter}; {e}"
        ) from e
