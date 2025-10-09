"""This module provides utilities for ensuring JSON schemas conform to a strict standard.

Functions:
    ensure_strict_json_schema: Ensures a given JSON schema adheres to the strict standard.
    resolve_ref: Resolves JSON Schema `$ref` pointers within a schema object.
    is_dict: Type guard to check if an object is a JSON-style dictionary.
    is_list: Type guard to check if an object is a list of items.
    has_more_than_n_keys: Checks if a dictionary has more than a specified number of keys.

Constants:
    _EMPTY_SCHEMA: A predefined empty JSON schema with strict settings.

Exceptions:
    UserError: Raised when invalid schema configurations are encountered.
"""

import logging
from typing import Any, Literal, TypeGuard, TypeVar

from openai import NOT_GIVEN
from pydantic import TypeAdapter, ValidationError

from guardrails.exceptions import ModelBehaviorError, UserError

logger = logging.getLogger(__name__)

T = TypeVar("T")

_EMPTY_SCHEMA = {
    "additionalProperties": False,
    "type": "object",
    "properties": {},
    "required": [],
}


def validate_json(json_str: str, type_adapter: TypeAdapter[T], partial: bool) -> T:
    """Validate and parse a JSON string using a Pydantic TypeAdapter.

    Args:
        json_str: The JSON string to validate and parse.
        type_adapter: A Pydantic TypeAdapter for the target type T.
        partial: If True, allow partial JSON parsing (trailing content).

    Returns:
        The parsed object of type T.

    Raises:
        ModelBehaviorError: If JSON parsing or validation fails.
    """
    partial_setting: bool | Literal["off", "on", "trailing-strings"] = "trailing-strings" if partial else False
    try:
        validated = type_adapter.validate_json(
            json_str,
            experimental_allow_partial=partial_setting,
        )
        return validated
    except ValidationError as e:
        logger.debug("JSON validation failed", exc_info=e)
        raise ModelBehaviorError(
            f"Invalid JSON when parsing {json_str} for {type_adapter}; {e}",
        ) from e


def ensure_strict_json_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Mutates the given JSON schema.

    This ensures it conforms to the `strict` standard that the OpenAI API expects.
    """
    if schema == {}:
        return _EMPTY_SCHEMA
    return _ensure_strict_json_schema(schema, path=(), root=schema)


# Adapted from https://github.com/openai/openai-python/blob/main/src/openai/lib/_pydantic.py
def _ensure_strict_json_schema(
    json_schema: object,
    *,
    path: tuple[str, ...],
    root: dict[str, object],
) -> dict[str, Any]:
    if not is_dict(json_schema):
        raise TypeError(f"Expected {json_schema} to be a dictionary; path={path}")

    defs = json_schema.get("$defs")
    if is_dict(defs):
        for def_name, def_schema in defs.items():
            _ensure_strict_json_schema(
                def_schema,
                path=(*path, "$defs", def_name),
                root=root,
            )

    definitions = json_schema.get("definitions")
    if is_dict(definitions):
        for definition_name, definition_schema in definitions.items():
            _ensure_strict_json_schema(
                definition_schema,
                path=(*path, "definitions", definition_name),
                root=root,
            )

    typ = json_schema.get("type")
    if typ == "object" and "additionalProperties" not in json_schema:
        json_schema["additionalProperties"] = False
    elif typ == "object" and "additionalProperties" in json_schema and json_schema["additionalProperties"]:
        raise UserError(
            "additionalProperties should not be set for object types. This could be because "
            "you're using an older version of Pydantic, or because you configured additional "
            "properties to be allowed. If you really need this, update the function or output tool "
            "to not use a strict schema.",
        )

    # object types
    # { 'type': 'object', 'properties': { 'a':  {...} } }
    properties = json_schema.get("properties")
    if is_dict(properties):
        json_schema["required"] = list(properties.keys())
        json_schema["properties"] = {
            key: _ensure_strict_json_schema(
                prop_schema,
                path=(*path, "properties", key),
                root=root,
            )
            for key, prop_schema in properties.items()
        }

    # arrays
    # { 'type': 'array', 'items': {...} }
    items = json_schema.get("items")
    if is_dict(items):
        json_schema["items"] = _ensure_strict_json_schema(
            items,
            path=(*path, "items"),
            root=root,
        )

    # unions
    any_of = json_schema.get("anyOf")
    if is_list(any_of):
        json_schema["anyOf"] = [
            _ensure_strict_json_schema(
                variant,
                path=(*path, "anyOf", str(i)),
                root=root,
            )
            for i, variant in enumerate(any_of)
        ]

    # intersections
    all_of = json_schema.get("allOf")
    if is_list(all_of):
        if len(all_of) == 1:
            json_schema.update(
                _ensure_strict_json_schema(
                    all_of[0],
                    path=(*path, "allOf", "0"),
                    root=root,
                ),
            )
            json_schema.pop("allOf")
        else:
            json_schema["allOf"] = [
                _ensure_strict_json_schema(
                    entry,
                    path=(*path, "allOf", str(i)),
                    root=root,
                )
                for i, entry in enumerate(all_of)
            ]

    # strip `None` defaults as there's no meaningful distinction here
    # the schema will still be `nullable` and the model will default
    # to using `None` anyway
    if json_schema.get("default", NOT_GIVEN) is None:
        json_schema.pop("default")

    # we can't use `$ref`s if there are also other properties defined, e.g.
    # `{"$ref": "...", "description": "my description"}`
    #
    # so we unravel the ref
    # `{"type": "string", "description": "my description"}`
    ref = json_schema.get("$ref")
    if ref and has_more_than_n_keys(json_schema, 1):
        assert isinstance(ref, str), f"Received non-string $ref - {ref}"

        resolved = resolve_ref(root=root, ref=ref)
        if not is_dict(resolved):
            raise ValueError(
                f"Expected `$ref: {ref}` to resolved to a dictionary but got {resolved}",
            )

        # properties from the json schema take priority over the ones on the `$ref`
        json_schema.update({**resolved, **json_schema})
        json_schema.pop("$ref")
        # Since the schema expanded from `$ref` might not have `additionalProperties: false` applied
        # we call `_ensure_strict_json_schema` again to fix the inlined schema and ensure it's valid
        return _ensure_strict_json_schema(json_schema, path=path, root=root)

    return json_schema


def resolve_ref(*, root: dict[str, object], ref: str) -> object:
    """Resolve a JSON Schema `$ref` pointer within a schema object.

    Args:
        root: The root JSON schema dictionary.
        ref: A reference string starting with "#/" indicating path in schema.

    Returns:
        The object within the schema that the reference points to.

    Raises:
        ValueError: If the reference format is invalid or resolution fails.
    """
    if not ref.startswith("#/"):
        raise ValueError(f"Unexpected $ref format {ref!r}; Does not start with #/")

    path = ref[2:].split("/")
    resolved = root
    for key in path:
        value = resolved[key]
        assert is_dict(value), f"encountered non-dictionary entry while resolving {ref} - {resolved}"
        resolved = value

    return resolved


def is_dict(obj: object) -> TypeGuard[dict[str, object]]:
    """Type guard to check if an object is a JSON-style dict.

    Args:
        obj: The object to test.

    Returns:
        True if `obj` is a dict, False otherwise.
    """
    # just pretend that we know there are only `str` keys
    # as that check is not worth the performance cost
    return isinstance(obj, dict)


def is_list(obj: object) -> TypeGuard[list[object]]:
    """Type guard to check if an object is a list of items.

    Args:
        obj: The object to test.

    Returns:
        True if `obj` is a list, False otherwise.
    """
    return isinstance(obj, list)


def has_more_than_n_keys(obj: dict[str, object], n: int) -> bool:
    """Check whether a dict has more than `n` keys without counting them all.

    Args:
        obj: The dictionary to inspect.
        n: The key-count threshold.

    Returns:
        True if `obj` contains more than `n` keys; False otherwise.
    """
    i = 0
    for _ in obj:
        i += 1
        if i > n:
            return True
    return False
