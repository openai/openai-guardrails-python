"""Utility helpers for dealing with guardrail execution contexts.

The module exposes :func:`proto_to_model` to turn a ``Protocol`` definition
into a minimal ``BaseModel`` schema and :func:`validate_guardrail_context` to
check runtime objects against such schema.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorDetails

from guardrails.exceptions import ContextValidationError
from guardrails.types import TContext, TIn

if TYPE_CHECKING:
    from ..runtime import ConfiguredGuardrail

import logging

__all__ = ["validate_guardrail_context"]

logger = logging.getLogger(__name__)


def _format_error(err: ErrorDetails) -> str:
    loc = ".".join(map(str, err.get("loc", [])))
    msg = err.get("msg", "Unknown validation error")
    typ = err.get("type", "unknown_error")
    return f"  Pydantic Validation Errors:\n  - {loc}: {msg} (type={typ})"


def validate_guardrail_context(
    guardrail: ConfiguredGuardrail[TContext, TIn, Any],
    ctx: TContext,
) -> None:
    """Validate ``ctx`` against ``guardrail``'s declared context schema.

    Args:
        guardrail: Guardrail whose ``ctx_requirements`` define the schema.
        ctx: Application context instance to validate.

    Raises:
        ContextValidationError: If ``ctx`` does not satisfy required fields.
        TypeError: If ``ctx``'s attributes cannot be introspected.
    """
    model: type[BaseModel] = guardrail.definition.ctx_requirements

    try:
        model.model_validate(ctx, from_attributes=True)
    except ValidationError as exc:
        logger.error(
            "Context validation failed for guardrail '%s'",
            guardrail.definition.name,
        )
        details = "\n".join(map(_format_error, exc.errors()))
        name = guardrail.definition.name
        ctx_requirements = guardrail.definition.ctx_requirements.model_fields
        # Attempt to get application context schema for better error message
        try:
            app_ctx_fields = get_type_hints(ctx)
        except TypeError as exc2:
            msg = f"Context must support attribute access, please pass Context as a class instead of '{type(ctx)}'."
            raise ContextValidationError(msg) from exc2
        # Raise a structured context validation error
        msg = f"Context for '{name}' guardrail expects {ctx_requirements} which does not match ctx schema '{app_ctx_fields}':\n{details}"
        raise ContextValidationError(msg) from exc
