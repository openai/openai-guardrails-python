"""Registry for managing GuardrailSpec instances and maintaining a catalog of guardrails.

This module provides the in-memory registry that acts as the authoritative
catalog for all available guardrail specifications. The registry supports
registration, lookup, removal, and metadata inspection for guardrails,
enabling extensibility and dynamic discovery across your application.
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from .spec import GuardrailSpec, GuardrailSpecMetadata

if TYPE_CHECKING:  # pragma: no cover - for typing only
    from .types import CheckFn, TContext, TIn

MIME_RE = re.compile(r"^[\w.+-]+/[\w.+-]+$")


class _NoConfig(BaseModel):
    """Sentinel config model for guardrails with no configuration options.

    Used to indicate that a guardrail does not require any config parameters.
    """


class _NoContextRequirements(BaseModel):
    """Sentinel context model for guardrails with no context requirements.

    Used to indicate that a guardrail can run with an empty context.
    """


def _resolve_ctx_requirements(fn: CheckFn) -> type[BaseModel]:  # type: ignore[type-arg]
    """Infer or synthesize the required context type for a guardrail check function.

    This function examines the type hints of the `ctx` parameter of the
    guardrail check function. If the type is a protocol, it generates a
    Pydantic model for context validation; if a `BaseModel`, it is used
    directly; otherwise, it falls back to the no-context sentinel.

    Args:
        fn (CheckFn): The guardrail check function.

    Returns:
        type[BaseModel]: The resolved context model type.
    """
    hints = get_type_hints(fn, include_extras=False)
    param_names = list(inspect.signature(fn).parameters)
    hint = hints.get(param_names[0], None)

    if inspect.isclass(hint) and issubclass(hint, BaseModel):
        return hint
    # If this is a Protocol, we need to create a model from it
    if inspect.isclass(hint) and hasattr(hint, "__annotations__"):
        cfg = ConfigDict(arbitrary_types_allowed=True)
        return create_model(
            f"{hint.__name__}Model",
            __config__=cfg,
            **get_type_hints(hint),
        )
    return _NoContextRequirements


def _resolve_config_schema(fn: CheckFn) -> type[BaseModel]:  # type: ignore[type-arg]
    """Infer the configuration schema model for a guardrail check function.

    Examines the type hint for the `config` parameter. If absent or None,
    returns the no-config sentinel; otherwise, returns the hinted type.

    Args:
        fn (CheckFn): The guardrail check function.

    Returns:
        type[BaseModel]: The resolved config model type.
    """
    hints = get_type_hints(fn, include_extras=False)
    param_names = list(inspect.signature(fn).parameters)
    hint = hints.get(param_names[2], None)
    return _NoConfig if (hint is None or hint is type(None)) else hint


@dataclass(frozen=True, slots=True)
class Metadata:
    """Metadata snapshot for a guardrail specification.

    This container bundles descriptive and structural details about a guardrail
    for inspection, discovery, or documentation.

    Attributes:
        name (str): Unique identifier for the guardrail.
        description (str): Explanation of what the guardrail checks.
        media_type (str): MIME type (e.g. "text/plain") the guardrail applies to.
        config_schema (dict[str, Any] | None): JSON schema for the guardrail's config model.
        metadata (dict[str, Any] | None): Additional metadata (e.g., engine type).
    """

    name: str
    description: str
    media_type: str
    config_schema: dict[str, Any] | None
    metadata: dict[str, Any] | None = None


class GuardrailRegistry:
    """Central registry for all registered guardrail specifications.

    This class provides methods to register, remove, and look up
    :class:`GuardrailSpec` objects by name. It supports dynamic extension
    of available guardrails and powers discovery and validation
    throughout the package.

    Typical usage:
        ```python
        registry = GuardrailRegistry()
        registry.register(...)
        spec = registry.get("my_guardrail")
        all_specs = registry.get_all()
        ```
    """

    def __init__(self) -> None:
        """Initialize an empty registry of guardrail specifications."""
        self._guardrailspecs: dict[str, GuardrailSpec[Any, Any, Any]] = {}
        self._logger = logging.getLogger(__name__ + ".GuardrailSpecRegistry")

    def register(
        self,
        name: str,
        check_fn: CheckFn[TContext, TIn, Any],
        description: str,
        media_type: str,
        *,
        metadata: GuardrailSpecMetadata | None = None,
    ) -> None:
        """Register a new guardrail specification.

        This adds a :class:`GuardrailSpec` to the registry, inferring the required
        context and configuration models from the function signature.

        Args:
            name (str): Unique identifier for the guardrail.
            check_fn (CheckFn): Function that implements the guardrail logic.
            description (str): Human-readable description for docs and discovery.
            media_type (str): MIME type this guardrail operates on.
            metadata (GuardrailSpecMetadata, optional): Additional details for UIs or tooling.

        Raises:
            ValueError: If `media_type` is not a valid MIME type, or if `name`
                is already registered.

        Example:
            ```python
            registry.register(
                name="keyword_filter",
                check_fn=keywords,
                description="Triggers if text contains banned keywords.",
                media_type="text/plain",
            )
            ```
        """
        if name in self._guardrailspecs:
            existing = self._guardrailspecs[name]
            self._logger.error("Duplicate registration attempted for '%s'", name)
            msg = f"Guardrail name '{name}' already bound to {existing.check_fn.__qualname__}. Pick a distinct name or rename the function."
            raise ValueError(msg)

        if isinstance(media_type, str) and not MIME_RE.match(media_type):
            msg = f"Invalid media-type '{media_type}'"
            raise ValueError(msg)

        resolved_ctx = _resolve_ctx_requirements(check_fn)
        resolved_config_schema = _resolve_config_schema(check_fn)

        _name = name or check_fn.__name__
        guardrailspec = GuardrailSpec(
            name=_name,
            description=description,
            media_type=media_type,
            config_schema=resolved_config_schema,
            check_fn=check_fn,
            ctx_requirements=resolved_ctx,
            metadata=metadata or GuardrailSpecMetadata(),
        )

        self._guardrailspecs[guardrailspec.name] = guardrailspec
        self._logger.debug("Registered guardrail spec '%s'", guardrailspec.name)

    def remove(self, name: str) -> None:
        """Remove a registered guardrail specification by name.

        Args:
            name (str): The guardrail name to remove.

        Raises:
            KeyError: If `name` is not present in the registry.
        """
        if name in self._guardrailspecs:
            del self._guardrailspecs[name]
            return
        msg = f"Guardrail spec '{name}' not found."
        raise KeyError(msg)

    def get(self, name: str) -> GuardrailSpec[Any, Any, Any]:
        """Retrieve a registered guardrail specification by name.

        Args:
            name (str): The name passed to :meth:`register`.

        Returns:
            GuardrailSpec: The registered guardrail specification.

        Raises:
            KeyError: If nothing is registered under `name`.
        """
        if name in self._guardrailspecs:
            return self._guardrailspecs[name]
        self._logger.warning("Attempted lookup of unknown guardrail '%s'", name)
        msg = f"Guardrail spec '{name}' not found."
        raise KeyError(msg)

    def get_all(self) -> list[GuardrailSpec[Any, Any, Any]]:
        """Return a list of all registered guardrail specifications.

        Returns:
            list[GuardrailSpec]: All registered specs, in registration order.
        """
        return list(self._guardrailspecs.values())

    def get_all_metadata(self) -> list[Metadata]:
        """Return summary metadata for all registered guardrail specifications.

        This provides lightweight, serializable descriptions of all guardrails,
        suitable for documentation, UI display, or catalog listing.

        Returns:
            list[Metadata]: List of metadata entries for each registered spec.
        """
        return [
            Metadata(
                name=d.name,
                description=d.description,
                media_type=d.media_type,
                config_schema=d.schema(),
                metadata=d.metadata.model_dump() if d.metadata else {},
            )
            for d in self._guardrailspecs.values()
        ]


default_spec_registry = GuardrailRegistry()
"""Global default registry for guardrail specifications.

This instance should be used for registration and lookup unless a custom
registry is explicitly required.
"""
