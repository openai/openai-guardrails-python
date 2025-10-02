"""Guardrail specification and model resolution.

This module defines the `GuardrailSpec` dataclass, which captures the metadata,
configuration schema, and logic for a guardrail. It also includes a structured
metadata model for attaching descriptive and extensible information to guardrails,
and instantiation logic for producing executable guardrail instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Generic,
)

from pydantic import BaseModel, ConfigDict, Field

from .types import (
    CheckFn,
    TCfg,
    TContext,
    TIn,
)

if TYPE_CHECKING:
    from .runtime import ConfiguredGuardrail


class GuardrailSpecMetadata(BaseModel):
    """Structured metadata for a guardrail specification.

    This model provides an extensible, strongly-typed way to attach metadata to
    guardrails for discovery, documentation, or engine-specific introspection.

    Attributes:
        engine (str | None): Short string identifying the implementation type or
            engine backing the guardrail (e.g., "regex", "LLM", "API"). Optional.
    """

    engine: str | None = Field(
        default=None,
        description="How the guardrail is implemented (regex/LLM/etc.)",
    )

    model_config = ConfigDict(extra="allow")


@dataclass(frozen=True, slots=True)
class GuardrailSpec(Generic[TContext, TIn, TCfg]):
    """Immutable descriptor for a registered guardrail.

    Encapsulates all static information about a guardrail, including its name,
    human description, supported media type, configuration schema, the validation
    function, context requirements, and optional metadata.

    GuardrailSpec instances are registered for cataloguing and introspection,
    but should be instantiated with user configuration to create a runnable guardrail
    for actual use.

    Attributes:
        name (str): Unique registry key for the guardrail.
        description (str): Human-readable explanation of the guardrail's purpose.
        media_type (str): MIME type to which the guardrail applies (e.g., "text/plain").
        config_schema (type[TCfg]): Pydantic model class describing configuration.
        check_fn (CheckFn[TContext, TIn, TCfg]): Function implementing the guardrail's logic.
        ctx_requirements (type[BaseModel]): Context model describing dependencies or runtime requirements.
        metadata (GuardrailSpecMetadata | None): Optional structured metadata for discovery and documentation.
    """  # noqa: E501

    name: str
    description: str
    media_type: str
    config_schema: type[TCfg]
    check_fn: CheckFn[TContext, TIn, TCfg]
    ctx_requirements: type[BaseModel]
    metadata: GuardrailSpecMetadata | None

    def schema(self) -> dict[str, Any]:
        """Return the JSON schema for the guardrail's configuration model.

        This method provides the schema needed for UI validation, documentation,
        or API introspection.

        Returns:
            dict[str, Any]: JSON schema describing the config model fields.
        """
        return self.config_schema.model_json_schema()

    def instantiate(self, *, config: TCfg) -> ConfiguredGuardrail[TContext, TIn, TCfg]:
        """Produce a configured, executable guardrail from this specification.

        This is the main entry point for creating guardrail instances that can
        be run in a validation pipeline. The returned object is fully bound to
        this definition and the provided configuration.

        Args:
            config (TCfg): Validated configuration for this guardrail.

        Returns:
            ConfiguredGuardrail[TContext, TIn, TCfg]: Runnable guardrail instance.
        """
        from .runtime import ConfiguredGuardrail  # local import to avoid cycle

        return ConfiguredGuardrail(definition=self, config=config)
