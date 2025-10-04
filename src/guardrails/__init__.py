"""Guardrails public API surface.

This package exposes utilities to define and run guardrails which validate
arbitrary data. The submodules provide runtime helpers, exception
types and a registry of built-in checks.

Modules within :mod:`guardrails` are imported lazily to keep the import surface
small when only a few utilities are needed.
"""

import logging as _logging
from importlib import metadata as _m

from . import checks
from .agents import GuardrailAgent
from .client import (
    GuardrailResults,
    GuardrailsAsyncOpenAI,
    GuardrailsOpenAI,
    GuardrailsResponse,
)

try:  # Optional Azure variants
    from .client import GuardrailsAsyncAzureOpenAI, GuardrailsAzureOpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency path
    GuardrailsAsyncAzureOpenAI = None  # type: ignore
    GuardrailsAzureOpenAI = None  # type: ignore
# Import resources for access to resource classes
from . import resources
from .exceptions import GuardrailTripwireTriggered
from .registry import default_spec_registry
from .runtime import (
    ConfigSource,
    ConfiguredGuardrail,
    JsonString,
    check_plain_text,
    instantiate_guardrails,
    load_config_bundle,
    load_pipeline_bundles,
    run_guardrails,
)
from .spec import GuardrailSpecMetadata
from .types import GuardrailResult

__all__ = [
    "ConfiguredGuardrail",  # configured, executable object
    "GuardrailAgent",  # drop-in replacement for Agents SDK Agent
    "GuardrailResult",
    "GuardrailResults",  # organized guardrail results by stage
    "GuardrailTripwireTriggered",
    "GuardrailsAsyncOpenAI",  # async OpenAI subclass with guardrails
    "GuardrailsOpenAI",  # sync OpenAI subclass with guardrails
    "GuardrailsAsyncAzureOpenAI",  # async Azure OpenAI subclass with guardrails
    "GuardrailsAzureOpenAI",  # sync Azure OpenAI subclass with guardrails
    "GuardrailsResponse",  # response wrapper with guardrail results
    "check_plain_text",
    "checks",
    "JsonString",
    "ConfigSource",
    "run_guardrails",
    "GuardrailSpecMetadata",
    "instantiate_guardrails",
    "load_config_bundle",
    "load_pipeline_bundles",
    "default_spec_registry",
    "resources",  # resource modules
]

__version__: str = _m.version("openai-guardrails")

# Expose a package-level logger and install a NullHandler so importing the
# library never configures global logging for the host application.
# Users can obtain module-specific loggers via ``logging.getLogger(__name__)``
# and configure handlers/levels as they see fit.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())
