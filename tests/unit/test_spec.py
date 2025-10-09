"""Unit tests for the :mod:`guardrails.spec` module."""

import sys
import types
from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import pytest
from pydantic import BaseModel

from guardrails.runtime import ConfiguredGuardrail
from guardrails.spec import GuardrailSpec, GuardrailSpecMetadata
from guardrails.types import GuardrailResult

CUSTOM_VALUE = 123
THRESHOLD_VALUE = 3


@pytest.fixture(autouse=True)
def stub_openai_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Provide a stub ``openai.AsyncOpenAI`` for modules under test."""
    module = types.ModuleType("openai")

    class AsyncOpenAI:
        def __init__(self, **_: object) -> None:
            pass

    module.__dict__["AsyncOpenAI"] = AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    yield module
    monkeypatch.delitem(sys.modules, "openai", raising=False)


class Cfg(BaseModel):
    """Simple configuration model for tests."""

    threshold: int


class Ctx(BaseModel):
    """Dummy context model."""

    model_config = {}


def check(_ctx: Ctx, value: int, cfg: Cfg) -> GuardrailResult:
    """Return ``tripwire_triggered`` if ``value`` exceeds the threshold."""
    _ = _ctx
    return GuardrailResult(tripwire_triggered=value > cfg.threshold)


def make_spec() -> GuardrailSpec[Ctx, int, Cfg]:
    """Create a GuardrailSpec instance for testing."""
    return GuardrailSpec(
        name="gt",
        description="threshold check",
        media_type="text/plain",
        config_schema=Cfg,
        check_fn=check,
        ctx_requirements=Ctx,
        metadata=GuardrailSpecMetadata(engine="python"),
    )


def test_guardrail_spec_is_frozen() -> None:
    """Instances are immutable dataclasses."""
    spec = make_spec()
    with pytest.raises(FrozenInstanceError):
        spec.name = "other"  # type: ignore[misc]


def test_schema_delegates_to_config_schema() -> None:
    """``schema`` returns the JSON schema of the config model."""
    spec = make_spec()
    assert spec.schema() == Cfg.model_json_schema()  # noqa: S101


def test_metadata_allows_extra_fields() -> None:
    """Extra fields are preserved in ``GuardrailSpecMetadata``."""
    data = {"engine": "regex", "custom": CUSTOM_VALUE}
    meta = GuardrailSpecMetadata(**data)
    assert meta.engine == "regex"
    assert meta.custom == CUSTOM_VALUE  # type: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_instantiate_runs_check_function() -> None:
    """``instantiate`` returns a runnable guardrail."""
    spec = make_spec()
    guardrail = spec.instantiate(config=Cfg(threshold=THRESHOLD_VALUE))

    assert isinstance(guardrail, ConfiguredGuardrail)  # noqa: S101
    assert guardrail.definition is spec  # noqa: S101
    assert guardrail.config.threshold == THRESHOLD_VALUE  # noqa: S101

    result = await guardrail.run(Ctx(), THRESHOLD_VALUE + 2)
    assert result.tripwire_triggered  # noqa: S101
