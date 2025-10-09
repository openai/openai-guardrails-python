"""Unit tests for registry module."""

import sys
import types
from collections.abc import Iterator
from typing import Protocol

import pytest


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


def test_resolve_ctx_protocol_creates_model() -> None:
    """Protocols yield pydantic models with matching fields."""
    from pydantic import BaseModel

    from guardrails.registry import _resolve_ctx_requirements
    from guardrails.types import GuardrailResult

    class CtxProto(Protocol):
        foo: int

    def check(_ctx: CtxProto, _value: str, _config: int) -> GuardrailResult:
        _, _, _ = _ctx, _value, _config
        return GuardrailResult(tripwire_triggered=False)

    model = _resolve_ctx_requirements(check)
    # Prefer Pydantic v2 API without eagerly touching deprecated v1 attributes
    fields = model.model_fields if hasattr(model, "model_fields") else getattr(model, "__fields__", {})
    assert issubclass(model, BaseModel)  # noqa: S101
    assert set(fields) == {"foo"}  # noqa: S101


def test_resolve_ctx_basemodel_passthrough() -> None:
    """BaseModel annotations are returned unchanged."""
    from pydantic import BaseModel

    from guardrails.registry import _resolve_ctx_requirements
    from guardrails.types import GuardrailResult

    class DummyCtx(BaseModel):
        bar: str

    def check(_ctx: DummyCtx, _value: str, _config: int) -> GuardrailResult:
        _, _, _ = _ctx, _value, _config
        return GuardrailResult(tripwire_triggered=False)

    assert _resolve_ctx_requirements(check) is DummyCtx


def test_resolve_config_schema() -> None:
    """Config type hints control schema resolution."""
    from pydantic import BaseModel

    from guardrails.registry import _NoConfig, _resolve_config_schema
    from guardrails.types import GuardrailResult

    class Cfg(BaseModel):
        threshold: int

    def with_cfg(_ctx: object, _value: str, _config: Cfg) -> GuardrailResult:
        _, _, _ = _ctx, _value, _config
        return GuardrailResult(tripwire_triggered=False)

    def without_cfg(_ctx: object, _value: str, _config: None) -> GuardrailResult:
        _, _, _ = _ctx, _value, _config
        return GuardrailResult(tripwire_triggered=False)

    assert _resolve_config_schema(with_cfg) is Cfg  # noqa: S101
    assert _resolve_config_schema(without_cfg) is _NoConfig  # noqa: S101


def test_registry_crud_and_metadata() -> None:
    """Registry registers, retrieves, and enumerates specs."""
    from pydantic import BaseModel

    from guardrails.registry import GuardrailRegistry
    from guardrails.spec import GuardrailSpecMetadata
    from guardrails.types import GuardrailResult

    class Ctx(BaseModel):
        user: str

    class Cfg(BaseModel):
        level: int

    def check(_ctx: Ctx, _value: str, _config: Cfg) -> GuardrailResult:
        _, _, _ = _ctx, _value, _config
        return GuardrailResult(tripwire_triggered=False)

    reg = GuardrailRegistry()
    reg.register(
        name="test",
        check_fn=check,
        description="desc",
        media_type="text/plain",
    )
    reg.register(
        name="other",
        check_fn=check,
        description="desc",
        media_type="text/plain",
        metadata=GuardrailSpecMetadata(engine="regex"),
    )

    spec = reg.get("test")
    assert spec.name == "test"  # noqa: S101
    assert spec.check_fn is check  # noqa: S101

    all_specs = {s.name for s in reg.get_all()}
    assert all_specs == {"test", "other"}  # noqa: S101

    meta_names = {m.name for m in reg.get_all_metadata()}
    assert meta_names == {"test", "other"}  # noqa: S101


def test_register_invalid_input_and_remove() -> None:
    """Duplicate names or bad media types raise errors."""
    from pydantic import BaseModel

    from guardrails.registry import GuardrailRegistry
    from guardrails.types import GuardrailResult

    class Ctx(BaseModel):
        pass

    class Cfg(BaseModel):
        pass

    def check(_ctx: Ctx, _value: str, _config: Cfg) -> GuardrailResult:
        _, _, _ = _ctx, _value, _config
        return GuardrailResult(tripwire_triggered=False)

    reg = GuardrailRegistry()
    reg.register(
        name="dup",
        check_fn=check,
        description="d",
        media_type="text/plain",
    )

    with pytest.raises(ValueError, match="already bound"):
        reg.register(
            name="dup",
            check_fn=check,
            description="d",
            media_type="text/plain",
        )

    with pytest.raises(ValueError, match="Invalid media-type"):
        reg.register(
            name="bad",
            check_fn=check,
            description="d",
            media_type="not-a-type",
        )

    reg.remove("dup")
    with pytest.raises(KeyError):
        reg.get("dup")
