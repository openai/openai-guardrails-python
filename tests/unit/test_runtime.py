"""Tests for the runtime module."""

import sys
import types
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

import pytest
from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError

from guardrails.exceptions import ConfigError, ContextValidationError, GuardrailTripwireTriggered
from guardrails.registry import GuardrailRegistry
from guardrails.runtime import (
    ConfigBundle,
    GuardrailConfig,
    PipelineBundles,
    check_plain_text,
    instantiate_guardrails,
    load_config_bundle,
    load_pipeline_bundles,
    run_guardrails,
)
from guardrails.types import GuardrailResult

THRESHOLD = 2


@pytest.fixture(autouse=True)
def stub_openai_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Provide a stub ``openai.AsyncOpenAI`` and patch imports in guardrails.*.

    Ensures tests don't require real OPENAI_API_KEY or networked clients.
    """
    module = types.ModuleType("openai")

    class AsyncOpenAI:  # noqa: D401 - simple stub
        """Stubbed AsyncOpenAI client."""

        def __init__(self, **_: object) -> None:
            pass

    module.__dict__["AsyncOpenAI"] = AsyncOpenAI
    # Ensure any downstream import finds our stub module
    monkeypatch.setitem(sys.modules, "openai", module)
    # Also patch already-imported symbols on guardrails modules
    try:
        import guardrails.runtime as gr_runtime  # type: ignore

        monkeypatch.setattr(gr_runtime, "AsyncOpenAI", AsyncOpenAI, raising=False)
    except Exception:
        pass
    try:
        import guardrails.types as gr_types  # type: ignore

        monkeypatch.setattr(gr_types, "AsyncOpenAI", AsyncOpenAI, raising=False)
    except Exception:
        pass
    # Provide dummy API key to satisfy any code paths that inspect env
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    yield module
    monkeypatch.delitem(sys.modules, "openai", raising=False)


class LenCfg(BaseModel):
    """Configuration specifying length threshold."""

    threshold: int


class CtxProto(Protocol):
    """Protocol for context object."""

    user: str


def len_check(ctx: CtxProto, data: str, config: LenCfg) -> GuardrailResult:
    """Return result triggered when ``data`` length exceeds threshold."""
    _ = ctx
    return GuardrailResult(tripwire_triggered=len(data) > config.threshold)


def no_ctx_len_check(ctx: Any, data: str, config: LenCfg) -> GuardrailResult:
    """Return result triggered when ``data`` length exceeds threshold."""
    _ = ctx
    return GuardrailResult(tripwire_triggered=len(data) > config.threshold)


def build_registry() -> GuardrailRegistry:
    """Return registry with length guard registered."""
    registry = GuardrailRegistry()
    registry.register(
        name="len",
        check_fn=len_check,
        description="length guard",
        media_type="text/plain",
    )
    registry.register(
        name="no ctx len",
        check_fn=no_ctx_len_check,
        description="no ctx length guard",
        media_type="text/plain",
    )
    return registry


@dataclass
class Ctx:
    """Minimal context containing ``user`` field."""

    user: str
    foo: str = "unused"


def test_load_config_bundle_dict_roundtrip() -> None:
    """Dict input converts to ConfigBundle."""
    bundle = load_config_bundle({"version": 1, "guardrails": []})
    assert isinstance(bundle, ConfigBundle)  # noqa: S101
    assert bundle.guardrails == []  # noqa: S101


def test_load_config_bundle_errors_on_invalid_dict() -> None:
    """Invalid structure raises ValidationError."""
    with pytest.raises(ValidationError):
        load_config_bundle({"version": 1})


@given(st.text())
def test_load_config_bundle_plain_string_invalid(text: str) -> None:
    """Plain strings are rejected."""
    with pytest.raises(ConfigError):
        load_config_bundle(text)  # type: ignore[arg-type]


def test_load_pipeline_bundles_dict_roundtrip() -> None:
    """Dict input converts to PipelineBundles with all stages."""
    bundle = load_pipeline_bundles(
        {
            "version": 1,
            "pre_flight": {"version": 1, "guardrails": []},
            "input": {"version": 1, "guardrails": []},
            "output": {"version": 1, "guardrails": []},
        }
    )
    assert isinstance(bundle, PipelineBundles)  # noqa: S101
    assert bundle.pre_flight is not None and bundle.pre_flight.guardrails == []  # noqa: S101
    assert bundle.input is not None and bundle.input.guardrails == []  # noqa: S101
    assert bundle.output is not None and bundle.output.guardrails == []  # noqa: S101


def test_load_pipeline_bundles_single_stage() -> None:
    """Dict input converts to PipelineBundles with just output stage."""
    bundle = load_pipeline_bundles(
        {
            "version": 1,
            "output": {"version": 1, "guardrails": []},
        }
    )
    assert isinstance(bundle, PipelineBundles)  # noqa: S101
    assert bundle.pre_flight is None  # noqa: S101
    assert bundle.input is None  # noqa: S101
    assert bundle.output is not None and bundle.output.guardrails == []  # noqa: S101
    assert len(bundle.stages()) == 1  # noqa: S101


def test_load_pipeline_bundles_no_stages() -> None:
    """PipelineBundles requires at least one stage."""
    with pytest.raises(ValueError, match="At least one stage"):
        load_pipeline_bundles({"version": 1})


def test_load_pipeline_bundles_errors_on_invalid_dict() -> None:
    """Invalid structure raises ValidationError."""
    with pytest.raises(ValidationError):
        load_pipeline_bundles({"version": 1, "invalid": "field"})


@given(st.text())
def test_load_pipeline_bundles_plain_string_invalid(text: str) -> None:
    """Plain strings are rejected."""
    with pytest.raises(ConfigError):
        load_pipeline_bundles(text)  # type: ignore[arg-type]


def test_instantiate_guardrails_happy_path() -> None:
    """Config data is validated and bound."""
    registry = build_registry()
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="len", config={"threshold": THRESHOLD})],
        version=1,
    )
    guardrails = instantiate_guardrails(bundle, registry)
    assert guardrails[0].config.threshold == THRESHOLD  # noqa: S101


def test_instantiate_guardrails_invalid_config() -> None:
    """Missing config fields raise ConfigError."""
    registry = build_registry()
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="len", config={})],
        version=1,
    )
    with pytest.raises(ConfigError):
        instantiate_guardrails(bundle, registry)


@pytest.mark.asyncio
async def test_run_guardrails_suppresses_tripwire_if_requested() -> None:
    """Tripwire results are returned but do not raise if suppression is enabled."""
    registry = build_registry()
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="len", config={"threshold": 0})],
        version=1,
    )
    guardrails = instantiate_guardrails(bundle, registry)
    results = await run_guardrails(
        ctx=Ctx("me"),
        data="trigger",
        media_type="text/plain",
        guardrails=guardrails,
        suppress_tripwire=True,
    )
    assert results[0].tripwire_triggered  # noqa: S101


@pytest.mark.asyncio
async def test_run_guardrails_raises_on_tripwire_by_default() -> None:
    """Tripwire results should raise exception by default."""
    registry = build_registry()
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="len", config={"threshold": 0})],
        version=1,
    )
    guardrails = instantiate_guardrails(bundle, registry)
    with pytest.raises(GuardrailTripwireTriggered):
        await run_guardrails(
            ctx=Ctx("me"),
            data="trigger",
            media_type="text/plain",
            guardrails=guardrails,
        )


@pytest.mark.asyncio
async def test_run_guardrails_with_handler_and_context() -> None:
    """Results flow through handler when no tripwire triggers."""
    registry = build_registry()
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="len", config={"threshold": 10})],
        version=1,
    )
    guardrails = instantiate_guardrails(bundle, registry)
    captured: list[GuardrailResult] = []

    async def handler(result: GuardrailResult) -> None:
        captured.append(result)

    results = await run_guardrails(
        ctx=Ctx("me"),
        data="ok",
        media_type="text/plain",
        guardrails=guardrails,
        result_handler=handler,
    )
    assert len(results) == 1  # noqa: S101
    assert captured == results  # noqa: S101


@pytest.mark.asyncio
async def test_context_validation_failure() -> None:
    """Invalid context is rejected."""
    registry = build_registry()
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="len", config={"threshold": 10})],
        version=1,
    )
    guardrails = instantiate_guardrails(bundle, registry)

    class BadCtx:
        pass

    with pytest.raises(ContextValidationError):
        await run_guardrails(
            ctx=BadCtx(),
            data="data",
            media_type="text/plain",
            guardrails=guardrails,
        )


@pytest.mark.asyncio
async def test_check_plain_text_integration(tmp_path) -> None:
    """End-to-end helper returns results for text input."""
    registry = build_registry()
    print("registry", registry)
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="len", config={"threshold": THRESHOLD})],
        version=1,
    )

    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    bundle_path = cfg_dir / "bundle.json"
    bundle_path.write_text(bundle.model_dump_json())
    result = await check_plain_text("hi", bundle_path, registry=registry, ctx=Ctx("me"))
    assert len(result) == 1  # noqa: S101


@pytest.mark.asyncio
async def test_check_plain_text_uses_default_context(tmp_path) -> None:
    """check_plain_text uses default fallback context when ctx is None."""
    registry = build_registry()
    bundle = ConfigBundle(
        guardrails=[GuardrailConfig(name="no ctx len", config={"threshold": 5})],
        version=1,
    )

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    path = cfg_dir / "bundle.json"
    path.write_text(bundle.model_dump_json())

    # This should not raise even with ctx=None
    results = await check_plain_text("short", path, registry=registry, ctx=None)
    assert len(results) == 1  # noqa: S101
    assert isinstance(results[0], GuardrailResult)  # noqa: S101
