"""Unit tests for types module."""

import sys
import types
from collections.abc import Iterator
from dataclasses import FrozenInstanceError

import pytest


@pytest.fixture(autouse=True)
def stub_openai_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    """Provide a stub ``openai.AsyncOpenAI`` and patch guardrails types symbol.

    Ensures tests don't require real OPENAI_API_KEY or networked clients.
    """
    module = types.ModuleType("openai")

    class AsyncOpenAI:  # noqa: D401 - simple stub
        """Stubbed AsyncOpenAI client."""

        def __init__(self, **_: object) -> None:
            pass

    module.__dict__["AsyncOpenAI"] = AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    # Patch already-imported symbol in guardrails.types if present
    try:
        import guardrails.types as gr_types  # type: ignore

        monkeypatch.setattr(gr_types, "AsyncOpenAI", AsyncOpenAI, raising=False)
    except Exception:
        pass
    # Provide dummy API key in case any code inspects env
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    yield module
    monkeypatch.delitem(sys.modules, "openai", raising=False)


def test_guardrail_result_is_frozen() -> None:
    """Attempting to mutate fields should raise ``FrozenInstanceError``."""
    from guardrails.types import GuardrailResult

    result = GuardrailResult(tripwire_triggered=True)
    with pytest.raises(FrozenInstanceError):
        result.tripwire_triggered = False  # type: ignore[assignment]


def test_guardrail_result_default_info_is_unique() -> None:
    """Instances should not share mutable ``info`` dicts."""
    from guardrails.types import GuardrailResult

    first = GuardrailResult(tripwire_triggered=False)
    second = GuardrailResult(tripwire_triggered=True)

    assert first.info == {}
    assert second.info == {}
    assert first.info is not second.info


def test_check_fn_typing_roundtrip() -> None:
    """A callable conforming to ``CheckFn`` returns a ``GuardrailResult``."""
    from pydantic import BaseModel

    from guardrails.types import CheckFn, GuardrailResult

    class Cfg(BaseModel):
        pass

    def check(_ctx: object, value: str, _cfg: Cfg) -> GuardrailResult:
        _, _ = _ctx, _cfg
        return GuardrailResult(tripwire_triggered=value == "fail")

    fn: CheckFn[object, str, Cfg] = check
    assert fn(None, "fail", Cfg()).tripwire_triggered
    assert not fn(None, "ok", Cfg()).tripwire_triggered


def test_guardrail_llm_context_proto_usage() -> None:
    """Objects with ``guardrail_llm`` attribute satisfy the protocol."""
    from guardrails.types import AsyncOpenAI, GuardrailLLMContextProto

    class DummyLLM(AsyncOpenAI):
        pass

    class DummyCtx:
        guardrail_llm: AsyncOpenAI

        def __init__(self) -> None:
            self.guardrail_llm = DummyLLM()

    def use(ctx: GuardrailLLMContextProto) -> object:
        return ctx.guardrail_llm

    assert isinstance(use(DummyCtx()), DummyLLM)
