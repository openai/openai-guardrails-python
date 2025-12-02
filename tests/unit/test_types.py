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


# ----- TokenUsage Tests -----


def test_token_usage_is_frozen() -> None:
    """TokenUsage instances should be immutable."""
    from guardrails.types import TokenUsage

    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    with pytest.raises(FrozenInstanceError):
        usage.prompt_tokens = 20  # type: ignore[assignment]


def test_token_usage_with_all_values() -> None:
    """TokenUsage should store all token counts."""
    from guardrails.types import TokenUsage

    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150
    assert usage.unavailable_reason is None


def test_token_usage_with_unavailable_reason() -> None:
    """TokenUsage should include reason when tokens are unavailable."""
    from guardrails.types import TokenUsage

    usage = TokenUsage(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        unavailable_reason="Third-party model",
    )
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert usage.unavailable_reason == "Third-party model"


def test_extract_token_usage_with_valid_response() -> None:
    """extract_token_usage should extract tokens from response with usage."""
    from guardrails.types import extract_token_usage

    class MockUsage:
        prompt_tokens = 100
        completion_tokens = 50
        total_tokens = 150

    class MockResponse:
        usage = MockUsage()

    usage = extract_token_usage(MockResponse())
    assert usage.prompt_tokens == 100
    assert usage.completion_tokens == 50
    assert usage.total_tokens == 150
    assert usage.unavailable_reason is None


def test_extract_token_usage_with_no_usage() -> None:
    """extract_token_usage should return unavailable when no usage attribute."""
    from guardrails.types import extract_token_usage

    class MockResponse:
        pass

    usage = extract_token_usage(MockResponse())
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert usage.unavailable_reason == "Token usage not available for this model provider"


def test_extract_token_usage_with_none_usage() -> None:
    """extract_token_usage should handle usage=None."""
    from guardrails.types import extract_token_usage

    class MockResponse:
        usage = None

    usage = extract_token_usage(MockResponse())
    assert usage.prompt_tokens is None
    assert usage.unavailable_reason == "Token usage not available for this model provider"


def test_extract_token_usage_with_empty_usage_object() -> None:
    """extract_token_usage should handle usage object with all None values."""
    from guardrails.types import extract_token_usage

    class MockUsage:
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None

    class MockResponse:
        usage = MockUsage()

    usage = extract_token_usage(MockResponse())
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert usage.unavailable_reason == "Token usage data not populated in response"


def test_token_usage_to_dict_with_values() -> None:
    """token_usage_to_dict should convert to dict with values."""
    from guardrails.types import TokenUsage, token_usage_to_dict

    usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
    result = token_usage_to_dict(usage)

    assert result == {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }


def test_token_usage_to_dict_with_unavailable_reason() -> None:
    """token_usage_to_dict should include unavailable_reason when present."""
    from guardrails.types import TokenUsage, token_usage_to_dict

    usage = TokenUsage(
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        unavailable_reason="No data",
    )
    result = token_usage_to_dict(usage)

    assert result == {
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "unavailable_reason": "No data",
    }


def test_token_usage_to_dict_without_unavailable_reason() -> None:
    """token_usage_to_dict should not include unavailable_reason when None."""
    from guardrails.types import TokenUsage, token_usage_to_dict

    usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    result = token_usage_to_dict(usage)

    assert "unavailable_reason" not in result


# ----- total_guardrail_token_usage Tests -----


def test_total_guardrail_token_usage_with_guardrails_response() -> None:
    """total_guardrail_token_usage should work with GuardrailsResponse objects."""
    from guardrails.types import total_guardrail_token_usage

    class MockGuardrailResults:
        @property
        def total_token_usage(self) -> dict:
            return {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}

    class MockResponse:
        guardrail_results = MockGuardrailResults()

    result = total_guardrail_token_usage(MockResponse())

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50
    assert result["total_tokens"] == 150


def test_total_guardrail_token_usage_with_guardrail_results_directly() -> None:
    """total_guardrail_token_usage should work with GuardrailResults directly."""
    from guardrails._base_client import GuardrailResults
    from guardrails.types import GuardrailResult, total_guardrail_token_usage

    results = GuardrailResults(
        preflight=[
            GuardrailResult(
                tripwire_triggered=False,
                info={
                    "guardrail_name": "Jailbreak",
                    "token_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 50,
                        "total_tokens": 150,
                    },
                },
            )
        ],
        input=[],
        output=[],
    )

    result = total_guardrail_token_usage(results)

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50
    assert result["total_tokens"] == 150


def test_total_guardrail_token_usage_with_agents_sdk_result() -> None:
    """total_guardrail_token_usage should work with Agents SDK RunResult-like objects."""
    from guardrails.types import total_guardrail_token_usage

    class MockOutput:
        output_info = {
            "guardrail_name": "Jailbreak",
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        }

    class MockGuardrailResult:
        output = MockOutput()

    class MockRunResult:
        input_guardrail_results = [MockGuardrailResult()]
        output_guardrail_results = []
        tool_input_guardrail_results = []
        tool_output_guardrail_results = []

    result = total_guardrail_token_usage(MockRunResult())

    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50
    assert result["total_tokens"] == 150


def test_total_guardrail_token_usage_with_multiple_agents_stages() -> None:
    """total_guardrail_token_usage should aggregate across all Agents SDK stages."""
    from guardrails.types import total_guardrail_token_usage

    class MockOutput:
        def __init__(self, tokens: dict) -> None:
            self.output_info = {"token_usage": tokens}

    class MockGuardrailResult:
        def __init__(self, tokens: dict) -> None:
            self.output = MockOutput(tokens)

    class MockRunResult:
        input_guardrail_results = [MockGuardrailResult({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150})]
        output_guardrail_results = [MockGuardrailResult({"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275})]
        tool_input_guardrail_results = []
        tool_output_guardrail_results = []

    result = total_guardrail_token_usage(MockRunResult())

    assert result["prompt_tokens"] == 300
    assert result["completion_tokens"] == 125
    assert result["total_tokens"] == 425


def test_total_guardrail_token_usage_with_unknown_result_type() -> None:
    """total_guardrail_token_usage should return None values for unknown types."""
    from guardrails.types import total_guardrail_token_usage

    class UnknownResult:
        pass

    result = total_guardrail_token_usage(UnknownResult())

    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["total_tokens"] is None


def test_total_guardrail_token_usage_with_none_output_info() -> None:
    """total_guardrail_token_usage should handle None output_info gracefully."""
    from guardrails.types import total_guardrail_token_usage

    class MockOutput:
        output_info = None

    class MockGuardrailResult:
        output = MockOutput()

    class MockRunResult:
        input_guardrail_results = [MockGuardrailResult()]
        output_guardrail_results = []
        tool_input_guardrail_results = []
        tool_output_guardrail_results = []

    result = total_guardrail_token_usage(MockRunResult())

    assert result["prompt_tokens"] is None
    assert result["completion_tokens"] is None
    assert result["total_tokens"] is None
