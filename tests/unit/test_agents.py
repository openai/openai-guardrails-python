"""Tests covering guardrails.agents helper functions."""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from guardrails.types import GuardrailResult

# ---------------------------------------------------------------------------
# Stub agents SDK module so guardrails.agents can import required symbols.
# ---------------------------------------------------------------------------

agents_module = types.ModuleType("agents")


@dataclass
class ToolContext:
    """Stub tool context carrying name, arguments, and optional call id."""

    tool_name: str
    tool_arguments: dict[str, Any] | str
    tool_call_id: str | None = None


@dataclass
class ToolInputGuardrailData:
    """Stub input guardrail payload."""

    context: ToolContext


@dataclass
class ToolOutputGuardrailData:
    """Stub output guardrail payload."""

    context: ToolContext
    output: Any


@dataclass
class GuardrailFunctionOutput:
    """Minimal guardrail function output stub."""

    output_info: Any
    tripwire_triggered: bool


@dataclass
class ToolGuardrailFunctionOutput:
    """Stub for tool guardrail responses."""

    message: str | None = None
    output_info: Any | None = None
    tripwire_triggered: bool = False

    @classmethod
    def raise_exception(cls, output_info: Any) -> ToolGuardrailFunctionOutput:
        """Return a response indicating an exception should be raised."""
        return cls(message="raise", output_info=output_info, tripwire_triggered=True)

    @classmethod
    def reject_content(
        cls,
        message: str,
        output_info: Any,
    ) -> ToolGuardrailFunctionOutput:
        """Return a response rejecting tool content."""
        return cls(message=message, output_info=output_info, tripwire_triggered=True)


def _decorator_passthrough(func: Callable) -> Callable:
    """Return the function unchanged (stand-in for agents decorators)."""
    return func


class RunContextWrapper:
    """Minimal RunContextWrapper stub."""

    def __init__(self, value: Any | None = None) -> None:
        """Store wrapped value."""
        self.value = value


@dataclass
class Agent:
    """Trivial Agent stub storing initialization args for assertions."""

    name: str
    instructions: str
    input_guardrails: list[Callable] | None = None
    output_guardrails: list[Callable] | None = None
    tools: list[Any] | None = None


class AgentRunner:
    """Minimal AgentRunner stub so guardrails patching succeeds."""

    async def run(self, *args: Any, **kwargs: Any) -> Any:
        """Return a sentinel result."""
        return SimpleNamespace()


agents_module.ToolGuardrailFunctionOutput = ToolGuardrailFunctionOutput
agents_module.ToolInputGuardrailData = ToolInputGuardrailData
agents_module.ToolOutputGuardrailData = ToolOutputGuardrailData
agents_module.tool_input_guardrail = _decorator_passthrough
agents_module.tool_output_guardrail = _decorator_passthrough
agents_module.RunContextWrapper = RunContextWrapper
agents_module.Agent = Agent
agents_module.GuardrailFunctionOutput = GuardrailFunctionOutput
agents_module.input_guardrail = _decorator_passthrough
agents_module.output_guardrail = _decorator_passthrough
agents_module.AgentRunner = AgentRunner

sys.modules.setdefault("agents", agents_module)

agents_run_module = types.ModuleType("agents.run")
agents_run_module.AgentRunner = AgentRunner
sys.modules.setdefault("agents.run", agents_run_module)
agents_module.run = agents_run_module

import guardrails.agents as agents  # noqa: E402  (import after stubbing)
import guardrails.runtime as runtime_module  # noqa: E402


def _make_guardrail(name: str) -> Any:
    class _DummyCtxModel:
        model_fields: dict[str, Any] = {}

        @staticmethod
        def model_validate(value: Any, **_: Any) -> Any:
            return value

    return SimpleNamespace(
        definition=SimpleNamespace(
            name=name,
            media_type="text/plain",
            ctx_requirements=_DummyCtxModel,
        ),
        ctx_requirements=[],
    )


@pytest.fixture(autouse=True)
def reset_agent_context() -> None:
    """Ensure agent conversation context vars are reset for each test."""
    agents._agent_session.set(None)
    agents._agent_conversation.set(None)


@pytest.mark.asyncio
async def test_conversation_with_tool_call_updates_fallback_history() -> None:
    """Fallback conversation should include previous history and new tool call."""
    agents._agent_session.set(None)
    agents._agent_conversation.set(({"role": "user", "content": "Hi there"},))
    data = SimpleNamespace(context=ToolContext(tool_name="math", tool_arguments={"x": 1}, tool_call_id="call-1"))

    conversation = await agents._conversation_with_tool_call(data)

    assert conversation[0]["content"] == "Hi there"  # noqa: S101
    assert conversation[-1]["type"] == "function_call"  # noqa: S101
    assert conversation[-1]["tool_name"] == "math"  # noqa: S101
    stored = agents._agent_conversation.get()
    assert stored is not None and stored[-1]["call_id"] == "call-1"  # type: ignore[index]  # noqa: S101


@pytest.mark.asyncio
async def test_conversation_with_tool_call_uses_session_history() -> None:
    """When session is available, its items form the conversation baseline."""

    class StubSession:
        def __init__(self) -> None:
            self.items = [{"role": "user", "content": "Remember me"}]

        async def get_items(self, limit: int | None = None) -> list[dict[str, Any]]:
            return self.items

        async def add_items(self, items: list[Any]) -> None:
            self.items.extend(items)

        async def pop_item(self) -> Any | None:
            return None

        async def clear_session(self) -> None:
            self.items.clear()

    session = StubSession()
    agents._agent_session.set(session)
    agents._agent_conversation.set(None)

    data = SimpleNamespace(context=ToolContext(tool_name="lookup", tool_arguments={"zip": 12345}, tool_call_id="call-2"))

    conversation = await agents._conversation_with_tool_call(data)

    assert conversation[0]["content"] == "Remember me"  # noqa: S101
    assert conversation[-1]["call_id"] == "call-2"  # noqa: S101
    cached = agents._agent_conversation.get()
    assert cached is not None and cached[-1]["call_id"] == "call-2"  # type: ignore[index]  # noqa: S101


@pytest.mark.asyncio
async def test_conversation_with_tool_output_includes_output() -> None:
    """Tool output conversation should include serialized output payload."""
    agents._agent_session.set(None)
    agents._agent_conversation.set(({"role": "user", "content": "Compute"},))
    data = SimpleNamespace(
        context=ToolContext(tool_name="calc", tool_arguments={"y": 2}, tool_call_id="call-3"),
        output={"result": 4},
    )

    conversation = await agents._conversation_with_tool_output(data)

    assert conversation[-1]["output"] == "{'result': 4}"  # noqa: S101


def test_create_conversation_context_exposes_history() -> None:
    """Conversation context should expose conversation history only."""
    base_context = SimpleNamespace(guardrail_llm="client")
    context = agents._create_conversation_context(["msg"], base_context)

    assert context.get_conversation_history() == ["msg"]  # noqa: S101
    assert not hasattr(context, "update_injection_last_checked_index")  # noqa: S101


def test_create_default_tool_context_provides_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Default tool context should return AsyncOpenAI client."""
    openai_mod = types.ModuleType("openai")

    class StubAsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            pass

    openai_mod.AsyncOpenAI = StubAsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_mod)

    context = agents._create_default_tool_context()

    assert isinstance(context.guardrail_llm, StubAsyncOpenAI)  # noqa: S101


def test_attach_guardrail_to_tools_initializes_lists() -> None:
    """Attaching guardrails should create input/output lists when missing."""
    tool = SimpleNamespace()
    fn = lambda data: data  # noqa: E731

    agents._attach_guardrail_to_tools([tool], fn, "input")
    agents._attach_guardrail_to_tools([tool], fn, "output")

    assert tool.tool_input_guardrails == [fn]  # type: ignore[attr-defined]  # noqa: S101
    assert tool.tool_output_guardrails == [fn]  # type: ignore[attr-defined]  # noqa: S101


def test_separate_tool_level_from_agent_level() -> None:
    """Prompt injection guardrails should be classified as tool-level."""
    tool, agent_level = agents._separate_tool_level_from_agent_level([_make_guardrail("Prompt Injection Detection"), _make_guardrail("Other Guard")])

    assert [g.definition.name for g in tool] == ["Prompt Injection Detection"]  # noqa: S101
    assert [g.definition.name for g in agent_level] == ["Other Guard"]  # noqa: S101


@pytest.mark.asyncio
async def test_create_tool_guardrail_rejects_on_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool guardrail should reject content when run_guardrails flags a violation."""
    guardrail = _make_guardrail("Test Guardrail")
    expected_info = {"observation": "violation"}
    agents._agent_session.set(None)
    agents._agent_conversation.set(({"role": "user", "content": "Original request"},))

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        assert kwargs["stage_name"] == "tool_input_test_guardrail"  # noqa: S101
        history = kwargs["ctx"].get_conversation_history()
        assert history[-1]["tool_name"] == "weather"  # noqa: S101
        return [GuardrailResult(tripwire_triggered=True, info=expected_info)]

    monkeypatch.setattr(runtime_module, "run_guardrails", fake_run_guardrails)

    tool_fn = agents._create_tool_guardrail(
        guardrail=guardrail,
        guardrail_type="input",
        context=SimpleNamespace(guardrail_llm="client"),
        raise_guardrail_errors=False,
        block_on_violations=False,
    )

    data = agents_module.ToolInputGuardrailData(context=ToolContext(tool_name="weather", tool_arguments={"city": "Paris"}))
    result = await tool_fn(data)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.output_info == expected_info  # noqa: S101
    assert "blocked by Test Guardrail" in result.message  # noqa: S101


@pytest.mark.asyncio
async def test_create_tool_guardrail_blocks_on_violation(monkeypatch: pytest.MonkeyPatch) -> None:
    """When block_on_violations is True, the guardrail should raise an exception output."""
    guardrail = _make_guardrail("Test Guardrail")

    async def fake_run_guardrails(**_: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=True, info={})]

    monkeypatch.setattr(runtime_module, "run_guardrails", fake_run_guardrails)
    agents._agent_session.set(None)
    agents._agent_conversation.set(({"role": "user", "content": "Hi"},))

    tool_fn = agents._create_tool_guardrail(
        guardrail=guardrail,
        guardrail_type="input",
        context=SimpleNamespace(guardrail_llm="client"),
        raise_guardrail_errors=False,
        block_on_violations=True,
    )

    data = agents_module.ToolInputGuardrailData(context=ToolContext(tool_name="weather", tool_arguments={}))
    result = await tool_fn(data)

    assert result.message == "raise"  # noqa: S101


@pytest.mark.asyncio
async def test_create_tool_guardrail_propagates_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guardrail errors should raise when raise_guardrail_errors is True."""
    guardrail = _make_guardrail("Failing Guardrail")

    async def failing_run_guardrails(**_: Any) -> list[GuardrailResult]:
        raise RuntimeError("guardrail failure")

    monkeypatch.setattr(runtime_module, "run_guardrails", failing_run_guardrails)
    agents._agent_session.set(None)
    agents._agent_conversation.set(({"role": "user", "content": "Hi"},))

    tool_fn = agents._create_tool_guardrail(
        guardrail=guardrail,
        guardrail_type="input",
        context=SimpleNamespace(guardrail_llm="client"),
        raise_guardrail_errors=True,
        block_on_violations=False,
    )

    data = agents_module.ToolInputGuardrailData(context=ToolContext(tool_name="weather", tool_arguments={}))
    result = await tool_fn(data)

    assert result.message == "raise"  # noqa: S101


@pytest.mark.asyncio
async def test_create_tool_guardrail_handles_empty_conversation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guardrail executes even when no prior conversation is present."""
    guardrail = _make_guardrail("Prompt Injection Detection")

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        history = kwargs["ctx"].get_conversation_history()
        assert history[-1]["output"] == "ok"  # noqa: S101
        return [GuardrailResult(tripwire_triggered=False, info={})]

    monkeypatch.setattr(runtime_module, "run_guardrails", fake_run_guardrails)
    agents._agent_session.set(None)
    agents._agent_conversation.set(None)

    tool_fn = agents._create_tool_guardrail(
        guardrail=guardrail,
        guardrail_type="output",
        context=SimpleNamespace(guardrail_llm="client"),
        raise_guardrail_errors=False,
        block_on_violations=False,
    )

    data = agents_module.ToolOutputGuardrailData(
        context=ToolContext(tool_name="math", tool_arguments={"value": 1}),
        output="ok",
    )
    result = await tool_fn(data)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_create_agents_guardrails_from_config_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent-level guardrail functions should execute run_guardrails."""
    pipeline = SimpleNamespace(pre_flight=None, input=SimpleNamespace(), output=None)
    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline)
    monkeypatch.setattr(
        runtime_module,
        "instantiate_guardrails",
        lambda stage, registry=None: [_make_guardrail("Input Guard")] if stage is pipeline.input else [],
    )

    captured: dict[str, Any] = {}

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        captured.update(kwargs)
        return [GuardrailResult(tripwire_triggered=False, info={})]

    monkeypatch.setattr(runtime_module, "run_guardrails", fake_run_guardrails)

    guardrails = agents._create_agents_guardrails_from_config(
        config={},
        stages=["input"],
        guardrail_type="input",
        context=None,
        raise_guardrail_errors=False,
    )

    assert len(guardrails) == 1  # noqa: S101
    output = await guardrails[0](agents_module.RunContextWrapper(None), Agent("a", "b"), "hello")

    assert output.tripwire_triggered is False  # noqa: S101
    assert captured["stage_name"] == "input"  # noqa: S101
    assert captured["data"] == "hello"  # noqa: S101


@pytest.mark.asyncio
async def test_create_agents_guardrails_from_config_tripwire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tripwire results should propagate to guardrail function output."""
    pipeline = SimpleNamespace(pre_flight=None, input=SimpleNamespace(), output=None)
    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline)
    monkeypatch.setattr(
        runtime_module,
        "instantiate_guardrails",
        lambda stage, registry=None: [_make_guardrail("Input Guard")] if stage is pipeline.input else [],
    )

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=True, info={"reason": "blocked"})]

    monkeypatch.setattr(runtime_module, "run_guardrails", fake_run_guardrails)

    guardrails = agents._create_agents_guardrails_from_config(
        config={},
        stages=["input"],
        guardrail_type="input",
        context=SimpleNamespace(guardrail_llm="llm"),
        raise_guardrail_errors=False,
    )

    result = await guardrails[0](agents_module.RunContextWrapper(None), Agent("a", "b"), "hi")

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.output_info == "Guardrail unknown triggered tripwire"  # noqa: S101


@pytest.mark.asyncio
async def test_create_agents_guardrails_from_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors should be converted to tripwire when raise_guardrail_errors=False."""
    pipeline = SimpleNamespace(pre_flight=None, input=SimpleNamespace(), output=None)
    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline)
    monkeypatch.setattr(
        runtime_module,
        "instantiate_guardrails",
        lambda stage, registry=None: [_make_guardrail("Input Guard")] if stage is pipeline.input else [],
    )

    async def failing_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        raise RuntimeError("boom")

    monkeypatch.setattr(runtime_module, "run_guardrails", failing_run_guardrails)

    guardrails = agents._create_agents_guardrails_from_config(
        config={},
        stages=["input"],
        guardrail_type="input",
        context=SimpleNamespace(guardrail_llm="llm"),
        raise_guardrail_errors=False,
    )

    result = await guardrails[0](agents_module.RunContextWrapper(None), Agent("name", "instr"), "msg")

    assert result.tripwire_triggered is True  # noqa: S101
    assert "Error running input guardrails" in result.output_info  # noqa: S101


@pytest.mark.asyncio
async def test_create_agents_guardrails_from_config_error_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Errors should bubble when raise_guardrail_errors=True."""
    pipeline = SimpleNamespace(pre_flight=None, input=SimpleNamespace(), output=None)
    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline)
    monkeypatch.setattr(
        runtime_module,
        "instantiate_guardrails",
        lambda stage, registry=None: [_make_guardrail("Input Guard")] if stage is pipeline.input else [],
    )

    async def failing_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        raise RuntimeError("failure")

    monkeypatch.setattr(runtime_module, "run_guardrails", failing_run_guardrails)

    guardrails = agents._create_agents_guardrails_from_config(
        config={},
        stages=["input"],
        guardrail_type="input",
        context=SimpleNamespace(guardrail_llm="llm"),
        raise_guardrail_errors=True,
    )

    with pytest.raises(RuntimeError):
        await guardrails[0](agents_module.RunContextWrapper(None), Agent("n", "i"), "msg")


@pytest.mark.asyncio
async def test_create_agents_guardrails_from_config_output_stage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Output guardrails should not capture user messages."""
    pipeline = SimpleNamespace(pre_flight=None, input=None, output=SimpleNamespace())
    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline)
    monkeypatch.setattr(
        runtime_module,
        "instantiate_guardrails",
        lambda stage, registry=None: [_make_guardrail("Output Guard")] if stage is pipeline.output else [],
    )

    async def fake_run_guardrails(**kwargs: Any) -> list[GuardrailResult]:
        return [GuardrailResult(tripwire_triggered=False, info={})]

    monkeypatch.setattr(runtime_module, "run_guardrails", fake_run_guardrails)

    guardrails = agents._create_agents_guardrails_from_config(
        config={},
        stages=["output"],
        guardrail_type="output",
        context=SimpleNamespace(guardrail_llm="llm"),
        raise_guardrail_errors=False,
    )

    result = await guardrails[0](agents_module.RunContextWrapper(None), Agent("n", "i"), "response")

    assert result.tripwire_triggered is False  # noqa: S101


def test_guardrail_agent_attaches_tool_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """GuardrailAgent should attach tool-level guardrails and return an Agent."""
    tool_guard = _make_guardrail("Prompt Injection Detection")
    agent_guard = _make_guardrail("Sensitive Data Check")

    class FakePipeline:
        def __init__(self) -> None:
            self.pre_flight = SimpleNamespace()
            self.input = SimpleNamespace()
            self.output = SimpleNamespace()

        def stages(self) -> list[Any]:
            return [self.pre_flight, self.input, self.output]

    pipeline = FakePipeline()

    def fake_load_pipeline_bundles(config: Any) -> FakePipeline:
        assert config == {"version": 1}  # noqa: S101
        return pipeline

    def fake_instantiate_guardrails(stage: Any, registry: Any | None = None) -> list[Any]:
        if stage is pipeline.pre_flight:
            return [tool_guard]
        if stage is pipeline.input:
            return [agent_guard]
        if stage is pipeline.output:
            return []
        return []

    from guardrails import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", fake_load_pipeline_bundles)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", fake_instantiate_guardrails)
    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", fake_load_pipeline_bundles, raising=False)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", fake_instantiate_guardrails, raising=False)

    tool = SimpleNamespace()
    agent_instance = agents.GuardrailAgent(
        config={"version": 1},
        name="Test Agent",
        instructions="Help users.",
        tools=[tool],
    )

    assert isinstance(agent_instance, agents_module.Agent)  # noqa: S101
    assert len(tool.tool_input_guardrails) == 1  # type: ignore[attr-defined]  # noqa: S101
    # Agent-level guardrails should be attached (one for Sensitive Data Check)
    assert len(agent_instance.input_guardrails or []) >= 1  # noqa: S101


def test_guardrail_agent_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agent with no tools should not attach tool guardrails."""
    pipeline = SimpleNamespace(pre_flight=None, input=None, output=None)

    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline, raising=False)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", lambda *args, **kwargs: [], raising=False)

    agent_instance = agents.GuardrailAgent(config={}, name="NoTools", instructions="None")

    assert getattr(agent_instance, "input_guardrails", []) == []  # noqa: S101


def test_guardrail_agent_without_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    """GuardrailAgent should work without instructions parameter."""
    pipeline = SimpleNamespace(pre_flight=None, input=None, output=None)

    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline, raising=False)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", lambda *args, **kwargs: [], raising=False)

    # Should not raise TypeError about missing instructions
    agent_instance = agents.GuardrailAgent(config={}, name="NoInstructions")

    assert isinstance(agent_instance, agents_module.Agent)  # noqa: S101
    assert agent_instance.instructions is None  # noqa: S101


def test_guardrail_agent_with_callable_instructions(monkeypatch: pytest.MonkeyPatch) -> None:
    """GuardrailAgent should accept callable instructions."""
    pipeline = SimpleNamespace(pre_flight=None, input=None, output=None)

    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline, raising=False)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", lambda *args, **kwargs: [], raising=False)

    def dynamic_instructions(ctx: Any, agent: Any) -> str:
        return f"You are {agent.name}"

    agent_instance = agents.GuardrailAgent(
        config={},
        name="DynamicAgent",
        instructions=dynamic_instructions,
    )

    assert isinstance(agent_instance, agents_module.Agent)  # noqa: S101
    assert callable(agent_instance.instructions)  # noqa: S101
    assert agent_instance.instructions == dynamic_instructions  # noqa: S101


def test_guardrail_agent_merges_user_input_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """User input guardrails should be merged with config guardrails."""
    agent_guard = _make_guardrail("Config Input Guard")

    class FakePipeline:
        def __init__(self) -> None:
            self.pre_flight = None
            self.input = SimpleNamespace()
            self.output = None

    pipeline = FakePipeline()

    def fake_load_pipeline_bundles(config: Any) -> FakePipeline:
        return pipeline

    def fake_instantiate_guardrails(stage: Any, registry: Any | None = None) -> list[Any]:
        if stage is pipeline.input:
            return [agent_guard]
        return []

    from guardrails import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", fake_load_pipeline_bundles)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", fake_instantiate_guardrails)

    # Create a custom user guardrail
    custom_guardrail = lambda ctx, agent, input: None  # noqa: E731

    agent_instance = agents.GuardrailAgent(
        config={},
        name="MergedAgent",
        instructions="Test",
        input_guardrails=[custom_guardrail],
    )

    # Should have both config and user guardrails merged
    assert isinstance(agent_instance, agents_module.Agent)  # noqa: S101
    assert len(agent_instance.input_guardrails) == 2  # noqa: S101
    # Config guardrail from _create_agents_guardrails_from_config, then user guardrail


def test_guardrail_agent_merges_user_output_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """User output guardrails should be merged with config guardrails."""
    agent_guard = _make_guardrail("Config Output Guard")

    class FakePipeline:
        def __init__(self) -> None:
            self.pre_flight = None
            self.input = None
            self.output = SimpleNamespace()

    pipeline = FakePipeline()

    def fake_load_pipeline_bundles(config: Any) -> FakePipeline:
        return pipeline

    def fake_instantiate_guardrails(stage: Any, registry: Any | None = None) -> list[Any]:
        if stage is pipeline.output:
            return [agent_guard]
        return []

    from guardrails import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", fake_load_pipeline_bundles)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", fake_instantiate_guardrails)

    # Create a custom user guardrail
    custom_guardrail = lambda ctx, agent, output: None  # noqa: E731

    agent_instance = agents.GuardrailAgent(
        config={},
        name="MergedAgent",
        instructions="Test",
        output_guardrails=[custom_guardrail],
    )

    # Should have both config and user guardrails merged
    assert isinstance(agent_instance, agents_module.Agent)  # noqa: S101
    assert len(agent_instance.output_guardrails) == 2  # noqa: S101
    # Config guardrail from _create_agents_guardrails_from_config, then user guardrail


def test_guardrail_agent_with_empty_user_guardrails(monkeypatch: pytest.MonkeyPatch) -> None:
    """GuardrailAgent should handle empty user guardrail lists gracefully."""
    pipeline = SimpleNamespace(pre_flight=None, input=None, output=None)

    monkeypatch.setattr(runtime_module, "load_pipeline_bundles", lambda config: pipeline, raising=False)
    monkeypatch.setattr(runtime_module, "instantiate_guardrails", lambda *args, **kwargs: [], raising=False)

    agent_instance = agents.GuardrailAgent(
        config={},
        name="EmptyListAgent",
        instructions="Test",
        input_guardrails=[],
        output_guardrails=[],
    )

    assert isinstance(agent_instance, agents_module.Agent)  # noqa: S101
    assert agent_instance.input_guardrails == []  # noqa: S101
    assert agent_instance.output_guardrails == []  # noqa: S101
