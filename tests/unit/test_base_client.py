"""Unit tests covering core GuardrailsBaseClient helper methods."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import guardrails.context as guardrails_context
from guardrails._base_client import GuardrailResults, GuardrailsBaseClient, GuardrailsResponse
from guardrails.context import GuardrailsContext
from guardrails.types import GuardrailResult


def test_extract_latest_user_message_dicts() -> None:
    """Ensure latest user message and index are returned for dict inputs."""
    client = GuardrailsBaseClient()
    messages = [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "  hi there  "},
    ]

    text, index = client._extract_latest_user_message(messages)

    assert text == "hi there"  # noqa: S101
    assert index == 1  # noqa: S101


def test_extract_latest_user_message_content_parts() -> None:
    """Support Responses API content part lists."""
    client = GuardrailsBaseClient()
    messages = [
        {"role": "assistant", "content": "prev"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "first"},
                {"type": "summary_text", "text": "second"},
            ],
        },
    ]

    text, index = client._extract_latest_user_message(messages)

    assert text == "first second"  # noqa: S101
    assert index == 1  # noqa: S101


def test_extract_latest_user_message_missing_user() -> None:
    """Return empty payload when no user role is present."""
    client = GuardrailsBaseClient()

    text, index = client._extract_latest_user_message([{"role": "assistant", "content": "x"}])

    assert text == ""  # noqa: S101
    assert index == -1  # noqa: S101


def test_apply_preflight_modifications_masks_user_message() -> None:
    """Mask PII tokens for the most recent user message."""
    client = GuardrailsBaseClient()
    guardrail_results = [
        GuardrailResult(
            tripwire_triggered=False,
            info={"detected_entities": {"PERSON": ["Alice Smith"]}},
        )
    ]
    messages = [
        {"role": "user", "content": "My name is Alice Smith."},
        {"role": "assistant", "content": "Hi Alice."},
    ]

    modified = client._apply_preflight_modifications(messages, guardrail_results)

    assert modified[0]["content"] == "My name is <PERSON>."  # noqa: S101
    assert messages[0]["content"] == "My name is Alice Smith."  # noqa: S101


def test_apply_preflight_modifications_handles_strings() -> None:
    """Apply masking for string payloads."""
    client = GuardrailsBaseClient()
    guardrail_results = [
        GuardrailResult(
            tripwire_triggered=False,
            info={"detected_entities": {"PHONE": ["+1-555-0100"]}},
        )
    ]

    masked = client._apply_preflight_modifications("+1-555-0100", guardrail_results)

    assert masked == "<PHONE>"  # noqa: S101


def test_apply_preflight_modifications_skips_when_no_entities() -> None:
    """Return original data when no guardrail metadata exists."""
    client = GuardrailsBaseClient()
    messages = [{"role": "user", "content": "Nothing to mask"}]
    guardrail_results = [GuardrailResult(tripwire_triggered=False)]

    modified = client._apply_preflight_modifications(messages, guardrail_results)

    assert modified is messages  # noqa: S101


def test_apply_preflight_modifications_structured_content() -> None:
    """Structured content parts should be masked individually."""
    client = GuardrailsBaseClient()
    guardrail_results = [
        GuardrailResult(
            tripwire_triggered=False,
            info={"detected_entities": {"PHONE": ["123-456"]}},
        )
    ]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Call 123-456"},
                {"type": "json", "value": {"raw": "no change"}},
            ],
        }
    ]

    modified = client._apply_preflight_modifications(messages, guardrail_results)

    assert modified[0]["content"][0]["text"] == "Call <PHONE>"  # noqa: S101
    assert modified[0]["content"][1]["value"] == {"raw": "no change"}  # noqa: S101


def test_apply_preflight_modifications_object_message_handles_failure() -> None:
    """If object content cannot be updated, original data should be returned."""
    client = GuardrailsBaseClient()
    guardrail_results = [
        GuardrailResult(
            tripwire_triggered=False,
            info={"detected_entities": {"NAME": ["Alice"]}},
        )
    ]

    class Message:
        def __init__(self) -> None:
            self.role = "user"
            self.content = "Alice"

        def __setattr__(self, key: str, value: Any) -> None:
            if key == "content" and hasattr(self, key):
                raise RuntimeError("cannot set")
            super().__setattr__(key, value)

    msg = Message()
    messages = [msg]

    modified = client._apply_preflight_modifications(messages, guardrail_results)

    assert modified is messages  # noqa: S101


def test_apply_preflight_modifications_no_user_message() -> None:
    """When no user message exists, data should be returned unchanged."""
    client = GuardrailsBaseClient()
    guardrail_results = [GuardrailResult(tripwire_triggered=False, info={"detected_entities": {"NAME": ["Alice"]}})]
    messages = [{"role": "assistant", "content": "hi"}]

    modified = client._apply_preflight_modifications(messages, guardrail_results)

    assert modified is messages  # noqa: S101


def test_apply_preflight_modifications_non_dict_part_preserved() -> None:
    """Non-dict content parts should be preserved as-is."""
    client = GuardrailsBaseClient()
    guardrail_results = [GuardrailResult(tripwire_triggered=False, info={"detected_entities": {"NAME": ["Alice"]}})]
    messages = [
        {
            "role": "user",
            "content": ["raw text"],
        }
    ]

    modified = client._apply_preflight_modifications(messages, guardrail_results)

    assert modified[0]["content"][0] == "raw text"  # noqa: S101


def test_create_guardrails_response_wraps_results() -> None:
    """Combine guardrail results by stage for response."""
    client = GuardrailsBaseClient()
    preflight = [GuardrailResult(tripwire_triggered=True)]
    input_stage = [GuardrailResult(tripwire_triggered=False)]
    output_stage = [GuardrailResult(tripwire_triggered=True)]

    response = client._create_guardrails_response(
        llm_response=SimpleNamespace(choices=[]),
        preflight_results=preflight,
        input_results=input_stage,
        output_results=output_stage,
    )

    assert isinstance(response, GuardrailsResponse)  # noqa: S101
    assert response.guardrail_results.tripwires_triggered is True  # noqa: S101
    assert len(response.guardrail_results.all_results) == 3  # noqa: S101


def test_extract_response_text_prefers_choice_message() -> None:
    """Extract message content from chat-style responses."""
    client = GuardrailsBaseClient()
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="hello"),
                delta=SimpleNamespace(content=None),
            )
        ],
        output_text=None,
        delta=None,
    )

    text = client._extract_response_text(response)

    assert text == "hello"  # noqa: S101


def test_extract_response_text_handles_delta_type() -> None:
    """Special delta responses should return delta text."""
    client = GuardrailsBaseClient()
    response = SimpleNamespace(type="response.output_text.delta", delta="partial")

    assert client._extract_response_text(response) == "partial"  # noqa: S101


class _DummyResourceClient:
    """Stub OpenAI resource client used during initialization tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _TestableClient(GuardrailsBaseClient):
    """Concrete subclass exposing _initialize_client for testing."""

    def __init__(self) -> None:
        self.override_called = False

    def _instantiate_all_guardrails(self) -> dict[str, list]:
        return {"pre_flight": [], "input": [], "output": []}

    def _create_default_context(self) -> SimpleNamespace:
        return SimpleNamespace(guardrail_llm="stub")

    def _override_resources(self) -> None:
        self.override_called = True


def test_initialize_client_sets_pipeline_and_context() -> None:
    """Ensure _initialize_client produces pipeline, guardrails, and context."""
    client = _TestableClient()

    client._initialize_client(
        config={"version": 1, "output": {"version": 1, "guardrails": []}},
        openai_kwargs={"api_key": "abc"},
        client_class=_DummyResourceClient,
    )

    assert client.pipeline.pre_flight is None  # type: ignore[attr-defined]  # noqa: S101
    assert client.pipeline.output.guardrails == []  # type: ignore[attr-defined]  # noqa: S101
    assert client.guardrails == {"pre_flight": [], "input": [], "output": []}  # noqa: S101
    assert client.context.guardrail_llm == "stub"  # type: ignore[attr-defined]  # noqa: S101
    assert client._resource_client.kwargs["api_key"] == "abc"  # type: ignore[attr-defined]  # noqa: S101
    assert client.override_called is True  # noqa: S101


def test_instantiate_all_guardrails_uses_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """_instantiate_all_guardrails should instantiate guardrails for each stage."""
    client = GuardrailsBaseClient()
    client.pipeline = SimpleNamespace(
        pre_flight=SimpleNamespace(),
        input=None,
        output=SimpleNamespace(),
    )

    instantiated: list[str] = []

    def fake_instantiate(stage: Any, registry: Any) -> list[str]:
        instantiated.append(str(stage))
        return ["g"]

    monkeypatch.setattr("guardrails.runtime.instantiate_guardrails", fake_instantiate)

    guardrails = client._instantiate_all_guardrails()

    assert guardrails["pre_flight"] == ["g"]  # noqa: S101
    assert guardrails["input"] == []  # noqa: S101
    assert guardrails["output"] == ["g"]  # noqa: S101
    assert len(instantiated) == 2  # noqa: S101


def test_validate_context_invokes_validator(monkeypatch: pytest.MonkeyPatch) -> None:
    """_validate_context should call validate_guardrail_context for each guardrail."""
    client = GuardrailsBaseClient()
    guardrail = SimpleNamespace()
    client.guardrails = {"pre_flight": [guardrail]}

    called: list[Any] = []

    def fake_validate(gr: Any, ctx: Any) -> None:
        called.append((gr, ctx))

    monkeypatch.setattr("guardrails._base_client.validate_guardrail_context", fake_validate)

    client._validate_context(context="ctx")

    assert called == [(guardrail, "ctx")]  # noqa: S101


def test_apply_preflight_modifications_leaves_unknown_content() -> None:
    """Unknown content types should remain untouched."""
    client = GuardrailsBaseClient()
    result = GuardrailResult(tripwire_triggered=False, info={"detected_entities": {"NAME": ["Alice"]}})
    messages = [{"role": "user", "content": {"unknown": "value"}}]

    modified = client._apply_preflight_modifications(messages, [result])

    assert modified is messages  # noqa: S101


def test_apply_preflight_modifications_non_string_text_retained() -> None:
    """Content parts without string text should remain unchanged."""
    client = GuardrailsBaseClient()
    result = GuardrailResult(tripwire_triggered=False, info={"detected_entities": {"PHONE": ["123"]}})
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": 123},
            ],
        }
    ]

    modified = client._apply_preflight_modifications(messages, [result])

    assert modified[0]["content"][0]["text"] == 123  # noqa: S101


def test_extract_latest_user_message_object_parts() -> None:
    """Object messages with attribute content should be handled."""
    client = GuardrailsBaseClient()

    class Msg:
        def __init__(self, role: str, content: Any) -> None:
            self.role = role
            self.content = content

    messages = [
        Msg("assistant", "ignored"),
        Msg("user", [SimpleNamespace(type="input_text", text="obj text")]),
    ]

    text, index = client._extract_latest_user_message(messages)

    assert text == "obj text"  # noqa: S101
    assert index == 1  # noqa: S101


def test_extract_response_text_fallback_returns_empty() -> None:
    """Unknown response types should return empty string."""
    client = GuardrailsBaseClient()
    response = SimpleNamespace(choices=[], output_text=None, delta=None)

    assert client._extract_response_text(response) == ""  # noqa: S101


def test_guardrail_results_properties() -> None:
    """GuardrailResults should aggregate and report tripwires."""
    results = GuardrailResults(
        preflight=[GuardrailResult(tripwire_triggered=False)],
        input=[GuardrailResult(tripwire_triggered=True)],
        output=[GuardrailResult(tripwire_triggered=False)],
    )

    assert len(results.all_results) == 3  # noqa: S101
    assert results.tripwires_triggered is True  # noqa: S101
    assert results.triggered_results == [results.input[0]]  # noqa: S101


def test_create_default_context_raises_without_subclass() -> None:
    """Base implementation should raise when no context available."""
    client = GuardrailsBaseClient()

    with pytest.raises(NotImplementedError):
        client._create_default_context()


def test_create_default_context_uses_existing_context() -> None:
    """Existing context var should be returned."""
    existing = GuardrailsContext(guardrail_llm="ctx")
    guardrails_context.set_context(existing)
    try:
        client = GuardrailsBaseClient()
        assert client._create_default_context() is existing  # noqa: S101
    finally:
        guardrails_context.clear_context()
