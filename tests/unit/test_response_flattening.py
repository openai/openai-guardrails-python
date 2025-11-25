"""Tests for GuardrailsResponse flattening and deprecation warnings.

This module tests that GuardrailsResponse acts as a transparent proxy to the
underlying OpenAI response, allowing direct attribute access while maintaining
backward compatibility with deprecation warnings for llm_response access.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from guardrails._base_client import GuardrailResults, GuardrailsResponse
from guardrails.types import GuardrailResult


def _create_mock_chat_completion() -> Any:
    """Create a mock ChatCompletion response."""
    return SimpleNamespace(
        id="chatcmpl-123",
        choices=[
            SimpleNamespace(
                index=0,
                message=SimpleNamespace(content="Hello, world!", role="assistant"),
                finish_reason="stop",
            )
        ],
        model="gpt-4",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def _create_mock_response() -> Any:
    """Create a mock Response (Responses API) response."""
    return SimpleNamespace(
        id="resp-123",
        output_text="Hello from responses API!",
        conversation=SimpleNamespace(id="conv-123"),
    )


def _create_mock_guardrail_results() -> GuardrailResults:
    """Create mock guardrail results."""
    return GuardrailResults(
        preflight=[GuardrailResult(tripwire_triggered=False, info={"stage": "preflight"})],
        input=[GuardrailResult(tripwire_triggered=False, info={"stage": "input"})],
        output=[GuardrailResult(tripwire_triggered=False, info={"stage": "output"})],
    )


def test_direct_attribute_access_works() -> None:
    """Test that attributes can be accessed directly without llm_response."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Direct access should work without warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        assert response.id == "chatcmpl-123"  # noqa: S101
        assert response.model == "gpt-4"  # noqa: S101
        assert response.choices[0].message.content == "Hello, world!"  # noqa: S101
        assert response.usage.total_tokens == 15  # noqa: S101


def test_responses_api_direct_access_works() -> None:
    """Test that Responses API attributes can be accessed directly."""
    mock_llm_response = _create_mock_response()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Direct access should work without warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert response.id == "resp-123"  # noqa: S101
        assert response.output_text == "Hello from responses API!"  # noqa: S101
        assert response.conversation.id == "conv-123"  # noqa: S101


def test_guardrail_results_access_no_warning() -> None:
    """Test that accessing guardrail_results does NOT emit deprecation warning."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # guardrail_results access should NOT emit warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        assert response.guardrail_results is not None  # noqa: S101
        assert len(response.guardrail_results.preflight) == 1  # noqa: S101
        assert len(response.guardrail_results.input) == 1  # noqa: S101
        assert len(response.guardrail_results.output) == 1  # noqa: S101


def test_llm_response_access_emits_deprecation_warning() -> None:
    """Test that accessing llm_response emits a deprecation warning."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Accessing llm_response should emit a deprecation warning
    with pytest.warns(DeprecationWarning, match="Accessing 'llm_response' is deprecated"):
        _ = response.llm_response


def test_llm_response_chained_access_emits_warning() -> None:
    """Test that accessing llm_response.attribute emits warning (only once)."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Accessing llm_response.attribute should emit a deprecation warning (first time)
    with pytest.warns(DeprecationWarning, match="Accessing 'llm_response' is deprecated"):
        _ = response.llm_response.id

    # Accessing again should NOT emit another warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        _ = response.llm_response.model  # Should not raise


def test_hasattr_works_correctly() -> None:
    """Test that hasattr works correctly for delegated attributes."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # hasattr should work for delegated attributes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert hasattr(response, "id")  # noqa: S101
        assert hasattr(response, "choices")  # noqa: S101
        assert hasattr(response, "model")  # noqa: S101
        assert hasattr(response, "guardrail_results")  # noqa: S101
        assert not hasattr(response, "nonexistent_attribute")  # noqa: S101


def test_getattr_works_correctly() -> None:
    """Test that getattr works correctly for delegated attributes."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # getattr should work for delegated attributes
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert response.id == "chatcmpl-123"  # noqa: S101
        assert response.model == "gpt-4"  # noqa: S101
        assert getattr(response, "nonexistent", "default") == "default"  # noqa: S101


def test_attribute_error_for_missing_attributes() -> None:
    """Test that AttributeError is raised for missing attributes."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Accessing non-existent attributes should raise AttributeError
    with pytest.raises(AttributeError):
        _ = response.nonexistent_attribute


def test_method_calls_work() -> None:
    """Test that method calls on delegated objects work correctly."""
    # Create a mock with a method
    mock_llm_response = SimpleNamespace(
        id="resp-123",
        custom_method=lambda: "method result",
    )
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Method calls should work without warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert response.custom_method() == "method result"  # noqa: S101


def test_nested_attribute_access_works() -> None:
    """Test that nested attribute access works correctly."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Nested access should work without warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert response.choices[0].message.content == "Hello, world!"  # noqa: S101
        assert response.choices[0].message.role == "assistant"  # noqa: S101
        assert response.choices[0].finish_reason == "stop"  # noqa: S101


def test_property_access_works() -> None:
    """Test that property access on delegated objects works correctly."""
    # Create a mock with a property
    class MockResponse:
        @property
        def computed_value(self) -> str:
            return "computed"

    mock_llm_response = MockResponse()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Property access should work without warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert response.computed_value == "computed"  # noqa: S101


def test_backward_compatibility_still_works() -> None:
    """Test that old pattern (response.llm_response.attr) still works despite warning."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Old pattern should still work (with warning on first access)
    with pytest.warns(DeprecationWarning):
        assert response.llm_response.id == "chatcmpl-123"  # noqa: S101

    # Subsequent accesses should work without warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        assert response.llm_response.model == "gpt-4"  # noqa: S101
        assert response.llm_response.choices[0].message.content == "Hello, world!"  # noqa: S101


def test_deprecation_warning_message_content() -> None:
    """Test that the deprecation warning contains the expected message."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Check the full warning message
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = response.llm_response

        assert len(w) == 1  # noqa: S101
        assert issubclass(w[0].category, DeprecationWarning)  # noqa: S101
        assert "Accessing 'llm_response' is deprecated" in str(w[0].message)  # noqa: S101
        assert "response.output_text" in str(w[0].message)  # noqa: S101
        assert "future versions" in str(w[0].message)  # noqa: S101


def test_warning_only_once_per_instance() -> None:
    """Test that deprecation warning is only emitted once per instance."""
    mock_llm_response = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response = GuardrailsResponse(
        _llm_response=mock_llm_response,
        guardrail_results=guardrail_results,
    )

    # Track all warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Access llm_response multiple times (simulating streaming chunks)
        _ = response.llm_response
        _ = response.llm_response.id
        _ = response.llm_response.model
        _ = response.llm_response.choices

        # Should only have ONE warning despite multiple accesses
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 1  # noqa: S101


def test_separate_instances_warn_independently() -> None:
    """Test that different GuardrailsResponse instances warn independently."""
    mock_llm_response1 = _create_mock_chat_completion()
    mock_llm_response2 = _create_mock_chat_completion()
    guardrail_results = _create_mock_guardrail_results()

    response1 = GuardrailsResponse(
        _llm_response=mock_llm_response1,
        guardrail_results=guardrail_results,
    )

    response2 = GuardrailsResponse(
        _llm_response=mock_llm_response2,
        guardrail_results=guardrail_results,
    )

    # Track all warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Each instance should warn once
        _ = response1.llm_response
        _ = response2.llm_response

        # Multiple accesses to same instance should not warn again
        _ = response1.llm_response
        _ = response2.llm_response

        # Should have exactly TWO warnings (one per instance)
        deprecation_warnings = [warning for warning in w if issubclass(warning.category, DeprecationWarning)]
        assert len(deprecation_warnings) == 2  # noqa: S101

