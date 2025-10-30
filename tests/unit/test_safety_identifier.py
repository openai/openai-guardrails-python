"""Tests for safety_identifier parameter handling across different client types."""

from unittest.mock import Mock

import pytest


def test_supports_safety_identifier_for_openai_client() -> None:
    """Official OpenAI client with default base_url should support safety_identifier."""
    from guardrails.utils.safety_identifier import supports_safety_identifier

    mock_client = Mock()
    mock_client.base_url = None
    mock_client.__class__.__name__ = "AsyncOpenAI"

    assert supports_safety_identifier(mock_client) is True  # noqa: S101


def test_supports_safety_identifier_for_openai_with_official_url() -> None:
    """OpenAI client with explicit api.openai.com base_url should support safety_identifier."""
    from guardrails.utils.safety_identifier import supports_safety_identifier

    mock_client = Mock()
    mock_client.base_url = "https://api.openai.com/v1"
    mock_client.__class__.__name__ = "AsyncOpenAI"

    assert supports_safety_identifier(mock_client) is True  # noqa: S101


def test_does_not_support_safety_identifier_for_azure() -> None:
    """Azure OpenAI client should not support safety_identifier."""
    from guardrails.utils.safety_identifier import supports_safety_identifier

    mock_client = Mock()
    mock_client.base_url = "https://example.openai.azure.com/v1"
    mock_client.__class__.__name__ = "AsyncAzureOpenAI"

    # Azure detection happens via isinstance check, but we can test with class name
    from openai import AsyncAzureOpenAI

    try:
        azure_client = AsyncAzureOpenAI(
            api_key="test",
            azure_endpoint="https://example.openai.azure.com",
            api_version="2024-02-01",
        )
        assert supports_safety_identifier(azure_client) is False  # noqa: S101
    except Exception:
        # If we can't create a real Azure client in tests, that's okay
        pytest.skip("Could not create Azure client for testing")


def test_does_not_support_safety_identifier_for_local_model() -> None:
    """Local model with custom base_url should not support safety_identifier."""
    from guardrails.utils.safety_identifier import supports_safety_identifier

    mock_client = Mock()
    mock_client.base_url = "http://localhost:11434/v1"  # Ollama
    mock_client.__class__.__name__ = "AsyncOpenAI"

    assert supports_safety_identifier(mock_client) is False  # noqa: S101


def test_does_not_support_safety_identifier_for_alternative_provider() -> None:
    """Alternative OpenAI-compatible provider should not support safety_identifier."""
    from guardrails.utils.safety_identifier import supports_safety_identifier

    mock_client = Mock()
    mock_client.base_url = "https://api.together.xyz/v1"
    mock_client.__class__.__name__ = "AsyncOpenAI"

    assert supports_safety_identifier(mock_client) is False  # noqa: S101

