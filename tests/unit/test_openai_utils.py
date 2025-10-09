"""Tests for OpenAI client helper utilities."""

from guardrails._openai_utils import SAFETY_IDENTIFIER_HEADER, SAFETY_IDENTIFIER_VALUE, ensure_safety_identifier_header, prepare_openai_kwargs


def test_prepare_openai_kwargs_adds_safety_identifier() -> None:
    """Default kwargs gain the Guardrails safety identifier."""
    result = prepare_openai_kwargs({})
    headers = result["default_headers"]
    assert headers[SAFETY_IDENTIFIER_HEADER] == SAFETY_IDENTIFIER_VALUE  # noqa: S101


def test_prepare_openai_kwargs_overrides_existing_identifier() -> None:
    """Existing identifier value is overwritten with Guardrails tag."""
    kwargs = {"default_headers": {SAFETY_IDENTIFIER_HEADER: "custom", "X-Test": "value"}}
    result = prepare_openai_kwargs(kwargs)
    headers = result["default_headers"]
    assert headers["X-Test"] == "value"  # noqa: S101
    assert headers[SAFETY_IDENTIFIER_HEADER] == SAFETY_IDENTIFIER_VALUE  # noqa: S101


def test_prepare_openai_kwargs_handles_non_mapping_as_none() -> None:
    """Non-mapping default headers fall back to an empty mapping."""
    result = prepare_openai_kwargs({"default_headers": object()})
    headers = result["default_headers"]
    assert headers == {SAFETY_IDENTIFIER_HEADER: SAFETY_IDENTIFIER_VALUE}  # noqa: S101


def test_ensure_safety_identifier_header_adds_identifier() -> None:
    """ensure_safety_identifier_header augments mappings."""
    headers = ensure_safety_identifier_header({"X-Test": "value"})
    assert headers["X-Test"] == "value"  # noqa: S101
    assert headers[SAFETY_IDENTIFIER_HEADER] == SAFETY_IDENTIFIER_VALUE  # noqa: S101
