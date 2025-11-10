"""Custom anonymizer for PII masking.

This module provides a lightweight replacement for presidio-anonymizer,
implementing text masking functionality for detected PII entities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol


class RecognizerResult(Protocol):
    """Protocol for analyzer results from presidio-analyzer.

    Attributes:
        start: Start position of the entity in text.
        end: End position of the entity in text.
        entity_type: Type of the detected entity (e.g., "EMAIL_ADDRESS").
    """

    start: int
    end: int
    entity_type: str


@dataclass(frozen=True, slots=True)
class OperatorConfig:
    """Configuration for an anonymization operator.

    Args:
        operator_name: Name of the operator (e.g., "replace").
        params: Parameters for the operator (e.g., {"new_value": "<EMAIL>"}).
    """

    operator_name: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AnonymizeResult:
    """Result of text anonymization.

    Attributes:
        text: The anonymized text with entities masked.
    """

    text: str


def _resolve_overlaps(results: Sequence[RecognizerResult]) -> list[RecognizerResult]:
    """Remove overlapping entity spans, keeping longer/earlier ones.

    When entities overlap, prioritize:
    1. Longer spans over shorter ones
    2. Earlier positions when spans are equal length

    Args:
        results: Sequence of recognizer results to resolve.

    Returns:
        List of non-overlapping recognizer results.

    Examples:
        >>> # If EMAIL_ADDRESS spans (0, 20) and PERSON spans (5, 10), keep EMAIL_ADDRESS
        >>> # If two entities span (0, 10) and (5, 15), keep the one starting at 0
    """
    if not results:
        return []

    # Sort by: 1) longer spans first, 2) earlier position for equal lengths
    sorted_results = sorted(
        results,
        key=lambda r: (-(r.end - r.start), r.start),
    )

    # Filter out overlapping spans
    non_overlapping: list[RecognizerResult] = []
    for result in sorted_results:
        # Check if this result overlaps with any already selected
        overlaps = False
        for selected in non_overlapping:
            # Two spans overlap if one starts before the other ends
            if (result.start < selected.end and result.end > selected.start):
                overlaps = True
                break

        if not overlaps:
            non_overlapping.append(result)

    return non_overlapping


def anonymize(
    text: str,
    analyzer_results: Sequence[RecognizerResult],
    operators: dict[str, OperatorConfig],
) -> AnonymizeResult:
    """Anonymize text by replacing detected entities with placeholders.

    This function replicates presidio-anonymizer's behavior for the "replace"
    operator, which we use to mask PII with placeholders like "<EMAIL_ADDRESS>".

    Args:
        text: The original text to anonymize.
        analyzer_results: Sequence of detected entities with positions.
        operators: Mapping from entity type to operator configuration.

    Returns:
        AnonymizeResult with masked text.

    Examples:
        >>> from collections import namedtuple
        >>> Result = namedtuple("Result", ["start", "end", "entity_type"])
        >>> results = [Result(start=10, end=25, entity_type="EMAIL_ADDRESS")]
        >>> operators = {"EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "<EMAIL_ADDRESS>"})}
        >>> result = anonymize("Contact: john@example.com", results, operators)
        >>> result.text
        'Contact: <EMAIL_ADDRESS>'
    """
    if not analyzer_results or not text:
        return AnonymizeResult(text=text)

    # Resolve overlapping entities
    non_overlapping = _resolve_overlaps(analyzer_results)

    # Sort by position (reverse order) to maintain correct offsets during replacement
    sorted_results = sorted(non_overlapping, key=lambda r: r.start, reverse=True)

    # Replace entities from end to start
    masked_text = text
    for result in sorted_results:
        entity_type = result.entity_type
        operator_config = operators.get(entity_type)

        if operator_config and operator_config.operator_name == "replace":
            # Extract the replacement value
            new_value = operator_config.params.get("new_value", f"<{entity_type}>")
            # Replace the text span
            masked_text = (
                masked_text[: result.start]
                + new_value
                + masked_text[result.end :]
            )

    return AnonymizeResult(text=masked_text)

