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


@dataclass(slots=True)
class _AnonymizationSpan:
    """A replacement span owned by the anonymizer, independent of detections."""

    start: int
    end: int
    entity_type: str


def _resolve_overlaps(results: Sequence[RecognizerResult]) -> list[RecognizerResult]:
    """Merge overlapping spans without losing any detected text coverage.

    Use the longest original detection's entity type for each connected group,
    preferring the earlier detection for equal lengths. Adjacent spans remain
    separate, and the analyzer's results are never mutated.

    Args:
        results: Sequence of recognizer results to resolve.

    Returns:
        Non-overlapping replacement spans covering every detection.
    """
    merged: list[RecognizerResult] = []
    preferred: RecognizerResult | None = None
    for result in sorted(results, key=lambda r: r.start):
        if not merged or result.start >= merged[-1].end:
            merged.append(_AnonymizationSpan(result.start, result.end, result.entity_type))
            preferred = result
            continue

        current = merged[-1]
        current.end = max(current.end, result.end)
        if preferred is not None and result.end - result.start > preferred.end - preferred.start:
            current.entity_type = result.entity_type
            preferred = result

    return merged


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
            masked_text = masked_text[: result.start] + new_value + masked_text[result.end :]

    return AnonymizeResult(text=masked_text)
