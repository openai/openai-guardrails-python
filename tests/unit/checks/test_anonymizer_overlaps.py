"""Regression coverage for overlapping PII detections."""

from __future__ import annotations

import pytest
from presidio_analyzer import RecognizerResult

from guardrails.checks.text.pii import PIIConfig, PIIEntity, _get_analyzer_engine, pii
from guardrails.utils.anonymizer import OperatorConfig, anonymize


@pytest.mark.asyncio
async def test_real_overlapping_detections_are_fully_masked() -> None:
    """Presidio's email and URL recognizers can return partially overlapping spans."""
    text = "Contact: jane@example.com/profile. End."
    entities = [PIIEntity.EMAIL_ADDRESS, PIIEntity.URL]
    config = PIIConfig(entities=entities, block=False)
    detections = _get_analyzer_engine().analyze(text, entities=[entity.value for entity in entities], language="en")
    assert {(r.entity_type, r.start, r.end) for r in detections} == {("EMAIL_ADDRESS", 9, 25), ("URL", 14, 34)}

    result = await pii(None, text, config)

    assert result.info["checked_text"] == "Contact: <URL> End."
    assert result.tripwire_triggered is False
    assert set(result.info["detected_entities"]) == {"EMAIL_ADDRESS", "URL"}


@pytest.mark.parametrize(
    ("spans", "expected"),
    [
        ([], "abcdefghijklmnop"),
        ([(0, 4, "A"), (4, 8, "B")], "<A><B>ijklmnop"),
        ([(0, 8, "A"), (2, 4, "B")], "<A>ijklmnop"),
        ([(0, 6, "A"), (4, 10, "B")], "<A>klmnop"),
        ([(0, 6, "A"), (4, 10, "B"), (9, 16, "C")], "<C>"),
        ([(0, 7, "A"), (5, 10, "B"), (9, 16, "C")], "<A>"),
    ],
    ids=["empty", "adjacent", "contained", "partial", "transitive-longest-original", "transitive-earliest-tie"],
)
def test_overlap_coverage_and_precedence(spans: list[tuple[int, int, str]], expected: str) -> None:
    """Cover connected detections while preserving marker precedence and input spans."""
    detections = [RecognizerResult(entity_type=entity, start=start, end=end, score=1.0) for start, end, entity in spans]
    operators = {entity: OperatorConfig("replace", {"new_value": f"<{entity}>"}) for _, _, entity in spans}

    for ordered in (detections, list(reversed(detections))):
        result = anonymize("abcdefghijklmnop", ordered, operators)
        assert result.text == expected
        assert [(r.start, r.end, r.entity_type) for r in detections] == spans
