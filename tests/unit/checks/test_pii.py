"""Tests for PII detection guardrail.

This module tests the PII detection functionality including entity detection,
masking behavior, and blocking behavior for various entity types.
"""

from __future__ import annotations

import pytest

from guardrails.checks.text.pii import PIIConfig, PIIEntity, pii
from guardrails.types import GuardrailResult


@pytest.mark.asyncio
async def test_pii_detects_korean_resident_registration_number() -> None:
    """Detect Korean Resident Registration Numbers with valid date and checksum."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=True)
    # Using valid RRN: 900101-2345670
    # Date: 900101 (Jan 1, 1990), Gender: 2, Serial: 34567, Checksum: 0
    result = await pii(None, "My RRN is 900101-2345670", config)

    assert isinstance(result, GuardrailResult)  # noqa: S101
    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["guardrail_name"] == "Contains PII"  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "KR_RRN" in result.info["detected_entities"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_masks_korean_rrn_in_non_blocking_mode() -> None:
    """Korean RRN with valid date and checksum should be masked when block=False."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=False)
    # Using valid RRN: 900101-2345670
    result = await pii(None, "My RRN is 900101-2345670", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert result.info["block_mode"] is False  # noqa: S101
    assert "<KR_RRN>" in result.info["checked_text"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_detects_multiple_entity_types() -> None:
    """Detect multiple PII entity types with valid dates and checksums."""
    config = PIIConfig(
        entities=[PIIEntity.EMAIL_ADDRESS, PIIEntity.KR_RRN],
        block=True,
    )
    result = await pii(
        None,
        "Contact: user@example.com, Korean RRN: 900101-2345670",
        config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    detected = result.info["detected_entities"]
    # Verify both entity types are detected
    assert "EMAIL_ADDRESS" in detected  # noqa: S101
    assert "KR_RRN" in detected  # noqa: S101
    # Verify actual values were captured
    assert detected["EMAIL_ADDRESS"] == ["user@example.com"]  # noqa: S101
    assert detected["KR_RRN"] == ["900101-2345670"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_masks_multiple_entity_types() -> None:
    """Mask multiple PII entity types with valid checksums."""
    config = PIIConfig(
        entities=[PIIEntity.EMAIL_ADDRESS, PIIEntity.KR_RRN],
        block=False,
    )
    result = await pii(
        None,
        "Contact: user@example.com, Korean RRN: 123456-1234563",
        config,
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    checked_text = result.info["checked_text"]
    assert "<EMAIL_ADDRESS>" in checked_text  # noqa: S101


@pytest.mark.asyncio
async def test_pii_does_not_trigger_on_clean_text() -> None:
    """Guardrail should not trigger when no PII is present."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN, PIIEntity.EMAIL_ADDRESS], block=True)
    result = await pii(None, "This is clean text with no PII", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is False  # noqa: S101
    assert result.info["detected_entities"] == {}  # noqa: S101


@pytest.mark.asyncio
async def test_pii_blocking_mode_triggers_tripwire() -> None:
    """Blocking mode should trigger tripwire when PII is detected."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS], block=True)
    result = await pii(None, "Contact me at test@example.com", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["block_mode"] is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101


@pytest.mark.asyncio
async def test_pii_masking_mode_does_not_trigger_tripwire() -> None:
    """Masking mode should not trigger tripwire even when PII is detected."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS], block=False)
    result = await pii(None, "Contact me at test@example.com", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["block_mode"] is False  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "<EMAIL_ADDRESS>" in result.info["checked_text"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_checked_text_unchanged_when_no_pii() -> None:
    """Checked text should remain unchanged when no PII is detected."""
    original_text = "This is clean text"
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS, PIIEntity.KR_RRN], block=False)
    result = await pii(None, original_text, config)

    assert result.info["checked_text"] == original_text  # noqa: S101
    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_pii_entity_types_checked_in_result() -> None:
    """Result should include list of entity types that were checked."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN, PIIEntity.EMAIL_ADDRESS, PIIEntity.US_SSN])
    result = await pii(None, "Clean text", config)

    entity_types = result.info["entity_types_checked"]
    assert PIIEntity.KR_RRN in entity_types  # noqa: S101
    assert PIIEntity.EMAIL_ADDRESS in entity_types  # noqa: S101
    assert PIIEntity.US_SSN in entity_types  # noqa: S101


@pytest.mark.asyncio
async def test_pii_config_defaults_to_masking_mode() -> None:
    """PIIConfig should default to masking mode (block=False)."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS])

    assert config.block is False  # noqa: S101


@pytest.mark.asyncio
async def test_pii_detects_us_ssn() -> None:
    """Detect US Social Security Numbers (regression test for existing functionality)."""
    config = PIIConfig(entities=[PIIEntity.US_SSN], block=True)
    # Use a valid SSN pattern that Presidio can detect (Presidio validates SSN patterns)
    result = await pii(None, "My social security number is 856-45-6789", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "US_SSN" in result.info["detected_entities"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_detects_phone_numbers() -> None:
    """Detect phone numbers (regression test for existing functionality)."""
    config = PIIConfig(entities=[PIIEntity.PHONE_NUMBER], block=True)
    result = await pii(None, "Call me at 555-123-4567", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "PHONE_NUMBER" in result.info["detected_entities"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_multiple_occurrences_of_same_entity() -> None:
    """Detect multiple occurrences of the same entity type."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS], block=True)
    result = await pii(
        None,
        "Contact alice@example.com or bob@example.com",
        config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "EMAIL_ADDRESS" in result.info["detected_entities"]  # noqa: S101
    assert len(result.info["detected_entities"]["EMAIL_ADDRESS"]) >= 1  # noqa: S101


@pytest.mark.asyncio
async def test_pii_detects_korean_rrn_with_invalid_checksum() -> None:
    """Presidio's KR_RRN recognizer detects patterns even with invalid checksums.

    Note: Presidio 2.2.360's implementation focuses on pattern matching rather than
    strict checksum validation, so it will detect RRN-like patterns regardless of
    checksum validity.
    """
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=True)
    # Using valid date but invalid checksum: 900101-2345679 (should be 900101-2345670)
    result = await pii(None, "My RRN is 900101-2345679", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "KR_RRN" in result.info["detected_entities"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_detects_korean_rrn_with_invalid_date() -> None:
    """Presidio's KR_RRN recognizer detects some patterns even with invalid dates.

    Note: Presidio 2.2.360's implementation may detect certain RRN-like patterns
    even if the date component is invalid (e.g., Feb 30). The recognizer prioritizes
    pattern matching over strict date validation.
    """
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=True)
    # Testing with Feb 30 which is an invalid date but matches the pattern
    result = await pii(None, "Korean RRN: 990230-1234567", config)

    # Presidio detects this pattern despite the invalid date
    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "KR_RRN" in result.info["detected_entities"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_accepts_valid_korean_rrn_dates() -> None:
    """Korean RRN with valid dates in different formats should be detected."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=False)
    valid_rrn = "900101-1234568"
    result = await pii(None, f"RRN: {valid_rrn}", config)

    # Should detect if date is valid
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "KR_RRN" in result.info["detected_entities"]  # noqa: S101
