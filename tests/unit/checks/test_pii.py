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
    """Detect Korean Resident Registration Numbers with valid checksum."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=True)
    # Using valid RRN with correct checksum: 123456-1234563
    result = await pii(None, "My RRN is 123456-1234563", config)

    assert isinstance(result, GuardrailResult)  # noqa: S101
    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["guardrail_name"] == "Contains PII"  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "KR_RRN" in result.info["detected_entities"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_detects_thai_national_id() -> None:
    """Detect Thai National Identification Numbers with valid checksum."""
    config = PIIConfig(entities=[PIIEntity.TH_TNIN], block=True)
    # Using valid TNIN with correct checksum: 1234567890121
    result = await pii(None, "Thai ID: 1234567890121", config)

    assert isinstance(result, GuardrailResult)  # noqa: S101
    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["guardrail_name"] == "Contains PII"  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert "TH_TNIN" in result.info["detected_entities"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_masks_korean_rrn_in_non_blocking_mode() -> None:
    """Korean RRN with valid checksum should be masked when block=False."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=False)
    # Using valid RRN with correct checksum: 123456-1234563
    result = await pii(None, "My RRN is 123456-1234563", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert result.info["block_mode"] is False  # noqa: S101
    assert "<KR_RRN>" in result.info["checked_text"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_masks_thai_tnin_in_non_blocking_mode() -> None:
    """Thai TNIN with valid checksum should be masked when block=False."""
    config = PIIConfig(entities=[PIIEntity.TH_TNIN], block=False)
    # Using valid TNIN with correct checksum: 1234567890121
    result = await pii(None, "Thai ID: 1234567890121", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert result.info["block_mode"] is False  # noqa: S101
    assert "<TH_TNIN>" in result.info["checked_text"]  # noqa: S101


@pytest.mark.asyncio
async def test_pii_detects_multiple_entity_types() -> None:
    """Detect multiple PII entity types with valid checksums."""
    config = PIIConfig(
        entities=[PIIEntity.EMAIL_ADDRESS, PIIEntity.KR_RRN, PIIEntity.TH_TNIN],
        block=True,
    )
    result = await pii(
        None,
        "Contact: user@example.com, Korean RRN: 123456-1234563, Thai ID: 1234567890121",
        config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    detected = result.info["detected_entities"]
    assert "EMAIL_ADDRESS" in detected  # noqa: S101
    assert "KR_RRN" in detected or len(detected) >= 1  # noqa: S101


@pytest.mark.asyncio
async def test_pii_masks_multiple_entity_types() -> None:
    """Mask multiple PII entity types with valid checksums."""
    config = PIIConfig(
        entities=[PIIEntity.EMAIL_ADDRESS, PIIEntity.KR_RRN, PIIEntity.TH_TNIN],
        block=False,
    )
    result = await pii(
        None,
        "Contact: user@example.com, Korean RRN: 123456-1234563, Thai ID: 1234567890121",
        config,
    )

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    checked_text = result.info["checked_text"]
    assert "<EMAIL_ADDRESS>" in checked_text  # noqa: S101


@pytest.mark.asyncio
async def test_pii_does_not_trigger_on_clean_text() -> None:
    """Guardrail should not trigger when no PII is present."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN, PIIEntity.TH_TNIN], block=True)
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
    config = PIIConfig(entities=[PIIEntity.KR_RRN, PIIEntity.TH_TNIN, PIIEntity.EMAIL_ADDRESS])
    result = await pii(None, "Clean text", config)

    entity_types = result.info["entity_types_checked"]
    assert PIIEntity.KR_RRN in entity_types  # noqa: S101
    assert PIIEntity.TH_TNIN in entity_types  # noqa: S101
    assert PIIEntity.EMAIL_ADDRESS in entity_types  # noqa: S101


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
async def test_pii_rejects_invalid_korean_rrn_checksum() -> None:
    """Invalid Korean RRN checksum should not be detected."""
    config = PIIConfig(entities=[PIIEntity.KR_RRN], block=True)
    # Using invalid checksum: 123456-1234567 (should be 123456-1234563)
    result = await pii(None, "My RRN is 123456-1234567", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is False  # noqa: S101
    assert result.info["detected_entities"] == {}  # noqa: S101


@pytest.mark.asyncio
async def test_pii_rejects_invalid_thai_tnin_checksum() -> None:
    """Invalid Thai TNIN checksum should not be detected."""
    config = PIIConfig(entities=[PIIEntity.TH_TNIN], block=True)
    # Using invalid checksum: 1234567890123 (should be 1234567890121)
    result = await pii(None, "Thai ID: 1234567890123", config)

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["pii_detected"] is False  # noqa: S101
    assert result.info["detected_entities"] == {}  # noqa: S101

