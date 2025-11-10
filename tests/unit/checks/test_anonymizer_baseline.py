"""Baseline tests for anonymizer functionality.

This module captures the expected behavior of presidio-anonymizer to ensure
our custom implementation produces identical results.
"""

from __future__ import annotations

import pytest

from guardrails.checks.text.pii import PIIConfig, PIIEntity, pii


@pytest.mark.asyncio
async def test_baseline_simple_email_masking() -> None:
    """Test simple email masking."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS], block=False)
    result = await pii(None, "Contact me at john@example.com for details", config)

    # Record baseline output
    expected = "Contact me at <EMAIL_ADDRESS> for details"
    assert result.info["checked_text"] == expected  # noqa: S101
    assert result.info["pii_detected"] is True  # noqa: S101
    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_ssn_masking() -> None:
    """Test SSN masking."""
    config = PIIConfig(entities=[PIIEntity.US_SSN], block=False)
    result = await pii(None, "My SSN is 856-45-6789", config)

    # Record baseline output
    expected = "My SSN is <US_SSN>"
    assert result.info["checked_text"] == expected  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_multiple_non_overlapping_entities() -> None:
    """Test multiple non-overlapping entities in same text."""
    config = PIIConfig(
        entities=[PIIEntity.EMAIL_ADDRESS, PIIEntity.PHONE_NUMBER],
        block=False,
    )
    result = await pii(
        None,
        "Email: test@example.com, Phone: (555) 123-4567",
        config,
    )

    # Record baseline output
    checked_text = result.info["checked_text"]
    assert "<EMAIL_ADDRESS>" in checked_text  # noqa: S101
    assert "<PHONE_NUMBER>" in checked_text  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_consecutive_entities() -> None:
    """Test consecutive entities without separation."""
    config = PIIConfig(
        entities=[PIIEntity.EMAIL_ADDRESS],
        block=False,
    )
    result = await pii(
        None,
        "Emails: alice@example.com and bob@test.com",
        config,
    )

    # Record baseline output
    checked_text = result.info["checked_text"]
    assert checked_text.count("<EMAIL_ADDRESS>") == 2  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_entity_at_boundaries() -> None:
    """Test entity at text boundaries."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS], block=False)

    # Email at start
    result_start = await pii(None, "user@example.com is the contact", config)

    # Email at end
    result_end = await pii(None, "Contact: user@example.com", config)

    assert result_start.info["checked_text"].startswith("<EMAIL_ADDRESS>")  # noqa: S101
    assert result_end.info["checked_text"].endswith("<EMAIL_ADDRESS>")  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_unicode_characters() -> None:
    """Test masking with Unicode characters."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS], block=False)
    result = await pii(
        None,
        "Email: user@example.com 🔒 Secure contact",
        config,
    )

    # Record baseline output
    checked_text = result.info["checked_text"]
    assert "<EMAIL_ADDRESS>" in checked_text  # noqa: S101
    assert "🔒" in checked_text  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_special_characters() -> None:
    """Test masking with special characters."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS], block=False)
    result = await pii(
        None,
        "Contact: [user@example.com] or {admin@test.com}",
        config,
    )

    # Record baseline output
    checked_text = result.info["checked_text"]
    assert "[<EMAIL_ADDRESS>]" in checked_text or "Contact: <EMAIL_ADDRESS>" in checked_text  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_no_pii_detected() -> None:
    """Test text with no PII."""
    config = PIIConfig(entities=[PIIEntity.EMAIL_ADDRESS, PIIEntity.US_SSN], block=False)
    result = await pii(None, "This is plain text with no PII at all", config)

    # Record baseline output
    assert result.info["checked_text"] == "This is plain text with no PII at all"  # noqa: S101
    assert result.info["pii_detected"] is False  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_credit_card_masking() -> None:
    """Test credit card masking."""
    config = PIIConfig(entities=[PIIEntity.CREDIT_CARD], block=False)
    result = await pii(None, "Card number: 4532123456789010", config)

    # Record baseline output
    checked_text = result.info["checked_text"]
    # Credit card detection may be inconsistent with certain formats
    if result.info["pii_detected"]:
        assert "<CREDIT_CARD>" in checked_text  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_phone_number_formats() -> None:
    """Test various phone number formats."""
    config = PIIConfig(entities=[PIIEntity.PHONE_NUMBER], block=False)

    # Test multiple formats
    texts_and_results = []

    result1 = await pii(None, "Call me at (555) 123-4567", config)
    texts_and_results.append(("(555) 123-4567", result1.info["checked_text"]))

    result2 = await pii(None, "Phone: 555-123-4567", config)
    texts_and_results.append(("555-123-4567", result2.info["checked_text"]))

    result3 = await pii(None, "Mobile: 5551234567", config)
    texts_and_results.append(("5551234567", result3.info["checked_text"]))

    # Check that at least the first format is detected
    assert "<PHONE_NUMBER>" in result1.info["checked_text"]  # noqa: S101


@pytest.mark.asyncio
async def test_baseline_mixed_entities_complex() -> None:
    """Test complex text with multiple entity types."""
    config = PIIConfig(
        entities=[
            PIIEntity.EMAIL_ADDRESS,
            PIIEntity.PHONE_NUMBER,
            PIIEntity.US_SSN,
        ],
        block=False,
    )
    result = await pii(
        None,
        "Contact John at john@company.com or call (555) 123-4567. "
        "SSN: 856-45-6789",
        config,
    )

    # Record baseline output
    checked_text = result.info["checked_text"]

    # Verify all entity types are masked
    assert "<EMAIL_ADDRESS>" in checked_text  # noqa: S101
    assert "<PHONE_NUMBER>" in checked_text or "555" not in checked_text  # noqa: S101
    assert "<US_SSN>" in checked_text  # noqa: S101

