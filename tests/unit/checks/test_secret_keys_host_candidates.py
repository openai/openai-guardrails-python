"""Regression coverage for URL host candidate extraction."""

from __future__ import annotations

import pytest

from guardrails.checks.text.secret_keys import SecretKeysCfg, secret_keys


@pytest.mark.asyncio
async def test_secret_keys_checks_prefixed_hostname_label() -> None:
    expected = "sk-AAAABBBBCCCCDDDD"
    result = await secret_keys(
        None,
        f"https://{expected}.example.com",
        SecretKeysCfg(threshold="balanced"),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert expected in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_checks_custom_hostname_label() -> None:
    result = await secret_keys(
        None,
        "https://internal-ab12.example.com",
        SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}"]),
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101
