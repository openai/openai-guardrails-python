"""Regression coverage for extracted Secret Keys candidates."""

from __future__ import annotations

import pytest

from guardrails.checks.text.secret_keys import SecretKeysCfg, secret_keys


@pytest.mark.asyncio
async def test_secret_key_path_candidate_with_allowed_extension_is_detected() -> None:
    config = SecretKeysCfg(threshold="balanced")
    result = await secret_keys(
        None,
        "https://example.com/sk-AAAABBBBCCCCDDDD.png?download=1",
        config,
    )

    assert result.tripwire_triggered is True  # noqa: S101
    assert "sk-AAAABBBBCCCCDDDD" in result.info["detected_secrets"]  # noqa: S101
