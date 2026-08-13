"""Focused regression coverage for parser edge cases."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from guardrails.checks.text.secret_keys import SecretKeysCfg, _embedded_secret_candidates, secret_keys

SYNTHETIC_VALUE = "Aa0Bb1Cc2Dd3Ee4Ff5"


@given(separator=st.sampled_from(["/", "\\", "%2F", "%2f", "%5C", "%5c"]))
def test_file_path_label_pairs_with_following_value(separator: str) -> None:
    token = f"files{separator}token{separator}{SYNTHETIC_VALUE}{separator}image.png"

    assert SYNTHETIC_VALUE in _embedded_secret_candidates(token)  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("punctuation", [".", ",", ";", ":", "!", "?", ")", "]"])
async def test_url_trailing_punctuation_is_removed_before_custom_match(punctuation: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}$"])
    result = await secret_keys(None, f"https://example.com/internal-ab12{punctuation}", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text",
    [
        "https://example.com/?redirect=https%3A%2F%2Fother.example%2F%3Ftoken%3Dinternal%252Bab12",
        "https://example.com/#redirect=https%3A%2F%2Fother.example%2F%3Ftoken%3Dinternal%252Bab12",
    ],
)
async def test_nested_url_escape_is_preserved(text: str) -> None:
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal\+ab12$"])
    result = await secret_keys(None, text, config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal+ab12" in result.info["detected_secrets"]  # noqa: S101
