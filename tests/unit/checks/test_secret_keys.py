"""Tests for secret key detection guardrail."""

from __future__ import annotations

import importlib
import threading

import pytest

from guardrails.checks.text.secret_keys import SecretKeysCfg, _detect_secret_keys, secret_keys


def test_detect_secret_keys_flags_high_entropy_strings() -> None:
    """High entropy tokens should be detected as potential secrets."""
    text = "API key sk-AAAABBBBCCCCDDDD"
    result = _detect_secret_keys(text, cfg={"min_length": 10, "min_entropy": 3.5, "min_diversity": 2, "strict_mode": True})

    assert result.tripwire_triggered is True  # noqa: S101
    assert "sk-AAAABBBBCCCCDDDD" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_with_custom_regex() -> None:
    """Custom regex patterns should trigger detection."""
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"internal-[a-z0-9]{4}"])
    result = await secret_keys(None, "internal-ab12 leaked", config)

    assert result.tripwire_triggered is True  # noqa: S101
    assert "internal-ab12" in result.info["detected_secrets"]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_revalidates_mutated_custom_regex() -> None:
    """Reject unsafe expressions appended after configuration validation."""
    config = SecretKeysCfg(threshold="balanced", custom_regex=[r"^safe$"])
    assert config.custom_regex is not None  # noqa: S101
    config.custom_regex.append(r"(a+)+$")

    with pytest.raises(ValueError, match="Unsafe regex pattern"):
        await secret_keys(None, "https://example.com/aaaaaaaaaaaaaaaa!", config)


@pytest.mark.asyncio
async def test_secret_keys_offloads_synchronous_scanning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run CPU-bound scanning outside the caller's event-loop thread.

    Args:
        monkeypatch: Pytest fixture used to replace the synchronous scanner.
    """
    module = importlib.import_module("guardrails.checks.text.secret_keys")
    caller_thread = threading.get_ident()
    scanner_threads: list[int] = []

    def fake_detect_secret_keys(*_args: object) -> object:
        scanner_threads.append(threading.get_ident())
        return module.GuardrailResult(tripwire_triggered=False, info={})

    monkeypatch.setattr(module, "_detect_secret_keys", fake_detect_secret_keys)

    result = await module.secret_keys(None, "benign", module.SecretKeysCfg())

    assert result.tripwire_triggered is False  # noqa: S101
    assert scanner_threads and scanner_threads[0] != caller_thread  # noqa: S101


@pytest.mark.parametrize(
    "pattern",
    [
        r"(a+)+$",
        r"(a|aa)+$",
        r"a+a+$",
        r".*a.*$",
        "^" + "(a|aa)" * 20 + "b$",
        "^" + "a?" * 20 + "a" * 20 + "$",
        r"(?i)(a.|Ab)+$",
        r"(?i:a+)(?-i:A+)$",
        r"^(a)?(?(1)(b+)+$|(c+)+$)",
        r"(?:a(?=a)?)+$",
        r"(a+){8}$",
        r"(a+){7}$",
        r"(a+){2}$",
        r"(a{1,100}){7}$",
        r"a*b?a*$",
        r"(a+)\1$",
        r"(?P<value>a+)(?P=value)$",
        r"^(?:(?!a*aX$).)*Z$",
    ],
)
def test_secret_keys_rejects_regexes_with_excessive_backtracking(pattern: str) -> None:
    """Reject custom patterns with excessive backtracking.

    Args:
        pattern: Unsafe custom expression to validate.
    """
    with pytest.raises(ValueError, match="Unsafe regex pattern"):
        SecretKeysCfg(threshold="balanced", custom_regex=[pattern])


@pytest.mark.parametrize(
    "pattern",
    [
        r"internal-[a-z0-9]{4}$",
        r"^[A-Za-z]+-[0-9]+$",
        r"(?:ab)+$",
        r"(?:ab?){2}$",
        r"a{1,2}a{1,2}$",
        r"(?:ab?)+$",
        r"(?:a+b)+$",
        r"(?:[A-Za-z0-9]+[-_]){2}[A-Za-z0-9]+$",
        r"(?>a+)+$",
        r"(?:a++)+$",
        r"custom-(?:one|two)$",
        r"foo(?:-bar)?(?:-baz)?$",
        r"(ab)\1$",
    ],
)
def test_secret_keys_accepts_bounded_custom_regexes(pattern: str) -> None:
    """Keep common deterministic custom patterns supported.

    Args:
        pattern: Bounded custom expression to validate.
    """
    assert SecretKeysCfg(threshold="balanced", custom_regex=[pattern]).custom_regex == [pattern]  # noqa: S101


@pytest.mark.asyncio
async def test_secret_keys_ignores_non_matching_input() -> None:
    """Benign inputs should not trigger the guardrail."""
    config = SecretKeysCfg(threshold="permissive", custom_regex=None)
    result = await secret_keys(None, "Hello world", config)

    assert result.tripwire_triggered is False  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", ["", " tail_"])
async def test_private_identifiers_are_not_markdown_openers(suffix: str) -> None:
    """Leading identifier underscores should not overflow presentation state."""
    identifiers = " ".join(f"_{character}" for character in "abcdefghijklmnopq") + suffix

    result = await secret_keys(None, identifiers, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_shell_globs_are_not_markdown_openers() -> None:
    """Unmatched leading stars should not overflow presentation state."""
    globs = " ".join(f"*{character}" for character in "abcdefghijklmnopq")

    result = await secret_keys(None, globs, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
@pytest.mark.parametrize("quote", ["'", '"'])
async def test_unmatched_quote_prefixed_words_are_not_presentation_openers(quote: str) -> None:
    """Unmatched prose quotes should not overflow presentation state."""
    words = " ".join(f"{quote}{character}" for character in "abcdefghijklmnopq")

    result = await secret_keys(None, words, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


@pytest.mark.asyncio
async def test_unmatched_asymmetric_delimiters_are_not_presentation_openers() -> None:
    """Do not retain unmatched asymmetric openers across lexical tokens."""
    words = " ".join(f"({character}" for character in "abcdefghijklmnopq")

    result = await secret_keys(None, words, SecretKeysCfg(threshold="balanced", custom_regex=None))

    assert result.tripwire_triggered is False  # noqa: S101
    assert result.info["detected_secrets"] == []  # noqa: S101


def test_secret_keys_limits_capture_bearing_custom_patterns() -> None:
    """Bound patterns that cannot use the combined boolean matcher."""
    patterns = [rf"^([A-Z]{{5}})x{index}$" for index in range(17)]

    with pytest.raises(ValueError, match="At most 16 custom regex patterns with capture groups"):
        SecretKeysCfg(threshold="balanced", custom_regex=patterns)
