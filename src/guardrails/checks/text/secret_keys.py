"""Secret key detection guardrail module.

This module provides functions and configuration for detecting potential API keys,
secrets, and credentials in text. It includes entropy and diversity checks, pattern
recognition, and a guardrail check_fn for runtime enforcement. File extensions and
URLs are optionally excluded, and custom detection criteria are supported.

Classes:
    SecretKeysCfg: Pydantic configuration for specifying secret key detection rules.

Functions:
    secret_keys: Async guardrail function for secret key detection.

Configuration Parameters:
    `threshold` (str): Detection sensitivity level. One of:

    - "strict": Most sensitive, may have more false positives
    - "balanced": Default setting, balanced between sensitivity and specificity
    - "permissive": Least sensitive, may have more false negatives

    `custom_regex` (list[str] | None): Optional list of custom regex patterns to check for secrets.
        If provided, these patterns will be used in addition to the default checks.
        Patterns with unsafe backtracking repetition are rejected.

Constants:
    COMMON_KEY_PREFIXES: Common prefixes used in secret keys.
    ALLOWED_EXTENSIONS: File extensions to ignore when strict_mode is False.

Examples:
```python
    >>> cfg = SecretKeysCfg(
    ...     threshold="balanced",
    ...     custom_regex=["my-custom-[a-zA-Z0-9]{32}", "internal-[a-zA-Z0-9]{16}-key"]
    ... )
    >>> result = await secret_keys(None, "my-custom-abc123xyz98765", cfg)
    >>> result.tripwire_triggered
    True
```
"""

from __future__ import annotations

import asyncio
import math
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections import Counter
from ipaddress import ip_address
from re import _constants as re_constants, _parser as re_parser  # type: ignore[attr-defined]
from typing import Any, NamedTuple, TypedDict
from urllib.parse import unquote, unquote_plus

from pydantic import BaseModel, ConfigDict, Field, field_validator

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["secret_keys"]


class SecretCfg(TypedDict, total=False):
    """Threshold settings used to classify candidate secrets.

    Args:
        strict_mode: Whether allowed-pattern exemptions are disabled.
        min_length: Minimum candidate length.
        min_diversity: Minimum character-class diversity.
        min_entropy: Minimum Shannon entropy.
    """

    strict_mode: bool
    min_length: int
    min_diversity: int
    min_entropy: float


class _EmbeddedCandidateGroup(NamedTuple):
    """Candidates produced by alternate component interpretations.

    Args:
        primary_index: Index of the preferred whole-component candidate.
        fallback_indexes: Indexes produced by definite parser views.
        ambiguous_suffixes: Decoded suffixes used to resolve path ambiguity.
        preserve_occurrence: Whether the primary represents a distinct source field.
    """

    primary_index: int
    fallback_indexes: tuple[int, ...]
    ambiguous_suffixes: tuple[str, ...] | None = None
    preserve_occurrence: bool = False


class _EmbeddedScanStatus(NamedTuple):
    """Completion state for one embedded-token scan.

    Args:
        structural_overflow: Whether bounded parser traversal was incomplete.
        custom_incomplete: Whether only custom-regex candidate work was truncated.
    """

    structural_overflow: bool
    custom_incomplete: bool


# Define common key prefixes
COMMON_KEY_PREFIXES = (
    "key-",
    "sk-",
    "sk_",
    "pk_",
    "pk-",
    "ghp_",
    "AKIA",
    "xox",
    "SG.",
    "hf_",
    "api-",
    "apikey-",
    "token-",
    "secret-",
    "SHA:",
    "Bearer ",
)

# Define allowed file extensions
ALLOWED_EXTENSIONS = (
    # Common file extensions
    ".py",
    ".js",
    ".html",
    ".css",
    ".json",
    ".md",
    ".txt",
    ".csv",
    ".xml",
    ".yaml",
    ".yml",
    ".ini",
    ".conf",
    ".config",
    ".log",
    ".sql",
    ".sh",
    ".bat",
    ".dll",
    ".so",
    ".dylib",
    ".jar",
    ".war",
    ".php",
    ".rb",
    ".go",
    ".rs",
    ".ts",
    ".jsx",
    ".vue",
    ".cpp",
    ".c",
    ".h",
    ".cs",
    ".fs",
    ".vb",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
)

MIN_EMBEDDED_SCAN_BUDGET = 16_384
EMBEDDED_SCAN_BUDGET_MULTIPLIER = 4
MAX_CUSTOM_CANDIDATE_RANGES = 262_144
MAX_CUSTOM_CANDIDATE_CHARACTERS = 134_217_728
MAX_CUSTOM_REGEX_EVALUATIONS = 8_388_608
MAX_CUSTOM_PRIORITY_REGEX_EVALUATIONS = 1_048_576
MAX_COMBINED_CUSTOM_REGEX_PATTERNS = 512
MAX_CAPTURE_CUSTOM_REGEX_PATTERNS = 16
MAX_CUSTOM_PRIORITY_BOUNDARIES = 2
MAX_CUSTOM_NEARBY_BOUNDARIES = 32
MAX_CUSTOM_TERMINAL_BOUNDARIES = 32
MAX_RETAINED_CUSTOM_CANDIDATES = 8_192
MAX_RETAINED_CUSTOM_CHARACTERS = 8_388_608
MAX_PRESENTATION_DEPTH = 16
MAX_PARSED_REGEX_WIDTH = int(re_parser.MAXWIDTH)
ASCII_URI_COMPONENT_SAFE = frozenset("-._~%")
NON_FORCING_COMPONENT_BOUNDARIES = frozenset("/\\:;=&+?#@")
URI_SUBDELIMITERS = frozenset("!$&'()*+,;=")
ENCLOSING_BOUNDARY_PAIRS = frozenset(
    {
        ("(", ")"),
        ("[", "]"),
        ("{", "}"),
        ("<", ">"),
        ('"', '"'),
        ("'", "'"),
        ("`", "`"),
        ("“", "”"),
        ("‘", "’"),
        ("«", "»"),
        ("‹", "›"),
        ("（", "）"),
        ("【", "】"),
        ("「", "」"),
        ("『", "』"),
        ("〈", "〉"),
        ("《", "》"),
        ("〔", "〕"),
    }
)


def _trim_trailing_prose(value: str) -> str:
    """Remove unbalanced trailing prose punctuation from a token.

    Args:
        value: URL- or file-shaped token to normalize.

    Returns:
        The token with trailing prose punctuation removed.
    """
    closing_pairs = {closing: opening for opening, closing in ENCLOSING_BOUNDARY_PAIRS if opening != closing}
    decoded_value = unquote(value)
    delimiter_counts = {delimiter: 0 for pair in closing_pairs.items() for delimiter in pair}
    for character in decoded_value:
        if character in delimiter_counts:
            delimiter_counts[character] += 1
    end = len(value)
    while end:
        character = value[end - 1]
        category = unicodedata.category(character)
        if not (character in ".,;:!?)]}" or category in {"Pe", "Pf"} or (not character.isascii() and category.startswith("P"))):
            break
        if character in closing_pairs:
            opener = closing_pairs[character]
            if delimiter_counts[opener] >= delimiter_counts[character]:
                break
            delimiter_counts[character] -= 1
        end -= 1
    return value[:end]


def _normalize_allowed_token(value: str) -> str:
    """Remove outer prose or Markdown wrappers from an allowed token.

    Args:
        value: Raw whitespace-delimited token.

    Returns:
        The normalized URL, file path, or Markdown link target.
    """
    normalized = value
    while normalized:
        previous = normalized
        markdown_target = normalized.rfind("](") if normalized.endswith(")") else -1
        if markdown_target >= 0 and markdown_target + 2 < len(normalized) - 1:
            normalized = normalized[markdown_target + 2 : -1]
            continue

        start = 0
        end = len(normalized)
        while start + 1 < end and (
            (normalized[start], normalized[end - 1]) in ENCLOSING_BOUNDARY_PAIRS
            or (normalized[start] == normalized[end - 1] and normalized[start] in "*_~`")
        ):
            start += 1
            end -= 1
        if start:
            normalized = normalized[start:end]
            continue

        normalized = _trim_trailing_prose(normalized)
        if normalized != previous:
            continue

        for marker in ("*", "_", "~", "`"):
            leading_run = len(normalized) - len(normalized.lstrip(marker))
            trailing_run = len(normalized) - len(normalized.rstrip(marker))
            if leading_run and trailing_run:
                normalized = normalized[min(leading_run, trailing_run) : -min(leading_run, trailing_run)]
                break
        if normalized == previous:
            normalized = normalized.lstrip("![](){}<>\"'“‘«‹（【「『〈《〔")
        if normalized != previous:
            continue
        break
    return normalized


CONFIGS: dict[str, SecretCfg] = {
    "strict": {
        "min_length": 10,
        "min_entropy": 3.5,
        "min_diversity": 2,
        "strict_mode": True,
    },
    "balanced": {
        "min_length": 15,
        "min_entropy": 3.8,
        "min_diversity": 3,
        "strict_mode": False,
    },
    "permissive": {
        "min_length": 20,
        "min_entropy": 4.0,
        "min_diversity": 3,
        "strict_mode": False,
    },
}


def _normalized_re_characters(character: int, flags: int) -> frozenset[int] | None:
    """Expand one literal for conservative alphabet comparisons.

    Args:
        character: Parsed Unicode code point.
        flags: Effective regular-expression flags.

    Returns:
        Comparable code points, or ``None`` when case folding cannot be
        represented conservatively.
    """
    if not flags & re.IGNORECASE:
        return frozenset({character})
    if character > 127:
        return None
    literal = chr(character)
    return frozenset({ord(literal.lower()), ord(literal.upper())})


def _subpattern_flags(argument: Any, flags: int) -> int:
    """Apply scoped flags from one parsed subpattern.

    Args:
        argument: Parsed ``SUBPATTERN`` payload.
        flags: Flags effective in the containing expression.

    Returns:
        Flags effective inside the subpattern.
    """
    return int((flags | argument[1]) & ~argument[2])


def _parsed_repeat_alphabet(parsed: Any, flags: int) -> frozenset[int] | None:
    """Return the finite character alphabet consumed by a parsed expression.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.
        flags: Flags effective in the parsed subpattern.

    Returns:
        The finite character set, or ``None`` when it cannot be proven.
    """
    characters: set[int] = set()
    for operation, argument in parsed:
        if operation is re_constants.LITERAL:
            normalized = _normalized_re_characters(argument, flags)
            if normalized is None:
                return None
            characters.update(normalized)
        elif operation is re_constants.IN:
            for item_operation, item_argument in argument:
                if item_operation is re_constants.LITERAL:
                    normalized = _normalized_re_characters(item_argument, flags)
                    if normalized is None:
                        return None
                    characters.update(normalized)
                elif item_operation is re_constants.RANGE and item_argument[1] - item_argument[0] <= 255:
                    for character in range(item_argument[0], item_argument[1] + 1):
                        normalized = _normalized_re_characters(character, flags)
                        if normalized is None:
                            return None
                        characters.update(normalized)
                else:
                    return None
        elif operation is re_constants.SUBPATTERN:
            nested = _parsed_repeat_alphabet(argument[-1], _subpattern_flags(argument, flags))
            if nested is None:
                return None
            characters.update(nested)
        elif operation in {
            re_constants.MAX_REPEAT,
            re_constants.MIN_REPEAT,
            re_constants.POSSESSIVE_REPEAT,
        }:
            nested = _parsed_repeat_alphabet(argument[2], flags)
            if nested is None:
                return None
            characters.update(nested)
        elif operation in {re_constants.AT, re_constants.ASSERT, re_constants.ASSERT_NOT}:
            continue
        else:
            return None
    return frozenset(characters)


def _parsed_first_alphabet(parsed: Any, flags: int) -> tuple[frozenset[int] | None, bool]:
    """Return possible first characters and nullability for a subpattern.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.
        flags: Flags effective in the parsed subpattern.

    Returns:
        A finite first-character set when provable, plus whether the
        subpattern can match an empty string.
    """
    characters: set[int] = set()
    for operation, argument in parsed:
        if operation in {re_constants.AT, re_constants.ASSERT, re_constants.ASSERT_NOT}:
            continue
        if operation is re_constants.LITERAL:
            normalized = _normalized_re_characters(argument, flags)
            if normalized is None:
                return None, False
            characters.update(normalized)
            return frozenset(characters), False
        if operation is re_constants.IN:
            alphabet = _parsed_repeat_alphabet([(operation, argument)], flags)
            if alphabet is None:
                return None, False
            characters.update(alphabet)
            return frozenset(characters), False
        if operation in {re_constants.SUBPATTERN, re_constants.ATOMIC_GROUP}:
            nested = argument[-1] if operation is re_constants.SUBPATTERN else argument
            nested_flags = _subpattern_flags(argument, flags) if operation is re_constants.SUBPATTERN else flags
            alphabet, nullable = _parsed_first_alphabet(nested, nested_flags)
            if alphabet is None:
                return None, nullable
            characters.update(alphabet)
            if not nullable:
                return frozenset(characters), False
            continue
        if operation is re_constants.BRANCH:
            nullable = False
            for branch in argument[1]:
                alphabet, branch_nullable = _parsed_first_alphabet(branch, flags)
                if alphabet is None:
                    return None, branch_nullable
                characters.update(alphabet)
                nullable = nullable or branch_nullable
            if not nullable:
                return frozenset(characters), False
            continue
        if operation in {
            re_constants.MAX_REPEAT,
            re_constants.MIN_REPEAT,
            re_constants.POSSESSIVE_REPEAT,
        }:
            minimum, _maximum, child = argument
            alphabet, nullable = _parsed_first_alphabet(child, flags)
            if alphabet is None:
                return None, minimum == 0 or nullable
            characters.update(alphabet)
            if minimum and not nullable:
                return frozenset(characters), False
            continue
        return None, False
    return frozenset(characters), True


def _parsed_has_disjoint_iteration_boundary(parsed: Any, flags: int) -> bool:
    """Check whether a repeated child has a distinct terminal alphabet.

    Args:
        parsed: Parsed repeated subpattern.
        flags: Flags effective in the repeated subpattern.

    Returns:
        True when the child's terminal token cannot begin its next iteration.
    """
    first_alphabet, nullable = _parsed_first_alphabet(parsed, flags)
    if first_alphabet is None or nullable:
        return False
    for operation, argument in reversed(parsed):
        if operation in {re_constants.AT, re_constants.ASSERT, re_constants.ASSERT_NOT}:
            continue
        if operation in {re_constants.LITERAL, re_constants.IN}:
            terminal_alphabet = _parsed_repeat_alphabet([(operation, argument)], flags)
        elif operation in {
            re_constants.MAX_REPEAT,
            re_constants.MIN_REPEAT,
            re_constants.POSSESSIVE_REPEAT,
        }:
            terminal_alphabet = _parsed_repeat_alphabet(argument[2], flags)
        elif operation is re_constants.SUBPATTERN:
            terminal_alphabet = _parsed_repeat_alphabet(argument[-1], _subpattern_flags(argument, flags))
        else:
            return False
        return terminal_alphabet is not None and bool(terminal_alphabet) and first_alphabet.isdisjoint(terminal_alphabet)
    return False


def _parsed_has_variable_zero_width(parsed: Any) -> bool:
    """Detect optional zero-width operations nested in a subpattern.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.

    Returns:
        True when a variable repetition can consume no characters.
    """
    for operation, argument in parsed:
        nested_patterns: tuple[Any, ...] = ()
        if operation in {
            re_constants.MAX_REPEAT,
            re_constants.MIN_REPEAT,
            re_constants.POSSESSIVE_REPEAT,
        }:
            minimum, maximum, child = argument
            if minimum != maximum and child.getwidth()[1] == 0:
                return True
            nested_patterns = (child,)
        elif operation is re_constants.SUBPATTERN:
            nested_patterns = (argument[-1],)
        elif operation is re_constants.BRANCH:
            nested_patterns = tuple(argument[1])
        elif operation in {re_constants.ASSERT, re_constants.ASSERT_NOT}:
            nested_patterns = (argument[-1],)
        elif operation is re_constants.ATOMIC_GROUP:
            nested_patterns = (argument,)
        elif operation is re_constants.GROUPREF_EXISTS:
            nested_patterns = tuple(branch for branch in argument[1:] if branch is not None)
        if any(_parsed_has_variable_zero_width(nested) for nested in nested_patterns):
            return True
    return False


def _parsed_has_variable_work_assertion(parsed: Any) -> bool:
    """Detect lookarounds whose work can grow with the remaining input.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.

    Returns:
        True when a nested assertion has variable width.
    """
    for operation, argument in parsed:
        nested_patterns: tuple[Any, ...] = ()
        if operation in {re_constants.ASSERT, re_constants.ASSERT_NOT}:
            assertion = argument[-1]
            minimum, maximum = assertion.getwidth()
            if minimum != maximum:
                return True
            nested_patterns = (assertion,)
        elif operation in {
            re_constants.MAX_REPEAT,
            re_constants.MIN_REPEAT,
            re_constants.POSSESSIVE_REPEAT,
        }:
            nested_patterns = (argument[2],)
        elif operation is re_constants.SUBPATTERN:
            nested_patterns = (argument[-1],)
        elif operation is re_constants.BRANCH:
            nested_patterns = tuple(argument[1])
        elif operation is re_constants.ATOMIC_GROUP:
            nested_patterns = (argument,)
        elif operation is re_constants.GROUPREF_EXISTS:
            nested_patterns = tuple(branch for branch in argument[1:] if branch is not None)
        if any(_parsed_has_variable_work_assertion(nested) for nested in nested_patterns):
            return True
    return False


def _parsed_ambiguous_branch_count(parsed: Any, flags: int) -> int:
    """Count branches whose alternatives cannot be distinguished safely.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.
        flags: Flags effective in the parsed subpattern.

    Returns:
        The number of branches with nullable, overlapping, or unknown starts.
    """
    count = 0
    for operation, argument in parsed:
        nested_patterns: tuple[tuple[Any, int], ...] = ()
        if operation is re_constants.BRANCH:
            branches = tuple(argument[1])
            first_sets: list[frozenset[int]] = []
            ambiguous = False
            for branch in branches:
                alphabet, nullable = _parsed_first_alphabet(branch, flags)
                if alphabet is None or nullable or any(not alphabet.isdisjoint(existing) for existing in first_sets):
                    ambiguous = True
                if alphabet is not None:
                    first_sets.append(alphabet)
            count += int(ambiguous)
            nested_patterns = tuple((branch, flags) for branch in branches)
        elif operation is re_constants.SUBPATTERN:
            nested_patterns = ((argument[-1], _subpattern_flags(argument, flags)),)
        elif operation in {
            re_constants.MAX_REPEAT,
            re_constants.MIN_REPEAT,
            re_constants.POSSESSIVE_REPEAT,
        }:
            nested_patterns = ((argument[2], flags),)
        elif operation in {re_constants.ASSERT, re_constants.ASSERT_NOT}:
            nested_patterns = ((argument[-1], flags),)
        elif operation is re_constants.ATOMIC_GROUP:
            nested_patterns = ((argument, flags),)
        elif operation is re_constants.GROUPREF_EXISTS:
            nested_patterns = tuple((branch, flags) for branch in argument[1:] if branch is not None)
        count += sum(_parsed_ambiguous_branch_count(nested, nested_flags) for nested, nested_flags in nested_patterns)
    return count


def _parsed_variable_decision_alphabet(
    operation: Any,
    argument: Any,
    flags: int,
) -> tuple[bool, bool, frozenset[int] | None]:
    """Describe one variable-width backtracking decision.

    Args:
        operation: Parsed ``re`` operation.
        argument: Operation payload.
        flags: Flags effective for the operation.

    Returns:
        Whether the operation is a variable-width decision, whether its width
        is unbounded, and its finite character alphabet when one is provable.
    """
    if operation in {re_constants.MAX_REPEAT, re_constants.MIN_REPEAT}:
        minimum, maximum, child = argument
        return maximum != minimum, maximum == re_constants.MAXREPEAT, _parsed_repeat_alphabet(child, flags)
    if operation is re_constants.SUBPATTERN:
        child = argument[-1]
        minimum, maximum = child.getwidth()
        return (
            minimum != maximum,
            maximum == MAX_PARSED_REGEX_WIDTH,
            _parsed_repeat_alphabet(child, _subpattern_flags(argument, flags)),
        )
    if operation is re_constants.BRANCH:
        branches = argument[1]
        widths = tuple(branch.getwidth() for branch in branches)
        variable = len(set(widths)) > 1 or any(minimum != maximum for minimum, maximum in widths)
        alphabet, _nullable = _parsed_first_alphabet([(operation, argument)], flags)
        return variable, any(maximum == MAX_PARSED_REGEX_WIDTH for _minimum, maximum in widths), alphabet
    return False, False, None


def _parsed_has_unsafe_repetition(parsed: Any, flags: int) -> bool:
    """Detect parsed repetition shapes with unbounded backtracking risk.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.
        flags: Flags effective in the parsed subpattern.

    Returns:
        True when the expression contains nested or overlapping repetition
        that is unsafe to run on untrusted input.
    """
    repeat_indexes: list[int] = []
    decision_run = 0
    previous_decision_unbounded = False
    previous_decision_alphabet: frozenset[int] | None = frozenset()
    for index, (operation, argument) in enumerate(parsed):
        is_decision, decision_unbounded, decision_alphabet = _parsed_variable_decision_alphabet(operation, argument, flags)
        if is_decision:
            overlaps = (
                decision_run == 0
                or previous_decision_alphabet is None
                or decision_alphabet is None
                or not previous_decision_alphabet.isdisjoint(decision_alphabet)
            )
            decision_run = decision_run + 1 if overlaps else 1
            if overlaps and decision_run > 1 and previous_decision_unbounded and decision_unbounded:
                return True
            previous_decision_unbounded = decision_unbounded
            previous_decision_alphabet = decision_alphabet
            if decision_run >= 8:
                return True
        elif operation not in {re_constants.AT, re_constants.ASSERT, re_constants.ASSERT_NOT}:
            decision_run = 0
            previous_decision_unbounded = False
            previous_decision_alphabet = frozenset()

        if operation in {re_constants.MAX_REPEAT, re_constants.MIN_REPEAT}:
            _minimum, maximum, child = argument
            child_minimum, child_maximum = child.getwidth()
            nested_ambiguity = _parsed_ambiguous_branch_count(child, flags)
            extension_is_atomic = len(child) == 1 and child[0][0] in {
                re_constants.ATOMIC_GROUP,
                re_constants.POSSESSIVE_REPEAT,
            }
            has_disjoint_boundary = _parsed_has_disjoint_iteration_boundary(child, flags)
            has_variable_zero_width = _parsed_has_variable_zero_width(child)
            if maximum > 8 and _parsed_has_variable_work_assertion(child):
                return True
            if (
                maximum == re_constants.MAXREPEAT
                and not extension_is_atomic
                and not has_disjoint_boundary
                and (child_minimum != child_maximum or nested_ambiguity or has_variable_zero_width)
            ):
                return True
            if maximum > 1 and not extension_is_atomic and not has_disjoint_boundary and child_minimum != child_maximum:
                return True
            if _parsed_has_unsafe_repetition(child, flags):
                return True
            if maximum > 8 and maximum != _minimum:
                repeat_indexes.append(index)
        elif operation is re_constants.SUBPATTERN:
            if _parsed_has_unsafe_repetition(argument[-1], _subpattern_flags(argument, flags)):
                return True
        elif operation is re_constants.BRANCH:
            if any(_parsed_has_unsafe_repetition(branch, flags) for branch in argument[1]):
                return True
        elif operation in {re_constants.ASSERT, re_constants.ASSERT_NOT}:
            if _parsed_has_unsafe_repetition(argument[-1], flags):
                return True
        elif operation is re_constants.ATOMIC_GROUP:
            if _parsed_has_unsafe_repetition(argument, flags):
                return True
        elif operation is re_constants.POSSESSIVE_REPEAT:
            if _parsed_has_unsafe_repetition(argument[2], flags):
                return True
        elif operation is re_constants.GROUPREF_EXISTS:
            if any(_parsed_has_unsafe_repetition(branch, flags) for branch in argument[1:] if branch is not None):
                return True

    for left_index, right_index in zip(repeat_indexes, repeat_indexes[1:], strict=False):
        left_alphabet = _parsed_repeat_alphabet(parsed[left_index][1][2], flags)
        right_alphabet = _parsed_repeat_alphabet(parsed[right_index][1][2], flags)
        if left_alphabet is not None and right_alphabet is not None and left_alphabet.isdisjoint(right_alphabet):
            continue
        has_barrier = False
        for operation, argument in parsed[left_index + 1 : right_index]:
            separator, nullable = _parsed_first_alphabet([(operation, argument)], flags)
            if nullable or not separator:
                continue
            if (left_alphabet is not None and separator.isdisjoint(left_alphabet)) or (
                right_alphabet is not None and separator.isdisjoint(right_alphabet)
            ):
                has_barrier = True
                break
        if not has_barrier:
            return True
    return False


def _parsed_has_variable_width_backreference(
    parsed: Any,
    group_widths: list[tuple[int, int] | None],
) -> bool:
    """Detect backreferences whose captured text has variable width.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.
        group_widths: Width bounds indexed by capture-group number.

    Returns:
        True when a backreference can trigger width-dependent backtracking.
    """
    for operation, argument in parsed:
        nested_patterns: tuple[Any, ...] = ()
        if operation is re_constants.GROUPREF:
            width = group_widths[argument]
            if width is None or width[0] != width[1]:
                return True
        elif operation in {
            re_constants.MAX_REPEAT,
            re_constants.MIN_REPEAT,
            re_constants.POSSESSIVE_REPEAT,
        }:
            nested_patterns = (argument[2],)
        elif operation is re_constants.SUBPATTERN:
            nested_patterns = (argument[-1],)
        elif operation is re_constants.BRANCH:
            nested_patterns = tuple(argument[1])
        elif operation in {re_constants.ASSERT, re_constants.ASSERT_NOT}:
            nested_patterns = (argument[-1],)
        elif operation is re_constants.ATOMIC_GROUP:
            nested_patterns = (argument,)
        elif operation is re_constants.GROUPREF_EXISTS:
            nested_patterns = tuple(branch for branch in argument[1:] if branch is not None)
        if any(_parsed_has_variable_width_backreference(nested, group_widths) for nested in nested_patterns):
            return True
    return False


def _custom_regex_has_backtracking_risk(pattern: re.Pattern[str]) -> bool:
    """Check whether a custom expression can backtrack excessively.

    Args:
        pattern: Compiled custom expression.

    Returns:
        True when the parsed expression contains a high-risk repetition
        structure that cannot be run safely on attacker-controlled values.
    """
    parsed = re_parser.parse(pattern.pattern, pattern.flags)
    return (
        _parsed_has_unsafe_repetition(parsed, pattern.flags)
        or _parsed_has_variable_width_backreference(parsed, parsed.state.groupwidths)
        or _parsed_ambiguous_branch_count(parsed, pattern.flags) >= 8
    )


def _trim_verbose_trailing_trivia(source: str) -> str:
    """Remove trailing whitespace and comments ignored by ``re.VERBOSE``.

    Args:
        source: Regular-expression source compiled in verbose mode.

    Returns:
        The source through its last syntactically significant character.
    """
    if "(?-x" in source:
        return source
    in_class = False
    escaped = False
    in_comment = False
    significant_end = 0
    for index, character in enumerate(source):
        if in_comment:
            if character in "\r\n":
                in_comment = False
            continue
        if escaped:
            escaped = False
            significant_end = index + 1
            continue
        if character == "\\":
            escaped = True
            significant_end = index + 1
        elif in_class:
            significant_end = index + 1
            if character == "]":
                in_class = False
        elif character == "[":
            in_class = True
            significant_end = index + 1
        elif character == "#":
            in_comment = True
        elif not character.isspace():
            significant_end = index + 1
    return source[:significant_end]


def _parsed_ends_with_terminal_anchor(parsed: Any) -> bool:
    r"""Return whether a parsed expression ends inside an absolute-end anchor.

    Args:
        parsed: Parsed ``re`` subpattern to inspect.

    Returns:
        True when the final consuming path ends with ``$``, ``\\Z``, or ``\\z``.
    """
    if not parsed:
        return False
    operation, argument = parsed[-1]
    if operation is re_constants.AT:
        return argument in {re_constants.AT_END, re_constants.AT_END_STRING}
    if operation is re_constants.SUBPATTERN:
        return _parsed_ends_with_terminal_anchor(argument[-1])
    if operation is re_constants.ATOMIC_GROUP:
        return _parsed_ends_with_terminal_anchor(argument)
    return False


def _terminal_unanchored_source(source: str, flags: int) -> str | None:
    """Remove a parsed terminal anchor while preserving surrounding groups.

    Args:
        source: One top-level regular-expression alternative.
        flags: Flags used to compile the original expression.

    Returns:
        Equivalent proposal source without its terminal anchor, or ``None``.
    """
    try:
        parsed = re_parser.parse(source, flags)
    except re.error:
        return None
    if not _parsed_ends_with_terminal_anchor(parsed):
        return None

    anchors: list[tuple[int, int]] = []
    in_class = False
    escaped = False
    in_comment = False
    for index, character in enumerate(source):
        if in_comment:
            if character in "\r\n":
                in_comment = False
            continue
        if escaped:
            escaped = False
            continue
        if character == "\\":
            if index + 1 < len(source) and source[index + 1] in "Zz":
                anchors.append((index, index + 2))
            escaped = True
        elif in_class:
            if character == "]":
                in_class = False
        elif character == "[":
            in_class = True
        elif flags & re.VERBOSE and character == "#":
            in_comment = True
        elif character == "$":
            anchors.append((index, index + 1))
    if not anchors:
        return None
    start, end = anchors[-1]
    return source[:start] + source[end:]


class SecretKeysCfg(BaseModel):
    """Configuration for secret key and credential detection.

    This configuration allows fine-tuning of secret detection sensitivity and
    adding custom patterns for project-specific secrets.

    Attributes:
        threshold (str): Detection sensitivity level. One of:

            - "strict": Most sensitive, may have more false positives
            - "balanced": Default setting, balanced between sensitivity and specificity
            - "permissive": Least sensitive, may have more false negatives

        custom_regex (list[str] | None): Optional list of custom regex patterns to check for secrets.
            If provided, these patterns will be used in addition to the default checks.
            Patterns with unsafe backtracking repetition are rejected.
    """

    threshold: str = Field(
        "balanced",
        description="Threshold level to use (strict, balanced, or permissive)",
        pattern="^(strict|balanced|permissive)$",
    )
    custom_regex: list[str] | None = Field(
        None, description="Optional list of custom regex patterns to check for secrets. Each pattern must be a valid regex string."
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("custom_regex")
    def validate_custom_regex(cls, v):
        """Validate that all custom regular expressions compile.

        Args:
            cls: Configuration model class supplied by Pydantic.
            v: Optional configured expression list.

        Returns:
            The validated expression list.

        Raises:
            ValueError: If an item is not a string or cannot compile.
        """
        if v is not None:
            capture_patterns = 0
            for pattern in v:
                if not isinstance(pattern, str):
                    raise ValueError("Each regex pattern must be a string")
                try:
                    compiled = re.compile(pattern)
                except re.error as exc:
                    raise ValueError(f"Invalid regex pattern '{pattern!r}': {exc}") from exc
                if _custom_regex_has_backtracking_risk(compiled):
                    raise ValueError(f"Unsafe regex pattern '{pattern!r}': its structure can cause excessive backtracking")
                capture_patterns += int(bool(compiled.groups))
            if capture_patterns > MAX_CAPTURE_CUSTOM_REGEX_PATTERNS:
                raise ValueError(f"At most {MAX_CAPTURE_CUSTOM_REGEX_PATTERNS} custom regex patterns with capture groups are supported")
        return v


def _entropy(s: str) -> float:
    """Calculate the Shannon entropy of a string.

    Args:
        s (str): The input string.

    Returns:
        float: The Shannon entropy of the string.
    """
    counts: dict[str, int] = {}
    for c in s:
        counts[c] = counts.get(c, 0) + 1

    return -sum((n := counts[c]) / len(s) * math.log2(n / len(s)) for c in counts)


def _char_diversity(s: str) -> int:
    """Count the number of character types present in a string.

    Returns the sum of booleans for presence of lowercase, uppercase, digits, and specials.

    Args:
        s (str): Input string.

    Returns:
        int: Number of unique character types in the string (1-4).
    """
    return sum(
        (
            any(c.islower() for c in s),
            any(c.isupper() for c in s),
            any(c.isdigit() for c in s),
            any(not c.isalnum() for c in s),
        )
    )


def _contains_allowed_pattern(text: str) -> bool:
    """Return True if text contains allowed URL or file extension patterns.

    Args:
        text (str): Input string.

    Returns:
        bool: True if text matches URL or allowed extension; otherwise False.
    """
    # Simple regex for URLs
    url_pattern = re.compile(r"https?://[^\s]+", re.IGNORECASE)
    if url_pattern.search(text):
        return True

    return text.lower().endswith(ALLOWED_EXTENSIONS)


def _strip_allowed_extension(value: str) -> str:
    """Remove one recognized file extension from a value.

    Args:
        value: Candidate value that may end in an allowed extension.

    Returns:
        The value without one trailing allowed extension, if present.
    """
    lowered = value.lower()
    for extension in ALLOWED_EXTENSIONS:
        if lowered.endswith(extension):
            return value[: -len(extension)]
    return value


def _matches_custom_pattern(value: str, custom_regex: list[str] | None) -> bool:
    """Check whether a value matches a configured secret pattern.

    Args:
        value: Candidate value to test.
        custom_regex: Optional configured regular expressions.

    Returns:
        True when any configured expression matches the value.
    """
    return bool(custom_regex and any(re.match(pattern, value) for pattern in custom_regex))


def _combined_custom_matchers(patterns: tuple[re.Pattern[str], ...]) -> tuple[re.Pattern[str], ...]:
    """Combine capture-free expressions for boolean matching.

    Args:
        patterns: Individually compiled custom expressions.

    Returns:
        Boolean-equivalent matchers. Expressions with capture groups remain
        individual so group numbering and references cannot change.
    """
    grouped: dict[int, list[str]] = {}
    individual: list[re.Pattern[str]] = []
    for pattern in patterns:
        source = re.sub(r"^(?:\(\?[aiLmsux]+\))+", "", pattern.pattern)
        if pattern.groups:
            individual.append(pattern)
        else:
            grouped.setdefault(pattern.flags, []).append(source)

    matchers = list(individual)
    for flags, compatible in grouped.items():
        for offset in range(0, len(compatible), MAX_COMBINED_CUSTOM_REGEX_PATTERNS):
            chunk = compatible[offset : offset + MAX_COMBINED_CUSTOM_REGEX_PATTERNS]
            separator = "\n" if flags & re.VERBOSE else ""
            try:
                matchers.append(re.compile("|".join(f"(?:{source}{separator})" for source in chunk), flags))
            except re.error:
                return patterns
    return tuple(matchers)


def _has_known_prefix(value: str) -> bool:
    """Check whether a value starts with a known credential prefix.

    Args:
        value: Candidate value to inspect.

    Returns:
        True when the value starts with a known credential prefix.
    """
    return any(value.startswith(prefix) for prefix in COMMON_KEY_PREFIXES)


def _literal_prefix_offsets(value: str, prefixes: tuple[str, ...]) -> tuple[int, ...]:
    """Find bounded first and last occurrences of literal prefixes.

    Args:
        value: Decoded syntax component to inspect.
        prefixes: Literal prefixes that can start a credential candidate.

    Returns:
        Source-ordered offsets for the first and last occurrence of each prefix.
    """
    offsets: set[int] = set()
    for prefix in prefixes:
        if not prefix:
            continue
        first = -1
        last = -1
        search_from = 0
        while (offset := value.find(prefix, search_from)) >= 0:
            if offset == 0 or not value[offset - 1].isalnum():
                first = offset if first < 0 else first
                last = offset
            search_from = offset + 1
        if first >= 0:
            offsets.add(first)
            offsets.add(last)
    return tuple(sorted(offsets))


def _component_candidate_boundaries(value: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return bounded candidate starts and ends for one component.

    Word starts and punctuation offsets may begin a custom-regex candidate.
    Ends remain lexical so adding punctuation starts does not also multiply the
    number of range endpoints. Direct regex matches may contribute a verified
    punctuation end separately.

    Args:
        value: One decoded URL, file, authority, or parameter component.

    Returns:
        Ordered candidate start offsets and lexical end offsets.
    """
    word_spans = tuple(re.finditer(r"[^\W_]+", value))
    starts = {match.start() for match in word_spans}
    starts.update(index for index, character in enumerate(value) if not character.isalnum())
    return tuple(sorted(starts)), tuple(match.end() for match in word_spans)


def _retain_longest_custom_range(
    findings: dict[int, str],
    start_offset: int,
    candidate: str,
    retained_count: int,
    retained_characters: int,
) -> tuple[bool, int, int]:
    """Retain the longest custom match for one component start.

    Args:
        findings: Matches retained for the current decoded component.
        start_offset: Candidate start within the decoded component.
        candidate: Matching candidate beginning at ``start_offset``.
        retained_count: Candidates retained across the token scan.
        retained_characters: Candidate characters retained across the scan.

    Returns:
        Whether retention succeeded and the updated count/character totals.
    """
    existing = findings.get(start_offset)
    if existing is not None and len(existing) >= len(candidate):
        return True, retained_count, retained_characters
    count_delta = int(existing is None)
    character_delta = len(candidate) - len(existing or "")
    if retained_count + count_delta > MAX_RETAINED_CUSTOM_CANDIDATES or retained_characters + character_delta > MAX_RETAINED_CUSTOM_CHARACTERS:
        return False, retained_count, retained_characters
    findings[start_offset] = candidate
    return True, retained_count + count_delta, retained_characters + character_delta


def _ordered_custom_range_findings(findings: dict[int, str]) -> tuple[str, ...]:
    """Return retained component matches in source order.

    Args:
        findings: Matches keyed by their decoded-component start offset.

    Returns:
        Candidate values ordered by source position.
    """
    return tuple(candidate for _, candidate in sorted(findings.items()))


def _ends_at_component_boundary(value: str, offset: int) -> bool:
    """Check whether an offset ends a complete component fragment.

    Args:
        value: Decoded syntax component containing the candidate.
        offset: Exclusive end offset of a regex match.

    Returns:
        True when the match reaches the component end or is followed by a
        non-word boundary.
    """
    return offset == len(value) or (0 <= offset < len(value) and not (value[offset].isalnum() or value[offset] == "_"))


def _bounded_component_match_end(value: str, offset: int, lexical_ends: tuple[int, ...]) -> int:
    """Extend a prefix match to a complete component fragment.

    Args:
        value: Decoded syntax component containing the match.
        offset: Exclusive end offset of the direct regex match.
        lexical_ends: Ordered lexical end offsets in the component.

    Returns:
        The direct end when it is already bounded, otherwise the next lexical
        end or the component end.
    """
    if _ends_at_component_boundary(value, offset):
        return offset
    next_end = bisect_right(lexical_ends, offset)
    return lexical_ends[next_end] if next_end < len(lexical_ends) else len(value)


def _embedded_known_prefix_suffixes(value: str) -> tuple[str, ...]:
    """Extract bounded suffixes beginning at known credential prefixes.

    The component has already been separated at literal path and parameter
    boundaries. Keeping the remainder intact preserves punctuation that may be
    part of a secret without joining unrelated route components.

    Args:
        value: One decoded syntax component.

    Returns:
        Known-prefix suffixes in source order.
    """
    findings: list[str] = []
    for offset in _literal_prefix_offsets(value, COMMON_KEY_PREFIXES):
        candidate = _strip_allowed_extension(value[offset:])
        if candidate and _has_known_prefix(candidate):
            findings.append(candidate)
    return tuple(dict.fromkeys(findings))


def _structural_index_payload(segment: str) -> str | None:
    """Return the payload of one complete bracket-style component.

    Args:
        segment: Decoded path component that may use bracket syntax.

    Returns:
        The bracket payload, including an empty payload, or ``None``.
    """
    if len(segment) >= 2 and segment.startswith("[") and segment.endswith("]"):
        return segment[1:-1]
    return None


def _associated_value_indexes(segments: tuple[str, ...], label_index: int) -> tuple[int, ...]:
    """Find ambiguous bracket payloads and the value after a label.

    Numeric components are unambiguous container indexes. Bracket components
    are ambiguous: their payload may itself be the value, while a later
    component may still be the container's associated value.

    Args:
        segments: Definite decoded path components.
        label_index: Index of the credential-bearing label.

    Returns:
        Candidate component indexes in source order.
    """
    indexes: list[int] = []
    value_index = label_index + 1
    while value_index < len(segments):
        segment = segments[value_index]
        if segment.isdigit():
            value_index += 1
            continue
        if _structural_index_payload(segment) is not None:
            indexes.append(value_index)
            value_index += 1
            continue
        indexes.append(value_index)
        break
    return tuple(indexes)


def _is_sensitive_parameter(name: str) -> bool:
    """Check whether a URL parameter conventionally carries credentials.

    Args:
        name: URL query or fragment parameter name.

    Returns:
        True when the normalized name indicates a credential-bearing field.
    """

    def normalized_variants(value: str) -> tuple[str, str]:
        """Normalize one field name and remove a trailing index/version.

        Args:
            value: Raw field-name component.

        Returns:
            The normalized value and its unversioned form.
        """
        separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value.strip())
        normalized = re.sub(r"[^a-zA-Z0-9]+", "_", separated).strip("_").lower()
        return normalized, re.sub(r"(?:_v?\d+)+$|\d+$", "", normalized)

    metadata_properties = {
        "algorithm",
        "bucket",
        "documentation",
        "endpoint",
        "expiration",
        "file",
        "header",
        "id",
        "issuer",
        "lifetime",
        "metadata",
        "method",
        "name",
        "policy",
        "provider",
        "rotation",
        "scope",
        "server",
        "store",
        "type",
        "uri",
        "url",
        "usage",
    }
    exact_names = {
        "apikey",
        "api_key",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "key",
        "passwd",
        "password",
        "secret",
        "token",
    }
    payload_suffixes = ("value", "data", "raw", "code", "confirmation", "confirm", "repeat")
    public_key_suffixes = ("public_key", "verification_key", "verify_key")

    def normalized_sensitive(value: str) -> bool:
        """Classify one flat field name without substring matching.

        Args:
            value: Raw field name or structural field component.

        Returns:
            Whether the complete normalized name denotes a credential value.
        """
        normalized = normalized_variants(value)[1]
        components = [component for component in normalized.split("_") if component]
        while components and (components[-1] in payload_suffixes or re.fullmatch(r"v?\d+", components[-1])):
            components.pop()
        normalized = "_".join(components)
        if any(normalized == suffix or normalized.endswith(f"_{suffix}") for suffix in public_key_suffixes):
            return False
        return normalized in exact_names or normalized.endswith(("_key", "_password", "_secret", "_token"))

    structural_components = tuple(component for component in re.split(r"[.\[\]]+", name) if component)
    if len(structural_components) > 1:
        meaningful_components = list(structural_components)
        while len(meaningful_components) > 1:
            normalized_terminal = normalized_variants(meaningful_components[-1])[1]
            if normalized_terminal in {"value", "data", "raw"} or not normalized_terminal:
                meaningful_components.pop()
                continue
            if re.fullmatch(r"(?:v?\d+)", normalized_terminal):
                meaningful_components.pop()
                continue
            break
        terminal = normalized_variants(meaningful_components[-1])[1]
        terminal_parts = tuple(part for part in terminal.split("_") if part)
        if terminal_parts and terminal_parts[-1] in metadata_properties:
            return False
        return any(normalized_sensitive(component) for component in structural_components)
    return normalized_sensitive(name)


def _extract_embedded_secret_candidates(
    token: str,
    custom_regex: list[str] | None = None,
) -> tuple[tuple[str, ...], _EmbeddedScanStatus, tuple[_EmbeddedCandidateGroup, ...]]:
    """Extract embedded candidates and report bounded-scan exhaustion.

    Args:
        token: Whitespace-delimited token that matched an allowed pattern.
        custom_regex: Optional configured regular expressions.

    Returns:
        Ordered candidate occurrences, independent parser/custom completion
        state, and same-component groups used for canonical output selection.
    """
    candidates: list[str] = []
    candidate_groups: list[_EmbeddedCandidateGroup] = []
    # Preserve the broad URL span used by the exemption check until another
    # HTTP(S) scheme begins. Component parsing and presentation normalization
    # decide which intervening punctuation is syntax.
    url_pattern = re.compile(r"https?://(?:(?!https?://)[^\s])+", re.IGNORECASE)

    def is_scheme_relative_url(value: str) -> bool:
        """Return whether a decoded value begins with a URL authority.

        Args:
            value: Decoded parameter or path-component value.

        Returns:
            True when the value is a non-empty ``//authority`` reference.
        """
        return len(value) > 2 and value.startswith("//") and value[2] not in "/\\?#"

    def decode_uri_component(value: str, *, query_form: bool = False) -> tuple[str, frozenset[int]]:
        """Decode URI data while retaining percent-escape provenance.

        Args:
            value: Raw URI component text.
            query_form: Whether literal plus signs decode as spaces.

        Returns:
            The decoded value and offsets originating from percent escapes.
        """
        decoded_parts: list[str] = []
        encoded_offsets: set[int] = set()
        decoded_length = 0
        index = 0
        while index < len(value):
            match = re.match(r"(?:%[0-9A-Fa-f]{2})+", value[index:])
            if match is not None:
                decoded_part = unquote(match.group(0))
                decoded_parts.append(decoded_part)
                encoded_offsets.update(range(decoded_length, decoded_length + len(decoded_part)))
                decoded_length += len(decoded_part)
                index += match.end()
                continue
            character = " " if query_form and value[index] == "+" else value[index]
            decoded_parts.append(character)
            decoded_length += len(character)
            index += 1
        return "".join(decoded_parts), frozenset(encoded_offsets)

    active_tokens: set[str] = set()
    scan_budget = max(MIN_EMBEDDED_SCAN_BUDGET, len(token) * EMBEDDED_SCAN_BUDGET_MULTIPLIER)
    custom_range_budget = max(MAX_CUSTOM_CANDIDATE_CHARACTERS, scan_budget * 16)
    scanned_characters = 0
    custom_range_count = 0
    custom_range_characters = 0
    priority_range_count = 0
    priority_range_characters = 0
    priority_regex_evaluations = 0
    retained_custom_count = 0
    retained_custom_characters = 0
    terminal_range_evaluations = 0
    custom_scan_truncated = False
    scan_overflow = False
    compiled_custom_regex = tuple(re.compile(pattern) for pattern in custom_regex or ())

    def top_level_alternatives(source: str) -> tuple[str, ...]:
        """Split an expression at ungrouped alternations.

        Args:
            source: Regular-expression source text.

        Returns:
            Source-ordered top-level alternatives.
        """
        depth = 0
        in_class = False
        escaped = False
        alternative_start = 0
        alternatives: list[str] = []
        for index, character in enumerate(source):
            if escaped:
                escaped = False
                continue
            if character == "\\":
                escaped = True
            elif character == "[":
                in_class = True
            elif character == "]" and in_class:
                in_class = False
            elif not in_class and character == "(":
                depth += 1
            elif not in_class and character == ")" and depth:
                depth -= 1
            elif not in_class and character == "|" and depth == 0:
                alternatives.append(source[alternative_start:index])
                alternative_start = index + 1
        alternatives.append(source[alternative_start:])
        return tuple(alternatives)

    custom_matchers = _combined_custom_matchers(compiled_custom_regex)
    terminal_range_patterns: list[re.Pattern[str]] = []
    for pattern in compiled_custom_regex:
        pattern_source = _trim_verbose_trailing_trivia(pattern.pattern) if pattern.flags & re.VERBOSE else pattern.pattern
        for source in top_level_alternatives(pattern_source):
            terminal_source = _terminal_unanchored_source(source, pattern.flags)
            if terminal_source is None:
                continue
            terminal_source = re.sub(r"^(?:\(\?[aiLmsux]+\))+", "", terminal_source)
            if terminal_source.startswith(r"\A"):
                terminal_source = terminal_source[2:]
            elif terminal_source.startswith("^"):
                terminal_source = terminal_source[1:]
            try:
                terminal_range_patterns.append(re.compile(terminal_source, pattern.flags))
            except re.error:
                continue
    custom_range_limit = min(
        MAX_CUSTOM_CANDIDATE_RANGES,
        MAX_CUSTOM_REGEX_EVALUATIONS // max(1, len(custom_matchers)),
    )

    def literal_custom_prefix(pattern: re.Pattern[str]) -> str | None:
        """Return a conservative fixed prefix for range-start pruning.

        Args:
            pattern: Compiled custom expression.

        Returns:
            A mandatory literal prefix, when one is provable.
        """
        if pattern.flags & re.VERBOSE:
            return None
        source = pattern.pattern
        while inline_flags := re.match(r"^\(\?[aiLmsux-]+\)", source):
            source = source[inline_flags.end() :]
        if source.startswith(r"\A"):
            source = source[2:]
        elif source.startswith("^"):
            source = source[1:]

        if len(top_level_alternatives(source)) > 1:
            return None

        # Only plain characters are a proof. In particular, an unescaped dot
        # is a wildcard and cannot constrain the candidate start.
        simple_lookahead = re.match(r"^\(\?=([A-Za-z0-9_/:-]+)\)", source)
        if simple_lookahead and source[simple_lookahead.end() : simple_lookahead.end() + 1] not in {"?", "*", "{"}:
            return simple_lookahead.group(1)
        single_literal_class = re.match(r"^\[([^\]\\])\]", source)
        if single_literal_class and source[single_literal_class.end() : single_literal_class.end() + 1] not in {
            "?",
            "*",
            "{",
        }:
            return single_literal_class.group(1)

        prefix: list[str] = []
        index = 0
        while index < len(source):
            character = source[index]
            if character == "\\":
                if index + 1 >= len(source) or source[index + 1].isalnum():
                    break
                prefix.append(source[index + 1])
                index += 2
                continue
            if character in "*+?{":
                if prefix:
                    prefix.pop()
                break
            if character in ".[$()}|^":
                break
            prefix.append(character)
            index += 1
        return "".join(prefix) or None

    custom_literal_prefixes = tuple(literal_custom_prefix(pattern) for pattern in compiled_custom_regex)
    custom_literal_prefix_matchers = tuple(
        re.compile(re.escape(prefix), re.IGNORECASE if pattern.flags & re.IGNORECASE else 0) if prefix is not None else None
        for pattern, prefix in zip(compiled_custom_regex, custom_literal_prefixes, strict=True)
    )

    def matches_custom_pattern(value: str) -> bool:
        """Match configured expressions without recompiling each range.

        Args:
            value: Candidate string to test.

        Returns:
            Whether any configured custom expression matches the value.
        """
        return any(pattern.match(value) for pattern in custom_matchers)

    def add_value_candidate(value: str, *, force: bool = False, encoded: bool = True) -> tuple[int, ...]:
        """Append an eligible value after at most one decoding pass.

        Args:
            value: Encoded query, fragment, or userinfo value.
            force: Whether the containing field is credential-bearing.
            encoded: Whether the current syntax layer still needs decoding.

        Returns:
            Candidate indexes appended for this component.
        """
        candidate = unquote(value) if encoded else value
        if candidate and (force or _has_known_prefix(candidate) or matches_custom_pattern(candidate)):
            candidates.append(candidate)
            return (len(candidates) - 1,)
        return ()

    def path_candidate_values(value: str) -> tuple[str, ...]:
        """Return eligible path variants without mutating scan state.

        Args:
            value: Decoded path or filename segment.

        Returns:
            Eligible whole and extension-stripped variants.
        """
        values: list[str] = []
        if not value:
            return ()
        if matches_custom_pattern(value):
            values.append(value)
        stripped = _strip_allowed_extension(value)
        if stripped and (_has_known_prefix(stripped) or matches_custom_pattern(stripped)):
            values.append(stripped)
        return tuple(dict.fromkeys(values))

    def add_path_candidate(value: str) -> tuple[int, ...]:
        """Append eligible path-like value variants.

        Args:
            value: Decoded path or filename segment.

        Returns:
            Candidate indexes appended for this component.
        """
        values = path_candidate_values(value)
        start = len(candidates)
        candidates.extend(values)
        return tuple(range(start, len(candidates)))

    def add_preferred_group(primary: tuple[int, ...], fallback_start: int) -> None:
        """Record that whole-component candidates outrank parser fallbacks.

        Args:
            primary: Definite candidates produced from the whole component.
            fallback_start: Candidate-list offset immediately before fallbacks.

        Returns:
            None.
        """
        for primary_index in primary:
            alternatives = tuple(index for index in range(fallback_start, len(candidates)) if index != primary_index)
            candidate_groups.append(
                _EmbeddedCandidateGroup(
                    primary_index,
                    alternatives,
                    preserve_occurrence=True,
                )
            )

    def split_prose_parts(
        value: str,
    ) -> tuple[tuple[str, str | None, str | None, int | None, int | None], ...]:
        """Split text at prose wrappers without treating URI data as paths.

        Args:
            value: Text surrounding or contained in an exempt URL span.

        Returns:
            Non-empty pieces paired with delimiter values and source offsets.
        """
        parts: list[tuple[str, str | None, str | None, int | None, int | None]] = []
        start = 0
        preceding_boundary: str | None = None
        preceding_boundary_offset: int | None = None
        for index, character in enumerate(value):
            is_boundary = character.isspace() or not (character.isalnum() or character in ASCII_URI_COMPONENT_SAFE)
            if not is_boundary:
                continue
            if start < index:
                parts.append((value[start:index], preceding_boundary, character, preceding_boundary_offset, index))
            preceding_boundary = character
            preceding_boundary_offset = index
            start = index + 1
        if start < len(value):
            parts.append((value[start:], preceding_boundary, None, preceding_boundary_offset, None))
        return tuple(parts)

    def trim_trailing_url_prose(value: str) -> str:
        """Remove trailing prose punctuation from a matched URL.

        Args:
            value: Raw regex match beginning with an HTTP(S) scheme.

        Returns:
            The URL match without closing prose punctuation.
        """
        return _trim_trailing_prose(value)

    def add_prose_candidates(
        value: str,
        *,
        force_all: bool = False,
        uri_data: bool = False,
        forcing_boundaries: frozenset[str] = frozenset(),
        encoded_boundary_offsets: frozenset[int] = frozenset(),
    ) -> None:
        """Append candidates separated from an exempt URL by prose.

        Args:
            value: Text outside a URL span or inside a parsed component.
            force_all: Whether every non-empty part lies outside the URL span.
            uri_data: Whether URI sub-delimiters are data rather than prose.
            forcing_boundaries: URI delimiters that remain unambiguous prose.
            encoded_boundary_offsets: Decoded offsets originating from percent escapes.

        Returns:
            None.
        """
        parts = split_prose_parts(value)
        non_forcing_boundaries = NON_FORCING_COMPONENT_BOUNDARIES | URI_SUBDELIMITERS if uri_data else NON_FORCING_COMPONENT_BOUNDARIES
        for part, preceding_boundary, following_boundary, preceding_offset, following_offset in parts:
            is_enclosed = (preceding_boundary, following_boundary) in ENCLOSING_BOUNDARY_PAIRS
            has_strong_boundary = (
                any(
                    boundary is not None
                    and offset not in encoded_boundary_offsets
                    and (boundary in forcing_boundaries or boundary not in non_forcing_boundaries)
                    and not (uri_data and boundary.isspace())
                    for boundary, offset in ((preceding_boundary, preceding_offset), (following_boundary, following_offset))
                )
                and not is_enclosed
            )
            if preceding_boundary is None and following_boundary is None and not force_all:
                continue
            for path_part in re.split(r"[\\/]", part):
                explicit_indexes = add_path_candidate(path_part)
                if path_part and not explicit_indexes and (force_all or has_strong_boundary):
                    candidates.append(path_part)

    def add_untrimmed_terminal_custom_candidates(value: str) -> None:
        """Scan a terminal component before prose trimming.

        Args:
            value: Raw URL match whose last component may end in punctuation.

        Returns:
            None.
        """
        nonlocal custom_range_characters, custom_scan_truncated, retained_custom_characters
        nonlocal retained_custom_count, terminal_range_evaluations
        if not custom_regex:
            return
        path_and_query, fragment_separator, fragment = value.partition("#")
        path, query_separator, query = path_and_query.partition("?")
        if fragment_separator:
            raw_terminal = fragment
            query_form = False
        elif query_separator:
            field = query.rsplit("&", 1)[-1]
            _, value_separator, raw_terminal = field.partition("=")
            raw_terminal = raw_terminal if value_separator else field
            query_form = True
        else:
            raw_terminal = path
            query_form = False
        raw_terminal = re.split(r"[/\\]", raw_terminal)[-1]
        terminal = unquote_plus(raw_terminal) if query_form else unquote(raw_terminal)
        trimmed = trim_trailing_url_prose(terminal)
        if not terminal or terminal == trimmed:
            return

        start_offsets, lexical_ends = _component_candidate_boundaries(terminal)
        if custom_literal_prefixes and all(prefix is not None for prefix in custom_literal_prefixes):
            start_offsets = tuple(
                start
                for start in start_offsets
                if any(matcher is not None and matcher.match(terminal, start) for matcher in custom_literal_prefix_matchers)
            )
        findings: list[tuple[int, str]] = []
        direct_pattern_count = len(compiled_custom_regex) + len(terminal_range_patterns)
        for start in start_offsets:
            suffix_length = len(terminal) - start
            if (
                terminal_range_evaluations + direct_pattern_count > MAX_CUSTOM_REGEX_EVALUATIONS
                or custom_range_characters + suffix_length > custom_range_budget
            ):
                candidates.extend(candidate for _, candidate in sorted(findings))
                custom_scan_truncated = True
                return
            custom_range_characters += suffix_length
            suffix = terminal[start:]
            longest_match: str | None = None
            for pattern in compiled_custom_regex:
                match = pattern.match(suffix)
                terminal_range_evaluations += 1
                if match and match.end():
                    match_end = start + match.end()
                    candidate_end = _bounded_component_match_end(terminal, match_end, lexical_ends)
                    candidate = terminal[start:candidate_end]
                    if candidate_end != match_end and pattern.match(candidate) is None:
                        candidate = suffix
                        candidate_end = len(terminal)
                    if candidate_end > len(trimmed) and (longest_match is None or len(candidate) > len(longest_match)):
                        longest_match = candidate
            for pattern in terminal_range_patterns:
                match = pattern.match(suffix)
                terminal_range_evaluations += 1
                if not match or not match.end():
                    continue
                candidate_end = start + match.end()
                candidate = terminal[start:candidate_end]
                if candidate_end > len(trimmed) and matches_custom_pattern(candidate):
                    if longest_match is None or len(candidate) > len(longest_match):
                        longest_match = candidate
            if longest_match is None:
                continue
            if (
                retained_custom_count >= MAX_RETAINED_CUSTOM_CANDIDATES
                or retained_custom_characters + len(longest_match) > MAX_RETAINED_CUSTOM_CHARACTERS
            ):
                candidates.extend(candidate for _, candidate in sorted(findings))
                custom_scan_truncated = True
                return
            findings.append((start, longest_match))
            retained_custom_count += 1
            retained_custom_characters += len(longest_match)
        candidates.extend(candidate for _, candidate in sorted(findings))

    def add_custom_component_ranges(
        value: str,
        start_offsets: tuple[int, ...],
        end_offsets: tuple[int, ...],
    ) -> None:
        """Apply configured regexes to bounded contiguous component ranges.

        Args:
            value: Decoded component containing the supplied boundaries.
            start_offsets: Ordered word and punctuation candidate starts.
            end_offsets: Ordered lexical candidate ends.

        Returns:
            None.
        """
        nonlocal custom_range_characters, custom_range_count, custom_scan_truncated, terminal_range_evaluations
        nonlocal priority_range_characters, priority_range_count, priority_regex_evaluations
        nonlocal retained_custom_characters, retained_custom_count
        if not custom_regex:
            return
        if custom_literal_prefixes and all(prefix is not None for prefix in custom_literal_prefixes):
            start_offsets = tuple(
                start
                for start in start_offsets
                if any(matcher is not None and matcher.match(value, start) for matcher in custom_literal_prefix_matchers)
            )
        punctuation_starts = tuple(start for start in start_offsets if not value[start].isalnum())
        range_starts = tuple(start for start in start_offsets if value[start].isalnum())
        projected_ranges = sum(max(0, len(end_offsets) - bisect_right(end_offsets, start) - MAX_CUSTOM_PRIORITY_BOUNDARIES) for start in range_starts)
        remaining_ranges = custom_range_limit - custom_range_count
        complete_range_scan = projected_ranges <= remaining_ranges
        if projected_ranges > remaining_ranges:
            range_starts = tuple(start for pair in zip(range_starts, reversed(range_starts), strict=True) for start in pair)
            range_starts = tuple(dict.fromkeys(range_starts))
        range_start_set = set(range_starts)
        scan_starts = range_starts + punctuation_starts
        range_findings: dict[int, str] = {}

        # Give each component a small, source-fair pass before any expensive
        # suffix work. A previous component's deep-scan truncation must not
        # suppress these short candidates in later URL or file components.
        for boundary_width in range(MAX_CUSTOM_PRIORITY_BOUNDARIES):
            for start_offset in scan_starts:
                end_index = bisect_right(end_offsets, start_offset) + boundary_width
                if end_index >= len(end_offsets):
                    continue
                end_offset = end_offsets[end_index]
                candidate_length = end_offset - start_offset
                matcher_count = len(custom_matchers)
                if (
                    priority_range_count >= custom_range_limit
                    or priority_range_characters + candidate_length > custom_range_budget
                    or priority_regex_evaluations + matcher_count > MAX_CUSTOM_PRIORITY_REGEX_EVALUATIONS
                ):
                    candidates.extend(_ordered_custom_range_findings(range_findings))
                    custom_scan_truncated = True
                    return
                priority_range_count += 1
                priority_range_characters += candidate_length
                priority_regex_evaluations += matcher_count
                candidate = value[start_offset:end_offset]
                if matches_custom_pattern(candidate):
                    retained, retained_custom_count, retained_custom_characters = _retain_longest_custom_range(
                        range_findings,
                        start_offset,
                        candidate,
                        retained_custom_count,
                        retained_custom_characters,
                    )
                    if not retained:
                        candidates.extend(_ordered_custom_range_findings(range_findings))
                        custom_scan_truncated = True
                        return

        if custom_scan_truncated:
            candidates.extend(_ordered_custom_range_findings(range_findings))
            return

        for start_offset in scan_starts:
            longest_match = range_findings.get(start_offset)
            suffix_length = len(value) - start_offset
            direct_pattern_count = len(compiled_custom_regex) + len(terminal_range_patterns)
            if (
                terminal_range_evaluations + direct_pattern_count > MAX_CUSTOM_REGEX_EVALUATIONS
                or custom_range_characters + suffix_length > custom_range_budget
            ):
                candidates.extend(_ordered_custom_range_findings(range_findings))
                custom_scan_truncated = True
                return
            custom_range_characters += suffix_length
            suffix = value[start_offset:]
            for pattern in compiled_custom_regex:
                match = pattern.match(suffix)
                terminal_range_evaluations += 1
                if match and match.end():
                    match_end = start_offset + match.end()
                    candidate_end = _bounded_component_match_end(value, match_end, end_offsets)
                    candidate = value[start_offset:candidate_end]
                    if candidate_end != match_end and pattern.match(candidate) is None:
                        candidate = suffix
                    if longest_match is None or len(candidate) > len(longest_match):
                        longest_match = candidate
            if terminal_range_patterns:
                for pattern in terminal_range_patterns:
                    match = pattern.match(suffix)
                    terminal_range_evaluations += 1
                    if match and match.end() and _ends_at_component_boundary(value, start_offset + match.end()):
                        candidate = suffix[: match.end()]
                    else:
                        candidate = ""
                    if candidate:
                        if matches_custom_pattern(candidate) and (longest_match is None or len(candidate) > len(longest_match)):
                            longest_match = candidate
            end_start = bisect_right(end_offsets, start_offset) + MAX_CUSTOM_PRIORITY_BOUNDARIES
            candidate_ends = end_offsets[end_start:] if start_offset in range_start_set else ()
            first_match_index: int | None = None
            if longest_match is not None:
                match_end = start_offset + len(longest_match)
                match_end_index = bisect_left(candidate_ends, match_end)
                if match_end_index < len(candidate_ends) and candidate_ends[match_end_index] == match_end:
                    first_match_index = match_end_index
            if first_match_index is None:
                for end_index, end_offset in enumerate(candidate_ends):
                    candidate_length = end_offset - start_offset
                    if custom_range_count >= custom_range_limit or custom_range_characters + candidate_length > custom_range_budget:
                        if longest_match is not None:
                            retained, retained_custom_count, retained_custom_characters = _retain_longest_custom_range(
                                range_findings,
                                start_offset,
                                longest_match,
                                retained_custom_count,
                                retained_custom_characters,
                            )
                            if not retained:
                                custom_scan_truncated = True
                        candidates.extend(_ordered_custom_range_findings(range_findings))
                        custom_scan_truncated = True
                        return
                    custom_range_count += 1
                    custom_range_characters += candidate_length
                    candidate = value[start_offset:end_offset]
                    if matches_custom_pattern(candidate):
                        if longest_match is None or len(candidate) > len(longest_match):
                            longest_match = candidate
                        first_match_index = end_index
                        break
            if first_match_index is not None:
                has_fixed_start = bool(custom_literal_prefixes) and all(prefix is not None for prefix in custom_literal_prefixes)
                nearby_end = (
                    len(candidate_ends)
                    if complete_range_scan or has_fixed_start
                    else min(len(candidate_ends), first_match_index + 1 + MAX_CUSTOM_NEARBY_BOUNDARIES)
                )
                for end_offset in candidate_ends[first_match_index + 1 : nearby_end]:
                    candidate_length = end_offset - start_offset
                    if custom_range_count >= custom_range_limit or custom_range_characters + candidate_length > custom_range_budget:
                        candidates.extend(_ordered_custom_range_findings(range_findings))
                        custom_scan_truncated = True
                        return
                    custom_range_count += 1
                    custom_range_characters += candidate_length
                    candidate = value[start_offset:end_offset]
                    if matches_custom_pattern(candidate):
                        if longest_match is None or len(candidate) > len(longest_match):
                            longest_match = candidate

                terminal_start = max(nearby_end, len(candidate_ends) - MAX_CUSTOM_TERMINAL_BOUNDARIES)
                for end_offset in reversed(candidate_ends[terminal_start:]):
                    candidate_length = end_offset - start_offset
                    if custom_range_count >= custom_range_limit or custom_range_characters + candidate_length > custom_range_budget:
                        candidates.extend(_ordered_custom_range_findings(range_findings))
                        custom_scan_truncated = True
                        return
                    custom_range_count += 1
                    custom_range_characters += candidate_length
                    candidate = value[start_offset:end_offset]
                    if matches_custom_pattern(candidate):
                        if longest_match is None or len(candidate) > len(longest_match):
                            longest_match = candidate
                        break
            if longest_match is not None:
                # Ranges from the same start are alternate parses, not source
                # occurrences. Retain the most complete matching boundary.
                retained, retained_custom_count, retained_custom_characters = _retain_longest_custom_range(
                    range_findings,
                    start_offset,
                    longest_match,
                    retained_custom_count,
                    retained_custom_characters,
                )
                if not retained:
                    candidates.extend(_ordered_custom_range_findings(range_findings))
                    custom_scan_truncated = True
                    return
        candidates.extend(_ordered_custom_range_findings(range_findings))

    def sensitive_assignment_value(value: str) -> tuple[str, int] | None:
        """Find a value assigned to a sensitive label.

        Args:
            value: Decoded component that may contain ``=`` or ``:``.

        Returns:
            The associated value and its offset, or ``None``.
        """
        label_start = 0
        for separator in re.finditer(r"[=:]", value):
            if _is_sensitive_parameter(value[label_start : separator.start()]):
                return value[separator.end() :], separator.end()
            label_start = separator.end()
        return None

    def add_intra_component_candidates(value: str) -> None:
        """Inspect slug-delimited suffixes inside one path component.

        Args:
            value: Decoded path component that may contain slug data.

        Returns:
            None.
        """
        candidates.extend(_embedded_known_prefix_suffixes(value))
        start_offsets, end_offsets = _component_candidate_boundaries(value)
        if len(start_offsets) < 2:
            return
        add_custom_component_ranges(value, start_offsets, end_offsets)

    def add_decoded_path_segment(
        segment: str,
        *,
        force_value: bool = False,
        encoded_boundary_offsets: frozenset[int] = frozenset(),
    ) -> None:
        """Inspect one segment separated by a definite path boundary.

        Args:
            segment: Decoded segment to inspect.
            force_value: Whether a preceding label associates this segment.
            encoded_boundary_offsets: Segment offsets originating from percent escapes.

        Returns:
            None.
        """
        add_path_candidate(segment)
        add_prose_candidates(segment, uri_data=True, encoded_boundary_offsets=encoded_boundary_offsets)
        add_intra_component_candidates(segment)

        assignment = sensitive_assignment_value(segment)
        if assignment is not None and assignment[0]:
            associated_value, _ = assignment
            add_value_candidate(associated_value, force=True, encoded=False)
        if force_value:
            payload = _structural_index_payload(segment)
            add_value_candidate(payload if payload is not None else segment, force=True, encoded=False)

    def add_ambiguous_path_segment(
        value: str,
        definite_indexes: tuple[int, ...],
        *,
        record_ambiguity: bool = True,
        encoded_boundary_offsets: frozenset[int] = frozenset(),
    ) -> None:
        """Inspect one raw segment after decoding encoded separators.

        Literal path separators have already been removed, so any separators
        remaining in this value came from percent encoding and may be either
        boundaries or credential data.

        Args:
            value: Individually decoded raw path segment.
            definite_indexes: Candidate indexes from the decoded boundary view.
            record_ambiguity: Whether to retain path-boundary provenance.
            encoded_boundary_offsets: Value offsets originating from percent escapes.

        Returns:
            None.
        """
        segments = tuple(re.finditer(r"[^\\/]+", value))
        segment_values = tuple(segment.group(0) for segment in segments)
        ambiguous_start = len(candidates)
        recorded_groups: set[_EmbeddedCandidateGroup] = set()
        group_specs: list[tuple[tuple[int, ...], tuple[str, ...]]] = []
        component_candidate_cache: dict[str, tuple[str, ...]] = {}
        fallback_values_cache: dict[frozenset[str], frozenset[str]] = {}
        candidate_indexes_by_value: dict[str, tuple[int, ...]] = {}

        def record_ambiguous_group(
            primary: tuple[int, ...],
            components: tuple[str, ...],
        ) -> None:
            """Relate whole candidates to encoded-boundary views.

            Args:
                primary: Whole-component candidate indexes.
                components: Decoded boundary-separated component values.

            Returns:
                None.
            """
            if len(components) < 2:
                return
            component_key = frozenset(components)
            fallback_values = fallback_values_cache.get(component_key)
            if fallback_values is None:
                fallback_values = frozenset(
                    candidate
                    for component in component_key
                    for candidate in component_candidate_cache.setdefault(component, path_candidate_values(component))
                )
                fallback_values_cache[component_key] = fallback_values
            for primary_index in primary:
                fallbacks = tuple(
                    sorted(
                        index
                        for fallback_value in fallback_values
                        if fallback_value != candidates[primary_index]
                        for index in candidate_indexes_by_value.get(fallback_value, ())
                    )
                )
                group = _EmbeddedCandidateGroup(primary_index, fallbacks, components[1:])
                if fallbacks and group not in recorded_groups:
                    candidate_groups.append(group)
                    recorded_groups.add(group)

        primary = add_path_candidate(value)
        if record_ambiguity:
            group_specs.append((primary, segment_values))
        add_prose_candidates(value, uri_data=True, encoded_boundary_offsets=encoded_boundary_offsets)

        if len(segments) > 1:
            custom_start = len(candidates)
            add_custom_component_ranges(value, *_component_candidate_boundaries(value))
            if record_ambiguity:
                for custom_index in range(custom_start, len(candidates)):
                    custom_components = tuple(component for component in re.split(r"[\\/]", candidates[custom_index]) if component)
                    if len(custom_components) > 1:
                        group_specs.append(((custom_index,), custom_components))

        for suffix in _embedded_known_prefix_suffixes(value):
            suffix_components = tuple(component for component in re.split(r"[\\/]", suffix) if component)
            candidates.append(suffix)
            primary = (len(candidates) - 1,)
            if record_ambiguity:
                group_specs.append((primary, suffix_components))

        forced_starts: list[int] = []
        for index, segment_match in enumerate(segments):
            segment = segment_match.group(0)
            assignment = sensitive_assignment_value(segment)
            if assignment is not None:
                associated_value, value_offset = assignment
                value_start = segment_match.start() + value_offset
                if associated_value:
                    forced_starts.append(value_start)
                else:
                    for value_index in _associated_value_indexes(segment_values, index):
                        payload = _structural_index_payload(segment_values[value_index])
                        if payload is not None:
                            add_value_candidate(payload, force=True, encoded=False)
                        else:
                            forced_starts.append(segments[value_index].start())
            if _is_sensitive_parameter(segment):
                for value_index in _associated_value_indexes(segment_values, index):
                    payload = _structural_index_payload(segment_values[value_index])
                    if payload is not None:
                        add_value_candidate(payload, force=True, encoded=False)
                    else:
                        forced_starts.append(segments[value_index].start())

        for value_start in dict.fromkeys(forced_starts[:1] + forced_starts[-1:]):
            add_value_candidate(value[value_start:], force=True, encoded=False)

        available_indexes = definite_indexes + tuple(range(ambiguous_start, len(candidates)))
        mutable_indexes_by_value: dict[str, list[int]] = {}
        for available_index in available_indexes:
            mutable_indexes_by_value.setdefault(candidates[available_index], []).append(available_index)
        candidate_indexes_by_value.update((candidate, tuple(indexes)) for candidate, indexes in mutable_indexes_by_value.items())
        for primary, components in group_specs:
            record_ambiguous_group(primary, components)

    def add_path_components(
        value: str,
        *,
        encoded: bool = True,
        query_encoded: bool = False,
        record_ambiguity: bool = True,
    ) -> None:
        """Append candidates without crossing literal path boundaries.

        Args:
            value: Raw URL path, fragment, parameter, or file-path content.
            encoded: Whether the current syntax layer still needs decoding.
            query_encoded: Whether plus signs use query-form decoding.
            record_ambiguity: Whether encoded separators are path-ambiguous.

        Returns:
            None.
        """
        raw_segments = tuple(segment for segment in re.split(r"[\\/]", value) if segment)
        decoded_raw_segments = tuple(
            decode_uri_component(raw_segment, query_form=query_encoded) if encoded else (raw_segment, frozenset()) for raw_segment in raw_segments
        )
        for decoded_raw_segment, _encoded_offsets in decoded_raw_segments:
            if url_pattern.search(decoded_raw_segment) or is_scheme_relative_url(decoded_raw_segment):
                scan_token(decoded_raw_segment)
                if scan_overflow:
                    return
        decoded_groups = tuple(tuple(re.finditer(r"[^\\/]+", decoded_raw_segment)) for decoded_raw_segment, _encoded_offsets in decoded_raw_segments)
        decoded_segments = tuple(segment.group(0) for group in decoded_groups for segment in group)
        forced_value_indexes = {
            value_index
            for index, segment in enumerate(decoded_segments)
            if _is_sensitive_parameter(segment)
            for value_index in _associated_value_indexes(decoded_segments, index)
        }

        decoded_index = 0
        for (decoded_raw_segment, encoded_offsets), decoded_group in zip(
            decoded_raw_segments,
            decoded_groups,
            strict=True,
        ):
            definite_start = len(candidates)
            for segment_match in decoded_group:
                segment = segment_match.group(0)
                occurrence_start = len(candidates)
                segment_encoded_offsets = frozenset(
                    offset - segment_match.start() for offset in encoded_offsets if segment_match.start() <= offset < segment_match.end()
                )
                add_decoded_path_segment(
                    segment,
                    force_value=decoded_index in forced_value_indexes,
                    encoded_boundary_offsets=segment_encoded_offsets,
                )
                occurrence_values: set[str] = set()
                for candidate_index in range(occurrence_start, len(candidates)):
                    candidate = candidates[candidate_index]
                    if candidate in occurrence_values:
                        continue
                    occurrence_values.add(candidate)
                    candidate_groups.append(
                        _EmbeddedCandidateGroup(
                            candidate_index,
                            (),
                            preserve_occurrence=True,
                        )
                    )
                decoded_index += 1
            definite_indexes = tuple(range(definite_start, len(candidates)))
            add_ambiguous_path_segment(
                decoded_raw_segment,
                definite_indexes,
                record_ambiguity=record_ambiguity,
                encoded_boundary_offsets=encoded_offsets,
            )

    def add_parameter_candidates(
        params: str,
        *,
        query_form: bool = True,
        bare_names_are_paths: bool = False,
    ) -> None:
        """Append candidates from raw query-style parameter fields.

        Args:
            params: Raw query or fragment parameter text.
            query_form: Whether plus signs represent spaces.
            bare_names_are_paths: Whether slash-containing bare fields are paths.

        Returns:
            None.
        """
        decode = unquote_plus if query_form else unquote
        for field in params.split("&"):
            if scan_overflow:
                return
            if not field:
                continue
            raw_name, value_separator, raw_value = field.partition("=")
            name = decode(raw_name)
            name_is_bare_path = bare_names_are_paths and not value_separator and bool(re.search(r"[\\/]", raw_name))
            name_primary = () if name_is_bare_path else add_value_candidate(name, encoded=False)
            fallback_start = len(candidates)
            add_path_components(raw_name, query_encoded=query_form, record_ambiguity=not name_primary)
            add_preferred_group(name_primary, fallback_start)

            if not value_separator:
                continue
            decoded_value, encoded_value_offsets = decode_uri_component(raw_value, query_form=query_form)
            if len(decoded_value) >= 2 and (decoded_value[0], decoded_value[-1]) in ENCLOSING_BOUNDARY_PAIRS:
                normalized_value = decoded_value[1:-1]
            else:
                normalized_value = trim_trailing_url_prose(decoded_value)
            if decoded_value != normalized_value and matches_custom_pattern(decoded_value) and not matches_custom_pattern(normalized_value):
                value = decoded_value
            else:
                value = normalized_value
            value_primary = add_value_candidate(value, force=_is_sensitive_parameter(name), encoded=False)
            fallback_start = len(candidates)
            add_prose_candidates(
                decoded_value,
                uri_data=True,
                encoded_boundary_offsets=encoded_value_offsets,
            )
            if url_pattern.search(value) or is_scheme_relative_url(value):
                scan_token(value)
            else:
                if value != decoded_value:
                    add_custom_component_ranges(decoded_value, *_component_candidate_boundaries(decoded_value))
                    add_path_components(value, encoded=False, record_ambiguity=not value_primary)
                else:
                    add_path_components(raw_value, query_encoded=query_form, record_ambiguity=not value_primary)
                add_preferred_group(value_primary, fallback_start)

    def add_host_components(value: str, *, encoded: bool = True) -> None:
        """Append credential-like labels from a raw host value.

        Args:
            value: Raw host text with case and escapes preserved.
            encoded: Whether the host still needs one percent-decoding pass.

        Returns:
            None.
        """
        decoded_value, encoded_offsets = decode_uri_component(value) if encoded else (value, frozenset())
        decoded = re.sub(r"[\u3002\uff0e\uff61]", ".", decoded_value)
        add_prose_candidates(
            decoded,
            uri_data=True,
            forcing_boundaries=frozenset({","}),
            encoded_boundary_offsets=encoded_offsets,
        )
        label_spans = tuple(re.finditer(r"[^.]+", decoded))
        labels = tuple(label.group(0) for label in label_spans)
        for index, label in enumerate(labels):
            add_value_candidate(label, encoded=False)
            if index + 1 < len(labels):
                pair = f"{label}.{labels[index + 1]}"
                if any("." in prefix and pair.startswith(prefix) for prefix in COMMON_KEY_PREFIXES):
                    dotted_candidate = pair
                    if (
                        pair.startswith("SG.")
                        and index + 2 < len(labels)
                        and re.fullmatch(r"[A-Za-z0-9_-]{22}", labels[index + 1])
                        and re.fullmatch(r"[A-Za-z0-9_-]{43}", labels[index + 2])
                    ):
                        dotted_candidate = f"{pair}.{labels[index + 2]}"
                    candidates.append(dotted_candidate)
        add_custom_component_ranges(decoded, *_component_candidate_boundaries(decoded))

        for raw_label in (label for label in re.split(r"[.\u3002\uff0e\uff61]", value) if label):
            add_path_components(raw_label, encoded=encoded)

    def add_authority_candidates(authority: str) -> None:
        """Append candidates from raw URL userinfo and host text.

        Args:
            authority: Raw URL authority without a scheme or suffix.

        Returns:
            None.
        """
        userinfo, userinfo_separator, host_and_port = authority.rpartition("@")
        if userinfo_separator:
            username, password_separator, password = userinfo.partition(":")
            if username:
                username_primary = add_value_candidate(username, force=True)
                fallback_start = len(candidates)
                add_path_components(username, record_ambiguity=False)
                add_preferred_group(username_primary, fallback_start)
            if password_separator and password:
                password_primary = add_value_candidate(password, force=True)
                fallback_start = len(candidates)
                add_path_components(password, record_ambiguity=False)
                add_preferred_group(password_primary, fallback_start)
        else:
            host_and_port = authority

        decoded_authority = unquote(host_and_port)
        authority_has_known_prefix = any(":" in prefix and decoded_authority.startswith(prefix) for prefix in COMMON_KEY_PREFIXES)
        if authority_has_known_prefix:
            candidates.append(decoded_authority)
        malformed_tail = ""
        bracketed_host = decoded_authority.startswith("[")
        host_components_encoded = True
        tail_encoded = True
        port_encoded = True
        if bracketed_host:
            closing_bracket = decoded_authority.find("]")
            raw_host = decoded_authority[1:closing_bracket] if closing_bracket >= 0 else decoded_authority[1:]
            authority_tail = decoded_authority[closing_bracket + 1 :] if closing_bracket >= 0 else ""
            raw_port = authority_tail[1:] if authority_tail.startswith(":") else ""
            if authority_tail and not authority_tail.startswith(":"):
                malformed_tail = authority_tail
            host_components_encoded = False
            tail_encoded = False
            port_encoded = False
        else:
            raw_host, port_separator, raw_port = host_and_port.rpartition(":")
            if not port_separator:
                raw_host = host_and_port
                raw_port = ""
        add_host_components(raw_host, encoded=host_components_encoded)
        decoded_host = unquote(raw_host) if host_components_encoded else raw_host
        try:
            ip_address(decoded_host)
            host_is_ip_literal = True
        except ValueError:
            host_is_ip_literal = bool(re.fullmatch(r"[vV][0-9A-Fa-f]+\.[A-Za-z0-9._~!$&'()*+,;=:-]+", decoded_host))
        if (bracketed_host or ":" in decoded_host) and not host_is_ip_literal:
            malformed_start = len(candidates)
            add_value_candidate(decoded_host, force=True, encoded=False)
            suffix_primary: tuple[int, ...] = ()
            if ":" in decoded_host:
                suffix_primary = add_value_candidate(decoded_host.split(":", 1)[1], force=True, encoded=False)
            for component in (component for component in decoded_host.split(":") if component):
                add_value_candidate(component, force=True, encoded=False)
            add_preferred_group(suffix_primary, malformed_start)
        if malformed_tail:
            tail_primary = add_value_candidate(malformed_tail, force=True, encoded=tail_encoded)
            fallback_start = len(candidates)
            add_path_components(malformed_tail, encoded=tail_encoded, record_ambiguity=False)
            add_preferred_group(tail_primary, fallback_start)
        if raw_port:
            decoded_port = unquote(raw_port) if port_encoded else raw_port
            port_primary = add_value_candidate(
                raw_port,
                force=not decoded_port.isdigit() and not authority_has_known_prefix,
                encoded=port_encoded,
            )
            fallback_start = len(candidates)
            add_path_components(raw_port, encoded=port_encoded, record_ambiguity=False)
            add_preferred_group(port_primary, fallback_start)

    def add_url_suffix_candidates(suffix: str) -> None:
        """Append candidates from a raw URL path, query, and fragment.

        Args:
            suffix: Raw URL content following the authority.

        Returns:
            None.
        """
        path_and_query, fragment_separator, fragment = suffix.partition("#")
        path, query_separator, query = path_and_query.partition("?")
        add_path_components(path)

        if query_separator:
            add_parameter_candidates(query)
        if fragment_separator:
            add_parameter_candidates(fragment, query_form=False, bare_names_are_paths=True)

    def scan_token_contents(current_token: str) -> None:
        """Scan one URL or file token within the shared safety budget.

        Nested URL values are scanned immediately so findings retain their
        left-to-right, depth-first occurrence order.

        Args:
            current_token: Raw token or decoded nested URL value to inspect.

        Returns:
            None.
        """
        nonlocal scan_overflow, scanned_characters

        if is_scheme_relative_url(current_token):
            raw_url_source = current_token
            raw_url = trim_trailing_url_prose(raw_url_source)
            authority_match = re.match(r"([^/\\?#]*)(.*)", raw_url[2:])
            if authority_match is not None:
                authority, suffix = authority_match.groups()
                add_authority_candidates(authority)
                add_url_suffix_candidates(suffix)
                add_untrimmed_terminal_custom_candidates(raw_url_source)
                add_prose_candidates(raw_url_source[len(raw_url) :])
                if scan_overflow:
                    return

        url_matches = tuple(url_pattern.finditer(current_token))
        if not url_matches and re.search(r"https?://[^\s]+", current_token, re.IGNORECASE):
            add_prose_candidates(current_token, force_all=True)
        match_end = 0
        for match_index, match in enumerate(url_matches):
            if match_index:
                nested_span_length = len(current_token) - match.start()
                if scanned_characters + nested_span_length > scan_budget:
                    scan_overflow = True
                    return
                scanned_characters += nested_span_length
            add_prose_candidates(current_token[match_end : match.start()], force_all=True)
            raw_match = match.group(0)
            leading_text = current_token[: match.start()]
            raw_url_source = raw_match
            if (link_end := raw_url_source.find("](")) >= 0:
                raw_url_source = raw_url_source[:link_end]

            leading_markers = re.search(r"[*_~`]+$", leading_text)
            wrapped_url = f"{leading_markers.group(0) if leading_markers else ''}{raw_url_source}"
            normalized_url = _normalize_allowed_token(wrapped_url)
            raw_url = normalized_url if re.match(r"^https?://", normalized_url, re.IGNORECASE) else trim_trailing_url_prose(raw_url_source)
            if leading_markers is None:
                for marker in ("**", "~~"):
                    if raw_url.endswith(marker):
                        raw_url = raw_url[: -len(marker)]
                        break
            remainder = raw_url.split("://", 1)[1]
            authority_match = re.match(r"([^/\\?#]*)(.*)", remainder)
            if authority_match is None:
                match_end = match.end()
                continue
            authority, suffix = authority_match.groups()
            add_authority_candidates(authority)
            add_url_suffix_candidates(suffix)
            add_untrimmed_terminal_custom_candidates(raw_url_source)
            add_prose_candidates(raw_match[len(raw_url) :])
            match_end = match.end()
            if scan_overflow:
                return
        if url_matches:
            add_prose_candidates(current_token[match_end:], force_all=True)

        # Markdown markers at the token edges are presentation syntax. Preserve
        # internal markers as real boundaries instead of concatenating the text
        # on either side and potentially hiding an embedded credential.
        raw_file_token = _normalize_allowed_token(current_token).strip("*#")
        file_token = _normalize_allowed_token(unquote(raw_file_token))
        lowered = file_token.lower()
        is_file_pattern = any(lowered.endswith(extension) for extension in ALLOWED_EXTENSIONS)
        markdown_file = re.fullmatch(r"!?\[([^\]]*)\]\((.+)\)", current_token)
        if is_file_pattern and (url_pattern.search(current_token) is None or markdown_file is not None):
            if markdown_file and url_pattern.search(markdown_file.group(1)) is None:
                add_prose_candidates(markdown_file.group(1), force_all=True)
            raw_root = re.split(r"[\\/]", raw_file_token, maxsplit=1)[0]
            root_parts = split_prose_parts(raw_root)
            non_forcing_root_boundaries = NON_FORCING_COMPONENT_BOUNDARIES | URI_SUBDELIMITERS
            has_explicit_file_root = bool(re.search(r"[\\/]", raw_file_token)) and bool(root_parts and root_parts[-1][0].lower() in {"file", "files"})
            if any(
                boundary is not None and (has_explicit_file_root or boundary not in non_forcing_root_boundaries)
                for _, preceding, following, _, _ in root_parts
                for boundary in (preceding, following)
            ):
                candidates.extend(unquote(part) for part, _, _, _, _ in root_parts)
            file_candidate_start = len(candidates)
            add_path_components(raw_file_token)
            file_indexes = tuple(range(file_candidate_start, len(candidates)))
            primary_by_value: dict[str, int] = {}
            for index in file_indexes:
                primary_by_value.setdefault(candidates[index], index)
            fallback_by_primary: dict[int, list[int]] = {}
            for index in file_indexes:
                value = candidates[index]
                normalized_value = _strip_allowed_extension(_trim_trailing_prose(value))
                primary_index = primary_by_value.get(normalized_value)
                if primary_index is not None and normalized_value != value and not matches_custom_pattern(value):
                    fallback_by_primary.setdefault(primary_index, []).append(index)
            candidate_groups.extend(
                _EmbeddedCandidateGroup(primary_index, tuple(fallback_indexes)) for primary_index, fallback_indexes in fallback_by_primary.items()
            )

    def scan_token(current_token: str) -> None:
        """Scan one token while preventing only active recursion cycles.

        Args:
            current_token: Raw token or decoded nested URL value to inspect.

        Returns:
            None.
        """
        nonlocal scanned_characters, scan_overflow
        if scan_overflow or current_token in active_tokens:
            return
        if scanned_characters + len(current_token) > scan_budget:
            scan_overflow = True
            return
        scanned_characters += len(current_token)
        active_tokens.add(current_token)
        try:
            scan_token_contents(current_token)
        finally:
            active_tokens.remove(current_token)

    scan_token(token)

    return (
        tuple(candidates),
        _EmbeddedScanStatus(scan_overflow, custom_scan_truncated),
        tuple(dict.fromkeys(candidate_groups)),
    )


def _embedded_secret_candidates(token: str, custom_regex: list[str] | None = None) -> tuple[str, ...]:
    """Extract credential-like values hidden by URL or file exemptions.

    Args:
        token: Whitespace-delimited token that matched an allowed pattern.
        custom_regex: Optional configured regular expressions.

    Returns:
        Unique embedded values that should be checked independently.
    """
    candidates, _, _ = _extract_embedded_secret_candidates(token, custom_regex)
    return tuple(dict.fromkeys(candidates))


def _is_secret_candidate(
    s: str,
    cfg: SecretCfg,
    custom_regex: list[str] | None = None,
    *,
    allow_pattern_exemption: bool = True,
    custom_matchers: tuple[re.Pattern[str], ...] | None = None,
) -> bool:
    """Check if a string is a secret key using the specified criteria.

    Skips candidates matching allowed patterns (when strict_mode=False),
    enforces minimum length, character diversity, common prefix, and entropy.
    Also checks against custom patterns if provided.

    Args:
        s (str): String to analyze.
        cfg (SecretCfg): Detection configuration.
        custom_regex (Optional[List[str]]): List of custom regex patterns to check.
        allow_pattern_exemption: Whether URL/file-pattern exemptions may suppress this candidate.
        custom_matchers: Optional precompiled boolean-equivalent matchers.

    Returns:
        bool: True if the string is a secret key; otherwise False.
    """
    # Check custom patterns first if provided
    if custom_matchers is not None:
        if any(pattern.match(s) for pattern in custom_matchers):
            return True
    elif custom_regex:
        for pattern in custom_regex:
            if re.match(pattern, s):
                return True

    if allow_pattern_exemption and not cfg.get("strict_mode", False) and _contains_allowed_pattern(s):
        return False

    long_enough = len(s) >= cfg.get("min_length", 15)
    diverse = _char_diversity(s) >= cfg.get("min_diversity", 2)

    if not (long_enough and diverse):
        return False

    if _has_known_prefix(s):
        return True

    return _entropy(s) >= cfg.get("min_entropy", 3.7)


def _canonical_embedded_findings(
    candidates: tuple[str, ...],
    groups: tuple[_EmbeddedCandidateGroup, ...],
    cfg: SecretCfg,
    custom_regex: list[str] | None,
) -> list[str]:
    """Select one policy-valid interpretation per parsed component.

    Args:
        candidates: Extracted values in occurrence order.
        groups: Provenance linking whole components to parser fallbacks.
        cfg: Active secret-detection policy.
        custom_regex: Optional configured expressions.

    Returns:
        Canonical policy-valid findings in occurrence order.
    """
    compiled_custom_regex = tuple(re.compile(pattern) for pattern in custom_regex or ())
    custom_matchers = _combined_custom_matchers(compiled_custom_regex)
    valid = tuple(
        _is_secret_candidate(
            candidate,
            cfg,
            allow_pattern_exemption=False,
            custom_matchers=custom_matchers,
        )
        for candidate in candidates
    )
    suppressed: set[int] = set()

    for group in groups:
        if not valid[group.primary_index]:
            continue
        valid_fallbacks = tuple(index for index in group.fallback_indexes if valid[index])
        if not valid_fallbacks:
            continue

        if group.ambiguous_suffixes is None:
            suppressed.update(valid_fallbacks)
            continue

        explicit_full_match = any(pattern.match(candidates[group.primary_index]) for pattern in custom_matchers)
        suffix_has_secret_evidence = any(
            _is_secret_candidate(
                _strip_allowed_extension(suffix),
                CONFIGS["strict"],
                allow_pattern_exemption=False,
                custom_matchers=custom_matchers,
            )
            for suffix in group.ambiguous_suffixes
        )
        if explicit_full_match or suffix_has_secret_evidence:
            suppressed.update(valid_fallbacks)
        else:
            suppressed.add(group.primary_index)

    occurrence_primaries = {group.primary_index for group in groups if group.preserve_occurrence}
    findings: list[str] = []
    seen: set[str] = set()
    seen_occurrence_primaries: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not valid[index] or index in suppressed:
            continue
        is_occurrence_primary = index in occurrence_primaries
        if candidate not in seen or (is_occurrence_primary and candidate in seen_occurrence_primaries):
            findings.append(candidate)
        seen.add(candidate)
        if is_occurrence_primary:
            seen_occurrence_primaries.add(candidate)
    return findings


def _extend_occurrence_findings(findings: list[str], additions: list[str]) -> None:
    """Merge parser findings without collapsing distinct occurrences.

    Args:
        findings: Findings already selected for the lexical token.
        additions: Findings from another parser view of the same token.

    Returns:
        None.
    """
    existing_counts = Counter(findings)
    addition_counts: Counter[str] = Counter()
    for finding in additions:
        addition_counts[finding] += 1
        if addition_counts[finding] > existing_counts[finding]:
            findings.append(finding)


def _strip_closed_presentation_regions(value: str, closers: list[str], minimum_index: int = 0) -> str:
    """Remove presentation closers already opened by preceding text.

    The closer list is ordered by source occurrence and is updated in place.
    Markdown label brackets remain in the value so their targets can still be
    parsed, while their presentation state is closed.

    Args:
        value: Current whitespace-delimited token.
        closers: Bounded presentation-closer state to update.
        minimum_index: Earliest position that can contain a closer.

    Returns:
        The token with matched presentation closers removed.
    """
    while closers:
        matched_closer = next(
            (
                (index, closer_index)
                for index, closer in enumerate(closers)
                if (closer_index := value.rfind(closer)) >= minimum_index and _trim_trailing_prose(value[closer_index + len(closer) :]) == ""
            ),
            None,
        )
        if matched_closer is not None:
            matched_index, closer_start = matched_closer
            closer = closers.pop(matched_index)
            value = value[:closer_start] + value[closer_start + len(closer) :]
            continue

        markdown_boundary = value.find("](", minimum_index)
        if markdown_boundary < 0 or "]" not in closers:
            break
        label_end = markdown_boundary
        while closers:
            label_closer_index = next(
                (index for index, closer in enumerate(closers) if closer != "]" and value[:label_end].endswith(closer)),
                None,
            )
            if label_closer_index is None:
                break
            closer = closers.pop(label_closer_index)
            closer_start = label_end - len(closer)
            value = value[:closer_start] + value[label_end:]
            label_end = closer_start
        closers.remove("]")

    return value


def _matched_presentation_openers(text: str) -> frozenset[int]:
    """Pair presentation openers with plausible later delimiters.

    Args:
        text: Complete text whose presentation delimiters are being parsed.

    Returns:
        Source offsets of presentation openers paired with later closers.
    """
    matched: set[int] = set()
    asymmetric_pairs = {opening: closing for opening, closing in ENCLOSING_BOUNDARY_PAIRS if opening != closing}
    asymmetric_pending: dict[str, list[int]] = {}
    for index, character in enumerate(text):
        closing = asymmetric_pairs.get(character)
        if closing is not None:
            asymmetric_pending.setdefault(closing, []).append(index)
            continue
        pending_openers = asymmetric_pending.get(character)
        if pending_openers:
            matched.add(pending_openers.pop())

    pending: dict[str, list[int]] = {}
    index = 0
    while index < len(text):
        character = text[index]
        if character not in "*_~`\"'":
            index += 1
            continue
        marker_end = index + 1
        while marker_end < len(text) and text[marker_end] == character:
            marker_end += 1
        marker = text[index:marker_end]
        can_close = index > 0 and not text[index - 1].isspace() and (marker_end == len(text) or not text[marker_end].isalnum())
        marker_pending = pending.setdefault(marker, [])
        if can_close and marker_pending:
            opening = marker_pending.pop()
            matched.update(range(opening, opening + len(marker)))
            index = marker_end
            continue
        can_open = marker_end < len(text) and not text[marker_end].isspace() and (index == 0 or not text[index - 1].isalnum())
        if can_open:
            marker_pending.append(index)
        index = marker_end
    return frozenset(matched)


def _detect_secret_keys(text: str, cfg: SecretCfg, custom_regex: list[str] | None = None) -> GuardrailResult:
    """Detect potential secret keys in text.

    Args:
        text (str): Input text to scan.
        cfg (SecretCfg): Secret detection criteria.
        custom_regex (Optional[List[str]]): List of custom regex patterns to check.

    Returns:
        GuardrailResult: Result containing flag status and detected secrets.
    """
    secrets: list[str] = []
    custom_scan_incomplete = False
    presentation_closers: list[str] = []
    matched_presentation_openers = _matched_presentation_openers(text)
    source_word_counts = Counter(match.group(0) for match in re.finditer(r"\S+", text))
    embedded_cache: dict[
        str,
        tuple[tuple[str, ...], _EmbeddedScanStatus, tuple[_EmbeddedCandidateGroup, ...]],
    ] = {}

    def extract_embedded(
        token: str,
        *,
        cacheable: bool = False,
    ) -> tuple[tuple[str, ...], _EmbeddedScanStatus, tuple[_EmbeddedCandidateGroup, ...]]:
        """Reuse embedded parsing for repeated source tokens.

        Args:
            token: Normalized token or Markdown label to inspect.
            cacheable: Whether the source token is known to repeat.

        Returns:
            Cached candidates, completion state, and provenance groups.
        """
        result = embedded_cache.get(token) if cacheable else None
        if result is None:
            result = _extract_embedded_secret_candidates(token, custom_regex)
        if cacheable:
            embedded_cache[token] = result
        return result

    for token_match in re.finditer(r"\S+", text):
        raw_word = token_match.group(0)
        source_word = raw_word
        presentation_overflow = False
        raw_word = _strip_closed_presentation_regions(raw_word, presentation_closers)

        opening_offset = 0
        token_closers: list[str] = []
        while opening_offset < len(raw_word):
            character = raw_word[opening_offset]
            recognized_opening = character in "*_~`" or any(opening == character for opening, _ in ENCLOSING_BOUNDARY_PAIRS)
            if len(presentation_closers) + len(token_closers) >= MAX_PRESENTATION_DEPTH:
                presentation_overflow = recognized_opening or raw_word.startswith("![", opening_offset)
                break
            if character == "!" and raw_word.startswith("![", opening_offset):
                opening_offset += 1
                continue
            if character in "*_~`":
                marker_end = opening_offset + 1
                while marker_end < len(raw_word) and raw_word[marker_end] == character:
                    marker_end += 1
                if token_match.start() + opening_offset not in matched_presentation_openers:
                    break
                token_closers.append(raw_word[opening_offset:marker_end])
                opening_offset = marker_end
                continue
            closing = next(
                (closing for opening, closing in ENCLOSING_BOUNDARY_PAIRS if opening == character),
                None,
            )
            if closing is None:
                break
            if token_match.start() + opening_offset not in matched_presentation_openers:
                break
            token_closers.append(closing)
            opening_offset += 1

        presentation_closers.extend(token_closers)
        raw_word = _strip_closed_presentation_regions(raw_word, presentation_closers, opening_offset)

        source_legacy_word = source_word.replace("*", "").replace("#", "")
        legacy_word = raw_word.replace("*", "").replace("#", "")
        word = _normalize_allowed_token(raw_word).strip("*#")
        decoded_word = _normalize_allowed_token(unquote(word))
        whole_word = legacy_word if re.search(r"https?://[^\s]+", legacy_word, re.IGNORECASE) else decoded_word
        word_findings: list[str] = []
        if presentation_overflow:
            word_findings.append(source_word)
        markdown_boundary = raw_word.find("](")
        if markdown_boundary >= 0:
            label_candidate = _normalize_allowed_token(raw_word[:markdown_boundary])
            if _is_secret_candidate(label_candidate, cfg, custom_regex):
                word_findings.append(label_candidate)
            if _contains_allowed_pattern(label_candidate):
                label_candidates, label_status, label_groups = extract_embedded(
                    label_candidate,
                    cacheable=source_word_counts[source_word] > 1,
                )
                _extend_occurrence_findings(
                    word_findings,
                    _canonical_embedded_findings(label_candidates, label_groups, cfg, custom_regex),
                )
                if label_status.structural_overflow and not word_findings:
                    word_findings.append(label_candidate)
                custom_scan_incomplete = custom_scan_incomplete or label_status.custom_incomplete

        whole_word_matched = False
        if _matches_custom_pattern(source_legacy_word, custom_regex):
            word_findings.append(source_legacy_word)
            whole_word_matched = True
        elif _is_secret_candidate(whole_word, cfg, custom_regex):
            word_findings.append(whole_word)
            whole_word_matched = True

        if not whole_word_matched and (
            _contains_allowed_pattern(word) or _contains_allowed_pattern(decoded_word) or _contains_allowed_pattern(legacy_word)
        ):
            raw_has_url = re.search(r"https?://[^\s]+", raw_word, re.IGNORECASE) is not None
            decoded_has_url = re.search(r"https?://[^\s]+", decoded_word, re.IGNORECASE) is not None
            embedded_word = decoded_word if decoded_has_url and not raw_has_url else raw_word
            candidates, scan_status, candidate_groups = extract_embedded(
                embedded_word,
                cacheable=source_word_counts[source_word] > 1,
            )
            _extend_occurrence_findings(
                word_findings,
                _canonical_embedded_findings(candidates, candidate_groups, cfg, custom_regex),
            )
            if scan_status.structural_overflow and not word_findings:
                word_findings.append(word)
            custom_scan_incomplete = custom_scan_incomplete or scan_status.custom_incomplete

        secrets.extend(word_findings)

    info: dict[str, Any] = {
        "guardrail_name": "Secret Keys",
        "detected_secrets": secrets,
    }
    if custom_scan_incomplete:
        info["custom_scan_incomplete"] = True
    return GuardrailResult(tripwire_triggered=bool(secrets), info=info)


async def secret_keys(
    ctx: Any,
    data: str,
    config: SecretKeysCfg,
) -> GuardrailResult:
    """Async guardrail function for secret key and credential detection.

    Scans the input for likely secrets or credentials (e.g., API keys, tokens)
    using entropy, diversity, and pattern rules.

    Args:
        ctx (Any): Guardrail context (unused).
        data (str): Input text to scan.
        config (SecretKeysCfg): Configuration for secret detection.

    Returns:
        GuardrailResult: Indicates if secrets were detected, with findings in info.
    """
    _ = ctx
    custom_regex_snapshot = tuple(config.custom_regex) if config.custom_regex is not None else None
    validated_config = SecretKeysCfg(
        threshold=config.threshold,
        custom_regex=list(custom_regex_snapshot) if custom_regex_snapshot is not None else None,
    )
    cfg = CONFIGS[validated_config.threshold]
    return await asyncio.to_thread(_detect_secret_keys, data, cfg, validated_config.custom_regex)


default_spec_registry.register(
    name="Secret Keys",
    check_fn=secret_keys,
    description=("Checks that the text does not contain potential API keys, secrets, or other credentials."),
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="RegEx"),
)
