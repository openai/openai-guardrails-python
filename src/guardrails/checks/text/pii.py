"""PII detection guardrail for sensitive text content.

This module implements a guardrail for detecting Personally Identifiable
Information (PII) in text using the Presidio analyzer. It defines the config
schema for entity selection, output/result structures, and the async guardrail
check_fn for runtime enforcement.

The guardrail supports two modes of operation:
- **Blocking mode** (block=True): Triggers tripwire when PII is detected, blocking the request
- **Masking mode** (block=False): Automatically masks PII with placeholder tokens without blocking

**IMPORTANT: PII masking is only supported in the pre-flight stage.**
- Use `block=False` (masking mode) in pre-flight to automatically mask PII from user input
- Use `block=True` (blocking mode) in output stage to prevent PII exposure in LLM responses
- Masking in output stage is not supported and will not work as expected

When used in pre-flight stage with masking mode, the masked text is automatically
passed to the LLM instead of the original text containing PII.

Classes:
    PIIEntity: Enum of supported PII entity types across global regions.
    PIIConfig: Pydantic config model specifying what entities to detect and behavior mode.
    PiiDetectionResult: Internal container for mapping entity types to findings.

Functions:
    pii: Async guardrail check_fn for PII detection.

Configuration Parameters:
    `entities` (list[PIIEntity]): List of PII entity types to detect.
    `block` (bool): If True, triggers tripwire when PII is detected (blocking behavior).
                   If False, only masks PII without blocking (masking behavior, default).
                   **Note: Masking only works in pre-flight stage. Use block=True for output stage.**

    Supported entities include:

    - "US_SSN": US Social Security Numbers
    - "PHONE_NUMBER": Phone numbers in various formats
    - "EMAIL_ADDRESS": Email addresses
    - "CREDIT_CARD": Credit card numbers
    - "US_BANK_ACCOUNT": US bank account numbers
    - And many more. See the full list at: [Presidio Supported Entities](https://microsoft.github.io/presidio/supported_entities/)

Example:
```python
    # Masking mode (default) - USE ONLY IN PRE-FLIGHT STAGE
    >>> config = PIIConfig(
    ...     entities=[PIIEntity.US_SSN, PIIEntity.EMAIL_ADDRESS],
    ...     block=False
    ... )
    >>> result = await pii(None, "Contact me at john@example.com, SSN: 111-22-3333", config)
    >>> result.tripwire_triggered
    False
    >>> result.info["checked_text"]
    "Contact me at <EMAIL_ADDRESS>, SSN: <US_SSN>"

    # Blocking mode - USE IN OUTPUT STAGE TO PREVENT PII EXPOSURE
    >>> config = PIIConfig(
    ...     entities=[PIIEntity.US_SSN, PIIEntity.EMAIL_ADDRESS],
    ...     block=True
    ... )
    >>> result = await pii(None, "Contact me at john@example.com, SSN: 111-22-3333", config)
    >>> result.tripwire_triggered
    True
```

Usage Guidelines:
    - PRE-FLIGHT STAGE: Use block=False for automatic PII masking of user input
    - OUTPUT STAGE: Use block=True to prevent PII exposure in LLM responses
    - Masking in output stage is not supported and will not work as expected
"""

from __future__ import annotations

import base64
import binascii
import functools
import logging
import re
import unicodedata
import urllib.parse
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_analyzer.predefined_recognizers.country_specific.korea.kr_rrn_recognizer import (
    KrRrnRecognizer,
)
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from pydantic import BaseModel, ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["pii"]

logger = logging.getLogger(__name__)

# Zero-width and invisible Unicode characters that can be used to bypass detection
_ZERO_WIDTH_CHARS = re.compile(
    r"[\u200b\u200c\u200d\u2060\ufeff]"  # Zero-width space, ZWNJ, ZWJ, word joiner, BOM
)

# Patterns for detecting encoded content
# Note: Hex must be checked BEFORE Base64 since hex strings can match Base64 pattern
_HEX_PATTERN = re.compile(r"\b[0-9a-fA-F]{24,}\b")  # Reduced from 32 to 24 (12 bytes min)
_BASE64_PATTERN = re.compile(r"(?:data:|base64,)?([A-Za-z0-9+/]{16,}={0,2})")  # Handle data: URI, min 16 chars
_URL_ENCODED_PATTERN = re.compile(r"(?:%[0-9A-Fa-f]{2})+")  # Match all consecutive sequences


@functools.lru_cache(maxsize=1)
def _get_analyzer_engine() -> AnalyzerEngine:
    """Return a cached AnalyzerEngine configured with Presidio recognizers.

    The engine loads Presidio's default recognizers for English and explicitly
    registers custom recognizers for KR_RRN, CVV/CVC codes, and BIC/SWIFT codes.

    Returns:
        AnalyzerEngine: Analyzer configured with English NLP support and
        region-specific recognizers backed by Presidio.
    """
    nlp_config: Final[dict[str, Any]] = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_sm"},
        ],
    }

    provider = NlpEngineProvider(nlp_configuration=nlp_config)
    nlp_engine = provider.create_engine()

    registry = RecognizerRegistry(supported_languages=["en"])
    registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)

    # Add custom recognizers
    registry.add_recognizer(KrRrnRecognizer(supported_language="en"))

    # CVV/CVC recognizer (3-4 digits, often near credit card context)
    cvv_pattern = Pattern(
        name="cvv_pattern",
        regex=r"\b(?:cvv|cvc|security\s*code|card\s*code)[:\s=]*(\d{3,4})\b",
        score=0.85,
    )
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="CVV",
            patterns=[cvv_pattern],
            supported_language="en",
        )
    )

    # BIC/SWIFT code recognizer (8 or 11 characters: 4 bank + 2 country + 2 location + 3 branch)
    bic_pattern = Pattern(
        name="bic_swift_pattern",
        regex=r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}([A-Z0-9]{3})?\b",
        score=0.75,
    )
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="BIC_SWIFT",
            patterns=[bic_pattern],
            supported_language="en",
        )
    )

    # Email in URL/query parameter context (Presidio's default fails in these contexts)
    # Matches: user=john@example.com, email=test@domain.org, etc.
    # Uses lookbehind to avoid capturing delimiters
    email_in_url_pattern = Pattern(
        name="email_in_url_pattern",
        regex=r"(?<=[\?&=\/])[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        score=0.9,
    )
    registry.add_recognizer(
        PatternRecognizer(
            supported_entity="EMAIL_ADDRESS",
            patterns=[email_in_url_pattern],
            supported_language="en",
        )
    )

    engine = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )
    return engine


@functools.lru_cache(maxsize=1)
def _get_anonymizer_engine() -> AnonymizerEngine:
    """Return a cached AnonymizerEngine for PII masking.

    Uses Presidio's built-in anonymization for optimal performance and
    correct handling of overlapping entities, Unicode, and special characters.

    Returns:
        AnonymizerEngine: Configured anonymizer for replacing PII entities.
    """
    return AnonymizerEngine()


class PIIEntity(str, Enum):
    """Supported PII entity types for detection.

    Includes global and region-specific types (US, UK, Spain, Italy, etc.).
    These map to Presidio's supported entity labels, plus custom recognizers.

    Example values: "US_SSN", "EMAIL_ADDRESS", "IP_ADDRESS", "IN_PAN", etc.
    """

    # Global
    CREDIT_CARD = "CREDIT_CARD"
    CRYPTO = "CRYPTO"
    DATE_TIME = "DATE_TIME"
    EMAIL_ADDRESS = "EMAIL_ADDRESS"
    IBAN_CODE = "IBAN_CODE"
    IP_ADDRESS = "IP_ADDRESS"
    NRP = "NRP"
    LOCATION = "LOCATION"
    PERSON = "PERSON"
    PHONE_NUMBER = "PHONE_NUMBER"
    MEDICAL_LICENSE = "MEDICAL_LICENSE"
    URL = "URL"

    # Custom recognizers
    CVV = "CVV"
    BIC_SWIFT = "BIC_SWIFT"

    # USA
    US_BANK_NUMBER = "US_BANK_NUMBER"
    US_DRIVER_LICENSE = "US_DRIVER_LICENSE"
    US_ITIN = "US_ITIN"
    US_PASSPORT = "US_PASSPORT"
    US_SSN = "US_SSN"

    # UK
    UK_NHS = "UK_NHS"
    UK_NINO = "UK_NINO"

    # Spain
    ES_NIF = "ES_NIF"
    ES_NIE = "ES_NIE"

    # Italy
    IT_FISCAL_CODE = "IT_FISCAL_CODE"
    IT_DRIVER_LICENSE = "IT_DRIVER_LICENSE"
    IT_VAT_CODE = "IT_VAT_CODE"
    IT_PASSPORT = "IT_PASSPORT"
    IT_IDENTITY_CARD = "IT_IDENTITY_CARD"

    # Poland
    PL_PESEL = "PL_PESEL"

    # Singapore
    SG_NRIC_FIN = "SG_NRIC_FIN"
    SG_UEN = "SG_UEN"

    # Australia
    AU_ABN = "AU_ABN"
    AU_ACN = "AU_ACN"
    AU_TFN = "AU_TFN"
    AU_MEDICARE = "AU_MEDICARE"

    # India
    IN_PAN = "IN_PAN"
    IN_AADHAAR = "IN_AADHAAR"
    IN_VEHICLE_REGISTRATION = "IN_VEHICLE_REGISTRATION"
    IN_VOTER = "IN_VOTER"
    IN_PASSPORT = "IN_PASSPORT"

    # Finland
    FI_PERSONAL_IDENTITY_CODE = "FI_PERSONAL_IDENTITY_CODE"

    # Korea
    KR_RRN = "KR_RRN"


class PIIConfig(BaseModel):
    """Configuration schema for PII detection.

    Used to control which entity types are checked and whether to block content
    containing PII or just mask it.

    Attributes:
        entities (list[PIIEntity]): List of PII entity types to detect. See the full list at: [Presidio Supported Entities](https://microsoft.github.io/presidio/supported_entities/)
        block (bool): If True, triggers tripwire when PII is detected (blocking behavior).
                     If False, only masks PII without blocking.
                     Defaults to False.
        detect_encoded_pii (bool): If True, detects PII in Base64/URL-encoded/hex strings.
                                  Adds ~30-40ms latency but catches obfuscated PII.
                     Defaults to False.
    """

    entities: list[PIIEntity] = Field(
        default_factory=lambda: list(PIIEntity),
        description="Entity types to detect (e.g., US_SSN, EMAIL_ADDRESS, etc.).",
    )
    block: bool = Field(
        default=False,
        description="If True, triggers tripwire when PII is detected (blocking mode). If False, masks PII without blocking (masking mode, only works in pre-flight stage).",  # noqa: E501
    )
    detect_encoded_pii: bool = Field(
        default=False,
        description="If True, detects PII in encoded content (Base64, URL-encoded, hex). Adds latency but improves security.",  # noqa: E501
    )

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class PiiDetectionResult:
    """Internal result structure for PII detection.

    Attributes:
        mapping (dict[str, list[str]]): Mapping from entity type to list of detected strings.
        analyzer_results (Sequence[RecognizerResult]): Raw analyzer results for position information.
        encoded_detections (dict[str, list[str]] | None): Optional mapping of encoded PII detections.
    """

    mapping: dict[str, list[str]]
    analyzer_results: Sequence[RecognizerResult]
    encoded_detections: dict[str, list[str]] | None = None

    def to_dict(self) -> dict[str, list[str]]:
        """Convert the result to a dictionary.

        Returns:
            dict[str, list[str]]: A copy of the entity mapping.
        """
        return {k: v.copy() for k, v in self.mapping.items()}

    def has_pii(self) -> bool:
        """Check if any PII was detected (plain or encoded).

        Returns:
            bool: True if PII was detected.
        """
        return bool(self.mapping) or bool(self.encoded_detections)


def _detect_pii(text: str, config: PIIConfig) -> PiiDetectionResult:
    """Run Presidio analysis and collect findings by entity type.

    Applies Unicode normalization before analysis to prevent bypasses using
    fullwidth characters or zero-width spaces. This ensures that obfuscation
    techniques cannot evade PII detection.

    Supports detection of Korean (KR_RRN) and other region-specific entities via
    Presidio recognizers registered with the analyzer engine.

    Args:
        text (str): The text to analyze for PII.
        config (PIIConfig): PII detection configuration.

    Returns:
        PiiDetectionResult: Object containing mapping of entities to detected snippets.

    Raises:
        ValueError: If text is empty or None.
    """
    if not text:
        raise ValueError("Text cannot be empty or None")

    # Normalize Unicode to prevent detection bypasses
    normalized_text = _normalize_unicode(text)

    engine = _get_analyzer_engine()

    # Run analysis for all configured entities
    # Region-specific recognizers (e.g., KR_RRN) are registered with language="en"
    entity_values = [e.value for e in config.entities]
    analyzer_results = engine.analyze(normalized_text, entities=entity_values, language="en")

    # Group results by entity type
    # Note: No filtering needed as engine.analyze already returns only requested entities
    grouped: dict[str, list[str]] = defaultdict(list)
    for res in analyzer_results:
        grouped[res.entity_type].append(normalized_text[res.start : res.end])

    return PiiDetectionResult(mapping=dict(grouped), analyzer_results=analyzer_results)


def _normalize_unicode(text: str) -> str:
    """Normalize Unicode text to prevent detection bypasses.

    Applies NFKC normalization to convert fullwidth and other variant characters
    to their canonical forms, then strips zero-width characters that could be
    used to corrupt detection spans.

    Security rationale:
    - Fullwidth characters (e.g., ＠ → @, ０ → 0) bypass regex patterns
    - Zero-width spaces (\u200b) corrupt entity spans and cause leaks
    - NFKC normalization handles ligatures, superscripts, circled chars, etc.

    Args:
        text (str): The text to normalize.

    Returns:
        str: Normalized text safe for PII detection.

    Examples:
        >>> _normalize_unicode("test＠example．com")  # Fullwidth @ and .
        'test@example.com'
        >>> _normalize_unicode("192\u200b.168.1.1")  # Zero-width space in IP
        '192.168.1.1'
    """
    if not text:
        return text

    # Step 1: NFKC normalization converts fullwidth → ASCII and decomposes ligatures
    normalized = unicodedata.normalize("NFKC", text)

    # Step 2: Strip zero-width and invisible characters
    cleaned = _ZERO_WIDTH_CHARS.sub("", normalized)

    return cleaned


@dataclass(frozen=True, slots=True)
class EncodedCandidate:
    """Represents a potentially encoded string found in text.

    Attributes:
        encoded_text: The encoded string as it appears in original text.
        decoded_text: The decoded version (may be None if decoding failed).
        encoding_type: Type of encoding (base64, url, hex).
        start: Start position in original text.
        end: End position in original text.
    """

    encoded_text: str
    decoded_text: str | None
    encoding_type: str
    start: int
    end: int


def _try_decode_base64(text: str) -> str | None:
    """Attempt to decode Base64 string.

    Limits decoded output to 10KB to prevent DoS attacks via memory exhaustion.
    Fails closed: raises error if decoded content exceeds limit to prevent PII leaks.

    Args:
        text: String that looks like Base64.

    Returns:
        Decoded string if valid and under size limit, None if invalid encoding.

    Raises:
        ValueError: If decoded content exceeds 10KB (security limit).
    """
    try:
        decoded_bytes = base64.b64decode(text, validate=True)
        # Security: Fail closed - reject content > 10KB to prevent memory DoS and PII bypass
        if len(decoded_bytes) > 10_000:
            msg = (
                f"Base64 decoded content too large ({len(decoded_bytes):,} bytes). Maximum allowed is 10KB."
            )
            raise ValueError(msg)
        # Check if result is valid UTF-8
        return decoded_bytes.decode("utf-8", errors="strict")
    except (binascii.Error, UnicodeDecodeError):
        return None


def _try_decode_hex(text: str) -> str | None:
    """Attempt to decode hex string.

    Limits decoded output to 10KB to prevent DoS attacks via memory exhaustion.
    Fails closed: raises error if decoded content exceeds limit to prevent PII leaks.

    Args:
        text: String that looks like hex.

    Returns:
        Decoded string if valid and under size limit, None if invalid encoding.

    Raises:
        ValueError: If decoded content exceeds 10KB (security limit).
    """
    try:
        decoded_bytes = bytes.fromhex(text)
    except ValueError:
        # Invalid hex string - return None
        return None

    # Security: Fail closed - reject content > 10KB to prevent memory DoS and PII bypass
    if len(decoded_bytes) > 10_000:
        msg = f"Hex decoded content too large ({len(decoded_bytes):,} bytes). Maximum allowed is 10KB."
        raise ValueError(msg)

    try:
        return decoded_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None


def _build_decoded_text(text: str) -> tuple[str, list[EncodedCandidate]]:
    """Build a fully decoded version of text by decoding all encoded chunks.

    Strategy:
    1. Find all encoded chunks (Hex, Base64, URL)
    2. Decode each chunk in place to build a fully decoded sentence
    3. Track mappings from decoded positions → original encoded spans

    This handles partial encodings like %6a%61%6e%65%40securemail.net → jane@securemail.net

    Args:
        text: Text that may contain encoded chunks.

    Returns:
        Tuple of (decoded_text, candidates_with_positions)
    """
    candidates = []
    used_spans = set()

    # Find hex candidates FIRST (most specific pattern)
    for match in _HEX_PATTERN.finditer(text):
        decoded = _try_decode_hex(match.group())
        if decoded and len(decoded) > 3:
            candidates.append(
                EncodedCandidate(
                    encoded_text=match.group(),
                    decoded_text=decoded,
                    encoding_type="hex",
                    start=match.start(),
                    end=match.end(),
                )
            )
            used_spans.add((match.start(), match.end()))

    # Find Base64 candidates
    for match in _BASE64_PATTERN.finditer(text):
        if (match.start(), match.end()) in used_spans:
            continue

        b64_string = match.group(1) if match.lastindex else match.group()
        decoded = _try_decode_base64(b64_string)
        if decoded and len(decoded) > 3:
            candidates.append(
                EncodedCandidate(
                    encoded_text=match.group(),
                    decoded_text=decoded,
                    encoding_type="base64",
                    start=match.start(),
                    end=match.end(),
                )
            )
            used_spans.add((match.start(), match.end()))

    # Build fully decoded text by replacing Hex and Base64 chunks first
    candidates.sort(key=lambda c: c.start, reverse=True)
    decoded_text = text
    for candidate in candidates:
        if candidate.decoded_text:
            decoded_text = decoded_text[: candidate.start] + candidate.decoded_text + decoded_text[candidate.end :]

    # URL decode the ENTIRE text (handles partial encodings like %6a%61%6e%65%40securemail.net)
    # This must happen AFTER Base64/Hex replacement to handle mixed encodings correctly
    url_decoded = urllib.parse.unquote(decoded_text)

    # If URL decoding changed the text, track encoded spans for masking
    if url_decoded != decoded_text:
        # Find URL-encoded spans in the ORIGINAL text for masking purposes
        for match in _URL_ENCODED_PATTERN.finditer(text):
            if any(start <= match.start() < end or start < match.end() <= end for start, end in used_spans):
                continue

            decoded_chunk = urllib.parse.unquote(match.group())
            if decoded_chunk != match.group():
                candidates.append(
                    EncodedCandidate(
                        encoded_text=match.group(),
                        decoded_text=decoded_chunk,
                        encoding_type="url",
                        start=match.start(),
                        end=match.end(),
                    )
                )
        decoded_text = url_decoded

    return decoded_text, candidates


def _mask_pii(text: str, detection: PiiDetectionResult, config: PIIConfig) -> tuple[str, dict[str, list[str]]]:
    """Mask detected PII using Presidio's AnonymizerEngine.

    Normalizes Unicode before masking to ensure consistency with detection.
    Uses Presidio's built-in anonymization for optimal performance and
    correct handling of overlapping entities, Unicode, and special characters.

    If detect_encoded_pii is enabled, also detects and masks PII in
    Base64/URL-encoded/hex strings using a hybrid approach.

    Args:
        text (str): The text to mask.
        detection (PiiDetectionResult): Results from PII detection.
        config (PIIConfig): PII detection configuration.

    Returns:
        Tuple of (masked_text, encoded_detections_mapping).

    Raises:
        ValueError: If text is empty or None.
    """
    if not text:
        raise ValueError("Text cannot be empty or None")

    # Normalize Unicode to match detection normalization
    normalized_text = _normalize_unicode(text)

    if not detection.analyzer_results:
        # Check encoded content even if no direct PII found
        if config.detect_encoded_pii:
            masked_text, encoded_detections = _mask_encoded_pii(normalized_text, config, original_text=text)
            # If no encoded PII found, return original text to preserve special characters
            if not encoded_detections:
                return text, {}
            return masked_text, encoded_detections
        # No PII detected - return original text to preserve special characters
        return text, {}

    # Use Presidio's optimized anonymizer with replace operator
    anonymizer = _get_anonymizer_engine()

    # Create operators mapping each entity type to a replace operator
    operators = {entity_type: OperatorConfig("replace", {"new_value": f"<{entity_type}>"}) for entity_type in detection.mapping.keys()}

    result = anonymizer.anonymize(
        text=normalized_text,
        analyzer_results=detection.analyzer_results,
        operators=operators,
    )

    masked_text = result.text
    encoded_detections = {}

    # If enabled, also check for encoded PII
    if config.detect_encoded_pii:
        masked_text, encoded_detections = _mask_encoded_pii(masked_text, config)

    return masked_text, encoded_detections


def _mask_encoded_pii(text: str, config: PIIConfig, original_text: str | None = None) -> tuple[str, dict[str, list[str]]]:
    """Detect and mask PII in encoded content (Base64, URL-encoded, hex).

    Strategy:
    1. Build fully decoded text by decoding all encoded chunks in place
    2. Pass the entire decoded text to Presidio once
    3. Map detections back to mask the encoded versions in original text

    Args:
        text: Normalized text potentially containing encoded PII.
        config: PII configuration specifying which entities to detect.
        original_text: Original (non-normalized) text to return if no PII found.

    Returns:
        Tuple of (masked_text, encoded_detections_mapping).
        Returns original_text if provided and no PII found, otherwise text.
    """
    # Build fully decoded text and get candidate mappings
    decoded_text, candidates = _build_decoded_text(text)

    if not candidates:
        return original_text or text, {}

    # Pass fully decoded text to Presidio ONCE
    engine = _get_analyzer_engine()
    analyzer_results = engine.analyze(decoded_text, entities=[e.value for e in config.entities], language="en")

    if not analyzer_results:
        return original_text or text, {}

    # Map detections back to encoded chunks in original text
    # Strategy: Check if the decoded chunk contributed to any PII detection
    masked_text = text
    encoded_detections: dict[str, list[str]] = defaultdict(list)

    # For each candidate, check if any PII was detected that includes its decoded content
    # Sort candidates by start position (reverse) to mask from end to start
    for candidate in sorted(candidates, key=lambda c: c.start, reverse=True):
        if not candidate.decoded_text:
            continue

        found_entities = set()
        for res in analyzer_results:
            detected_value = decoded_text[res.start : res.end]
            candidate_lower = candidate.decoded_text.lower()
            detected_lower = detected_value.lower()

            # Check if candidate's decoded text overlaps with the detection
            # Handle partial encodings where encoded span may include extra characters
            # e.g., %3A%6a%6f%65%40 → ":joe@" but only "joe@" is in email "joe@domain.com"
            has_overlap = (
                candidate_lower in detected_lower  # Candidate is substring of detection
                or detected_lower in candidate_lower  # Detection is substring of candidate
                or (
                    len(candidate_lower) >= 3
                    and any(  # Any 3-char chunk overlaps
                        candidate_lower[i : i + 3] in detected_lower
                        for i in range(0, len(candidate_lower) - 2, 2)  # Step by 2 for efficiency
                    )
                )
            )

            if has_overlap:
                found_entities.add(res.entity_type)
                encoded_detections[res.entity_type].append(candidate.encoded_text)

        if found_entities:
            # Mask the encoded version in original text
            entity_marker = f"<{next(iter(found_entities))}_ENCODED>"
            masked_text = masked_text[: candidate.start] + entity_marker + masked_text[candidate.end :]

    return masked_text, dict(encoded_detections)


def _as_result(detection: PiiDetectionResult, config: PIIConfig, name: str, text: str) -> GuardrailResult:
    """Convert detection results to a GuardrailResult for reporting.

    Args:
        detection (PiiDetectionResult): Results of the PII scan.
        config (PIIConfig): Original detection configuration.
        name (str): Name for the guardrail in result metadata.
        text (str): Original input text for masking.

    Returns:
        GuardrailResult: Always includes checked_text. Triggers tripwire only if
        PII found AND block=True.
    """
    # Mask the text (returns masked text and any encoded detections)
    checked_text, encoded_detections = _mask_pii(text, detection, config) if detection.mapping or config.detect_encoded_pii else (text, {})

    # Merge plain and encoded detections
    all_detections = dict(detection.mapping)
    for entity_type, values in encoded_detections.items():
        if entity_type in all_detections:
            all_detections[entity_type].extend(values)
        else:
            all_detections[entity_type] = values

    # Only trigger tripwire if PII is found AND block=True
    has_pii = bool(all_detections)
    tripwire_triggered = has_pii and config.block

    return GuardrailResult(
        tripwire_triggered=tripwire_triggered,
        info={
            "guardrail_name": name,
            "detected_entities": all_detections,
            "entity_types_checked": config.entities,
            "checked_text": checked_text,
            "block_mode": config.block,
            "pii_detected": has_pii,
            "detect_encoded_pii": config.detect_encoded_pii,
        },
    )


async def pii(
    ctx: Any,
    data: str,
    config: PIIConfig,
) -> GuardrailResult:
    """Async guardrail check_fn for PII entity detection in text.

    Analyzes text for any configured PII entity types and reports results.
    Behavior depends on the `block` configuration:

    - If `block=True`: Triggers tripwire when PII is detected (blocking behavior)
    - If `block=False`: Only masks PII without blocking (masking behavior, default)

    **IMPORTANT: PII masking (block=False) only works in pre-flight stage.**
    - Use masking mode in pre-flight to automatically clean user input
    - Use blocking mode in output stage to prevent PII exposure in LLM responses
    - Masking in output stage will not work as expected

    Args:
        ctx (Any): Guardrail runtime context (unused).
        data (str): The input text to scan.
        config (PIIConfig): Guardrail configuration for PII detection.

    Returns:
        GuardrailResult: Indicates if PII was found and whether to block based on config.
                        Always includes checked_text in the info.

    Raises:
        ValueError: If input text is empty or None.
    """
    _ = ctx
    result = _detect_pii(data, config)
    return _as_result(result, config, "Contains PII", data)


default_spec_registry.register(
    name="Contains PII",
    check_fn=pii,
    description=(
        "Checks that the text does not contain personally identifiable information (PII) such as "
        "SSNs, phone numbers, credit card numbers, etc., based on configured entity types."
    ),
    media_type="text/plain",
    metadata=GuardrailSpecMetadata(engine="Presidio"),
)
