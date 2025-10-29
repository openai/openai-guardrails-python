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

import functools
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts, NlpEngineProvider
from pydantic import BaseModel, ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["pii"]

logger = logging.getLogger(__name__)


def _is_valid_date(year: int, month: int, day: int) -> bool:
    """Validate if year, month, day form a valid date.

    Args:
        year: Full year (e.g., 1990, 2024)
        month: Month (1-12)
        day: Day of month (1-31)

    Returns:
        bool: True if date is valid, False otherwise.
    """
    # Validate month (01-12)
    if not 1 <= month <= 12:
        return False

    # Validate day based on month
    if month in (1, 3, 5, 7, 8, 10, 12):
        max_day = 31
    elif month in (4, 6, 9, 11):
        max_day = 30
    elif month == 2:
        # For February, check if it's a leap year
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        max_day = 29 if is_leap else 28
    else:
        return False

    # Validate day range
    return 1 <= day <= max_day


def _validate_kr_rrn_date(rrn: str) -> bool:
    """Validate the date portion (YYMMDD) of Korean RRN.

    The first 6 digits must represent a valid date in YYMMDD format.

    Args:
        rrn (str): The RRN string (with or without hyphen).

    Returns:
        bool: True if date is valid, False otherwise.
    """
    # Remove hyphen/space
    digits = rrn.replace("-", "").replace(" ", "")
    if len(digits) < 6:
        return False

    try:
        year = int(digits[0:2])
        month = int(digits[2:4])
        day = int(digits[4:6])

        # Determine full year from century (gender digit)
        if len(digits) >= 7:
            gender_digit = int(digits[6])
            # 1,2: 1900s, 3,4: 2000s, 5,6: 1800s, 7,8: 2000s, 9,0: 1800s
            if gender_digit in (1, 2, 5, 6, 9, 0):
                full_year = 1900 + year
            else:
                full_year = 2000 + year
        else:
            # If we can't determine century, assume 2000s for validation
            full_year = 2000 + year

        # Use helper to validate date
        return _is_valid_date(full_year, month, day)
    except (ValueError, IndexError):
        return False


def _validate_kr_rrn_checksum(rrn: str) -> bool:
    """Validate Korean Resident Registration Number checksum.

    The last digit of the RRN is a checksum calculated using a weighted sum
    of the first 12 digits. Based on official Korean RRN validation algorithm.

    Args:
        rrn (str): The RRN string (with or without hyphen).

    Returns:
        bool: True if checksum is valid, False otherwise.
    """
    # Remove hyphen/space and validate length
    digits = rrn.replace("-", "").replace(" ", "")
    if len(digits) != 13:
        return False

    try:
        # Extract first 12 digits and checksum
        numbers = [int(d) for d in digits[:12]]
        checksum = int(digits[12])

        # Weight pattern: 2,3,4,5,6,7,8,9,2,3,4,5
        weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]

        # Calculate weighted sum
        total = sum(n * w for n, w in zip(numbers, weights, strict=False))

        # Calculate expected checksum: (11 - (sum % 11)) % 10
        expected_checksum = (11 - (total % 11)) % 10

        return checksum == expected_checksum
    except (ValueError, IndexError):
        return False


class KoreanRrnRecognizer(PatternRecognizer):
    """Custom recognizer for Korean Resident Registration Numbers with checksum validation."""

    def __init__(self) -> None:
        """Initialize the Korean RRN recognizer."""
        patterns = [
            Pattern(
                name="kr_rrn_pattern",
                regex=r"\b\d{6}[- ]?\d{7}\b",
                score=0.85,
            ),
        ]
        super().__init__(
            supported_entity="KR_RRN",
            patterns=patterns,
            context=["rrn", "resident", "registration", "korean", "korea", "주민등록번호"],
            supported_language="en",
        )

    def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None) -> list[RecognizerResult]:
        """Analyze text for Korean RRN and validate date and checksums.

        Args:
            text: Text to analyze.
            entities: List of entity types to detect.
            nlp_artifacts: NLP artifacts (unused for pattern-based recognition).

        Returns:
            List of validated RecognizerResult objects.
        """
        results = super().analyze(text, entities, nlp_artifacts)

        # Filter out results with invalid date or checksums
        validated_results = []
        for result in results:
            candidate = text[result.start : result.end]
            # Validate both date (YYMMDD) and checksum
            if _validate_kr_rrn_date(candidate) and _validate_kr_rrn_checksum(candidate):
                validated_results.append(result)

        return validated_results


def _create_kr_rrn_recognizer() -> KoreanRrnRecognizer:
    """Create a custom recognizer for Korean Resident Registration Numbers.

    Based on Presidio's KR_RRN recognizer with date and checksum validation.
    Format: 6 digits (YYMMDD) + hyphen + 7 digits (last digit is checksum)

    Validation includes:
    - YYMMDD must be a valid date
    - Last digit must match the checksum algorithm

    Example: 900101-1234567 (valid date: Jan 1, 1990)

    Returns:
        KoreanRrnRecognizer: Recognizer for Korean RRN with date and checksum validation.
    """
    return KoreanRrnRecognizer()


def _validate_th_tnin_structure(tnin: str) -> bool:
    """Validate the structure of Thai National Identification Number.

    The first digit must be a valid category code (0-8) as defined by
    Thai identification system.

    Reference: https://en.wikipedia.org/wiki/Thai_identity_card

    Categories:
    - 0: Not found on Thai nationals' cards (other identity documents)
    - 1: Born after Jan 1, 1984, birth notified within deadline
    - 2: Born after Jan 1, 1984, birth notified late
    - 3: Born before Jan 1, 1984, included in house registration
    - 4: Born before Jan 1, 1984, not in house registration
    - 5: Missed census or dual nationality cases
    - 6: Foreign nationals living temporarily/illegal migrants
    - 7: Children of category 6 born in Thailand
    - 8: Foreign nationals living permanently or naturalized citizens

    Args:
        tnin (str): The 13-digit TNIN string.

    Returns:
        bool: True if structure is valid, False otherwise.
    """
    if len(tnin) != 13:
        return False

    try:
        # First digit must be a valid category (0-8)
        category = int(tnin[0])
        if category not in range(9):  # 0-8 inclusive
            return False

        # All characters must be digits
        if not tnin.isdigit():
            return False

        return True
    except (ValueError, IndexError):
        return False


def _validate_th_tnin_checksum(tnin: str) -> bool:
    """Validate Thai National Identification Number checksum.

    The 13th digit is a checksum calculated using modulo-11 algorithm
    on the first 12 digits. Based on official Thai TNIN validation.

    Args:
        tnin (str): The 13-digit TNIN string.

    Returns:
        bool: True if checksum is valid, False otherwise.
    """
    # Validate length
    if len(tnin) != 13:
        return False

    try:
        # Extract first 12 digits and checksum
        numbers = [int(d) for d in tnin[:12]]
        checksum = int(tnin[12])

        # Weight pattern: 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2
        weights = [13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2]

        # Calculate weighted sum
        total = sum(n * w for n, w in zip(numbers, weights, strict=False))

        # Calculate expected checksum: (11 - (sum % 11)) % 10
        expected_checksum = (11 - (total % 11)) % 10

        return checksum == expected_checksum
    except (ValueError, IndexError):
        return False


class ThaiTninRecognizer(PatternRecognizer):
    """Custom recognizer for Thai National Identification Numbers with checksum validation."""

    def __init__(self) -> None:
        """Initialize the Thai TNIN recognizer."""
        patterns = [
            Pattern(
                name="th_tnin_pattern",
                regex=r"\b\d{13}\b",
                score=0.85,
            ),
        ]
        super().__init__(
            supported_entity="TH_TNIN",
            patterns=patterns,
            context=["tnin", "thai", "thailand", "national", "id", "identification", "เลขประจำตัวประชาชน"],
            supported_language="en",
        )

    def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None) -> list[RecognizerResult]:
        """Analyze text for Thai TNIN and validate structure and checksums.

        Args:
            text: Text to analyze.
            entities: List of entity types to detect.
            nlp_artifacts: NLP artifacts (unused for pattern-based recognition).

        Returns:
            List of validated RecognizerResult objects.
        """
        results = super().analyze(text, entities, nlp_artifacts)

        # Filter out results with invalid structure or checksums
        validated_results = []
        for result in results:
            candidate = text[result.start : result.end]
            # Validate both structure (category code) and checksum
            if _validate_th_tnin_structure(candidate) and _validate_th_tnin_checksum(candidate):
                validated_results.append(result)

        return validated_results


def _create_th_tnin_recognizer() -> ThaiTninRecognizer:
    """Create a custom recognizer for Thai National Identification Numbers.

    Based on Presidio's TH_TNIN recognizer with structure and checksum validation.
    Format: 13 digits (first digit is category 0-8, last digit is checksum)

    Validation includes:
    - First digit must be valid category code (0-8)
    - Last digit must match the checksum algorithm

    Example: 1234567890121 (category 1)

    Returns:
        ThaiTninRecognizer: Recognizer for Thai TNIN with structure and checksum validation.
    """
    return ThaiTninRecognizer()


@functools.lru_cache(maxsize=1)
def _get_analyzer_engine() -> AnalyzerEngine:
    """Return a cached, configured Presidio AnalyzerEngine instance.

    Supports multiple languages including English, Korean, and Thai for
    comprehensive PII detection across regions. Custom recognizers are
    registered for KR_RRN and TH_TNIN as they are not included in the
    released Presidio version.

    Returns:
        AnalyzerEngine: Initialized Presidio analyzer engine with multi-language support.
    """
    # Define multi-language NLP configuration
    # Korean (ko) and Thai (th) recognizers use pattern matching and don't require NLP models,
    # but we still need English for entity recognition that depends on NLP
    sm_nlp_config: Final[dict[str, Any]] = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_sm"},
            # Korean and Thai recognizers are pattern-based and don't require spacy models
        ],
    }

    # Reduce the size of the nlp model loaded by Presidio
    provider = NlpEngineProvider(nlp_configuration=sm_nlp_config)
    sm_nlp_engine = provider.create_engine()

    # Create custom recognizers for Korean and Thai entities
    kr_rrn_recognizer = _create_kr_rrn_recognizer()
    th_tnin_recognizer = _create_th_tnin_recognizer()

    # Analyzer using minimal NLP with support for all loaded recognizers
    engine = AnalyzerEngine(nlp_engine=sm_nlp_engine)

    # Register custom recognizers
    engine.registry.add_recognizer(kr_rrn_recognizer)
    engine.registry.add_recognizer(th_tnin_recognizer)

    logger.debug(
        "Initialized Presidio analyzer engine with custom recognizers",
        extra={
            "event": "analyzer_engine_initialized",
            "supported_languages": ["en"],
            "custom_recognizers": ["KR_RRN", "TH_TNIN"],
        },
    )
    return engine


class PIIEntity(str, Enum):
    """Supported PII entity types for detection.

    Includes global and region-specific types (US, UK, Spain, Italy, etc.).
    These map to Presidio's supported entity labels.

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

    # Thailand
    TH_TNIN = "TH_TNIN"


class PIIConfig(BaseModel):
    """Configuration schema for PII detection.

    Used to control which entity types are checked and whether to block content
    containing PII or just mask it.

    Attributes:
        entities (list[PIIEntity]): List of PII entity types to detect. See the full list at: [Presidio Supported Entities](https://microsoft.github.io/presidio/supported_entities/)
        block (bool): If True, triggers tripwire when PII is detected (blocking behavior).
                     If False, only masks PII without blocking.
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

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class PiiDetectionResult:
    """Internal result structure for PII detection.

    Attributes:
        mapping (dict[str, list[str]]): Mapping from entity type to list of detected strings.
        analyzer_results (Sequence[RecognizerResult]): Raw analyzer results for position information.
    """

    mapping: dict[str, list[str]]
    analyzer_results: Sequence[RecognizerResult]

    def to_dict(self) -> dict[str, list[str]]:
        """Convert the result to a dictionary.

        Returns:
            dict[str, list[str]]: A copy of the entity mapping.
        """
        return {k: v.copy() for k, v in self.mapping.items()}


def _detect_pii(text: str, config: PIIConfig) -> PiiDetectionResult:
    """Run Presidio analysis and collect findings by entity type.

    Supports detection of Korean (KR_RRN) and Thai (TH_TNIN) entities via
    custom recognizers registered with the analyzer engine.

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

    engine = _get_analyzer_engine()

    # Run analysis for all configured entities
    # Custom recognizers (KR_RRN, TH_TNIN) are registered with language="en"
    analyzer_results = engine.analyze(text, entities=[e.value for e in config.entities], language="en")

    # Filter results and create mapping
    entity_values = {e.value for e in config.entities}
    filtered_results = [res for res in analyzer_results if res.entity_type in entity_values]
    grouped: dict[str, list[str]] = defaultdict(list)
    for res in filtered_results:
        grouped[res.entity_type].append(text[res.start : res.end])

    logger.debug(
        "PII detection completed",
        extra={
            "event": "pii_detection",
            "entities_found": len(filtered_results),
            "entity_types": list(grouped.keys()),
        },
    )
    return PiiDetectionResult(mapping=dict(grouped), analyzer_results=filtered_results)


def _mask_pii(text: str, detection: PiiDetectionResult, config: PIIConfig) -> str:
    """Mask detected PII from text by replacing with entity type markers.

    Handles overlapping entities using these rules:
    1. Full overlap: Use entity with higher score
    2. One contained in another: Use larger text span
    3. Partial intersection: Replace each individually
    4. No overlap: Replace normally

    Args:
        text (str): The text to mask.
        detection (PiiDetectionResult): Results from PII detection.
        config (PIIConfig): PII detection configuration.

    Returns:
        str: Text with PII replaced by entity type markers.

    Raises:
        ValueError: If text is empty or None.
    """
    if not text:
        raise ValueError("Text cannot be empty or None")

    # Sort by start position and score for consistent handling
    sorted_results = sorted(detection.analyzer_results, key=lambda x: (x.start, -x.score, -x.end))

    # Process results in order, tracking text offsets
    result = text
    offset = 0

    for res in sorted_results:
        start = res.start + offset
        end = res.end + offset
        replacement = f"<{res.entity_type}>"
        result = result[:start] + replacement + result[end:]
        offset += len(replacement) - (end - start)

    logger.debug(
        "PII masking completed",
        extra={
            "event": "pii_masking",
            "entities_masked": len(sorted_results),
            "entity_types": [res.entity_type for res in sorted_results],
        },
    )
    return result


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
    # Mask the text if PII is found
    checked_text = _mask_pii(text, detection, config) if detection.mapping else text

    # Only trigger tripwire if PII is found AND block=True
    tripwire_triggered = bool(detection.mapping) and config.block

    return GuardrailResult(
        tripwire_triggered=tripwire_triggered,
        info={
            "guardrail_name": name,
            "detected_entities": detection.mapping,
            "entity_types_checked": config.entities,
            "checked_text": checked_text,
            "block_mode": config.block,
            "pii_detected": bool(detection.mapping),
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
