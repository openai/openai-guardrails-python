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

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider
from pydantic import BaseModel, ConfigDict, Field

from guardrails.registry import default_spec_registry
from guardrails.spec import GuardrailSpecMetadata
from guardrails.types import GuardrailResult

__all__ = ["pii"]

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def _get_analyzer_engine() -> AnalyzerEngine:
    """Return a cached, configured Presidio AnalyzerEngine instance.

    Returns:
        AnalyzerEngine: Initialized Presidio analyzer engine.
    """
    # Define a smaller NLP configuration
    sm_nlp_config: Final[dict[str, Any]] = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
    }

    # Reduce the size of the nlp model loaded by Presidio
    provider = NlpEngineProvider(nlp_configuration=sm_nlp_config)
    sm_nlp_engine = provider.create_engine()

    # Analyzer using minimal NLP
    engine = AnalyzerEngine(nlp_engine=sm_nlp_engine)
    logger.debug("Initialized Presidio analyzer engine")
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
    analyzer_results = engine.analyze(text, entities=[e.value for e in config.entities], language="en")

    # Filter results once and create both mapping and filtered results
    filtered_results = [res for res in analyzer_results if res.entity_type in config.entities]
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
