# Contains PII

Detects personally identifiable information (PII) such as SSNs, phone numbers, credit card numbers, and email addresses using Microsoft's [Presidio library](https://microsoft.github.io/presidio/). Will automatically mask detected PII or block content based on configuration.

## Configuration

```json
{
    "name": "Contains PII",
    "config": {
        "entities": ["EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD", "PHONE_NUMBER"],
        "block": false
    }
}
```

### Parameters

- **`entities`** (required): List of PII entity types to detect. See the full list of [supported entities](https://microsoft.github.io/presidio/supported_entities/).
- **`block`** (optional): Whether to block content or just mask PII (default: `false`)

## Implementation Notes

**Stage-specific behavior is critical:**

- **Pre-flight stage**: Use `block=false` (default) for automatic PII masking of user input
- **Output stage**: Use `block=true` to prevent PII exposure in LLM responses
- **Masking in output stage is not supported** and will not work as expected

**PII masking mode** (default, `block=false`):

- Automatically replaces detected PII with placeholder tokens like `<EMAIL_ADDRESS>`, `<US_SSN>`
- Does not trigger tripwire - allows content through with PII removed

**Blocking mode** (`block=true`):

- Triggers tripwire when PII is detected
- Prevents content from being delivered to users

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Contains PII",
    "detected_entities": {
        "EMAIL_ADDRESS": ["user@email.com"],
        "US_SSN": ["123-45-6789"]
    },
    "entity_types_checked": ["EMAIL_ADDRESS", "US_SSN", "CREDIT_CARD"],
    "checked_text": "Contact me at <EMAIL_ADDRESS>, SSN: <US_SSN>",
    "block_mode": false,
    "pii_detected": true
}
```

- **`detected_entities`**: Detected entities and their values
- **`entity_types_checked`**: List of entity types that were configured for detection
- **`checked_text`**: Text with PII masked (if PII was found) or original text (if no PII was found)
- **`block_mode`**: Whether the check was configured to block or mask
- **`pii_detected`**: Boolean indicating if any PII was found
