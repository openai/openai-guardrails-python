# Competitor Detection

Flags mentions of competitors from a configurable list. Scans text for mentions of configured competitor names, uses case-insensitive matching to identify competitor references, triggers tripwire when competitor mentions are detected, and helps maintain business focus and prevent information sharing.

## Configuration

```json
{
    "name": "Competitor Detection",
    "config": {
        "competitors": ["competitor1", "rival-company.com", "alternative-provider"]
    }
}
```

### Parameters

- **`competitors`** (required): List of competitor names, domains, or identifiers to detect

## Implementation Notes

- **Exact Matching**: Matches the exact competitor names you configure
- **Case Insensitive**: Detects variations in capitalization

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Competitor Detection",
    "competitors_found": ["competitor1"],
    "checked_competitors": ["competitor1", "rival-company.com"],
    "checked_text": "Original input text"
}
```

- **`competitors_found`**: List of competitors detected in the text
- **`checked_competitors`**: List of competitors that were configured for detection
- **`checked_text`**: Original input text
