# Keyword Filter

Detects and blocks text containing specified banned keywords or phrases. Uses case-insensitive matching with word boundaries to identify forbidden terms and triggers if any configured keyword is found.

## Configuration

```json
{
    "name": "Keyword Filter",
    "config": {
        "keywords": ["confidential", "secret", "internal only", "do not share"]
    }
}
```

### Parameters

- **`keywords`** (required): List of banned keywords or phrases to detect

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Keyword Filter",
    "matched": ["confidential", "secret"],
    "checked": ["confidential", "secret", "internal only"],
    "checked_text": "This is confidential information that should be kept secret"
}
```

- **`matched`**: List of keywords found in the text
- **`checked`**: List of keywords that were configured for detection
- **`checked_text`**: Original input text
