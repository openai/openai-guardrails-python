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

- **`keywords`** (required): List of banned keywords or phrases to detect. Trailing `.`, `,`, `!`, `?`, `;`, and `:` characters are removed before matching. Values containing only those characters are rejected during configuration validation.

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Keyword Filter",
    "matched": ["confidential", "secret"],
    "checked": ["confidential", "secret", "internal only", "do not share"],
    "sanitized_keywords": ["confidential", "secret", "internal only", "do not share"]
}
```

- **`matched`**: List of keywords found in the text
- **`checked`**: List of keywords that were configured for detection
- **`sanitized_keywords`**: List of keywords used for matching after trailing punctuation is removed
