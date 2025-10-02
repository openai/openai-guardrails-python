# Secret Keys

Identifies potential API keys, secrets, and credentials in text using entropy analysis and pattern matching. Scans text for high-entropy strings that look like secrets, uses pattern matching to identify common secret formats, applies entropy analysis to detect random-looking strings, and helps prevent accidental exposure of sensitive credentials.

## Configuration

```json
{
    "name": "Secret Keys",
    "config": {
        "threshold": "balanced",
        "custom_regex": ["my-custom-[a-zA-Z0-9]{32}", "internal-[a-zA-Z0-9]{16}-key"]
    }
}
```

### Parameters

- **`threshold`** (optional): Detection sensitivity level (default: "balanced")
    - `"strict"` - Most sensitive, may have more false positives (commonly flag high entropy filenames or code)
    - `"balanced"` - Default setting, balanced between sensitivity and specificity  
    - `"permissive"` - Least sensitive, may have more false negatives
- **`custom_regex`** (optional): List of custom regex patterns to check for secrets

## Implementation Notes

- **Pre-configured Sensitivity**: Threshold values automatically set appropriate entropy, length, and diversity requirements
- **Pattern Matching**: Looks for common secret prefixes and formats

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Secret Keys",
    "detected_secrets": ["sk-abc123...", "Bearer xyz789..."],
    "checked_text": "Original input text"
}
```

- **`detected_secrets`**: List of potential secrets detected in the text
- **`checked_text`**: Original input text (unchanged)
