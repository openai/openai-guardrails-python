# Custom Prompt Check

Implements custom content checks using configurable LLM prompts. Uses your custom LLM prompts to perform specialized validation, allows you to define exactly what constitutes a violation, provides flexibility for business-specific validation rules, and returns structured results based on your prompt design.

## Configuration

```json
{
    "name": "Custom Prompt Check",
    "config": {
        "model": "gpt-5",
        "confidence_threshold": 0.7,
        "system_prompt_details": "Determine if the user's request needs to be escalated to a senior support agent. Indications of escalation include: ..."
    }
}
```

### Parameters

- **`model`** (required): Model to use for the check (e.g., "gpt-5")
- **`confidence_threshold`** (required): Minimum confidence score to trigger tripwire (0.0 to 1.0)
- **`system_prompt_details`** (required): Custom instructions defining the content detection criteria

## Implementation Notes

- **Custom Logic**: You define the validation criteria through prompts
- **Prompt Engineering**: Quality of results depends on your prompt design

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Custom Prompt Check",
    "flagged": true,
    "confidence": 0.85,
    "threshold": 0.7,
    "checked_text": "Original input text"
}
```

- **`flagged`**: Whether the custom validation criteria were met
- **`confidence`**: Confidence score (0.0 to 1.0) for the validation
- **`threshold`**: The confidence threshold that was configured
- **`checked_text`**: Original input text
