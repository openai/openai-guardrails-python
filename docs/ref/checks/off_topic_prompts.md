# Off Topic Prompts

Ensures content stays within defined business scope using LLM analysis. Flags content that goes off-topic or outside your scope to help maintain focus and prevent scope creep.

## Configuration

```json
{
    "name": "Off Topic Prompts",
    "config": {
        "model": "gpt-5",
        "confidence_threshold": 0.7,
        "system_prompt_details": "Customer support for our e-commerce platform. Topics include order status, returns, shipping, and product questions."
    }
}
```

### Parameters

- **`model`** (required): Model to use for analysis (e.g., "gpt-5")
- **`confidence_threshold`** (required): Minimum confidence score to trigger tripwire (0.0 to 1.0)
- **`system_prompt_details`** (required): Description of your business scope and acceptable topics

## Implementation Notes

- **LLM Required**: Uses an LLM for analysis
- **Business Scope**: `system_prompt_details` should clearly define your business scope and acceptable topics. Effective prompt engineering is essential for optimal LLM performance and accurate off-topic detection.

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Off Topic Prompts",
    "flagged": false,
    "confidence": 0.85,
    "threshold": 0.7,
    "checked_text": "Original input text"
}
```

- **`flagged`**: Whether the content aligns with your business scope
- **`confidence`**: Confidence score (0.0 to 1.0) for the prompt injection detection assessment
- **`threshold`**: The confidence threshold that was configured
- **`checked_text`**: Original input text
