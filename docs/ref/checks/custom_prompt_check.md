# Custom Prompt Check

Implements custom content checks using configurable LLM prompts. Uses your custom LLM prompts to perform specialized validation, allows you to define exactly what constitutes a violation, provides flexibility for business-specific validation rules, and returns structured results based on your prompt design.

## Configuration

```json
{
    "name": "Custom Prompt Check",
    "config": {
        "model": "gpt-5",
        "confidence_threshold": 0.7,
        "system_prompt_details": "Determine if the user's request needs to be escalated to a senior support agent. Indications of escalation include: ...",
        "max_turns": 10
    }
}
```

### Parameters

- **`model`** (required): Model to use for the check (e.g., "gpt-5")
- **`confidence_threshold`** (required): Minimum confidence score to trigger tripwire (0.0 to 1.0)
- **`system_prompt_details`** (required): Custom instructions defining the content detection criteria
- **`max_turns`** (optional): Maximum number of conversation turns to include for multi-turn analysis. Default: 10. Set to 1 for single-turn mode.
- **`include_reasoning`** (optional): Whether to include reasoning/explanation fields in the guardrail output (default: `false`)
    - When `false`: The LLM only generates the essential fields (`flagged` and `confidence`), reducing token generation costs
    - When `true`: Additionally, returns detailed reasoning for its decisions
    - **Performance**: In our evaluations, disabling reasoning reduces median latency by 40% on average (ranging from 18% to 67% depending on model) while maintaining detection performance
    - **Use Case**: Keep disabled for production to minimize costs and latency; enable for development and debugging

## Implementation Notes

- **LLM Required**: Uses an LLM for analysis
- **Business Scope**: `system_prompt_details` should clearly define your policy and acceptable topics. Effective prompt engineering is essential for optimal LLM performance and detection accuracy.

## What It Returns

Returns a `GuardrailResult` with the following `info` dictionary:

```json
{
    "guardrail_name": "Custom Prompt Check",
    "flagged": true,
    "confidence": 0.85,
    "threshold": 0.7,
    "token_usage": {
        "prompt_tokens": 1234,
        "completion_tokens": 56,
        "total_tokens": 1290
    }
}
```

- **`flagged`**: Whether the custom validation criteria were met
- **`confidence`**: Confidence score (0.0 to 1.0) for the validation
- **`threshold`**: The confidence threshold that was configured
- **`token_usage`**: Token usage statistics from the LLM call
- **`reason`**: Explanation of why the input was flagged (or not flagged) - *only included when `include_reasoning=true`*
