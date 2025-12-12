# Off Topic Prompts

Ensures content stays within defined business scope using LLM analysis. Flags content that goes off-topic or outside your scope to help maintain focus and prevent scope creep.

## Configuration

```json
{
    "name": "Off Topic Prompts",
    "config": {
        "model": "gpt-5",
        "confidence_threshold": 0.7,
        "system_prompt_details": "Customer support for our e-commerce platform. Topics include order status, returns, shipping, and product questions.",
        "max_turns": 10
    }
}
```

### Parameters

- **`model`** (required): Model to use for analysis (e.g., "gpt-5")
- **`confidence_threshold`** (required): Minimum confidence score to trigger tripwire (0.0 to 1.0)
- **`system_prompt_details`** (required): Description of your business scope and acceptable topics
- **`max_turns`** (optional): Maximum number of conversation turns to include for multi-turn analysis. Default: 10. Set to 1 for single-turn mode.
- **`include_reasoning`** (optional): Whether to include reasoning/explanation fields in the guardrail output (default: `false`)
    - When `false`: The LLM only generates the essential fields (`flagged` and `confidence`), reducing token generation costs
    - When `true`: Additionally, returns detailed reasoning for its decisions
    - **Performance**: In our evaluations, disabling reasoning reduces median latency by 40% on average (ranging from 18% to 67% depending on model) while maintaining detection performance
    - **Use Case**: Keep disabled for production to minimize costs and latency; enable for development and debugging

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
    "token_usage": {
        "prompt_tokens": 1234,
        "completion_tokens": 56,
        "total_tokens": 1290
    }
}
```

- **`flagged`**: Whether the content is off-topic (true = off-topic, false = on-topic)
- **`confidence`**: Confidence score (0.0 to 1.0) for the assessment
- **`threshold`**: The confidence threshold that was configured
- **`token_usage`**: Token usage statistics from the LLM call
- **`reason`**: Explanation of why the input was flagged (or not flagged) - *only included when `include_reasoning=true`*
