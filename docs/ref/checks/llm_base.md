# LLM Base

Base configuration for LLM-based guardrails. Provides common configuration options used by other LLM-powered checks, including multi-turn conversation support.

## Configuration

```json
{
    "name": "LLM Base",
    "config": {
        "model": "gpt-5",
        "confidence_threshold": 0.7,
        "max_turns": 10
    }
}
```

### Parameters

- **`model`** (required): OpenAI model to use for the check (e.g., "gpt-5")
- **`confidence_threshold`** (required): Minimum confidence score to trigger tripwire (0.0 to 1.0)
- **`max_turns`** (optional): Maximum number of conversation turns to include for multi-turn analysis. Default: 10. Set to 1 for single-turn mode.

## What It Does

- Provides base configuration for LLM-based guardrails
- Defines common parameters used across multiple LLM checks
- Enables multi-turn conversation analysis across all LLM-based guardrails
- Not typically used directly - serves as foundation for other checks

## Multi-Turn Support

All LLM-based guardrails support multi-turn conversation analysis:

- **Default behavior**: Analyzes up to the last 10 conversation turns
- **Single-turn mode**: Set `max_turns: 1` to analyze only the current input
- **Custom history length**: Adjust `max_turns` based on your use case

When conversation history is available, guardrails can detect patterns that span multiple turns, such as gradual escalation attacks or context manipulation.

## Special Considerations

- **Base Class**: This is a configuration base class, not a standalone guardrail
- **Inheritance**: Other LLM-based checks extend this configuration
- **Common Parameters**: Standardizes model, confidence, and multi-turn settings across checks

## What It Returns

This is a base configuration class and does not return results directly. It provides the foundation for other LLM-based guardrails that return `GuardrailResult` objects.

## Usage

This configuration is used by these guardrails:
- Jailbreak Detection
- NSFW Detection
- Off Topic Prompts
- Custom Prompt Check
- Competitors Detection
