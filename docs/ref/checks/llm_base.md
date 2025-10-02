# LLM Base

Base configuration for LLM-based guardrails. Provides common configuration options used by other LLM-powered checks.

## Configuration

```json
{
    "name": "LLM Base",
    "config": {
        "model": "gpt-5",
        "confidence_threshold": 0.7
    }
}
```

### Parameters

- **`model`** (required): OpenAI model to use for the check (e.g., "gpt-5")
- **`confidence_threshold`** (required): Minimum confidence score to trigger tripwire (0.0 to 1.0)

## What It Does

- Provides base configuration for LLM-based guardrails
- Defines common parameters used across multiple LLM checks
- Not typically used directly - serves as foundation for other checks

## Special Considerations

- **Base Class**: This is a configuration base class, not a standalone guardrail
- **Inheritance**: Other LLM-based checks extend this configuration
- **Common Parameters**: Standardizes model and confidence settings across checks

## What It Returns

This is a base configuration class and does not return results directly. It provides the foundation for other LLM-based guardrails that return `GuardrailResult` objects.

## Usage

This configuration is typically used by other guardrails like:
- Hallucination Detection
- Jailbreak Detection
- NSFW Detection
- Off Topic Prompts
- Custom Prompt Check
