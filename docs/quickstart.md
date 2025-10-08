# Quickstart

Get started with Guardrails in minutes. Guardrails provides drop-in replacements for OpenAI clients that automatically validate inputs and outputs using configurable safety checks.

## Install

```bash
pip install openai-guardrails
```

## Set API Key

```bash
export OPENAI_API_KEY=sk-...
```

## Create Pipeline Configuration

The fastest way is using the [Guardrails Wizard](https://guardrails.openai.com/) - a no-code tool for creating configurations.

Or define manually:

```json
{
    "version": 1,
    "input": {
        "version": 1,
        "guardrails": [
            {"name": "URL Filter", "config": {}},
            {"name": "Moderation", "config": {"categories": ["hate", "violence"]}}
        ]
    },
    "output": {
        "version": 1,
        "guardrails": [
            {"name": "Contains PII", "config": {"entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]}}
        ]
    }
}
```

### Pipeline Stages

Guardrails use a **pipeline configuration** with 1 to 3 stages:

- **Preflight** - Runs before the LLM call (e.g., mask PII, moderation)
- **Input** - Runs in parallel with the LLM call (e.g., jailbreak detection)
- **Output** - Runs over the LLM generated content (e.g., fact checking, compliance)

**Not all stages are required** - you can use just input, just output, or any combination.

## Use as Drop-in Replacement

Replace your OpenAI client with the Guardrails version (GuardrailsAsyncOpenAI or GuardrailsOpenAI):

We support `chat.completions.create` and `responses.create` as well as `responses.parse` for structured outputs.

```python
import asyncio
from pathlib import Path
from guardrails import GuardrailsAsyncOpenAI

async def main():
    # Use GuardrailsAsyncOpenAI instead of AsyncOpenAI
    client = GuardrailsAsyncOpenAI(config=Path("guardrails_config.json"))
    
    try:
        response = await client.responses.create(
            model="gpt-5",
            input="Hello world"
        )
        
        # Access OpenAI response via .llm_response
        print(response.llm_response.output_text)
        
    except GuardrailTripwireTriggered as exc:
        print(f"Guardrail triggered: {exc.guardrail_result.info}")

asyncio.run(main())
```

**That's it!** Your existing OpenAI code now includes automatic guardrail validation based on your pipeline configuration. Just use `response.llm_response` instead of `response`.

## Guardrail Execution Error Handling

Guardrails supports two error handling modes for guardrail execution failures:

### Fail-Safe Mode (Default)
If a guardrail fails to execute (e.g., invalid model name), the system continues passing back `tripwire_triggered=False`:

```python
# Default: raise_guardrail_errors=False
client = GuardrailsAsyncOpenAI(config="config.json")
# Continues execution even if guardrails have any errors
```

### Fail-Secure Mode
Enable strict mode to raise exceptions on guardrail execution failures:

```python
# Strict mode: raise_guardrail_errors=True  
client = GuardrailsAsyncOpenAI(
    config="config.json",
    raise_guardrail_errors=True
)
# Raises exceptions if guardrails fail to execute properly
```

**Note**: This only affects guardrail execution errors. Safety violations (tripwires) are handled separately - see [Tripwires](./tripwires.md) for details.

## Agents SDK Integration

For OpenAI Agents SDK users, we provide `GuardrailAgent` as a drop-in replacement:

```python
from pathlib import Path
from guardrails import GuardrailAgent
from agents import Runner

# Create agent with guardrails automatically configured
agent = GuardrailAgent(
    config=Path("guardrails_config.json"),
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
)

# Use exactly like a regular Agent
result = await Runner.run(agent, "Hello, can you help me?")
```

`GuardrailAgent` automatically configures guardrails from your pipeline:

- **Prompt Injection Detection**: Applied at the **tool level** (before and after each tool call) to ensure tool actions align with user intent
- **Other guardrails** (PII, Jailbreak, etc.): Applied at the **agent level** (input/output boundaries)

#### Tool Violation Handling

By default (`block_on_tool_violations=False`), tool guardrail violations use `reject_content` to block the violative tool call/output while allowing the agent to continue execution and explain the issue to the user.

Set `block_on_tool_violations=True` to raise exceptions and halt execution immediately when tool guardrails are violated.

## Azure OpenAI

Use the Azure-specific client (GuardrailsAsyncAzureOpenAI or GuardrailsAzureOpenAI):

```python
from pathlib import Path
from guardrails import GuardrailsAsyncAzureOpenAI

client = GuardrailsAsyncAzureOpenAI(
    config=Path("guardrails_config.json"),
    azure_endpoint="https://your-resource.openai.azure.com/",
    api_key="your-azure-key",
    api_version="2025-01-01-preview"
)
```

## Third-Party Models

Works with any OpenAI-compatible API:

```python
from pathlib import Path
from guardrails import GuardrailsAsyncOpenAI

# Local Ollama model
client = GuardrailsAsyncOpenAI(
    config=Path("guardrails_config.json"),
    base_url="http://127.0.0.1:11434/v1/",
    api_key="ollama"
)
```

## Next Steps

- Explore [examples](./examples.md) for advanced patterns
- Learn about [streaming considerations](./streaming_output.md)
