# Using Guardrails with Agents SDK

Our Guardrails can easily be integrated with OpenAI's Agents SDK using the **GuardrailAgent** class for a seamless drop-in replacement.

## Overview

**GuardrailAgent** provides the simplest integration - just replace `Agent` with `GuardrailAgent` and add your config:

- Drop-in replacement for Agents SDK's `Agent` class
- Automatically configures guardrails from your pipeline configuration 
- Returns a regular `Agent` instance that works with all Agents SDK features
- **Prompt Injection Detection runs at the tool level** - checks EACH tool call and output
- Other guardrails run at the agent level for efficiency
- Keep your existing pipeline configuration - no need to rewrite
- Use Agents SDK's native exception handling for guardrail violations

## Quick Start with GuardrailAgent

The easiest way to integrate guardrails is using `GuardrailAgent` as a drop-in replacement:

```python
import asyncio
from pathlib import Path
from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered, Runner
from agents.run import RunConfig
from guardrails import GuardrailAgent

# Create agent with guardrails automatically configured from your config file
agent = GuardrailAgent(
    config=Path("guardrails_config.json"),
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
)

async def main():
    while True:
        try:
            user_input = input("Enter a message: ")
            result = await Runner.run(
                agent,
                user_input,
                run_config=RunConfig(tracing_disabled=True),
            )
            print(f"Assistant: {result.final_output}")
        except InputGuardrailTripwireTriggered:
            print("🛑 Input guardrail triggered!")
            continue
        except OutputGuardrailTripwireTriggered:
            print("🛑 Output guardrail triggered!")
            continue

if __name__ == "__main__":
    asyncio.run(main())
```

That's it! `GuardrailAgent` automatically:

- Parses your pipeline configuration
- Creates the appropriate guardrail functions 
- Wires them to a regular `Agent` instance
- Returns the configured agent ready for use with `Runner.run()`

## Configuration Options

GuardrailAgent supports the same configuration formats as our other clients:

```python
# File path (recommended)
agent = GuardrailAgent(config=Path("guardrails_config.json"), ...)

# Dictionary (for dynamic configuration)
config_dict = {
    "version": 1,
    "input": {"version": 1, "guardrails": [...]},
    "output": {"version": 1, "guardrails": [...]}
}
agent = GuardrailAgent(config=config_dict, ...)

# JSON string (with JsonString wrapper)
from guardrails import JsonString
agent = GuardrailAgent(config=JsonString('{"version": 1, ...}'), ...)
```

## Next Steps

- Use the [Guardrails Wizard](https://guardrails.openai.com/) to generate your configuration
- Explore available guardrails for your use case  
- Learn about pipeline configuration in our [quickstart](./quickstart.md)
- For more details on the OpenAI Agents SDK, refer to the [Agent SDK documentation](https://openai.github.io/openai-agents-python/).
