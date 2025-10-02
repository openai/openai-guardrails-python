"""Example: Basic async guardrail bundle using Agents SDK with GuardrailAgent."""

import asyncio
from contextlib import suppress

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
)
from agents.run import RunConfig

from guardrails import GuardrailAgent

# Define your pipeline configuration
PIPELINE_CONFIG = {
    "version": 1,
    "pre_flight": {
        "version": 1,
        "guardrails": [
            {
                "name": "Moderation",
                "config": {
                    "categories": ["hate", "violence", "self-harm"],
                },
            },
        ],
    },
    "input": {
        "version": 1,
        "guardrails": [
            {
                "name": "Custom Prompt Check",
                "config": {
                    "model": "gpt-4.1-nano-2025-04-14",
                    "confidence_threshold": 0.7,
                    "system_prompt_details": "Check if the text contains any math problems.",
                },
            },
        ],
    },
    "output": {
        "version": 1,
        "guardrails": [
            {"name": "URL Filter", "config": {"url_allow_list": ["example.com"]}},
        ],
    },
}


async def main() -> None:
    """Main input loop for the customer support agent with input/output guardrails."""
    # Create agent with guardrails automatically configured from pipeline configuration
    AGENT = GuardrailAgent(
        config=PIPELINE_CONFIG,
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
    )

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                user_input = input("Enter a message: ")
                result = await Runner.run(
                    AGENT,
                    user_input,
                    run_config=RunConfig(tracing_disabled=True),
                )
                print(f"Assistant: {result.final_output}")
            except EOFError:
                print("\nExiting.")
                break
            except InputGuardrailTripwireTriggered:
                print("🛑 Input guardrail triggered!")
                continue
            except OutputGuardrailTripwireTriggered:
                print("🛑 Output guardrail triggered!")
                continue


if __name__ == "__main__":
    asyncio.run(main())
    print("\nExiting the program.")
