"""Implementation with guardrails using Agents SDK and GuardrailAgent."""

import asyncio
from pathlib import Path

from agents import (
    InputGuardrailTripwireTriggered,
    OutputGuardrailTripwireTriggered,
    Runner,
)

from guardrails import GuardrailAgent


async def main():
    # Create agent with guardrails configured from config file
    agent = GuardrailAgent(
        config=Path("guardrails_config.json"),
        name="Customer support agent",
        instructions="You are a customer support agent. You help customers with their questions.",
    )

    while True:
        try:
            prompt = input("\nEnter a message: ")
            result = await Runner.run(agent, prompt)

            print(f"\nAssistant: {result.final_output}")

        except (EOFError, KeyboardInterrupt):
            break
        except (InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered) as exc:
            stage_name = "input" if isinstance(exc, InputGuardrailTripwireTriggered) else "output"
            print(f"\n🛑 Guardrail triggered in stage '{stage_name}'!")
            continue


if __name__ == "__main__":
    asyncio.run(main())
