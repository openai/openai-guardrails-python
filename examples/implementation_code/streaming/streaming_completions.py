"""Minimal streaming implementation with guardrails using the new GuardrailsClient."""

# Please integrate openai-guardrails into my LLM calls.

# Use "guardrails_config.json" as the config file name,
# and follow the example below that illustrates a sample installation.

import asyncio
import os
from pathlib import Path

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered


async def process_input(guardrails_client: GuardrailsAsyncOpenAI, user_input: str) -> str:
    """Process user input with streaming output and guardrails using the GuardrailsClient."""
    try:
        # Use the GuardrailsClient - it handles all guardrail validation automatically
        # including pre-flight, input, and output stages, plus the LLM call
        stream = await guardrails_client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="gpt-4.1-nano",
            stream=True,
        )

        # Stream with output guardrail checks
        async for chunk in stream:
            if chunk.llm_response.choices[0].delta.content:
                print(chunk.llm_response.choices[0].delta.content, end="", flush=True)
        return "Stream completed successfully"

    except GuardrailTripwireTriggered:
        raise


async def main():
    # Initialize GuardrailsAsyncOpenAI with the config file
    guardrails_client = GuardrailsAsyncOpenAI(config=Path("guardrails_config.json"))

    while True:
        try:
            prompt = input("\nEnter a message: ")
            await process_input(guardrails_client, prompt)
        except (EOFError, KeyboardInterrupt):
            break
        except GuardrailTripwireTriggered as exc:
            # The stream will have already yielded the violation chunk before raising
            os.system("cls" if os.name == "nt" else "clear")
            stage_name = exc.guardrail_result.info.get("stage_name", "unknown")
            guardrail_name = exc.guardrail_result.info.get("guardrail_name", "unknown")
            print(f"\n🛑 Guardrail '{guardrail_name}' triggered in stage '{stage_name}'!")
            continue


if __name__ == "__main__":
    asyncio.run(main())
