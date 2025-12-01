"""Minimal streaming implementation with guardrails using the new GuardrailsClient."""

# Please integrate openai-guardrails into my LLM calls.

# Use "guardrails_config.json" as the config file name,
# and follow the example below that illustrates a sample installation.

import asyncio
import os
from pathlib import Path

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered


async def process_input(
    guardrails_client: GuardrailsAsyncOpenAI,
    user_input: str,
    messages: list[dict[str, str]],
) -> str:
    """Process user input with streaming output and guardrails using the GuardrailsClient."""
    try:
        # Pass user input inline WITHOUT mutating messages first
        # Only add to messages AFTER guardrails pass and streaming completes
        stream = await guardrails_client.chat.completions.create(
            messages=messages + [{"role": "user", "content": user_input}],
            model="gpt-4.1-mini",
            stream=True,
        )

        # Stream with output guardrail checks and accumulate response
        response_content = ""
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                print(delta, end="", flush=True)
                response_content += delta

        print()  # New line after streaming

        # Guardrails passed - now safe to add to conversation history
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": response_content})

    except GuardrailTripwireTriggered:
        # Guardrail blocked - user message NOT added to history
        raise


async def main():
    # Initialize GuardrailsAsyncOpenAI with the config file
    guardrails_client = GuardrailsAsyncOpenAI(config=Path("guardrails_config.json"))

    messages: list[dict[str, str]] = []

    while True:
        try:
            prompt = input("\nEnter a message: ")
            await process_input(guardrails_client, prompt, messages)
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
