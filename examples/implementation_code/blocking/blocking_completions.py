"""Minimal non-streaming (blocking) implementation with guardrails using the new GuardrailsClient."""

# Please integrate openai-guardrails into my LLM calls.

# Use "guardrails_config.json" as the config file name,
# and follow the example below that illustrates a sample installation.

import asyncio
from pathlib import Path

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered


async def process_input(
    guardrails_client: GuardrailsAsyncOpenAI,
    user_input: str,
    messages: list[dict[str, str]],
) -> None:
    """Process user input with complete response validation using the new GuardrailsClient."""
    try:
        # Pass user input inline WITHOUT mutating messages first
        # Only add to messages AFTER guardrails pass and LLM call succeeds
        response = await guardrails_client.chat.completions.create(
            messages=messages + [{"role": "user", "content": user_input}],
            model="gpt-4.1-mini",
        )

        response_content = response.llm_response.choices[0].message.content
        print(f"\nAssistant: {response_content}")

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
        except GuardrailTripwireTriggered as e:
            stage_name = e.guardrail_result.info.get("stage_name", "unknown")
            guardrail_name = e.guardrail_result.info.get("guardrail_name", "unknown")
            print(f"\n🛑 Guardrail '{guardrail_name}' triggered in stage '{stage_name}'!")
            continue


if __name__ == "__main__":
    asyncio.run(main())
