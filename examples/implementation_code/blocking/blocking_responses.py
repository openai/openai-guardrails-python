"""Minimal non-streaming (blocking) implementation with guardrails using the new GuardrailsClient."""

# Please integrate openai-guardrails into my LLM calls.

# Use "guardrails_config.json" as the config file name,
# and follow the example below that illustrates a sample installation.

import asyncio
from pathlib import Path

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered


async def process_input(guardrails_client: GuardrailsAsyncOpenAI, user_input: str, response_id: str | None = None) -> str | None:
    """Process user input with complete response validation using the new GuardrailsClient."""
    try:
        # Use the GuardrailsClient - it handles all guardrail validation automatically
        # including pre-flight, input, and output stages, plus the LLM call
        response = await guardrails_client.responses.create(input=user_input, model="gpt-4.1-nano", previous_response_id=response_id)

        print(f"\nAssistant: {response.llm_response.output_text}")

        return response.llm_response.id

    except GuardrailTripwireTriggered:
        # GuardrailsClient automatically handles tripwire exceptions
        raise


async def main():
    # Initialize GuardrailsAsyncOpenAI with the config file
    guardrails_client = GuardrailsAsyncOpenAI(config=Path("guardrails_config.json"))

    response_id: str | None = None

    while True:
        try:
            prompt = input("\nEnter a message: ")
            response_id = await process_input(guardrails_client, prompt, response_id)
        except (EOFError, KeyboardInterrupt):
            break
        except GuardrailTripwireTriggered as exc:
            stage_name = exc.guardrail_result.info.get("stage_name", "unknown")
            guardrail_name = exc.guardrail_result.info.get("guardrail_name", "unknown")
            print(f"\n🛑 Guardrail '{guardrail_name}' triggered in stage '{stage_name}'!")
            continue


if __name__ == "__main__":
    asyncio.run(main())
