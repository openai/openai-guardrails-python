"""Minimal streaming implementation with guardrails using the new GuardrailsClient."""

# Please integrate openai-guardrails into my LLM calls.

# Use "guardrails_config.json" as the config file name,
# and follow the example below that illustrates a sample installation.

import asyncio
import os
from pathlib import Path

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered


async def process_input(guardrails_client: GuardrailsAsyncOpenAI, user_input: str, response_id: str | None = None) -> str | None:
    """Process user input with streaming output and guardrails using the new GuardrailsClient."""
    try:
        # Use the GuardrailsClient - it handles all guardrail validation automatically
        # including pre-flight, input, and output stages, plus the LLM call
        stream = await guardrails_client.responses.create(
            input=user_input,
            model="gpt-4.1-nano",
            previous_response_id=response_id,
            stream=True,
        )

        # Stream with output guardrail checks
        async for chunk in stream:
            # Access streaming response exactly like native OpenAI API through .llm_response
            # For responses API streaming, check for delta content
            if hasattr(chunk.llm_response, "delta") and chunk.llm_response.delta:
                print(chunk.llm_response.delta, end="", flush=True)

        # Get the response ID from the final chunk
        response_id_to_return = None
        if hasattr(chunk.llm_response, "response") and hasattr(chunk.llm_response.response, "id"):
            response_id_to_return = chunk.llm_response.response.id

        return response_id_to_return

    except GuardrailTripwireTriggered:
        # The stream will have already yielded the violation chunk before raising
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
            # Clear output and handle violation
            os.system("cls" if os.name == "nt" else "clear")
            stage_name = exc.guardrail_result.info.get("stage_name", "unknown")
            guardrail_name = exc.guardrail_result.info.get("guardrail_name", "unknown")
            print(f"\n🛑 Guardrail '{guardrail_name}' triggered in stage '{stage_name}'!")
            continue


if __name__ == "__main__":
    asyncio.run(main())
