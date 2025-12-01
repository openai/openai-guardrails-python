"""Custom context example for LLM-based guardrails.

This example shows how to:
- Use the normal OpenAI API (AsyncOpenAI-compatible) for LLM calls
- Use a different client (Ollama) for LLM-based guardrail checks via ContextVars
"""

import asyncio
from contextlib import suppress

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered
from guardrails.context import GuardrailsContext, set_context

# Pipeline config with an LLM-based guardrail using Gemma3 via Ollama
PIPELINE_CONFIG = {
    "version": 1,
    "input": {
        "version": 1,
        "guardrails": [
            {"name": "Moderation", "config": {"categories": ["hate", "violence"]}},
            {
                "name": "Custom Prompt Check",
                "config": {
                    "model": "gemma3",
                    "confidence_threshold": 0.7,
                    "system_prompt_details": "Check if the text contains any math problems.",
                },
            },
        ],
    },
}


async def main() -> None:
    # Use Ollama for guardrail LLM checks
    from openai import AsyncOpenAI

    guardrail_llm = AsyncOpenAI(
        base_url="http://127.0.0.1:11434/v1/",  # Ollama endpoint
        api_key="ollama",
    )

    # Set custom context for guardrail execution
    set_context(GuardrailsContext(guardrail_llm=guardrail_llm))

    # Instantiate GuardrailsAsyncOpenAI with the pipeline configuration and
    # the default OpenAI for main LLM calls
    client = GuardrailsAsyncOpenAI(config=PIPELINE_CONFIG)

    messages: list[dict[str, str]] = []

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                user_input = input("Enter a message: ")
                # Pass user input inline WITHOUT mutating messages first
                response = await client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=messages + [{"role": "user", "content": user_input}],
                )
                response_content = response.choices[0].message.content
                print("Assistant:", response_content)

                # Guardrails passed - now safe to add to conversation history
                messages.append({"role": "user", "content": user_input})
                messages.append({"role": "assistant", "content": response_content})
            except EOFError:
                break
            except GuardrailTripwireTriggered as exc:
                # Guardrail blocked - user message NOT added to history
                print("🛑 Guardrail triggered.", str(exc))
                continue


if __name__ == "__main__":
    asyncio.run(main())
