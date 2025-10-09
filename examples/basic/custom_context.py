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

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                user_input = input("Enter a message: ")
                response = await client.chat.completions.create(model="gpt-4.1-nano", messages=[{"role": "user", "content": user_input}])
                print("Assistant:", response.llm_response.choices[0].message.content)
            except EOFError:
                break
            except GuardrailTripwireTriggered as exc:
                # Minimal handling; guardrail details available on exc.guardrail_result
                print("🛑 Guardrail triggered.", str(exc))
                continue


if __name__ == "__main__":
    asyncio.run(main())
