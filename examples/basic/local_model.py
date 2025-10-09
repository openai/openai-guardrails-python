"""Example: Guardrail bundle using Ollama's Gemma3 model with GuardrailsClient."""

import asyncio
from contextlib import suppress

from openai.types.chat import ChatCompletionMessageParam
from rich.console import Console
from rich.panel import Panel

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered

console = Console()

# Define your pipeline configuration for Gemma3
GEMMA3_PIPELINE_CONFIG = {
    "version": 1,
    "input": {
        "version": 1,
        "guardrails": [
            {"name": "Moderation", "config": {"categories": ["hate", "violence"]}},
            {
                "name": "URL Filter",
                "config": {"url_allow_list": ["example.com", "baz.com"]},
            },
            {
                "name": "Jailbreak",
                "config": {
                    "model": "gemma3",
                    "confidence_threshold": 0.7,
                },
            },
        ],
    },
}


async def process_input(
    guardrails_client: GuardrailsAsyncOpenAI,
    user_input: str,
    input_data: list[ChatCompletionMessageParam],
) -> None:
    """Process user input through Gemma3 guardrails using GuardrailsClient."""
    try:
        # Use GuardrailsClient for chat completions with guardrails
        response = await guardrails_client.chat.completions.create(
            messages=input_data + [{"role": "user", "content": user_input}],
            model="gemma3",
        )

        # Access response content using standard OpenAI API
        response_content = response.llm_response.choices[0].message.content
        console.print(f"\nAssistant output: {response_content}", end="\n\n")

        # Add to conversation history
        input_data.append({"role": "user", "content": user_input})
        input_data.append({"role": "assistant", "content": response_content})

    except GuardrailTripwireTriggered:
        # Handle guardrail violations
        raise


async def main() -> None:
    """Main async input loop for user interaction."""
    # Initialize GuardrailsAsyncOpenAI with Ollama configuration
    guardrails_client = GuardrailsAsyncOpenAI(
        config=GEMMA3_PIPELINE_CONFIG,
        base_url="http://127.0.0.1:11434/v1/",
        api_key="ollama",
    )

    input_data: list[ChatCompletionMessageParam] = []

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                user_input = input("Enter a message: ")
                await process_input(guardrails_client, user_input, input_data)
            except EOFError:
                break
            except GuardrailTripwireTriggered as exc:
                stage_name = exc.guardrail_result.info.get("stage_name", "unknown")
                guardrail_name = exc.guardrail_result.info.get("guardrail_name", "unknown")
                console.print(f"\n🛑 [bold red]Guardrail '{guardrail_name}' triggered in stage '{stage_name}'![/bold red]")
                console.print(
                    Panel(
                        str(exc.guardrail_result),
                        title="Guardrail Result",
                        border_style="red",
                    )
                )
                continue


if __name__ == "__main__":
    asyncio.run(main())
    console.print("\nExiting the program.")
