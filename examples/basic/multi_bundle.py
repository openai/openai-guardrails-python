"""Example: Async customer support agent with multiple guardrail bundles using GuardrailsClient. Streams output using Rich."""

import asyncio
from contextlib import suppress

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered

console = Console()

# Define your pipeline configuration
PIPELINE_CONFIG = {
    "version": 1,
    "pre_flight": {
        "version": 1,
        "guardrails": [
            {"name": "Moderation", "config": {"categories": ["hate", "violence"]}},
            {
                "name": "URL Filter",
                "config": {"url_allow_list": ["example.com", "baz.com"]},
            },
        ],
    },
    "input": {
        "version": 1,
        "guardrails": [
            {
                "name": "Custom Prompt Check",
                "config": {
                    "model": "gpt-4.1-nano",
                    "confidence_threshold": 0.7,
                    "system_prompt_details": "Check if the text contains any math problems.",
                },
            },
        ],
    },
    "output": {
        "version": 1,
        "guardrails": [
            {"name": "URL Filter", "config": {"url_allow_list": ["openai.com"]}},
        ],
    },
}


async def process_input(
    guardrails_client: GuardrailsAsyncOpenAI,
    user_input: str,
    response_id: str | None = None,
) -> str | None:
    """Process user input with streaming output and guardrails using GuardrailsClient."""
    # Use the new GuardrailsClient - it handles all guardrail validation automatically
    # including pre-flight, input, and output stages, plus the LLM call
    stream = await guardrails_client.responses.create(
        input=user_input,
        model="gpt-4.1-nano",
        previous_response_id=response_id,
        stream=True,
    )

    # Stream the assistant's output inside a Rich Live panel
    output_text = "Assistant output: "
    with Live(output_text, console=console, refresh_per_second=10) as live:
        try:
            async for chunk in stream:
                # Access streaming response exactly like native OpenAI API through .llm_response
                if hasattr(chunk.llm_response, "delta") and chunk.llm_response.delta:
                    output_text += chunk.llm_response.delta
                    live.update(output_text)

            # Get the response ID from the final chunk
            response_id_to_return = None
            if hasattr(chunk.llm_response, "response") and hasattr(chunk.llm_response.response, "id"):
                response_id_to_return = chunk.llm_response.response.id

            return response_id_to_return

        except GuardrailTripwireTriggered:
            # Clear the live display when output guardrail is triggered
            live.update("")
            console.clear()
            raise


async def main() -> None:
    """Simple REPL loop: read from stdin, process, and stream results."""
    # Initialize GuardrailsAsyncOpenAI with the pipeline configuration
    guardrails_client = GuardrailsAsyncOpenAI(config=PIPELINE_CONFIG)

    response_id: str | None = None

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                prompt = input("Enter a message: ")
                response_id = await process_input(guardrails_client, prompt, response_id)
            except (EOFError, KeyboardInterrupt):
                break
            except GuardrailTripwireTriggered as exc:
                stage_name = exc.guardrail_result.info.get("stage_name", "unknown")
                guardrail_name = exc.guardrail_result.info.get("guardrail_name", "unknown")
                console.print(
                    f"🛑 Guardrail '{guardrail_name}' triggered in stage '{stage_name}'!",
                    style="bold red",
                )
                console.print(
                    Panel(
                        str(exc.guardrail_result),
                        title="Guardrail Result",
                        border_style="red",
                    )
                )
                # on guardrail trip, just continue to next prompt
                continue

    console.print("👋 Goodbye!", style="bold green")


if __name__ == "__main__":
    asyncio.run(main())
