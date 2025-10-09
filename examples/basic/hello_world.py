"""Hello World: Minimal async customer support agent with guardrails using the new GuardrailsAsyncOpenAI."""

import asyncio
from contextlib import suppress

from rich.console import Console
from rich.panel import Panel

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered

console = Console()

# Pipeline configuration with pre_flight and input guardrails
PIPELINE_CONFIG = {
    "version": 1,
    "pre_flight": {
        "version": 1,
        "guardrails": [
            {"name": "Contains PII", "config": {"entities": ["US_SSN", "PHONE_NUMBER", "EMAIL_ADDRESS"]}},
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
}


async def process_input(
    guardrails_client: GuardrailsAsyncOpenAI,
    user_input: str,
    response_id: str | None = None,
) -> str:
    """Process user input using the new GuardrailsAsyncOpenAI."""
    try:
        # Use the new GuardrailsAsyncOpenAI - it handles all guardrail validation automatically
        response = await guardrails_client.responses.create(
            input=user_input,
            model="gpt-4.1-nano",
            previous_response_id=response_id,
        )

        console.print(f"\nAssistant output: {response.llm_response.output_text}", end="\n\n")

        # Show guardrail results if any were run
        if response.guardrail_results.all_results:
            console.print(f"[dim]Guardrails checked: {len(response.guardrail_results.all_results)}[/dim]")

        return response.llm_response.id

    except GuardrailTripwireTriggered:
        raise


async def main() -> None:
    """Main async input loop for user interaction."""
    # Initialize GuardrailsAsyncOpenAI with our pipeline configuration
    guardrails_client = GuardrailsAsyncOpenAI(config=PIPELINE_CONFIG)

    response_id = None

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                user_input = input("Enter a message: ")
                response_id = await process_input(guardrails_client, user_input, response_id)
            except EOFError:
                break
            except GuardrailTripwireTriggered as exc:
                stage_name = exc.guardrail_result.info.get("stage_name", "unknown")
                console.print(f"\n🛑 [bold red]Guardrail triggered in stage '{stage_name}'![/bold red]")
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
