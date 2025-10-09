"""Example: Guardrail bundle with suppressed tripwire exception using GuardrailsClient."""

import asyncio
from contextlib import suppress
from typing import Any

from rich.console import Console
from rich.panel import Panel

from guardrails import GuardrailsAsyncOpenAI

console = Console()

# Define your pipeline configuration
PIPELINE_CONFIG: dict[str, Any] = {
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
                "name": "Custom Prompt Check",
                "config": {
                    "model": "gpt-4.1-nano-2025-04-14",
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
    """Process user input, run guardrails (tripwire suppressed)."""
    try:
        # Use GuardrailsClient with suppress_tripwire=True
        response = await guardrails_client.responses.create(
            input=user_input,
            model="gpt-4.1-nano-2025-04-14",
            previous_response_id=response_id,
            suppress_tripwire=True,
        )

        # Check if any guardrails were triggered
        if response.guardrail_results.all_results:
            for result in response.guardrail_results.all_results:
                guardrail_name = result.info.get("guardrail_name", "Unknown Guardrail")
                if result.tripwire_triggered:
                    console.print(f"[bold yellow]Guardrail '{guardrail_name}' triggered![/bold yellow]")
                    console.print(
                        Panel(
                            str(result),
                            title=f"Guardrail Result: {guardrail_name}",
                            border_style="yellow",
                        )
                    )
                else:
                    console.print(f"[bold green]Guardrail '{guardrail_name}' passed.[/bold green]")
        else:
            console.print("[bold green]No guardrails triggered.[/bold green]")

        console.print(f"\n[bold blue]Assistant output:[/bold blue] {response.llm_response.output_text}\n")
        return response.llm_response.id

    except Exception as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        return response_id


async def main() -> None:
    """Main async input loop for user interaction."""
    # Initialize GuardrailsAsyncOpenAI with the pipeline configuration
    guardrails_client = GuardrailsAsyncOpenAI(config=PIPELINE_CONFIG)

    response_id: str | None = None

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                user_input = input("Enter a message: ")
            except EOFError:
                break
            response_id = await process_input(guardrails_client, user_input, response_id)


if __name__ == "__main__":
    asyncio.run(main())
    console.print("\n[bold blue]Exiting the program.[/bold blue]")
