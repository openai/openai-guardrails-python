import asyncio

from rich.console import Console
from rich.panel import Panel

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered

# Initialize Rich console
console = Console()

# Replace with your actual vector store ID from the vector store creation step
VECTOR_STORE_ID = "<YOUR VECTOR STORE ID>"  # <-- UPDATE THIS WITH YOUR VECTOR STORE ID


async def main():
    # Define the anti-hallucination guardrail config
    pipeline_config = {
        "version": 1,
        "input": {
            "version": 1,
            "guardrails": [
                {
                    "name": "Hallucination Detection",
                    "config": {
                        "model": "gpt-4.1-mini",
                        "confidence_threshold": 0.7,
                        "knowledge_source": VECTOR_STORE_ID,
                    },
                },
            ],
        },
    }

    # Initialize the guardrails client
    client = GuardrailsAsyncOpenAI(config=pipeline_config)

    messages: list[dict[str, str]] = []

    # Example inputs to test
    test_cases = [
        "Microsoft's annual revenue was $500 billion in 2023.",  # hallucination
        "Microsoft's annual revenue was $56.5 billion in 2023.",  # non-hallucination
    ]

    for candidate in test_cases:
        console.print(f"\n[bold cyan]Testing:[/bold cyan] {candidate}\n")

        try:
            # Pass user input inline WITHOUT mutating messages first
            response = await client.chat.completions.create(
                messages=messages + [{"role": "user", "content": candidate}],
                model="gpt-4.1-mini",
            )

            response_content = response.choices[0].message.content
            console.print(
                Panel(
                    f"[bold green]Tripwire not triggered[/bold green]\n\nResponse: {response_content}",
                    title="✅ Guardrail Check Passed",
                    border_style="green",
                )
            )

            # Guardrails passed - now safe to add to conversation history
            messages.append({"role": "user", "content": candidate})
            messages.append({"role": "assistant", "content": response_content})

        except GuardrailTripwireTriggered as exc:
            # Guardrail blocked - user message NOT added to history
            console.print(
                Panel(
                    f"[bold red]Guardrail triggered: {exc.guardrail_result.info.get('guardrail_name', 'unnamed')}[/bold red]",
                    title="⚠️  Guardrail Alert",
                    border_style="red",
                )
            )
            print(f"Result details: {exc.guardrail_result.info}")


if __name__ == "__main__":
    asyncio.run(main())
