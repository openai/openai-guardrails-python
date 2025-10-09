"""Example: Async customer support agent with PII masking using GuardrailsClient.

This example demonstrates how to use the PII guardrail in masking mode (block=False)
to automatically mask PII from user input using the GuardrailsClient.
The PII is replaced with placeholder tokens like <EMAIL_ADDRESS> or <US_SSN>.

Example input: "My SSN is 123-45-6789 and email is john@example.com"

PII will block the output if it is detected in the LLM response. (masking of the output is not supported yet)
"""

import asyncio
from contextlib import suppress

from rich.console import Console
from rich.panel import Panel

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered

console = Console()

# Define pipeline configuration with PII masking
PIPELINE_CONFIG = {
    "version": 1,
    "pre_flight": {
        "version": 1,
        "guardrails": [
            {
                "name": "Contains PII",
                "config": {
                    "entities": [
                        "EMAIL_ADDRESS",
                        "PHONE_NUMBER",
                        "US_SSN",
                        "CREDIT_CARD",
                    ],
                    "block": False,  # Default - won't block, just mask
                },
            }
        ],
        "config": {"concurrency": 5, "suppress_tripwire": False},
    },
    "input": {
        "version": 1,
        "guardrails": [{"name": "Moderation", "config": {"categories": ["hate", "violence"]}}],
        "config": {"concurrency": 5, "suppress_tripwire": False},
    },
    "output": {
        "version": 1,
        "guardrails": [
            {
                "name": "Contains PII",
                "config": {
                    "entities": [
                        "EMAIL_ADDRESS",
                        "PHONE_NUMBER",
                        "US_SSN",
                        "CREDIT_CARD",
                    ],
                    "block": True,  # Will block output if PII is detected
                },
            }
        ],
        "config": {"concurrency": 5, "suppress_tripwire": False},
    },
}


async def process_input(
    guardrails_client: GuardrailsAsyncOpenAI,
    user_input: str,
) -> None:
    """Process user input using GuardrailsClient with automatic PII masking.

    Args:
        guardrails_client: GuardrailsClient instance with PII masking configuration.
        user_input: User's input text.
    """
    try:
        # Use GuardrailsClient - it handles all PII masking automatically
        response = await guardrails_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Comply with the user's request.",
                },
                {"role": "user", "content": user_input},
            ],
            model="gpt-4",
        )

        # Show the LLM response (already masked if PII was detected)
        content = response.llm_response.choices[0].message.content
        console.print(f"\n[bold blue]Assistant output:[/bold blue] {content}\n")

        # Show PII masking information if detected in pre-flight
        if response.guardrail_results.preflight:
            for result in response.guardrail_results.preflight:
                if result.info.get("guardrail_name") == "Contains PII" and result.info.get("pii_detected", False):
                    detected_entities = result.info.get("detected_entities", {})
                    masked_text = result.info.get("checked_text", user_input)

                    # Show what text was actually sent to the LLM
                    console.print(
                        Panel(
                            f"PII detected and masked before sending to LLM:\n"
                            f"Original: {user_input}\n"
                            f"Sent to LLM: {masked_text}\n"
                            f"Entities found: {', '.join(detected_entities.keys())}",
                            title="PII Masking Applied",
                            border_style="yellow",
                        )
                    )
        # Show if PII was detected in output
        if response.guardrail_results.output:
            for result in response.guardrail_results.output:
                if result.info.get("guardrail_name") == "Contains PII" and result.info.get("pii_detected", False):
                    detected_entities = result.info.get("detected_entities", {})
                    console.print(
                        Panel(
                            f"Warning: PII detected in LLM output (Was not masked. Set block=True to block the output):\n"
                            f"Entities found: {', '.join(detected_entities.keys())}",
                            title="PII in Output",
                            border_style="yellow",
                        )
                    )

    except GuardrailTripwireTriggered as exc:
        stage_name = exc.guardrail_result.info.get("stage_name", "unknown")
        guardrail_name = exc.guardrail_result.info.get("guardrail_name", "unknown")
        console.print(f"[bold red]Guardrail '{guardrail_name}' triggered in stage '{stage_name}'![/bold red]")
        console.print(Panel(str(exc.guardrail_result), title="Guardrail Result", border_style="red"))
        raise


async def main() -> None:
    """Main async input loop for user interaction."""
    # Initialize GuardrailsAsyncOpenAI with PII masking configuration
    guardrails_client = GuardrailsAsyncOpenAI(config=PIPELINE_CONFIG)

    with suppress(KeyboardInterrupt, asyncio.CancelledError):
        while True:
            try:
                user_input = input("\nEnter a message: ").strip()
                if user_input.lower() == "exit":
                    break

                await process_input(guardrails_client, user_input)

            except EOFError:
                break
            except GuardrailTripwireTriggered:
                # Already handled in process_input
                continue
            except Exception as e:
                console.print(f"\nError: {e}")
                continue


if __name__ == "__main__":
    asyncio.run(main())
    console.print("\nExiting the program.")
