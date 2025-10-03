"""This module defines a test suite.

The test suite validates guardrails using various test cases.
It includes test definitions, execution logic, and result summarization.
"""

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from guardrails import (
    GuardrailTripwireTriggered,
    check_plain_text,
)


@dataclass
class Context:  # noqa: D101
    guardrail_llm: AsyncOpenAI


async def process_input(
    input_data: str,
    config_path: Path,
    context: Context,
    media_type: str = "text/plain",
) -> None:
    """Process a single input through the guardrails.

    Args:
        input_data (str): The text to check against guardrails.
        config_path (Path): Path to guardrail configuration.
        context (Context): Runtime context for guardrails.
        media_type (str, optional): Type of input data. Defaults to "text/plain".
    """
    try:
        # Run guardrails using check_plain_text
        results = await check_plain_text(
            text=input_data,
            bundle_path=config_path,
            ctx=context,
        )

        print("\n--- Results ---")
        if results:
            for result in results:
                print("- Guardrail passed.")
                print(f"  Tripwire triggered: {result.tripwire_triggered}")
                print(f"  Info: {result.info}")
        else:
            print(
                "No guardrails were triggered or returned results (check media type?).",
            )

    except GuardrailTripwireTriggered as e:
        print("\n--- Guardrail Tripwire Triggered! ---")
        result = e.guardrail_result
        print("Guardrail triggered")
        print(f"Info: {result.info}")

    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


async def get_user_input() -> str | None:
    """Get input from the user.

    Returns:
        Optional[str]: User input or None if they want to exit.
    """
    print("\nEnter text to check (or 'exit' to quit):")
    user_input = input("> ").strip()
    return None if user_input.lower() == "exit" else user_input


async def main(config_path: Path, media_type: str = "text/plain") -> None:
    """Run an interactive session for guardrail testing.

    Args:
        config_path (Path): Path to guardrail configuration.
        media_type (str, optional): Type of input data. Defaults to "text/plain".
    """
    print(f"Loading guardrail configuration from: {config_path}")
    try:
        openai_client = AsyncOpenAI()
        context = Context(guardrail_llm=openai_client)

        while True:
            user_input = await get_user_input()
            if user_input is None:
                print("\nExiting...")
                break

            await process_input(user_input, config_path, context, media_type)

    except FileNotFoundError as e:
        print(f"\nError: Configuration file not found: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive Guardrail Testing Tool.",
    )
    parser.add_argument(
        "config_file",
        type=str,
        help="Path to the guardrails JSON configuration file.",
    )
    parser.add_argument(
        "-m",
        "--media-type",
        type=str,
        default="text/plain",
        help="The media type of the input data (e.g., 'text/plain'). Default: text/plain",
    )

    args = parser.parse_args()
    config_path = Path(args.config_file)

    asyncio.run(main(config_path, args.media_type))
