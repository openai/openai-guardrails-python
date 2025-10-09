"""Guardrail implementation using Azure OpenAI for all LLM calls.

Configure the following environment variables (e.g., via .env):
- AZURE_ENDPOINT: https://<resource-name>.openai.azure.com
- AZURE_DEPLOYMENT: your deployment name (e.g., gpt-4o-mini)
- AZURE_API_KEY: your Azure OpenAI API key
"""

import asyncio
import os

from dotenv import load_dotenv
from openai import BadRequestError

from guardrails import (
    GuardrailsAsyncAzureOpenAI,
    GuardrailTripwireTriggered,
)

load_dotenv()

# Import Azure credentials from secret file
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

# Define your pipeline configuration
PIPELINE_CONFIG = {
    "version": 1,
    "pre_flight": {
        "version": 1,
        "guardrails": [
            {"name": "Moderation", "config": {"categories": ["hate", "violence"]}},
            {
                "name": "Jailbreak",
                "config": {"confidence_threshold": 0.7, "model": AZURE_DEPLOYMENT},
            },
        ],
    },
    "input": {
        "version": 1,
        "guardrails": [
            {"name": "URL Filter", "config": {"url_allow_list": ["www.openai.com"]}},
            {
                "name": "Custom Prompt Check",
                "config": {
                    "model": AZURE_DEPLOYMENT,
                    "confidence_threshold": 0.7,
                    "system_prompt_details": "Check if the text contains any math problems.",
                },
            },
        ],
    },
}


async def process_input(guardrails_client: GuardrailsAsyncAzureOpenAI, user_input: str) -> None:
    """Process user input with complete response validation using GuardrailsClient."""
    try:
        # Use GuardrailsClient to handle all guardrail checks and LLM calls
        response = await guardrails_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": user_input}],
        )

        # Extract the response content from the GuardrailsResponse
        response_text = response.llm_response.choices[0].message.content

        # Only show output if all guardrails pass
        print(f"\nAssistant: {response_text}")

    except GuardrailTripwireTriggered as e:
        # Extract information from the triggered guardrail
        triggered_result = e.guardrail_result
        print("   Input blocked. Please try a different message.")
        print(f"   Full result: {triggered_result}")
        raise
    except BadRequestError as e:
        # Handle Azure's built-in content filter errors
        # Will be triggered not when the guardrail is tripped, but when the LLM is filtered by Azure.
        if "content_filter" in str(e):
            print("\n🚨 Third party content filter triggered during LLM call.")
            print(f"   Error: {e}")
            raise
        else:
            # Re-raise other BadRequestError types
            print(f"   API Error: {e}")
            raise


async def main():
    # Initialize GuardrailsAsyncAzureOpenAI with native Azure arguments
    guardrails_client = GuardrailsAsyncAzureOpenAI(
        config=PIPELINE_CONFIG,
        azure_endpoint=AZURE_ENDPOINT,
        api_key=AZURE_API_KEY,
        api_version="2025-01-01-preview",
    )

    while True:
        try:
            prompt = input("\nEnter a message: ")

            if prompt.lower() == "exit":
                print("Goodbye!")
                break

            await process_input(guardrails_client, prompt)
        except (EOFError, KeyboardInterrupt):
            break
        except (GuardrailTripwireTriggered, BadRequestError):
            continue


if __name__ == "__main__":
    asyncio.run(main())
