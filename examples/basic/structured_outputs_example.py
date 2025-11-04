"""Simple example demonstrating structured outputs with GuardrailsClient."""

import asyncio

from pydantic import BaseModel, Field

from guardrails import GuardrailsAsyncOpenAI, GuardrailTripwireTriggered


# Define a simple Pydantic model for structured output
class UserInfo(BaseModel):
    """User information extracted from text."""

    name: str = Field(description="Full name of the user")
    age: int = Field(description="Age of the user")
    email: str = Field(description="Email address of the user")


# Pipeline configuration with basic guardrails
PIPELINE_CONFIG = {
    "version": 1,
    "input": {
        "version": 1,
        "guardrails": [
            {"name": "Moderation", "config": {"categories": ["hate", "violence"]}},
            {
                "name": "Custom Prompt Check",
                "config": {
                    "model": "gpt-4.1-mini",
                    "confidence_threshold": 0.7,
                    "system_prompt_details": "Check if the text contains any math problems.",
                },
            },
        ],
    },
}


async def extract_user_info(
    guardrails_client: GuardrailsAsyncOpenAI,
    text: str,
    previous_response_id: str | None = None,
) -> tuple[UserInfo, str]:
    """Extract user information using responses.parse with structured output."""
    try:
        # Use responses.parse() for structured outputs with guardrails
        # Note: responses.parse() requires input as a list of message dicts
        response = await guardrails_client.responses.parse(
            input=[
                {"role": "system", "content": "Extract user information from the provided text."},
                {"role": "user", "content": text},
            ],
            model="gpt-4.1-mini",
            text_format=UserInfo,
            previous_response_id=previous_response_id,
        )

        # Access the parsed structured output
        user_info = response.llm_response.output_parsed
        print(f"✅ Successfully extracted: {user_info.name}, {user_info.age}, {user_info.email}")

        # Return user info and response ID (only returned if guardrails pass)
        return user_info, response.llm_response.id

    except GuardrailTripwireTriggered:
        # Guardrail blocked - no response ID returned, conversation history unchanged
        raise


async def main() -> None:
    """Interactive loop demonstrating structured outputs with conversation history."""
    # Initialize GuardrailsAsyncOpenAI
    guardrails_client = GuardrailsAsyncOpenAI(config=PIPELINE_CONFIG)

    # Use previous_response_id to maintain conversation history with responses API
    response_id: str | None = None

    while True:
        try:
            text = input("Enter text to extract user info. Include name, age, and email: ")

            # Extract user info - only updates response_id if guardrails pass
            user_info, response_id = await extract_user_info(guardrails_client, text, response_id)

            # Demonstrate structured output clearly
            print("\n✅ Parsed structured output:")
            print(user_info.model_dump())
            print()

        except EOFError:
            print("\nExiting.")
            break
        except GuardrailTripwireTriggered as exc:
            # Guardrail blocked - response_id unchanged, so blocked message not in history
            print(f"🛑 Guardrail triggered: {exc}")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
