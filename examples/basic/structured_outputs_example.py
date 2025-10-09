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
        ],
    },
}


async def extract_user_info(guardrails_client: GuardrailsAsyncOpenAI, text: str) -> UserInfo:
    """Extract user information using responses_parse with structured output."""
    try:
        response = await guardrails_client.responses.parse(
            input=[{"role": "system", "content": "Extract user information from the provided text."}, {"role": "user", "content": text}],
            model="gpt-4.1-nano",
            text_format=UserInfo,
        )

        # Access the parsed structured output
        user_info = response.llm_response.output_parsed
        print(f"✅ Successfully extracted: {user_info.name}, {user_info.age}, {user_info.email}")

        return user_info

    except GuardrailTripwireTriggered as exc:
        print(f"❌ Guardrail triggered: {exc}")
        raise


async def main() -> None:
    """Interactive loop demonstrating structured outputs."""
    # Initialize GuardrailsAsyncOpenAI
    guardrails_client = GuardrailsAsyncOpenAI(config=PIPELINE_CONFIG)
    while True:
        try:
            text = input("Enter text to extract user info. Include name, age, and email: ")
            user_info = await extract_user_info(guardrails_client, text)

            # Demonstrate structured output clearly
            print("\n✅ Parsed structured output:")
            print(user_info.model_dump())
            print()

        except EOFError:
            print("\nExiting.")
            break
        except GuardrailTripwireTriggered as exc:
            print(f"🛑 Guardrail triggered: {exc}")
            continue
        except Exception as e:
            print(f"Error: {e}")
            continue


if __name__ == "__main__":
    asyncio.run(main())
