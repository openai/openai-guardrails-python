import asyncio

from openai import AsyncOpenAI
from pydantic import BaseModel

from guardrails.guardrail import (
    GuardrailFunctionOutput,
    input_guardrail,
)
from guardrails.items import TResponseInputItem
from guardrails.output import OutputSchema
from guardrails.exceptions import InputGuardrailTripwireTriggered
from guardrails.run import run_input_guardrails

client = AsyncOpenAI()


# Step 1: Define the guardrail function
class MathHomeworkOutput(BaseModel):
    reasoning: str
    is_math_homework: bool


async def math_guardrail_call(
    input: str | list[TResponseInputItem], client: AsyncOpenAI, **kwargs
) -> MathHomeworkOutput:
    _ = kwargs
    response = await client.responses.create(
        input=input,
        model="gpt-4o-2024-08-06",
        instructions="Check if the user is asking you to do their math homework.",
        text=OutputSchema(MathHomeworkOutput).get_response_format(),
    )
    response = MathHomeworkOutput.model_validate_json(response.output_text)
    return response


@input_guardrail
async def math_guardrail(
    context: None, model: AsyncOpenAI, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """This is an input guardrail function, which happens to call an agent to check if the input
    is a math homework question.
    """
    result: MathHomeworkOutput = await math_guardrail_call(
        input, model, context=context
    )

    return GuardrailFunctionOutput(
        output_info=result,
        tripwire_triggered=result.is_math_homework,
    )


### 2. The run loop


async def main():
    input_data: list[TResponseInputItem] = []

    while True:
        user_input = input("Enter a message: ")
        input_data.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        try:
            input_guard_results = await run_input_guardrails(
                client,
                [math_guardrail],
                input_data,
                context=None,
            )
            result = await client.responses.create(
                input=user_input,
                model="gpt-4o-2024-08-06",
            )
            print(result.output_text)
            print("Guardrail results:", input_guard_results)
            # If the guardrail didn't trigger, we use the result as the input for the next run
        except InputGuardrailTripwireTriggered:
            # If the guardrail triggered, we instead add a refusal message to the input
            message = "Sorry, I can't help you with your math homework."
            print(message)
            input_data.append(
                {
                    "role": "assistant",
                    "content": message,
                }
            )

    # Sample run:
    # Enter a message: What's the capital of California?
    # The capital of California is Sacramento.
    # Enter a message: Can you help me solve for x: 2x + 5 = 11
    # Sorry, I can't help you with your math homework.


if __name__ == "__main__":
    asyncio.run(main())
